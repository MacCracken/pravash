//! Grid-based Euler/Navier-Stokes fluid solver.
//!
//! 2D incompressible Navier-Stokes on a uniform grid using operator splitting:
//! advection (semi-Lagrangian) → diffusion (Gauss-Seidel) → force application
//! → pressure projection (Gauss-Seidel Poisson solver).

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

use hisab::num::{dst, idst};
use tracing::trace_span;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Boundary condition type for the grid solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BoundaryCondition {
    /// Velocity = 0 at walls (viscous wall).
    NoSlip,
    /// Normal velocity = 0, tangential velocity free (inviscid wall).
    FreeSlip,
    /// Opposite edges connected (wrapping). Enables FFT pressure solve.
    Periodic,
}

/// Configuration for a grid fluid simulation step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GridConfig {
    pub dt: f64,
    pub viscosity: f64,
    pub diffusion_iterations: usize,
    pub pressure_iterations: usize,
    pub boundary: BoundaryCondition,
    /// Vorticity confinement strength (0.0 to disable).
    /// Requires grid size ≥ 6x6 to have effect.
    pub vorticity_confinement: f64,
    /// Buoyancy coefficient. Applied as `vy += alpha * (density - ambient) * dt`.
    ///
    /// Sign convention (y-up): positive alpha pushes high-density fluid upward.
    /// For thermal buoyancy (hot = light = rises), use negative alpha with
    /// density representing temperature, or invert the density field.
    pub buoyancy_alpha: f64,
    /// Ambient density for buoyancy reference.
    pub ambient_density: f64,
    /// Use MacCormack advection (higher-order, less diffusive).
    /// Falls back to semi-Lagrangian at grid boundaries.
    pub use_maccormack: bool,
    /// Smagorinsky SGS turbulence model coefficient (0.0 to disable).
    /// ν_t = (Cs·dx)²·|S|. Typical Cs: 0.1–0.2. Default: 0.0.
    pub smagorinsky_cs: f64,
}

impl GridConfig {
    /// Default config for smoke simulation (y-up coordinates).
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            dt: 0.1,
            viscosity: 0.0001,
            diffusion_iterations: 20,
            pressure_iterations: 40,
            boundary: BoundaryCondition::NoSlip,
            vorticity_confinement: 0.5,
            buoyancy_alpha: 0.1,
            ambient_density: 0.0,
            use_maccormack: false,
            smagorinsky_cs: 0.0,
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self::smoke()
    }
}

/// 2D grid-based fluid state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FluidGrid {
    pub nx: usize,
    pub ny: usize,
    /// Cell size in physical units.
    pub dx: f64,
    /// Velocity field (x-component), row-major. Units: physical (m/s).
    pub vx: Vec<f64>,
    /// Velocity field (y-component), row-major. Units: physical (m/s).
    pub vy: Vec<f64>,
    /// Pressure field, row-major.
    pub pressure: Vec<f64>,
    /// Density/scalar field (smoke, temperature, dye).
    pub density: Vec<f64>,
    /// Scratch buffers (persistent across steps to avoid allocation).
    #[serde(skip)]
    scratch_a: Vec<f64>,
    #[serde(skip)]
    scratch_b: Vec<f64>,
    #[serde(skip)]
    scratch_c: Vec<f64>,
    /// Diffusion RHS buffer.
    #[serde(skip)]
    diffuse_rhs: Vec<f64>,
}

impl FluidGrid {
    /// Minimum grid size for the solver to work (boundary + interior + boundary).
    const MIN_SIZE: usize = 4;

    /// Create a new grid initialized to zero. Minimum size is 4x4.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx < Self::MIN_SIZE || ny < Self::MIN_SIZE {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 || !dx.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive and finite: {dx}").into(),
            });
        }
        let size = nx.checked_mul(ny).ok_or(PravashError::InvalidParameter {
            reason: format!("grid size overflow: {nx} x {ny}").into(),
        })?;
        Ok(Self {
            nx,
            ny,
            dx,
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            pressure: vec![0.0; size],
            density: vec![0.0; size],
            scratch_a: vec![0.0; size],
            scratch_b: vec![0.0; size],
            scratch_c: vec![0.0; size],
            diffuse_rhs: vec![0.0; size],
        })
    }

    #[inline]
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        debug_assert!(x < self.nx && y < self.ny, "grid index out of bounds");
        y * self.nx + x
    }

    #[inline]
    #[must_use]
    pub fn velocity_at(&self, x: usize, y: usize) -> (f64, f64) {
        let i = self.idx(x, y);
        (self.vx[i], self.vy[i])
    }

    #[inline]
    #[must_use]
    pub fn speed_at(&self, x: usize, y: usize) -> f64 {
        let (vx, vy) = self.velocity_at(x, y);
        (vx * vx + vy * vy).sqrt()
    }

    #[must_use]
    pub fn max_speed(&self) -> f64 {
        self.vx
            .iter()
            .zip(self.vy.iter())
            .map(|(&vx, &vy)| vx * vx + vy * vy)
            .fold(0.0f64, f64::max)
            .sqrt()
    }

    /// Compute CFL-limited timestep for the grid solver.
    ///
    /// dt = cfl · min(dx/v_max, dx²/(4·ν))
    #[must_use]
    pub fn cfl_dt(&self, viscosity: f64, cfl_factor: f64) -> f64 {
        let mut dt = f64::MAX;
        let max_v = self.max_speed();
        if max_v > 1e-20 {
            dt = dt.min(self.dx / max_v);
        }
        if viscosity > 1e-20 {
            dt = dt.min(self.dx * self.dx / (4.0 * viscosity));
        }
        cfl_factor * dt
    }

    #[must_use]
    pub fn total_kinetic_energy(&self) -> f64 {
        let dx2 = self.dx * self.dx;
        self.vx
            .iter()
            .zip(self.vy.iter())
            .zip(self.density.iter())
            .map(|((&vx, &vy), &rho)| 0.5 * rho * (vx * vx + vy * vy) * dx2)
            .sum()
    }

    // ── Advection ───────────────────────────────────────────────────────────

    /// Bilinear interpolation at continuous position (fx, fy) in grid-cell units.
    #[inline]
    fn sample(field: &[f64], nx: usize, ny: usize, fx: f64, fy: f64) -> f64 {
        // Propagate NaN from advection backtrace (fail-fast for divergence detection)
        if !fx.is_finite() || !fy.is_finite() {
            return f64::NAN;
        }
        let x0 = fx.floor().max(0.0).min((nx - 2) as f64) as usize;
        let y0 = fy.floor().max(0.0).min((ny - 2) as f64) as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let sx = (fx - x0 as f64).clamp(0.0, 1.0);
        let sy = (fy - y0 as f64).clamp(0.0, 1.0);

        let v00 = field[y0 * nx + x0];
        let v10 = field[y0 * nx + x1];
        let v01 = field[y1 * nx + x0];
        let v11 = field[y1 * nx + x1];

        v00 * (1.0 - sx) * (1.0 - sy)
            + v10 * sx * (1.0 - sy)
            + v01 * (1.0 - sx) * sy
            + v11 * sx * sy
    }

    /// Semi-Lagrangian advection: trace back through velocity field, sample.
    /// Velocity is in physical units; dx converts to grid-cell displacement.
    #[allow(clippy::too_many_arguments)]
    fn advect(
        dst: &mut [f64],
        src: &[f64],
        vx: &[f64],
        vy: &[f64],
        nx: usize,
        ny: usize,
        dt: f64,
        inv_dx: f64,
    ) {
        let _span = trace_span!("grid::advect", nx, ny).entered();
        let dt_inv_dx = dt * inv_dx;

        #[cfg(feature = "parallel")]
        {
            let rows: Vec<(usize, &mut [f64])> = dst.chunks_mut(nx).enumerate().collect();
            rows.into_par_iter().for_each(|(y, row)| {
                if y >= 1 && y < ny - 1 {
                    #[allow(clippy::needless_range_loop)]
                    for x in 1..nx - 1 {
                        let i = y * nx + x;
                        let fx = x as f64 - dt_inv_dx * vx[i];
                        let fy = y as f64 - dt_inv_dx * vy[i];
                        row[x] = Self::sample(src, nx, ny, fx, fy);
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    let fx = x as f64 - dt_inv_dx * vx[i];
                    let fy = y as f64 - dt_inv_dx * vy[i];
                    dst[i] = Self::sample(src, nx, ny, fx, fy);
                }
            }
        }
    }

    /// Semi-Lagrangian advection with periodic wrapping.
    #[allow(clippy::too_many_arguments)]
    fn advect_periodic(
        dst: &mut [f64],
        src: &[f64],
        vx: &[f64],
        vy: &[f64],
        nx: usize,
        ny: usize,
        dt: f64,
        inv_dx: f64,
    ) {
        let _span = trace_span!("grid::advect_periodic", nx, ny).entered();
        let dt_inv_dx = dt * inv_dx;

        #[cfg(feature = "parallel")]
        {
            let rows: Vec<(usize, &mut [f64])> = dst.chunks_mut(nx).enumerate().collect();
            rows.into_par_iter().for_each(|(y, row)| {
                #[allow(clippy::needless_range_loop)]
                for x in 0..nx {
                    let i = y * nx + x;
                    let fx = x as f64 - dt_inv_dx * vx[i];
                    let fy = y as f64 - dt_inv_dx * vy[i];
                    row[x] = Self::sample_periodic(src, nx, ny, fx, fy);
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for y in 0..ny {
                for x in 0..nx {
                    let i = y * nx + x;
                    let fx = x as f64 - dt_inv_dx * vx[i];
                    let fy = y as f64 - dt_inv_dx * vy[i];
                    dst[i] = Self::sample_periodic(src, nx, ny, fx, fy);
                }
            }
        }
    }

    /// Sample with periodic wrapping (for periodic boundary conditions).
    #[inline]
    fn sample_periodic(field: &[f64], nx: usize, ny: usize, fx: f64, fy: f64) -> f64 {
        if !fx.is_finite() || !fy.is_finite() {
            return f64::NAN;
        }
        let nxf = nx as f64;
        let nyf = ny as f64;
        let fx = ((fx % nxf) + nxf) % nxf;
        let fy = ((fy % nyf) + nyf) % nyf;

        let x0 = fx.floor() as usize % nx;
        let y0 = fy.floor() as usize % ny;
        let x1 = (x0 + 1) % nx;
        let y1 = (y0 + 1) % ny;
        let sx = fx - fx.floor();
        let sy = fy - fy.floor();

        let v00 = field[y0 * nx + x0];
        let v10 = field[y0 * nx + x1];
        let v01 = field[y1 * nx + x0];
        let v11 = field[y1 * nx + x1];

        v00 * (1.0 - sx) * (1.0 - sy)
            + v10 * sx * (1.0 - sy)
            + v01 * (1.0 - sx) * sy
            + v11 * sx * sy
    }

    /// MacCormack advection: forward step, backward correction, clamp.
    /// Reduces numerical diffusion compared to basic semi-Lagrangian.
    #[allow(clippy::too_many_arguments)]
    fn advect_maccormack(
        dst: &mut [f64],
        src: &[f64],
        vx: &[f64],
        vy: &[f64],
        nx: usize,
        ny: usize,
        dt: f64,
        inv_dx: f64,
        temp: &mut [f64],
    ) {
        let _span = trace_span!("grid::advect_maccormack", nx, ny).entered();
        let dt_inv_dx = dt * inv_dx;

        // Forward semi-Lagrangian (standard)
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 - dt_inv_dx * vx[i];
                let fy = y as f64 - dt_inv_dx * vy[i];
                dst[i] = Self::sample(src, nx, ny, fx, fy);
            }
        }

        // Backward step from the forward result
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 + dt_inv_dx * vx[i];
                let fy = y as f64 + dt_inv_dx * vy[i];
                temp[i] = Self::sample(dst, nx, ny, fx, fy);
            }
        }

        // Corrected result with clamping to neighbor min/max
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let corrected = dst[i] + 0.5 * (src[i] - temp[i]);

                // Clamp to min/max of original neighbors (loop bounds guarantee in-range)
                let s = src[i];
                let mut lo = s;
                let mut hi = s;
                lo = lo.min(src[i - 1]);
                hi = hi.max(src[i - 1]);
                lo = lo.min(src[i + 1]);
                hi = hi.max(src[i + 1]);
                lo = lo.min(src[i - nx]);
                hi = hi.max(src[i - nx]);
                lo = lo.min(src[i + nx]);
                hi = hi.max(src[i + nx]);

                dst[i] = corrected.clamp(lo, hi);
            }
        }
    }

    // ── Diffusion ───────────────────────────────────────────────────────────

    /// Diffuse a field using Gauss-Seidel iteration.
    ///
    /// Solves the implicit equation `(I - a·∇²)·x_new = x_old` where
    /// `a = dt · diff_rate / dx²`. The original field is preserved as the RHS.
    pub fn diffuse(
        field: &mut [f64],
        nx: usize,
        ny: usize,
        diff_rate: f64,
        dt: f64,
        dx: f64,
        iterations: usize,
    ) {
        let mut rhs = vec![0.0f64; nx * ny];
        Self::diffuse_with_buf(field, nx, ny, diff_rate, dt, dx, iterations, &mut rhs);
    }

    /// Diffuse using a caller-provided RHS buffer (avoids allocation).
    #[allow(clippy::too_many_arguments)]
    fn diffuse_with_buf(
        field: &mut [f64],
        nx: usize,
        ny: usize,
        diff_rate: f64,
        dt: f64,
        dx: f64,
        iterations: usize,
        rhs: &mut Vec<f64>,
    ) {
        let _span = trace_span!("grid::diffuse", nx, ny, iterations).entered();
        debug_assert_eq!(field.len(), nx * ny, "field size must match nx*ny");
        let a = dt * diff_rate / (dx * dx);
        let inv_denom = 1.0 / (1.0 + 4.0 * a);
        let n = nx * ny;
        rhs.resize(n, 0.0);
        rhs.copy_from_slice(field);
        for _ in 0..iterations {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let idx = y * nx + x;
                    let neighbors =
                        field[idx - 1] + field[idx + 1] + field[idx - nx] + field[idx + nx];
                    field[idx] = (rhs[idx] + a * neighbors) * inv_denom;
                }
            }
        }
    }

    // ── Pressure Projection ─────────────────────────────────────────────────

    /// Compute velocity divergence with correct dx scaling.
    fn divergence(div: &mut [f64], vx: &[f64], vy: &[f64], nx: usize, ny: usize, inv_2dx: f64) {
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                div[i] = -((vx[i + 1] - vx[i - 1]) + (vy[i + nx] - vy[i - nx])) * inv_2dx;
            }
        }
    }

    /// Gauss-Seidel Poisson solver: ∇²p = div.
    /// Uses the existing pressure field as initial guess (warm start).
    fn pressure_solve(pressure: &mut [f64], div: &[f64], nx: usize, ny: usize, iterations: usize) {
        let _span = trace_span!("grid::pressure_solve", nx, ny, iterations).entered();
        for _ in 0..iterations {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    let neighbors =
                        pressure[i - 1] + pressure[i + 1] + pressure[i - nx] + pressure[i + nx];
                    pressure[i] = (div[i] + neighbors) * 0.25;
                }
            }
        }
    }

    /// DST-based Poisson solver for Dirichlet (wall) boundary conditions.
    ///
    /// Solves ∇²p = div on interior cells using 2D Discrete Sine Transform.
    /// Exact solve (no iterations needed). Works for any grid size.
    fn pressure_solve_dst(pressure: &mut [f64], div: &[f64], nx: usize, ny: usize) -> Result<()> {
        let _span = trace_span!("grid::pressure_solve_dst", nx, ny).entered();
        let inx = nx - 2; // interior dimensions
        let iny = ny - 2;
        if inx == 0 || iny == 0 {
            return Ok(());
        }

        // Extract interior divergence into a flat buffer
        let mut rhs = vec![0.0f64; inx * iny];
        for iy in 0..iny {
            for ix in 0..inx {
                rhs[iy * inx + ix] = div[(iy + 1) * nx + (ix + 1)];
            }
        }

        let map_err = |_| PravashError::Diverged {
            reason: "DST/IDST transform failed in pressure solve".into(),
        };

        // 2D DST: rows then columns
        // Row DSTs
        for iy in 0..iny {
            let row: Vec<f64> = rhs[iy * inx..(iy + 1) * inx].to_vec();
            let transformed = dst(&row).map_err(map_err)?;
            rhs[iy * inx..(iy + 1) * inx].copy_from_slice(&transformed);
        }
        // Column DSTs
        let mut col = vec![0.0f64; iny];
        for ix in 0..inx {
            for iy in 0..iny {
                col[iy] = rhs[iy * inx + ix];
            }
            let transformed = dst(&col).map_err(map_err)?;
            for iy in 0..iny {
                rhs[iy * inx + ix] = transformed[iy];
            }
        }

        // Divide by eigenvalues of the discrete Laplacian (DST-I)
        // λ(kx,ky) = 2cos(π(kx+1)/(inx+1)) + 2cos(π(ky+1)/(iny+1)) - 4
        let pi = std::f64::consts::PI;
        for ky in 0..iny {
            for kx in 0..inx {
                let lx = 2.0 * (pi * (kx + 1) as f64 / (inx + 1) as f64).cos() - 2.0;
                let ly = 2.0 * (pi * (ky + 1) as f64 / (iny + 1) as f64).cos() - 2.0;
                let eigenvalue = lx + ly;
                if eigenvalue.abs() > 1e-20 {
                    rhs[ky * inx + kx] /= eigenvalue;
                }
            }
        }

        // 2D IDST: rows then columns
        for iy in 0..iny {
            let row: Vec<f64> = rhs[iy * inx..(iy + 1) * inx].to_vec();
            let transformed = idst(&row).map_err(map_err)?;
            rhs[iy * inx..(iy + 1) * inx].copy_from_slice(&transformed);
        }
        for ix in 0..inx {
            for iy in 0..iny {
                col[iy] = rhs[iy * inx + ix];
            }
            let transformed = idst(&col).map_err(map_err)?;
            for iy in 0..iny {
                rhs[iy * inx + ix] = transformed[iy];
            }
        }

        // Write back to pressure (interior only, boundaries stay zero)
        for i in pressure.iter_mut() {
            *i = 0.0;
        }
        for iy in 0..iny {
            for ix in 0..inx {
                pressure[(iy + 1) * nx + (ix + 1)] = rhs[iy * inx + ix];
            }
        }
        Ok(())
    }

    /// Subtract pressure gradient from velocity to enforce divergence-free.
    fn project_velocity(
        vx: &mut [f64],
        vy: &mut [f64],
        pressure: &[f64],
        nx: usize,
        ny: usize,
        inv_2dx: f64,
    ) {
        let _span = trace_span!("grid::project").entered();

        #[cfg(feature = "parallel")]
        {
            // Compute corrections, then apply (avoids split-borrow issues)
            let corrections: Vec<(f64, f64)> = (1..ny - 1)
                .into_par_iter()
                .flat_map(|y| {
                    (1..nx - 1).into_par_iter().map(move |x| {
                        let i = y * nx + x;
                        (
                            (pressure[i + 1] - pressure[i - 1]) * inv_2dx,
                            (pressure[i + nx] - pressure[i - nx]) * inv_2dx,
                        )
                    })
                })
                .collect();
            let mut idx = 0;
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    vx[i] -= corrections[idx].0;
                    vy[i] -= corrections[idx].1;
                    idx += 1;
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    vx[i] -= (pressure[i + 1] - pressure[i - 1]) * inv_2dx;
                    vy[i] -= (pressure[i + nx] - pressure[i - nx]) * inv_2dx;
                }
            }
        }
    }

    // ── Vorticity Confinement ───────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn apply_vorticity_confinement(
        vx: &mut [f64],
        vy: &mut [f64],
        nx: usize,
        ny: usize,
        dx: f64,
        dt: f64,
        epsilon: f64,
        vort_buf: &mut [f64],
    ) {
        let _span = trace_span!("grid::vorticity_confinement").entered();
        let inv_2dx = 0.5 / dx;

        // Compute vorticity: ω = ∂vy/∂x - ∂vx/∂y
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let dvydx = (vy[i + 1] - vy[i - 1]) * inv_2dx;
                let dvxdy = (vx[i + nx] - vx[i - nx]) * inv_2dx;
                vort_buf[i] = dvydx - dvxdy;
            }
        }

        for y in 2..ny.saturating_sub(2) {
            for x in 2..nx.saturating_sub(2) {
                let i = y * nx + x;
                let abs_w = vort_buf[i].abs();
                if abs_w < 1e-10 {
                    continue;
                }

                let eta_x = (vort_buf[i + 1].abs() - vort_buf[i - 1].abs()) * inv_2dx;
                let eta_y = (vort_buf[i + nx].abs() - vort_buf[i - nx].abs()) * inv_2dx;
                let eta_mag = (eta_x * eta_x + eta_y * eta_y).sqrt();
                if eta_mag < 1e-10 {
                    continue;
                }

                let nx_v = eta_x / eta_mag;
                let ny_v = eta_y / eta_mag;
                // 2D: N × ω = (ny·ω, -nx·ω)
                vx[i] += epsilon * dx * dt * ny_v * vort_buf[i];
                vy[i] -= epsilon * dx * dt * nx_v * vort_buf[i];
            }
        }
    }

    // ── Boundary Conditions ─────────────────────────────────────────────────

    fn enforce_boundary(
        vx: &mut [f64],
        vy: &mut [f64],
        nx: usize,
        ny: usize,
        bc: BoundaryCondition,
    ) {
        match bc {
            BoundaryCondition::NoSlip => {
                for x in 0..nx {
                    vx[x] = -vx[nx + x];
                    vy[x] = -vy[nx + x];
                    let top = (ny - 1) * nx + x;
                    let below = (ny - 2) * nx + x;
                    vx[top] = -vx[below];
                    vy[top] = -vy[below];
                }
                for y in 0..ny {
                    let l = y * nx;
                    vx[l] = -vx[l + 1];
                    vy[l] = -vy[l + 1];
                    let r = y * nx + nx - 1;
                    vx[r] = -vx[r - 1];
                    vy[r] = -vy[r - 1];
                }
            }
            BoundaryCondition::FreeSlip => {
                for x in 0..nx {
                    vx[x] = vx[nx + x];
                    vy[x] = 0.0;
                    let top = (ny - 1) * nx + x;
                    let below = (ny - 2) * nx + x;
                    vx[top] = vx[below];
                    vy[top] = 0.0;
                }
                for y in 0..ny {
                    let l = y * nx;
                    vx[l] = 0.0;
                    vy[l] = vy[l + 1];
                    let r = y * nx + nx - 1;
                    vx[r] = 0.0;
                    vy[r] = vy[r - 1];
                }
            }
            BoundaryCondition::Periodic => {
                for x in 0..nx {
                    vx[x] = vx[(ny - 2) * nx + x];
                    vy[x] = vy[(ny - 2) * nx + x];
                    vx[(ny - 1) * nx + x] = vx[nx + x];
                    vy[(ny - 1) * nx + x] = vy[nx + x];
                }
                for y in 0..ny {
                    vx[y * nx] = vx[y * nx + nx - 2];
                    vy[y * nx] = vy[y * nx + nx - 2];
                    vx[y * nx + nx - 1] = vx[y * nx + 1];
                    vy[y * nx + nx - 1] = vy[y * nx + 1];
                }
                // Fix corners: must be consistent diagonal periodic cell
                let c = (ny - 2) * nx + nx - 2;
                vx[0] = vx[c];
                vy[0] = vy[c];
                vx[nx - 1] = vx[(ny - 2) * nx + 1];
                vy[nx - 1] = vy[(ny - 2) * nx + 1];
                vx[(ny - 1) * nx] = vx[nx + nx - 2];
                vy[(ny - 1) * nx] = vy[nx + nx - 2];
                vx[(ny - 1) * nx + nx - 1] = vx[nx + 1];
                vy[(ny - 1) * nx + nx - 1] = vy[nx + 1];
            }
        }
    }

    fn enforce_scalar_boundary(field: &mut [f64], nx: usize, ny: usize) {
        for x in 0..nx {
            field[x] = field[nx + x];
            field[(ny - 1) * nx + x] = field[(ny - 2) * nx + x];
        }
        for y in 0..ny {
            field[y * nx] = field[y * nx + 1];
            field[y * nx + nx - 1] = field[y * nx + nx - 2];
        }
    }

    // ── Full Simulation Step ────────────────────────────────────────────────

    /// Perform one Navier-Stokes simulation step.
    ///
    /// Pipeline: advect velocity → diffuse → apply forces (buoyancy, vorticity
    /// confinement) → pressure projection → advect density.
    pub fn step(&mut self, config: &GridConfig) -> Result<()> {
        let _span = trace_span!("grid::step", nx = self.nx, ny = self.ny).entered();

        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt: config.dt });
        }
        if config.viscosity < 0.0 || !config.viscosity.is_finite() {
            return Err(PravashError::InvalidViscosity {
                viscosity: config.viscosity,
            });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = config.dt;
        let n = nx * ny;
        let inv_dx = 1.0 / dx;
        let inv_2dx = 0.5 * inv_dx;

        // Take scratch buffers out of self to avoid borrow conflicts with field slices.
        // Resize handles post-deserialization empty buffers; fill zeroes for reuse.
        let mut sa = std::mem::take(&mut self.scratch_a);
        let mut sb = std::mem::take(&mut self.scratch_b);
        let mut sc = std::mem::take(&mut self.scratch_c);
        let mut drhs = std::mem::take(&mut self.diffuse_rhs);
        sa.resize(n, 0.0);
        sb.resize(n, 0.0);
        sc.resize(n, 0.0);
        drhs.resize(n, 0.0);
        sa.fill(0.0);
        sb.fill(0.0);

        // 1. Advect velocity
        if config.use_maccormack && config.boundary != BoundaryCondition::Periodic {
            sc.fill(0.0);
            Self::advect_maccormack(
                &mut sa, &self.vx, &self.vx, &self.vy, nx, ny, dt, inv_dx, &mut sc,
            );
            Self::advect_maccormack(
                &mut sb, &self.vy, &self.vx, &self.vy, nx, ny, dt, inv_dx, &mut sc,
            );
            self.vx.copy_from_slice(&sa);
            self.vy.copy_from_slice(&sb);
        } else if config.boundary == BoundaryCondition::Periodic {
            Self::advect_periodic(&mut sa, &self.vx, &self.vx, &self.vy, nx, ny, dt, inv_dx);
            Self::advect_periodic(&mut sb, &self.vy, &self.vx, &self.vy, nx, ny, dt, inv_dx);
            self.vx.copy_from_slice(&sa);
            self.vy.copy_from_slice(&sb);
        } else {
            Self::advect(&mut sa, &self.vx, &self.vx, &self.vy, nx, ny, dt, inv_dx);
            Self::advect(&mut sb, &self.vy, &self.vx, &self.vy, nx, ny, dt, inv_dx);
            self.vx.copy_from_slice(&sa);
            self.vy.copy_from_slice(&sb);
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 2. Diffuse velocity (with Smagorinsky SGS turbulent viscosity if enabled)
        let eff_viscosity = if config.smagorinsky_cs > 0.0 {
            // ν_t = (Cs·dx)²·|S| averaged over the grid
            let cs_dx = config.smagorinsky_cs * dx;
            let cs_dx2 = cs_dx * cs_dx;
            let inv_2dx_local = inv_2dx;
            let mut sum_nu_t = 0.0;
            let mut count = 0.0;
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    let dudx = (self.vx[i + 1] - self.vx[i - 1]) * inv_2dx_local;
                    let dudy = (self.vx[i + nx] - self.vx[i - nx]) * inv_2dx_local;
                    let dvdx = (self.vy[i + 1] - self.vy[i - 1]) * inv_2dx_local;
                    let dvdy = (self.vy[i + nx] - self.vy[i - nx]) * inv_2dx_local;
                    let s11 = dudx;
                    let s22 = dvdy;
                    let s12 = 0.5 * (dudy + dvdx);
                    let s_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();
                    sum_nu_t += cs_dx2 * s_mag;
                    count += 1.0;
                }
            }
            config.viscosity + if count > 0.0 { sum_nu_t / count } else { 0.0 }
        } else {
            config.viscosity
        };

        if eff_viscosity > 0.0 {
            Self::diffuse_with_buf(
                &mut self.vx,
                nx,
                ny,
                eff_viscosity,
                dt,
                dx,
                config.diffusion_iterations,
                &mut drhs,
            );
            Self::diffuse_with_buf(
                &mut self.vy,
                nx,
                ny,
                eff_viscosity,
                dt,
                dx,
                config.diffusion_iterations,
                &mut drhs,
            );
            Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);
        }

        // 3. Apply forces
        if config.buoyancy_alpha != 0.0 {
            for i in 0..n {
                self.vy[i] +=
                    config.buoyancy_alpha * (self.density[i] - config.ambient_density) * dt;
            }
        }

        if config.vorticity_confinement > 0.0 {
            sa.fill(0.0);
            Self::apply_vorticity_confinement(
                &mut self.vx,
                &mut self.vy,
                nx,
                ny,
                dx,
                dt,
                config.vorticity_confinement,
                &mut sa,
            );
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 4. Pressure projection
        {
            sa.fill(0.0);
            Self::divergence(&mut sa, &self.vx, &self.vy, nx, ny, inv_2dx);
            if config.boundary == BoundaryCondition::Periodic {
                Self::pressure_solve(&mut self.pressure, &sa, nx, ny, config.pressure_iterations);
            } else {
                Self::pressure_solve_dst(&mut self.pressure, &sa, nx, ny)?;
            }
            Self::project_velocity(&mut self.vx, &mut self.vy, &self.pressure, nx, ny, inv_2dx);
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 5. Advect density
        if config.use_maccormack && config.boundary != BoundaryCondition::Periodic {
            sc.fill(0.0);
            Self::advect_maccormack(
                &mut sa,
                &self.density,
                &self.vx,
                &self.vy,
                nx,
                ny,
                dt,
                inv_dx,
                &mut sc,
            );
            self.density.copy_from_slice(&sa);
        } else if config.boundary == BoundaryCondition::Periodic {
            Self::advect_periodic(
                &mut sa,
                &self.density,
                &self.vx,
                &self.vy,
                nx,
                ny,
                dt,
                inv_dx,
            );
            self.density.copy_from_slice(&sa);
        } else {
            Self::advect(
                &mut sa,
                &self.density,
                &self.vx,
                &self.vy,
                nx,
                ny,
                dt,
                inv_dx,
            );
            self.density.copy_from_slice(&sa);
        }
        if config.boundary != BoundaryCondition::Periodic {
            Self::enforce_scalar_boundary(&mut self.density, nx, ny);
        }

        // Return scratch buffers to self
        self.scratch_a = sa;
        self.scratch_b = sb;
        self.scratch_c = sc;
        self.diffuse_rhs = drhs;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_new() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert_eq!(g.cell_count(), 100);
        assert_eq!(g.vx.len(), 100);
    }

    #[test]
    fn test_grid_invalid() {
        assert!(FluidGrid::new(0, 10, 0.1).is_err());
        assert!(FluidGrid::new(10, 0, 0.1).is_err());
        // Below minimum size
        assert!(FluidGrid::new(3, 10, 0.1).is_err());
        assert!(FluidGrid::new(10, 3, 0.1).is_err());
    }

    #[test]
    fn test_grid_invalid_dx() {
        assert!(FluidGrid::new(10, 10, 0.0).is_err());
        assert!(FluidGrid::new(10, 10, -1.0).is_err());
        assert!(FluidGrid::new(10, 10, f64::NAN).is_err());
    }

    #[test]
    fn test_grid_idx() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(5, 3), 35);
    }

    #[test]
    fn test_velocity_at() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let i = g.idx(3, 4);
        g.vx[i] = 1.5;
        g.vy[i] = 2.5;
        let (vx, vy) = g.velocity_at(3, 4);
        assert!((vx - 1.5).abs() < f64::EPSILON);
        assert!((vy - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_at() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let i = g.idx(5, 5);
        g.vx[i] = 3.0;
        g.vy[i] = 4.0;
        assert!((g.speed_at(5, 5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_speed_zero() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert!(g.max_speed().abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_speed() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        g.vx[50] = 3.0;
        g.vy[50] = 4.0;
        assert!((g.max_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diffuse_preserves_zeros() {
        let mut field = vec![0.0; 100];
        FluidGrid::diffuse(&mut field, 10, 10, 0.01, 0.001, 0.1, 10);
        assert!(field.iter().all(|&v| v.abs() < f64::EPSILON));
    }

    #[test]
    fn test_diffuse_spreads() {
        let mut field = vec![0.0; 100];
        field[55] = 100.0;
        FluidGrid::diffuse(&mut field, 10, 10, 0.1, 0.01, 0.1, 20);
        assert!(field[54] > 0.0 || field[56] > 0.0);
    }

    #[test]
    fn test_grid_serde() {
        let g = FluidGrid::new(5, 5, 0.2).unwrap();
        let json = serde_json::to_string(&g).unwrap();
        let g2: FluidGrid = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.nx, 5);
        assert_eq!(g2.ny, 5);
    }

    // ── Advection tests ─────────────────────────────────────────────────────

    #[test]
    fn test_sample_center() {
        let nx = 4;
        let ny = 4;
        let mut field = vec![0.0; nx * ny];
        field[nx + 1] = 1.0;
        let val = FluidGrid::sample(&field, nx, ny, 1.0, 1.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_interpolation() {
        let nx = 4;
        let ny = 4;
        let mut field = vec![0.0; nx * ny];
        field[nx + 1] = 1.0;
        field[nx + 2] = 3.0;
        let val = FluidGrid::sample(&field, nx, ny, 1.5, 1.0);
        assert!((val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_advect_zero_velocity() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let mut src = vec![0.0; n];
        src[4 * nx + 4] = 1.0;
        let mut dst = vec![0.0; n];
        FluidGrid::advect(&mut dst, &src, &vx, &vy, nx, ny, 0.1, 1.0);
        assert!((dst[4 * nx + 4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_advect_uniform_velocity() {
        let nx = 16;
        let ny = 16;
        let n = nx * ny;
        // Velocity = 1 cell/s (with inv_dx=1.0)
        let vx = vec![1.0; n];
        let vy = vec![0.0; n];
        let mut src = vec![0.0; n];
        src[8 * nx + 8] = 1.0;
        let mut dst = vec![0.0; n];
        FluidGrid::advect(&mut dst, &src, &vx, &vy, nx, ny, 1.0, 1.0);
        // dst[8,9] traces back to (8,8) which has 1.0
        assert!(
            dst[8 * nx + 9] > 0.5,
            "advected value should appear at (9,8): got {}",
            dst[8 * nx + 9]
        );
    }

    // ── Pressure projection tests ───────────────────────────────────────────

    #[test]
    fn test_divergence_zero_field() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let mut div = vec![0.0; n];
        FluidGrid::divergence(&mut div, &vx, &vy, nx, ny, 0.5);
        assert!(div.iter().all(|&v| v.abs() < f64::EPSILON));
    }

    #[test]
    fn test_pressure_solve_zero_rhs() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let div = vec![0.0; n];
        let mut pressure = vec![0.0; n];
        FluidGrid::pressure_solve(&mut pressure, &div, nx, ny, 50);
        // Zero RHS + zero initial guess → stays zero
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                assert!(pressure[y * nx + x].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_projection_reduces_divergence() {
        let nx = 15;
        let ny = 15;
        let n = nx * ny;
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        let cx = nx / 2;
        let cy = ny / 2;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                vx[i] = (x as f64 - cx as f64) * 0.1;
                vy[i] = (y as f64 - cy as f64) * 0.1;
            }
        }

        let inv_2dx = 0.5;
        let mut div_before = vec![0.0; n];
        FluidGrid::divergence(&mut div_before, &vx, &vy, nx, ny, inv_2dx);
        let mag_before: f64 = div_before.iter().map(|d| d * d).sum();

        let mut pressure = vec![0.0; n];
        FluidGrid::pressure_solve(&mut pressure, &div_before, nx, ny, 100);
        FluidGrid::project_velocity(&mut vx, &mut vy, &pressure, nx, ny, inv_2dx);

        let mut div_after = vec![0.0; n];
        FluidGrid::divergence(&mut div_after, &vx, &vy, nx, ny, inv_2dx);
        let mag_after: f64 = div_after.iter().map(|d| d * d).sum();

        assert!(
            mag_after < mag_before,
            "projection should reduce divergence: before={mag_before}, after={mag_after}"
        );
    }

    // ── Boundary condition tests ────────────────────────────────────────────

    #[test]
    fn test_no_slip_boundary() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let mut vx = vec![1.0; n];
        let mut vy = vec![1.0; n];
        FluidGrid::enforce_boundary(&mut vx, &mut vy, nx, ny, BoundaryCondition::NoSlip);
        // Left wall, row 1: vx[1*8+0] should mirror -vx[1*8+1]
        assert!((vx[nx] + vx[nx + 1]).abs() < 1e-10);
        // Bottom wall, col 1: vy[0*8+1] should mirror -vy[1*8+1]
        assert!((vy[1] + vy[nx + 1]).abs() < 1e-10);
    }

    #[test]
    fn test_free_slip_boundary() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let mut vx = vec![1.0; n];
        let mut vy = vec![2.0; n];
        FluidGrid::enforce_boundary(&mut vx, &mut vy, nx, ny, BoundaryCondition::FreeSlip);
        assert!(vx[0].abs() < f64::EPSILON); // left wall: normal vx = 0
        assert!(vy[1].abs() < f64::EPSILON); // bottom wall: normal vy = 0
    }

    // ── Full step tests ─────────────────────────────────────────────────────

    #[test]
    fn test_step_empty_grid_stable() {
        let mut g = FluidGrid::new(16, 16, 0.1).unwrap();
        let config = GridConfig::smoke();
        g.step(&config).unwrap();
        assert!(g.max_speed() < f64::EPSILON);
    }

    #[test]
    fn test_step_smoke_plume() {
        let nx = 30;
        let dx = 1.0 / nx as f64;
        let mut g = FluidGrid::new(nx, nx, dx).unwrap();
        let mut config = GridConfig::smoke();
        config.dt = 0.01;

        for x in 13..17 {
            let i = 2 * nx + x;
            g.density[i] = 1.0;
            g.vy[i] = 0.5;
        }

        for _ in 0..50 {
            for x in 13..17 {
                g.density[2 * nx + x] = 1.0;
            }
            g.step(&config).unwrap();
        }

        // Velocity should develop and all values should be finite
        assert!(g.max_speed().is_finite());
        for v in &g.density {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_step_invalid_dt() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        let mut config = GridConfig::smoke();
        config.dt = -1.0;
        assert!(g.step(&config).is_err());
    }

    #[test]
    fn test_step_nan_dt() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        let mut config = GridConfig::smoke();
        config.dt = f64::NAN;
        assert!(g.step(&config).is_err());
    }

    #[test]
    fn test_step_no_divergence_after_projection() {
        let mut g = FluidGrid::new(15, 15, 0.1).unwrap();
        let config = GridConfig::smoke();

        let center = 7 * 15 + 7;
        g.vx[center] = 5.0;
        g.vy[center] = 3.0;

        g.step(&config).unwrap();

        let mut div = vec![0.0; 15 * 15];
        FluidGrid::divergence(&mut div, &g.vx, &g.vy, 15, 15, 0.5 / g.dx);
        let max_div: f64 = div.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
        assert!(
            max_div < 5.0,
            "divergence should be bounded after projection: {max_div}"
        );
    }

    // ── MacCormack tests ────────────────────────────────────────────────────

    #[test]
    fn test_maccormack_zero_velocity() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let mut src = vec![0.0; n];
        src[4 * nx + 4] = 1.0;
        let mut dst = vec![0.0; n];
        let mut temp = vec![0.0; n];
        FluidGrid::advect_maccormack(&mut dst, &src, &vx, &vy, nx, ny, 0.1, 1.0, &mut temp);
        assert!((dst[4 * nx + 4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_maccormack_less_diffusive() {
        let nx = 32;
        let ny = 32;
        let n = nx * ny;
        let vx = vec![1.0; n];
        let vy = vec![0.0; n];

        // Create a narrow peak
        let mut src = vec![0.0; n];
        src[16 * nx + 16] = 1.0;

        let mut dst_sl = vec![0.0; n];
        FluidGrid::advect(&mut dst_sl, &src, &vx, &vy, nx, ny, 1.0, 1.0);

        let mut dst_mc = vec![0.0; n];
        let mut temp = vec![0.0; n];
        FluidGrid::advect_maccormack(&mut dst_mc, &src, &vx, &vy, nx, ny, 1.0, 1.0, &mut temp);

        // MacCormack should preserve the peak better (higher max value)
        let max_sl: f64 = dst_sl.iter().cloned().fold(0.0f64, f64::max);
        let max_mc: f64 = dst_mc.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max_mc >= max_sl,
            "MacCormack should be at least as sharp: mc={max_mc}, sl={max_sl}"
        );
    }

    #[test]
    fn test_step_maccormack_stable() {
        let mut g = FluidGrid::new(16, 16, 0.1).unwrap();
        let mut config = GridConfig::smoke();
        config.use_maccormack = true;
        config.dt = 0.01;

        let center = 8 * 16 + 8;
        g.density[center] = 1.0;
        g.vy[center] = 0.5;

        for _ in 0..20 {
            g.step(&config).unwrap();
        }
        assert!(g.max_speed().is_finite());
    }

    // ── Periodic boundary tests ─────────────────────────────────────────────

    #[test]
    fn test_periodic_boundary_enforcement() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        // Set interior value
        vx[nx + 1] = 1.0;
        FluidGrid::enforce_boundary(&mut vx, &mut vy, nx, ny, BoundaryCondition::Periodic);
        // Right edge should match left interior
        assert!((vx[nx + nx - 1] - vx[nx + 1]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_step_periodic_stable() {
        let mut g = FluidGrid::new(16, 16, 0.1).unwrap();
        let mut config = GridConfig::smoke();
        config.boundary = BoundaryCondition::Periodic;
        config.dt = 0.01;

        let center = 8 * 16 + 8;
        g.vx[center] = 1.0;

        for _ in 0..20 {
            g.step(&config).unwrap();
        }
        assert!(g.max_speed().is_finite());
    }

    #[test]
    fn test_sample_periodic_wrapping() {
        let nx = 4;
        let ny = 4;
        let mut field = vec![0.0; nx * ny];
        field[0] = 1.0; // (0,0)
        // Sampling at (-0.5, 0) should wrap to right side
        let val = FluidGrid::sample_periodic(&field, nx, ny, -0.5, 0.0);
        assert!(val.is_finite());
    }

    #[test]
    fn test_no_alloc_after_first_step() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let config = GridConfig::smoke();
        g.step(&config).unwrap();
        let cap_a = g.scratch_a.capacity();
        let cap_b = g.scratch_b.capacity();
        g.step(&config).unwrap();
        assert_eq!(g.scratch_a.capacity(), cap_a);
        assert_eq!(g.scratch_b.capacity(), cap_b);
    }

    #[test]
    fn test_grid_serde_preserves_state() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        let config = GridConfig::smoke();
        let center = g.idx(4, 4);
        g.density[center] = 1.0;
        g.step(&config).unwrap();

        let json = serde_json::to_string(&g).unwrap();
        let mut g2: FluidGrid = serde_json::from_str(&json).unwrap();
        // Scratch buffers are skipped but should be rebuilt on next step
        g2.step(&config).unwrap();
        assert!(g2.max_speed().is_finite());
    }

    #[test]
    fn test_grid_step_deterministic() {
        let config = GridConfig::smoke();
        let run = || -> FluidGrid {
            let mut g = FluidGrid::new(20, 20, 0.1).unwrap();
            let center = g.idx(10, 10);
            g.density[center] = 1.0;
            g.vy[center] = 0.5;
            for _ in 0..10 {
                g.step(&config).unwrap();
            }
            g
        };
        let g1 = run();
        let g2 = run();
        for i in 0..g1.vx.len() {
            assert!(
                (g1.vx[i] - g2.vx[i]).abs() < 1e-10,
                "grid step should be deterministic"
            );
        }
    }

    #[test]
    fn test_free_slip_tangential_preserved() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        // Set tangential velocity on left boundary
        let i = g.idx(1, 4);
        g.vy[i] = 1.0;
        FluidGrid::enforce_boundary(
            &mut g.vx,
            &mut g.vy,
            g.nx,
            g.ny,
            BoundaryCondition::FreeSlip,
        );
        // Left boundary: normal (vx) should be 0, tangential (vy) should be preserved
        let l = g.idx(0, 4);
        assert!(g.vx[l].abs() < 1e-10, "free-slip: normal vx should be 0");
        assert!(
            (g.vy[l] - 1.0).abs() < 1e-10,
            "free-slip: tangential vy should be preserved: {}",
            g.vy[l]
        );
    }
}
