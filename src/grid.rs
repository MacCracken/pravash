//! Grid-based Euler/Navier-Stokes fluid solver.
//!
//! 2D incompressible Navier-Stokes on a uniform grid using operator splitting:
//! advection (semi-Lagrangian) → diffusion (Gauss-Seidel) → force application
//! → pressure projection (Gauss-Seidel Poisson solver).

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

use hisab::num::{conjugate_gradient, dst, idst};
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
    /// Opposite edges connected (wrapping) for advection and velocity ghosts.
    /// The pressure solve still uses Gauss-Seidel with zero (Dirichlet)
    /// boundary pressure — a true periodic solve is on the roadmap.
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
    /// Use MacCormack advection (higher-order, less diffusive), limited to
    /// the bilinear stencil of the backtraced point. Ignored for `Periodic`.
    pub use_maccormack: bool,
    /// Use BFECC advection (Back and Forth Error Compensation and Correction).
    /// Better feature preservation than MacCormack. Overrides MacCormack if both set.
    /// Default: false.
    pub use_bfecc: bool,
    /// Smagorinsky SGS turbulence model coefficient (0.0 to disable).
    /// ν_t = (Cs·dx)²·|S|. Typical Cs: 0.1–0.2. Default: 0.0.
    pub smagorinsky_cs: f64,
    /// Use multigrid V-cycles for the pressure solve (O(N) per cycle).
    /// Runs `pressure_iterations / 10` (at least 1) V-cycles, about the work
    /// of `pressure_iterations` Gauss-Seidel sweeps. When disabled, DST is
    /// used for walls and GS for periodic. Default: false.
    pub use_multigrid: bool,
    /// Use conjugate gradient for pressure solve.
    /// Matrix-free CG via hisab. Good for large grids where GS converges slowly.
    /// Takes priority over multigrid/DST/GS when enabled.
    /// Default: false.
    pub use_cg: bool,
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
            use_bfecc: false,
            smagorinsky_cs: 0.0,
            use_multigrid: false,
            use_cg: false,
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
    /// Persistent multigrid level buffers (residual, coarse_rhs, coarse_p per level).
    #[serde(skip)]
    mg_levels: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
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
            mg_levels: Vec::new(),
        })
    }

    #[inline]
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    #[inline]
    #[must_use]
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
        (cfl_factor * dt).min(f64::MAX)
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

    /// Min/max of the four bilinear stencil values used by [`sample`](Self::sample)
    /// at `(fx, fy)` — the limiter bounds for MacCormack/BFECC (Selle et al. 2008).
    #[inline]
    fn sample_bounds(field: &[f64], nx: usize, ny: usize, fx: f64, fy: f64) -> (f64, f64) {
        if !fx.is_finite() || !fy.is_finite() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let x0 = fx.floor().max(0.0).min((nx - 2) as f64) as usize;
        let y0 = fy.floor().max(0.0).min((ny - 2) as f64) as usize;
        let v00 = field[y0 * nx + x0];
        let v10 = field[y0 * nx + x0 + 1];
        let v01 = field[(y0 + 1) * nx + x0];
        let v11 = field[(y0 + 1) * nx + x0 + 1];
        (
            v00.min(v10).min(v01).min(v11),
            v00.max(v10).max(v01).max(v11),
        )
    }

    /// Clamp that never panics: returns `v` unchanged if the bounds are NaN or inverted.
    #[inline]
    fn clamp_safe(v: f64, lo: f64, hi: f64) -> f64 {
        if lo <= hi { v.clamp(lo, hi) } else { v }
    }

    /// Copy the boundary ring of `src` into `dst`.
    fn copy_ring(dst: &mut [f64], src: &[f64], nx: usize, ny: usize) {
        dst[..nx].copy_from_slice(&src[..nx]);
        let top = (ny - 1) * nx;
        dst[top..top + nx].copy_from_slice(&src[top..top + nx]);
        for y in 1..ny - 1 {
            dst[y * nx] = src[y * nx];
            dst[y * nx + nx - 1] = src[y * nx + nx - 1];
        }
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
    ///
    /// The corrected value is clamped to the bilinear stencil of the
    /// backtraced point, so the scheme never creates new extrema.
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

        // The backward pass samples the forward result near the walls too:
        // give it the source's boundary ring instead of stale scratch data.
        Self::copy_ring(dst, src, nx, ny);

        // Backward step from the forward result
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 + dt_inv_dx * vx[i];
                let fy = y as f64 + dt_inv_dx * vy[i];
                temp[i] = Self::sample(dst, nx, ny, fx, fy);
            }
        }

        // Corrected result, clamped to the stencil of the backtraced point
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let corrected = dst[i] + 0.5 * (src[i] - temp[i]);
                let fx = x as f64 - dt_inv_dx * vx[i];
                let fy = y as f64 - dt_inv_dx * vy[i];
                let (lo, hi) = Self::sample_bounds(src, nx, ny, fx, fy);
                dst[i] = Self::clamp_safe(corrected, lo, hi);
            }
        }
    }

    /// BFECC advection: forward, backward, error estimate, corrected forward.
    /// Better feature preservation than MacCormack.
    ///
    /// φ̂ = B(F(φⁿ)), φ̃ = φⁿ + ½(φⁿ − φ̂), φⁿ⁺¹ = F(φ̃), clamped to the stencil
    /// of the backtraced point in φⁿ.
    #[allow(clippy::too_many_arguments)]
    fn advect_bfecc(
        dst: &mut [f64],
        src: &[f64],
        vx: &[f64],
        vy: &[f64],
        nx: usize,
        ny: usize,
        dt: f64,
        inv_dx: f64,
        temp1: &mut [f64],
        temp2: &mut [f64],
    ) {
        let _span = trace_span!("grid::advect_bfecc", nx, ny).entered();
        let dt_inv_dx = dt * inv_dx;

        // Step 1: Forward semi-Lagrangian → temp1
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 - dt_inv_dx * vx[i];
                let fy = y as f64 - dt_inv_dx * vy[i];
                temp1[i] = Self::sample(src, nx, ny, fx, fy);
            }
        }

        // Step 2: Backward from temp1 → temp2
        Self::copy_ring(temp1, src, nx, ny);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 + dt_inv_dx * vx[i];
                let fy = y as f64 + dt_inv_dx * vy[i];
                temp2[i] = Self::sample(temp1, nx, ny, fx, fy);
            }
        }

        // Step 3: Error-compensated source φ̃ = φ + ½(φ − φ̂) → temp2
        Self::copy_ring(temp2, src, nx, ny);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                temp2[i] = src[i] + 0.5 * (src[i] - temp2[i]);
            }
        }

        // Step 4: Forward advect the compensated source, clamp to original stencil
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 - dt_inv_dx * vx[i];
                let fy = y as f64 - dt_inv_dx * vy[i];
                let v = Self::sample(temp2, nx, ny, fx, fy);
                let (lo, hi) = Self::sample_bounds(src, nx, ny, fx, fy);
                dst[i] = Self::clamp_safe(v, lo, hi);
            }
        }
    }

    // ── Diffusion ───────────────────────────────────────────────────────────

    /// Diffuse a field using Gauss-Seidel iteration.
    ///
    /// Solves the implicit equation `(I - a·∇²)·x_new = x_old` where
    /// `a = dt · diff_rate / dx²`. The original field is preserved as the RHS.
    ///
    /// Does nothing (with a warning) if `field.len() != nx·ny` or the grid is
    /// smaller than 3×3.
    pub fn diffuse(
        field: &mut [f64],
        nx: usize,
        ny: usize,
        diff_rate: f64,
        dt: f64,
        dx: f64,
        iterations: usize,
    ) {
        let _span = trace_span!("grid::diffuse", nx, ny).entered();
        if nx < 3 || ny < 3 || nx.checked_mul(ny) != Some(field.len()) {
            tracing::warn!(
                nx,
                ny,
                len = field.len(),
                "grid::diffuse: size mismatch ignored"
            );
            return;
        }
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

    /// Compute `-(central difference sum) · scale` at interior cells.
    ///
    /// With `scale = 1/(2dx)` this is the negated physical divergence `-∇·u`.
    /// The projection passes `scale = dx/2`, i.e. `-dx²·∇·u`, which is the
    /// right-hand side of the unit-stencil pressure Poisson equation.
    fn divergence(div: &mut [f64], vx: &[f64], vy: &[f64], nx: usize, ny: usize, scale: f64) {
        let _span = trace_span!("grid::divergence", nx, ny).entered();
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                div[i] = -((vx[i + 1] - vx[i - 1]) + (vy[i + nx] - vy[i - nx])) * scale;
            }
        }
    }

    /// Gauss-Seidel Poisson solver for the unit-stencil system
    /// `4·p[i] − Σ p[neighbors] = rhs[i]`, i.e. `−dx²·∇²p = rhs`.
    ///
    /// All pressure solvers (GS, DST, multigrid, CG) solve this same system
    /// with homogeneous Dirichlet boundaries. With `rhs = −dx²·∇·u` the
    /// subsequent `u −= ∇p` makes `u` divergence-free.
    /// Uses the existing pressure field as initial guess (warm start).
    fn pressure_solve(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        iterations: usize,
    ) -> Result<()> {
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
        Ok(())
    }

    /// DST-based Poisson solver for Dirichlet (wall) boundary conditions.
    ///
    /// Solves `4p − Σn = rhs` on interior cells using a 2D Discrete Sine Transform.
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

        // Divide by eigenvalues of the unit-stencil operator 4p − Σn (DST-I):
        // μ(kx,ky) = 4 − 2cos(π(kx+1)/(inx+1)) − 2cos(π(ky+1)/(iny+1)) > 0
        let pi = std::f64::consts::PI;
        for ky in 0..iny {
            for kx in 0..inx {
                let lx = 2.0 * (pi * (kx + 1) as f64 / (inx + 1) as f64).cos() - 2.0;
                let ly = 2.0 * (pi * (ky + 1) as f64 / (iny + 1) as f64).cos() - 2.0;
                let eigenvalue = -(lx + ly);
                if eigenvalue.abs() > 1e-12 {
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

    /// Multigrid V-cycle Poisson solver. O(N) total work.
    ///
    /// Recursively restricts the residual to coarser grids, solves, and
    /// prolongates the correction. Uses Gauss-Seidel smoothing.
    /// `levels` provides persistent scratch buffers (residual, coarse_rhs, coarse_p) per level.
    /// Runs `cycles` V-cycles with `nu_smooth` Gauss-Seidel sweeps per level.
    fn pressure_solve_multigrid(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        nu_smooth: usize,
        cycles: usize,
        levels: &mut Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    ) -> Result<()> {
        let _span = trace_span!("grid::multigrid", nx, ny, cycles).entered();
        for _ in 0..cycles.max(1) {
            Self::mg_vcycle(pressure, div, nx, ny, nu_smooth, levels, 0);
        }
        Ok(())
    }

    fn mg_vcycle(
        p: &mut [f64],
        rhs: &[f64],
        nx: usize,
        ny: usize,
        nu: usize,
        levels: &mut Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
        depth: usize,
    ) {
        if nx <= 6 || ny <= 6 || depth >= 10 {
            let _ = Self::pressure_solve(p, rhs, nx, ny, 50);
            return;
        }

        // Pre-smooth
        let _ = Self::pressure_solve(p, rhs, nx, ny, nu);

        let n = nx * ny;
        // Vertex-centred coarsening: coarse (cx, cy) ↔ fine (2cx, 2cy).
        let cnx = nx.div_ceil(2);
        let cny = ny.div_ceil(2);
        if cnx < 4 || cny < 4 {
            let _ = Self::pressure_solve(p, rhs, nx, ny, nu * 4);
            return;
        }
        let cn = cnx * cny;

        // Ensure persistent buffers exist for this level
        while levels.len() <= depth {
            levels.push((Vec::new(), Vec::new(), Vec::new()));
        }

        // Take this level's buffers out to avoid borrow conflict with recursive call
        let (mut residual, mut c_rhs, mut c_p) = std::mem::take(&mut levels[depth]);
        residual.resize(n, 0.0);
        c_rhs.resize(cn, 0.0);
        c_p.resize(cn, 0.0);
        c_p.fill(0.0);

        // Residual of the smoother's system: r = rhs − (4p − Σn)
        residual.fill(0.0);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let a_p = 4.0 * p[i] - (p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx]);
                residual[i] = rhs[i] - a_p;
            }
        }

        // Restrict (full weighting). The unit stencil hides h², so the coarse
        // (2h) system's right-hand side is 4× the restricted residual.
        c_rhs.fill(0.0);
        for cy in 1..cny - 1 {
            for cx in 1..cnx - 1 {
                let fx = cx * 2;
                let fy = cy * 2;
                let fi = fy * nx + fx;
                c_rhs[cy * cnx + cx] = (4.0 * residual[fi]
                    + 2.0
                        * (residual[fi - 1]
                            + residual[fi + 1]
                            + residual[fi - nx]
                            + residual[fi + nx])
                    + residual[fi - nx - 1]
                    + residual[fi - nx + 1]
                    + residual[fi + nx - 1]
                    + residual[fi + nx + 1])
                    * 0.25;
            }
        }

        // Recurse
        Self::mg_vcycle(&mut c_p, &c_rhs, cnx, cny, nu, levels, depth + 1);

        // Prolongate (bilinear): even fine points take the coarse value, odd
        // points average their 2 or 4 coarse parents. Boundary ring stays 0.
        let c_at = |cx: usize, cy: usize| -> f64 {
            if cx < cnx && cy < cny {
                c_p[cy * cnx + cx]
            } else {
                0.0
            }
        };
        for y in 1..ny - 1 {
            let cy = y / 2;
            let odd_y = y % 2 == 1;
            for x in 1..nx - 1 {
                let cx = x / 2;
                let odd_x = x % 2 == 1;
                let corr = match (odd_x, odd_y) {
                    (false, false) => c_at(cx, cy),
                    (true, false) => 0.5 * (c_at(cx, cy) + c_at(cx + 1, cy)),
                    (false, true) => 0.5 * (c_at(cx, cy) + c_at(cx, cy + 1)),
                    (true, true) => {
                        0.25 * (c_at(cx, cy)
                            + c_at(cx + 1, cy)
                            + c_at(cx, cy + 1)
                            + c_at(cx + 1, cy + 1))
                    }
                };
                p[y * nx + x] += corr;
            }
        }

        // Return buffers to level cache
        levels[depth] = (residual, c_rhs, c_p);

        // Post-smooth
        let _ = Self::pressure_solve(p, rhs, nx, ny, nu);
    }

    /// Conjugate gradient Poisson solver for `4p − Σn = rhs`.
    ///
    /// Matrix-free CG using `hisab::num::conjugate_gradient`. The unit-stencil
    /// operator `4p − Σn` is symmetric positive definite on the interior with
    /// Dirichlet boundaries, as CG requires. Converges to a relative residual
    /// of 1e-6 or stops after `iterations`.
    fn pressure_solve_cg(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        iterations: usize,
    ) -> Result<()> {
        let _span = trace_span!("grid::pressure_solve_cg", nx, ny, iterations).entered();

        let n = nx * ny;

        // SPD operator 4·p[center] − (p[left]+p[right]+p[up]+p[down]) on
        // interior cells; boundary rows are zero (Dirichlet BCs).
        let a_mul = move |p: &[f64]| -> Vec<f64> {
            let mut result = vec![0.0; n];
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    result[i] = 4.0 * p[i] - (p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx]);
                }
            }
            result
        };

        let b_norm = div.iter().map(|d| d * d).sum::<f64>().sqrt();
        let tol = (1e-6 * b_norm).max(1e-14);

        let solution = conjugate_gradient(a_mul, div, pressure, tol, iterations).map_err(|e| {
            PravashError::Diverged {
                reason: format!("CG pressure solve failed: {e}").into(),
            }
        })?;

        pressure.copy_from_slice(&solution);
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
            // Rows are independent: update vx/vy row slices in parallel.
            vx.par_chunks_mut(nx)
                .zip(vy.par_chunks_mut(nx))
                .enumerate()
                .filter(|(y, _)| *y >= 1 && *y < ny - 1)
                .for_each(|(y, (row_x, row_y))| {
                    for x in 1..nx - 1 {
                        let i = y * nx + x;
                        row_x[x] -= (pressure[i + 1] - pressure[i - 1]) * inv_2dx;
                        row_y[x] -= (pressure[i + nx] - pressure[i - nx]) * inv_2dx;
                    }
                });
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
                // Wall sits on the face between ghost and first interior cell
                // (as for NoSlip): normal component reflected (zero at the
                // face), tangential component copied (zero gradient).
                for x in 0..nx {
                    vx[x] = vx[nx + x];
                    vy[x] = -vy[nx + x];
                    let top = (ny - 1) * nx + x;
                    let below = (ny - 2) * nx + x;
                    vx[top] = vx[below];
                    vy[top] = -vy[below];
                }
                for y in 0..ny {
                    let l = y * nx;
                    vx[l] = -vx[l + 1];
                    vy[l] = vy[l + 1];
                    let r = y * nx + nx - 1;
                    vx[r] = -vx[r - 1];
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
        if !config.smagorinsky_cs.is_finite() || config.smagorinsky_cs < 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: "smagorinsky_cs must be finite and >= 0".into(),
            });
        }
        if !config.vorticity_confinement.is_finite() || config.vorticity_confinement < 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: "vorticity_confinement must be finite and >= 0".into(),
            });
        }
        if !config.buoyancy_alpha.is_finite() || !config.ambient_density.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: "buoyancy_alpha and ambient_density must be finite".into(),
            });
        }
        if config.pressure_iterations == 0 {
            return Err(PravashError::InvalidParameter {
                reason: "pressure_iterations must be > 0".into(),
            });
        }
        if config.diffusion_iterations == 0 {
            return Err(PravashError::InvalidParameter {
                reason: "diffusion_iterations must be > 0".into(),
            });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = config.dt;
        // Public fields / Deserialize can bypass `new()`: re-check invariants.
        let n = match nx.checked_mul(ny) {
            Some(n)
                if nx >= Self::MIN_SIZE
                    && ny >= Self::MIN_SIZE
                    && self.vx.len() == n
                    && self.vy.len() == n
                    && self.pressure.len() == n
                    && self.density.len() == n =>
            {
                n
            }
            _ => return Err(PravashError::InvalidGridResolution { nx, ny }),
        };
        if !dx.is_finite() || dx <= 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
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
        if config.use_bfecc && config.boundary != BoundaryCondition::Periodic {
            sc.fill(0.0);
            drhs.fill(0.0);
            Self::advect_bfecc(
                &mut sa, &self.vx, &self.vx, &self.vy, nx, ny, dt, inv_dx, &mut sc, &mut drhs,
            );
            sc.fill(0.0);
            drhs.fill(0.0);
            Self::advect_bfecc(
                &mut sb, &self.vy, &self.vx, &self.vy, nx, ny, dt, inv_dx, &mut sc, &mut drhs,
            );
            self.vx.copy_from_slice(&sa);
            self.vy.copy_from_slice(&sb);
        } else if config.use_maccormack && config.boundary != BoundaryCondition::Periodic {
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

        // Check for NaN after advection
        if self.vx.iter().any(|v| v.is_nan()) || self.vy.iter().any(|v| v.is_nan()) {
            // Restore scratch buffers before returning error
            self.scratch_a = sa;
            self.scratch_b = sb;
            self.scratch_c = sc;
            self.diffuse_rhs = drhs;
            return Err(PravashError::Diverged {
                reason: "NaN detected in velocity field after advection".into(),
            });
        }

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
            // RHS = −dx²·∇·u for the unit-stencil Poisson system (see pressure_solve).
            Self::divergence(&mut sa, &self.vx, &self.vy, nx, ny, 0.5 * dx);
            if config.use_cg {
                Self::pressure_solve_cg(
                    &mut self.pressure,
                    &sa,
                    nx,
                    ny,
                    config.pressure_iterations,
                )?;
            } else if config.use_multigrid {
                // ~10 GS sweeps of work per V-cycle: match pressure_iterations' budget.
                let cycles = config.pressure_iterations.div_ceil(10);
                let mut mg = std::mem::take(&mut self.mg_levels);
                let res = Self::pressure_solve_multigrid(
                    &mut self.pressure,
                    &sa,
                    nx,
                    ny,
                    4,
                    cycles,
                    &mut mg,
                );
                self.mg_levels = mg;
                res?;
            } else if config.boundary == BoundaryCondition::Periodic {
                Self::pressure_solve(&mut self.pressure, &sa, nx, ny, config.pressure_iterations)?;
            } else {
                Self::pressure_solve_dst(&mut self.pressure, &sa, nx, ny)?;
            }
            Self::project_velocity(&mut self.vx, &mut self.vy, &self.pressure, nx, ny, inv_2dx);
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        if self.vx.iter().chain(&self.vy).any(|v| !v.is_finite()) {
            self.scratch_a = sa;
            self.scratch_b = sb;
            self.scratch_c = sc;
            self.diffuse_rhs = drhs;
            return Err(PravashError::Diverged {
                reason: "non-finite velocity after pressure projection".into(),
            });
        }

        // 5. Advect density
        if config.use_bfecc && config.boundary != BoundaryCondition::Periodic {
            sc.fill(0.0);
            drhs.fill(0.0);
            Self::advect_bfecc(
                &mut sa,
                &self.density,
                &self.vx,
                &self.vy,
                nx,
                ny,
                dt,
                inv_dx,
                &mut sc,
                &mut drhs,
            );
            self.density.copy_from_slice(&sa);
        } else if config.use_maccormack && config.boundary != BoundaryCondition::Periodic {
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

// ── k-epsilon Turbulence Model ───────────────────────────────────────────────

/// k-epsilon turbulence state for a 2D grid.
///
/// Two-equation RANS model for:
/// - k: turbulent kinetic energy
/// - ε: turbulent dissipation rate
///
/// [`step`](KEpsilon::step) integrates the production/dissipation source
/// terms only; advection and diffusion of k and ε are not modelled yet.
///
/// Effective viscosity: ν_t = C_μ · k² / ε
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KEpsilon {
    /// Turbulent kinetic energy per cell.
    pub k: Vec<f64>,
    /// Turbulent dissipation rate per cell.
    pub epsilon: Vec<f64>,
    /// C_μ constant. Default: 0.09.
    pub c_mu: f64,
    /// C_ε1 constant. Default: 1.44.
    pub c_eps1: f64,
    /// C_ε2 constant. Default: 1.92.
    pub c_eps2: f64,
    /// Scratch buffer for k (avoids per-step allocation).
    #[serde(skip)]
    scratch_k: Vec<f64>,
    /// Scratch buffer for epsilon (avoids per-step allocation).
    #[serde(skip)]
    scratch_eps: Vec<f64>,
}

impl KEpsilon {
    /// Create k-epsilon state for a grid.
    #[must_use]
    pub fn new(nx: usize, ny: usize) -> Self {
        let n = nx * ny;
        Self {
            k: vec![1e-4; n],
            epsilon: vec![1e-5; n],
            c_mu: 0.09,
            c_eps1: 1.44,
            c_eps2: 1.92,
            scratch_k: vec![0.0; n],
            scratch_eps: vec![0.0; n],
        }
    }

    /// Compute turbulent viscosity at a cell: ν_t = C_μ · k² / ε.
    #[inline]
    #[must_use]
    pub fn turbulent_viscosity(&self, i: usize) -> f64 {
        self.c_mu * self.k[i] * self.k[i] / self.epsilon[i].max(1e-20)
    }

    /// Step the k-ε source terms (production and dissipation).
    ///
    /// Does nothing (with a warning) if the slice lengths do not match
    /// `nx·ny`, the grid is smaller than 3×3, or `dx`/`dt` are invalid.
    pub fn step(&mut self, vx: &[f64], vy: &[f64], nx: usize, ny: usize, dx: f64, dt: f64) {
        let _span = trace_span!("grid::kepsilon", nx, ny).entered();
        let n = match nx.checked_mul(ny) {
            Some(n)
                if nx >= 3
                    && ny >= 3
                    && vx.len() == n
                    && vy.len() == n
                    && self.k.len() == n
                    && self.epsilon.len() == n
                    && dx.is_finite()
                    && dx > 0.0
                    && dt.is_finite()
                    && dt > 0.0 =>
            {
                n
            }
            _ => {
                tracing::warn!(
                    nx,
                    ny,
                    "KEpsilon::step: invalid sizes or parameters ignored"
                );
                return;
            }
        };
        let inv_2dx = 0.5 / dx;

        // Swap k/epsilon into scratch buffers to read old values without cloning.
        self.scratch_k.resize(n, 0.0);
        self.scratch_eps.resize(n, 0.0);
        std::mem::swap(&mut self.k, &mut self.scratch_k);
        std::mem::swap(&mut self.epsilon, &mut self.scratch_eps);
        // Now scratch_k/scratch_eps hold the old values; k/epsilon are the (stale) scratch buffers.
        // Copy old values into k/epsilon as starting point for the update.
        self.k.resize(n, 0.0);
        self.epsilon.resize(n, 0.0);
        self.k.copy_from_slice(&self.scratch_k);
        self.epsilon.copy_from_slice(&self.scratch_eps);

        let k_old = &self.scratch_k;
        let eps_old = &self.scratch_eps;

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let nu_t = self.c_mu * k_old[i] * k_old[i] / eps_old[i].max(1e-20);

                let dudx = (vx[i + 1] - vx[i - 1]) * inv_2dx;
                let dudy = (vx[i + nx] - vx[i - nx]) * inv_2dx;
                let dvdx = (vy[i + 1] - vy[i - 1]) * inv_2dx;
                let dvdy = (vy[i + nx] - vy[i - nx]) * inv_2dx;
                let s_mag_sq = 2.0 * (dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx).powi(2));

                let production = nu_t * s_mag_sq;

                self.k[i] += (production - eps_old[i]) * dt;
                self.k[i] = self.k[i].max(1e-10);

                let eps_over_k = eps_old[i] / k_old[i].max(1e-10);
                self.epsilon[i] +=
                    (self.c_eps1 * production - self.c_eps2 * eps_old[i]) * eps_over_k * dt;
                self.epsilon[i] = self.epsilon[i].max(1e-10);
            }
        }
    }
}

// ── Ghost Fluid Method ──────────────────────────────────────────────────────

/// Apply ghost fluid pressure correction at a multi-phase interface.
///
/// Modifies the pressure field to account for the surface-tension-induced
/// pressure discontinuity `[p] = σ·κ` at the interface defined by `level_set`.
///
/// `level_set`: signed distance (negative = inside fluid, positive = outside).
/// `surface_tension`: σ coefficient.
pub fn apply_ghost_fluid(
    pressure: &mut [f64],
    level_set: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    surface_tension: f64,
) {
    let _span = trace_span!("grid::ghost_fluid", nx, ny).entered();
    let valid = nx >= 3
        && ny >= 3
        && nx
            .checked_mul(ny)
            .is_some_and(|n| pressure.len() == n && level_set.len() == n)
        && dx.is_finite()
        && dx > 0.0;
    if !valid {
        tracing::warn!(nx, ny, "apply_ghost_fluid: invalid sizes or dx ignored");
        return;
    }
    let inv_dx = 1.0 / dx;
    let inv_dx2 = inv_dx * inv_dx;

    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            let i = y * nx + x;
            let phi = level_set[i];
            let phi_r = level_set[i + 1];
            let phi_l = level_set[i - 1];
            let phi_t = level_set[i + nx];
            let phi_b = level_set[i - nx];

            let crosses =
                phi * phi_r < 0.0 || phi * phi_l < 0.0 || phi * phi_t < 0.0 || phi * phi_b < 0.0;
            if !crosses {
                continue;
            }

            // Curvature: κ ≈ -(∇²φ) / |∇φ|
            let gx = (phi_r - phi_l) * 0.5 * inv_dx;
            let gy = (phi_t - phi_b) * 0.5 * inv_dx;
            let grad_mag = (gx * gx + gy * gy).sqrt().max(1e-20);
            let lxx = (phi_r - 2.0 * phi + phi_l) * inv_dx2;
            let lyy = (phi_t - 2.0 * phi + phi_b) * inv_dx2;
            let kappa = -(lxx + lyy) / grad_mag;

            // Pressure jump at interface: [p] = σ·κ
            if phi > 0.0 {
                pressure[i] -= surface_tension * kappa * phi / grad_mag;
            }
        }
    }
}

// ── Staggered MAC Grid ──────────────────────────────────────────────────────

/// Staggered (MAC) grid — velocity at faces, pressure at centers.
///
/// Eliminates checkerboard pressure oscillations by storing vx at
/// x-face centers (i+½, j) and vy at y-face centers (i, j+½).
/// Pressure and density are stored at cell centers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MacGrid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    /// x-velocity at x-face centers. Size: (nx+1) × ny.
    pub vx: Vec<f64>,
    /// y-velocity at y-face centers. Size: nx × (ny+1).
    pub vy: Vec<f64>,
    /// Pressure at cell centers. Size: nx × ny.
    pub pressure: Vec<f64>,
    /// Density/scalar at cell centers. Size: nx × ny.
    pub density: Vec<f64>,
    /// Scratch buffers (divergence, advected density) reused across steps.
    #[serde(skip)]
    scratch_div: Vec<f64>,
    #[serde(skip)]
    scratch_density: Vec<f64>,
}

impl MacGrid {
    /// Create a new staggered grid. Minimum size 4×4.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx < 4 || ny < 4 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 || !dx.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
        let cell_n = nx * ny;
        let vx_n = (nx + 1) * ny;
        let vy_n = nx * (ny + 1);
        Ok(Self {
            nx,
            ny,
            dx,
            vx: vec![0.0; vx_n],
            vy: vec![0.0; vy_n],
            pressure: vec![0.0; cell_n],
            density: vec![0.0; cell_n],
            scratch_div: vec![0.0; cell_n],
            scratch_density: vec![0.0; cell_n],
        })
    }

    /// Cell-center velocity by averaging face values.
    #[inline]
    #[must_use]
    pub fn velocity_at(&self, x: usize, y: usize) -> (f64, f64) {
        let vx = 0.5 * (self.vx[y * (self.nx + 1) + x] + self.vx[y * (self.nx + 1) + x + 1]);
        let vy = 0.5 * (self.vy[y * self.nx + x] + self.vy[(y + 1) * self.nx + x]);
        (vx, vy)
    }

    /// Maximum speed across all cell centers.
    #[must_use]
    pub fn max_speed(&self) -> f64 {
        let mut max_sq = 0.0f64;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let (vx, vy) = self.velocity_at(x, y);
                max_sq = max_sq.max(vx * vx + vy * vy);
            }
        }
        max_sq.sqrt()
    }

    /// Compute divergence at cell centers (exact on staggered grid).
    #[inline]
    #[must_use]
    pub fn divergence_at(&self, x: usize, y: usize) -> f64 {
        let inv_dx = 1.0 / self.dx;
        let dvx = self.vx[y * (self.nx + 1) + x + 1] - self.vx[y * (self.nx + 1) + x];
        let dvy = self.vy[(y + 1) * self.nx + x] - self.vy[y * self.nx + x];
        (dvx + dvy) * inv_dx
    }

    /// Perform one Navier-Stokes step on the staggered grid.
    ///
    /// Simplified pipeline: buoyancy → pressure projection → advect density.
    ///
    /// The domain is closed by solid walls on all four sides: the normal face
    /// velocity on the domain boundary is held at zero and pressure uses
    /// Neumann (zero normal gradient) wall conditions, so the projection is
    /// exact on the staggered layout. `config.boundary` is not used (there is
    /// no tangential wall velocity on a MAC grid without viscosity).
    pub fn step(&mut self, config: &GridConfig) -> Result<()> {
        let _span = trace_span!("mac::step", nx = self.nx, ny = self.ny).entered();

        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt: config.dt });
        }
        if config.pressure_iterations == 0 {
            return Err(PravashError::InvalidParameter {
                reason: "pressure_iterations must be > 0".into(),
            });
        }
        if !config.buoyancy_alpha.is_finite() || !config.ambient_density.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: "buoyancy_alpha and ambient_density must be finite".into(),
            });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let dt = config.dt;
        let cell_n = match nx.checked_mul(ny) {
            Some(n)
                if nx >= 4
                    && ny >= 4
                    && self.vx.len() == (nx + 1) * ny
                    && self.vy.len() == nx * (ny + 1)
                    && self.pressure.len() == n
                    && self.density.len() == n =>
            {
                n
            }
            _ => return Err(PravashError::InvalidGridResolution { nx, ny }),
        };
        if !dx.is_finite() || dx <= 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
        let inv_dx = 1.0 / dx;

        // 0. Solid walls: zero normal velocity on the domain boundary faces.
        for y in 0..ny {
            self.vx[y * (nx + 1)] = 0.0;
            self.vx[y * (nx + 1) + nx] = 0.0;
        }
        for x in 0..nx {
            self.vy[x] = 0.0;
            self.vy[ny * nx + x] = 0.0;
        }

        // 1. Buoyancy on interior y-faces (density averaged to the face)
        if config.buoyancy_alpha != 0.0 {
            for y in 1..ny {
                for x in 0..nx {
                    let rho = 0.5 * (self.density[(y - 1) * nx + x] + self.density[y * nx + x]);
                    self.vy[y * nx + x] +=
                        config.buoyancy_alpha * (rho - config.ambient_density) * dt;
                }
            }
        }

        // 2. Pressure projection: Σ_nbr (p_i − p_j) = −dx·(flux sum) = −dx²·∇·u,
        //    neighbours across a wall are omitted (Neumann).
        let mut rhs = std::mem::take(&mut self.scratch_div);
        rhs.resize(cell_n, 0.0);
        for y in 0..ny {
            for x in 0..nx {
                rhs[y * nx + x] = -((self.vx[y * (nx + 1) + x + 1] - self.vx[y * (nx + 1) + x])
                    + (self.vy[(y + 1) * nx + x] - self.vy[y * nx + x]))
                    * dx;
            }
        }
        for _ in 0..config.pressure_iterations {
            for y in 0..ny {
                for x in 0..nx {
                    let i = y * nx + x;
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    if x > 0 {
                        sum += self.pressure[i - 1];
                        count += 1.0;
                    }
                    if x + 1 < nx {
                        sum += self.pressure[i + 1];
                        count += 1.0;
                    }
                    if y > 0 {
                        sum += self.pressure[i - nx];
                        count += 1.0;
                    }
                    if y + 1 < ny {
                        sum += self.pressure[i + nx];
                        count += 1.0;
                    }
                    self.pressure[i] = (rhs[i] + sum) / count;
                }
            }
        }
        // Pure-Neumann pressure is defined up to a constant: pin the mean to 0.
        let mean = self.pressure.iter().sum::<f64>() / cell_n as f64;
        self.pressure.iter_mut().for_each(|p| *p -= mean);
        self.scratch_div = rhs;

        // Subtract pressure gradient at interior faces (wall faces stay 0)
        for y in 0..ny {
            for x in 1..nx {
                self.vx[y * (nx + 1) + x] -=
                    (self.pressure[y * nx + x] - self.pressure[y * nx + x - 1]) * inv_dx;
            }
        }
        for y in 1..ny {
            for x in 0..nx {
                self.vy[y * nx + x] -=
                    (self.pressure[y * nx + x] - self.pressure[(y - 1) * nx + x]) * inv_dx;
            }
        }

        // 3. Advect density (semi-Lagrangian from cell centres)
        let mut advected = std::mem::take(&mut self.scratch_density);
        advected.resize(cell_n, 0.0);
        let dt_inv_dx = dt * inv_dx;
        for y in 0..ny {
            for x in 0..nx {
                let (u, v) = self.velocity_at(x, y);
                let fx = x as f64 - dt_inv_dx * u;
                let fy = y as f64 - dt_inv_dx * v;
                advected[y * nx + x] = FluidGrid::sample(&self.density, nx, ny, fx, fy);
            }
        }
        std::mem::swap(&mut self.density, &mut advected);
        self.scratch_density = advected;

        if self
            .vx
            .iter()
            .chain(&self.vy)
            .chain(&self.density)
            .any(|v| !v.is_finite())
        {
            return Err(PravashError::Diverged {
                reason: "non-finite value in MAC grid step".into(),
            });
        }
        Ok(())
    }

    /// CFL timestep for the staggered grid.
    #[must_use]
    pub fn cfl_dt(&self, cfl_factor: f64) -> f64 {
        let max_v = self.max_speed();
        if max_v < 1e-20 {
            // No flow: no advective limit (consistent with FluidGrid::cfl_dt).
            return f64::MAX;
        }
        cfl_factor * self.dx / max_v
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
        let _ = FluidGrid::pressure_solve(&mut pressure, &div, nx, ny, 50);
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
        let _ = FluidGrid::pressure_solve(&mut pressure, &div_before, nx, ny, 100);
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
        // Wall on the ghost/interior face: normal velocity averages to zero there.
        let l = 3 * nx; // left ghost, row 3
        assert!((vx[l] + vx[l + 1]).abs() < f64::EPSILON); // left wall: normal vx
        assert!((vy[1] + vy[nx + 1]).abs() < f64::EPSILON); // bottom wall: normal vy
        // Tangential components are copied (zero gradient).
        assert!((vy[l] - vy[l + 1]).abs() < f64::EPSILON);
        assert!((vx[1] - vx[nx + 1]).abs() < f64::EPSILON);
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

    // ── Audit regressions (1.3.0) ──────────────────────────────────────

    /// Smooth potential-flow field u = ∇φ (φ a Gaussian bump), zero near walls.
    fn potential_flow(nx: usize, ny: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
        let n = nx * ny;
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        let (cx, cy) = (nx as f64 / 2.0, ny as f64 / 2.0);
        let sigma = nx as f64 / 8.0;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let (rx, ry) = (x as f64 - cx, y as f64 - cy);
                let phi = (-(rx * rx + ry * ry) / (sigma * sigma)).exp();
                // ∂φ/∂x in physical units
                vx[y * nx + x] = -2.0 * rx / (sigma * sigma) * phi / dx;
                vy[y * nx + x] = -2.0 * ry / (sigma * sigma) * phi / dx;
            }
        }
        (vx, vy)
    }

    fn div_norm(vx: &[f64], vy: &[f64], nx: usize, ny: usize, dx: f64) -> f64 {
        let mut div = vec![0.0; nx * ny];
        FluidGrid::divergence(&mut div, vx, vy, nx, ny, 0.5 / dx);
        div.iter().map(|d| d * d).sum::<f64>().sqrt()
    }

    /// Project with each solver at dx ≠ 1; every one must cut divergence.
    #[test]
    fn test_projection_all_solvers_dx_scaled() {
        let (nx, ny, dx) = (33, 33, 0.1);
        for solver in ["gs", "dst", "mg", "cg"] {
            let (mut vx, mut vy) = potential_flow(nx, ny, dx);
            let before = div_norm(&vx, &vy, nx, ny, dx);
            let mut rhs = vec![0.0; nx * ny];
            FluidGrid::divergence(&mut rhs, &vx, &vy, nx, ny, 0.5 * dx);
            let mut p = vec![0.0; nx * ny];
            let mut levels = Vec::new();
            match solver {
                "gs" => FluidGrid::pressure_solve(&mut p, &rhs, nx, ny, 2000).unwrap(),
                "dst" => FluidGrid::pressure_solve_dst(&mut p, &rhs, nx, ny).unwrap(),
                "mg" => {
                    FluidGrid::pressure_solve_multigrid(&mut p, &rhs, nx, ny, 4, 10, &mut levels)
                        .unwrap()
                }
                _ => FluidGrid::pressure_solve_cg(&mut p, &rhs, nx, ny, 500).unwrap(),
            }
            FluidGrid::project_velocity(&mut vx, &mut vy, &p, nx, ny, 0.5 / dx);
            let after = div_norm(&vx, &vy, nx, ny, dx);
            assert!(
                after < 0.5 * before,
                "{solver}: divergence {before} -> {after} (must shrink)"
            );
        }
    }

    /// All solvers solve the same system: their pressures must agree.
    #[test]
    fn test_pressure_solvers_agree() {
        let (nx, ny, dx) = (17, 17, 0.25);
        let (vx, vy) = potential_flow(nx, ny, dx);
        let mut rhs = vec![0.0; nx * ny];
        FluidGrid::divergence(&mut rhs, &vx, &vy, nx, ny, 0.5 * dx);
        let mut p_dst = vec![0.0; nx * ny];
        FluidGrid::pressure_solve_dst(&mut p_dst, &rhs, nx, ny).unwrap();
        let mut p_gs = vec![0.0; nx * ny];
        FluidGrid::pressure_solve(&mut p_gs, &rhs, nx, ny, 5000).unwrap();
        let mut p_cg = vec![0.0; nx * ny];
        FluidGrid::pressure_solve_cg(&mut p_cg, &rhs, nx, ny, 1000).unwrap();
        let mut p_mg = vec![0.0; nx * ny];
        let mut levels = Vec::new();
        FluidGrid::pressure_solve_multigrid(&mut p_mg, &rhs, nx, ny, 4, 30, &mut levels).unwrap();
        let scale = p_dst.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0);
        for (name, p) in [("gs", &p_gs), ("cg", &p_cg), ("mg", &p_mg)] {
            let err = p
                .iter()
                .zip(&p_dst)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                err < 1e-3 * scale,
                "{name} disagrees with DST: {err} vs {scale}"
            );
        }
    }

    /// Repeated V-cycles must drive the residual down monotonically.
    #[test]
    fn test_multigrid_converges() {
        let (nx, ny) = (33, 33);
        let mut rhs = vec![0.0; nx * ny];
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                rhs[y * nx + x] = ((x * 7 + y * 3) % 5) as f64 - 2.0;
            }
        }
        let residual = |p: &[f64]| -> f64 {
            let mut r = 0.0;
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    let a = 4.0 * p[i] - (p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx]);
                    r += (rhs[i] - a).powi(2);
                }
            }
            r.sqrt()
        };
        let mut p = vec![0.0; nx * ny];
        let mut levels = Vec::new();
        let mut last = residual(&p);
        for _ in 0..6 {
            FluidGrid::pressure_solve_multigrid(&mut p, &rhs, nx, ny, 4, 1, &mut levels).unwrap();
            let r = residual(&p);
            assert!(r < 0.5 * last, "V-cycle residual {last} -> {r}");
            last = r;
        }
    }

    /// A blob advected more than one cell per step must survive the limiter.
    #[test]
    fn test_maccormack_bfecc_preserve_moving_blob() {
        let (nx, ny) = (40, 9);
        let n = nx * ny;
        let mut src = vec![0.0; n];
        for x in 8..12 {
            src[4 * nx + x] = 1.0;
        }
        let vx = vec![2.0; n]; // CFL = 2 with dt = dx = 1
        let vy = vec![0.0; n];
        let mass0: f64 = src.iter().sum();
        for which in ["maccormack", "bfecc"] {
            let mut dst = vec![0.0; n];
            let mut t1 = vec![0.0; n];
            let mut t2 = vec![0.0; n];
            if which == "maccormack" {
                FluidGrid::advect_maccormack(&mut dst, &src, &vx, &vy, nx, ny, 1.0, 1.0, &mut t1);
            } else {
                FluidGrid::advect_bfecc(
                    &mut dst, &src, &vx, &vy, nx, ny, 1.0, 1.0, &mut t1, &mut t2,
                );
            }
            let mass: f64 = dst.iter().sum();
            assert!(
                (mass - mass0).abs() < 1e-9,
                "{which}: blob mass {mass0} -> {mass}"
            );
            assert!(
                (dst[4 * nx + 12] - 1.0).abs() < 1e-9,
                "{which}: blob moved by 2"
            );
            assert!(dst.iter().all(|&v| (-1e-12..=1.0 + 1e-12).contains(&v)));
        }
    }

    #[test]
    fn test_step_rejects_nan_config() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        let mut c = GridConfig::smoke();
        c.smagorinsky_cs = f64::NAN;
        assert!(g.step(&c).is_err());
        let mut c = GridConfig::smoke();
        c.buoyancy_alpha = f64::NAN;
        assert!(g.step(&c).is_err());
    }

    #[test]
    fn test_step_rejects_inconsistent_grid() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        g.vx.truncate(10);
        assert!(g.step(&GridConfig::smoke()).is_err());
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        g.nx = 1;
        assert!(g.step(&GridConfig::smoke()).is_err());
    }

    #[test]
    fn test_diffuse_size_mismatch_no_panic() {
        let mut f = vec![1.0; 10];
        FluidGrid::diffuse(&mut f, 8, 8, 0.1, 0.1, 0.1, 5);
        FluidGrid::diffuse(&mut f, 10, 0, 0.1, 0.1, 0.1, 5);
        assert!(f.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_kepsilon_size_mismatch_no_panic() {
        let mut ke = KEpsilon::new(8, 8);
        ke.step(&[0.0; 10], &[0.0; 10], 8, 8, 0.1, 0.01);
        ke.step(&[0.0; 64], &[0.0; 64], 16, 4, 0.1, 0.01);
    }

    #[test]
    fn test_cfl_dt_finite_without_flow() {
        let g = FluidGrid::new(8, 8, 0.1).unwrap();
        assert!(g.cfl_dt(0.0, 2.0).is_finite());
        let m = MacGrid::new(8, 8, 0.1).unwrap();
        assert!(m.cfl_dt(0.5) > 1e300);
    }

    #[test]
    fn test_mac_projection_divergence_free() {
        let mut m = MacGrid::new(16, 16, 0.1).unwrap();
        // Arbitrary interior face velocities.
        for (k, v) in m.vx.iter_mut().enumerate() {
            *v = ((k * 13) % 7) as f64 - 3.0;
        }
        for (k, v) in m.vy.iter_mut().enumerate() {
            *v = ((k * 5) % 11) as f64 - 5.0;
        }
        let mut c = GridConfig::smoke();
        c.buoyancy_alpha = 0.0;
        c.pressure_iterations = 3000;
        m.step(&c).unwrap();
        let mut max_div = 0.0f64;
        for y in 0..16 {
            for x in 0..16 {
                max_div = max_div.max(m.divergence_at(x, y).abs());
            }
        }
        assert!(
            max_div < 1e-3,
            "MAC projection residual divergence {max_div}"
        );
        // Walls are closed.
        assert!(m.vy[..16].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mac_buoyancy_does_not_accumulate_on_walls() {
        let mut m = MacGrid::new(8, 8, 0.1).unwrap();
        m.density.fill(1.0);
        let c = GridConfig::smoke();
        for _ in 0..10 {
            m.step(&c).unwrap();
        }
        assert!(m.vy[..8].iter().all(|&v| v == 0.0));
        assert!(m.vy[8 * 8..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mac_advects_density() {
        let mut m = MacGrid::new(16, 16, 0.1).unwrap();
        m.density[8 * 16 + 4] = 1.0;
        for y in 0..16 {
            for x in 1..16 {
                m.vx[y * 17 + x] = 0.5;
            }
        }
        let mut c = GridConfig::smoke();
        c.buoyancy_alpha = 0.0;
        let before = m.density[8 * 16 + 4];
        m.step(&c).unwrap();
        assert!(
            m.density[8 * 16 + 4] < before,
            "density must move with the flow"
        );
    }

    #[test]
    fn test_ghost_fluid_size_mismatch_no_panic() {
        let mut p = vec![0.0; 10];
        apply_ghost_fluid(&mut p, &[0.0; 64], 8, 8, 0.1, 0.07);
    }

    /// Full `step()` at dx = 0.1 must remove divergence for every solver path
    /// (the unscaled RHS used to amplify it by ~1/dx²).
    #[test]
    fn test_step_reduces_divergence_small_dx() {
        let (nx, ny, dx) = (33, 33, 0.1);
        for (cg, mg) in [(false, false), (false, true), (true, false)] {
            let mut g = FluidGrid::new(nx, ny, dx).unwrap();
            let (vx, vy) = potential_flow(nx, ny, dx);
            g.vx = vx;
            g.vy = vy;
            let before = div_norm(&g.vx, &g.vy, nx, ny, dx);
            let mut c = GridConfig::smoke();
            c.dt = 1e-4;
            c.viscosity = 0.0;
            c.buoyancy_alpha = 0.0;
            c.vorticity_confinement = 0.0;
            c.pressure_iterations = 200;
            c.use_cg = cg;
            c.use_multigrid = mg;
            g.step(&c).unwrap();
            let after = div_norm(&g.vx, &g.vy, nx, ny, dx);
            assert!(
                after < 0.5 * before,
                "cg={cg} mg={mg}: divergence {before} -> {after}"
            );
        }
    }
}
