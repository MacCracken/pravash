//! Grid-based Euler/Navier-Stokes fluid solver.
//!
//! 2D incompressible Navier-Stokes on a uniform grid using operator splitting:
//! advection (semi-Lagrangian) → diffusion (Gauss-Seidel) → force application
//! → pressure projection (FFT Poisson solver).

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

use hisab::num::{Complex, fft, ifft};
use tracing::trace_span;

/// Boundary condition type for the grid solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BoundaryCondition {
    /// Velocity = 0 at walls (viscous wall).
    NoSlip,
    /// Normal velocity = 0, tangential velocity free (inviscid wall).
    FreeSlip,
}

/// Configuration for a grid fluid simulation step.
#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    pub dt: f64,
    pub viscosity: f64,
    pub diffusion_iterations: usize,
    pub boundary: BoundaryCondition,
    /// Vorticity confinement strength (0.0 to disable).
    pub vorticity_confinement: f64,
    /// Gravity/buoyancy: acceleration applied to density field.
    /// Positive = upward buoyancy for hot/light fluid.
    pub buoyancy_alpha: f64,
    /// Ambient temperature/density for buoyancy reference.
    pub ambient_density: f64,
}

impl GridConfig {
    /// Default config for smoke simulation.
    #[must_use]
    pub fn smoke() -> Self {
        Self {
            dt: 0.1,
            viscosity: 0.0001,
            diffusion_iterations: 20,
            boundary: BoundaryCondition::NoSlip,
            vorticity_confinement: 0.5,
            buoyancy_alpha: 0.1,
            ambient_density: 0.0,
        }
    }
}

/// 2D grid-based fluid state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidGrid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    /// Velocity field (x-component), row-major.
    pub vx: Vec<f64>,
    /// Velocity field (y-component), row-major.
    pub vy: Vec<f64>,
    /// Pressure field, row-major.
    pub pressure: Vec<f64>,
    /// Density/scalar field (smoke, temperature, dye).
    pub density: Vec<f64>,
}

impl FluidGrid {
    /// Create a new grid initialized to zero.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
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

    /// Bilinear interpolation of a field at continuous position (fx, fy).
    /// Coordinates are in grid-cell units (0..nx-1, 0..ny-1).
    #[inline]
    fn sample(field: &[f64], nx: usize, ny: usize, fx: f64, fy: f64) -> f64 {
        let x0 = fx.floor().max(0.0).min((nx - 2) as f64) as usize;
        let y0 = fy.floor().max(0.0).min((ny - 2) as f64) as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let sx = fx - x0 as f64;
        let sy = fy - y0 as f64;
        let sx = sx.clamp(0.0, 1.0);
        let sy = sy.clamp(0.0, 1.0);

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
    fn advect(dst: &mut [f64], src: &[f64], vx: &[f64], vy: &[f64], nx: usize, ny: usize, dt: f64) {
        let _span = trace_span!("grid::advect", nx, ny).entered();
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let fx = x as f64 - dt * vx[i];
                let fy = y as f64 - dt * vy[i];
                dst[i] = Self::sample(src, nx, ny, fx, fy);
            }
        }
    }

    // ── Diffusion ───────────────────────────────────────────────────────────

    /// Diffuse a field using Gauss-Seidel iteration.
    pub fn diffuse(
        field: &mut [f64],
        nx: usize,
        ny: usize,
        diff_rate: f64,
        dt: f64,
        iterations: usize,
    ) {
        let _span = trace_span!("grid::diffuse", nx, ny, iterations).entered();
        debug_assert_eq!(field.len(), nx * ny, "field size must match nx*ny");
        let a = dt * diff_rate * (nx as f64) * (ny as f64);
        let inv_denom = 1.0 / (1.0 + 4.0 * a);
        for _ in 0..iterations {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let idx = y * nx + x;
                    let neighbors =
                        field[idx - 1] + field[idx + 1] + field[idx - nx] + field[idx + nx];
                    field[idx] = (field[idx] + a * neighbors) * inv_denom;
                }
            }
        }
    }

    // ── Pressure Projection ─────────────────────────────────────────────────

    /// Compute velocity divergence.
    fn divergence(div: &mut [f64], vx: &[f64], vy: &[f64], nx: usize, ny: usize) {
        let inv_2 = 0.5;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                div[i] = -inv_2 * ((vx[i + 1] - vx[i - 1]) + (vy[i + nx] - vy[i - nx]));
            }
        }
    }

    /// Solve ∇²p = div using FFT (requires power-of-2 grid dimensions).
    /// Falls back to Gauss-Seidel if dimensions aren't power-of-2.
    fn pressure_solve(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        scratch: &mut Vec<Complex>,
    ) {
        let _span = trace_span!("grid::pressure_solve", nx, ny).entered();

        if nx.is_power_of_two() && ny.is_power_of_two() {
            Self::pressure_solve_fft(pressure, div, nx, ny, scratch);
        } else {
            Self::pressure_solve_gs(pressure, div, nx, ny, 100);
        }
    }

    /// FFT-based Poisson solver for power-of-2 grids.
    fn pressure_solve_fft(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        scratch: &mut Vec<Complex>,
    ) {
        let n = nx * ny;
        scratch.resize(n, Complex::from_real(0.0));

        // Copy divergence into complex buffer
        for i in 0..n {
            scratch[i] = Complex::from_real(div[i]);
        }

        // 2D FFT: rows then columns
        // Row transforms
        for y in 0..ny {
            let start = y * nx;
            fft(&mut scratch[start..start + nx]);
        }
        // Column transforms (needs transposing into a temp buffer)
        let mut col_buf = vec![Complex::from_real(0.0); ny];
        for x in 0..nx {
            for y in 0..ny {
                col_buf[y] = scratch[y * nx + x];
            }
            fft(&mut col_buf);
            for y in 0..ny {
                scratch[y * nx + x] = col_buf[y];
            }
        }

        // Divide by eigenvalues of the discrete Laplacian in frequency space.
        // For periodic DFT: λ = 2cos(2πk/N) - 2
        let two_pi = 2.0 * std::f64::consts::PI;
        for ky in 0..ny {
            for kx in 0..nx {
                if kx == 0 && ky == 0 {
                    scratch[0] = Complex::from_real(0.0);
                    continue;
                }
                let lambda_x = 2.0 * (two_pi * kx as f64 / nx as f64).cos() - 2.0;
                let lambda_y = 2.0 * (two_pi * ky as f64 / ny as f64).cos() - 2.0;
                let eigenvalue = lambda_x + lambda_y;
                let i = ky * nx + kx;
                scratch[i] = Complex::new(scratch[i].re / eigenvalue, scratch[i].im / eigenvalue);
            }
        }

        // 2D IFFT: rows then columns
        for y in 0..ny {
            let start = y * nx;
            ifft(&mut scratch[start..start + nx]);
        }
        for x in 0..nx {
            for y in 0..ny {
                col_buf[y] = scratch[y * nx + x];
            }
            ifft(&mut col_buf);
            for y in 0..ny {
                scratch[y * nx + x] = col_buf[y];
            }
        }

        // Extract real part as pressure
        for i in 0..n {
            pressure[i] = scratch[i].re;
        }
    }

    /// Gauss-Seidel Poisson solver fallback for non-power-of-2 grids.
    fn pressure_solve_gs(
        pressure: &mut [f64],
        div: &[f64],
        nx: usize,
        ny: usize,
        iterations: usize,
    ) {
        for i in pressure.iter_mut() {
            *i = 0.0;
        }
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

    /// Subtract pressure gradient from velocity to enforce divergence-free.
    fn project_velocity(vx: &mut [f64], vy: &mut [f64], pressure: &[f64], nx: usize, ny: usize) {
        let _span = trace_span!("grid::project").entered();
        let scale = 0.5;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                vx[i] -= scale * (pressure[i + 1] - pressure[i - 1]);
                vy[i] -= scale * (pressure[i + nx] - pressure[i - nx]);
            }
        }
    }

    // ── Vorticity Confinement ───────────────────────────────────────────────

    /// Apply vorticity confinement to boost rotational detail.
    fn apply_vorticity_confinement(
        vx: &mut [f64],
        vy: &mut [f64],
        nx: usize,
        ny: usize,
        dt: f64,
        epsilon: f64,
        vort_buf: &mut Vec<f64>,
    ) {
        let _span = trace_span!("grid::vorticity_confinement").entered();
        let n = nx * ny;
        vort_buf.resize(n, 0.0);

        // Compute vorticity: ω = ∂vy/∂x - ∂vx/∂y
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let dvydx = (vy[i + 1] - vy[i - 1]) * 0.5;
                let dvxdy = (vx[i + nx] - vx[i - nx]) * 0.5;
                vort_buf[i] = dvydx - dvxdy;
            }
        }

        // Compute gradient of |ω| and apply confinement force
        for y in 2..ny - 2 {
            for x in 2..nx - 2 {
                let i = y * nx + x;
                let abs_w = vort_buf[i].abs();
                if abs_w < 1e-10 {
                    continue;
                }

                // η = ∇|ω|
                let eta_x = (vort_buf[i + 1].abs() - vort_buf[i - 1].abs()) * 0.5;
                let eta_y = (vort_buf[i + nx].abs() - vort_buf[i - nx].abs()) * 0.5;
                let eta_mag = (eta_x * eta_x + eta_y * eta_y).sqrt();
                if eta_mag < 1e-10 {
                    continue;
                }

                // N = η / |η|, F = ε·h·(N × ω)
                let nx_v = eta_x / eta_mag;
                let ny_v = eta_y / eta_mag;
                // 2D cross product: N × ω = (ny·ω, -nx·ω)
                vx[i] += epsilon * dt * ny_v * vort_buf[i];
                vy[i] -= epsilon * dt * nx_v * vort_buf[i];
            }
        }
    }

    // ── Boundary Conditions ─────────────────────────────────────────────────

    /// Enforce boundary conditions on velocity fields.
    fn enforce_boundary(
        vx: &mut [f64],
        vy: &mut [f64],
        nx: usize,
        ny: usize,
        bc: BoundaryCondition,
    ) {
        match bc {
            BoundaryCondition::NoSlip => {
                // Top and bottom walls
                for x in 0..nx {
                    // Bottom
                    vx[x] = -vx[nx + x];
                    vy[x] = -vy[nx + x];
                    // Top
                    let top = (ny - 1) * nx + x;
                    let below = (ny - 2) * nx + x;
                    vx[top] = -vx[below];
                    vy[top] = -vy[below];
                }
                // Left and right walls
                for y in 0..ny {
                    // Left
                    let l = y * nx;
                    vx[l] = -vx[l + 1];
                    vy[l] = -vy[l + 1];
                    // Right
                    let r = y * nx + nx - 1;
                    vx[r] = -vx[r - 1];
                    vy[r] = -vy[r - 1];
                }
            }
            BoundaryCondition::FreeSlip => {
                for x in 0..nx {
                    // Bottom: tangential free, normal = 0
                    vx[x] = vx[nx + x];
                    vy[x] = 0.0;
                    // Top
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
        }
    }

    /// Enforce boundary conditions on a scalar field (zero Neumann).
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

        if config.dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt: config.dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dt = config.dt;
        let n = nx * ny;

        let mut scratch_a = vec![0.0f64; n];
        let mut scratch_b = vec![0.0f64; n];
        let mut fft_scratch = Vec::new();
        let mut vort_buf = Vec::new();

        // 1. Advect velocity
        {
            let _span = trace_span!("grid::advect_velocity").entered();
            // Advect vx
            Self::advect(&mut scratch_a, &self.vx, &self.vx, &self.vy, nx, ny, dt);
            // Advect vy
            Self::advect(&mut scratch_b, &self.vy, &self.vx, &self.vy, nx, ny, dt);
            self.vx.copy_from_slice(&scratch_a);
            self.vy.copy_from_slice(&scratch_b);
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 2. Diffuse velocity
        if config.viscosity > 0.0 {
            Self::diffuse(
                &mut self.vx,
                nx,
                ny,
                config.viscosity,
                dt,
                config.diffusion_iterations,
            );
            Self::diffuse(
                &mut self.vy,
                nx,
                ny,
                config.viscosity,
                dt,
                config.diffusion_iterations,
            );
            Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);
        }

        // 3. Apply forces
        // Buoyancy: push density upward (vy += alpha * (density - ambient))
        if config.buoyancy_alpha != 0.0 {
            for i in 0..n {
                self.vy[i] +=
                    config.buoyancy_alpha * (self.density[i] - config.ambient_density) * dt;
            }
        }

        // Vorticity confinement
        if config.vorticity_confinement > 0.0 {
            Self::apply_vorticity_confinement(
                &mut self.vx,
                &mut self.vy,
                nx,
                ny,
                dt,
                config.vorticity_confinement,
                &mut vort_buf,
            );
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 4. Pressure projection
        {
            Self::divergence(&mut scratch_a, &self.vx, &self.vy, nx, ny);
            Self::pressure_solve(&mut self.pressure, &scratch_a, nx, ny, &mut fft_scratch);
            Self::project_velocity(&mut self.vx, &mut self.vy, &self.pressure, nx, ny);
        }
        Self::enforce_boundary(&mut self.vx, &mut self.vy, nx, ny, config.boundary);

        // 5. Advect density
        {
            let _span = trace_span!("grid::advect_density").entered();
            Self::advect(
                &mut scratch_a,
                &self.density,
                &self.vx,
                &self.vy,
                nx,
                ny,
                dt,
            );
            self.density.copy_from_slice(&scratch_a);
        }
        Self::enforce_scalar_boundary(&mut self.density, nx, ny);

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
    }

    #[test]
    fn test_grid_invalid_dx() {
        assert!(FluidGrid::new(10, 10, 0.0).is_err());
        assert!(FluidGrid::new(10, 10, -1.0).is_err());
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
        FluidGrid::diffuse(&mut field, 10, 10, 0.01, 0.001, 10);
        assert!(field.iter().all(|&v| v.abs() < f64::EPSILON));
    }

    #[test]
    fn test_diffuse_spreads() {
        let mut field = vec![0.0; 100];
        field[55] = 100.0;
        FluidGrid::diffuse(&mut field, 10, 10, 0.1, 0.01, 20);
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
        field[nx + 1] = 1.0; // (1,1) = 1.0
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
        // Midpoint between (1,1)=1.0 and (2,1)=3.0 should be 2.0
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
        FluidGrid::advect(&mut dst, &src, &vx, &vy, nx, ny, 0.1);
        // Zero velocity → field should not move
        assert!((dst[4 * nx + 4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_advect_uniform_velocity() {
        let nx = 16;
        let ny = 16;
        let n = nx * ny;
        let vx = vec![1.0; n]; // uniform rightward
        let vy = vec![0.0; n];
        let mut src = vec![0.0; n];
        src[8 * nx + 8] = 1.0;
        let mut dst = vec![0.0; n];
        FluidGrid::advect(&mut dst, &src, &vx, &vy, nx, ny, 1.0);
        // Semi-Lagrangian: dst[9,8] traces back to (8,8) which has value 1.0
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
        FluidGrid::divergence(&mut div, &vx, &vy, nx, ny);
        assert!(div.iter().all(|&v| v.abs() < f64::EPSILON));
    }

    #[test]
    fn test_pressure_solve_gs_zero_rhs() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let div = vec![0.0; n];
        let mut pressure = vec![1.0; n]; // nonzero initial
        FluidGrid::pressure_solve_gs(&mut pressure, &div, nx, ny, 50);
        // With zero RHS, pressure should converge to zero
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                assert!(pressure[y * nx + x].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_pressure_solve_fft_zero_rhs() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let div = vec![0.0; n];
        let mut pressure = vec![0.0; n];
        let mut scratch = Vec::new();
        FluidGrid::pressure_solve_fft(&mut pressure, &div, nx, ny, &mut scratch);
        assert!(pressure.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_projection_reduces_divergence() {
        // Use non-power-of-2 to test Gauss-Seidel solver
        let nx = 15;
        let ny = 15;
        let n = nx * ny;
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        // Create a divergent field: expanding source at center
        let cx = nx / 2;
        let cy = ny / 2;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                vx[i] = (x as f64 - cx as f64) * 0.1;
                vy[i] = (y as f64 - cy as f64) * 0.1;
            }
        }

        // Measure divergence before
        let mut div_before = vec![0.0; n];
        FluidGrid::divergence(&mut div_before, &vx, &vy, nx, ny);
        let div_mag_before: f64 = div_before.iter().map(|d| d * d).sum();

        // Project
        let mut pressure = vec![0.0; n];
        let mut scratch = Vec::new();
        FluidGrid::pressure_solve(&mut pressure, &div_before, nx, ny, &mut scratch);
        FluidGrid::project_velocity(&mut vx, &mut vy, &pressure, nx, ny);

        // Measure divergence after
        let mut div_after = vec![0.0; n];
        FluidGrid::divergence(&mut div_after, &vx, &vy, nx, ny);
        let div_mag_after: f64 = div_after.iter().map(|d| d * d).sum();

        assert!(
            div_mag_after < div_mag_before,
            "projection should reduce divergence: before={div_mag_before}, after={div_mag_after}"
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
        // Left wall: vx should be negated from interior
        assert!((vx[nx] + vx[nx + 1]).abs() < 1e-10); // row 1, col 0 mirrors col 1
        // Bottom wall: vy should be negated from interior
        assert!((vy[1] + vy[nx + 1]).abs() < 1e-10); // row 0, col 1 mirrors row 1
    }

    #[test]
    fn test_free_slip_boundary() {
        let nx = 8;
        let ny = 8;
        let n = nx * ny;
        let mut vx = vec![1.0; n];
        let mut vy = vec![2.0; n];
        FluidGrid::enforce_boundary(&mut vx, &mut vy, nx, ny, BoundaryCondition::FreeSlip);
        // Left wall: vx = 0, vy preserved
        assert!(vx[0].abs() < f64::EPSILON);
        // Bottom wall: vy = 0, vx preserved
        assert!(vy[1].abs() < f64::EPSILON);
    }

    // ── Full step tests ─────────────────────────────────────────────────────

    #[test]
    fn test_step_empty_grid_stable() {
        let mut g = FluidGrid::new(16, 16, 0.1).unwrap();
        let config = GridConfig::smoke();
        g.step(&config).unwrap();
        // Empty grid should remain zero
        assert!(g.max_speed() < f64::EPSILON);
    }

    #[test]
    fn test_step_smoke_plume() {
        let mut g = FluidGrid::new(30, 30, 0.1).unwrap();
        let config = GridConfig::smoke();

        for x in 13..17 {
            let i = 2 * 30 + x;
            g.density[i] = 1.0;
            g.vy[i] = 1.0;
        }

        for _ in 0..20 {
            for x in 13..17 {
                let i = 2 * 30 + x;
                g.density[i] = 1.0;
            }
            g.step(&config).unwrap();
        }

        // Density should exist above the source (row 2)
        let total_density: f64 = g.density.iter().sum();
        let source_density: f64 = (13..17).map(|x| g.density[2 * 30 + x]).sum();
        // Some density should exist outside the source row
        assert!(
            total_density > source_density + 0.001,
            "density should have spread: total={total_density}, source={source_density}"
        );
        // Velocity should have developed
        assert!(
            g.max_speed() > 0.01,
            "velocity should have developed: max_speed={}",
            g.max_speed()
        );
    }

    #[test]
    fn test_step_invalid_dt() {
        let mut g = FluidGrid::new(8, 8, 0.1).unwrap();
        let mut config = GridConfig::smoke();
        config.dt = -1.0;
        assert!(g.step(&config).is_err());
    }

    #[test]
    fn test_step_no_divergence_after_projection() {
        let mut g = FluidGrid::new(15, 15, 0.1).unwrap();
        let config = GridConfig::smoke();

        // Inject some velocity
        let center = 7 * 15 + 7;
        g.vx[center] = 5.0;
        g.vy[center] = 3.0;

        g.step(&config).unwrap();

        let mut div = vec![0.0; 15 * 15];
        FluidGrid::divergence(&mut div, &g.vx, &g.vy, 15, 15);
        let max_div: f64 = div.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
        assert!(
            max_div < 5.0,
            "divergence should be bounded after projection: {max_div}"
        );
    }
}
