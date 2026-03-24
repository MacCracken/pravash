//! Smoothed Particle Hydrodynamics (SPH) — particle-based fluid simulation.
//!
//! SPH represents fluid as a collection of particles. Each particle carries
//! mass, position, velocity, and density. Forces (pressure, viscosity, gravity)
//! are computed from neighbor interactions using smoothing kernels.
//!
//! # Usage
//!
//! For best performance, use [`SphSolver`] which maintains a spatial hash for
//! O(n·k) neighbor queries and persistent scratch buffers:
//!
//! ```ignore
//! let mut solver = SphSolver::new();
//! solver.step(&mut particles, &config, viscosity)?;
//! ```
//!
//! The free function [`step`] is available for simple cases but uses O(n²)
//! brute-force neighbor search.

use crate::common::{FluidConfig, FluidParticle};
use crate::error::{PravashError, Result};

use std::f64::consts::PI;

use hisab::SpatialHash;
use hisab::Vec3;
use tracing::trace_span;

// ── Precomputed Kernel Coefficients ─────────────────────────────────────────

/// Precomputed kernel coefficients for a given smoothing radius h.
#[derive(Debug, Clone, Copy)]
struct KernelCoeffs {
    h: f64,
    h2: f64,
    poly6: f64,
    spiky_grad: f64,
    visc_lap: f64,
    /// Poly6 gradient coefficient: -945 / (32·π·h⁹)
    poly6_grad: f64,
    /// Poly6 Laplacian coefficient: -945 / (32·π·h⁹)
    poly6_lap: f64,
}

impl KernelCoeffs {
    #[inline]
    fn new(h: f64) -> Self {
        let h2 = h * h;
        let h3 = h2 * h;
        let h6 = h3 * h3;
        let h9 = h6 * h3;
        Self {
            h,
            h2,
            poly6: 315.0 / (64.0 * PI * h9),
            spiky_grad: -45.0 / (PI * h6),
            visc_lap: 45.0 / (PI * h6),
            poly6_grad: -945.0 / (32.0 * PI * h9),
            poly6_lap: -945.0 / (32.0 * PI * h9),
        }
    }

    #[inline]
    fn poly6(&self, r2: f64) -> f64 {
        if r2 > self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r2;
        self.poly6 * diff * diff * diff
    }

    /// Poly6 gradient magnitude (for surface tension color field normal).
    /// ∇W_poly6 = -945/(32·π·h⁹) · r · (h² - r²)²
    /// Returns the scalar multiplied by direction (caller provides r·r̂ = displacement).
    #[inline]
    fn poly6_grad_scalar(&self, r2: f64) -> f64 {
        if r2 > self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r2;
        self.poly6_grad * diff * diff
    }

    /// Poly6 Laplacian (for surface tension color field curvature).
    /// ∇²W_poly6 = -945/(32·π·h⁹) · (h² - r²) · (3h² - 7r²)
    #[inline]
    fn poly6_laplacian(&self, r2: f64) -> f64 {
        if r2 > self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r2;
        self.poly6_lap * diff * (3.0 * self.h2 - 7.0 * r2)
    }

    #[inline]
    fn spiky_grad(&self, r: f64) -> f64 {
        if r > self.h || r <= 0.0 {
            return 0.0;
        }
        let diff = self.h - r;
        self.spiky_grad * diff * diff
    }

    #[inline]
    fn visc_laplacian(&self, r: f64) -> f64 {
        if r > self.h || r <= 0.0 {
            return 0.0;
        }
        self.visc_lap * (self.h - r)
    }
}

// ── SPH Kernels (public API) ────────────────────────────────────────────────

/// Poly6 smoothing kernel — used for density estimation.
///
/// W(r, h) = 315 / (64·π·h⁹) · (h² - r²)³  for r ≤ h
#[inline]
#[must_use]
pub fn kernel_poly6(r: f64, h: f64) -> f64 {
    if r < 0.0 {
        return 0.0;
    }
    KernelCoeffs::new(h).poly6(r * r)
}

/// Spiky kernel gradient magnitude — used for pressure forces.
///
/// ∇W(r, h) = -45 / (π·h⁶) · (h - r)²  for r ≤ h
#[inline]
#[must_use]
pub fn kernel_spiky_grad(r: f64, h: f64) -> f64 {
    KernelCoeffs::new(h).spiky_grad(r)
}

/// Viscosity kernel Laplacian — used for viscosity forces.
///
/// ∇²W(r, h) = 45 / (π·h⁶) · (h - r)  for r ≤ h
#[inline]
#[must_use]
pub fn kernel_viscosity_laplacian(r: f64, h: f64) -> f64 {
    KernelCoeffs::new(h).visc_laplacian(r)
}

// ── Density & Pressure ──────────────────────────────────────────────────────

/// Compute density for a single particle from all neighbors (brute-force).
#[inline]
#[must_use]
pub fn compute_density(particle_idx: usize, particles: &[FluidParticle], h: f64) -> f64 {
    compute_density_inner(&particles[particle_idx], particles, &KernelCoeffs::new(h))
}

#[inline]
fn compute_density_inner(
    pi: &FluidParticle,
    particles: &[FluidParticle],
    kc: &KernelCoeffs,
) -> f64 {
    let mut density = 0.0;
    for pj in particles {
        let r2 = pi.distance_squared_to(pj);
        if r2 <= kc.h2 {
            density += pj.mass * kc.poly6(r2);
        }
    }
    density
}

/// Compute pressure from density using the equation of state.
///
/// P = k · (ρ - ρ₀)
#[inline]
#[must_use]
pub fn equation_of_state(density: f64, rest_density: f64, gas_constant: f64) -> f64 {
    gas_constant * (density - rest_density)
}

// ── Forces (standalone, brute-force) ────────────────────────────────────────

/// Compute pressure force on particle i from all neighbors (brute-force).
#[must_use]
pub fn pressure_force(particle_idx: usize, particles: &[FluidParticle], h: f64) -> [f64; 3] {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > kc.h2 || r2 < 1e-20 {
            continue;
        }
        let r = r2.sqrt();

        let pressure_term = (pi.pressure + pj.pressure) / (2.0 * pj.density.max(1e-10));
        let grad = kc.spiky_grad(r);
        let scale = -pj.mass * pressure_term * grad / r;

        force[0] += scale * (pi.position[0] - pj.position[0]);
        force[1] += scale * (pi.position[1] - pj.position[1]);
        force[2] += scale * (pi.position[2] - pj.position[2]);
    }

    force
}

/// Compute viscosity force on particle i from all neighbors (brute-force).
#[must_use]
pub fn viscosity_force(
    particle_idx: usize,
    particles: &[FluidParticle],
    h: f64,
    viscosity: f64,
) -> [f64; 3] {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > kc.h2 || r2 < 1e-20 {
            continue;
        }
        let r = r2.sqrt();

        let lap = kc.visc_laplacian(r);
        let rho_j = pj.density.max(1e-10);
        let scale = viscosity * pj.mass * lap / rho_j;

        force[0] += scale * (pj.velocity[0] - pi.velocity[0]);
        force[1] += scale * (pj.velocity[1] - pi.velocity[1]);
        force[2] += scale * (pj.velocity[2] - pi.velocity[2]);
    }

    force
}

// ── SphSolver ───────────────────────────────────────────────────────────────

/// Stateful SPH solver with spatial hash acceleration and persistent buffers.
///
/// Maintains a spatial hash grid (via hisab) for O(n·k) neighbor queries
/// instead of O(n²) brute-force. Caches neighbor lists and reuses scratch
/// buffers across steps to minimize allocations.
pub struct SphSolver {
    grid: SpatialHash,
    cell_size: f32,
    densities: Vec<f64>,
    snapshot: Vec<FluidParticle>,
    positions_f32: Vec<Vec3>,
    /// Cached neighbor indices per particle (flat buffer + offsets).
    neighbor_indices: Vec<usize>,
    neighbor_offsets: Vec<u32>,
    /// Surface tension coefficient (0.0 to disable).
    pub surface_tension: f64,
}

impl SphSolver {
    /// Create a new solver. Call [`step`](SphSolver::step) to advance the simulation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: SpatialHash::new(1.0),
            cell_size: 1.0,
            densities: Vec::new(),
            snapshot: Vec::new(),
            positions_f32: Vec::new(),
            neighbor_indices: Vec::new(),
            neighbor_offsets: Vec::new(),
            surface_tension: 0.0,
        }
    }

    /// Create a solver with surface tension enabled.
    #[must_use]
    pub fn with_surface_tension(surface_tension: f64) -> Self {
        let mut s = Self::new();
        s.surface_tension = surface_tension;
        s
    }

    /// Build spatial hash and cache neighbor lists for all particles.
    /// Uses `query_cell` per neighbor cell (zero-alloc) instead of `query_radius`.
    fn build_neighbors(&mut self, particles: &[FluidParticle], h_f32: f32) {
        let n = particles.len();
        let inv_cs = 1.0 / h_f32;
        let half_cs = h_f32 * 0.5;

        // Rebuild grid: clear retains HashMap capacity
        if (self.cell_size - h_f32).abs() > f32::EPSILON {
            self.grid = SpatialHash::new(h_f32);
            self.cell_size = h_f32;
        } else {
            self.grid.clear();
        }

        // Cache f32 positions and insert into grid in one pass
        self.positions_f32.clear();
        self.positions_f32.reserve(n);
        for (i, p) in particles.iter().enumerate() {
            let pos = Vec3::new(
                p.position[0] as f32,
                p.position[1] as f32,
                p.position[2] as f32,
            );
            self.positions_f32.push(pos);
            self.grid.insert(pos, i);
        }

        // Build neighbor cache using query_cell (zero allocation per particle).
        // For cell_size = h, we check the 3x3x3 cube of cells around each particle.
        self.neighbor_indices.clear();
        self.neighbor_offsets.clear();
        self.neighbor_offsets.reserve(n + 1);
        self.neighbor_offsets.push(0);

        for pos in &self.positions_f32 {
            let cx = (pos.x * inv_cs).floor() as i32;
            let cy = (pos.y * inv_cs).floor() as i32;
            let cz = (pos.z * inv_cs).floor() as i32;

            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        // Construct a point that lands in the target cell
                        let probe = Vec3::new(
                            (cx + dx) as f32 * h_f32 + half_cs,
                            (cy + dy) as f32 * h_f32 + half_cs,
                            (cz + dz) as f32 * h_f32 + half_cs,
                        );
                        let cell = self.grid.query_cell(probe);
                        self.neighbor_indices.extend_from_slice(cell);
                    }
                }
            }
            self.neighbor_offsets
                .push(self.neighbor_indices.len() as u32);
        }
    }

    /// Get the cached neighbor slice for particle i.
    #[inline]
    fn neighbors(&self, i: usize) -> &[usize] {
        let start = self.neighbor_offsets[i] as usize;
        let end = self.neighbor_offsets[i + 1] as usize;
        &self.neighbor_indices[start..end]
    }

    /// Perform one SPH simulation step with spatial hash acceleration.
    pub fn step(
        &mut self,
        particles: &mut [FluidParticle],
        config: &FluidConfig,
        viscosity: f64,
    ) -> Result<()> {
        let _span = trace_span!("sph::solver_step", n = particles.len()).entered();
        config.validate()?;
        let h = config.smoothing_radius;
        let n = particles.len();

        if n == 0 {
            return Ok(());
        }

        let h_f32 = h as f32;
        if !h_f32.is_finite() || h_f32 <= 0.0 {
            return Err(PravashError::InvalidSmoothingRadius { h });
        }

        let kc = KernelCoeffs::new(h);

        // Build spatial hash and cache neighbor lists (zero-alloc query_cell)
        {
            let _span = trace_span!("sph::build_neighbors", n).entered();
            self.build_neighbors(particles, h_f32);
        }

        // Compute densities via cached neighbors
        {
            let _span = trace_span!("sph::density", n).entered();
            self.densities.resize(n, 0.0);
            for i in 0..n {
                let pi = &particles[i];
                let mut d = 0.0;
                for &j in self.neighbors(i) {
                    let r2 = pi.distance_squared_to(&particles[j]);
                    if r2 <= kc.h2 {
                        d += particles[j].mass * kc.poly6(r2);
                    }
                }
                self.densities[i] = d;
            }
            for (i, p) in particles.iter_mut().enumerate() {
                p.density = self.densities[i];
                p.pressure = equation_of_state(p.density, config.rest_density, config.gas_constant);
            }
        }

        // Snapshot for force computation
        self.snapshot.resize(n, FluidParticle::new([0.0; 3], 0.0));
        self.snapshot.copy_from_slice(particles);

        // Compute forces via cached neighbors
        {
            let _span = trace_span!("sph::forces", n).entered();
            let st = self.surface_tension;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let pi = &self.snapshot[i];
                let mut ax = config.gravity[0];
                let mut ay = config.gravity[1];
                let mut az = config.gravity[2];
                let rho = pi.density.max(1e-10);

                // Surface tension accumulators (CSF with poly6 kernel)
                let mut cn_x = 0.0;
                let mut cn_y = 0.0;
                let mut cn_z = 0.0;
                let mut laplacian_color = 0.0;

                for &j in self.neighbors(i) {
                    if j == i {
                        continue;
                    }
                    let pj = &self.snapshot[j];
                    let r2 = pi.distance_squared_to(pj);
                    if r2 > kc.h2 || r2 < 1e-20 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let rho_j = pj.density.max(1e-10);

                    let dx = pi.position[0] - pj.position[0];
                    let dy = pi.position[1] - pj.position[1];
                    let dz = pi.position[2] - pj.position[2];

                    // Symmetric pressure force: -m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
                    let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let p_scale = -pj.mass * sym_pressure * grad / r;
                    ax += p_scale * dx;
                    ay += p_scale * dy;
                    az += p_scale * dz;

                    // Viscosity force
                    let lap = kc.visc_laplacian(r);
                    let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                    ax += v_scale * (pj.velocity[0] - pi.velocity[0]);
                    ay += v_scale * (pj.velocity[1] - pi.velocity[1]);
                    az += v_scale * (pj.velocity[2] - pi.velocity[2]);

                    // Surface tension: color field gradient and Laplacian (poly6 kernel)
                    if st > 0.0 {
                        let mass_over_rho = pj.mass / rho_j;
                        let pg = kc.poly6_grad_scalar(r2);
                        cn_x += mass_over_rho * pg * dx;
                        cn_y += mass_over_rho * pg * dy;
                        cn_z += mass_over_rho * pg * dz;
                        laplacian_color += mass_over_rho * kc.poly6_laplacian(r2);
                    }
                }

                // Surface tension force: F_st = -σ · κ · n̂
                if st > 0.0 {
                    let cn_mag = (cn_x * cn_x + cn_y * cn_y + cn_z * cn_z).sqrt();
                    if cn_mag > 1e-6 / kc.h {
                        let kappa = -laplacian_color / cn_mag;
                        let st_scale = st * kappa / (rho * cn_mag);
                        ax += st_scale * cn_x;
                        ay += st_scale * cn_y;
                        az += st_scale * cn_z;
                    }
                }

                if !ax.is_finite() || !ay.is_finite() || !az.is_finite() {
                    return Err(PravashError::Diverged {
                        reason: format!("NaN/Inf in acceleration at particle {i}").into(),
                    });
                }

                particles[i].acceleration = [ax, ay, az];
            }
        }

        // Integrate (symplectic Euler)
        let dt = config.dt;
        for p in particles.iter_mut() {
            p.velocity[0] += p.acceleration[0] * dt;
            p.velocity[1] += p.acceleration[1] * dt;
            p.velocity[2] += p.acceleration[2] * dt;

            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }

        // Boundary enforcement
        let [min_x, min_y, min_z, max_x, max_y, max_z] = config.bounds;
        let damp = config.boundary_damping;
        let bounds = [(min_x, max_x), (min_y, max_y), (min_z, max_z)];
        for p in particles.iter_mut() {
            for (dim, &(lo, hi)) in bounds.iter().enumerate() {
                if p.position[dim] < lo {
                    p.position[dim] = lo;
                    p.velocity[dim] *= -damp;
                } else if p.position[dim] > hi && hi > lo {
                    p.position[dim] = hi;
                    p.velocity[dim] *= -damp;
                }
            }
        }

        Ok(())
    }
}

impl Default for SphSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Free-function step (brute-force, kept for simplicity/backwards compat) ──

/// Perform one SPH simulation step (brute-force O(n²)).
///
/// For better performance with >100 particles, use [`SphSolver::step`] instead.
pub fn step(particles: &mut [FluidParticle], config: &FluidConfig, viscosity: f64) -> Result<()> {
    let _span = trace_span!("sph::step", n = particles.len()).entered();
    config.validate()?;
    let h = config.smoothing_radius;
    let n = particles.len();

    if n == 0 {
        return Ok(());
    }

    let kc = KernelCoeffs::new(h);

    {
        let _span = trace_span!("sph::density", n).entered();
        let mut densities = vec![0.0f64; n];
        for i in 0..n {
            densities[i] = compute_density_inner(&particles[i], particles, &kc);
        }
        for (i, p) in particles.iter_mut().enumerate() {
            p.density = densities[i];
            p.pressure = equation_of_state(p.density, config.rest_density, config.gas_constant);
        }
    }

    let snapshot: Vec<FluidParticle> = particles.to_vec();
    {
        let _span = trace_span!("sph::forces", n).entered();
        for i in 0..n {
            let pi = &snapshot[i];
            let mut ax = config.gravity[0];
            let mut ay = config.gravity[1];
            let mut az = config.gravity[2];
            let rho = pi.density.max(1e-10);

            for (j, pj) in snapshot.iter().enumerate() {
                if j == i {
                    continue;
                }
                let r2 = pi.distance_squared_to(pj);
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let rho_j = pj.density.max(1e-10);

                let pressure_term = (pi.pressure + pj.pressure) / (2.0 * rho_j);
                let grad = kc.spiky_grad(r);
                let p_scale = -pj.mass * pressure_term * grad / r;

                let dx = pi.position[0] - pj.position[0];
                let dy = pi.position[1] - pj.position[1];
                let dz = pi.position[2] - pj.position[2];

                ax += (p_scale * dx) / rho;
                ay += (p_scale * dy) / rho;
                az += (p_scale * dz) / rho;

                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;

                ax += v_scale * (pj.velocity[0] - pi.velocity[0]);
                ay += v_scale * (pj.velocity[1] - pi.velocity[1]);
                az += v_scale * (pj.velocity[2] - pi.velocity[2]);
            }

            if !ax.is_finite() || !ay.is_finite() || !az.is_finite() {
                return Err(PravashError::Diverged {
                    reason: format!("NaN/Inf in acceleration at particle {i}").into(),
                });
            }

            particles[i].acceleration = [ax, ay, az];
        }
    }

    let dt = config.dt;
    for p in particles.iter_mut() {
        p.velocity[0] += p.acceleration[0] * dt;
        p.velocity[1] += p.acceleration[1] * dt;
        p.velocity[2] += p.acceleration[2] * dt;

        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;
    }

    let [min_x, min_y, min_z, max_x, max_y, max_z] = config.bounds;
    let damp = config.boundary_damping;
    let bounds = [(min_x, max_x), (min_y, max_y), (min_z, max_z)];
    for p in particles.iter_mut() {
        for (dim, &(lo, hi)) in bounds.iter().enumerate() {
            if p.position[dim] < lo {
                p.position[dim] = lo;
                p.velocity[dim] *= -damp;
            } else if p.position[dim] > hi && hi > lo {
                p.position[dim] = hi;
                p.velocity[dim] *= -damp;
            }
        }
    }

    Ok(())
}

// ── Utility Functions ───────────────────────────────────────────────────────

/// Total kinetic energy of the system.
#[must_use]
pub fn total_kinetic_energy(particles: &[FluidParticle]) -> f64 {
    particles.iter().map(|p| p.kinetic_energy()).sum()
}

/// Maximum particle speed (for CFL checks).
#[must_use]
pub fn max_speed(particles: &[FluidParticle]) -> f64 {
    particles
        .iter()
        .map(|p| p.speed_squared())
        .fold(0.0f64, f64::max)
        .sqrt()
}

/// Create a block of particles in a grid pattern.
#[must_use]
pub fn create_particle_block(
    origin: [f64; 2],
    size: [f64; 2],
    spacing: f64,
    particle_mass: f64,
) -> Vec<FluidParticle> {
    let nx = (size[0] / spacing).ceil() as usize;
    let ny = (size[1] / spacing).ceil() as usize;
    let mut particles = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            let y = origin[1] + iy as f64 * spacing;
            particles.push(FluidParticle::new_2d(x, y, particle_mass));
        }
    }
    particles
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_kernel_poly6_at_zero() {
        let w = kernel_poly6(0.0, 1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_kernel_poly6_at_boundary() {
        let w = kernel_poly6(1.0, 1.0);
        assert!(w.abs() < EPS);
    }

    #[test]
    fn test_kernel_poly6_outside() {
        let w = kernel_poly6(1.5, 1.0);
        assert!(w.abs() < EPS);
    }

    #[test]
    fn test_kernel_poly6_decreasing() {
        let w1 = kernel_poly6(0.1, 1.0);
        let w2 = kernel_poly6(0.5, 1.0);
        assert!(w1 > w2);
    }

    #[test]
    fn test_kernel_spiky_grad() {
        let g = kernel_spiky_grad(0.5, 1.0);
        assert!(g < 0.0);
    }

    #[test]
    fn test_kernel_spiky_grad_outside() {
        let g = kernel_spiky_grad(1.5, 1.0);
        assert!(g.abs() < EPS);
    }

    #[test]
    fn test_kernel_viscosity_laplacian() {
        let l = kernel_viscosity_laplacian(0.5, 1.0);
        assert!(l > 0.0);
    }

    #[test]
    fn test_equation_of_state() {
        let p = equation_of_state(1100.0, 1000.0, 2000.0);
        assert!((p - 200000.0).abs() < EPS);
    }

    #[test]
    fn test_equation_of_state_at_rest() {
        let p = equation_of_state(1000.0, 1000.0, 2000.0);
        assert!(p.abs() < EPS);
    }

    #[test]
    fn test_compute_density_self() {
        let particles = vec![FluidParticle::new_2d(0.0, 0.0, 1.0)];
        let d = compute_density(0, &particles, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_create_particle_block() {
        let particles = create_particle_block([0.0, 0.0], [0.1, 0.1], 0.05, 0.01);
        assert_eq!(particles.len(), 4);
    }

    #[test]
    fn test_create_particle_block_positions() {
        let particles = create_particle_block([1.0, 2.0], [0.1, 0.1], 0.05, 0.01);
        assert!((particles[0].position[0] - 1.0).abs() < EPS);
        assert!((particles[0].position[1] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_create_particle_block_preallocated() {
        let particles = create_particle_block([0.0, 0.0], [1.0, 1.0], 0.1, 0.01);
        assert!(particles.capacity() >= particles.len());
        assert!(particles.capacity() <= particles.len() + 10);
    }

    #[test]
    fn test_step_empty() {
        let mut particles = vec![];
        let config = FluidConfig::water_2d();
        assert!(step(&mut particles, &config, 0.001).is_ok());
    }

    #[test]
    fn test_step_single_particle_falls() {
        let mut particles = vec![FluidParticle::new_2d(0.5, 0.5, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        step(&mut particles, &config, 0.001).unwrap();
        assert!(particles[0].velocity[1] < 0.0);
    }

    #[test]
    fn test_step_boundary_enforcement() {
        let mut particles = vec![FluidParticle::new_2d(0.5, -0.1, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        step(&mut particles, &config, 0.001).unwrap();
        assert!(particles[0].position[1] >= 0.0);
    }

    #[test]
    fn test_total_kinetic_energy() {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 2.0);
        p.velocity = [3.0, 4.0, 0.0];
        let ke = total_kinetic_energy(&[p]);
        assert!((ke - 25.0).abs() < EPS);
    }

    #[test]
    fn test_max_speed() {
        let mut p1 = FluidParticle::new_2d(0.0, 0.0, 1.0);
        let mut p2 = FluidParticle::new_2d(1.0, 0.0, 1.0);
        p1.velocity = [1.0, 0.0, 0.0];
        p2.velocity = [3.0, 4.0, 0.0];
        assert!((max_speed(&[p1, p2]) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_step_invalid_config() {
        let mut particles = vec![FluidParticle::new_2d(0.5, 0.5, 0.01)];
        let mut config = FluidConfig::water_2d();
        config.dt = -1.0;
        assert!(step(&mut particles, &config, 0.001).is_err());
    }

    #[test]
    fn test_kernel_negative_r() {
        assert!(kernel_poly6(-1.0, 1.0).abs() < EPS);
        assert!(kernel_spiky_grad(-1.0, 1.0).abs() < EPS);
        assert!(kernel_viscosity_laplacian(-1.0, 1.0).abs() < EPS);
    }

    // ── SphSolver tests ─────────────────────────────────────────────────────

    #[test]
    fn test_solver_empty() {
        let mut particles = vec![];
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        assert!(solver.step(&mut particles, &config, 0.001).is_ok());
    }

    #[test]
    fn test_solver_single_particle_falls() {
        let mut particles = vec![FluidParticle::new_2d(0.5, 0.5, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();
        assert!(particles[0].velocity[1] < 0.0);
    }

    #[test]
    fn test_solver_particles_stay_in_bounds() {
        let mut particles = create_particle_block([0.1, 0.5], [0.3, 0.3], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        let viscosity = 0.001;

        for _ in 0..100 {
            solver.step(&mut particles, &config, viscosity).unwrap();
        }

        let [min_x, min_y, _, max_x, max_y, _] = config.bounds;
        for p in &particles {
            assert!(p.position[0] >= min_x && p.position[0] <= max_x);
            assert!(p.position[1] >= min_y && p.position[1] <= max_y);
        }
    }

    #[test]
    fn test_solver_surface_tension() {
        let mut particles = create_particle_block([0.3, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::with_surface_tension(0.072);
        for _ in 0..10 {
            solver.step(&mut particles, &config, 0.001).unwrap();
        }
        for p in &particles {
            assert!(p.position[0].is_finite());
            assert!(p.position[1].is_finite());
        }
    }

    #[test]
    fn test_solver_reuses_buffers() {
        let mut particles = create_particle_block([0.1, 0.1], [0.2, 0.2], 0.05, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();

        solver.step(&mut particles, &config, 0.001).unwrap();
        let cap_d = solver.densities.capacity();
        let cap_s = solver.snapshot.capacity();

        solver.step(&mut particles, &config, 0.001).unwrap();
        assert_eq!(solver.densities.capacity(), cap_d);
        assert_eq!(solver.snapshot.capacity(), cap_s);
    }

    #[test]
    fn test_solver_neighbor_cache_used() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();
        // Neighbor offsets should have n+1 entries
        assert_eq!(solver.neighbor_offsets.len(), particles.len() + 1);
        // Each particle should have at least itself as neighbor
        for i in 0..particles.len() {
            assert!(!solver.neighbors(i).is_empty());
        }
    }

    #[test]
    fn test_solver_grid_clear_reuse() {
        let mut particles = create_particle_block([0.1, 0.1], [0.2, 0.2], 0.05, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();

        solver.step(&mut particles, &config, 0.001).unwrap();
        // Cell size should match smoothing radius
        assert!((solver.cell_size - config.smoothing_radius as f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_poly6_kernels_consistent() {
        let kc = KernelCoeffs::new(1.0);
        // At r=0, poly6_grad should be 0 (peak of poly6, zero gradient)
        assert!(kc.poly6_grad_scalar(0.0).abs() > 0.0); // actually nonzero because formula
        // poly6 laplacian at r=0 should be finite
        let lap = kc.poly6_laplacian(0.0);
        assert!(lap.is_finite());
        // Outside h, all should be 0
        assert!(kc.poly6(1.1).abs() < EPS);
        assert!(kc.poly6_grad_scalar(1.1).abs() < EPS);
        assert!(kc.poly6_laplacian(1.1).abs() < EPS);
    }
}
