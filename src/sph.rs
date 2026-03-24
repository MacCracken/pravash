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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
///
/// Uses the symmetric momentum-conserving formula:
/// F_i = -m_i Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W(r_ij, h)
#[must_use]
pub fn pressure_force(particle_idx: usize, particles: &[FluidParticle], h: f64) -> [f64; 3] {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];
    let rho_i = pi.density.max(1e-10);

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > kc.h2 || r2 < 1e-20 {
            continue;
        }
        let r = r2.sqrt();
        let rho_j = pj.density.max(1e-10);

        let sym_pressure = pi.pressure / (rho_i * rho_i) + pj.pressure / (rho_j * rho_j);
        let grad = kc.spiky_grad(r);
        let scale = -pi.mass * pj.mass * sym_pressure * grad / r;

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
    // PCISPH scratch buffers (persistent across steps)
    pcisph_accel: Vec<[f64; 3]>,
    pcisph_pressures: Vec<f64>,
    pcisph_pred_pos: Vec<[f64; 3]>,
    pcisph_pred_vel: Vec<[f64; 3]>,
}

impl SphSolver {
    /// Create a new solver. Call [`step`](SphSolver::step) to advance the simulation.
    #[must_use]
    pub fn new() -> Self {
        // SAFETY: 1.0 is always a valid cell size (positive, finite).
        // SpatialHash::new only fails for non-positive or non-finite values.
        let grid = match SpatialHash::new(1.0) {
            Ok(g) => g,
            Err(_) => unreachable!(),
        };
        Self {
            grid,
            cell_size: 1.0,
            densities: Vec::new(),
            snapshot: Vec::new(),
            positions_f32: Vec::new(),
            neighbor_indices: Vec::new(),
            neighbor_offsets: Vec::new(),
            surface_tension: 0.0,
            pcisph_accel: Vec::new(),
            pcisph_pressures: Vec::new(),
            pcisph_pred_pos: Vec::new(),
            pcisph_pred_vel: Vec::new(),
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
    fn build_neighbors(&mut self, particles: &[FluidParticle], h: f64, h_f32: f32) -> Result<()> {
        let n = particles.len();
        let inv_cs = 1.0 / h_f32;
        let half_cs = h_f32 * 0.5;

        // Rebuild grid: clear retains HashMap capacity
        if (self.cell_size - h_f32).abs() > f32::EPSILON {
            self.grid =
                SpatialHash::new(h_f32).map_err(|_| PravashError::InvalidSmoothingRadius { h })?;
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
        Ok(())
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
            self.build_neighbors(particles, h, h_f32)?;
        }

        // Compute densities via cached neighbors
        {
            let _span = trace_span!("sph::density", n).entered();
            self.densities.resize(n, 0.0);
            let offsets = &self.neighbor_offsets;
            let indices = &self.neighbor_indices;
            let compute_density_i = |i: usize| -> f64 {
                let pi = &particles[i];
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                let mut d = 0.0;
                for &j in &indices[start..end] {
                    let r2 = pi.distance_squared_to(&particles[j]);
                    if r2 <= kc.h2 {
                        d += particles[j].mass * kc.poly6(r2);
                    }
                }
                d
            };

            #[cfg(feature = "parallel")]
            {
                let densities: Vec<f64> = (0..n).into_par_iter().map(compute_density_i).collect();
                self.densities.copy_from_slice(&densities);
            }
            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..n {
                    self.densities[i] = compute_density_i(i);
                }
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
            let snapshot = &self.snapshot;
            let offsets = &self.neighbor_offsets;
            let indices = &self.neighbor_indices;
            let gravity = config.gravity;

            let compute_accel_i = |i: usize| -> [f64; 3] {
                let pi = &snapshot[i];
                let mut ax = gravity[0];
                let mut ay = gravity[1];
                let mut az = gravity[2];
                let rho = pi.density.max(1e-10);

                let mut cn_x = 0.0;
                let mut cn_y = 0.0;
                let mut cn_z = 0.0;
                let mut laplacian_color = 0.0;

                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                for &j in &indices[start..end] {
                    if j == i {
                        continue;
                    }
                    let pj = &snapshot[j];
                    let r2 = pi.distance_squared_to(pj);
                    if r2 > kc.h2 || r2 < 1e-20 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let rho_j = pj.density.max(1e-10);

                    let dx = pi.position[0] - pj.position[0];
                    let dy = pi.position[1] - pj.position[1];
                    let dz = pi.position[2] - pj.position[2];

                    let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let p_scale = -pj.mass * sym_pressure * grad / r;
                    ax += p_scale * dx;
                    ay += p_scale * dy;
                    az += p_scale * dz;

                    let lap = kc.visc_laplacian(r);
                    let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                    ax += v_scale * (pj.velocity[0] - pi.velocity[0]);
                    ay += v_scale * (pj.velocity[1] - pi.velocity[1]);
                    az += v_scale * (pj.velocity[2] - pi.velocity[2]);

                    if st > 0.0 {
                        let mass_over_rho = pj.mass / rho_j;
                        let pg = kc.poly6_grad_scalar(r2);
                        cn_x += mass_over_rho * pg * dx;
                        cn_y += mass_over_rho * pg * dy;
                        cn_z += mass_over_rho * pg * dz;
                        laplacian_color += mass_over_rho * kc.poly6_laplacian(r2);
                    }
                }

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

                [ax, ay, az]
            };

            #[cfg(feature = "parallel")]
            let accels: Vec<[f64; 3]> = (0..n).into_par_iter().map(compute_accel_i).collect();
            #[cfg(not(feature = "parallel"))]
            let accels: Vec<[f64; 3]> = (0..n).map(compute_accel_i).collect();

            for (i, accel) in accels.iter().enumerate() {
                if !accel[0].is_finite() || !accel[1].is_finite() || !accel[2].is_finite() {
                    return Err(PravashError::Diverged {
                        reason: format!("NaN/Inf in acceleration at particle {i}").into(),
                    });
                }
                particles[i].acceleration = *accel;
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

    /// Perform one PCISPH step (Predictive-Corrective Incompressible SPH).
    ///
    /// Iteratively corrects pressure to enforce near-constant density.
    /// `max_iterations` controls convergence (typically 3-10).
    /// `max_density_error` is the convergence threshold (e.g., 0.01 = 1%).
    pub fn step_pcisph(
        &mut self,
        particles: &mut [FluidParticle],
        config: &FluidConfig,
        viscosity: f64,
        max_iterations: usize,
        max_density_error: f64,
    ) -> Result<()> {
        let _span = trace_span!("sph::pcisph", n = particles.len()).entered();
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
        let dt = config.dt;
        let rest_density = config.rest_density;

        // Build neighbors
        {
            let _span = trace_span!("sph::build_neighbors", n).entered();
            self.build_neighbors(particles, h, h_f32)?;
        }

        // Compute densities
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
            p.pressure = 0.0; // start with zero pressure
        }

        // Compute non-pressure accelerations (viscosity + gravity + surface tension)
        self.snapshot.resize(n, FluidParticle::new([0.0; 3], 0.0));
        self.snapshot.copy_from_slice(particles);

        let st = self.surface_tension;
        // Take persistent buffers out of self to avoid borrow conflicts
        let mut non_pressure_accel = std::mem::take(&mut self.pcisph_accel);
        non_pressure_accel.resize(n, [0.0; 3]);
        non_pressure_accel.fill([0.0; 3]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let pi = &self.snapshot[i];
            let mut ax = config.gravity[0];
            let mut ay = config.gravity[1];
            let mut az = config.gravity[2];
            let rho = pi.density.max(1e-10);

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

                // Viscosity
                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                ax += v_scale * (pj.velocity[0] - pi.velocity[0]);
                ay += v_scale * (pj.velocity[1] - pi.velocity[1]);
                az += v_scale * (pj.velocity[2] - pi.velocity[2]);

                if st > 0.0 {
                    let dx = pi.position[0] - pj.position[0];
                    let dy = pi.position[1] - pj.position[1];
                    let dz = pi.position[2] - pj.position[2];
                    let mass_over_rho = pj.mass / rho_j;
                    let pg = kc.poly6_grad_scalar(r2);
                    cn_x += mass_over_rho * pg * dx;
                    cn_y += mass_over_rho * pg * dy;
                    cn_z += mass_over_rho * pg * dz;
                    laplacian_color += mass_over_rho * kc.poly6_laplacian(r2);
                }
            }

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
                self.pcisph_accel = non_pressure_accel;
                return Err(PravashError::Diverged {
                    reason: format!("NaN/Inf in non-pressure acceleration at particle {i}").into(),
                });
            }

            non_pressure_accel[i] = [ax, ay, az];
        }

        // PCISPH pressure correction loop
        // Precompute scaling factor δ from kernel gradient sums (Solenthaler 2009).
        // Use a representative particle (particle 0) to estimate the scaling.
        let delta = {
            let pi = &self.snapshot[0];
            let mut sum_grad = [0.0f64; 3];
            let mut sum_grad_sq = 0.0f64;
            for &j in self.neighbors(0) {
                if j == 0 {
                    continue;
                }
                let pj = &self.snapshot[j];
                let r2 = pi.distance_squared_to(pj);
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let grad = kc.spiky_grad(r);
                let grad_w = [
                    grad / r * (pi.position[0] - pj.position[0]),
                    grad / r * (pi.position[1] - pj.position[1]),
                    grad / r * (pi.position[2] - pj.position[2]),
                ];
                sum_grad[0] += grad_w[0];
                sum_grad[1] += grad_w[1];
                sum_grad[2] += grad_w[2];
                sum_grad_sq +=
                    grad_w[0] * grad_w[0] + grad_w[1] * grad_w[1] + grad_w[2] * grad_w[2];
            }
            let sum_dot =
                sum_grad[0] * sum_grad[0] + sum_grad[1] * sum_grad[1] + sum_grad[2] * sum_grad[2];
            let beta = 2.0 * (dt * self.snapshot[0].mass / rest_density).powi(2);
            let denom = beta * (sum_dot + sum_grad_sq);
            if denom.abs() < 1e-20 {
                0.0 // degenerate (single particle or no neighbors)
            } else {
                1.0 / denom
            }
        };

        let mut pressures = std::mem::take(&mut self.pcisph_pressures);
        pressures.resize(n, 0.0);
        pressures.fill(0.0);
        let mut predicted_pos = std::mem::take(&mut self.pcisph_pred_pos);
        predicted_pos.resize(n, [0.0; 3]);
        let mut predicted_vel = std::mem::take(&mut self.pcisph_pred_vel);
        predicted_vel.resize(n, [0.0; 3]);

        for iter in 0..max_iterations {
            let _span = trace_span!("sph::pcisph_iter", iter).entered();

            // Predict position/velocity with current pressure + non-pressure forces
            for i in 0..n {
                let pi = &self.snapshot[i];
                let rho = pi.density.max(1e-10);

                // Compute pressure acceleration from current pressures
                let mut pax = 0.0;
                let mut pay = 0.0;
                let mut paz = 0.0;

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

                    let sym = pressures[i] / (rho * rho) + pressures[j] / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let s = -pj.mass * sym * grad / r;

                    pax += s * (pi.position[0] - pj.position[0]);
                    pay += s * (pi.position[1] - pj.position[1]);
                    paz += s * (pi.position[2] - pj.position[2]);
                }

                let total_ax = non_pressure_accel[i][0] + pax;
                let total_ay = non_pressure_accel[i][1] + pay;
                let total_az = non_pressure_accel[i][2] + paz;

                predicted_vel[i] = [
                    pi.velocity[0] + total_ax * dt,
                    pi.velocity[1] + total_ay * dt,
                    pi.velocity[2] + total_az * dt,
                ];
                predicted_pos[i] = [
                    pi.position[0] + predicted_vel[i][0] * dt,
                    pi.position[1] + predicted_vel[i][1] * dt,
                    pi.position[2] + predicted_vel[i][2] * dt,
                ];
            }

            // Compute predicted density and density error
            let mut max_error = 0.0f64;
            for i in 0..n {
                let mut pred_density = 0.0;
                for &j in self.neighbors(i) {
                    let dx = predicted_pos[i][0] - predicted_pos[j][0];
                    let dy = predicted_pos[i][1] - predicted_pos[j][1];
                    let dz = predicted_pos[i][2] - predicted_pos[j][2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 <= kc.h2 {
                        pred_density += self.snapshot[j].mass * kc.poly6(r2);
                    }
                }

                let density_error = pred_density - rest_density;
                max_error = max_error.max(density_error.abs() / rest_density);

                // Update pressure: p = max(0, p + δ * density_error)
                pressures[i] = (pressures[i] + density_error * delta).max(0.0);
            }

            if max_error < max_density_error {
                break;
            }
        }

        // Apply final velocities and positions (with divergence check)
        for i in 0..n {
            let pos = predicted_pos[i];
            if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
                self.pcisph_accel = non_pressure_accel;
                self.pcisph_pressures = pressures;
                self.pcisph_pred_pos = predicted_pos;
                self.pcisph_pred_vel = predicted_vel;
                return Err(PravashError::Diverged {
                    reason: format!("NaN/Inf in predicted position at particle {i}").into(),
                });
            }
            particles[i].velocity = predicted_vel[i];
            particles[i].position = pos;
            particles[i].pressure = pressures[i];
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

        // Return persistent buffers to self
        self.pcisph_accel = non_pressure_accel;
        self.pcisph_pressures = pressures;
        self.pcisph_pred_pos = predicted_pos;
        self.pcisph_pred_vel = predicted_vel;

        Ok(())
    }

    /// Compute CFL-limited timestep: dt = cfl_factor * h / max_velocity.
    ///
    /// Returns the computed dt clamped to `[dt_min, dt_max]`.
    #[must_use]
    pub fn adaptive_dt(
        particles: &[FluidParticle],
        smoothing_radius: f64,
        cfl_factor: f64,
        dt_min: f64,
        dt_max: f64,
    ) -> f64 {
        if !smoothing_radius.is_finite()
            || smoothing_radius <= 0.0
            || !cfl_factor.is_finite()
            || cfl_factor <= 0.0
            || !dt_min.is_finite()
            || !dt_max.is_finite()
        {
            return dt_max.max(dt_min);
        }
        let max_v = max_speed(particles);
        if max_v < 1e-20 {
            return dt_max;
        }
        (cfl_factor * smoothing_radius / max_v).clamp(dt_min, dt_max)
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
/// Both use the symmetric momentum-conserving pressure formula.
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

                // Symmetric momentum-conserving pressure: -m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
                let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                let grad = kc.spiky_grad(r);
                let p_scale = -pj.mass * sym_pressure * grad / r;

                let dx = pi.position[0] - pj.position[0];
                let dy = pi.position[1] - pj.position[1];
                let dz = pi.position[2] - pj.position[2];

                ax += p_scale * dx;
                ay += p_scale * dy;
                az += p_scale * dz;

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

    // ── Parallel consistency tests ────────────────────────────────────────

    #[test]
    fn test_solver_step_deterministic() {
        // Two identical runs should produce identical results
        // (validates that parallel path doesn't introduce nondeterminism in ordering)
        let config = FluidConfig::water_2d();
        let particles_init = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);

        let run = || -> Vec<FluidParticle> {
            let mut p = particles_init.clone();
            let mut solver = SphSolver::new();
            for _ in 0..5 {
                solver.step(&mut p, &config, 0.001).unwrap();
            }
            p
        };

        let r1 = run();
        let r2 = run();
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a.position[0] - b.position[0]).abs() < 1e-10,
                "runs should be deterministic"
            );
        }
    }

    #[test]
    fn test_solver_large_particle_count() {
        // Stress test with many particles — should not crash or produce NaN
        let mut particles = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.01, 0.0001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();

        solver.step(&mut particles, &config, 0.001).unwrap();

        assert!(
            particles.iter().all(|p| p.position[0].is_finite()),
            "large particle simulation should stay finite"
        );
    }

    // ── PCISPH tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pcisph_empty() {
        let mut particles = vec![];
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        assert!(
            solver
                .step_pcisph(&mut particles, &config, 0.001, 5, 0.01)
                .is_ok()
        );
    }

    #[test]
    fn test_pcisph_single_particle_falls() {
        let mut particles = vec![FluidParticle::new_2d(0.5, 0.5, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver
            .step_pcisph(&mut particles, &config, 0.001, 5, 0.01)
            .unwrap();
        assert!(particles[0].velocity[1] < 0.0);
    }

    #[test]
    fn test_pcisph_density_bounded() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();

        for _ in 0..20 {
            solver
                .step_pcisph(&mut particles, &config, 0.001, 5, 0.01)
                .unwrap();
        }
        // All densities should be finite and positive
        for p in &particles {
            assert!(p.density.is_finite() && p.density >= 0.0);
            assert!(p.position[0].is_finite());
        }
    }

    // ── Adaptive dt tests ───────────────────────────────────────────────────

    #[test]
    fn test_adaptive_dt_stationary() {
        let particles = create_particle_block([0.1, 0.1], [0.2, 0.2], 0.05, 0.001);
        // Stationary particles → max dt
        let dt = SphSolver::adaptive_dt(&particles, 0.05, 0.4, 0.0001, 0.01);
        assert!((dt - 0.01).abs() < EPS);
    }

    #[test]
    fn test_adaptive_dt_fast_particles() {
        let mut particles = create_particle_block([0.1, 0.1], [0.2, 0.2], 0.05, 0.001);
        particles[0].velocity = [100.0, 0.0, 0.0];
        let dt = SphSolver::adaptive_dt(&particles, 0.05, 0.4, 0.0001, 0.01);
        // dt = 0.4 * 0.05 / 100 = 0.0002
        assert!(dt < 0.001);
        assert!(dt >= 0.0001);
    }
}
