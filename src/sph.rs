//! Smoothed Particle Hydrodynamics (SPH) — particle-based fluid simulation.
//!
//! SPH represents fluid as a collection of particles. Each particle carries
//! mass, position, velocity, and density. Forces (pressure, viscosity, gravity)
//! are computed from neighbor interactions using smoothing kernels.

use crate::common::{FluidConfig, FluidParticle};
use crate::error::{PravashError, Result};

use std::f64::consts::PI;

use tracing::trace_span;

// ── Precomputed Kernel Coefficients ─────────────────────────────────────────

/// Precomputed kernel coefficients for a given smoothing radius h.
/// Avoids recomputing h^6, h^9 per particle pair.
#[derive(Debug, Clone, Copy)]
struct KernelCoeffs {
    h: f64,
    h2: f64,
    poly6: f64,
    spiky_grad: f64,
    visc_lap: f64,
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
        }
    }

    /// Poly6 kernel from squared distance.
    #[inline]
    fn poly6(&self, r2: f64) -> f64 {
        if r2 > self.h2 {
            return 0.0;
        }
        let diff = self.h2 - r2;
        self.poly6 * diff * diff * diff
    }

    /// Spiky gradient magnitude (requires actual distance r).
    #[inline]
    fn spiky_grad(&self, r: f64) -> f64 {
        if r > self.h || r <= 0.0 {
            return 0.0;
        }
        let diff = self.h - r;
        self.spiky_grad * diff * diff
    }

    /// Viscosity Laplacian (requires actual distance r).
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
    if r > h || r < 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    let r2 = r * r;
    let diff = h2 - r2;
    let coeff = 315.0 / (64.0 * PI * h.powi(9));
    coeff * diff * diff * diff
}

/// Spiky kernel gradient magnitude — used for pressure forces.
///
/// ∇W(r, h) = -45 / (π·h⁶) · (h - r)²  for r ≤ h
#[inline]
#[must_use]
pub fn kernel_spiky_grad(r: f64, h: f64) -> f64 {
    if r > h || r <= 0.0 {
        return 0.0;
    }
    let diff = h - r;
    let coeff = -45.0 / (PI * h.powi(6));
    coeff * diff * diff
}

/// Viscosity kernel Laplacian — used for viscosity forces.
///
/// ∇²W(r, h) = 45 / (π·h⁶) · (h - r)  for r ≤ h
#[inline]
#[must_use]
pub fn kernel_viscosity_laplacian(r: f64, h: f64) -> f64 {
    if r > h || r <= 0.0 {
        return 0.0;
    }
    let coeff = 45.0 / (PI * h.powi(6));
    coeff * (h - r)
}

// ── Density & Pressure ──────────────────────────────────────────────────────

/// Compute density for a particle from its neighbors.
///
/// ρᵢ = Σⱼ mⱼ · W(|rᵢ - rⱼ|, h)
#[inline]
#[must_use]
pub fn compute_density(particle_idx: usize, particles: &[FluidParticle], h: f64) -> f64 {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let h2 = kc.h2;
    let mut density = 0.0;
    for pj in particles {
        let r2 = pi.distance_squared_to(pj);
        if r2 <= h2 {
            density += pj.mass * kc.poly6(r2);
        }
    }
    density
}

/// Compute pressure from density using the equation of state.
///
/// P = k · (ρ - ρ₀)
/// where k = gas constant, ρ₀ = rest density.
#[inline]
#[must_use]
pub fn equation_of_state(density: f64, rest_density: f64, gas_constant: f64) -> f64 {
    gas_constant * (density - rest_density)
}

// ── Forces ──────────────────────────────────────────────────────────────────

/// Compute pressure force on particle i from all neighbors.
///
/// fᵢᵖʳᵉˢˢ = -Σⱼ mⱼ · (Pᵢ + Pⱼ)/(2·ρⱼ) · ∇W(rᵢⱼ, h) · r̂ᵢⱼ
#[must_use]
pub fn pressure_force(particle_idx: usize, particles: &[FluidParticle], h: f64) -> [f64; 3] {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let h2 = kc.h2;
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > h2 || r2 < 1e-20 {
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

/// Compute viscosity force on particle i from all neighbors.
///
/// fᵢᵛⁱˢᶜ = μ · Σⱼ mⱼ · (vⱼ - vᵢ)/ρⱼ · ∇²W(rᵢⱼ, h)
#[must_use]
pub fn viscosity_force(
    particle_idx: usize,
    particles: &[FluidParticle],
    h: f64,
    viscosity: f64,
) -> [f64; 3] {
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let h2 = kc.h2;
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > h2 || r2 < 1e-20 {
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

// ── Simulation Step ─────────────────────────────────────────────────────────

/// Perform one SPH simulation step.
///
/// 1. Compute densities
/// 2. Compute pressures
/// 3. Compute forces (pressure + viscosity + gravity)
/// 4. Integrate (symplectic Euler)
/// 5. Enforce boundary conditions
pub fn step(particles: &mut [FluidParticle], config: &FluidConfig, viscosity: f64) -> Result<()> {
    let _span = trace_span!("sph::step", n = particles.len()).entered();
    config.validate()?;
    let h = config.smoothing_radius;
    let n = particles.len();

    if n == 0 {
        return Ok(());
    }

    let kc = KernelCoeffs::new(h);

    // 1. Compute densities using precomputed coefficients
    {
        let _span = trace_span!("sph::density", n).entered();
        let h2 = kc.h2;
        // Compute densities into a temporary buffer to avoid aliasing
        let mut densities = vec![0.0f64; n];
        for i in 0..n {
            let pi = &particles[i];
            let mut d = 0.0;
            for pj in particles.iter() {
                let r2 = pi.distance_squared_to(pj);
                if r2 <= h2 {
                    d += pj.mass * kc.poly6(r2);
                }
            }
            densities[i] = d;
        }
        for (i, p) in particles.iter_mut().enumerate() {
            p.density = densities[i];
            p.pressure = equation_of_state(p.density, config.rest_density, config.gas_constant);
        }
    }

    // 2. Compute forces
    // Snapshot only position/velocity/density/pressure/mass — needed for force computation.
    // With Copy on FluidParticle, this is a memcpy.
    let snapshot: Vec<FluidParticle> = particles.to_vec();
    {
        let _span = trace_span!("sph::forces", n).entered();
        let h2 = kc.h2;
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
                if r2 > h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let rho_j = pj.density.max(1e-10);

                // Pressure force contribution
                let pressure_term = (pi.pressure + pj.pressure) / (2.0 * rho_j);
                let grad = kc.spiky_grad(r);
                let p_scale = -pj.mass * pressure_term * grad / r;

                let dx = pi.position[0] - pj.position[0];
                let dy = pi.position[1] - pj.position[1];
                let dz = pi.position[2] - pj.position[2];

                ax += (p_scale * dx) / rho;
                ay += (p_scale * dy) / rho;
                az += (p_scale * dz) / rho;

                // Viscosity force contribution
                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;

                ax += v_scale * (pj.velocity[0] - pi.velocity[0]);
                ay += v_scale * (pj.velocity[1] - pi.velocity[1]);
                az += v_scale * (pj.velocity[2] - pi.velocity[2]);
            }

            particles[i].acceleration = [ax, ay, az];
        }
    }

    // Check for NaN/Inf divergence
    for (i, p) in particles.iter().enumerate() {
        if !p.acceleration[0].is_finite()
            || !p.acceleration[1].is_finite()
            || !p.acceleration[2].is_finite()
        {
            return Err(PravashError::Diverged {
                reason: format!("NaN/Inf in acceleration at particle {i}").into(),
            });
        }
    }

    // 3. Integrate (symplectic Euler)
    let dt = config.dt;
    for p in particles.iter_mut() {
        p.velocity[0] += p.acceleration[0] * dt;
        p.velocity[1] += p.acceleration[1] * dt;
        p.velocity[2] += p.acceleration[2] * dt;

        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;
    }

    // 4. Boundary enforcement
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

/// Total kinetic energy of the system.
#[must_use]
pub fn total_kinetic_energy(particles: &[FluidParticle]) -> f64 {
    particles.iter().map(|p| p.kinetic_energy()).sum()
}

/// Maximum particle speed (for CFL checks).
#[must_use]
pub fn max_speed(particles: &[FluidParticle]) -> f64 {
    particles.iter().map(|p| p.speed()).fold(0.0f64, f64::max)
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
        assert!(w.abs() < EPS); // W(h, h) = 0
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
        assert!(w1 > w2); // closer → stronger
    }

    #[test]
    fn test_kernel_spiky_grad() {
        let g = kernel_spiky_grad(0.5, 1.0);
        assert!(g < 0.0); // negative gradient (repulsive)
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
        assert!(d > 0.0); // self-contribution
    }

    #[test]
    fn test_create_particle_block() {
        let particles = create_particle_block([0.0, 0.0], [0.1, 0.1], 0.05, 0.01);
        assert_eq!(particles.len(), 4); // 2x2 grid
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
        // Set initial density to avoid division by zero
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        step(&mut particles, &config, 0.001).unwrap();
        // Particle should have moved downward due to gravity
        assert!(particles[0].velocity[1] < 0.0);
    }

    #[test]
    fn test_step_boundary_enforcement() {
        let mut particles = vec![FluidParticle::new_2d(0.5, -0.1, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        step(&mut particles, &config, 0.001).unwrap();
        // Should be clamped to min_y = 0.0
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

    #[test]
    fn test_kernel_coeffs_match_standalone() {
        let h = 0.5;
        let r = 0.3;
        let kc = KernelCoeffs::new(h);
        let r2 = r * r;
        assert!((kc.poly6(r2) - kernel_poly6(r, h)).abs() < 1e-6);
        assert!((kc.spiky_grad(r) - kernel_spiky_grad(r, h)).abs() < 1e-6);
        assert!((kc.visc_laplacian(r) - kernel_viscosity_laplacian(r, h)).abs() < 1e-6);
    }
}
