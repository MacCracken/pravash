//! Smoothed Particle Hydrodynamics (SPH) — particle-based fluid simulation.
//!
//! SPH represents fluid as a collection of particles. Each particle carries
//! mass, position, velocity, and density. Forces (pressure, viscosity, gravity)
//! are computed from neighbor interactions using smoothing kernels.

use crate::common::{FluidConfig, FluidParticle};
use crate::error::{PravashError, Result};

use std::f64::consts::PI;

// ── SPH Kernels ─────────────────────────────────────────────────────────────

/// Poly6 smoothing kernel — used for density estimation.
///
/// W(r, h) = 315 / (64·π·h⁹) · (h² - r²)³  for r ≤ h
#[inline]
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
pub fn compute_density(particle_idx: usize, particles: &[FluidParticle], h: f64) -> f64 {
    let pi = &particles[particle_idx];
    let mut density = 0.0;
    for pj in particles {
        let r = pi.distance_to(pj);
        density += pj.mass * kernel_poly6(r, h);
    }
    density
}

/// Compute pressure from density using the equation of state.
///
/// P = k · (ρ - ρ₀)
/// where k = gas constant, ρ₀ = rest density.
#[inline]
pub fn equation_of_state(density: f64, rest_density: f64, gas_constant: f64) -> f64 {
    gas_constant * (density - rest_density)
}

// ── Forces ──────────────────────────────────────────────────────────────────

/// Compute pressure force on particle i from all neighbors.
///
/// fᵢᵖʳᵉˢˢ = -Σⱼ mⱼ · (Pᵢ + Pⱼ)/(2·ρⱼ) · ∇W(rᵢⱼ, h) · r̂ᵢⱼ
pub fn pressure_force(
    particle_idx: usize,
    particles: &[FluidParticle],
    h: f64,
) -> [f64; 3] {
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r = pi.distance_to(pj);
        if r > h || r < 1e-10 {
            continue;
        }

        let pressure_term = (pi.pressure + pj.pressure) / (2.0 * pj.density.max(1e-10));
        let grad = kernel_spiky_grad(r, h);
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
pub fn viscosity_force(
    particle_idx: usize,
    particles: &[FluidParticle],
    h: f64,
    viscosity: f64,
) -> [f64; 3] {
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r = pi.distance_to(pj);
        if r > h || r < 1e-10 {
            continue;
        }

        let lap = kernel_viscosity_laplacian(r, h);
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
    config.validate()?;
    let h = config.smoothing_radius;
    let n = particles.len();

    if n == 0 {
        return Ok(());
    }

    // 1. Compute densities
    let densities: Vec<f64> = (0..n)
        .map(|i| compute_density(i, particles, h))
        .collect();

    // 2. Compute pressures
    for (i, p) in particles.iter_mut().enumerate() {
        p.density = densities[i];
        p.pressure = equation_of_state(p.density, config.rest_density, config.gas_constant);
    }

    // 3. Compute forces
    // Need immutable snapshot for force computation
    let snapshot: Vec<FluidParticle> = particles.to_vec();
    for i in 0..n {
        let fp = pressure_force(i, &snapshot, h);
        let fv = viscosity_force(i, &snapshot, h, viscosity);
        let rho = snapshot[i].density.max(1e-10);

        particles[i].acceleration[0] = (fp[0] + fv[0]) / rho + config.gravity[0];
        particles[i].acceleration[1] = (fp[1] + fv[1]) / rho + config.gravity[1];
        particles[i].acceleration[2] = (fp[2] + fv[2]) / rho + config.gravity[2];
    }

    // 4. Integrate (symplectic Euler)
    let dt = config.dt;
    for p in particles.iter_mut() {
        p.velocity[0] += p.acceleration[0] * dt;
        p.velocity[1] += p.acceleration[1] * dt;
        p.velocity[2] += p.acceleration[2] * dt;

        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;
    }

    // 5. Boundary enforcement
    let [min_x, min_y, min_z, max_x, max_y, max_z] = config.bounds;
    let damp = config.boundary_damping;
    for p in particles.iter_mut() {
        for (pos, vel, lo, hi) in [
            (&mut p.position[0], &mut p.velocity[0], min_x, max_x),
            (&mut p.position[1], &mut p.velocity[1], min_y, max_y),
            (&mut p.position[2], &mut p.velocity[2], min_z, max_z),
        ] {
            if *pos < lo {
                *pos = lo;
                *vel *= -damp;
            } else if *pos > hi && hi > lo {
                *pos = hi;
                *vel *= -damp;
            }
        }
    }

    Ok(())
}

/// Total kinetic energy of the system.
pub fn total_kinetic_energy(particles: &[FluidParticle]) -> f64 {
    particles.iter().map(|p| p.kinetic_energy()).sum()
}

/// Maximum particle speed (for CFL checks).
pub fn max_speed(particles: &[FluidParticle]) -> f64 {
    particles
        .iter()
        .map(|p| p.speed())
        .fold(0.0f64, f64::max)
}

/// Create a block of particles in a grid pattern.
pub fn create_particle_block(
    origin: [f64; 2],
    size: [f64; 2],
    spacing: f64,
    particle_mass: f64,
) -> Vec<FluidParticle> {
    let mut particles = Vec::new();
    let nx = (size[0] / spacing).ceil() as usize;
    let ny = (size[1] / spacing).ceil() as usize;
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
}
