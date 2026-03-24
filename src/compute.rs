//! Compute backend abstraction — GPU-agnostic acceleration interface.
//!
//! Pravash exposes simulation kernels as trait methods with packed buffer
//! descriptors. External backends (wgpu, Vulkan, CUDA, distributed compute)
//! implement [`ComputeBackend`] to accelerate specific operations.
//!
//! Pravash never depends on any GPU library — it only describes **what** to
//! compute via parameter structs and flat f32 buffer layouts.

use serde::{Deserialize, Serialize};

use crate::common::ParticleSoa;
use crate::error::Result;

// ── Packed Buffers ──────────────────────────────────────────────────────────

/// Flat f32 buffer packed from SOA particle data for GPU upload.
///
/// All positions, velocities, etc. are converted from f64 to f32 and stored
/// in contiguous arrays. The backend uploads these directly to storage buffers.
#[derive(Debug, Clone)]
pub struct PackedParticles {
    pub count: u32,
    /// Interleaved [x, y, z, mass] per particle (4 floats each).
    pub position_mass: Vec<f32>,
    /// Interleaved [vx, vy, vz, padding] per particle.
    pub velocity: Vec<f32>,
    /// Density and pressure per particle: [density, pressure] interleaved.
    pub density_pressure: Vec<f32>,
}

impl PackedParticles {
    /// Pack from SOA data into GPU-friendly f32 buffers.
    #[must_use]
    pub fn from_soa(soa: &ParticleSoa) -> Self {
        let n = soa.len();
        let mut position_mass = Vec::with_capacity(n * 4);
        let mut velocity = Vec::with_capacity(n * 4);
        let mut density_pressure = Vec::with_capacity(n * 2);

        for i in 0..n {
            position_mass.push(soa.pos_x[i] as f32);
            position_mass.push(soa.pos_y[i] as f32);
            position_mass.push(soa.pos_z[i] as f32);
            position_mass.push(soa.mass[i] as f32);

            velocity.push(soa.vel_x[i] as f32);
            velocity.push(soa.vel_y[i] as f32);
            velocity.push(soa.vel_z[i] as f32);
            velocity.push(0.0); // padding for 16-byte alignment

            density_pressure.push(soa.density[i] as f32);
            density_pressure.push(soa.pressure[i] as f32);
        }

        Self {
            count: n as u32,
            position_mass,
            velocity,
            density_pressure,
        }
    }

    /// Unpack density and pressure results back into SOA.
    pub fn unpack_density_pressure(&self, soa: &mut ParticleSoa) {
        let n = self.count as usize;
        for i in 0..n.min(soa.len()) {
            soa.density[i] = f64::from(self.density_pressure[i * 2]);
            soa.pressure[i] = f64::from(self.density_pressure[i * 2 + 1]);
        }
    }

    /// Unpack velocity results back into SOA.
    pub fn unpack_velocity(&self, soa: &mut ParticleSoa) {
        let n = self.count as usize;
        for i in 0..n.min(soa.len()) {
            soa.vel_x[i] = f64::from(self.velocity[i * 4]);
            soa.vel_y[i] = f64::from(self.velocity[i * 4 + 1]);
            soa.vel_z[i] = f64::from(self.velocity[i * 4 + 2]);
        }
    }

    /// Unpack position results back into SOA.
    pub fn unpack_positions(&self, soa: &mut ParticleSoa) {
        let n = self.count as usize;
        for i in 0..n.min(soa.len()) {
            soa.pos_x[i] = f64::from(self.position_mass[i * 4]);
            soa.pos_y[i] = f64::from(self.position_mass[i * 4 + 1]);
            soa.pos_z[i] = f64::from(self.position_mass[i * 4 + 2]);
        }
    }
}

// ── Kernel Parameters ──────────────────────────────────────────────────────

/// Parameters for SPH density/force computation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SphKernelParams {
    pub smoothing_radius: f32,
    pub rest_density: f32,
    pub gas_constant: f32,
    pub viscosity: f32,
    pub gravity: [f32; 3],
    pub dt: f32,
    pub particle_count: u32,
}

/// Parameters for grid-based Navier-Stokes step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridKernelParams {
    pub nx: u32,
    pub ny: u32,
    pub dx: f32,
    pub dt: f32,
    pub viscosity: f32,
    pub buoyancy_alpha: f32,
    pub ambient_density: f32,
}

/// Parameters for shallow water step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShallowKernelParams {
    pub nx: u32,
    pub ny: u32,
    pub dx: f32,
    pub dt: f32,
    pub gravity: f32,
    pub damping: f32,
    pub dry_threshold: f32,
}

// ── Compute Backend Trait ──────────────────────────────────────────────────

/// Describes which operation a backend should perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComputeOp {
    /// Compute SPH particle densities from neighbor interactions.
    SphDensity,
    /// Compute SPH pressure + viscosity forces and integrate.
    SphStep,
    /// Perform one grid Navier-Stokes advection step.
    GridAdvect,
    /// Perform one shallow water step.
    ShallowStep,
}

/// Backend-agnostic compute acceleration interface.
///
/// Implementors provide GPU or distributed acceleration for simulation
/// kernels. Pravash calls these methods with packed buffers and parameters;
/// the backend executes the computation and writes results back.
///
/// # Example (pseudocode)
///
/// ```ignore
/// struct WgpuBackend { device: wgpu::Device, /* ... */ }
///
/// impl ComputeBackend for WgpuBackend {
///     fn supports(&self, op: ComputeOp) -> bool { /* check compiled shaders */ }
///     fn sph_density(&self, particles: &mut PackedParticles, params: &SphKernelParams) -> Result<()> {
///         // upload particles.position_mass to GPU storage buffer
///         // dispatch density compute shader
///         // readback density_pressure
///         Ok(())
///     }
///     // ...
/// }
/// ```
pub trait ComputeBackend {
    /// Whether this backend supports a given operation.
    fn supports(&self, op: ComputeOp) -> bool;

    /// Compute SPH densities. Updates `particles.density_pressure`.
    fn sph_density(&self, particles: &mut PackedParticles, params: &SphKernelParams) -> Result<()>;

    /// Full SPH step (density + forces + integrate). Updates all particle buffers.
    fn sph_step(&self, particles: &mut PackedParticles, params: &SphKernelParams) -> Result<()>;

    /// Grid advection. Reads `src`, writes `dst`.
    fn grid_advect(
        &self,
        dst: &mut [f32],
        src: &[f32],
        vx: &[f32],
        vy: &[f32],
        params: &GridKernelParams,
    ) -> Result<()>;

    /// Shallow water step. Updates height and velocity buffers in place.
    fn shallow_step(
        &self,
        height: &mut [f32],
        vx: &mut [f32],
        vy: &mut [f32],
        ground: &[f32],
        params: &ShallowKernelParams,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{FluidParticle, ParticleSoa};

    #[test]
    fn test_packed_particles_roundtrip() {
        let particles = vec![
            FluidParticle::new([1.0, 2.0, 3.0], 0.5),
            FluidParticle::new([4.0, 5.0, 6.0], 0.25),
        ];
        let soa = ParticleSoa::from_aos(&particles);
        let packed = PackedParticles::from_soa(&soa);

        assert_eq!(packed.count, 2);
        // Check first particle position
        assert!((packed.position_mass[0] - 1.0).abs() < 1e-6);
        assert!((packed.position_mass[1] - 2.0).abs() < 1e-6);
        assert!((packed.position_mass[2] - 3.0).abs() < 1e-6);
        assert!((packed.position_mass[3] - 0.5).abs() < 1e-6); // mass
        // Check second particle
        assert!((packed.position_mass[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_packed_unpack_density() {
        let particles = vec![FluidParticle::new([0.0; 3], 1.0)];
        let mut soa = ParticleSoa::from_aos(&particles);
        let mut packed = PackedParticles::from_soa(&soa);

        // Simulate GPU writing density/pressure
        packed.density_pressure[0] = 1000.0;
        packed.density_pressure[1] = 500.0;
        packed.unpack_density_pressure(&mut soa);

        assert!((soa.density[0] - 1000.0).abs() < 1e-3);
        assert!((soa.pressure[0] - 500.0).abs() < 1e-3);
    }

    #[test]
    fn test_packed_unpack_velocity() {
        let particles = vec![FluidParticle::new([0.0; 3], 1.0)];
        let mut soa = ParticleSoa::from_aos(&particles);
        let mut packed = PackedParticles::from_soa(&soa);

        packed.velocity[0] = 1.5;
        packed.velocity[1] = 2.5;
        packed.velocity[2] = 3.5;
        packed.unpack_velocity(&mut soa);

        assert!((soa.vel_x[0] - 1.5).abs() < 1e-3);
        assert!((soa.vel_y[0] - 2.5).abs() < 1e-3);
        assert!((soa.vel_z[0] - 3.5).abs() < 1e-3);
    }

    #[test]
    fn test_packed_empty() {
        let soa = ParticleSoa::new();
        let packed = PackedParticles::from_soa(&soa);
        assert_eq!(packed.count, 0);
        assert!(packed.position_mass.is_empty());
    }

    #[test]
    fn test_sph_kernel_params() {
        let params = SphKernelParams {
            smoothing_radius: 0.05,
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 0.001,
            gravity: [0.0, -9.81, 0.0],
            dt: 0.001,
            particle_count: 100,
        };
        assert!(params.dt > 0.0);
        assert_eq!(params.particle_count, 100);
    }

    #[test]
    fn test_compute_op_variants() {
        // Ensure all variants are distinct
        assert_ne!(ComputeOp::SphDensity, ComputeOp::SphStep);
        assert_ne!(ComputeOp::GridAdvect, ComputeOp::ShallowStep);
    }
}
