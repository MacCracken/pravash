//! Compute backend abstraction — GPU-agnostic acceleration interface.
//!
//! Pravash exposes simulation kernels as trait methods with packed buffer
//! descriptors. External backends (wgpu, Vulkan, CUDA, distributed compute)
//! implement [`ComputeBackend`] to accelerate specific operations.
//!
//! Pravash never depends on any GPU library — it only describes **what** to
//! compute via parameter structs and flat f32 buffer layouts.
//!
//! # No vendor lock-in
//!
//! ```ignore
//! // In soorat (or any GPU backend):
//! struct WgpuBackend { device: wgpu::Device, /* ... */ }
//! impl pravash::compute::ComputeBackend for WgpuBackend {
//!     fn supports(&self, op: ComputeOp) -> bool { true }
//!     fn sph_density(&self, p: &mut PackedParticles, params: &SphKernelParams) -> Result<()> {
//!         // upload, dispatch compute shader, readback
//!         Ok(())
//!     }
//!     // ...
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::common::ParticleSoa;
use crate::error::Result;

// ── Packed Buffers ──────────────────────────────────────────────────────────

/// Flat f32 buffer packed from SOA particle data for GPU upload.
///
/// All positions, velocities, etc. are converted from f64 to f32 and stored
/// in contiguous arrays. The backend uploads these directly to storage buffers.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
        debug_assert_eq!(soa.pos_x.len(), n, "pos_x length mismatch");
        debug_assert_eq!(soa.pos_y.len(), n, "pos_y length mismatch");
        debug_assert_eq!(soa.pos_z.len(), n, "pos_z length mismatch");
        debug_assert_eq!(soa.vel_x.len(), n, "vel_x length mismatch");
        debug_assert_eq!(soa.vel_y.len(), n, "vel_y length mismatch");
        debug_assert_eq!(soa.vel_z.len(), n, "vel_z length mismatch");
        debug_assert_eq!(soa.mass.len(), n, "mass length mismatch");
        debug_assert_eq!(soa.density.len(), n, "density length mismatch");
        debug_assert_eq!(soa.pressure.len(), n, "pressure length mismatch");
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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

// ── Neural Operator Acceleration ────────────────────────────────────────────

/// Trait for neural network-based simulation correction.
///
/// Learned correctors take a coarse simulation state and return a correction
/// that approximates the fine-resolution result. Enables 4-16x speedup by
/// running a coarser sim and applying a learned fix.
///
/// # No vendor lock-in
///
/// Pravash defines this trait. ML frameworks (PyTorch, ONNX, candle, etc.)
/// implement it. The consumer wires them together.
///
/// ```ignore
/// struct FnoCorrector { model: onnxruntime::Session }
/// impl NeuralCorrector for FnoCorrector {
///     fn correct(&self, state: &SimState) -> Result<SimCorrection> {
///         // run inference, return velocity/pressure corrections
///     }
/// }
/// ```
pub trait NeuralCorrector {
    /// Apply a learned correction to the simulation state.
    ///
    /// `velocities`: flat f32 buffer of velocity components.
    /// `pressures`: flat f32 buffer of pressure values.
    /// Returns corrections to be added to the current state.
    fn correct(
        &self,
        velocities: &[f32],
        pressures: &[f32],
        params: &GridKernelParams,
    ) -> Result<NeuralCorrection>;
}

/// Correction output from a neural operator.
#[derive(Debug, Clone)]
pub struct NeuralCorrection {
    /// Velocity correction (same layout as input).
    pub velocity_delta: Vec<f32>,
    /// Pressure correction.
    pub pressure_delta: Vec<f32>,
}

// ── Differentiable Simulation ───────────────────────────────────────────────

/// Analytical derivatives of SPH kernels for differentiable simulation.
///
/// Enables gradient-based optimization of simulation parameters,
/// inverse design, and integration with autodiff frameworks.
pub struct KernelDerivatives;

impl KernelDerivatives {
    /// d(poly6)/d(r²) — derivative of poly6 kernel w.r.t. squared distance.
    ///
    /// W = c · (h² - r²)³ → dW/d(r²) = -3c · (h² - r²)²
    #[inline]
    #[must_use]
    pub fn dpoly6_dr2(r2: f64, h: f64) -> f64 {
        let h2 = h * h;
        if r2 > h2 {
            return 0.0;
        }
        let h3 = h2 * h;
        let h9 = h3 * h3 * h3;
        let c = 315.0 / (64.0 * std::f64::consts::PI * h9);
        let diff = h2 - r2;
        -3.0 * c * diff * diff
    }

    /// d(EOS)/dρ — derivative of linear equation of state w.r.t. density.
    ///
    /// P = k(ρ - ρ₀) → dP/dρ = k
    #[inline]
    #[must_use]
    pub fn deos_drho(gas_constant: f64) -> f64 {
        gas_constant
    }

    /// d(Tait EOS)/dρ — derivative of Tait equation of state.
    ///
    /// P = B((ρ/ρ₀)^γ - 1) → dP/dρ = B·γ·(ρ/ρ₀)^(γ-1) / ρ₀
    #[inline]
    #[must_use]
    pub fn dtait_drho(density: f64, rest_density: f64, speed_of_sound: f64, gamma: f64) -> f64 {
        let rho0 = rest_density.max(1e-10);
        let b = rho0 * speed_of_sound * speed_of_sound / gamma;
        b * gamma * (density / rho0).powf(gamma - 1.0) / rho0
    }

    /// d(Wendland C2)/dr — analytical derivative for gradient computation.
    #[inline]
    #[must_use]
    pub fn dwendland_c2_dr(r: f64, h: f64) -> f64 {
        crate::sph::kernel_wendland_c2_grad(r, h)
    }

    /// Compute gradient of an arbitrary scalar function using hisab's
    /// reverse-mode automatic differentiation.
    ///
    /// `f` takes a tape and input variables, returns a scalar output.
    /// Returns the gradient ∂f/∂x for each input dimension.
    ///
    /// ```ignore
    /// let grad = KernelDerivatives::autodiff_gradient(
    ///     |tape, vars| {
    ///         let r2 = tape.mul(vars[0], vars[0]);
    ///         // ... kernel computation ...
    ///         result
    ///     },
    ///     &[0.5], // input: r = 0.5
    /// );
    /// ```
    ///
    /// Compute gradient using hisab's reverse-mode automatic differentiation.
    ///
    /// Wraps `hisab::autodiff::reverse_gradient` for convenience.
    #[must_use]
    pub fn autodiff_gradient(
        f: impl Fn(&mut hisab::autodiff::Tape, &[hisab::autodiff::Var]) -> hisab::autodiff::Var,
        x: &[f64],
    ) -> Vec<f64> {
        hisab::autodiff::reverse_gradient(f, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{FluidParticle, ParticleSoa};
    use hisab::DVec3;

    #[test]
    fn test_packed_particles_roundtrip() {
        let particles = vec![
            FluidParticle::new(DVec3::new(1.0, 2.0, 3.0), 0.5),
            FluidParticle::new(DVec3::new(4.0, 5.0, 6.0), 0.25),
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
        let particles = vec![FluidParticle::new(DVec3::ZERO, 1.0)];
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
        let particles = vec![FluidParticle::new(DVec3::ZERO, 1.0)];
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
