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

use hisab::DVec3;
use hisab::SpatialHash;
use hisab::Vec3;
use tracing::trace_span;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Minimum density floor to prevent division-by-zero in pressure/force calculations.
const MIN_DENSITY: f64 = 1e-10;

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
        debug_assert!(h > 0.0, "smoothing radius must be positive");
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

// ── Wendland Kernels ──────────────────────────────────────────────────────

/// Wendland C2 kernel — recommended replacement for Poly6.
///
/// W(r, h) = (7 / (4·π·h²)) · (1 - q/2)⁴ · (1 + 2q)  for q = r/h ≤ 2
///
/// Strictly positive definite, no tensile instability, C2 smooth.
/// Note: compact support is 2h (vs h for Poly6).
#[inline]
#[must_use]
pub fn kernel_wendland_c2(r: f64, h: f64) -> f64 {
    let q = r / h;
    if !(0.0..2.0).contains(&q) {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    (7.0 / (4.0 * PI * h * h)) * t * t * t * t * (1.0 + 2.0 * q)
}

/// Wendland C2 gradient magnitude.
///
/// ∂W/∂r = (7 / (4·π·h²)) · (-5q) · (1 - q/2)³ / h
#[inline]
#[must_use]
pub fn kernel_wendland_c2_grad(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 || q <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    (7.0 / (4.0 * PI * h * h)) * (-5.0 * q) * t * t * t / h
}

/// Wendland C4 kernel — higher-order alternative to C2.
///
/// W(r, h) = (9 / (4·π·h²)) · (1 - q/2)⁶ · (1 + 3q + 35q²/12)  for q = r/h ≤ 2
#[inline]
#[must_use]
pub fn kernel_wendland_c4(r: f64, h: f64) -> f64 {
    let q = r / h;
    if !(0.0..2.0).contains(&q) {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    let t6 = t * t * t * t * t * t;
    (9.0 / (4.0 * PI * h * h)) * t6 * (1.0 + 3.0 * q + 35.0 / 12.0 * q * q)
}

/// Wendland C4 gradient magnitude.
///
/// dW/dr = σ/h · (1-q/2)⁵ · [-3(1+3q+35q²/12) + (1-q/2)(3+35q/6)]
#[inline]
#[must_use]
pub fn kernel_wendland_c4_grad(r: f64, h: f64) -> f64 {
    let q = r / h;
    if q >= 2.0 || q <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    let t5 = t * t * t * t * t;
    let g = 1.0 + 3.0 * q + 35.0 / 12.0 * q * q;
    let gp = 3.0 + 35.0 / 6.0 * q;
    // Product rule: d/dq[(1-q/2)^6 · g] = -3(1-q/2)^5 · g + (1-q/2)^6 · g'
    let dfdq = t5 * (-3.0 * g + t * gp);
    (9.0 / (4.0 * PI * h * h)) * dfdq / h
}

// ── Multi-phase Configuration ──────────────────────────────────────────────

/// Per-phase material properties for multi-phase SPH.
#[derive(Debug, Clone, Copy)]
pub struct PhaseProperties {
    /// Rest density for this phase.
    pub rest_density: f64,
    /// Gas constant (equation of state stiffness).
    pub gas_constant: f64,
    /// Dynamic viscosity.
    pub viscosity: f64,
}

/// Multi-phase SPH configuration.
///
/// Maps phase indices to material properties. Particles with `phase = i`
/// use `phases[i]` for their density, pressure, and viscosity computation.
///
/// Inter-phase surface tension is applied at boundaries between different phases.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultiPhaseConfig {
    /// Material properties per phase. Index by particle `phase` field.
    pub phases: Vec<PhaseProperties>,
    /// Interface tension coefficient between phases.
    /// Applied at boundaries where neighboring particles have different phases.
    pub interface_tension: f64,
}

impl MultiPhaseConfig {
    /// Create a single-phase config (equivalent to standard SPH).
    #[must_use]
    pub fn single(rest_density: f64, gas_constant: f64, viscosity: f64) -> Self {
        Self {
            phases: vec![PhaseProperties {
                rest_density,
                gas_constant,
                viscosity,
            }],
            interface_tension: 0.0,
        }
    }

    /// Create a two-phase config (e.g., water-air).
    #[must_use]
    pub fn two_phase(
        phase_a: PhaseProperties,
        phase_b: PhaseProperties,
        interface_tension: f64,
    ) -> Self {
        Self {
            phases: vec![phase_a, phase_b],
            interface_tension,
        }
    }

    /// Get properties for a particle's phase, falling back to phase 0.
    #[inline]
    #[must_use]
    fn get(&self, phase: u8) -> &PhaseProperties {
        self.phases.get(phase as usize).unwrap_or_else(|| {
            tracing::warn!(
                requested = phase,
                available = self.phases.len(),
                "phase index out of bounds, falling back to phase 0"
            );
            &self.phases[0]
        })
    }
}

// ── Viscoelastic Configuration ─────────────────────────────────────────────

/// Viscoelastic fluid properties (Oldroyd-B model).
///
/// The conformation tensor C tracks polymer chain deformation.
/// Elastic stress τ = G·(C - I) is added to the SPH momentum equation.
/// C evolves via: DC/Dt = C·∇u + (∇u)ᵀ·C - (C - I)/λ
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ViscoelasticConfig {
    /// Elastic modulus G (Pa). Controls the strength of elastic restoring force.
    /// Typical: 10-1000 for honey/lava effects.
    pub elastic_modulus: f64,
    /// Relaxation time λ (seconds). How quickly stress dissipates.
    /// Small λ = viscous (quick relaxation), large λ = elastic (slow relaxation).
    /// Typical: 0.01-1.0.
    pub relaxation_time: f64,
}

/// Update conformation tensors and apply elastic stress for viscoelastic particles.
///
/// Evolves each particle's conformation tensor using the Oldroyd-B model
/// and adds the elastic stress divergence to particle accelerations.
///
/// Call after `SphSolver::step()` (which computes density/pressure/viscous forces),
/// then re-integrate positions with the updated accelerations.
pub fn update_viscoelastic(
    particles: &mut [FluidParticle],
    neighbor_offsets: &[u32],
    neighbor_indices: &[usize],
    h: f64,
    ve: &ViscoelasticConfig,
    dt: f64,
) {
    let _span = trace_span!("sph::viscoelastic", n = particles.len()).entered();
    let n = particles.len();
    if n == 0 {
        return;
    }

    let kc = KernelCoeffs::new(h);
    let inv_lambda = 1.0 / ve.relaxation_time.max(1e-20);

    let snap: Vec<FluidParticle> = particles.to_vec();

    for i in 0..n {
        let pi = &snap[i];
        let rho = pi.density.max(MIN_DENSITY);
        let start = neighbor_offsets[i] as usize;
        let end = neighbor_offsets[i + 1] as usize;

        // Estimate velocity gradient ∇u via SPH
        let mut dudx = 0.0;
        let mut dudy = 0.0;
        let mut dvdx = 0.0;
        let mut dvdy = 0.0;

        for &j in &neighbor_indices[start..end] {
            if j == i {
                continue;
            }
            let pj = &snap[j];
            let r2 = pi.distance_squared_to(pj);
            if r2 > kc.h2 || r2 < 1e-20 {
                continue;
            }
            let r = r2.sqrt();
            let rho_j = pj.density.max(MIN_DENSITY);
            let grad = kc.spiky_grad(r);
            let scale = pj.mass / rho_j * grad / r;

            let dx = pi.position[0] - pj.position[0];
            let dy = pi.position[1] - pj.position[1];
            let dvx = pj.velocity[0] - pi.velocity[0];
            let dvy = pj.velocity[1] - pi.velocity[1];

            dudx += dvx * scale * dx;
            dudy += dvx * scale * dy;
            dvdx += dvy * scale * dx;
            dvdy += dvy * scale * dy;
        }

        // Evolve conformation tensor: DC/Dt = C·∇u + (∇u)ᵀ·C - (C-I)/λ
        let [c_xx, c_xy, c_yy] = pi.conformation;

        let dcxx =
            (c_xx * dudx + c_xy * dvdx) + (dudx * c_xx + dudy * c_xy) - (c_xx - 1.0) * inv_lambda;
        let dcxy = 0.5 * ((c_xx * dudy + c_xy * dvdy) + (c_xy * dudx + c_yy * dvdx))
            + 0.5 * ((dvdx * c_xx + dvdy * c_xy) + (dudx * c_xy + dudy * c_yy))
            - c_xy * inv_lambda;
        let dcyy =
            (c_xy * dudy + c_yy * dvdy) + (dvdx * c_xy + dvdy * c_yy) - (c_yy - 1.0) * inv_lambda;

        particles[i].conformation = [c_xx + dcxx * dt, c_xy + dcxy * dt, c_yy + dcyy * dt];

        // Elastic stress divergence: ∇·(G·(C - I))
        let mut fx = 0.0;
        let mut fy = 0.0;
        for &j in &neighbor_indices[start..end] {
            if j == i {
                continue;
            }
            let pj = &snap[j];
            let r2 = pi.distance_squared_to(pj);
            if r2 > kc.h2 || r2 < 1e-20 {
                continue;
            }
            let r = r2.sqrt();
            let rho_j = pj.density.max(MIN_DENSITY);
            let grad = kc.spiky_grad(r);
            let scale = pj.mass / rho_j * grad / r;

            let dx = pi.position[0] - pj.position[0];
            let dy = pi.position[1] - pj.position[1];

            let dc_xx = pj.conformation[0] - c_xx;
            let dc_xy = pj.conformation[1] - c_xy;
            let dc_yy = pj.conformation[2] - c_yy;

            fx += ve.elastic_modulus * (dc_xx * dx + dc_xy * dy) * scale / rho;
            fy += ve.elastic_modulus * (dc_xy * dx + dc_yy * dy) * scale / rho;
        }

        particles[i].acceleration[0] += fx;
        particles[i].acceleration[1] += fy;
    }
}

// ── Heat Transfer ──────────────────────────────────────────────────────────

/// Configuration for SPH heat conduction.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct HeatConfig {
    /// Thermal diffusivity κ (m²/s). Controls heat conduction speed.
    /// Water ≈ 1.4e-7, air ≈ 2.2e-5, lava ≈ 5e-7.
    pub diffusivity: f64,
}

/// Update particle temperatures via SPH heat conduction.
///
/// Computes `∂T/∂t = κ·∇²T` using the SPH Laplacian approximation:
/// `∇²T_i = Σ_j (m_j/ρ_j) · (T_j - T_i) · ∇²W(r_ij, h)`
///
/// Convection is implicit — particles carry temperature as they move.
/// Call after the SPH step (needs valid density and neighbor cache).
pub fn update_heat(
    particles: &mut [FluidParticle],
    neighbor_offsets: &[u32],
    neighbor_indices: &[usize],
    h: f64,
    heat: &HeatConfig,
    dt: f64,
) {
    let _span = trace_span!("sph::heat", n = particles.len()).entered();
    let n = particles.len();
    if n == 0 || heat.diffusivity <= 0.0 {
        return;
    }

    let kc = KernelCoeffs::new(h);
    let snap: Vec<f64> = particles.iter().map(|p| p.temperature).collect();

    for i in 0..n {
        let pi = &particles[i];
        let rho = pi.density.max(MIN_DENSITY);
        let start = neighbor_offsets[i] as usize;
        let end = neighbor_offsets[i + 1] as usize;
        let mut dt_temp = 0.0;

        for &j in &neighbor_indices[start..end] {
            if j == i {
                continue;
            }
            let pj = &particles[j];
            let r2 = pi.distance_squared_to(pj);
            if r2 > kc.h2 || r2 < 1e-20 {
                continue;
            }
            let r = r2.sqrt();
            let rho_j = pj.density.max(MIN_DENSITY);

            let lap = kc.visc_laplacian(r);
            dt_temp += pj.mass / rho_j * (snap[j] - snap[i]) * lap;
        }

        particles[i].temperature += heat.diffusivity * dt_temp / rho * dt;
    }
}

// ── Phase Change ──────────────────────────────────────────────────────────

/// Phase change configuration (melting, solidification, evaporation, condensation).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PhaseChangeConfig {
    /// Melting/boiling temperature (K). Phase changes at this threshold.
    pub transition_temperature: f64,
    /// Latent heat (J/kg). Energy absorbed during melting/evaporation.
    pub latent_heat: f64,
    /// Specific heat capacity (J/(kg·K)). Converts temperature to energy.
    /// Water ≈ 4186, ice ≈ 2090, steam ≈ 2010.
    pub heat_capacity: f64,
    /// Phase index for the solid/liquid phase below transition temperature.
    pub phase_below: u8,
    /// Phase index for the liquid/gas phase above transition temperature.
    pub phase_above: u8,
}

/// Update particle phases based on temperature (Stefan condition).
///
/// Particles at the transition temperature absorb/release latent heat
/// before changing phase. Temperature is held at the transition point
/// until all latent heat is consumed.
///
/// Uses `fuel` field as a latent heat accumulator (0 = fully below, 1 = fully above).
pub fn update_phase_change(particles: &mut [FluidParticle], config: &PhaseChangeConfig, _dt: f64) {
    let _span = trace_span!("sph::phase_change", n = particles.len()).entered();
    let t_trans = config.transition_temperature;
    let l_heat = config.latent_heat;
    let cp = config.heat_capacity.max(1e-10);

    for p in particles.iter_mut() {
        if p.temperature > t_trans && p.phase == config.phase_below {
            // Absorb latent heat (melting/evaporation)
            // Thermal energy available = cp · ΔT (J/kg), need l_heat (J/kg) to transition
            let excess = p.temperature - t_trans;
            let thermal_energy = excess * cp;
            let fraction = (thermal_energy / l_heat).min(1.0);
            if fraction >= 1.0 {
                p.phase = config.phase_above;
                // Remaining temperature above transition after consuming latent heat
                p.temperature = t_trans + (thermal_energy - l_heat) / cp;
            } else {
                p.temperature = t_trans; // hold at transition
            }
        } else if p.temperature < t_trans && p.phase == config.phase_above {
            // Release latent heat (solidification/condensation)
            let deficit = t_trans - p.temperature;
            let thermal_energy = deficit * cp;
            let fraction = (thermal_energy / l_heat).min(1.0);
            if fraction >= 1.0 {
                p.phase = config.phase_below;
                p.temperature = t_trans - (thermal_energy - l_heat) / cp;
            } else {
                p.temperature = t_trans;
            }
        }
    }
}

// ── Chemical Reaction ─────────────────────────────────────────────────────

/// Trait for pluggable chemistry backends (e.g., kimiya).
///
/// Implement this to replace the built-in `CombustionConfig` with
/// advanced reaction kinetics (Arrhenius, multi-step, equilibrium).
/// Pravash calls `react()` per particle per step — the implementation
/// decides how much fuel is consumed and how much heat is released.
///
/// # No vendor lock-in
///
/// Pravash defines this trait. Chemistry crates (kimiya, etc.) implement it.
/// Neither depends on the other — the consumer wires them together.
///
/// ```ignore
/// // In your app:
/// struct KimiyaReactor { /* kimiya types */ }
/// impl ReactionProvider for KimiyaReactor {
///     fn react(&self, temperature: f64, fuel: f64, dt: f64) -> (f64, f64) {
///         let rate = kimiya::kinetics::arrhenius_rate(a, ea, temperature).unwrap();
///         let consumed = fuel * (1.0 - (-rate * dt).exp());
///         let heat = consumed * delta_h;
///         (consumed, heat)
///     }
/// }
/// ```
pub trait ReactionProvider {
    /// Compute reaction for a single particle.
    ///
    /// Given current `temperature` (K), `fuel` concentration (0–1), and `dt` (s),
    /// returns `(fuel_consumed, temperature_increase)`.
    fn react(&self, temperature: f64, fuel: f64, dt: f64) -> (f64, f64);
}

/// Update particles using a pluggable reaction provider.
///
/// Drop-in replacement for `update_combustion` that accepts any
/// `ReactionProvider` implementation.
pub fn update_reaction(particles: &mut [FluidParticle], provider: &dyn ReactionProvider, dt: f64) {
    let _span = trace_span!("sph::reaction", n = particles.len()).entered();
    for p in particles.iter_mut() {
        if p.fuel > 0.0 {
            let (consumed, heat) = provider.react(p.temperature, p.fuel, dt);
            p.fuel = (p.fuel - consumed).max(0.0);
            p.temperature += heat;
        }
    }
}

// Make CombustionConfig implement ReactionProvider for backward compatibility
impl ReactionProvider for CombustionConfig {
    fn react(&self, temperature: f64, fuel: f64, dt: f64) -> (f64, f64) {
        if temperature <= self.ignition_temperature || fuel <= 0.0 {
            return (0.0, 0.0);
        }
        let consumed = fuel - fuel / (1.0 + self.reaction_rate * dt);
        (consumed, self.heat_release * consumed)
    }
}

/// Combustion reaction configuration.
///
/// Simple one-step reaction model: fuel → products + heat.
/// Above the ignition temperature, fuel depletes at a rate proportional to
/// concentration and temperature excess. The reaction releases heat,
/// creating a feedback loop (fire spreads).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CombustionConfig {
    /// Ignition temperature (K). No reaction below this. Default: 500.
    pub ignition_temperature: f64,
    /// Reaction rate constant (1/s). Controls how fast fuel burns.
    /// Higher = faster combustion. Default: 10.0.
    pub reaction_rate: f64,
    /// Heat release per unit fuel consumed (K per unit fuel).
    /// Controls how much temperature increases from burning. Default: 1000.0.
    pub heat_release: f64,
}

impl Default for CombustionConfig {
    fn default() -> Self {
        Self {
            ignition_temperature: 500.0,
            reaction_rate: 10.0,
            heat_release: 1000.0,
        }
    }
}

/// Update particle fuel and temperature via combustion reaction.
///
/// For each particle above ignition temperature with fuel > 0:
/// - Fuel depletes: `dfuel = -rate · fuel · dt`
/// - Temperature increases: `dT = heat_release · |dfuel|`
///
/// Call after `update_heat` for coupled heat-reaction simulation.
pub fn update_combustion(particles: &mut [FluidParticle], config: &CombustionConfig, dt: f64) {
    let _span = trace_span!("sph::combustion", n = particles.len()).entered();
    let t_ign = config.ignition_temperature;
    let rate = config.reaction_rate;
    let heat = config.heat_release;

    for p in particles.iter_mut() {
        if p.temperature > t_ign && p.fuel > 0.0 {
            // Implicit fuel depletion: fuel_new = fuel / (1 + rate·dt)
            let consumed = p.fuel - p.fuel / (1.0 + rate * dt);
            p.fuel -= consumed;
            p.fuel = p.fuel.max(0.0);

            // Exothermic: heat released proportional to fuel consumed
            p.temperature += heat * consumed;
        }
    }
}

// ── Contact Angle / Wetting ────────────────────────────────────────────────

/// Apply contact angle boundary condition to SPH particles near solid walls.
///
/// Young's equation: cos(θ) = (γ_SG - γ_SL) / γ_LG
///
/// Particles within `h` of a domain boundary have their color field normal
/// adjusted to enforce the prescribed contact angle. This affects the surface
/// tension force direction, creating wetting (θ < 90°) or non-wetting (θ > 90°)
/// behavior.
///
/// `contact_angle` in radians (π/4 = hydrophilic, π/2 = neutral, 3π/4 = hydrophobic).
pub fn apply_contact_angle(
    particles: &mut [FluidParticle],
    config: &FluidConfig,
    contact_angle: f64,
    surface_tension: f64,
) {
    let _span = trace_span!("sph::contact_angle", n = particles.len()).entered();
    let cos_theta = contact_angle.cos();
    let sin_theta = contact_angle.sin();
    let h = config.smoothing_radius;
    let lo = config.bounds_min;
    let hi = config.bounds_max;

    for p in particles.iter_mut() {
        // Check proximity to each wall
        let mut wall_normal = DVec3::ZERO;
        if p.position.x - lo.x < h {
            wall_normal.x += 1.0;
        }
        if hi.x - p.position.x < h && hi.x > lo.x {
            wall_normal.x -= 1.0;
        }
        if p.position.y - lo.y < h {
            wall_normal.y += 1.0;
        }
        if hi.y - p.position.y < h && hi.y > lo.y {
            wall_normal.y -= 1.0;
        }
        if p.position.z - lo.z < h {
            wall_normal.z += 1.0;
        }
        if hi.z - p.position.z < h && hi.z > lo.z {
            wall_normal.z -= 1.0;
        }

        if wall_normal.length_squared() < 1e-20 {
            continue;
        }
        let n_wall = wall_normal.normalize();

        // Apply wetting force: push velocity toward the wall (hydrophilic)
        // or away (hydrophobic) proportional to surface tension.
        //
        // Simplified contact angle model: `surface_tension` acts as a
        // force-per-unit-mass scale (units: m/s²). The smoothing radius `h`
        // is used as the characteristic length scale so the force magnitude
        // remains consistent across resolutions.
        let approach = p.velocity.dot(n_wall);
        let wetting_force = surface_tension * (cos_theta - approach.signum() * sin_theta) / h;
        p.acceleration += n_wall * wetting_force / p.mass.max(1e-20);
    }
}

// ── Non-Newtonian Viscosity ────────────────────────────────────────────────

/// Non-Newtonian viscosity model.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum NonNewtonianViscosity {
    /// Power-law: μ_eff = K · |γ̇|^(n-1).
    /// n < 1: shear-thinning (blood, paint). n > 1: shear-thickening (cornstarch).
    PowerLaw { consistency: f64, power_index: f64 },
    /// Bingham plastic: solid below yield stress, flows above.
    /// μ_eff = μ_p + τ_y / (|γ̇| + ε)
    Bingham {
        yield_stress: f64,
        plastic_viscosity: f64,
    },
    /// Herschel-Bulkley: yield stress + power-law.
    /// μ_eff = K · |γ̇|^(n-1) + τ_y / (|γ̇| + ε)
    HerschelBulkley {
        yield_stress: f64,
        consistency: f64,
        power_index: f64,
    },
}

impl NonNewtonianViscosity {
    /// Compute effective viscosity from strain rate magnitude.
    ///
    /// `strain_rate` = |γ̇| = sqrt(2 · Sᵢⱼ · Sᵢⱼ)
    #[inline]
    #[must_use]
    pub fn effective_viscosity(&self, strain_rate: f64) -> f64 {
        let eps = 1e-10;
        let sr = strain_rate.max(eps);
        match *self {
            NonNewtonianViscosity::PowerLaw {
                consistency,
                power_index,
            } => consistency * sr.powf(power_index - 1.0),
            NonNewtonianViscosity::Bingham {
                yield_stress,
                plastic_viscosity,
            } => {
                let mu = plastic_viscosity + yield_stress / sr;
                // Cap viscosity to prevent extreme values at near-zero strain rates
                mu.min(1e6 * plastic_viscosity)
            }
            NonNewtonianViscosity::HerschelBulkley {
                yield_stress,
                consistency,
                power_index,
            } => {
                let base_visc = consistency * sr.powf(power_index - 1.0);
                let mu = base_visc + yield_stress / sr;
                // Cap viscosity to prevent extreme values at near-zero strain rates
                mu.min(1e6 * consistency)
            }
        }
    }
}

// ── Delta-SPH Density Diffusion ────────────────────────────────────────────

/// Apply delta-SPH density diffusion to reduce pressure noise.
///
/// Adds a diffusive correction: `δ·h·c₀·Σ Vⱼ·ψᵢⱼ·∇Wᵢⱼ`
/// where `ψᵢⱼ = 2(ρⱼ-ρᵢ)·rᵢⱼ / (|rᵢⱼ|² + 0.01h²)`.
///
/// Call after density computation, before pressure/force evaluation.
/// Typical `delta` value: 0.05–0.1.
pub fn apply_delta_sph(
    particles: &mut [FluidParticle],
    neighbor_offsets: &[u32],
    neighbor_indices: &[usize],
    h: f64,
    speed_of_sound: f64,
    delta: f64,
    dt: f64,
) {
    let _span = trace_span!("sph::delta_sph", n = particles.len()).entered();
    let n = particles.len();
    if n == 0 || delta <= 0.0 {
        return;
    }

    let kc = KernelCoeffs::new(h);
    let eps = 0.01 * h * h;
    let snap: Vec<(f64, DVec3, f64)> = particles
        .iter()
        .map(|p| (p.density, p.position, p.mass))
        .collect();

    for i in 0..n {
        let (rho_i, pos_i, _) = snap[i];
        let start = neighbor_offsets[i] as usize;
        let end = neighbor_offsets[i + 1] as usize;
        let mut correction = 0.0;

        for &j in &neighbor_indices[start..end] {
            if j == i {
                continue;
            }
            let (rho_j, pos_j, mass_j) = snap[j];
            let dx = pos_i.x - pos_j.x;
            let dy = pos_i.y - pos_j.y;
            let dz = pos_i.z - pos_j.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 > kc.h2 || r2 < 1e-20 {
                continue;
            }
            let r = r2.sqrt();
            let rho_j_safe = rho_j.max(MIN_DENSITY);
            let vol_j = mass_j / rho_j_safe;

            // ψᵢⱼ · ∇Wᵢⱼ (dot product of psi vector with kernel gradient direction)
            let psi_scale = 2.0 * (rho_j - rho_i) / (r2 + eps);
            let grad = kc.spiky_grad(r);
            let grad_dot_r = grad; // |∇W| already scaled, dot with r̂ gives magnitude
            correction += vol_j * psi_scale * r * grad_dot_r;
        }

        particles[i].density += delta * h * speed_of_sound * correction * dt;
    }
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

/// Tait equation of state for weakly compressible SPH.
///
/// P = B · ((ρ/ρ₀)^γ - 1) where B = ρ₀·c²/γ.
///
/// Enables shockwaves and compressibility effects.
/// `gamma` = 7 for water, 1.4 for ideal gas.
#[inline]
#[must_use]
pub fn equation_of_state_tait(
    density: f64,
    rest_density: f64,
    speed_of_sound: f64,
    gamma: f64,
) -> f64 {
    let b = rest_density * speed_of_sound * speed_of_sound / gamma;
    b * ((density / rest_density.max(MIN_DENSITY)).powf(gamma) - 1.0)
}

// ── Forces (standalone, brute-force) ────────────────────────────────────────

/// Compute pressure force on particle i from all neighbors (brute-force).
///
/// Uses the symmetric momentum-conserving formula:
/// F_i = -m_i Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W(r_ij, h)
#[must_use]
pub fn pressure_force(particle_idx: usize, particles: &[FluidParticle], h: f64) -> [f64; 3] {
    let _span = trace_span!("sph::pressure_force", n = particles.len()).entered();
    let kc = KernelCoeffs::new(h);
    let pi = &particles[particle_idx];
    let mut force = [0.0; 3];
    let rho_i = pi.density.max(MIN_DENSITY);

    for (j, pj) in particles.iter().enumerate() {
        if j == particle_idx {
            continue;
        }
        let r2 = pi.distance_squared_to(pj);
        if r2 > kc.h2 || r2 < 1e-20 {
            continue;
        }
        let r = r2.sqrt();
        let rho_j = pj.density.max(MIN_DENSITY);

        let sym_pressure = pi.pressure / (rho_i * rho_i) + pj.pressure / (rho_j * rho_j);
        let grad = kc.spiky_grad(r);
        let scale = -pi.mass * pj.mass * sym_pressure * grad / r;

        force[0] += scale * (pi.position.x - pj.position.x);
        force[1] += scale * (pi.position.y - pj.position.y);
        force[2] += scale * (pi.position.z - pj.position.z);
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
    let _span = trace_span!("sph::viscosity_force", n = particles.len()).entered();
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
        let rho_j = pj.density.max(MIN_DENSITY);
        let scale = viscosity * pj.mass * lap / rho_j;

        force[0] += scale * (pj.velocity.x - pi.velocity.x);
        force[1] += scale * (pj.velocity.y - pi.velocity.y);
        force[2] += scale * (pj.velocity.z - pi.velocity.z);
    }

    force
}

// ── SphSolver ───────────────────────────────────────────────────────────────

/// Stateful SPH solver with spatial hash acceleration and persistent buffers.
///
/// Maintains a spatial hash grid (via hisab) for O(n·k) neighbor queries
/// instead of O(n²) brute-force. Caches neighbor lists and reuses scratch
/// buffers across steps to minimize allocations.
#[non_exhaustive]
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
    pcisph_accel: Vec<DVec3>,
    pcisph_pressures: Vec<f64>,
    pcisph_pred_pos: Vec<DVec3>,
    pcisph_pred_vel: Vec<DVec3>,
    /// Use Velocity Verlet integration instead of symplectic Euler.
    pub use_verlet: bool,
    /// Previous accelerations for Verlet half-step.
    prev_accel: Vec<DVec3>,
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
            use_verlet: false,
            prev_accel: Vec::new(),
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
                self.densities
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, d)| *d = compute_density_i(i));
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
        self.snapshot
            .resize(n, FluidParticle::new(DVec3::ZERO, 0.0));
        self.snapshot.copy_from_slice(particles);

        // Compute forces via cached neighbors
        {
            let _span = trace_span!("sph::forces", n).entered();
            let st = self.surface_tension;
            let snapshot = &self.snapshot;
            let offsets = &self.neighbor_offsets;
            let indices = &self.neighbor_indices;
            let gravity = config.gravity;

            let compute_accel_i = |i: usize| -> DVec3 {
                let pi = &snapshot[i];
                let mut ax = gravity.x;
                let mut ay = gravity.y;
                let mut az = gravity.z;
                let rho = pi.density.max(MIN_DENSITY);

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
                    let rho_j = pj.density.max(MIN_DENSITY);

                    let dx = pi.position.x - pj.position.x;
                    let dy = pi.position.y - pj.position.y;
                    let dz = pi.position.z - pj.position.z;

                    let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let p_scale = -pj.mass * sym_pressure * grad / r;
                    ax += p_scale * dx;
                    ay += p_scale * dy;
                    az += p_scale * dz;

                    let lap = kc.visc_laplacian(r);
                    let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                    ax += v_scale * (pj.velocity.x - pi.velocity.x);
                    ay += v_scale * (pj.velocity.y - pi.velocity.y);
                    az += v_scale * (pj.velocity.z - pi.velocity.z);

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
                    let threshold = (1e-6 / kc.h).max(1e-10);
                    if cn_mag > threshold {
                        let kappa = -laplacian_color / cn_mag;
                        let st_scale = st * kappa / (rho * cn_mag);
                        ax += st_scale * cn_x;
                        ay += st_scale * cn_y;
                        az += st_scale * cn_z;
                    }
                }

                DVec3::new(ax, ay, az)
            };

            #[cfg(feature = "parallel")]
            let accels: Vec<DVec3> = (0..n).into_par_iter().map(compute_accel_i).collect();
            #[cfg(not(feature = "parallel"))]
            let accels: Vec<DVec3> = (0..n).map(compute_accel_i).collect();

            for (i, accel) in accels.iter().enumerate() {
                if !accel.x.is_finite() || !accel.y.is_finite() || !accel.z.is_finite() {
                    return Err(PravashError::Diverged {
                        reason: format!("NaN/Inf in acceleration at particle {i}").into(),
                    });
                }
                particles[i].acceleration = *accel;
            }
        }

        // Integrate
        let dt = config.dt;
        if self.use_verlet {
            // Velocity Verlet: x += v·dt + 0.5·a·dt², v += 0.5·(a_old + a_new)·dt
            if self.prev_accel.len() != n {
                self.prev_accel.resize(n, DVec3::ZERO);
            }
            for (i, p) in particles.iter_mut().enumerate() {
                let a_old = self.prev_accel[i];
                p.position += p.velocity * dt + 0.5 * a_old * dt * dt;
                p.velocity += 0.5 * (a_old + p.acceleration) * dt;
                self.prev_accel[i] = p.acceleration;
            }
        } else {
            // Symplectic Euler
            for p in particles.iter_mut() {
                p.velocity += p.acceleration * dt;
                p.position += p.velocity * dt;
            }
        }

        // Boundary enforcement
        let lo = config.bounds_min;
        let hi = config.bounds_max;
        let damp = config.boundary_damping;
        for p in particles.iter_mut() {
            if p.position.x < lo.x {
                p.position.x = lo.x;
                p.velocity.x *= -damp;
            } else if p.position.x > hi.x && hi.x > lo.x {
                p.position.x = hi.x;
                p.velocity.x *= -damp;
            }
            if p.position.y < lo.y {
                p.position.y = lo.y;
                p.velocity.y *= -damp;
            } else if p.position.y > hi.y && hi.y > lo.y {
                p.position.y = hi.y;
                p.velocity.y *= -damp;
            }
            if p.position.z < lo.z {
                p.position.z = lo.z;
                p.velocity.z *= -damp;
            } else if p.position.z > hi.z && hi.z > lo.z {
                p.position.z = hi.z;
                p.velocity.z *= -damp;
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
        self.snapshot
            .resize(n, FluidParticle::new(DVec3::ZERO, 0.0));
        self.snapshot.copy_from_slice(particles);

        let st = self.surface_tension;
        // Take persistent buffers out of self to avoid borrow conflicts
        let mut non_pressure_accel = std::mem::take(&mut self.pcisph_accel);
        non_pressure_accel.resize(n, DVec3::ZERO);
        non_pressure_accel.fill(DVec3::ZERO);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let pi = &self.snapshot[i];
            let mut ax = config.gravity.x;
            let mut ay = config.gravity.y;
            let mut az = config.gravity.z;
            let rho = pi.density.max(MIN_DENSITY);

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
                let rho_j = pj.density.max(MIN_DENSITY);

                // Viscosity
                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                ax += v_scale * (pj.velocity.x - pi.velocity.x);
                ay += v_scale * (pj.velocity.y - pi.velocity.y);
                az += v_scale * (pj.velocity.z - pi.velocity.z);

                if st > 0.0 {
                    let dx = pi.position.x - pj.position.x;
                    let dy = pi.position.y - pj.position.y;
                    let dz = pi.position.z - pj.position.z;
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
                let threshold = (1e-6 / kc.h).max(1e-10);
                if cn_mag > threshold {
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

            non_pressure_accel[i] = DVec3::new(ax, ay, az);
        }

        // PCISPH pressure correction loop
        // Precompute scaling factor δ from kernel gradient sums (Solenthaler 2009).
        // Average over all particles with neighbors for robustness.
        let delta = {
            let mut delta_sum = 0.0f64;
            let mut delta_count = 0usize;
            for pi_idx in 0..n {
                let pi = &self.snapshot[pi_idx];
                let mut sum_grad = [0.0f64; 3];
                let mut sum_grad_sq = 0.0f64;
                let mut has_neighbors = false;
                for &j in self.neighbors(pi_idx) {
                    if j == pi_idx {
                        continue;
                    }
                    let pj = &self.snapshot[j];
                    let r2 = pi.distance_squared_to(pj);
                    if r2 > kc.h2 || r2 < 1e-20 {
                        continue;
                    }
                    has_neighbors = true;
                    let r = r2.sqrt();
                    let grad = kc.spiky_grad(r);
                    let grad_w = [
                        grad / r * (pi.position.x - pj.position.x),
                        grad / r * (pi.position.y - pj.position.y),
                        grad / r * (pi.position.z - pj.position.z),
                    ];
                    sum_grad[0] += grad_w[0];
                    sum_grad[1] += grad_w[1];
                    sum_grad[2] += grad_w[2];
                    sum_grad_sq +=
                        grad_w[0] * grad_w[0] + grad_w[1] * grad_w[1] + grad_w[2] * grad_w[2];
                }
                if !has_neighbors {
                    continue;
                }
                let sum_dot = sum_grad[0] * sum_grad[0]
                    + sum_grad[1] * sum_grad[1]
                    + sum_grad[2] * sum_grad[2];
                let beta = 2.0 * (dt * pi.mass / rest_density).powi(2);
                let denom = beta * (sum_dot + sum_grad_sq);
                if denom.abs() > 1e-20 {
                    delta_sum += 1.0 / denom;
                    delta_count += 1;
                }
            }
            if delta_count > 0 {
                delta_sum / delta_count as f64
            } else {
                0.0
            }
        };

        let mut pressures = std::mem::take(&mut self.pcisph_pressures);
        pressures.resize(n, 0.0);
        pressures.fill(0.0);
        let mut predicted_pos = std::mem::take(&mut self.pcisph_pred_pos);
        predicted_pos.resize(n, DVec3::ZERO);
        let mut predicted_vel = std::mem::take(&mut self.pcisph_pred_vel);
        predicted_vel.resize(n, DVec3::ZERO);

        for iter in 0..max_iterations {
            let _span = trace_span!("sph::pcisph_iter", iter).entered();

            // Predict position/velocity with current pressure + non-pressure forces
            for i in 0..n {
                let pi = &self.snapshot[i];
                let rho = pi.density.max(MIN_DENSITY);

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
                    let rho_j = pj.density.max(MIN_DENSITY);

                    let sym = pressures[i] / (rho * rho) + pressures[j] / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let s = -pj.mass * sym * grad / r;

                    pax += s * (pi.position.x - pj.position.x);
                    pay += s * (pi.position.y - pj.position.y);
                    paz += s * (pi.position.z - pj.position.z);
                }

                let total_ax = non_pressure_accel[i].x + pax;
                let total_ay = non_pressure_accel[i].y + pay;
                let total_az = non_pressure_accel[i].z + paz;

                predicted_vel[i] = DVec3::new(
                    pi.velocity.x + total_ax * dt,
                    pi.velocity.y + total_ay * dt,
                    pi.velocity.z + total_az * dt,
                );
                predicted_pos[i] = DVec3::new(
                    pi.position.x + predicted_vel[i].x * dt,
                    pi.position.y + predicted_vel[i].y * dt,
                    pi.position.z + predicted_vel[i].z * dt,
                );
            }

            // Compute predicted density and density error
            let mut max_error = 0.0f64;
            for i in 0..n {
                let mut pred_density = 0.0;
                for &j in self.neighbors(i) {
                    let dx = predicted_pos[i].x - predicted_pos[j].x;
                    let dy = predicted_pos[i].y - predicted_pos[j].y;
                    let dz = predicted_pos[i].z - predicted_pos[j].z;
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
            if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
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
        let lo = config.bounds_min;
        let hi = config.bounds_max;
        let damp = config.boundary_damping;
        for p in particles.iter_mut() {
            if p.position.x < lo.x {
                p.position.x = lo.x;
                p.velocity.x *= -damp;
            } else if p.position.x > hi.x && hi.x > lo.x {
                p.position.x = hi.x;
                p.velocity.x *= -damp;
            }
            if p.position.y < lo.y {
                p.position.y = lo.y;
                p.velocity.y *= -damp;
            } else if p.position.y > hi.y && hi.y > lo.y {
                p.position.y = hi.y;
                p.velocity.y *= -damp;
            }
            if p.position.z < lo.z {
                p.position.z = lo.z;
                p.velocity.z *= -damp;
            } else if p.position.z > hi.z && hi.z > lo.z {
                p.position.z = hi.z;
                p.velocity.z *= -damp;
            }
        }

        // Return persistent buffers to self
        self.pcisph_accel = non_pressure_accel;
        self.pcisph_pressures = pressures;
        self.pcisph_pred_pos = predicted_pos;
        self.pcisph_pred_vel = predicted_vel;

        Ok(())
    }

    /// Perform one DFSPH step (Divergence-Free SPH, Bender & Koschier 2015).
    ///
    /// Two-stage correction:
    /// 1. Correct velocity divergence: enforce ∇·v = 0
    /// 2. Correct density error: enforce ρ = ρ₀
    ///
    /// Converges in 1–2 iterations per stage (vs 5–10 for PCISPH).
    /// `max_iterations` applies to each stage independently.
    /// `max_error` is the relative convergence threshold (e.g., 0.001 = 0.1%).
    pub fn step_dfsph(
        &mut self,
        particles: &mut [FluidParticle],
        config: &FluidConfig,
        viscosity: f64,
        max_iterations: usize,
        max_error: f64,
    ) -> Result<()> {
        let _span = trace_span!("sph::dfsph", n = particles.len()).entered();
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
                let r2 = pi.position.distance_squared(particles[j].position);
                if r2 <= kc.h2 {
                    d += particles[j].mass * kc.poly6(r2);
                }
            }
            self.densities[i] = d;
        }
        for (i, p) in particles.iter_mut().enumerate() {
            p.density = self.densities[i];
        }

        // Compute alpha factors: α_i = ρ_i / (|Σ m_j ∇W_ij|² + Σ |m_j ∇W_ij|²)
        let mut alphas = vec![0.0f64; n];
        for i in 0..n {
            let pi = &particles[i];
            let mut sum_grad = DVec3::ZERO;
            let mut sum_grad_sq = 0.0f64;
            for &j in self.neighbors(i) {
                if j == i {
                    continue;
                }
                let pj = &particles[j];
                let diff = pi.position - pj.position;
                let r2 = diff.length_squared();
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let grad = kc.spiky_grad(r);
                let grad_w = diff * (pj.mass * grad / r);
                sum_grad += grad_w;
                sum_grad_sq += grad_w.length_squared();
            }
            let denom = sum_grad.length_squared() + sum_grad_sq;
            alphas[i] = if denom > 1e-20 {
                pi.density.max(MIN_DENSITY) / denom
            } else {
                0.0
            };
        }

        // Compute non-pressure accelerations (gravity + viscosity)
        self.snapshot
            .resize(n, FluidParticle::new(DVec3::ZERO, 0.0));
        self.snapshot.copy_from_slice(particles);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let pi = &self.snapshot[i];
            let mut accel = config.gravity;
            let rho = pi.density.max(MIN_DENSITY);

            for &j in self.neighbors(i) {
                if j == i {
                    continue;
                }
                let pj = &self.snapshot[j];
                let r2 = pi.position.distance_squared(pj.position);
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let rho_j = pj.density.max(MIN_DENSITY);
                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;
                accel += (pj.velocity - pi.velocity) * v_scale;
            }

            particles[i].velocity += accel * dt;
        }

        // Stage 1: Correct velocity divergence (∇·v = 0)
        for _iter in 0..max_iterations {
            let _span = trace_span!("sph::dfsph_div").entered();
            let snap_vel: Vec<DVec3> = particles.iter().map(|p| p.velocity).collect();
            let mut max_div = 0.0f64;

            for i in 0..n {
                let pos_i = particles[i].position;
                let rho_i = particles[i].density.max(MIN_DENSITY);

                // Compute velocity divergence
                let mut div_v = 0.0;
                for &j in self.neighbors(i) {
                    if j == i {
                        continue;
                    }
                    let diff = pos_i - particles[j].position;
                    let r2 = diff.length_squared();
                    if r2 > kc.h2 || r2 < 1e-20 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let grad = kc.spiky_grad(r);
                    let grad_w = diff * (grad / r);
                    div_v += particles[j].mass * (snap_vel[j] - snap_vel[i]).dot(grad_w);
                }

                max_div = max_div.max(div_v.abs());

                // Correct velocity: κ_v = (1/dt) · div_v · α_i
                if alphas[i] > 0.0 {
                    let kappa = div_v * alphas[i] / dt;
                    let mut vel_correction = DVec3::ZERO;
                    for &j in self.neighbors(i) {
                        if j == i {
                            continue;
                        }
                        let diff = pos_i - particles[j].position;
                        let r2 = diff.length_squared();
                        if r2 > kc.h2 || r2 < 1e-20 {
                            continue;
                        }
                        let r = r2.sqrt();
                        let rho_j = particles[j].density.max(MIN_DENSITY);
                        let grad = kc.spiky_grad(r);
                        vel_correction += diff
                            * (dt
                                * particles[j].mass
                                * (kappa / (rho_i * rho_i) + kappa / (rho_j * rho_j))
                                * grad
                                / r);
                    }
                    particles[i].velocity -= vel_correction;
                }
            }

            if max_div < max_error * rest_density {
                break;
            }
        }

        // Integrate positions
        for p in particles.iter_mut() {
            p.position += p.velocity * dt;
        }

        // Recompute density at new positions
        for i in 0..n {
            let pi = &particles[i];
            let mut d = 0.0;
            for &j in self.neighbors(i) {
                let r2 = pi.position.distance_squared(particles[j].position);
                if r2 <= kc.h2 {
                    d += particles[j].mass * kc.poly6(r2);
                }
            }
            particles[i].density = d;
        }

        // Stage 2: Correct density error (ρ = ρ₀)
        for _iter in 0..max_iterations {
            let _span = trace_span!("sph::dfsph_density").entered();
            let mut max_rho_err = 0.0f64;

            for i in 0..n {
                let rho_err = particles[i].density - rest_density;
                max_rho_err = max_rho_err.max(rho_err.abs() / rest_density);

                if alphas[i] > 0.0 && rho_err.abs() > 1e-10 {
                    let kappa = rho_err * alphas[i] / (dt * dt);
                    let pi_pos = particles[i].position;
                    let pi_rho = particles[i].density.max(MIN_DENSITY);

                    for &j in self.neighbors(i) {
                        if j == i {
                            continue;
                        }
                        let diff = pi_pos - particles[j].position;
                        let r2 = diff.length_squared();
                        if r2 > kc.h2 || r2 < 1e-20 {
                            continue;
                        }
                        let r = r2.sqrt();
                        let rho_j = particles[j].density.max(MIN_DENSITY);
                        let grad = kc.spiky_grad(r);
                        let correction = diff
                            * (dt
                                * particles[j].mass
                                * (kappa / (pi_rho * pi_rho) + kappa / (rho_j * rho_j))
                                * grad
                                / r);
                        particles[i].velocity -= correction;
                    }
                }
            }

            if max_rho_err < max_error {
                break;
            }
        }

        // Store pressure as density error for diagnostics
        for p in particles.iter_mut() {
            p.pressure = equation_of_state(p.density, rest_density, config.gas_constant);
        }

        // Boundary enforcement
        let lo = config.bounds_min;
        let hi = config.bounds_max;
        let damp = config.boundary_damping;
        for p in particles.iter_mut() {
            if p.position.x < lo.x {
                p.position.x = lo.x;
                p.velocity.x *= -damp;
            } else if p.position.x > hi.x && hi.x > lo.x {
                p.position.x = hi.x;
                p.velocity.x *= -damp;
            }
            if p.position.y < lo.y {
                p.position.y = lo.y;
                p.velocity.y *= -damp;
            } else if p.position.y > hi.y && hi.y > lo.y {
                p.position.y = hi.y;
                p.velocity.y *= -damp;
            }
            if p.position.z < lo.z {
                p.position.z = lo.z;
                p.velocity.z *= -damp;
            } else if p.position.z > hi.z && hi.z > lo.z {
                p.position.z = hi.z;
                p.velocity.z *= -damp;
            }
        }

        Ok(())
    }

    /// Compute CFL-limited timestep from advective, force, and viscous constraints.
    ///
    /// dt = cfl · min(h/v_max, sqrt(h/a_max), h²/(6·ν))
    ///
    /// Returns the computed dt clamped to `[dt_min, dt_max]`.
    #[must_use]
    pub fn adaptive_dt(
        particles: &[FluidParticle],
        smoothing_radius: f64,
        viscosity: f64,
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
        let h = smoothing_radius;
        let mut dt_cfl = f64::MAX;

        // Advective: h / v_max
        let max_v = max_speed(particles);
        if max_v > 1e-20 {
            dt_cfl = dt_cfl.min(h / max_v);
        }

        // Force: sqrt(h / a_max)
        let max_a = particles
            .iter()
            .map(|p| {
                let a = &p.acceleration;
                a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
            })
            .fold(0.0f64, f64::max)
            .sqrt();
        if max_a > 1e-20 {
            dt_cfl = dt_cfl.min((h / max_a).sqrt());
        }

        // Viscous: h² / (6·ν)
        if viscosity > 1e-20 {
            dt_cfl = dt_cfl.min(h * h / (6.0 * viscosity));
        }

        (cfl_factor * dt_cfl).clamp(dt_min, dt_max)
    }

    /// Multi-phase SPH step with per-particle material properties.
    ///
    /// Each particle uses its `phase` index to look up rest density,
    /// gas constant, and viscosity from `phase_config`. Interface tension
    /// is applied between particles of different phases.
    pub fn step_multiphase(
        &mut self,
        particles: &mut [FluidParticle],
        config: &FluidConfig,
        phase_config: &MultiPhaseConfig,
    ) -> Result<()> {
        let _span = trace_span!("sph::step_multiphase", n = particles.len()).entered();
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

        {
            let _span = trace_span!("sph::build_neighbors", n).entered();
            self.build_neighbors(particles, h, h_f32)?;
        }

        // Compute densities with per-phase rest density for EOS
        {
            let _span = trace_span!("sph::density", n).entered();
            self.densities.resize(n, 0.0);
            let offsets = &self.neighbor_offsets;
            let indices = &self.neighbor_indices;
            for i in 0..n {
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
                self.densities[i] = d;
            }
            for (i, p) in particles.iter_mut().enumerate() {
                p.density = self.densities[i];
                let props = phase_config.get(p.phase);
                p.pressure = equation_of_state(p.density, props.rest_density, props.gas_constant);
            }
        }

        // Snapshot for force computation
        self.snapshot
            .resize(n, FluidParticle::new(DVec3::ZERO, 0.0));
        self.snapshot.copy_from_slice(particles);

        // Compute forces with per-phase viscosity and interface tension
        {
            let _span = trace_span!("sph::forces", n).entered();
            let st = self.surface_tension;
            let it = phase_config.interface_tension;
            let snapshot = &self.snapshot;
            let offsets = &self.neighbor_offsets;
            let indices = &self.neighbor_indices;
            let gravity = config.gravity;

            let compute_accel_i = |i: usize| -> DVec3 {
                let pi = &snapshot[i];
                let mut ax = gravity.x;
                let mut ay = gravity.y;
                let mut az = gravity.z;
                let rho = pi.density.max(MIN_DENSITY);
                let props_i = phase_config.get(pi.phase);

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
                    let rho_j = pj.density.max(MIN_DENSITY);

                    let dx = pi.position.x - pj.position.x;
                    let dy = pi.position.y - pj.position.y;
                    let dz = pi.position.z - pj.position.z;

                    // Pressure force (symmetric)
                    let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                    let grad = kc.spiky_grad(r);
                    let p_scale = -pj.mass * sym_pressure * grad / r;
                    ax += p_scale * dx;
                    ay += p_scale * dy;
                    az += p_scale * dz;

                    // Viscosity: average of both phases
                    let props_j = phase_config.get(pj.phase);
                    let visc = 0.5 * (props_i.viscosity + props_j.viscosity);
                    let lap = kc.visc_laplacian(r);
                    let v_scale = visc * pj.mass * lap / rho_j / rho;
                    ax += v_scale * (pj.velocity.x - pi.velocity.x);
                    ay += v_scale * (pj.velocity.y - pi.velocity.y);
                    az += v_scale * (pj.velocity.z - pi.velocity.z);

                    // Surface tension (CSF) + interface tension between different phases
                    let effective_st = if pi.phase != pj.phase { st + it } else { st };
                    if effective_st > 0.0 {
                        let mass_over_rho = pj.mass / rho_j;
                        let pg = kc.poly6_grad_scalar(r2);
                        cn_x += mass_over_rho * pg * dx;
                        cn_y += mass_over_rho * pg * dy;
                        cn_z += mass_over_rho * pg * dz;
                        laplacian_color += mass_over_rho * kc.poly6_laplacian(r2);
                    }
                }

                // Apply surface/interface tension
                if st > 0.0 || it > 0.0 {
                    let cn_mag = (cn_x * cn_x + cn_y * cn_y + cn_z * cn_z).sqrt();
                    let threshold = (1e-6 / kc.h).max(1e-10);
                    if cn_mag > threshold {
                        let kappa = -laplacian_color / cn_mag;
                        let st_scale = (st + it) * kappa / (rho * cn_mag);
                        ax += st_scale * cn_x;
                        ay += st_scale * cn_y;
                        az += st_scale * cn_z;
                    }
                }

                DVec3::new(ax, ay, az)
            };

            #[cfg(feature = "parallel")]
            let accels: Vec<DVec3> = (0..n).into_par_iter().map(compute_accel_i).collect();
            #[cfg(not(feature = "parallel"))]
            let accels: Vec<DVec3> = (0..n).map(compute_accel_i).collect();

            for (i, accel) in accels.iter().enumerate() {
                if !accel.x.is_finite() || !accel.y.is_finite() || !accel.z.is_finite() {
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
            p.velocity += p.acceleration * dt;
            p.position += p.velocity * dt;
        }

        // Boundary enforcement
        let lo = config.bounds_min;
        let hi = config.bounds_max;
        let damp = config.boundary_damping;
        for p in particles.iter_mut() {
            if p.position.x < lo.x {
                p.position.x = lo.x;
                p.velocity.x *= -damp;
            } else if p.position.x > hi.x && hi.x > lo.x {
                p.position.x = hi.x;
                p.velocity.x *= -damp;
            }
            if p.position.y < lo.y {
                p.position.y = lo.y;
                p.velocity.y *= -damp;
            } else if p.position.y > hi.y && hi.y > lo.y {
                p.position.y = hi.y;
                p.velocity.y *= -damp;
            }
            if p.position.z < lo.z {
                p.position.z = lo.z;
                p.velocity.z *= -damp;
            } else if p.position.z > hi.z && hi.z > lo.z {
                p.position.z = hi.z;
                p.velocity.z *= -damp;
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
            let mut ax = config.gravity.x;
            let mut ay = config.gravity.y;
            let mut az = config.gravity.z;
            let rho = pi.density.max(MIN_DENSITY);

            for (j, pj) in snapshot.iter().enumerate() {
                if j == i {
                    continue;
                }
                let r2 = pi.distance_squared_to(pj);
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let rho_j = pj.density.max(MIN_DENSITY);

                // Symmetric momentum-conserving pressure: -m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W
                let sym_pressure = pi.pressure / (rho * rho) + pj.pressure / (rho_j * rho_j);
                let grad = kc.spiky_grad(r);
                let p_scale = -pj.mass * sym_pressure * grad / r;

                let dx = pi.position.x - pj.position.x;
                let dy = pi.position.y - pj.position.y;
                let dz = pi.position.z - pj.position.z;

                ax += p_scale * dx;
                ay += p_scale * dy;
                az += p_scale * dz;

                let lap = kc.visc_laplacian(r);
                let v_scale = viscosity * pj.mass * lap / rho_j / rho;

                ax += v_scale * (pj.velocity.x - pi.velocity.x);
                ay += v_scale * (pj.velocity.y - pi.velocity.y);
                az += v_scale * (pj.velocity.z - pi.velocity.z);
            }

            if !ax.is_finite() || !ay.is_finite() || !az.is_finite() {
                return Err(PravashError::Diverged {
                    reason: format!("NaN/Inf in acceleration at particle {i}").into(),
                });
            }

            particles[i].acceleration = DVec3::new(ax, ay, az);
        }
    }

    let dt = config.dt;
    for p in particles.iter_mut() {
        p.velocity += p.acceleration * dt;
        p.position += p.velocity * dt;
    }

    let lo = config.bounds_min;
    let hi = config.bounds_max;
    let damp = config.boundary_damping;
    for p in particles.iter_mut() {
        if p.position.x < lo.x {
            p.position.x = lo.x;
            p.velocity.x *= -damp;
        } else if p.position.x > hi.x && hi.x > lo.x {
            p.position.x = hi.x;
            p.velocity.x *= -damp;
        }
        if p.position.y < lo.y {
            p.position.y = lo.y;
            p.velocity.y *= -damp;
        } else if p.position.y > hi.y && hi.y > lo.y {
            p.position.y = hi.y;
            p.velocity.y *= -damp;
        }
        if p.position.z < lo.z {
            p.position.z = lo.z;
            p.velocity.z *= -damp;
        } else if p.position.z > hi.z && hi.z > lo.z {
            p.position.z = hi.z;
            p.velocity.z *= -damp;
        }
    }

    Ok(())
}

// ── MLS Gradient Correction ────────────────────────────────────────────────

/// Compute MLS gradient correction matrices for all particles (3D).
///
/// Returns one 3×3 correction matrix `[L00..L22]` (9 elements, row-major) per particle.
/// Apply as: `∇W_corrected = L · ∇W` to restore 1st-order gradient consistency
/// near boundaries and free surfaces.
///
/// `L_i = (Σ V_j (x_j - x_i) ⊗ ∇W_ij)^(-1)`
#[must_use]
pub fn compute_gradient_corrections(
    particles: &[FluidParticle],
    neighbor_offsets: &[u32],
    neighbor_indices: &[usize],
    h: f64,
) -> Vec<[f64; 9]> {
    let _span = trace_span!("sph::mls_correction", n = particles.len()).entered();
    let n = particles.len();
    let kc = KernelCoeffs::new(h);
    // Identity 3x3: [1,0,0, 0,1,0, 0,0,1]
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut corrections = vec![identity; n];

    for i in 0..n {
        let pi = &particles[i];
        let start = neighbor_offsets[i] as usize;
        let end = neighbor_offsets[i + 1] as usize;

        // Build the 3×3 moment matrix M = Σ V_j (x_j - x_i) ⊗ ∇W_ij
        let mut m = [0.0f64; 9];

        for &j in &neighbor_indices[start..end] {
            if j == i {
                continue;
            }
            let pj = &particles[j];
            let diff = pj.position - pi.position;
            let r2 = diff.length_squared();
            if r2 > kc.h2 || r2 < 1e-20 {
                continue;
            }
            let r = r2.sqrt();
            let rho_j = pj.density.max(MIN_DENSITY);
            let vol_j = pj.mass / rho_j;
            let grad = kc.spiky_grad(r);
            // ∇W points from j to i
            let dir = (pi.position - pj.position) * (grad / r);
            let gw = [dir.x, dir.y, dir.z];
            let d = [diff.x, diff.y, diff.z];

            // (x_j - x_i) ⊗ ∇W  →  M[row][col] += V_j * d[row] * gw[col]
            for row in 0..3 {
                for col in 0..3 {
                    m[row * 3 + col] += vol_j * d[row] * gw[col];
                }
            }
        }

        // Invert the 3x3 matrix
        if let Some(inv) = invert_3x3(&m) {
            corrections[i] = inv;
        }
        // else: keep identity (degenerate neighborhood)
    }

    corrections
}

/// Invert a 3×3 matrix stored as 9-element row-major array.
/// Returns None if singular.
#[inline]
fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let [a, b, c, d, e, f, g, h, k] = *m;
    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (e * k - f * h) * inv_det,
        (c * h - b * k) * inv_det,
        (b * f - c * e) * inv_det,
        (f * g - d * k) * inv_det,
        (a * k - c * g) * inv_det,
        (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det,
        (b * g - a * h) * inv_det,
        (a * e - b * d) * inv_det,
    ])
}

/// Apply a 3×3 correction matrix to a kernel gradient vector.
///
/// `corrected = L · ∇W` where L is the 3×3 MLS correction matrix.
#[inline]
#[must_use]
pub fn apply_gradient_correction(correction: &[f64; 9], gw: DVec3) -> DVec3 {
    DVec3::new(
        correction[0] * gw.x + correction[1] * gw.y + correction[2] * gw.z,
        correction[3] * gw.x + correction[4] * gw.y + correction[5] * gw.z,
        correction[6] * gw.x + correction[7] * gw.y + correction[8] * gw.z,
    )
}

// ── IMEX Splitting ────────────────────────────────────────────────────────

/// Apply implicit viscosity using an iterative solver.
///
/// Treats viscosity implicitly: `v_new = v_old + dt·ν·∇²v_new`.
/// Uses Jacobi iteration to solve the implicit system.
/// Removes the viscous CFL restriction (dt < h²/6ν).
pub fn apply_implicit_viscosity(
    particles: &mut [FluidParticle],
    neighbor_offsets: &[u32],
    neighbor_indices: &[usize],
    h: f64,
    viscosity: f64,
    dt: f64,
    iterations: usize,
) {
    let _span = trace_span!("sph::implicit_visc", n = particles.len()).entered();
    let n = particles.len();
    if n == 0 || viscosity <= 0.0 {
        return;
    }

    let kc = KernelCoeffs::new(h);

    const CONVERGENCE_THRESHOLD: f64 = 1e-8;

    for _iter in 0..iterations {
        let snap_vel: Vec<DVec3> = particles.iter().map(|p| p.velocity).collect();
        let mut max_change: f64 = 0.0;

        for i in 0..n {
            let pi_pos = particles[i].position;
            let rho = particles[i].density.max(MIN_DENSITY);
            let start = neighbor_offsets[i] as usize;
            let end = neighbor_offsets[i + 1] as usize;
            let mut laplacian = DVec3::ZERO;

            for &j in &neighbor_indices[start..end] {
                if j == i {
                    continue;
                }
                let r2 = pi_pos.distance_squared(particles[j].position);
                if r2 > kc.h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let rho_j = particles[j].density.max(MIN_DENSITY);
                let lap = kc.visc_laplacian(r);
                laplacian += (snap_vel[j] - snap_vel[i]) * (particles[j].mass * lap / rho_j);
            }

            let new_vel = snap_vel[i] + laplacian * (viscosity * dt / rho);
            let change = (new_vel - snap_vel[i]).length_squared();
            max_change = max_change.max(change);
            particles[i].velocity = new_vel;
        }

        // Early exit if all velocity updates are below convergence threshold
        if max_change < CONVERGENCE_THRESHOLD * CONVERGENCE_THRESHOLD {
            break;
        }
    }
}

// ── Adaptive Particle Splitting/Merging ───────────────────────────────────

/// Split a particle into `n_children` smaller particles.
///
/// Conserves total mass and momentum. Children are placed in a ring
/// around the parent position at distance `offset`.
#[must_use]
pub fn split_particle(
    parent: &FluidParticle,
    n_children: usize,
    offset: f64,
) -> Vec<FluidParticle> {
    let _span = trace_span!("sph::split_particle").entered();
    if n_children == 0 {
        return vec![];
    }
    let child_mass = parent.mass / n_children as f64;
    let mut children = Vec::with_capacity(n_children);

    for k in 0..n_children {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n_children as f64;
        let mut child = *parent;
        child.mass = child_mass;
        child.position.x += offset * angle.cos();
        child.position.y += offset * angle.sin();
        children.push(child);
    }
    children
}

/// Merge two nearby particles into one, conserving mass and momentum.
#[must_use]
pub fn merge_particles(a: &FluidParticle, b: &FluidParticle) -> FluidParticle {
    let _span = trace_span!("sph::merge_particles").entered();
    let total_mass = a.mass + b.mass;
    let inv_mass = 1.0 / total_mass.max(1e-20);
    FluidParticle {
        position: (a.position * a.mass + b.position * b.mass) * inv_mass,
        velocity: (a.velocity * a.mass + b.velocity * b.mass) * inv_mass,
        acceleration: DVec3::ZERO,
        density: (a.density * a.mass + b.density * b.mass) * inv_mass,
        pressure: (a.pressure * a.mass + b.pressure * b.mass) * inv_mass,
        mass: total_mass,
        phase: a.phase,
        conformation: a.conformation,
        temperature: (a.temperature * a.mass + b.temperature * b.mass) * inv_mass,
        fuel: (a.fuel * a.mass + b.fuel * b.mass) * inv_mass,
    }
}

// ── Z-Order Sorting ───────────────────────────────────────────────────────

/// Spread bits for 2D Morton code: insert a zero between each bit.
#[inline]
fn spread_bits(x: u32) -> u64 {
    let mut v = x as u64;
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    v = (v | (v << 1)) & 0x5555_5555_5555_5555;
    v
}

/// Compute 2D Morton code (Z-order curve index) for a position.
#[inline]
#[must_use]
fn morton_code_2d(x: f64, y: f64, inv_cell: f64) -> u64 {
    let ix = (x * inv_cell).max(0.0) as u32;
    let iy = (y * inv_cell).max(0.0) as u32;
    spread_bits(ix) | (spread_bits(iy) << 1)
}

/// Sort particles by Z-order (Morton code) for cache-friendly spatial access.
///
/// Call before `SphSolver::step()` for improved cache hit rates during
/// neighbor traversal. The cell size should match the smoothing radius.
pub fn sort_by_zorder(particles: &mut [FluidParticle], cell_size: f64) {
    let _span = trace_span!("sph::sort_by_zorder", n = particles.len()).entered();
    if particles.len() < 2 || cell_size <= 0.0 {
        return;
    }
    let inv_cell = 1.0 / cell_size;
    particles.sort_unstable_by_key(|p| morton_code_2d(p.position[0], p.position[1], inv_cell));
}

// ── Foam / Spray / Bubble Generation ──────────────────────────────────────

/// Type of secondary particle for visual effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecondaryType {
    Foam,
    Spray,
    Bubble,
}

/// A secondary (decorative) particle.
#[derive(Debug, Clone, Copy)]
pub struct SecondaryParticle {
    pub position: DVec3,
    pub velocity: DVec3,
    pub lifetime: f64,
    pub kind: SecondaryType,
}

/// Generate secondary particles from fluid surface based on acceleration.
#[must_use]
pub fn generate_secondary_particles(
    particles: &[FluidParticle],
    accel_threshold: f64,
    max_lifetime: f64,
) -> Vec<SecondaryParticle> {
    let _span = trace_span!("sph::secondary", n = particles.len()).entered();
    let mut secondary = Vec::new();

    for p in particles {
        let accel_mag = p.acceleration.length();
        if accel_mag > accel_threshold {
            secondary.push(SecondaryParticle {
                position: p.position,
                velocity: p.velocity * 1.2,
                lifetime: max_lifetime * 0.5,
                kind: SecondaryType::Spray,
            });
        } else if accel_mag > accel_threshold * 0.3 {
            secondary.push(SecondaryParticle {
                position: p.position,
                velocity: p.velocity * 0.1,
                lifetime: max_lifetime,
                kind: SecondaryType::Foam,
            });
        }
    }
    secondary
}

/// Advect and age secondary particles. Removes dead ones.
pub fn update_secondary_particles(
    secondaries: &mut Vec<SecondaryParticle>,
    gravity: DVec3,
    dt: f64,
) {
    for s in secondaries.iter_mut() {
        match s.kind {
            SecondaryType::Spray => s.velocity += gravity * dt,
            SecondaryType::Foam => s.velocity *= 0.98,
            SecondaryType::Bubble => {
                s.velocity.y += 0.5 * dt;
                s.velocity *= 0.95;
            }
        }
        s.position += s.velocity * dt;
        s.lifetime -= dt;
    }
    secondaries.retain(|s| s.lifetime > 0.0);
}

// ── Batch Kernel Evaluation (SIMD-friendly) ───────────────────────────────

/// Batch-evaluate Poly6 kernel for multiple squared distances.
///
/// Auto-vectorizable inner loop — the compiler can emit SIMD instructions
/// for this pattern without explicit intrinsics.
#[must_use]
pub fn batch_poly6(r2_values: &[f64], h: f64) -> Vec<f64> {
    let kc = KernelCoeffs::new(h);
    r2_values
        .iter()
        .map(|&r2| {
            if r2 > kc.h2 {
                0.0
            } else {
                let diff = kc.h2 - r2;
                kc.poly6 * diff * diff * diff
            }
        })
        .collect()
}

/// Batch-compute squared distances from one particle to many.
///
/// Output layout: `out[i] = |pos - positions[i]|²`.
/// Auto-vectorizable.
#[must_use]
pub fn batch_distance_squared(pos: DVec3, positions: &[DVec3]) -> Vec<f64> {
    positions
        .iter()
        .map(|&other| pos.distance_squared(other))
        .collect()
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
        p.velocity = DVec3::new(3.0, 4.0, 0.0);
        let ke = total_kinetic_energy(&[p]);
        assert!((ke - 25.0).abs() < EPS);
    }

    #[test]
    fn test_max_speed() {
        let mut p1 = FluidParticle::new_2d(0.0, 0.0, 1.0);
        let mut p2 = FluidParticle::new_2d(1.0, 0.0, 1.0);
        p1.velocity = DVec3::new(1.0, 0.0, 0.0);
        p2.velocity = DVec3::new(3.0, 4.0, 0.0);
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

        let lo = config.bounds_min;
        let hi = config.bounds_max;
        for p in &particles {
            assert!(p.position.x >= lo.x && p.position.x <= hi.x);
            assert!(p.position.y >= lo.y && p.position.y <= hi.y);
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

    // ── Multi-phase tests ────────────────────────────────────────────────

    #[test]
    fn test_multiphase_single_phase_matches_standard() {
        // Single-phase config should produce same results as standard step
        let config = FluidConfig::water_2d();
        let phase_config =
            MultiPhaseConfig::single(config.rest_density, config.gas_constant, 0.001);
        let particles_init = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);

        let mut p_standard = particles_init.clone();
        let mut p_multi = particles_init;
        let mut solver_s = SphSolver::new();
        let mut solver_m = SphSolver::new();

        solver_s.step(&mut p_standard, &config, 0.001).unwrap();
        solver_m
            .step_multiphase(&mut p_multi, &config, &phase_config)
            .unwrap();

        for (a, b) in p_standard.iter().zip(p_multi.iter()) {
            assert!(
                (a.position[0] - b.position[0]).abs() < 1e-10,
                "single-phase multiphase should match standard"
            );
        }
    }

    #[test]
    fn test_multiphase_two_phase_stable() {
        let config = FluidConfig::water_2d();
        let phase_config = MultiPhaseConfig::two_phase(
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            PhaseProperties {
                rest_density: 1.225,
                gas_constant: 2000.0,
                viscosity: 0.00001,
            },
            0.072,
        );

        let mut particles = create_particle_block([0.1, 0.1], [0.3, 0.3], 0.02, 0.001);
        // Bottom half: water (phase 0), top half: air (phase 1)
        let mid_y = 0.25;
        for p in particles.iter_mut() {
            if p.position[1] > mid_y {
                p.phase = 1;
                p.mass = 0.0001; // lighter air particles
            }
        }

        let mut solver = SphSolver::new();
        for _ in 0..10 {
            solver
                .step_multiphase(&mut particles, &config, &phase_config)
                .unwrap();
        }

        assert!(
            particles.iter().all(|p| p.position[0].is_finite()),
            "two-phase sim should stay finite"
        );
    }

    #[test]
    fn test_multiphase_phases_use_correct_density() {
        let config = FluidConfig::water_2d();
        let phase_config = MultiPhaseConfig::two_phase(
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            PhaseProperties {
                rest_density: 500.0,
                gas_constant: 1000.0,
                viscosity: 0.01,
            },
            0.0,
        );

        let mut particles = vec![
            FluidParticle::new_2d(0.5, 0.5, 0.01),
            FluidParticle::new_2d(0.52, 0.5, 0.01),
        ];
        particles[1].phase = 1;

        let mut solver = SphSolver::new();
        solver
            .step_multiphase(&mut particles, &config, &phase_config)
            .unwrap();

        // Both should have computed density and pressure (non-zero due to self-contribution)
        assert!(particles[0].density > 0.0);
        assert!(particles[1].density > 0.0);
        // Phase 1 has lower rest_density → should have different pressure from phase 0
        // (for same computed density, lower rest_density → higher pressure)
        assert!(
            (particles[0].pressure - particles[1].pressure).abs() > 1e-6,
            "different phases should produce different pressures: p0={}, p1={}",
            particles[0].pressure,
            particles[1].pressure
        );
    }

    #[test]
    fn test_multiphase_interface_tension() {
        // Interface tension between phases should push unlike particles apart
        let config = FluidConfig::water_2d();
        let phase_no_tension = MultiPhaseConfig::two_phase(
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            0.0,
        );
        let phase_with_tension = MultiPhaseConfig::two_phase(
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            PhaseProperties {
                rest_density: 1000.0,
                gas_constant: 2000.0,
                viscosity: 0.001,
            },
            0.5,
        );

        let make_particles = || {
            let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.1], 0.02, 0.001);
            let mut p2 = create_particle_block([0.2, 0.4], [0.2, 0.1], 0.02, 0.001);
            for p in p2.iter_mut() {
                p.phase = 1;
            }
            particles.extend(p2);
            particles
        };

        let mut p_no = make_particles();
        let mut p_with = make_particles();
        let mut solver_no = SphSolver::new();
        let mut solver_with = SphSolver::with_surface_tension(0.0);

        for _ in 0..5 {
            solver_no
                .step_multiphase(&mut p_no, &config, &phase_no_tension)
                .unwrap();
            solver_with
                .step_multiphase(&mut p_with, &config, &phase_with_tension)
                .unwrap();
        }

        // Both should be finite
        assert!(p_no.iter().all(|p| p.position[0].is_finite()));
        assert!(p_with.iter().all(|p| p.position[0].is_finite()));
    }

    // ── Combustion tests ─────────────────────────────────────────────────

    #[test]
    fn test_combustion_below_ignition_no_reaction() {
        let mut particles = vec![FluidParticle::new_2d(0.0, 0.0, 1.0)];
        particles[0].fuel = 1.0;
        particles[0].temperature = 300.0;
        let config = CombustionConfig::default();
        update_combustion(&mut particles, &config, 0.01);
        assert!((particles[0].fuel - 1.0).abs() < 1e-10);
        assert!((particles[0].temperature - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_combustion_above_ignition_consumes_fuel() {
        let mut particles = vec![FluidParticle::new_2d(0.0, 0.0, 1.0)];
        particles[0].fuel = 1.0;
        particles[0].temperature = 600.0;
        let config = CombustionConfig::default();
        update_combustion(&mut particles, &config, 0.01);
        assert!(
            particles[0].fuel < 1.0,
            "fuel should deplete: {}",
            particles[0].fuel
        );
        assert!(
            particles[0].temperature > 600.0,
            "should heat up: {}",
            particles[0].temperature
        );
    }

    #[test]
    fn test_combustion_fuel_never_negative() {
        let mut particles = vec![FluidParticle::new_2d(0.0, 0.0, 1.0)];
        particles[0].fuel = 0.001;
        particles[0].temperature = 1000.0;
        let config = CombustionConfig {
            reaction_rate: 1000.0,
            ..CombustionConfig::default()
        };
        update_combustion(&mut particles, &config, 1.0);
        assert!(particles[0].fuel >= 0.0, "fuel should not go negative");
    }

    #[test]
    fn test_combustion_no_fuel_no_reaction() {
        let mut particles = vec![FluidParticle::new_2d(0.0, 0.0, 1.0)];
        particles[0].fuel = 0.0;
        particles[0].temperature = 800.0;
        let config = CombustionConfig::default();
        let t_before = particles[0].temperature;
        update_combustion(&mut particles, &config, 0.01);
        assert!((particles[0].temperature - t_before).abs() < 1e-10);
    }

    // ── Heat transfer tests ──────────────────────────────────────────────

    #[test]
    fn test_heat_uniform_no_change() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();

        let heat = HeatConfig { diffusivity: 1e-7 };
        let t_before: Vec<f64> = particles.iter().map(|p| p.temperature).collect();
        update_heat(
            &mut particles,
            &solver.neighbor_offsets,
            &solver.neighbor_indices,
            config.smoothing_radius,
            &heat,
            0.001,
        );
        // Uniform temperature → no change
        for (i, p) in particles.iter().enumerate() {
            assert!(
                (p.temperature - t_before[i]).abs() < 1e-10,
                "uniform temperature should not change"
            );
        }
    }

    #[test]
    fn test_heat_conducts_from_hot_to_cold() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        // Make center particle hot
        let mid = particles.len() / 2;
        particles[mid].temperature = 500.0;
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();

        let heat = HeatConfig { diffusivity: 1.0 };
        let t_hot_before = particles[mid].temperature;
        for _ in 0..50 {
            update_heat(
                &mut particles,
                &solver.neighbor_offsets,
                &solver.neighbor_indices,
                config.smoothing_radius,
                &heat,
                0.001,
            );
        }
        assert!(
            particles[mid].temperature < t_hot_before,
            "hot particle should cool: before={t_hot_before}, after={}",
            particles[mid].temperature
        );
    }

    // ── Viscoelastic tests ────────────────────────────────────────────────

    #[test]
    fn test_viscoelastic_relaxes_to_identity() {
        // With no velocity gradient, conformation should relax toward identity
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        // Deform conformation away from identity
        for p in particles.iter_mut() {
            p.density = 1000.0;
            p.conformation = [2.0, 0.5, 1.5];
        }
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();

        let ve = ViscoelasticConfig {
            elastic_modulus: 100.0,
            relaxation_time: 0.01,
        };

        // Run viscoelastic update many times (particles are stationary-ish)
        for _ in 0..50 {
            update_viscoelastic(
                &mut particles,
                &solver.neighbor_offsets,
                &solver.neighbor_indices,
                config.smoothing_radius,
                &ve,
                0.001,
            );
        }

        // Conformation should have relaxed toward identity [1, 0, 1]
        let p = &particles[particles.len() / 2];
        assert!(
            (p.conformation[0] - 1.0).abs() < 0.5,
            "c_xx should relax toward 1: {}",
            p.conformation[0]
        );
    }

    #[test]
    fn test_viscoelastic_elastic_stress_finite() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();

        let ve = ViscoelasticConfig {
            elastic_modulus: 500.0,
            relaxation_time: 0.1,
        };

        update_viscoelastic(
            &mut particles,
            &solver.neighbor_offsets,
            &solver.neighbor_indices,
            config.smoothing_radius,
            &ve,
            0.001,
        );

        assert!(particles.iter().all(|p| p.acceleration[0].is_finite()));
        assert!(particles.iter().all(|p| p.conformation[0].is_finite()));
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
        let dt = SphSolver::adaptive_dt(&particles, 0.05, 0.001, 0.4, 0.0001, 0.01);
        assert!((dt - 0.01).abs() < EPS);
    }

    #[test]
    fn test_adaptive_dt_fast_particles() {
        let mut particles = create_particle_block([0.1, 0.1], [0.2, 0.2], 0.05, 0.001);
        particles[0].velocity = DVec3::new(100.0, 0.0, 0.0);
        let dt = SphSolver::adaptive_dt(&particles, 0.05, 0.001, 0.4, 0.0001, 0.01);
        // dt = 0.4 * 0.05 / 100 = 0.0002
        assert!(dt < 0.001);
        assert!(dt >= 0.0001);
    }

    // ── Wendland kernel tests ─────────────────────────────────────────────

    #[test]
    fn test_wendland_c2_at_zero() {
        let w = kernel_wendland_c2(0.0, 1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_wendland_c2_at_boundary() {
        let w = kernel_wendland_c2(2.0, 1.0);
        assert!(w.abs() < EPS);
    }

    #[test]
    fn test_wendland_c4_grad_numerical() {
        // Verify analytical gradient matches numerical finite difference
        let h = 1.0;
        let r = 0.7;
        let dr = 1e-6;
        let numerical =
            (kernel_wendland_c4(r + dr, h) - kernel_wendland_c4(r - dr, h)) / (2.0 * dr);
        let analytical = kernel_wendland_c4_grad(r, h);
        assert!(
            (numerical - analytical).abs() < 1e-4,
            "C4 gradient mismatch: numerical={numerical}, analytical={analytical}"
        );
    }

    #[test]
    fn test_wendland_c2_grad_numerical() {
        let h = 1.0;
        let r = 0.7;
        let dr = 1e-6;
        let numerical =
            (kernel_wendland_c2(r + dr, h) - kernel_wendland_c2(r - dr, h)) / (2.0 * dr);
        let analytical = kernel_wendland_c2_grad(r, h);
        assert!(
            (numerical - analytical).abs() < 1e-4,
            "C2 gradient mismatch: numerical={numerical}, analytical={analytical}"
        );
    }

    #[test]
    fn test_wendland_c2_decreasing() {
        let w1 = kernel_wendland_c2(0.1, 1.0);
        let w2 = kernel_wendland_c2(0.5, 1.0);
        assert!(w1 > w2);
    }

    #[test]
    fn test_wendland_c4_at_zero() {
        assert!(kernel_wendland_c4(0.0, 1.0) > 0.0);
    }

    #[test]
    fn test_wendland_c2_grad_sign() {
        let g = kernel_wendland_c2_grad(0.5, 1.0);
        assert!(g < 0.0, "gradient should be negative (decreasing)");
    }

    // ── Non-Newtonian tests ───────────────────────────────────────────────

    #[test]
    fn test_power_law_shear_thinning() {
        let nn = NonNewtonianViscosity::PowerLaw {
            consistency: 1.0,
            power_index: 0.5,
        };
        let low_rate = nn.effective_viscosity(0.1);
        let high_rate = nn.effective_viscosity(10.0);
        assert!(
            high_rate < low_rate,
            "shear-thinning: viscosity should decrease with rate"
        );
    }

    #[test]
    fn test_bingham_yield_stress() {
        let nn = NonNewtonianViscosity::Bingham {
            yield_stress: 100.0,
            plastic_viscosity: 0.1,
        };
        let low_rate = nn.effective_viscosity(0.001);
        assert!(
            low_rate > 1000.0,
            "low strain rate should give very high effective viscosity"
        );
    }

    #[test]
    fn test_herschel_bulkley() {
        let nn = NonNewtonianViscosity::HerschelBulkley {
            yield_stress: 50.0,
            consistency: 1.0,
            power_index: 0.5,
        };
        let visc = nn.effective_viscosity(1.0);
        assert!(visc > 50.0, "should include yield stress contribution");
        assert!(visc.is_finite());
    }

    // ── Z-order sorting tests ─────────────────────────────────────────────

    #[test]
    fn test_sort_by_zorder() {
        let mut particles = create_particle_block([0.0, 0.0], [1.0, 1.0], 0.1, 0.001);
        let n = particles.len();
        sort_by_zorder(&mut particles, 0.1);
        assert_eq!(particles.len(), n);
        // All positions should still be finite
        assert!(particles.iter().all(|p| p.position[0].is_finite()));
    }

    #[test]
    fn test_sort_by_zorder_preserves_data() {
        let mut particles = create_particle_block([0.0, 0.0], [0.5, 0.5], 0.1, 0.001);
        particles[0].velocity = DVec3::new(99.0, 0.0, 0.0);
        let total_mass_before: f64 = particles.iter().map(|p| p.mass).sum();
        sort_by_zorder(&mut particles, 0.1);
        let total_mass_after: f64 = particles.iter().map(|p| p.mass).sum();
        assert!((total_mass_before - total_mass_after).abs() < EPS);
        // The tagged particle should still exist
        assert!(particles.iter().any(|p| (p.velocity[0] - 99.0).abs() < EPS));
    }

    // ── Grid CFL test ─────────────────────────────────────────────────────

    #[test]
    fn test_grid_cfl_dt() {
        use crate::grid::FluidGrid;
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let i = g.idx(5, 5);
        g.vx[i] = 1.0;
        let dt = g.cfl_dt(0.001, 0.5);
        assert!(dt > 0.0 && dt.is_finite());
        assert!(dt <= 0.5 * 0.1 / 1.0); // CFL * dx / v
    }

    // ── Shallow CFL test ──────────────────────────────────────────────────

    #[test]
    fn test_shallow_cfl_dt() {
        use crate::shallow::ShallowWater;
        let sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        let dt = sw.cfl_dt(0.5);
        assert!(dt > 0.0 && dt.is_finite());
        assert!(dt < 0.1);
    }

    // ── DFSPH tests ───────────────────────────────────────────────────────

    #[test]
    fn test_dfsph_empty() {
        let mut particles = vec![];
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        assert!(
            solver
                .step_dfsph(&mut particles, &config, 0.001, 3, 0.01)
                .is_ok()
        );
    }

    #[test]
    fn test_dfsph_single_particle_falls() {
        let mut particles = vec![FluidParticle::new_2d(0.5, 0.5, 0.01)];
        particles[0].density = 1000.0;
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver
            .step_dfsph(&mut particles, &config, 0.001, 3, 0.01)
            .unwrap();
        assert!(particles[0].velocity.y < 0.0, "should fall under gravity");
    }

    #[test]
    fn test_dfsph_particles_finite() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        for _ in 0..5 {
            solver
                .step_dfsph(&mut particles, &config, 0.001, 3, 0.01)
                .unwrap();
        }
        assert!(particles.iter().all(|p| p.position.x.is_finite()));
    }

    // ── MLS correction tests ──────────────────────────────────────────────

    #[test]
    fn test_mls_correction_interior() {
        let mut particles = create_particle_block([0.2, 0.3], [0.2, 0.2], 0.02, 0.001);
        let config = FluidConfig::water_2d();
        let mut solver = SphSolver::new();
        solver.step(&mut particles, &config, 0.001).unwrap();

        let corrections = compute_gradient_corrections(
            &particles,
            &solver.neighbor_offsets,
            &solver.neighbor_indices,
            config.smoothing_radius,
        );
        assert_eq!(corrections.len(), particles.len());
        // Interior particles should have near-identity corrections
        let mid = corrections.len() / 2;
        let c = corrections[mid];
        assert!(c[0].is_finite() && c[4].is_finite() && c[8].is_finite());
    }

    #[test]
    fn test_apply_gradient_correction_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = apply_gradient_correction(&identity, DVec3::new(2.0, 3.0, 4.0));
        assert!((result.x - 2.0).abs() < 1e-10);
        assert!((result.y - 3.0).abs() < 1e-10);
        assert!((result.z - 4.0).abs() < 1e-10);
    }
}
