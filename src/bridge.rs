//! Cross-crate bridges — convert primitive values from other AGNOS science crates
//! into pravash fluid dynamics parameters and vice versa.
//!
//! Always available — takes primitive values (f64), no science crate deps.
//!
//! # Architecture
//!
//! ```text
//! pavan (aerodynamics) ──┐
//! ushma (thermodynamics) ┼──> bridge ──> pravash fluid parameters
//! goonj (acoustics)      ┘
//! ```

// ── Pavan bridges (aerodynamics) ───────────────────────────────────────────

/// Convert airfoil surface pressure coefficient (Cp) and free-stream
/// dynamic pressure (Pa) to surface pressure (Pa).
///
/// P = P_inf + Cp × q_inf, where q_inf = 0.5 × ρ × V².
/// Returns absolute surface pressure in Pa.
#[must_use]
#[inline]
pub fn cp_to_surface_pressure(cp: f64, dynamic_pressure_pa: f64, static_pressure_pa: f64) -> f64 {
    static_pressure_pa + cp * dynamic_pressure_pa
}

/// Convert free-stream velocity (m/s) and air density (kg/m³)
/// to dynamic pressure (Pa).
///
/// q = 0.5 × ρ × V²
#[must_use]
#[inline]
pub fn freestream_dynamic_pressure(velocity_ms: f64, density_kg_m3: f64) -> f64 {
    0.5 * density_kg_m3 * velocity_ms * velocity_ms
}

/// Convert aerodynamic lift and drag coefficients to force magnitudes (N).
///
/// F = C × q × S, where q is dynamic pressure and S is reference area.
/// Returns `(lift_n, drag_n)`.
#[must_use]
#[inline]
pub fn aero_coefficients_to_forces(
    cl: f64,
    cd: f64,
    dynamic_pressure_pa: f64,
    reference_area_m2: f64,
) -> (f64, f64) {
    let lift = cl * dynamic_pressure_pa * reference_area_m2;
    let drag = cd * dynamic_pressure_pa * reference_area_m2;
    (lift, drag)
}

/// Convert wind field velocity (m/s) to a free-stream boundary condition
/// for fluid simulation.
///
/// Returns the velocity clamped to a physically reasonable range.
#[must_use]
#[inline]
pub fn wind_to_freestream(wind_velocity_ms: f64) -> f64 {
    wind_velocity_ms.clamp(-340.0, 340.0) // subsonic limit
}

// ── Ushma bridges (thermodynamics) ─────────────────────────────────────────

/// Convert fluid temperature (K) to dynamic viscosity (Pa·s) for water
/// using the Andrade equation approximation.
///
/// μ(T) = A × exp(B / T), with water constants.
/// Valid range: 273–373 K.
#[must_use]
#[inline]
pub fn temperature_to_water_viscosity(temperature_k: f64) -> f64 {
    let t = temperature_k.clamp(273.0, 373.0);
    // Andrade coefficients for water
    let a = 2.414e-5;
    let b = 247.8;
    let c = 140.0;
    a * 10.0_f64.powf(b / (t - c))
}

/// Convert heat source power (W) and fluid volume (m³) to volumetric
/// thermal buoyancy forcing (N/m³).
///
/// Buoyancy force per unit volume: f_b = ρ × g × β × ΔT
/// where ΔT = Q / (ρ × c_p × V) for a single timestep.
///
/// `heat_power_w`: heat source power in watts.
/// `fluid_volume_m3`: volume of affected fluid.
/// `density_kg_m3`: fluid density.
/// `specific_heat_j_per_kg_k`: fluid specific heat.
/// `expansion_coeff_per_k`: thermal expansion coefficient (water ≈ 2.1e-4 /K).
/// `dt_s`: timestep in seconds.
#[must_use]
pub fn heat_to_buoyancy_forcing(
    heat_power_w: f64,
    fluid_volume_m3: f64,
    density_kg_m3: f64,
    specific_heat_j_per_kg_k: f64,
    expansion_coeff_per_k: f64,
    dt_s: f64,
) -> f64 {
    if fluid_volume_m3 <= 0.0 || specific_heat_j_per_kg_k <= 0.0 {
        return 0.0;
    }
    let delta_t =
        heat_power_w * dt_s / (density_kg_m3 * specific_heat_j_per_kg_k * fluid_volume_m3);
    density_kg_m3 * 9.81 * expansion_coeff_per_k * delta_t
}

/// Convert turbulent kinetic energy (m²/s²) to eddy thermal diffusivity (m²/s)
/// using the Reynolds analogy.
///
/// α_t ≈ ν_t / Pr_t, where ν_t ≈ C_μ × k² / ε.
/// Simplified: α_t ≈ C × sqrt(k) × L, with default turbulent Prandtl = 0.85.
/// `tke`: turbulent kinetic energy (m²/s²).
/// `turbulent_length_scale_m`: characteristic eddy size.
#[must_use]
#[inline]
pub fn tke_to_eddy_diffusivity(tke: f64, turbulent_length_scale_m: f64) -> f64 {
    if tke <= 0.0 {
        return 0.0;
    }
    // ν_t ≈ C_μ^(1/4) × sqrt(k) × L, then α_t = ν_t / Pr_t
    let c_mu_quarter = 0.5477; // C_μ^(1/4) ≈ 0.09^0.25
    let pr_t = 0.85;
    c_mu_quarter * tke.sqrt() * turbulent_length_scale_m / pr_t
}

// ── Goonj bridges (acoustics) ──────────────────────────────────────────────

/// Convert fluid velocity field magnitude (m/s) to acoustic source strength
/// for aeroacoustic analysis (Lighthill's analogy).
///
/// Acoustic power scales as P ∝ ρ × V⁸ / c⁵ for free turbulence (Lighthill 8th power law).
/// Returns a dimensionless intensity scaling factor (relative to reference).
#[must_use]
#[inline]
pub fn velocity_to_acoustic_source_strength(velocity_ms: f64, speed_of_sound: f64) -> f64 {
    if speed_of_sound <= 0.0 {
        return 0.0;
    }
    let mach = velocity_ms.abs() / speed_of_sound;
    mach.powi(8) // Lighthill 8th power law
}

/// Convert turbulent kinetic energy (m²/s²) to estimated broadband noise
/// level in dB SPL.
///
/// Empirical: SPL ≈ 10 × log10(ρ × k × V / (p_ref² / (ρ × c)))
/// Simplified scaling: SPL ≈ 20 × log10(sqrt(2/3 × k) / v_ref) + offset.
/// `tke`: turbulent kinetic energy (m²/s²).
/// Returns approximate dB SPL (reference: 20 μPa).
#[must_use]
#[inline]
pub fn tke_to_broadband_noise_db(tke: f64) -> f64 {
    if tke <= 0.0 {
        return 0.0;
    }
    // Turbulent velocity fluctuation
    let u_rms = (2.0 / 3.0 * tke).sqrt();
    // Reference velocity for 0 dB SPL ≈ 5e-8 m/s (in air)
    let v_ref = 5e-8;
    if u_rms <= v_ref {
        return 0.0;
    }
    20.0 * (u_rms / v_ref).log10()
}

/// Convert fluid surface wave height (m) and frequency (Hz) to estimated
/// acoustic pressure contribution (Pa) at the surface.
///
/// Simplified dipole radiation model: p ≈ ρ × g × h × (k × a)
/// where k = 2πf/c, a is a characteristic length.
#[must_use]
#[inline]
pub fn wave_height_to_acoustic_pressure(
    wave_height_m: f64,
    frequency_hz: f64,
    speed_of_sound: f64,
) -> f64 {
    if speed_of_sound <= 0.0 || frequency_hz <= 0.0 {
        return 0.0;
    }
    let rho = 1.225; // air density
    let g = 9.81;
    let k = 2.0 * std::f64::consts::PI * frequency_hz / speed_of_sound;
    rho * g * wave_height_m.abs() * k * wave_height_m.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pavan bridges ──────────────────────────────────────────────────

    #[test]
    fn cp_to_pressure_stagnation() {
        // Cp = 1.0 at stagnation point
        let p = cp_to_surface_pressure(1.0, 500.0, 101325.0);
        assert!((p - 101825.0).abs() < 0.1);
    }

    #[test]
    fn dynamic_pressure_basic() {
        // 10 m/s in air (1.225 kg/m³) → q = 61.25 Pa
        let q = freestream_dynamic_pressure(10.0, 1.225);
        assert!((q - 61.25).abs() < 0.01);
    }

    #[test]
    fn aero_forces_basic() {
        let (lift, drag) = aero_coefficients_to_forces(1.0, 0.05, 100.0, 10.0);
        assert!((lift - 1000.0).abs() < 0.1);
        assert!((drag - 50.0).abs() < 0.1);
    }

    #[test]
    fn wind_to_freestream_clamps() {
        assert_eq!(wind_to_freestream(500.0), 340.0);
        assert_eq!(wind_to_freestream(-500.0), -340.0);
    }

    // ── Ushma bridges ──────────────────────────────────────────────────

    #[test]
    fn water_viscosity_room_temp() {
        // Water at 293K (~20°C) ≈ 0.001 Pa·s
        let mu = temperature_to_water_viscosity(293.0);
        assert!(mu > 0.0005 && mu < 0.002, "got {mu}");
    }

    #[test]
    fn water_viscosity_decreases_with_temp() {
        let cold = temperature_to_water_viscosity(280.0);
        let hot = temperature_to_water_viscosity(350.0);
        assert!(cold > hot);
    }

    #[test]
    fn buoyancy_forcing_positive() {
        let f = heat_to_buoyancy_forcing(1000.0, 1.0, 1000.0, 4186.0, 2.1e-4, 1.0);
        assert!(f > 0.0);
    }

    #[test]
    fn buoyancy_forcing_zero_volume() {
        assert_eq!(
            heat_to_buoyancy_forcing(1000.0, 0.0, 1000.0, 4186.0, 2.1e-4, 1.0),
            0.0
        );
    }

    #[test]
    fn eddy_diffusivity_zero_tke() {
        assert_eq!(tke_to_eddy_diffusivity(0.0, 1.0), 0.0);
    }

    #[test]
    fn eddy_diffusivity_positive() {
        let alpha = tke_to_eddy_diffusivity(1.0, 0.1);
        assert!(alpha > 0.0);
    }

    // ── Goonj bridges ──────────────────────────────────────────────────

    #[test]
    fn acoustic_source_strength_subsonic() {
        let s = velocity_to_acoustic_source_strength(34.3, 343.0); // Mach 0.1
        assert!((s - 1e-8).abs() < 1e-9); // 0.1^8 = 1e-8
    }

    #[test]
    fn acoustic_source_zero_velocity() {
        assert_eq!(velocity_to_acoustic_source_strength(0.0, 343.0), 0.0);
    }

    #[test]
    fn tke_noise_zero() {
        assert_eq!(tke_to_broadband_noise_db(0.0), 0.0);
    }

    #[test]
    fn tke_noise_increases_with_tke() {
        let low = tke_to_broadband_noise_db(1.0);
        let high = tke_to_broadband_noise_db(10.0);
        assert!(high > low);
    }

    #[test]
    fn wave_acoustic_pressure_positive() {
        let p = wave_height_to_acoustic_pressure(0.1, 100.0, 343.0);
        assert!(p > 0.0);
    }

    #[test]
    fn wave_acoustic_pressure_zero_freq() {
        assert_eq!(wave_height_to_acoustic_pressure(0.1, 0.0, 343.0), 0.0);
    }
}
