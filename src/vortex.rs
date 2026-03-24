//! Vortex dynamics — vorticity, circulation, turbulence.

use std::f64::consts::PI;

/// Vorticity from a 2D velocity field (curl of velocity).
///
/// ω = ∂vy/∂x - ∂vx/∂y
#[inline]
#[must_use]
pub fn vorticity_2d(dvydx: f64, dvxdy: f64) -> f64 {
    dvydx - dvxdy
}

/// Lamb-Oseen vortex — decaying vortex with viscous core.
///
/// Returns tangential velocity at radius r from center.
/// Γ = circulation, ν = kinematic viscosity, t = time since creation.
#[inline]
#[must_use]
pub fn lamb_oseen_velocity(circulation: f64, radius: f64, viscosity: f64, time: f64) -> f64 {
    if radius <= 0.0 || time <= 0.0 {
        return 0.0;
    }
    let r2 = radius * radius;
    let core = 4.0 * viscosity * time;
    (circulation / (2.0 * PI * radius)) * (1.0 - (-r2 / core).exp())
}

/// Rankine vortex — solid body rotation inside core, irrotational outside.
///
/// v(r) = Γ·r/(2π·R²)  for r ≤ R (solid body)
/// v(r) = Γ/(2π·r)      for r > R (irrotational)
#[inline]
#[must_use]
pub fn rankine_velocity(circulation: f64, radius: f64, core_radius: f64) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    if radius <= core_radius {
        circulation * radius / (2.0 * PI * core_radius * core_radius)
    } else {
        circulation / (2.0 * PI * radius)
    }
}

/// Enstrophy — measure of total vorticity (integral of ω²).
///
/// For a discrete 2D field: Σ ω² × dx²
#[must_use]
pub fn enstrophy(vorticity_field: &[f64], dx: f64) -> f64 {
    let dx2 = dx * dx;
    vorticity_field.iter().map(|&w| w * w * dx2).sum()
}

/// Kolmogorov microscale — smallest turbulent eddy size.
///
/// η = (ν³/ε)^(1/4)
/// where ε = turbulent dissipation rate.
#[inline]
#[must_use]
pub fn kolmogorov_scale(viscosity: f64, dissipation_rate: f64) -> f64 {
    if dissipation_rate <= 0.0 {
        return f64::INFINITY;
    }
    (viscosity.powi(3) / dissipation_rate).powf(0.25)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_vorticity_2d() {
        assert!((vorticity_2d(1.0, 0.0) - 1.0).abs() < EPS);
        assert!((vorticity_2d(0.0, 1.0) - (-1.0)).abs() < EPS);
        assert!((vorticity_2d(2.0, 2.0)).abs() < EPS);
    }

    #[test]
    fn test_lamb_oseen_far_field() {
        let v = lamb_oseen_velocity(1.0, 10.0, 0.001, 1.0);
        // Far from core, approaches Γ/(2πr)
        let irrotational = 1.0 / (2.0 * PI * 10.0);
        assert!((v - irrotational).abs() < 0.01);
    }

    #[test]
    fn test_lamb_oseen_zero_radius() {
        assert!(lamb_oseen_velocity(1.0, 0.0, 0.001, 1.0).abs() < EPS);
    }

    #[test]
    fn test_lamb_oseen_zero_time() {
        assert!(lamb_oseen_velocity(1.0, 1.0, 0.001, 0.0).abs() < EPS);
    }

    #[test]
    fn test_rankine_inside_core() {
        let v = rankine_velocity(2.0 * PI, 0.5, 1.0);
        // Inside: v = Γr/(2πR²) = 2π·0.5/(2π·1) = 0.5
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn test_rankine_outside_core() {
        let v = rankine_velocity(2.0 * PI, 2.0, 1.0);
        // Outside: v = Γ/(2πr) = 2π/(2π·2) = 0.5
        assert!((v - 0.5).abs() < EPS);
    }

    #[test]
    fn test_rankine_continuous_at_core() {
        let gamma = 5.0;
        let r_core = 1.0;
        let v_inside = rankine_velocity(gamma, r_core, r_core);
        let v_outside = rankine_velocity(gamma, r_core + EPS, r_core);
        assert!((v_inside - v_outside).abs() < 0.01);
    }

    #[test]
    fn test_enstrophy_zero() {
        assert!(enstrophy(&[0.0, 0.0, 0.0], 0.1).abs() < EPS);
    }

    #[test]
    fn test_enstrophy_positive() {
        let field = vec![1.0, -1.0, 2.0, 0.5];
        let e = enstrophy(&field, 0.1);
        assert!(e > 0.0);
    }

    #[test]
    fn test_kolmogorov_scale() {
        let eta = kolmogorov_scale(1e-6, 0.01);
        assert!(eta > 0.0);
        assert!(eta.is_finite());
    }

    #[test]
    fn test_kolmogorov_zero_dissipation() {
        assert!(kolmogorov_scale(1e-6, 0.0).is_infinite());
    }
}
