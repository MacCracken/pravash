//! Buoyancy and drag forces — fluid-body interaction.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

/// Buoyancy force on a submerged body (Archimedes' principle).
///
/// F = ρ_fluid × g × V_displaced
#[inline]
#[must_use]
pub fn buoyancy_force(fluid_density: f64, gravity: f64, displaced_volume: f64) -> f64 {
    fluid_density * gravity * displaced_volume
}

/// Drag force magnitude on a body moving through fluid.
///
/// F = 0.5 × ρ × v² × Cd × A
///
/// Returns the magnitude of the drag force (always non-negative).
/// The caller is responsible for applying the force in the correct direction
/// (opposing the velocity).
#[inline]
#[must_use]
pub fn drag_force(
    fluid_density: f64,
    velocity: f64,
    drag_coefficient: f64,
    cross_section_area: f64,
) -> f64 {
    0.5 * fluid_density * velocity * velocity * drag_coefficient * cross_section_area
}

/// Common drag coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DragCoefficient;

impl DragCoefficient {
    pub const SPHERE: f64 = 0.47;
    pub const CUBE: f64 = 1.05;
    pub const CYLINDER: f64 = 0.82;
    pub const STREAMLINED: f64 = 0.04;
    pub const FLAT_PLATE: f64 = 1.28;
}

/// Terminal velocity — when drag equals gravitational force.
///
/// v_t = sqrt(2·m·g / (ρ·Cd·A))
///
/// Returns an error if any denominator term is zero or negative, or if
/// the argument to sqrt would be negative.
#[inline]
pub fn terminal_velocity(
    mass: f64,
    gravity: f64,
    fluid_density: f64,
    drag_coefficient: f64,
    area: f64,
) -> Result<f64> {
    if !gravity.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: "terminal_velocity: gravity must be finite".into(),
        });
    }
    let denom = fluid_density * drag_coefficient * area;
    if denom <= 0.0 {
        return Err(PravashError::InvalidParameter {
            reason: "terminal_velocity denominator (ρ·Cd·A) must be positive".into(),
        });
    }
    let arg = 2.0 * mass * gravity / denom;
    if arg < 0.0 {
        return Err(PravashError::InvalidParameter {
            reason: "terminal_velocity: m·g must be non-negative".into(),
        });
    }
    Ok(arg.sqrt())
}

/// Reynolds number — determines flow regime (laminar vs turbulent).
///
/// Re = ρ·v·L / μ
///
/// Returns an error if viscosity is zero (inviscid limit).
#[inline]
pub fn reynolds_number(
    fluid_density: f64,
    velocity: f64,
    length: f64,
    viscosity: f64,
) -> Result<f64> {
    if viscosity <= 0.0 {
        return Err(PravashError::InvalidViscosity { viscosity });
    }
    if !fluid_density.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: "reynolds_number: fluid_density must be finite".into(),
        });
    }
    if !velocity.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: "reynolds_number: velocity must be finite".into(),
        });
    }
    if !length.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: "reynolds_number: length must be finite".into(),
        });
    }
    Ok(fluid_density * velocity * length / viscosity)
}

/// Flow regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FlowRegime {
    Laminar,
    Transitional,
    Turbulent,
}

/// Classify flow regime from Reynolds number.
#[inline]
#[must_use]
pub fn classify_flow(re: f64) -> FlowRegime {
    if re < 2300.0 {
        FlowRegime::Laminar
    } else if re < 4000.0 {
        FlowRegime::Transitional
    } else {
        FlowRegime::Turbulent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buoyancy_water() {
        // 1 m³ of water displaced, density 1000, g=9.81
        let f = buoyancy_force(1000.0, 9.81, 1.0);
        assert!((f - 9810.0).abs() < 0.1);
    }

    #[test]
    fn test_buoyancy_zero_volume() {
        assert!(buoyancy_force(1000.0, 9.81, 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drag_force() {
        let f = drag_force(1.225, 10.0, DragCoefficient::SPHERE, 0.01);
        assert!(f > 0.0);
    }

    #[test]
    fn test_drag_zero_velocity() {
        assert!(drag_force(1.225, 0.0, DragCoefficient::SPHERE, 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_terminal_velocity() {
        let vt = terminal_velocity(1.0, 9.81, 1.225, DragCoefficient::SPHERE, 0.01).unwrap();
        assert!(vt > 0.0);
        assert!(vt.is_finite());
    }

    #[test]
    fn test_terminal_velocity_zero_denom() {
        assert!(terminal_velocity(1.0, 9.81, 0.0, DragCoefficient::SPHERE, 0.01).is_err());
    }

    #[test]
    fn test_terminal_velocity_negative_gravity() {
        assert!(terminal_velocity(1.0, -9.81, 1.225, DragCoefficient::SPHERE, 0.01).is_err());
    }

    #[test]
    fn test_reynolds_number() {
        // Water flowing at 1 m/s through 0.1m pipe
        let re = reynolds_number(1000.0, 1.0, 0.1, 0.001).unwrap();
        assert!((re - 100000.0).abs() < 1.0);
    }

    #[test]
    fn test_reynolds_number_zero_viscosity() {
        assert!(reynolds_number(1000.0, 1.0, 0.1, 0.0).is_err());
    }

    #[test]
    fn test_flow_regime_laminar() {
        assert_eq!(classify_flow(1000.0), FlowRegime::Laminar);
    }

    #[test]
    fn test_flow_regime_transitional() {
        assert_eq!(classify_flow(3000.0), FlowRegime::Transitional);
    }

    #[test]
    fn test_flow_regime_turbulent() {
        assert_eq!(classify_flow(10000.0), FlowRegime::Turbulent);
    }

    #[test]
    fn test_drag_coefficients() {
        const { assert!(DragCoefficient::STREAMLINED < DragCoefficient::SPHERE) };
        const { assert!(DragCoefficient::SPHERE < DragCoefficient::FLAT_PLATE) };
    }
}
