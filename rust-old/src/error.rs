//! Error types for pravash.

use std::borrow::Cow;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PravashError {
    #[error("invalid particle count: {count} (must be > 0)")]
    InvalidParticleCount { count: usize },

    #[error("invalid grid resolution: {nx}x{ny} (must be > 0)")]
    InvalidGridResolution { nx: usize, ny: usize },

    #[error("invalid timestep: {dt} (must be > 0)")]
    InvalidTimestep { dt: f64 },

    #[error("density out of range: {density} kg/m³ (must be > 0)")]
    InvalidDensity { density: f64 },

    #[error("viscosity out of range: {viscosity} (must be >= 0)")]
    InvalidViscosity { viscosity: f64 },

    #[error("smoothing radius must be positive: {h}")]
    InvalidSmoothingRadius { h: f64 },

    #[error(
        "CFL condition violated: velocity {velocity} exceeds stable limit for dx={dx}, dt={dt}"
    )]
    CflViolation { velocity: f64, dx: f64, dt: f64 },

    #[error("simulation diverged: {reason}")]
    Diverged { reason: Cow<'static, str> },

    #[error("invalid parameter: {reason}")]
    InvalidParameter { reason: Cow<'static, str> },
}

pub type Result<T> = std::result::Result<T, PravashError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_particle_count() {
        let e = PravashError::InvalidParticleCount { count: 0 };
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn test_invalid_grid() {
        let e = PravashError::InvalidGridResolution { nx: 0, ny: 10 };
        assert!(e.to_string().contains("0x10"));
    }

    #[test]
    fn test_cfl_violation() {
        let e = PravashError::CflViolation {
            velocity: 100.0,
            dx: 0.01,
            dt: 0.001,
        };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn test_diverged() {
        let e = PravashError::Diverged {
            reason: "NaN in pressure".into(),
        };
        assert!(e.to_string().contains("NaN"));
    }

    #[test]
    fn test_diverged_static() {
        let e = PravashError::Diverged {
            reason: Cow::Borrowed("static reason"),
        };
        assert!(e.to_string().contains("static"));
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<f64> = Ok(1.0);
        assert!(ok.is_ok());
    }
}
