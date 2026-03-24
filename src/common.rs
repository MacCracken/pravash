//! Shared fluid types — particles, materials, configuration.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

/// Physical properties of a fluid material.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FluidMaterial {
    /// Rest density in kg/m³ (water ≈ 1000).
    pub density: f64,
    /// Dynamic viscosity in Pa·s (water ≈ 0.001).
    pub viscosity: f64,
    /// Surface tension coefficient in N/m (water ≈ 0.072).
    pub surface_tension: f64,
    /// Speed of sound in the fluid (for pressure computation).
    pub speed_of_sound: f64,
}

impl FluidMaterial {
    pub const WATER: Self = Self {
        density: 1000.0,
        viscosity: 0.001,
        surface_tension: 0.072,
        speed_of_sound: 1480.0,
    };

    pub const OIL: Self = Self {
        density: 900.0,
        viscosity: 0.03,
        surface_tension: 0.03,
        speed_of_sound: 1200.0,
    };

    pub const HONEY: Self = Self {
        density: 1400.0,
        viscosity: 2.0,
        surface_tension: 0.06,
        speed_of_sound: 1500.0,
    };

    pub const AIR: Self = Self {
        density: 1.225,
        viscosity: 0.0000181,
        surface_tension: 0.0,
        speed_of_sound: 343.0,
    };

    pub const LAVA: Self = Self {
        density: 2700.0,
        viscosity: 100.0,
        surface_tension: 0.4,
        speed_of_sound: 2000.0,
    };

    /// Custom material with validation.
    pub fn custom(density: f64, viscosity: f64, surface_tension: f64, speed_of_sound: f64) -> Result<Self> {
        if density <= 0.0 {
            return Err(PravashError::InvalidDensity { density });
        }
        if viscosity < 0.0 {
            return Err(PravashError::InvalidViscosity { viscosity });
        }
        Ok(Self { density, viscosity, surface_tension, speed_of_sound })
    }
}

/// A fluid particle (SPH).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidParticle {
    /// Position [x, y] (2D) or [x, y, z] (3D — z=0 for 2D).
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz].
    pub velocity: [f64; 3],
    /// Acceleration (accumulated forces / mass).
    pub acceleration: [f64; 3],
    /// Particle density (computed from neighbors).
    pub density: f64,
    /// Pressure (derived from density via equation of state).
    pub pressure: f64,
    /// Mass of this particle.
    pub mass: f64,
}

impl FluidParticle {
    /// Create a particle at rest.
    pub fn new(position: [f64; 3], mass: f64) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            acceleration: [0.0; 3],
            density: 0.0,
            pressure: 0.0,
            mass,
        }
    }

    /// Create a 2D particle (z = 0).
    #[inline]
    pub fn new_2d(x: f64, y: f64, mass: f64) -> Self {
        Self::new([x, y, 0.0], mass)
    }

    /// Speed (magnitude of velocity).
    #[inline]
    pub fn speed(&self) -> f64 {
        let v = &self.velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Kinetic energy: 0.5 * m * v².
    #[inline]
    pub fn kinetic_energy(&self) -> f64 {
        let v2 = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        0.5 * self.mass * v2
    }

    /// Distance to another particle.
    #[inline]
    pub fn distance_to(&self, other: &FluidParticle) -> f64 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidConfig {
    /// Timestep in seconds.
    pub dt: f64,
    /// Gravity vector [gx, gy, gz].
    pub gravity: [f64; 3],
    /// SPH smoothing radius (particle interaction distance).
    pub smoothing_radius: f64,
    /// Gas constant for equation of state (pressure from density).
    pub gas_constant: f64,
    /// Rest density of the fluid.
    pub rest_density: f64,
    /// Domain bounds [min_x, min_y, min_z, max_x, max_y, max_z].
    pub bounds: [f64; 6],
    /// Boundary damping (velocity multiplier on wall collision, 0-1).
    pub boundary_damping: f64,
}

impl FluidConfig {
    /// Default 2D water simulation config.
    pub fn water_2d() -> Self {
        Self {
            dt: 0.001,
            gravity: [0.0, -9.81, 0.0],
            smoothing_radius: 0.05,
            gas_constant: 2000.0,
            rest_density: FluidMaterial::WATER.density,
            bounds: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            boundary_damping: 0.5,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt: self.dt });
        }
        if self.smoothing_radius <= 0.0 {
            return Err(PravashError::InvalidSmoothingRadius { h: self.smoothing_radius });
        }
        if self.rest_density <= 0.0 {
            return Err(PravashError::InvalidDensity { density: self.rest_density });
        }
        Ok(())
    }

    /// CFL stability check: dt must be small enough for the given velocity.
    #[inline]
    pub fn check_cfl(&self, max_velocity: f64) -> Result<()> {
        let dx = self.smoothing_radius;
        let cfl = max_velocity * self.dt / dx;
        if cfl > 1.0 {
            return Err(PravashError::CflViolation {
                velocity: max_velocity,
                dx,
                dt: self.dt,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_material() {
        assert!((FluidMaterial::WATER.density - 1000.0).abs() < f64::EPSILON);
        assert!(FluidMaterial::WATER.viscosity > 0.0);
    }

    #[test]
    fn test_custom_material_valid() {
        let m = FluidMaterial::custom(800.0, 0.01, 0.05, 1400.0).unwrap();
        assert!((m.density - 800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_material_invalid_density() {
        assert!(FluidMaterial::custom(-1.0, 0.01, 0.05, 1400.0).is_err());
        assert!(FluidMaterial::custom(0.0, 0.01, 0.05, 1400.0).is_err());
    }

    #[test]
    fn test_custom_material_invalid_viscosity() {
        assert!(FluidMaterial::custom(1000.0, -0.01, 0.05, 1400.0).is_err());
    }

    #[test]
    fn test_particle_new() {
        let p = FluidParticle::new([1.0, 2.0, 3.0], 0.01);
        assert!((p.position[0] - 1.0).abs() < f64::EPSILON);
        assert!((p.speed()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_particle_2d() {
        let p = FluidParticle::new_2d(5.0, 10.0, 0.01);
        assert!((p.position[2]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_particle_speed() {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        p.velocity = [3.0, 4.0, 0.0];
        assert!((p.speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_particle_kinetic_energy() {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 2.0);
        p.velocity = [3.0, 4.0, 0.0];
        // KE = 0.5 * 2.0 * 25.0 = 25.0
        assert!((p.kinetic_energy() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_particle_distance() {
        let a = FluidParticle::new_2d(0.0, 0.0, 1.0);
        let b = FluidParticle::new_2d(3.0, 4.0, 1.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_water_2d() {
        let c = FluidConfig::water_2d();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_dt() {
        let mut c = FluidConfig::water_2d();
        c.dt = -0.001;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_invalid_smoothing() {
        let mut c = FluidConfig::water_2d();
        c.smoothing_radius = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_cfl_ok() {
        let c = FluidConfig::water_2d();
        assert!(c.check_cfl(1.0).is_ok());
    }

    #[test]
    fn test_cfl_violated() {
        let c = FluidConfig::water_2d();
        assert!(c.check_cfl(1000.0).is_err());
    }

    #[test]
    fn test_material_serde() {
        let m = FluidMaterial::WATER;
        let json = serde_json::to_string(&m).unwrap();
        let m2: FluidMaterial = serde_json::from_str(&json).unwrap();
        assert!((m2.density - m.density).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_serde() {
        let c = FluidConfig::water_2d();
        let json = serde_json::to_string(&c).unwrap();
        let c2: FluidConfig = serde_json::from_str(&json).unwrap();
        assert!((c2.dt - c.dt).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_predefined_materials() {
        let materials = [
            FluidMaterial::WATER,
            FluidMaterial::OIL,
            FluidMaterial::HONEY,
            FluidMaterial::AIR,
            FluidMaterial::LAVA,
        ];
        for m in &materials {
            assert!(m.density > 0.0);
            assert!(m.viscosity >= 0.0);
            assert!(m.speed_of_sound > 0.0);
        }
    }
}
