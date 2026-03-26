//! Shared fluid types — particles, materials, configuration.
//!
//! Uses hisab's `DVec3` (glam SIMD-accelerated) for all 3D vector quantities.

use serde::{Deserialize, Serialize};

use hisab::DVec3;

use crate::error::{PravashError, Result};

/// Physical properties of a fluid material.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
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
    #[must_use = "returns a new material, does not modify in place"]
    pub fn custom(
        density: f64,
        viscosity: f64,
        surface_tension: f64,
        speed_of_sound: f64,
    ) -> Result<Self> {
        if density <= 0.0 {
            return Err(PravashError::InvalidDensity { density });
        }
        if viscosity < 0.0 {
            return Err(PravashError::InvalidViscosity { viscosity });
        }
        if speed_of_sound <= 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!("speed of sound must be positive: {speed_of_sound}").into(),
            });
        }
        Ok(Self {
            density,
            viscosity,
            surface_tension,
            speed_of_sound,
        })
    }
}

/// A fluid particle (SPH).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FluidParticle {
    /// Position (2D: z=0, 3D: full).
    pub position: DVec3,
    /// Velocity.
    pub velocity: DVec3,
    /// Acceleration (accumulated forces / mass).
    pub acceleration: DVec3,
    /// Particle density (computed from neighbors).
    pub density: f64,
    /// Pressure (derived from density via equation of state).
    pub pressure: f64,
    /// Mass of this particle.
    pub mass: f64,
    /// Phase index for multi-phase simulation (0 = default/primary phase).
    pub phase: u8,
    /// Conformation tensor for viscoelastic fluids (2D symmetric: [c_xx, c_xy, c_yy]).
    /// Initialized to identity [1, 0, 1]. Only used when viscoelastic config is active.
    pub conformation: [f64; 3],
    /// Temperature (Kelvin). Default: 293.15 (20°C). Used for heat transfer.
    pub temperature: f64,
    /// Fuel concentration (0.0–1.0). Depleted by combustion above ignition temperature.
    pub fuel: f64,
}

impl FluidParticle {
    /// Create a particle at rest.
    #[must_use]
    pub fn new(position: DVec3, mass: f64) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            acceleration: DVec3::ZERO,
            density: 0.0,
            pressure: 0.0,
            mass,
            phase: 0,
            conformation: [1.0, 0.0, 1.0], // identity
            temperature: 293.15,           // 20°C
            fuel: 0.0,
        }
    }

    /// Create a 2D particle (z = 0).
    #[inline]
    #[must_use]
    pub fn new_2d(x: f64, y: f64, mass: f64) -> Self {
        Self::new(DVec3::new(x, y, 0.0), mass)
    }

    /// Squared speed (avoids sqrt, useful for comparisons).
    #[inline]
    #[must_use]
    pub fn speed_squared(&self) -> f64 {
        self.velocity.length_squared()
    }

    /// Speed (magnitude of velocity).
    #[inline]
    #[must_use]
    pub fn speed(&self) -> f64 {
        self.velocity.length()
    }

    /// Kinetic energy: 0.5 * m * v².
    #[inline]
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.speed_squared()
    }

    /// Squared distance to another particle (avoids sqrt).
    #[inline]
    #[must_use]
    pub fn distance_squared_to(&self, other: &FluidParticle) -> f64 {
        self.position.distance_squared(other.position)
    }

    /// Distance to another particle.
    #[inline]
    #[must_use]
    pub fn distance_to(&self, other: &FluidParticle) -> f64 {
        self.position.distance(other.position)
    }
}

/// Structure-of-Arrays layout for particle data.
///
/// Stores each field component in a separate contiguous array for
/// cache-friendly SIMD access. Use [`from_aos`](ParticleSoa::from_aos)
/// and [`to_aos`](ParticleSoa::to_aos) to convert to/from `FluidParticle` slices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSoa {
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub pos_z: Vec<f64>,
    pub vel_x: Vec<f64>,
    pub vel_y: Vec<f64>,
    pub vel_z: Vec<f64>,
    pub accel_x: Vec<f64>,
    pub accel_y: Vec<f64>,
    pub accel_z: Vec<f64>,
    pub density: Vec<f64>,
    pub pressure: Vec<f64>,
    pub mass: Vec<f64>,
    pub phase: Vec<u8>,
    pub conf_xx: Vec<f64>,
    pub conf_xy: Vec<f64>,
    pub conf_yy: Vec<f64>,
    pub temperature: Vec<f64>,
    pub fuel: Vec<f64>,
}

impl ParticleSoa {
    /// Create an empty SOA store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            pos_z: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            vel_z: Vec::new(),
            accel_x: Vec::new(),
            accel_y: Vec::new(),
            accel_z: Vec::new(),
            density: Vec::new(),
            pressure: Vec::new(),
            mass: Vec::new(),
            phase: Vec::new(),
            conf_xx: Vec::new(),
            conf_xy: Vec::new(),
            conf_yy: Vec::new(),
            temperature: Vec::new(),
            fuel: Vec::new(),
        }
    }

    /// Create a SOA store with given capacity.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            pos_x: Vec::with_capacity(n),
            pos_y: Vec::with_capacity(n),
            pos_z: Vec::with_capacity(n),
            vel_x: Vec::with_capacity(n),
            vel_y: Vec::with_capacity(n),
            vel_z: Vec::with_capacity(n),
            accel_x: Vec::with_capacity(n),
            accel_y: Vec::with_capacity(n),
            accel_z: Vec::with_capacity(n),
            density: Vec::with_capacity(n),
            pressure: Vec::with_capacity(n),
            mass: Vec::with_capacity(n),
            phase: Vec::with_capacity(n),
            conf_xx: Vec::with_capacity(n),
            conf_xy: Vec::with_capacity(n),
            conf_yy: Vec::with_capacity(n),
            temperature: Vec::with_capacity(n),
            fuel: Vec::with_capacity(n),
        }
    }

    /// Number of particles.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }

    /// Whether the store is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    /// Convert from a slice of AOS particles.
    #[must_use]
    pub fn from_aos(particles: &[FluidParticle]) -> Self {
        let n = particles.len();
        let mut soa = Self::with_capacity(n);
        for p in particles {
            soa.pos_x.push(p.position.x);
            soa.pos_y.push(p.position.y);
            soa.pos_z.push(p.position.z);
            soa.vel_x.push(p.velocity.x);
            soa.vel_y.push(p.velocity.y);
            soa.vel_z.push(p.velocity.z);
            soa.accel_x.push(p.acceleration.x);
            soa.accel_y.push(p.acceleration.y);
            soa.accel_z.push(p.acceleration.z);
            soa.density.push(p.density);
            soa.pressure.push(p.pressure);
            soa.mass.push(p.mass);
            soa.phase.push(p.phase);
            soa.conf_xx.push(p.conformation[0]);
            soa.conf_xy.push(p.conformation[1]);
            soa.conf_yy.push(p.conformation[2]);
            soa.temperature.push(p.temperature);
            soa.fuel.push(p.fuel);
        }
        soa
    }

    /// Convert back to AOS particles.
    #[must_use]
    pub fn to_aos(&self) -> Vec<FluidParticle> {
        let n = self.len();
        let mut particles = Vec::with_capacity(n);
        for i in 0..n {
            particles.push(FluidParticle {
                position: DVec3::new(self.pos_x[i], self.pos_y[i], self.pos_z[i]),
                velocity: DVec3::new(self.vel_x[i], self.vel_y[i], self.vel_z[i]),
                acceleration: DVec3::new(self.accel_x[i], self.accel_y[i], self.accel_z[i]),
                density: self.density[i],
                pressure: self.pressure[i],
                mass: self.mass[i],
                phase: self.phase[i],
                conformation: [self.conf_xx[i], self.conf_xy[i], self.conf_yy[i]],
                temperature: self.temperature[i],
                fuel: self.fuel[i],
            });
        }
        particles
    }

    /// Write SOA data back into an existing AOS slice (avoids allocation).
    pub fn write_to_aos(&self, particles: &mut [FluidParticle]) {
        let n = self.len().min(particles.len());
        for (i, p) in particles.iter_mut().enumerate().take(n) {
            p.position = DVec3::new(self.pos_x[i], self.pos_y[i], self.pos_z[i]);
            p.velocity = DVec3::new(self.vel_x[i], self.vel_y[i], self.vel_z[i]);
            p.acceleration = DVec3::new(self.accel_x[i], self.accel_y[i], self.accel_z[i]);
            p.density = self.density[i];
            p.pressure = self.pressure[i];
            p.mass = self.mass[i];
            p.phase = self.phase[i];
            p.conformation = [self.conf_xx[i], self.conf_xy[i], self.conf_yy[i]];
            p.temperature = self.temperature[i];
            p.fuel = self.fuel[i];
        }
    }

    /// Squared distance between particles i and j.
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, i: usize, j: usize) -> f64 {
        let dx = self.pos_x[i] - self.pos_x[j];
        let dy = self.pos_y[i] - self.pos_y[j];
        let dz = self.pos_z[i] - self.pos_z[j];
        dx * dx + dy * dy + dz * dz
    }
}

impl Default for ParticleSoa {
    fn default() -> Self {
        Self::new()
    }
}

// ── Particle Arena ──────────────────────────────────────────────────────────

/// Handle to an allocated block of particles in a [`ParticleArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaHandle {
    start: usize,
    len: usize,
}

impl ArenaHandle {
    /// Starting index into the arena's particle buffer.
    #[inline]
    #[must_use]
    pub fn start(&self) -> usize {
        self.start
    }

    /// Number of particles in this block.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this handle is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Pre-allocated arena for particle buffers.
///
/// Avoids per-step heap allocation by reserving capacity upfront.
/// Hands out contiguous blocks via [`alloc`](ParticleArena::alloc) and reclaims
/// them via [`free`](ParticleArena::free). Freed blocks are merged and reused.
///
/// Active particles can be accessed as a contiguous slice via
/// [`active_particles`](ParticleArena::active_particles) after calling
/// [`compact`](ParticleArena::compact).
#[non_exhaustive]
pub struct ParticleArena {
    particles: Vec<FluidParticle>,
    capacity: usize,
    /// High-water mark: first unused index.
    watermark: usize,
    /// Free list: (start, len) sorted by start index.
    free_list: Vec<(usize, usize)>,
}

impl ParticleArena {
    /// Create an arena with the given maximum particle capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let default_particle = FluidParticle::new(DVec3::ZERO, 0.0);
        Self {
            particles: vec![default_particle; capacity],
            capacity,
            watermark: 0,
            free_list: Vec::new(),
        }
    }

    /// Total capacity of the arena.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of currently allocated (active) particles.
    #[inline]
    #[must_use]
    pub fn active_count(&self) -> usize {
        let free_total: usize = self.free_list.iter().map(|(_, len)| len).sum();
        self.watermark.saturating_sub(free_total)
    }

    /// Allocate a contiguous block of `n` particles. Returns `None` if
    /// there isn't enough space.
    pub fn alloc(&mut self, n: usize) -> Option<ArenaHandle> {
        if n == 0 {
            return Some(ArenaHandle { start: 0, len: 0 });
        }

        // Try to find a free block that fits
        for i in 0..self.free_list.len() {
            let (start, free_len) = self.free_list[i];
            if free_len >= n {
                let handle = ArenaHandle { start, len: n };
                if free_len == n {
                    self.free_list.remove(i);
                } else {
                    self.free_list[i] = (start + n, free_len - n);
                }
                // Zero the allocated particles
                let default = FluidParticle::new(DVec3::ZERO, 0.0);
                for p in &mut self.particles[start..start + n] {
                    *p = default;
                }
                return Some(handle);
            }
        }

        // Allocate from the watermark
        if self.watermark + n <= self.capacity {
            let handle = ArenaHandle {
                start: self.watermark,
                len: n,
            };
            self.watermark += n;
            return Some(handle);
        }

        None // out of space
    }

    /// Free a previously allocated block, returning its slots to the pool.
    pub fn free(&mut self, handle: ArenaHandle) {
        if handle.len == 0 {
            return;
        }

        // Insert into free list sorted by start, then merge adjacent
        let entry = (handle.start, handle.len);
        let pos = self
            .free_list
            .binary_search_by_key(&handle.start, |&(s, _)| s)
            .unwrap_or_else(|p| p);
        self.free_list.insert(pos, entry);

        // Merge with next block
        if pos + 1 < self.free_list.len() {
            let (s1, l1) = self.free_list[pos];
            let (s2, l2) = self.free_list[pos + 1];
            if s1 + l1 == s2 {
                self.free_list[pos] = (s1, l1 + l2);
                self.free_list.remove(pos + 1);
            }
        }
        // Merge with previous block
        if pos > 0 {
            let (s0, l0) = self.free_list[pos - 1];
            let (s1, l1) = self.free_list[pos];
            if s0 + l0 == s1 {
                self.free_list[pos - 1] = (s0, l0 + l1);
                self.free_list.remove(pos);
            }
        }

        // Shrink watermark if the freed block was at the end
        if let Some(&(s, l)) = self.free_list.last()
            && s + l == self.watermark
        {
            self.watermark = s;
            self.free_list.pop();
        }
    }

    /// Get a slice of particles for a given handle.
    #[inline]
    #[must_use]
    pub fn get(&self, handle: ArenaHandle) -> &[FluidParticle] {
        &self.particles[handle.start..handle.start + handle.len]
    }

    /// Get a mutable slice of particles for a given handle.
    #[inline]
    pub fn get_mut(&mut self, handle: ArenaHandle) -> &mut [FluidParticle] {
        &mut self.particles[handle.start..handle.start + handle.len]
    }

    /// All particles from index 0 to watermark (includes freed gaps).
    /// For gap-free iteration, use `compact()` first then `active_particles()`.
    #[inline]
    #[must_use]
    pub fn raw_slice(&self) -> &[FluidParticle] {
        &self.particles[..self.watermark]
    }

    /// Compact the arena by moving all active particles to the front.
    /// Returns the number of active particles. Invalidates all existing handles.
    pub fn compact(&mut self) -> usize {
        if self.free_list.is_empty() {
            return self.watermark;
        }

        let mut write = 0;
        let mut read = 0;
        let wm = self.watermark;
        let mut free_idx = 0;

        while read < wm {
            // Skip over free blocks
            if free_idx < self.free_list.len() && read == self.free_list[free_idx].0 {
                read += self.free_list[free_idx].1;
                free_idx += 1;
                continue;
            }
            if read != write {
                self.particles[write] = self.particles[read];
            }
            write += 1;
            read += 1;
        }

        self.watermark = write;
        self.free_list.clear();
        write
    }

    /// Contiguous slice of all active particles after compaction.
    /// Call `compact()` first to ensure no gaps.
    #[inline]
    #[must_use]
    pub fn active_particles(&self) -> &[FluidParticle] {
        &self.particles[..self.watermark]
    }

    /// Mutable contiguous slice of all active particles after compaction.
    #[inline]
    pub fn active_particles_mut(&mut self) -> &mut [FluidParticle] {
        &mut self.particles[..self.watermark]
    }
}

/// Simulation configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FluidConfig {
    /// Timestep in seconds.
    pub dt: f64,
    /// Gravity vector.
    pub gravity: DVec3,
    /// SPH smoothing radius (particle interaction distance).
    pub smoothing_radius: f64,
    /// Gas constant for equation of state (pressure from density).
    pub gas_constant: f64,
    /// Rest density of the fluid.
    pub rest_density: f64,
    /// Domain bounds minimum corner.
    pub bounds_min: DVec3,
    /// Domain bounds maximum corner.
    pub bounds_max: DVec3,
    /// Boundary damping (velocity multiplier on wall collision, 0-1).
    pub boundary_damping: f64,
}

impl FluidConfig {
    /// Default 2D water simulation config.
    #[must_use]
    pub fn water_2d() -> Self {
        Self {
            dt: 0.001,
            gravity: DVec3::new(0.0, -9.81, 0.0),
            smoothing_radius: 0.05,
            gas_constant: 2000.0,
            rest_density: FluidMaterial::WATER.density,
            bounds_min: DVec3::ZERO,
            bounds_max: DVec3::new(1.0, 1.0, 0.0),
            boundary_damping: 0.5,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt: self.dt });
        }
        if !self.smoothing_radius.is_finite() || self.smoothing_radius <= 0.0 {
            return Err(PravashError::InvalidSmoothingRadius {
                h: self.smoothing_radius,
            });
        }
        if !self.rest_density.is_finite() || self.rest_density <= 0.0 {
            return Err(PravashError::InvalidDensity {
                density: self.rest_density,
            });
        }
        if !self.gas_constant.is_finite() || self.gas_constant <= 0.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!("gas constant must be positive: {}", self.gas_constant).into(),
            });
        }
        if self.boundary_damping < 0.0 || self.boundary_damping > 1.0 {
            return Err(PravashError::InvalidParameter {
                reason: format!(
                    "boundary damping must be in [0, 1]: {}",
                    self.boundary_damping
                )
                .into(),
            });
        }
        // Check bounds ordering: min must be <= max for each axis
        let lo = self.bounds_min;
        let hi = self.bounds_max;
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            return Err(PravashError::InvalidParameter {
                reason: format!(
                    "bounds min must be <= max: [{}, {}, {}] to [{}, {}, {}]",
                    lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
                )
                .into(),
            });
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
        const { assert!(FluidMaterial::WATER.viscosity > 0.0) };
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
    fn test_custom_material_invalid_speed_of_sound() {
        assert!(FluidMaterial::custom(1000.0, 0.01, 0.05, 0.0).is_err());
        assert!(FluidMaterial::custom(1000.0, 0.01, 0.05, -1.0).is_err());
    }

    #[test]
    fn test_particle_new() {
        let p = FluidParticle::new(DVec3::new(1.0, 2.0, 3.0), 0.01);
        assert!((p.position.x - 1.0).abs() < f64::EPSILON);
        assert!((p.speed()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_particle_copy() {
        let p = FluidParticle::new_2d(1.0, 2.0, 0.01);
        let p2 = p; // Copy, not move
        assert!((p.position.x - p2.position.x).abs() < f64::EPSILON);
    }

    #[test]
    fn test_particle_2d() {
        let p = FluidParticle::new_2d(5.0, 10.0, 0.01);
        assert!((p.position.z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_particle_speed() {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        p.velocity = DVec3::new(3.0, 4.0, 0.0);
        assert!((p.speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_particle_kinetic_energy() {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 2.0);
        p.velocity = DVec3::new(3.0, 4.0, 0.0);
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
    fn test_particle_distance_squared() {
        let a = FluidParticle::new_2d(0.0, 0.0, 1.0);
        let b = FluidParticle::new_2d(3.0, 4.0, 1.0);
        assert!((a.distance_squared_to(&b) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_water_2d() {
        let c = FluidConfig::water_2d();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_config_nan_dt_rejected() {
        let mut c = FluidConfig::water_2d();
        c.dt = f64::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_nan_smoothing_rejected() {
        let mut c = FluidConfig::water_2d();
        c.smoothing_radius = f64::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_inf_density_rejected() {
        let mut c = FluidConfig::water_2d();
        c.rest_density = f64::INFINITY;
        assert!(c.validate().is_err());
    }

    // ── ParticleSoa tests ────────────────────────────────────────────────

    #[test]
    fn test_soa_roundtrip() {
        let mut p = FluidParticle::new(DVec3::new(1.0, 2.0, 3.0), 0.5);
        p.velocity = DVec3::new(4.0, 5.0, 6.0);
        p.density = 1000.0;
        p.pressure = 200.0;
        let particles = vec![p];

        let soa = ParticleSoa::from_aos(&particles);
        assert_eq!(soa.len(), 1);
        assert!((soa.pos_x[0] - 1.0).abs() < f64::EPSILON);
        assert!((soa.vel_y[0] - 5.0).abs() < f64::EPSILON);
        assert!((soa.mass[0] - 0.5).abs() < f64::EPSILON);

        let back = soa.to_aos();
        assert!((back[0].position.x - 1.0).abs() < f64::EPSILON);
        assert!((back[0].velocity.y - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_soa_write_to_aos() {
        let particles_orig = vec![
            FluidParticle::new_2d(0.0, 0.0, 1.0),
            FluidParticle::new_2d(1.0, 1.0, 1.0),
        ];
        let mut soa = ParticleSoa::from_aos(&particles_orig);
        soa.pos_x[0] = 99.0;
        soa.density[1] = 500.0;

        let mut particles = particles_orig;
        soa.write_to_aos(&mut particles);
        assert!((particles[0].position.x - 99.0).abs() < f64::EPSILON);
        assert!((particles[1].density - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_soa_distance_squared() {
        let particles = vec![
            FluidParticle::new_2d(0.0, 0.0, 1.0),
            FluidParticle::new_2d(3.0, 4.0, 1.0),
        ];
        let soa = ParticleSoa::from_aos(&particles);
        assert!((soa.distance_squared(0, 1) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_soa_empty() {
        let soa = ParticleSoa::new();
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
        assert!(soa.to_aos().is_empty());
    }

    // ── ParticleArena tests ──────────────────────────────────────────────

    #[test]
    fn test_arena_alloc_and_get() {
        let mut arena = ParticleArena::new(100);
        let h = arena.alloc(10).unwrap();
        assert_eq!(h.len(), 10);
        assert_eq!(arena.active_count(), 10);
        let slice = arena.get(h);
        assert_eq!(slice.len(), 10);
    }

    #[test]
    fn test_arena_alloc_zero() {
        let mut arena = ParticleArena::new(10);
        let h = arena.alloc(0).unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn test_arena_capacity_exceeded() {
        let mut arena = ParticleArena::new(10);
        assert!(arena.alloc(11).is_none());
        assert!(arena.alloc(10).is_some());
        assert!(arena.alloc(1).is_none());
    }

    #[test]
    fn test_arena_free_and_reuse() {
        let mut arena = ParticleArena::new(20);
        let h1 = arena.alloc(10).unwrap();
        let _h2 = arena.alloc(5).unwrap();
        assert_eq!(arena.active_count(), 15);

        arena.free(h1);
        assert_eq!(arena.active_count(), 5);

        // Should reuse the freed block
        let h3 = arena.alloc(8).unwrap();
        assert_eq!(h3.start(), h1.start());
        assert_eq!(arena.active_count(), 13);
    }

    #[test]
    fn test_arena_free_merges_adjacent() {
        let mut arena = ParticleArena::new(30);
        let h1 = arena.alloc(10).unwrap();
        let h2 = arena.alloc(10).unwrap();
        let _h3 = arena.alloc(10).unwrap();

        arena.free(h1);
        arena.free(h2);
        // h1 and h2 should merge into a single 20-slot block
        let h4 = arena.alloc(20).unwrap();
        assert_eq!(h4.start(), 0);
    }

    #[test]
    fn test_arena_compact() {
        let mut arena = ParticleArena::new(30);
        let h1 = arena.alloc(10).unwrap();
        let h2 = arena.alloc(10).unwrap();

        // Write recognizable data
        arena.get_mut(h1)[0] = FluidParticle::new(DVec3::new(1.0, 0.0, 0.0), 1.0);
        arena.get_mut(h2)[0] = FluidParticle::new(DVec3::new(2.0, 0.0, 0.0), 2.0);

        // Free first block, creating a gap
        arena.free(h1);
        assert_eq!(arena.active_count(), 10);

        // Compact moves h2 data to front
        let active = arena.compact();
        assert_eq!(active, 10);
        assert!((arena.active_particles()[0].position.x - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_arena_free_at_end_shrinks_watermark() {
        let mut arena = ParticleArena::new(20);
        let _h1 = arena.alloc(5).unwrap();
        let h2 = arena.alloc(5).unwrap();

        // Freeing the last block should shrink the watermark
        arena.free(h2);
        // Now we should be able to alloc 15 (5 used + 15 remaining)
        let h3 = arena.alloc(15).unwrap();
        assert_eq!(h3.start(), 5);
    }

    #[test]
    fn test_arena_get_mut_writes() {
        let mut arena = ParticleArena::new(10);
        let h = arena.alloc(3).unwrap();
        let slice = arena.get_mut(h);
        slice[0] = FluidParticle::new(DVec3::new(99.0, 0.0, 0.0), 1.0);
        assert!((arena.get(h)[0].position.x - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_copy() {
        let c = FluidConfig::water_2d();
        let c2 = c; // Copy, not move
        assert!((c.dt - c2.dt).abs() < f64::EPSILON);
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
    fn test_config_invalid_gas_constant() {
        let mut c = FluidConfig::water_2d();
        c.gas_constant = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_invalid_boundary_damping() {
        let mut c = FluidConfig::water_2d();
        c.boundary_damping = 1.5;
        assert!(c.validate().is_err());
        c.boundary_damping = -0.1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_invalid_bounds() {
        let mut c = FluidConfig::water_2d();
        c.bounds_min = DVec3::new(1.0, 0.0, 0.0); // min_x > max_x
        c.bounds_max = DVec3::new(0.0, 1.0, 0.0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_equal_bounds_valid() {
        let mut c = FluidConfig::water_2d();
        c.bounds_min = DVec3::new(0.0, 0.0, 0.0); // min_x == max_x, valid (2D)
        c.bounds_max = DVec3::new(0.0, 1.0, 0.0);
        assert!(c.validate().is_ok());
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
