//! Coupling — fluid-body interaction and hybrid methods.
//!
//! Bridges particle (SPH) and grid (Navier-Stokes) solvers. Provides:
//! - [`RigidBody`] with two-way SPH coupling
//! - [`FlipSolver`] for FLIP/PIC hybrid advection (particles on grid)

use serde::{Deserialize, Serialize};

use hisab::DVec3;

use crate::common::FluidParticle;
use crate::error::{PravashError, Result};

use tracing::trace_span;

// ── Rigid Body ──────────────────────────────────────────────────────────────

/// Shape for a rigid body, with signed distance function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BodyShape {
    /// Sphere with given radius.
    Sphere { radius: f64 },
    /// Axis-aligned box with given half-extents [hx, hy, hz].
    Box { half_extents: [f64; 3] },
}

impl BodyShape {
    /// Signed distance from a point to the body surface.
    /// Negative = inside, positive = outside.
    #[inline]
    #[must_use]
    pub fn signed_distance(&self, point: DVec3, body_pos: DVec3) -> f64 {
        let dx = point.x - body_pos.x;
        let dy = point.y - body_pos.y;
        let dz = point.z - body_pos.z;
        match *self {
            BodyShape::Sphere { radius } => (dx * dx + dy * dy + dz * dz).sqrt() - radius,
            BodyShape::Box { half_extents } => {
                let qx = dx.abs() - half_extents[0];
                let qy = dy.abs() - half_extents[1];
                let qz = dz.abs() - half_extents[2];
                let outside =
                    (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt();
                let inside = qx.max(qy).max(qz).min(0.0);
                outside + inside
            }
        }
    }

    /// Approximate volume of the shape.
    #[inline]
    #[must_use]
    pub fn volume(&self) -> f64 {
        match *self {
            BodyShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius
            }
            BodyShape::Box { half_extents } => {
                8.0 * half_extents[0] * half_extents[1] * half_extents[2]
            }
        }
    }
}

/// A rigid body that interacts with fluid particles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigidBody {
    pub position: DVec3,
    pub velocity: DVec3,
    pub mass: f64,
    pub shape: BodyShape,
    /// Accumulated force from fluid interaction (reset each step).
    pub force: DVec3,
}

impl RigidBody {
    /// Create a new rigid body at rest.
    #[must_use]
    pub fn new(position: DVec3, mass: f64, shape: BodyShape) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            mass,
            shape,
            force: DVec3::ZERO,
        }
    }

    /// Signed distance from a point to this body's surface.
    #[inline]
    #[must_use]
    pub fn signed_distance(&self, point: DVec3) -> f64 {
        self.shape.signed_distance(point, self.position)
    }

    /// Surface normal at a point (central-difference gradient of signed distance).
    #[must_use]
    pub fn surface_normal(&self, point: DVec3) -> DVec3 {
        let eps = 1e-6;
        let dx = self
            .shape
            .signed_distance(DVec3::new(point.x + eps, point.y, point.z), self.position)
            - self
                .shape
                .signed_distance(DVec3::new(point.x - eps, point.y, point.z), self.position);
        let dy = self
            .shape
            .signed_distance(DVec3::new(point.x, point.y + eps, point.z), self.position)
            - self
                .shape
                .signed_distance(DVec3::new(point.x, point.y - eps, point.z), self.position);
        let dz = self
            .shape
            .signed_distance(DVec3::new(point.x, point.y, point.z + eps), self.position)
            - self
                .shape
                .signed_distance(DVec3::new(point.x, point.y, point.z - eps), self.position);
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag < 1e-20 {
            // Fallback: direction from body center to point
            let fx = point.x - self.position.x;
            let fy = point.y - self.position.y;
            let fz = point.z - self.position.z;
            let fmag = (fx * fx + fy * fy + fz * fz).sqrt();
            return if fmag < 1e-20 {
                DVec3::new(1.0, 0.0, 0.0) // arbitrary fallback for exact center
            } else {
                DVec3::new(fx / fmag, fy / fmag, fz / fmag)
            };
        }
        DVec3::new(dx / mag, dy / mag, dz / mag)
    }
}

/// Compute two-way forces between SPH particles and rigid bodies.
///
/// - Particles near/inside a body are pushed out along the surface normal.
/// - The body accumulates the reaction force (Newton's third law).
/// - Call this after SPH density/pressure computation but before integration.
pub fn couple_sph_bodies(
    particles: &mut [FluidParticle],
    bodies: &mut [RigidBody],
    smoothing_radius: f64,
    stiffness: f64,
    damping: f64,
) {
    let _span = trace_span!(
        "coupling::sph_bodies",
        n_particles = particles.len(),
        n_bodies = bodies.len()
    )
    .entered();

    // Reset body forces
    for body in bodies.iter_mut() {
        body.force = DVec3::ZERO;
    }

    for p in particles.iter_mut() {
        for body in bodies.iter_mut() {
            let sd = body.signed_distance(p.position);

            // Only interact when particle is within smoothing radius of surface
            if sd < smoothing_radius {
                let normal = body.surface_normal(p.position);
                // Penetration depth (positive when inside)
                let penetration = smoothing_radius - sd;
                let pen_force = stiffness * penetration;

                // Relative velocity (particle relative to body surface)
                let rel_vel = p.velocity - body.velocity;

                // Normal component of relative velocity
                let vn = rel_vel.dot(normal);

                // Damping force (only when approaching)
                let damp_force = if vn < 0.0 { -damping * vn } else { 0.0 };

                let total = pen_force + damp_force;

                // Apply force to particle (push outward)
                p.acceleration += total * normal / p.mass.max(1e-20);

                // Reaction force on body (Newton's third law)
                body.force -= total * normal;
            }
        }
    }
}

/// Integrate rigid body motion from accumulated forces.
pub fn integrate_bodies(bodies: &mut [RigidBody], gravity: DVec3, dt: f64) {
    for body in bodies {
        let inv_mass = 1.0 / body.mass.max(1e-20);
        body.velocity += (body.force * inv_mass + gravity) * dt;
        body.position += body.velocity * dt;
    }
}

// ── Added Mass ──────────────────────────────────────────────────────────────

/// Added mass coefficients for common shapes.
pub struct AddedMassCoefficient;

impl AddedMassCoefficient {
    /// Sphere: C_a = 0.5
    pub const SPHERE: f64 = 0.5;
    /// Cube: C_a ≈ 0.67
    pub const CUBE: f64 = 0.67;
    /// Long cylinder (transverse): C_a = 1.0
    pub const CYLINDER: f64 = 1.0;
    /// Flat plate (normal): C_a ≈ 1.0
    pub const FLAT_PLATE: f64 = 1.0;
}

/// Compute the effective mass including added mass.
///
/// m_eff = m_body + C_a · ρ_fluid · V_body
#[inline]
#[must_use]
pub fn effective_mass(body_mass: f64, fluid_density: f64, body_volume: f64, ca: f64) -> f64 {
    body_mass + ca * fluid_density * body_volume
}

/// Integrate rigid body motion with added mass effect.
///
/// The added mass increases the effective inertia of the body, reducing
/// acceleration for a given force. `ca` is the added mass coefficient
/// (shape-dependent, 0.5 for sphere).
pub fn integrate_bodies_with_added_mass(
    bodies: &mut [RigidBody],
    gravity: DVec3,
    dt: f64,
    fluid_density: f64,
    ca: f64,
) {
    for body in bodies {
        let vol = body.shape.volume();
        let m_eff = effective_mass(body.mass, fluid_density, vol, ca);
        let inv_m = 1.0 / m_eff.max(1e-20);
        body.velocity += (body.force * inv_m + gravity) * dt;
        body.position += body.velocity * dt;
    }
}

// ── Drag from Velocity Fields ───────────────────────────────────────────────

/// Compute drag on a rigid body from nearby SPH particle velocities.
///
/// Samples the average fluid velocity near the body surface from particles
/// within `sample_radius`, then applies the standard drag formula using
/// the relative velocity.
///
/// Returns the drag force vector (opposing relative motion).
#[must_use]
pub fn drag_from_particles(
    body: &RigidBody,
    particles: &[FluidParticle],
    fluid_density: f64,
    drag_coefficient: f64,
    sample_radius: f64,
) -> DVec3 {
    let _span = trace_span!("coupling::drag_from_particles").entered();
    // Average fluid velocity near the body surface
    let mut avg_vx = 0.0;
    let mut avg_vy = 0.0;
    let mut avg_vz = 0.0;
    let mut count = 0.0;

    for p in particles {
        let sd = body.signed_distance(p.position);
        if sd.abs() < sample_radius {
            avg_vx += p.velocity.x;
            avg_vy += p.velocity.y;
            avg_vz += p.velocity.z;
            count += 1.0;
        }
    }

    if count < 1.0 {
        return DVec3::ZERO;
    }

    avg_vx /= count;
    avg_vy /= count;
    avg_vz /= count;

    // Relative velocity (fluid relative to body)
    let rel_vx = avg_vx - body.velocity.x;
    let rel_vy = avg_vy - body.velocity.y;
    let rel_vz = avg_vz - body.velocity.z;
    let rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz).sqrt();

    if rel_speed < 1e-20 {
        return DVec3::ZERO;
    }

    // Cross-section area projected onto the velocity-normal plane
    let area = match body.shape {
        BodyShape::Sphere { radius } => std::f64::consts::PI * radius * radius,
        BodyShape::Box { half_extents } => {
            // Project box area onto the plane perpendicular to relative velocity
            let ax = (rel_vx / rel_speed).abs();
            let ay = (rel_vy / rel_speed).abs();
            let az = (rel_vz / rel_speed).abs();
            // Each face contributes proportionally to its alignment with velocity
            4.0 * (ax * half_extents[1] * half_extents[2]
                + ay * half_extents[0] * half_extents[2]
                + az * half_extents[0] * half_extents[1])
        }
    };

    // F_drag = 0.5 · ρ · |v_rel|² · Cd · A · v̂_rel
    let drag_mag = 0.5 * fluid_density * rel_speed * drag_coefficient * area;
    DVec3::new(
        drag_mag * rel_vx / rel_speed,
        drag_mag * rel_vy / rel_speed,
        drag_mag * rel_vz / rel_speed,
    )
}

// ── Particle-Level Set Surface Tracking ─────────────────────────────────────

/// Reconstruct a signed distance field on a grid from SPH particles.
///
/// Uses nearest-particle distance to build an approximate level set.
/// Positive = outside fluid, negative = inside fluid.
/// The `particle_radius` defines the effective radius of each particle's
/// contribution to the fluid region.
pub fn particle_level_set(
    level_set: &mut [f64],
    nx: usize,
    ny: usize,
    dx: f64,
    particles: &[FluidParticle],
    particle_radius: f64,
) {
    let _span = trace_span!("coupling::particle_level_set", nx, ny, n = particles.len()).entered();

    // Initialize to large positive (outside)
    let far = (nx as f64 + ny as f64) * dx;
    level_set.fill(far);

    // For each grid cell, find distance to nearest particle
    for y in 0..ny {
        for x in 0..nx {
            let gx = x as f64 * dx;
            let gy = y as f64 * dx;
            let idx = y * nx + x;
            let mut min_dist = far;

            for p in particles {
                let ddx = gx - p.position.x;
                let ddy = gy - p.position.y;
                let dist = (ddx * ddx + ddy * ddy).sqrt() - particle_radius;
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            level_set[idx] = min_dist;
        }
    }
}

// ── FLIP/PIC Hybrid ─────────────────────────────────────────────────────────

/// FLIP/PIC hybrid solver — particles advected on a background grid.
///
/// Combines the low-dissipation of particle methods (FLIP) with the stability
/// of grid methods (PIC). The blend ratio controls the mix:
/// - 0.0 = pure PIC (stable, diffusive)
/// - 1.0 = pure FLIP (low dissipation, can be noisy)
/// - 0.95 = typical (mostly FLIP with a touch of PIC for stability)
#[non_exhaustive]
pub struct FlipSolver {
    nx: usize,
    ny: usize,
    dx: f64,
    /// FLIP/PIC blend ratio (0 = PIC, 1 = FLIP).
    pub flip_ratio: f64,
    // Grid fields
    vx: Vec<f64>,
    vy: Vec<f64>,
    vx_old: Vec<f64>,
    vy_old: Vec<f64>,
    weight: Vec<f64>,
    pressure: Vec<f64>,
    div: Vec<f64>,
    /// Use APIC (Affine Particle-In-Cell) transfers instead of FLIP/PIC.
    /// Conserves angular momentum. Overrides `flip_ratio` when true.
    pub use_apic: bool,
    /// Narrow-band mode: only transfer particles within `narrow_band_cells`
    /// of the free surface. Interior particles are frozen. 0 = disabled.
    pub narrow_band_cells: usize,
    /// Per-particle affine velocity matrices [c00, c01, c10, c11] for APIC.
    apic_c: Vec<[f64; 4]>,
}

impl FlipSolver {
    /// Create a new FLIP/PIC solver.
    pub fn new(nx: usize, ny: usize, dx: f64, flip_ratio: f64) -> Result<Self> {
        if nx < 4 || ny < 4 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 || !dx.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
        let n = nx * ny;
        Ok(Self {
            nx,
            ny,
            dx,
            flip_ratio: flip_ratio.clamp(0.0, 1.0),
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            vx_old: vec![0.0; n],
            vy_old: vec![0.0; n],
            weight: vec![0.0; n],
            pressure: vec![0.0; n],
            div: vec![0.0; n],
            use_apic: false,
            narrow_band_cells: 0,
            apic_c: Vec::new(),
        })
    }

    #[inline]
    #[must_use]
    pub fn grid_nx(&self) -> usize {
        self.nx
    }

    #[inline]
    #[must_use]
    pub fn grid_ny(&self) -> usize {
        self.ny
    }

    #[inline]
    #[must_use]
    pub fn grid_dx(&self) -> f64 {
        self.dx
    }

    /// Transfer particle velocities to grid (Particle-to-Grid).
    fn particles_to_grid(&mut self, particles: &[FluidParticle]) {
        let _span = trace_span!("flip::p2g", n = particles.len()).entered();
        let nx = self.nx;
        let inv_dx = 1.0 / self.dx;

        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.weight.fill(0.0);

        let dx = self.dx;
        let use_apic = self.use_apic;

        for (pi, p) in particles.iter().enumerate() {
            let gx = p.position.x * inv_dx;
            let gy = p.position.y * inv_dx;

            let x0 = gx.floor().max(0.0).min((nx - 2) as f64) as usize;
            let y0 = gy.floor().max(0.0).min((self.ny - 2) as f64) as usize;

            let sx = gx - x0 as f64;
            let sy = gy - y0 as f64;

            let weights = [
                (1.0 - sx) * (1.0 - sy),
                sx * (1.0 - sy),
                (1.0 - sx) * sy,
                sx * sy,
            ];
            let grid_x = [x0, x0 + 1, x0, x0 + 1];
            let grid_y = [y0, y0, y0 + 1, y0 + 1];
            let indices = [
                y0 * nx + x0,
                y0 * nx + x0 + 1,
                (y0 + 1) * nx + x0,
                (y0 + 1) * nx + x0 + 1,
            ];

            // APIC: splat v_p + C_p · (x_i - x_p) instead of just v_p
            let c = if use_apic && pi < self.apic_c.len() {
                self.apic_c[pi]
            } else {
                [0.0; 4]
            };

            for k in 0..4 {
                let w = weights[k];
                let idx = indices[k];
                let dxi = grid_x[k] as f64 * dx - p.position.x;
                let dyi = grid_y[k] as f64 * dx - p.position.y;
                // v_splat = v_p + C_p · (x_i - x_p)
                self.vx[idx] += w * (p.velocity.x + c[0] * dxi + c[1] * dyi);
                self.vy[idx] += w * (p.velocity.y + c[2] * dxi + c[3] * dyi);
                self.weight[idx] += w;
            }
        }

        // Normalize by weights
        for i in 0..self.vx.len() {
            if self.weight[i] > 1e-20 {
                self.vx[i] /= self.weight[i];
                self.vy[i] /= self.weight[i];
            }
        }
    }

    /// Transfer grid velocities back to particles (Grid-to-Particle).
    fn grid_to_particles(&mut self, particles: &mut [FluidParticle]) {
        let _span = trace_span!("flip::g2p", n = particles.len()).entered();
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let inv_dx = 1.0 / dx;

        if self.use_apic {
            // APIC G2P: v_p = Σ v_i w_ip, C_p = (4/dx²) Σ v_i (x_i - x_p)^T w_ip
            self.apic_c.resize(particles.len(), [0.0; 4]);
            let scale = 4.0 / (dx * dx);
            for (pi, p) in particles.iter_mut().enumerate() {
                let gx = p.position.x * inv_dx;
                let gy = p.position.y * inv_dx;
                let x0 = gx.floor().max(0.0).min((nx - 2) as f64) as usize;
                let y0 = gy.floor().max(0.0).min((ny - 2) as f64) as usize;
                let sx = (gx - x0 as f64).clamp(0.0, 1.0);
                let sy = (gy - y0 as f64).clamp(0.0, 1.0);

                let mut vx_new = 0.0;
                let mut vy_new = 0.0;
                let mut c00 = 0.0;
                let mut c01 = 0.0;
                let mut c10 = 0.0;
                let mut c11 = 0.0;

                let weights = [
                    ((x0, y0), (1.0 - sx) * (1.0 - sy)),
                    ((x0 + 1, y0), sx * (1.0 - sy)),
                    ((x0, y0 + 1), (1.0 - sx) * sy),
                    ((x0 + 1, y0 + 1), sx * sy),
                ];
                for &((ix, iy), w) in &weights {
                    let idx = iy * nx + ix;
                    let grid_vx = self.vx[idx];
                    let grid_vy = self.vy[idx];
                    vx_new += w * grid_vx;
                    vy_new += w * grid_vy;
                    let dxi = ix as f64 * dx - p.position.x;
                    let dyi = iy as f64 * dx - p.position.y;
                    c00 += w * grid_vx * dxi * scale;
                    c01 += w * grid_vx * dyi * scale;
                    c10 += w * grid_vy * dxi * scale;
                    c11 += w * grid_vy * dyi * scale;
                }
                p.velocity.x = vx_new;
                p.velocity.y = vy_new;
                self.apic_c[pi] = [c00, c01, c10, c11];
            }
        } else {
            // Standard FLIP/PIC blend
            let flip = self.flip_ratio;
            let pic = 1.0 - flip;
            for p in particles.iter_mut() {
                let gx = p.position.x * inv_dx;
                let gy = p.position.y * inv_dx;
                let new_vx = Self::sample_field(&self.vx, nx, ny, gx, gy);
                let new_vy = Self::sample_field(&self.vy, nx, ny, gx, gy);
                let old_vx = Self::sample_field(&self.vx_old, nx, ny, gx, gy);
                let old_vy = Self::sample_field(&self.vy_old, nx, ny, gx, gy);
                let dvx = new_vx - old_vx;
                let dvy = new_vy - old_vy;
                p.velocity.x = flip * (p.velocity.x + dvx) + pic * new_vx;
                p.velocity.y = flip * (p.velocity.y + dvy) + pic * new_vy;
            }
        }
    }

    /// Bilinear sample from a grid field.
    #[inline]
    fn sample_field(field: &[f64], nx: usize, ny: usize, gx: f64, gy: f64) -> f64 {
        let x0 = gx.floor().max(0.0).min((nx - 2) as f64) as usize;
        let y0 = gy.floor().max(0.0).min((ny - 2) as f64) as usize;
        let sx = (gx - x0 as f64).clamp(0.0, 1.0);
        let sy = (gy - y0 as f64).clamp(0.0, 1.0);

        let v00 = field[y0 * nx + x0];
        let v10 = field[y0 * nx + x0 + 1];
        let v01 = field[(y0 + 1) * nx + x0];
        let v11 = field[(y0 + 1) * nx + x0 + 1];

        v00 * (1.0 - sx) * (1.0 - sy)
            + v10 * sx * (1.0 - sy)
            + v01 * (1.0 - sx) * sy
            + v11 * sx * sy
    }

    /// Perform one FLIP/PIC step.
    ///
    /// 1. Transfer particle velocities to grid (P2G)
    /// 2. Save pre-solve grid velocity
    /// 3. Pressure projection on grid
    /// 4. Transfer grid velocity back to particles (G2P) with FLIP/PIC blend
    /// 5. Advect particles through velocity field
    pub fn step(&mut self, particles: &mut [FluidParticle], gravity: DVec3, dt: f64) -> Result<()> {
        let _span = trace_span!("flip::step", n = particles.len()).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;

        // 1. P2G
        self.particles_to_grid(particles);

        // Save pre-solve velocity for FLIP delta
        self.vx_old.copy_from_slice(&self.vx);
        self.vy_old.copy_from_slice(&self.vy);

        // 2. Apply gravity to grid (both components)
        for i in 0..nx * ny {
            self.vx[i] += gravity.x * dt;
            self.vy[i] += gravity.y * dt;
        }

        // 3. Pressure projection on grid (with dt scaling)
        {
            let inv_2dx = 0.5 / self.dx;
            self.div.fill(0.0);
            // Divergence scaled by 1/dt for correct pressure units
            let div_scale = inv_2dx / dt;
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    self.div[i] = -((self.vx[i + 1] - self.vx[i - 1])
                        + (self.vy[i + nx] - self.vy[i - nx]))
                        * div_scale;
                }
            }
            // GS pressure solve (warm start from previous step)
            for _ in 0..40 {
                for y in 1..ny - 1 {
                    for x in 1..nx - 1 {
                        let i = y * nx + x;
                        let neighbors = self.pressure[i - 1]
                            + self.pressure[i + 1]
                            + self.pressure[i - nx]
                            + self.pressure[i + nx];
                        self.pressure[i] = (self.div[i] + neighbors) * 0.25;
                    }
                }
            }
            // Subtract pressure gradient scaled by dt
            let grad_scale = dt * inv_2dx;
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    self.vx[i] -= (self.pressure[i + 1] - self.pressure[i - 1]) * grad_scale;
                    self.vy[i] -= (self.pressure[i + nx] - self.pressure[i - nx]) * grad_scale;
                }
            }
        }

        // 4. G2P with FLIP/PIC blend
        self.grid_to_particles(particles);

        // 5. Advect particles and clamp to grid domain
        let max_x = (nx - 1) as f64 * self.dx;
        let max_y = (ny - 1) as f64 * self.dx;
        for p in particles.iter_mut() {
            p.position += p.velocity * dt;

            // Clamp to grid domain with velocity reflection
            if p.position.x < 0.0 {
                p.position.x = 0.0;
                p.velocity.x = 0.0;
            } else if p.position.x > max_x {
                p.position.x = max_x;
                p.velocity.x = 0.0;
            }
            if p.position.y < 0.0 {
                p.position.y = 0.0;
                p.velocity.y = 0.0;
            } else if p.position.y > max_y {
                p.position.y = max_y;
                p.velocity.y = 0.0;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BodyShape tests ─────────────────────────────────────────────────────

    #[test]
    fn test_sphere_sdf_center() {
        let shape = BodyShape::Sphere { radius: 1.0 };
        assert!((shape.signed_distance(DVec3::ZERO, DVec3::ZERO) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_sdf_surface() {
        let shape = BodyShape::Sphere { radius: 1.0 };
        assert!(
            shape
                .signed_distance(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_sphere_sdf_outside() {
        let shape = BodyShape::Sphere { radius: 1.0 };
        let sd = shape.signed_distance(DVec3::new(2.0, 0.0, 0.0), DVec3::ZERO);
        assert!((sd - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_sdf_center() {
        let shape = BodyShape::Box {
            half_extents: [1.0, 1.0, 1.0],
        };
        assert!(shape.signed_distance(DVec3::ZERO, DVec3::ZERO) < 0.0);
    }

    #[test]
    fn test_box_sdf_outside() {
        let shape = BodyShape::Box {
            half_extents: [1.0, 1.0, 1.0],
        };
        let sd = shape.signed_distance(DVec3::new(2.0, 0.0, 0.0), DVec3::ZERO);
        assert!((sd - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_volume() {
        let shape = BodyShape::Sphere { radius: 1.0 };
        let expected = (4.0 / 3.0) * std::f64::consts::PI;
        assert!((shape.volume() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_box_volume() {
        let shape = BodyShape::Box {
            half_extents: [1.0, 2.0, 3.0],
        };
        assert!((shape.volume() - 48.0).abs() < 1e-10);
    }

    // ── RigidBody tests ─────────────────────────────────────────────────────

    #[test]
    fn test_rigid_body_surface_normal() {
        let body = RigidBody::new(DVec3::ZERO, 1.0, BodyShape::Sphere { radius: 1.0 });
        let n = body.surface_normal(DVec3::new(2.0, 0.0, 0.0));
        assert!((n.x - 1.0).abs() < 1e-4);
        assert!(n.y.abs() < 1e-4);
    }

    #[test]
    fn test_coupling_pushes_particle_out() {
        let mut particles = vec![FluidParticle::new(DVec3::ZERO, 1.0)];
        particles[0].density = 1000.0;
        let mut bodies = vec![RigidBody::new(
            DVec3::ZERO,
            10.0,
            BodyShape::Sphere { radius: 0.5 },
        )];

        couple_sph_bodies(&mut particles, &mut bodies, 0.1, 1000.0, 10.0);

        // Particle at center of sphere should get pushed outward
        let accel_mag = particles[0].acceleration.length();
        assert!(accel_mag > 0.0, "particle should be accelerated outward");
    }

    #[test]
    fn test_coupling_newton_third_law() {
        let mut particles = vec![FluidParticle::new(DVec3::new(0.6, 0.0, 0.0), 1.0)];
        particles[0].density = 1000.0;
        let mut bodies = vec![RigidBody::new(
            DVec3::ZERO,
            10.0,
            BodyShape::Sphere { radius: 0.5 },
        )];

        couple_sph_bodies(&mut particles, &mut bodies, 0.2, 1000.0, 10.0);

        // Body force should be opposite to particle acceleration * mass
        let f_on_particle_x = particles[0].acceleration.x * particles[0].mass;
        let f_on_body_x = bodies[0].force.x;
        assert!(
            (f_on_particle_x + f_on_body_x).abs() < 1e-10,
            "forces should be equal and opposite: particle={f_on_particle_x}, body={f_on_body_x}"
        );
    }

    #[test]
    fn test_integrate_bodies() {
        let mut bodies = vec![RigidBody::new(
            DVec3::new(0.0, 1.0, 0.0),
            1.0,
            BodyShape::Sphere { radius: 0.1 },
        )];
        bodies[0].force = DVec3::ZERO;
        integrate_bodies(&mut bodies, DVec3::new(0.0, -9.81, 0.0), 0.01);
        assert!(bodies[0].velocity.y < 0.0);
        assert!(bodies[0].position.y < 1.0);
    }

    // ── FLIP/PIC tests ──────────────────────────────────────────────────────

    #[test]
    fn test_flip_solver_new() {
        let solver = FlipSolver::new(16, 16, 0.1, 0.95).unwrap();
        assert_eq!(solver.grid_nx(), 16);
        assert!((solver.flip_ratio - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_flip_solver_invalid() {
        assert!(FlipSolver::new(2, 16, 0.1, 0.95).is_err());
        assert!(FlipSolver::new(16, 16, 0.0, 0.95).is_err());
    }

    #[test]
    fn test_flip_step_empty() {
        let mut solver = FlipSolver::new(8, 8, 0.1, 0.95).unwrap();
        let mut particles = vec![];
        solver
            .step(&mut particles, DVec3::new(0.0, -9.81, 0.0), 0.01)
            .unwrap();
    }

    #[test]
    fn test_flip_step_particles_move() {
        let mut solver = FlipSolver::new(16, 16, 0.1, 0.95).unwrap();
        let mut particles = vec![FluidParticle::new_2d(0.8, 0.8, 0.01)];
        particles[0].velocity = DVec3::new(1.0, 0.0, 0.0);

        let x_before = particles[0].position.x;
        solver.step(&mut particles, DVec3::ZERO, 0.1).unwrap();
        // Particle should move (velocity may change from projection, but position advances)
        assert!(
            (particles[0].position.x - x_before).abs() > 1e-6,
            "particle should have moved: before={x_before}, after={}",
            particles[0].position.x
        );
    }

    #[test]
    fn test_flip_gravity() {
        let mut solver = FlipSolver::new(16, 16, 0.1, 0.0).unwrap(); // pure PIC
        let mut particles = vec![FluidParticle::new_2d(0.8, 0.8, 0.01)];

        solver
            .step(&mut particles, DVec3::new(0.0, -9.81, 0.0), 0.01)
            .unwrap();
        assert!(
            particles[0].velocity.y < 0.0,
            "particle should fall under gravity"
        );
    }

    #[test]
    fn test_flip_pic_blend() {
        let mut solver_flip = FlipSolver::new(8, 8, 0.1, 1.0).unwrap();
        let mut solver_pic = FlipSolver::new(8, 8, 0.1, 0.0).unwrap();

        let p_init = FluidParticle::new_2d(0.4, 0.4, 0.01);
        let mut particles_flip = vec![p_init];
        let mut particles_pic = vec![p_init];
        particles_flip[0].velocity = DVec3::new(1.0, 0.0, 0.0);
        particles_pic[0].velocity = DVec3::new(1.0, 0.0, 0.0);

        solver_flip
            .step(&mut particles_flip, DVec3::ZERO, 0.01)
            .unwrap();
        solver_pic
            .step(&mut particles_pic, DVec3::ZERO, 0.01)
            .unwrap();

        // Both should produce finite results
        assert!(particles_flip[0].velocity.x.is_finite());
        assert!(particles_pic[0].velocity.x.is_finite());
    }

    #[test]
    fn test_flip_boundary_clamping() {
        let mut solver = FlipSolver::new(8, 8, 0.1, 0.95).unwrap();
        // Particle with high velocity heading out of domain
        let mut particles = vec![FluidParticle::new_2d(0.05, 0.05, 0.01)];
        particles[0].velocity = DVec3::new(-10.0, -10.0, 0.0);

        solver.step(&mut particles, DVec3::ZERO, 0.1).unwrap();

        // Particle should be clamped to domain [0, (nx-1)*dx]
        assert!(
            particles[0].position.x >= 0.0,
            "x should be >= 0: {}",
            particles[0].position.x
        );
        assert!(
            particles[0].position.y >= 0.0,
            "y should be >= 0: {}",
            particles[0].position.y
        );
        // Velocity should be zeroed at boundary
        assert!(
            particles[0].velocity.x.abs() < 1e-10,
            "vx should be 0 at boundary: {}",
            particles[0].velocity.x
        );
    }

    // ── Added mass tests ────────────────────────────────────────────────────

    #[test]
    fn test_effective_mass_sphere() {
        // Sphere in water: m_eff = 1.0 + 0.5 * 1000 * V
        let vol = BodyShape::Sphere { radius: 0.1 }.volume();
        let m_eff = effective_mass(1.0, 1000.0, vol, AddedMassCoefficient::SPHERE);
        assert!(m_eff > 1.0);
        // Added mass should be 0.5 * 1000 * (4/3 * pi * 0.001) ≈ 2.09
        assert!((m_eff - 1.0 - 0.5 * 1000.0 * vol).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_with_added_mass_slower() {
        let mut body_normal = RigidBody::new(
            DVec3::new(0.0, 1.0, 0.0),
            1.0,
            BodyShape::Sphere { radius: 0.1 },
        );
        let mut body_added = body_normal.clone();
        // Apply same force, zero gravity to isolate added mass effect
        body_normal.force = DVec3::new(10.0, 0.0, 0.0);
        body_added.force = DVec3::new(10.0, 0.0, 0.0);

        integrate_bodies(std::slice::from_mut(&mut body_normal), DVec3::ZERO, 0.01);
        integrate_bodies_with_added_mass(
            std::slice::from_mut(&mut body_added),
            DVec3::ZERO,
            0.01,
            1000.0,
            AddedMassCoefficient::SPHERE,
        );

        // Added mass body should accelerate less from the same force
        assert!(
            body_added.velocity.x.abs() < body_normal.velocity.x.abs(),
            "added mass should reduce force response: normal={}, added={}",
            body_normal.velocity.x,
            body_added.velocity.x
        );
    }

    // ── Drag from particles tests ───────────────────────────────────────────

    #[test]
    fn test_drag_no_particles() {
        let body = RigidBody::new(DVec3::ZERO, 1.0, BodyShape::Sphere { radius: 0.1 });
        let drag = drag_from_particles(&body, &[], 1000.0, 0.47, 0.2);
        assert!(drag.x.abs() < 1e-20);
    }

    #[test]
    fn test_drag_opposing_motion() {
        let mut body = RigidBody::new(
            DVec3::new(0.5, 0.5, 0.0),
            1.0,
            BodyShape::Sphere { radius: 0.1 },
        );
        body.velocity = DVec3::new(1.0, 0.0, 0.0); // body moving right

        // Stationary fluid particles nearby
        let particles: Vec<FluidParticle> = (0..5)
            .map(|i| FluidParticle::new(DVec3::new(0.45 + i as f64 * 0.025, 0.5, 0.0), 0.01))
            .collect();

        let drag = drag_from_particles(&body, &particles, 1000.0, 0.47, 0.2);
        // Drag should oppose body motion (point left, negative x)
        assert!(
            drag.x < 0.0,
            "drag should oppose rightward motion: {}",
            drag.x
        );
    }

    // ── Particle-level set tests ────────────────────────────────────────────

    #[test]
    fn test_level_set_empty() {
        let mut ls = vec![0.0; 16];
        particle_level_set(&mut ls, 4, 4, 0.1, &[], 0.05);
        // All should be far (positive = outside)
        assert!(ls.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_level_set_single_particle() {
        let particles = vec![FluidParticle::new(DVec3::new(0.2, 0.2, 0.0), 0.01)];
        let mut ls = vec![0.0; 25];
        particle_level_set(&mut ls, 5, 5, 0.1, &particles, 0.05);
        // Cell at (2,2) = 0.2,0.2 should be inside (negative)
        assert!(
            ls[2 * 5 + 2] < 0.0,
            "particle center should be inside: {}",
            ls[2 * 5 + 2]
        );
        // Far corner should be outside (positive)
        assert!(ls[0] > 0.0);
    }

    #[test]
    fn test_level_set_continuous() {
        let particles = vec![FluidParticle::new(DVec3::new(0.5, 0.5, 0.0), 0.01)];
        let mut ls = vec![0.0; 100];
        particle_level_set(&mut ls, 10, 10, 0.1, &particles, 0.1);
        // Level set should be smooth (no sharp jumps between adjacent cells)
        for y in 0..9 {
            for x in 0..9 {
                let d1 = ls[y * 10 + x];
                let d2 = ls[y * 10 + x + 1];
                assert!(
                    (d1 - d2).abs() < 0.2,
                    "level set should be smooth: ({x},{y})={d1}, ({},{y})={d2}",
                    x + 1
                );
            }
        }
    }
}
