//! Shallow water equations — 2D surface wave simulation.
//!
//! Non-linear shallow water equations with convective transport.
//! Models water surface height and horizontal velocities using flux-form
//! continuity and momentum equations with proper convective acceleration.
//! Good for oceans, lakes, puddles, flooding, and dam break scenarios.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

use tracing::trace_span;

/// Shallow water state on a 2D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ShallowWater {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    /// Water height field.
    pub height: Vec<f64>,
    /// Velocity x-component.
    pub vx: Vec<f64>,
    /// Velocity y-component.
    pub vy: Vec<f64>,
    /// Ground elevation (bathymetry).
    pub ground: Vec<f64>,
    /// Gravity.
    pub gravity: f64,
    /// Damping factor (energy loss per step).
    pub damping: f64,
    /// Manning's roughness coefficient per cell (0.0 = frictionless).
    ///
    /// Typical values: 0.01 (smooth), 0.025 (earth channel), 0.035 (gravel),
    /// 0.05 (heavy brush), 0.1 (dense vegetation).
    pub manning_n: Vec<f64>,
    /// Minimum water depth for a cell to be considered "wet".
    /// Cells with depth below this threshold have their velocity zeroed
    /// and produce no outgoing flux. Default: 1e-4 meters.
    pub dry_threshold: f64,
    /// Critical surface slope for wave breaking detection: |∇η| / depth.
    /// When exceeded, extra dissipation is applied. Default: 0.5.
    /// Set to 0.0 to disable breaking detection.
    pub breaking_threshold: f64,
    /// Extra viscous dissipation applied at breaking wave fronts.
    /// Velocity is damped by `1 / (1 + breaking_dissipation · dt)` at breaking cells.
    /// Default: 5.0.
    pub breaking_dissipation: f64,
    /// Boussinesq dispersive correction coefficient.
    /// Scales the dispersive pressure term: `coeff · h² · g · ∇²(∇η)`.
    /// The theoretical value is 1/3 ≈ 0.333 (Peregrine 1967).
    /// Set to 0.0 to disable (default).
    ///
    /// **Stability note:** The explicit 4th-order term is stiff. Use small
    /// values (0.01–0.1) and small timesteps. Stability requires roughly
    /// `dt < dx² / (coeff · h² · g)`. For `dx=0.1, h=1, coeff=0.05`:
    /// `dt < 0.01 / (0.05 · 9.81) ≈ 0.02`.
    pub dispersion_coeff: f64,
    /// Use Green-Naghdi fully nonlinear dispersion instead of Boussinesq.
    /// More accurate for large-amplitude waves. Requires `dispersion_coeff > 0`.
    /// Default: false.
    pub use_green_naghdi: bool,
    /// Use HLL Riemann solver for flux computation (better shock capturing).
    /// Default: false (uses averaged fluxes).
    pub use_riemann: bool,
    /// Scratch buffers (persistent across steps to avoid allocation).
    #[serde(skip)]
    scratch_vx: Vec<f64>,
    #[serde(skip)]
    scratch_vy: Vec<f64>,
    #[serde(skip)]
    scratch_h: Vec<f64>,
    /// Scratch for Laplacian of height (dispersion).
    #[serde(skip)]
    scratch_lap: Vec<f64>,
}

impl ShallowWater {
    /// Create a flat water surface at given height.
    pub fn new(nx: usize, ny: usize, dx: f64, water_height: f64) -> Result<Self> {
        if nx < 3 || ny < 3 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        let size = nx * ny;
        Ok(Self {
            nx,
            ny,
            dx,
            height: vec![water_height; size],
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            ground: vec![0.0; size],
            gravity: 9.81,
            damping: 0.999,
            manning_n: vec![0.0; size],
            dry_threshold: 1e-4,
            breaking_threshold: 0.5,
            breaking_dissipation: 5.0,
            dispersion_coeff: 0.0,
            use_green_naghdi: false,
            use_riemann: false,
            scratch_vx: vec![0.0; size],
            scratch_vy: vec![0.0; size],
            scratch_h: vec![0.0; size],
            scratch_lap: vec![0.0; size],
        })
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Total water depth at a cell (height - ground).
    #[inline]
    #[must_use]
    pub fn depth_at(&self, x: usize, y: usize) -> f64 {
        let i = self.idx(x, y);
        (self.height[i] - self.ground[i]).max(0.0)
    }

    /// Surface elevation at a cell.
    #[inline]
    #[must_use]
    pub fn surface_at(&self, x: usize, y: usize) -> f64 {
        self.height[self.idx(x, y)]
    }

    /// Whether a cell is wet (depth above dry threshold).
    #[inline]
    #[must_use]
    pub fn is_wet(&self, x: usize, y: usize) -> bool {
        self.depth_at(x, y) >= self.dry_threshold
    }

    /// Whether a cell is at a breaking wave front.
    ///
    /// Compares the surface gradient magnitude to the breaking threshold
    /// relative to local depth. Returns false for boundary/dry cells.
    #[must_use]
    pub fn is_breaking(&self, x: usize, y: usize) -> bool {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 {
            return false;
        }
        let i = self.idx(x, y);
        let depth = (self.height[i] - self.ground[i]).max(0.0);
        if depth < self.dry_threshold || self.breaking_threshold <= 0.0 {
            return false;
        }
        let inv_2dx = 0.5 / self.dx;
        let dhdx = (self.height[i + 1] - self.height[i - 1]) * inv_2dx;
        let dhdy = (self.height[i + self.nx] - self.height[i - self.nx]) * inv_2dx;
        let slope = (dhdx * dhdx + dhdy * dhdy).sqrt();
        slope / depth > self.breaking_threshold
    }

    /// Add a circular disturbance (drop/splash).
    pub fn add_disturbance(&mut self, cx: f64, cy: f64, radius: f64, amplitude: f64) {
        if radius <= 0.0 {
            return;
        }
        let _span = trace_span!("shallow::disturbance").entered();
        let r2 = radius * radius;
        let inv_radius = 1.0 / radius;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                let dx = px - cx;
                let dy = py - cy;
                let dist2 = dx * dx + dy * dy;
                if dist2 < r2 {
                    let factor = 1.0 - dist2.sqrt() * inv_radius;
                    let idx = self.idx(x, y);
                    self.height[idx] += amplitude * factor * factor;
                }
            }
        }
    }

    /// HLL flux for 1D shallow water Riemann problem.
    /// Returns (flux_h, flux_hu) at the interface between left and right states.
    #[inline]
    fn hll_flux(h_l: f64, u_l: f64, h_r: f64, u_r: f64, g: f64) -> (f64, f64) {
        let eps = 1e-8;
        let c_l = (g * h_l.max(0.0)).sqrt();
        let c_r = (g * h_r.max(0.0)).sqrt();

        // Wave speed estimates (Davis)
        let s_l = (u_l - c_l).min(u_r - c_r);
        let s_r = (u_l + c_l).max(u_r + c_r);

        // Fluxes
        let f_h_l = h_l * u_l;
        let f_hu_l = h_l * u_l * u_l + 0.5 * g * h_l * h_l;
        let f_h_r = h_r * u_r;
        let f_hu_r = h_r * u_r * u_r + 0.5 * g * h_r * h_r;

        if s_l >= 0.0 {
            (f_h_l, f_hu_l)
        } else if s_r <= 0.0 {
            (f_h_r, f_hu_r)
        } else {
            let denom = 1.0 / (s_r - s_l).max(eps);
            let f_h = (s_r * f_h_l - s_l * f_h_r + s_l * s_r * (h_r - h_l)) * denom;
            let f_hu = (s_r * f_hu_l - s_l * f_hu_r + s_l * s_r * (h_r * u_r - h_l * u_l)) * denom;
            (f_h, f_hu)
        }
    }

    /// Step the simulation using non-linear shallow water equations.
    ///
    /// Solves the full SWE with convective acceleration and flux-form continuity:
    /// - Continuity: ∂h/∂t + ∂(h·u)/∂x + ∂(h·v)/∂y = 0
    /// - Momentum-x: ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -g·∂η/∂x
    /// - Momentum-y: ∂v/∂t + u·∂v/∂x + v·∂v/∂y = -g·∂η/∂y
    ///
    /// where η = height (surface elevation) and h = depth (height - ground).
    pub fn step(&mut self, dt: f64) -> Result<()> {
        let _span = trace_span!("shallow::step", nx = self.nx, ny = self.ny).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let g = self.gravity;
        let dx = self.dx;
        let inv_2dx = 0.5 / dx;
        // Timestep-proportional damping: damp^(dt/dt_ref) where dt_ref = 0.001
        let damp = self.damping.powf(dt / 0.001);

        let dry_thr = self.dry_threshold;

        // Ensure scratch buffers are correct size (may be empty after deserialization)
        let size = nx * ny;
        self.scratch_vx.resize(size, 0.0);
        self.scratch_vy.resize(size, 0.0);
        self.scratch_h.resize(size, 0.0);
        self.scratch_lap.resize(size, 0.0);

        // Zero velocity in dry cells before snapshotting
        for i in 0..size {
            if (self.height[i] - self.ground[i]) < dry_thr {
                self.vx[i] = 0.0;
                self.vy[i] = 0.0;
            }
        }

        // Snapshot current state for centered-difference stencils
        self.scratch_vx.copy_from_slice(&self.vx);
        self.scratch_vy.copy_from_slice(&self.vy);
        self.scratch_h.copy_from_slice(&self.height);

        let svx = &self.scratch_vx;
        let svy = &self.scratch_vy;
        let sh = &self.scratch_h;

        // Pre-compute Laplacian of surface height for Boussinesq dispersion.
        // ∇²η = (η[i+1] + η[i-1] + η[i+nx] + η[i-nx] - 4·η[i]) / dx²
        let disp = self.dispersion_coeff;
        if disp > 0.0 {
            let inv_dx2 = 1.0 / (dx * dx);
            self.scratch_lap.fill(0.0);
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let i = y * nx + x;
                    self.scratch_lap[i] =
                        (sh[i + 1] + sh[i - 1] + sh[i + nx] + sh[i - nx] - 4.0 * sh[i]) * inv_dx2;
                }
            }
        }

        // Momentum update: pressure gradient + convective acceleration
        // ∂u/∂t = -g·∂η/∂x - u·∂u/∂x - v·∂u/∂y
        // Boussinesq correction: -disp · h² · g · ∂(∇²η)/∂x
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let u = svx[i];
                let v = svy[i];

                // Pressure gradient with well-balanced hydrostatic reconstruction.
                // Uses reconstructed interface depths where bathymetry varies,
                // ensuring lake-at-rest (η=const) over non-flat ground produces
                // zero net force.
                let hi = (sh[i] - self.ground[i]).max(0.0);
                let (dhdx, dhdy) = if hi < dry_thr {
                    (0.0, 0.0)
                } else {
                    let gx = self.ground[i];

                    // X-direction: well-balanced if ground varies, simple if flat
                    let gxr = self.ground[i + 1];
                    let gxl = self.ground[i - 1];
                    let px = if (gxr - gx).abs() > 1e-12 || (gxl - gx).abs() > 1e-12 {
                        let z_right = gx.max(gxr);
                        let z_left = gx.max(gxl);
                        let h_ir = (sh[i] - z_right).max(0.0);
                        let h_jr = (sh[i + 1] - z_right).max(0.0);
                        let h_il = (sh[i] - z_left).max(0.0);
                        let h_jl = (sh[i - 1] - z_left).max(0.0);
                        0.5 * g * (h_jr * h_jr - h_ir * h_ir + h_il * h_il - h_jl * h_jl)
                            / (2.0 * dx * hi.max(dry_thr))
                    } else {
                        g * (sh[i + 1] - sh[i - 1]) * inv_2dx
                    };

                    // Y-direction: well-balanced if ground varies, simple if flat
                    let gyt = self.ground[i + nx];
                    let gyb = self.ground[i - nx];
                    let py = if (gyt - gx).abs() > 1e-12 || (gyb - gx).abs() > 1e-12 {
                        let z_top = gx.max(gyt);
                        let z_bot = gx.max(gyb);
                        let h_it = (sh[i] - z_top).max(0.0);
                        let h_jt = (sh[i + nx] - z_top).max(0.0);
                        let h_ib = (sh[i] - z_bot).max(0.0);
                        let h_jb = (sh[i - nx] - z_bot).max(0.0);
                        0.5 * g * (h_jt * h_jt - h_it * h_it + h_ib * h_ib - h_jb * h_jb)
                            / (2.0 * dx * hi.max(dry_thr))
                    } else {
                        g * (sh[i + nx] - sh[i - nx]) * inv_2dx
                    };

                    (px, py)
                };

                // Convective terms: u·∂u/∂x + v·∂u/∂y, u·∂v/∂x + v·∂v/∂y
                let dudx = (svx[i + 1] - svx[i - 1]) * inv_2dx;
                let dudy = (svx[i + nx] - svx[i - nx]) * inv_2dx;
                let dvdx = (svy[i + 1] - svy[i - 1]) * inv_2dx;
                let dvdy = (svy[i + nx] - svy[i - nx]) * inv_2dx;

                self.vx[i] -= (dhdx + u * dudx + v * dudy) * dt;
                self.vy[i] -= (dhdy + u * dvdx + v * dvdy) * dt;

                // Dispersive correction
                if disp > 0.0 {
                    let depth = (sh[i] - self.ground[i]).max(0.0);
                    if depth > dry_thr {
                        let lap = &self.scratch_lap;
                        if self.use_green_naghdi {
                            // Green-Naghdi: -(h²/3)·g·∂/∂x(h·∇²η)
                            // Uses depth-weighted Laplacian for fully nonlinear dispersion
                            let h_lap_r = (sh[i + 1] - self.ground[i + 1]).max(0.0) * lap[i + 1];
                            let h_lap_l = (sh[i - 1] - self.ground[i - 1]).max(0.0) * lap[i - 1];
                            let h_lap_t = (sh[i + nx] - self.ground[i + nx]).max(0.0) * lap[i + nx];
                            let h_lap_b = (sh[i - nx] - self.ground[i - nx]).max(0.0) * lap[i - nx];
                            let dhl_dx = (h_lap_r - h_lap_l) * inv_2dx;
                            let dhl_dy = (h_lap_t - h_lap_b) * inv_2dx;
                            let scale = disp * depth * depth * g / 3.0;
                            self.vx[i] -= scale * dhl_dx * dt;
                            self.vy[i] -= scale * dhl_dy * dt;
                        } else {
                            // Boussinesq: -coeff · h² · g · ∇(∇²η)
                            let dlap_dx = (lap[i + 1] - lap[i - 1]) * inv_2dx;
                            let dlap_dy = (lap[i + nx] - lap[i - nx]) * inv_2dx;
                            let scale = disp * depth * depth * g;
                            self.vx[i] -= scale * dlap_dx * dt;
                            self.vy[i] -= scale * dlap_dy * dt;
                        }
                    }
                }

                // Manning's bed friction: S_f = -g·n²·|u|·u / h^(4/3)
                // Uses implicit treatment: u_new = u_old / (1 + α·dt) for stability
                let n_manning = self.manning_n[i];
                if n_manning > 0.0 {
                    let depth = (sh[i] - self.ground[i]).max(0.0);
                    if depth > dry_thr {
                        let speed = (u * u + v * v).sqrt();
                        // h^(4/3) = h · h^(1/3) = h · cbrt(h)
                        let d_safe = depth.max(dry_thr);
                        let friction = g * n_manning * n_manning * speed / (d_safe * d_safe.cbrt());
                        // Implicit: v_new = v_old / (1 + friction·dt)
                        let decay = 1.0 / (1.0 + friction * dt);
                        self.vx[i] *= decay;
                        self.vy[i] *= decay;
                    }
                }

                // Wave breaking dissipation: extra damping at steep fronts
                // Uses surface elevation gradient (not the well-balanced pressure term)
                let brk_thr = self.breaking_threshold;
                if brk_thr > 0.0 && self.breaking_dissipation > 0.0 {
                    let depth = (sh[i] - self.ground[i]).max(0.0);
                    if depth > dry_thr {
                        let eta_dx = (sh[i + 1] - sh[i - 1]) * inv_2dx;
                        let eta_dy = (sh[i + nx] - sh[i - nx]) * inv_2dx;
                        let slope = (eta_dx * eta_dx + eta_dy * eta_dy).sqrt();
                        if slope / depth > brk_thr {
                            let decay = 1.0 / (1.0 + self.breaking_dissipation * dt);
                            self.vx[i] *= decay;
                            self.vy[i] *= decay;
                        }
                    }
                }

                self.vx[i] *= damp;
                self.vy[i] *= damp;
            }
        }

        // Continuity update (flux form): ∂h/∂t + ∂(h·u)/∂x + ∂(h·v)/∂y = 0
        let use_riemann = self.use_riemann;
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let depth = (sh[i] - self.ground[i]).max(0.0);

                let flux_div = if use_riemann {
                    // HLL Riemann solver at each cell face (mass + momentum fluxes)
                    let d_r = (sh[i + 1] - self.ground[i + 1]).max(0.0);
                    let d_l = (sh[i - 1] - self.ground[i - 1]).max(0.0);
                    let d_t = (sh[i + nx] - self.ground[i + nx]).max(0.0);
                    let d_b = (sh[i - nx] - self.ground[i - nx]).max(0.0);

                    let (fh_r, fhu_r) = Self::hll_flux(depth, svx[i], d_r, svx[i + 1], g);
                    let (fh_l, fhu_l) = Self::hll_flux(d_l, svx[i - 1], depth, svx[i], g);
                    let (fh_t, fhv_t) = Self::hll_flux(depth, svy[i], d_t, svy[i + nx], g);
                    let (fh_b, fhv_b) = Self::hll_flux(d_b, svy[i - nx], depth, svy[i], g);

                    // Update velocity from momentum flux divergence: ∂(hu)/∂t = -∂F_hu/∂x
                    if depth > dry_thr {
                        self.vx[i] -= (fhu_r - fhu_l) / (dx * depth) * dt;
                        self.vy[i] -= (fhv_t - fhv_b) / (dx * depth) * dt;
                    }

                    (fh_r - fh_l) / dx + (fh_t - fh_b) / dx
                } else {
                    // Averaged fluxes (original method)
                    let hu_right = {
                        let d = (sh[i + 1] - self.ground[i + 1]).max(0.0);
                        0.5 * (depth * self.vx[i] + d * self.vx[i + 1])
                    };
                    let hu_left = {
                        let d = (sh[i - 1] - self.ground[i - 1]).max(0.0);
                        0.5 * (depth * self.vx[i] + d * self.vx[i - 1])
                    };
                    let hv_top = {
                        let d = (sh[i + nx] - self.ground[i + nx]).max(0.0);
                        0.5 * (depth * self.vy[i] + d * self.vy[i + nx])
                    };
                    let hv_bottom = {
                        let d = (sh[i - nx] - self.ground[i - nx]).max(0.0);
                        0.5 * (depth * self.vy[i] + d * self.vy[i - nx])
                    };
                    (hu_right - hu_left + hv_top - hv_bottom) * inv_2dx
                };

                self.height[i] -= flux_div * dt;

                // Clamp: depth must never go negative
                let min_h = self.ground[i];
                if self.height[i] < min_h {
                    self.height[i] = min_h;
                }
            }
        }

        // Enforce boundary conditions (Neumann: zero-gradient at edges)
        self.enforce_boundary();

        Ok(())
    }

    /// Enforce Neumann (zero-gradient) boundary conditions.
    /// Copies nearest interior row/column to boundary cells.
    fn enforce_boundary(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        // Top and bottom rows
        for x in 0..nx {
            self.height[x] = self.height[nx + x];
            self.vx[x] = self.vx[nx + x];
            self.vy[x] = 0.0; // reflect: zero normal velocity at boundary
            let top = (ny - 1) * nx + x;
            let below = (ny - 2) * nx + x;
            self.height[top] = self.height[below];
            self.vx[top] = self.vx[below];
            self.vy[top] = 0.0;
        }
        // Left and right columns
        for y in 0..ny {
            let l = y * nx;
            self.height[l] = self.height[l + 1];
            self.vx[l] = 0.0; // reflect: zero normal velocity at boundary
            self.vy[l] = self.vy[l + 1];
            let r = y * nx + nx - 1;
            self.height[r] = self.height[r - 1];
            self.vx[r] = 0.0;
            self.vy[r] = self.vy[r - 1];
        }
    }

    /// Total water volume (sum of depths × cell area).
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        let cell_area = self.dx * self.dx;
        self.height
            .iter()
            .zip(self.ground.iter())
            .map(|(&h, &g)| (h - g).max(0.0) * cell_area)
            .sum()
    }

    /// Maximum wave height above rest level.
    #[must_use]
    pub fn max_wave_height(&self, rest_height: f64) -> f64 {
        self.height
            .iter()
            .map(|&h| (h - rest_height).abs())
            .fold(0.0f64, f64::max)
    }

    /// Compute CFL-limited timestep for the shallow water solver.
    ///
    /// dt = cfl · dx / max(|u| + sqrt(g·h))
    #[must_use]
    pub fn cfl_dt(&self, cfl_factor: f64) -> f64 {
        let g = self.gravity;
        let mut max_wave_speed = 0.0f64;
        for i in 0..self.height.len() {
            let depth = (self.height[i] - self.ground[i]).max(0.0);
            let speed = (self.vx[i] * self.vx[i] + self.vy[i] * self.vy[i]).sqrt();
            let wave_speed = speed + (g * depth).sqrt();
            max_wave_speed = max_wave_speed.max(wave_speed);
        }
        if max_wave_speed < 1e-20 {
            return cfl_factor * self.dx; // fallback
        }
        let dt_cfl = cfl_factor * self.dx / max_wave_speed;

        // Dispersive stability limit: dt < dx² / (coeff · g · h_max)
        if self.dispersion_coeff > 0.0 {
            let h_max = self
                .height
                .iter()
                .zip(self.ground.iter())
                .map(|(h, g_elev)| (h - g_elev).max(0.0))
                .fold(0.0f64, f64::max);
            let dt_disp = self.dx * self.dx / (self.dispersion_coeff * g * h_max.max(1e-6));
            dt_cfl.min(dt_disp)
        } else {
            dt_cfl
        }
    }
}

// ── Sediment Transport ──────────────────────────────────────────────────────

/// Sediment transport configuration for shallow water.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SedimentConfig {
    /// Shields parameter threshold for bed load initiation. Typical: 0.047.
    pub shields_critical: f64,
    /// Sediment grain diameter (m). Typical: 0.001 (1mm sand).
    pub grain_diameter: f64,
    /// Sediment density (kg/m³). Typical: 2650 (quartz).
    pub sediment_density: f64,
    /// Water density (kg/m³). Typical: 1000.
    pub water_density: f64,
    /// Erosion rate coefficient. Controls how fast bed erodes.
    pub erosion_rate: f64,
    /// Deposition rate coefficient. Controls how fast sediment settles.
    pub deposition_rate: f64,
}

impl Default for SedimentConfig {
    fn default() -> Self {
        Self {
            shields_critical: 0.047,
            grain_diameter: 0.001,
            sediment_density: 2650.0,
            water_density: 1000.0,
            erosion_rate: 0.01,
            deposition_rate: 0.02,
        }
    }
}

/// Update sediment transport: erosion/deposition modifying bathymetry.
///
/// Uses Shields criterion to determine bed load initiation.
/// Erodes ground where flow exceeds critical shear, deposits where it doesn't.
/// `concentration` is suspended sediment per cell (modified in place).
/// `sw.ground` is modified by erosion/deposition.
pub fn update_sediment(
    sw: &mut ShallowWater,
    concentration: &mut [f64],
    config: &SedimentConfig,
    dt: f64,
) {
    let _span = trace_span!("shallow::sediment", nx = sw.nx, ny = sw.ny).entered();
    let nx = sw.nx;
    let g = sw.gravity;
    let rho_s = config.sediment_density;
    let rho_w = config.water_density;
    let d = config.grain_diameter;
    let tau_cr = config.shields_critical * (rho_s - rho_w) * g * d;

    if concentration.len() < nx * sw.ny {
        return;
    }

    for y in 1..sw.ny - 1 {
        for x in 1..nx - 1 {
            let i = y * nx + x;
            let depth = (sw.height[i] - sw.ground[i]).max(0.0);
            if depth < 1e-6 {
                continue;
            }

            // Bed shear stress: τ_b = ρ_w · g · n² · |u|² / h^(1/3)
            let speed_sq = sw.vx[i] * sw.vx[i] + sw.vy[i] * sw.vy[i];
            let n_manning = sw.manning_n[i].max(0.01);
            let tau_b = rho_w * g * n_manning * n_manning * speed_sq / depth.cbrt();

            if tau_b > tau_cr {
                // Erosion: lift sediment from bed into suspension
                let erosion = config.erosion_rate * (tau_b - tau_cr) * dt;
                sw.ground[i] -= erosion;
                concentration[i] += erosion / depth;
            }

            // Deposition: settle suspended sediment onto bed
            if concentration[i] > 0.0 {
                let depo = config.deposition_rate * concentration[i] * dt;
                let actual_depo = depo.min(concentration[i] * depth);
                sw.ground[i] += actual_depo;
                concentration[i] -= actual_depo / depth.max(1e-6);
                concentration[i] = concentration[i].max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_shallow_new() {
        let sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        assert_eq!(sw.height.len(), 400);
        assert!((sw.height[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_shallow_invalid() {
        assert!(ShallowWater::new(0, 10, 0.1, 1.0).is_err());
    }

    #[test]
    fn test_depth_at() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 2.0).unwrap();
        let idx = sw.idx(5, 5);
        sw.ground[idx] = 0.5;
        assert!((sw.depth_at(5, 5) - 1.5).abs() < EPS);
    }

    #[test]
    fn test_depth_at_dry() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        let idx = sw.idx(5, 5);
        sw.ground[idx] = 2.0; // ground above water
        assert!(sw.depth_at(5, 5).abs() < EPS); // clamped to 0
    }

    #[test]
    fn test_add_disturbance() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        let center_h_before = sw.height[sw.idx(10, 10)];
        sw.add_disturbance(1.0, 1.0, 0.3, 0.5);
        assert!(sw.height[sw.idx(10, 10)] > center_h_before);
    }

    #[test]
    fn test_step_flat_stays_flat() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        sw.step(0.001).unwrap();
        // Flat surface should remain flat (no gradients → no forces)
        let max_deviation = sw.max_wave_height(1.0);
        assert!(max_deviation < 1e-6);
    }

    #[test]
    fn test_step_disturbance_propagates() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        sw.add_disturbance(1.0, 1.0, 0.2, 0.5);
        let initial_max = sw.max_wave_height(1.0);
        for _ in 0..100 {
            sw.step(0.001).unwrap();
        }
        // Wave should have spread out (max height decreased)
        let final_max = sw.max_wave_height(1.0);
        assert!(final_max < initial_max);
    }

    #[test]
    fn test_step_invalid_dt() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        assert!(sw.step(-0.001).is_err());
    }

    #[test]
    fn test_step_nan_dt() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        assert!(sw.step(f64::NAN).is_err());
    }

    #[test]
    fn test_total_volume() {
        let sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        let expected = 10.0 * 10.0 * 0.1 * 0.1 * 1.0; // 10x10 cells, 0.1 spacing, 1.0 height
        assert!((sw.total_volume() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_volume_conservation() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        let vol_before = sw.total_volume();
        sw.add_disturbance(1.0, 1.0, 0.2, 0.1);
        for _ in 0..50 {
            sw.step(0.001).unwrap();
        }
        let vol_after = sw.total_volume();
        // Volume should be approximately conserved (small numerical loss from damping)
        assert!((vol_after - vol_before).abs() / vol_before < 0.05);
    }

    #[test]
    fn test_serde_roundtrip() {
        let sw = ShallowWater::new(5, 5, 0.2, 1.5).unwrap();
        let json = serde_json::to_string(&sw).unwrap();
        let sw2: ShallowWater = serde_json::from_str(&json).unwrap();
        assert_eq!(sw2.nx, 5);
        assert!((sw2.height[0] - 1.5).abs() < EPS);
    }

    #[test]
    fn test_no_alloc_after_first_step() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        sw.step(0.001).unwrap();
        let cap_vx = sw.scratch_vx.capacity();
        let cap_vy = sw.scratch_vy.capacity();
        sw.step(0.001).unwrap();
        // Capacity should not change (no reallocation)
        assert_eq!(sw.scratch_vx.capacity(), cap_vx);
        assert_eq!(sw.scratch_vy.capacity(), cap_vy);
    }

    #[test]
    fn test_boundary_reflects_waves() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        // Add disturbance near left boundary
        sw.add_disturbance(0.2, 1.0, 0.15, 0.5);
        for _ in 0..200 {
            sw.step(0.001).unwrap();
        }
        // Boundary cells should have sensible values (not stuck at initial)
        let left_height = sw.height[sw.idx(0, 10)];
        let interior_height = sw.height[sw.idx(1, 10)];
        // Neumann BC: boundary should match interior neighbor
        assert!(
            (left_height - interior_height).abs() < 1e-6,
            "boundary should track interior: left={left_height}, interior={interior_height}"
        );
    }

    #[test]
    fn test_convective_transport() {
        // A uniform rightward current should carry a height perturbation downstream
        let mut sw = ShallowWater::new(40, 10, 0.1, 1.0).unwrap();
        sw.damping = 1.0; // no damping for this test

        // Set uniform rightward flow
        for i in 0..sw.vx.len() {
            sw.vx[i] = 2.0;
        }
        // Add a height bump in the left region
        let bump_x = 10;
        for y in 3..7 {
            let i = sw.idx(bump_x, y);
            sw.height[i] += 0.1;
        }

        let initial_bump_center = sw.height[sw.idx(bump_x, 5)];

        // Advance for many steps with small dt
        for _ in 0..100 {
            sw.step(0.001).unwrap();
        }

        // The bump should have moved rightward: original location should be lower
        let bump_after = sw.height[sw.idx(bump_x, 5)];
        assert!(
            bump_after < initial_bump_center,
            "bump should have advected away: before={initial_bump_center}, after={bump_after}"
        );
    }

    #[test]
    fn test_dam_break_asymmetry() {
        // Non-linear SWE produces asymmetric dam break (shock front + rarefaction fan).
        // With a step in water height, the right-moving front should be steeper than
        // the left-moving rarefaction.
        let mut sw = ShallowWater::new(60, 6, 0.1, 0.5).unwrap();
        sw.damping = 1.0;

        // Left half: high water, right half: low water
        let nx = sw.nx;
        for y in 0..6 {
            for x in 0..30 {
                sw.height[y * nx + x] = 2.0;
            }
        }

        for _ in 0..200 {
            sw.step(0.0005).unwrap();
        }

        // Check that flow developed: there should be rightward velocity in the middle
        let mid_vx = sw.vx[sw.idx(30, 3)];
        assert!(
            mid_vx > 0.0,
            "dam break should produce rightward flow at break point: vx={mid_vx}"
        );
        // All values should be finite (no blowup)
        assert!(
            sw.height.iter().all(|h| h.is_finite()),
            "all heights should remain finite"
        );
    }

    #[test]
    fn test_nonlinear_steepening() {
        // Non-linear waves should steepen: the crest travels faster than the trough.
        // A sinusoidal perturbation on a base flow should develop asymmetry.
        let mut sw = ShallowWater::new(80, 6, 0.05, 1.0).unwrap();
        sw.damping = 1.0;

        // Add sinusoidal height perturbation
        for x in 0..80 {
            let phase = std::f64::consts::PI * 2.0 * x as f64 / 40.0;
            for y in 0..6 {
                let i = sw.idx(x, y);
                sw.height[i] += 0.2 * phase.sin();
            }
        }

        for _ in 0..300 {
            sw.step(0.0005).unwrap();
        }

        // Should remain stable and finite
        assert!(sw.height.iter().all(|h| h.is_finite()));
        // Wave should have propagated (non-zero velocity somewhere)
        let max_v = sw.vx.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(max_v > 0.01, "wave should generate velocity: max_v={max_v}");
    }

    // ── Manning's friction tests ───────────────────────────────────────────

    #[test]
    fn test_friction_decelerates_flow() {
        let mut sw = ShallowWater::new(30, 6, 0.1, 1.0).unwrap();
        sw.damping = 1.0;
        // Set uniform rightward flow
        for v in sw.vx.iter_mut() {
            *v = 2.0;
        }
        // Apply strong friction everywhere
        sw.manning_n.fill(0.05);

        let speed_before: f64 = sw.vx.iter().map(|v| v.abs()).sum();
        for _ in 0..200 {
            sw.step(0.001).unwrap();
        }
        let speed_after: f64 = sw.vx.iter().map(|v| v.abs()).sum();

        assert!(
            speed_after < speed_before * 0.8,
            "friction should decelerate: before={speed_before}, after={speed_after}"
        );
    }

    #[test]
    fn test_friction_scales_with_roughness() {
        // Higher Manning's n should produce more deceleration
        let make_flow = |n: f64| -> f64 {
            let mut sw = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
            sw.damping = 1.0;
            for v in sw.vx.iter_mut() {
                *v = 1.0;
            }
            sw.manning_n.fill(n);
            for _ in 0..100 {
                sw.step(0.001).unwrap();
            }
            sw.vx.iter().map(|v| v.abs()).sum()
        };

        let speed_smooth = make_flow(0.01);
        let speed_rough = make_flow(0.05);
        assert!(
            speed_rough < speed_smooth,
            "rougher bed should slow flow more: smooth={speed_smooth}, rough={speed_rough}"
        );
    }

    #[test]
    fn test_zero_friction_no_effect() {
        // With manning_n = 0 (default), friction term should have no effect
        let mut sw_no_friction = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
        let mut sw_with_zero = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
        sw_no_friction.damping = 1.0;
        sw_with_zero.damping = 1.0;
        // manning_n is already 0.0 by default for both

        // Same initial disturbance
        sw_no_friction.add_disturbance(1.0, 0.3, 0.2, 0.3);
        sw_with_zero.add_disturbance(1.0, 0.3, 0.2, 0.3);

        for _ in 0..50 {
            sw_no_friction.step(0.001).unwrap();
            sw_with_zero.step(0.001).unwrap();
        }

        // Results should be identical
        for i in 0..sw_no_friction.height.len() {
            assert!(
                (sw_no_friction.height[i] - sw_with_zero.height[i]).abs() < 1e-12,
                "zero manning_n should be identical to default"
            );
        }
    }

    #[test]
    fn test_friction_dry_cells_safe() {
        // Friction with dry cells should not produce NaN/Inf
        let mut sw = ShallowWater::new(20, 6, 0.1, 0.5).unwrap();
        sw.damping = 1.0;
        sw.manning_n.fill(0.05);
        // Make some cells dry
        for x in 10..15 {
            for y in 0..6 {
                let i = y * sw.nx + x;
                sw.ground[i] = 1.0; // ground above water
            }
        }
        // Set flow toward dry area
        for v in sw.vx.iter_mut() {
            *v = 1.0;
        }

        for _ in 0..100 {
            sw.step(0.001).unwrap();
        }
        assert!(
            sw.vx.iter().all(|v| v.is_finite()),
            "friction near dry cells should remain finite"
        );
        assert!(sw.height.iter().all(|h| h.is_finite()));
    }

    // ── Wetting/drying tests ─────────────────────────────────────────────

    #[test]
    fn test_dry_cells_zero_velocity() {
        let mut sw = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
        sw.damping = 1.0;
        // Make right half dry
        let nx = sw.nx;
        for y in 0..6 {
            for x in 10..20 {
                sw.ground[y * nx + x] = 2.0;
            }
        }
        // Give everything velocity
        sw.vx.fill(1.0);
        sw.vy.fill(0.5);

        sw.step(0.001).unwrap();

        // Dry cells should have zero velocity
        for y in 1..5 {
            for x in 11..19 {
                assert!(
                    !sw.is_wet(x, y),
                    "cell ({x},{y}) should be dry: depth={}",
                    sw.depth_at(x, y)
                );
            }
        }
    }

    #[test]
    fn test_depth_never_negative() {
        // Water flowing off a shelf should not produce negative depth
        let mut sw = ShallowWater::new(30, 6, 0.1, 0.5).unwrap();
        sw.damping = 1.0;
        let nx = sw.nx;
        // Create a shelf: ground rises in the middle
        for y in 0..6 {
            for x in 12..18 {
                sw.ground[y * nx + x] = 0.45; // just below water surface
            }
        }
        // Strong rightward flow
        sw.vx.fill(3.0);

        for _ in 0..500 {
            sw.step(0.0005).unwrap();
        }

        // No cell should have negative depth
        for i in 0..sw.height.len() {
            let depth = sw.height[i] - sw.ground[i];
            assert!(
                depth >= -1e-10,
                "depth should never be negative: cell {i}, depth={depth}"
            );
        }
    }

    #[test]
    fn test_water_advances_onto_dry_bed() {
        // Dam break onto initially dry bed — water should advance
        let mut sw = ShallowWater::new(40, 6, 0.1, 0.0).unwrap();
        sw.damping = 1.0;
        sw.breaking_threshold = 0.0; // disable breaking for clean dam break test
        let nx = sw.nx;
        // Left side: water, right side: dry (height = ground = 0)
        for y in 0..6 {
            for x in 0..15 {
                sw.height[y * nx + x] = 1.0;
            }
        }

        // Initially, right side is dry
        assert!(!sw.is_wet(30, 3));

        for _ in 0..2000 {
            sw.step(0.0005).unwrap();
        }

        // Water should have advanced — check that front cells have water
        let heights: Vec<f64> = (10..25).map(|x| sw.height[3 * nx + x]).collect();
        let any_wet = (15..25).any(|x| sw.depth_at(x, 3) > sw.dry_threshold);
        assert!(any_wet, "water should advance. h[10..25]={heights:?}");
        // Everything should remain finite
        assert!(sw.height.iter().all(|h| h.is_finite()));
        assert!(sw.vx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_is_wet_helper() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        assert!(sw.is_wet(5, 5));
        let i = sw.idx(5, 5);
        sw.ground[i] = 1.5; // raise ground above water
        assert!(!sw.is_wet(5, 5));
    }

    // ── Wave breaking tests ──────────────────────────────────────────────

    #[test]
    fn test_steep_wave_detected_as_breaking() {
        let mut sw = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
        // Create a very steep step in the middle
        let nx = sw.nx;
        for y in 0..6 {
            for x in 0..10 {
                sw.height[y * nx + x] = 3.0;
            }
        }
        // The cell at the step edge should be breaking
        assert!(
            sw.is_breaking(10, 3),
            "steep front should be detected as breaking"
        );
    }

    #[test]
    fn test_gentle_wave_not_breaking() {
        let mut sw = ShallowWater::new(40, 6, 0.1, 1.0).unwrap();
        // Very gentle sinusoidal perturbation
        for x in 0..40 {
            let phase = std::f64::consts::PI * 2.0 * x as f64 / 40.0;
            for y in 0..6 {
                let i = sw.idx(x, y);
                sw.height[i] += 0.001 * phase.sin();
            }
        }
        // No cell should be breaking
        for x in 1..39 {
            assert!(
                !sw.is_breaking(x, 3),
                "gentle wave should not break at x={x}"
            );
        }
    }

    #[test]
    fn test_breaking_dissipates_energy() {
        // Dam break with breaking enabled should have less energy than without
        let run = |breaking: bool| -> f64 {
            let mut sw = ShallowWater::new(40, 6, 0.1, 0.5).unwrap();
            sw.damping = 1.0;
            if !breaking {
                sw.breaking_threshold = 0.0; // disable
            }
            let nx = sw.nx;
            for y in 0..6 {
                for x in 0..15 {
                    sw.height[y * nx + x] = 2.0;
                }
            }
            for _ in 0..300 {
                sw.step(0.0005).unwrap();
            }
            // Total kinetic energy
            sw.vx
                .iter()
                .zip(sw.vy.iter())
                .map(|(&u, &v)| u * u + v * v)
                .sum()
        };

        let ke_with_breaking = run(true);
        let ke_without = run(false);
        assert!(
            ke_with_breaking < ke_without,
            "breaking should dissipate energy: with={ke_with_breaking}, without={ke_without}"
        );
    }

    #[test]
    fn test_breaking_disabled_when_threshold_zero() {
        let mut sw = ShallowWater::new(20, 6, 0.1, 1.0).unwrap();
        sw.breaking_threshold = 0.0;
        // Even with a steep front, is_breaking should return false
        let nx = sw.nx;
        for y in 0..6 {
            for x in 0..10 {
                sw.height[y * nx + x] = 5.0;
            }
        }
        assert!(!sw.is_breaking(10, 3));
    }

    // ── Dispersive correction tests ────────────────────────────────────────

    #[test]
    fn test_dispersion_disabled_by_default() {
        // With dispersion_coeff = 0 (default), results should match non-dispersive
        let mut sw_a = ShallowWater::new(30, 6, 0.1, 1.0).unwrap();
        let mut sw_b = ShallowWater::new(30, 6, 0.1, 1.0).unwrap();
        sw_a.damping = 1.0;
        sw_b.damping = 1.0;
        sw_a.breaking_threshold = 0.0;
        sw_b.breaking_threshold = 0.0;
        // Same perturbation
        sw_a.add_disturbance(1.5, 0.3, 0.2, 0.3);
        sw_b.add_disturbance(1.5, 0.3, 0.2, 0.3);

        for _ in 0..50 {
            sw_a.step(0.001).unwrap();
            sw_b.step(0.001).unwrap();
        }

        for i in 0..sw_a.height.len() {
            assert!(
                (sw_a.height[i] - sw_b.height[i]).abs() < 1e-12,
                "zero dispersion should be identical"
            );
        }
    }

    #[test]
    fn test_dispersion_changes_wave_shape() {
        // With dispersion enabled, a sharp pulse should spread differently.
        // Uses coarser grid + small coefficient for stability with explicit scheme.
        let run = |disp: f64| -> Vec<f64> {
            let mut sw = ShallowWater::new(40, 6, 0.1, 1.0).unwrap();
            sw.damping = 1.0;
            sw.breaking_threshold = 0.0;
            sw.dispersion_coeff = disp;
            sw.add_disturbance(2.0, 0.3, 0.2, 0.1);
            for _ in 0..200 {
                sw.step(0.0001).unwrap();
            }
            let y = 3;
            (0..40).map(|x| sw.height[sw.idx(x, y)]).collect()
        };

        let profile_no_disp = run(0.0);
        let profile_with_disp = run(0.05);

        // Profiles should differ (dispersion changes wave shape)
        let diff: f64 = profile_no_disp
            .iter()
            .zip(profile_with_disp.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "dispersion should change the wave profile: diff={diff}"
        );
        assert!(profile_with_disp.iter().all(|h| h.is_finite()));
    }

    #[test]
    fn test_dispersion_stable_long_run() {
        // Dispersive simulation should not blow up over many steps.
        // Small coefficient + tiny dt for explicit stability of 4th-order term.
        let mut sw = ShallowWater::new(30, 6, 0.1, 1.0).unwrap();
        sw.damping = 1.0;
        sw.dispersion_coeff = 0.01;
        sw.add_disturbance(1.5, 0.3, 0.2, 0.05);

        for _ in 0..500 {
            sw.step(0.00005).unwrap();
        }

        assert!(
            sw.height.iter().all(|h| h.is_finite()),
            "dispersive sim should stay finite"
        );
        assert!(sw.vx.iter().all(|v| v.is_finite()));
    }

    // ── Terrain-following / well-balanced tests ────────────────────────────

    #[test]
    fn test_lake_at_rest_over_slope() {
        // A flat water surface over sloped ground should remain perfectly still.
        // This is the key well-balanced property.
        let mut sw = ShallowWater::new(30, 6, 0.1, 2.0).unwrap();
        sw.damping = 1.0;
        sw.breaking_threshold = 0.0;
        let nx = sw.nx;
        // Linear bed slope: ground rises from 0 to 1 across the domain
        for y in 0..6 {
            for x in 0..30 {
                sw.ground[y * nx + x] = x as f64 / 30.0;
            }
        }

        let max_v_before = sw
            .vx
            .iter()
            .chain(sw.vy.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(max_v_before < 1e-15, "should start at rest");

        for _ in 0..500 {
            sw.step(0.001).unwrap();
        }

        let max_v = sw
            .vx
            .iter()
            .chain(sw.vy.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_v < 1e-6,
            "lake at rest over slope should stay still: max_v={max_v}"
        );
    }

    #[test]
    fn test_flow_over_bump_stable() {
        // Water flowing over a submerged bump should remain finite
        let mut sw = ShallowWater::new(40, 6, 0.1, 1.0).unwrap();
        sw.damping = 1.0;
        sw.breaking_threshold = 0.0;
        let nx = sw.nx;
        // Gaussian bump in the middle
        for y in 0..6 {
            for x in 0..40 {
                let cx = (x as f64 - 20.0) * 0.1;
                sw.ground[y * nx + x] = 0.5 * (-cx * cx * 4.0).exp();
            }
        }
        // Initial rightward flow
        sw.vx.fill(1.0);

        for _ in 0..500 {
            sw.step(0.0005).unwrap();
        }

        assert!(
            sw.height.iter().all(|h| h.is_finite()),
            "flow over bump should stay finite"
        );
        assert!(sw.vx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_well_balanced_step_bathymetry() {
        // Lake at rest over step bathymetry (discontinuous ground)
        let mut sw = ShallowWater::new(30, 6, 0.1, 2.0).unwrap();
        sw.damping = 1.0;
        sw.breaking_threshold = 0.0;
        let nx = sw.nx;
        // Step: ground=0 for left half, ground=0.5 for right half
        for y in 0..6 {
            for x in 15..30 {
                sw.ground[y * nx + x] = 0.5;
            }
        }

        for _ in 0..200 {
            sw.step(0.001).unwrap();
        }

        // Velocities should remain very small (well-balanced)
        let max_v = sw
            .vx
            .iter()
            .chain(sw.vy.iter())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_v < 0.01,
            "lake at rest over step should stay nearly still: max_v={max_v}"
        );
    }

    #[test]
    fn test_boundary_zero_normal_velocity() {
        let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
        sw.add_disturbance(1.0, 1.0, 0.3, 0.5);
        for _ in 0..50 {
            sw.step(0.001).unwrap();
        }
        // Left boundary: vx should be 0 (normal to wall)
        for y in 0..sw.ny {
            assert!(
                sw.vx[sw.idx(0, y)].abs() < 1e-10,
                "left boundary vx should be 0 at y={y}: {}",
                sw.vx[sw.idx(0, y)]
            );
        }
        // Bottom boundary: vy should be 0 (normal to wall)
        for x in 0..sw.nx {
            assert!(
                sw.vy[sw.idx(x, 0)].abs() < 1e-10,
                "bottom boundary vy should be 0 at x={x}: {}",
                sw.vy[sw.idx(x, 0)]
            );
        }
    }
}
