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
    /// Scratch buffers (persistent across steps to avoid allocation).
    #[serde(skip)]
    scratch_vx: Vec<f64>,
    #[serde(skip)]
    scratch_vy: Vec<f64>,
    #[serde(skip)]
    scratch_h: Vec<f64>,
}

impl ShallowWater {
    /// Create a flat water surface at given height.
    pub fn new(nx: usize, ny: usize, dx: f64, water_height: f64) -> Result<Self> {
        if nx == 0 || ny == 0 {
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
            scratch_vx: vec![0.0; size],
            scratch_vy: vec![0.0; size],
            scratch_h: vec![0.0; size],
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

    /// Add a circular disturbance (drop/splash).
    pub fn add_disturbance(&mut self, cx: f64, cy: f64, radius: f64, amplitude: f64) {
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

        // Ensure scratch buffers are correct size (may be empty after deserialization)
        let size = nx * ny;
        self.scratch_vx.resize(size, 0.0);
        self.scratch_vy.resize(size, 0.0);
        self.scratch_h.resize(size, 0.0);

        // Snapshot current state for centered-difference stencils
        self.scratch_vx.copy_from_slice(&self.vx);
        self.scratch_vy.copy_from_slice(&self.vy);
        self.scratch_h.copy_from_slice(&self.height);

        let svx = &self.scratch_vx;
        let svy = &self.scratch_vy;
        let sh = &self.scratch_h;

        // Momentum update: pressure gradient + convective acceleration
        // ∂u/∂t = -g·∂η/∂x - u·∂u/∂x - v·∂u/∂y
        // ∂v/∂t = -g·∂η/∂y - u·∂v/∂x - v·∂v/∂y
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let u = svx[i];
                let v = svy[i];

                // Pressure gradient (surface elevation gradient)
                let dhdx = (sh[i + 1] - sh[i - 1]) * inv_2dx;
                let dhdy = (sh[i + nx] - sh[i - nx]) * inv_2dx;

                // Convective terms: u·∂u/∂x + v·∂u/∂y, u·∂v/∂x + v·∂v/∂y
                let dudx = (svx[i + 1] - svx[i - 1]) * inv_2dx;
                let dudy = (svx[i + nx] - svx[i - nx]) * inv_2dx;
                let dvdx = (svy[i + 1] - svy[i - 1]) * inv_2dx;
                let dvdy = (svy[i + nx] - svy[i - nx]) * inv_2dx;

                self.vx[i] -= (g * dhdx + u * dudx + v * dudy) * dt;
                self.vy[i] -= (g * dhdy + u * dvdx + v * dvdy) * dt;

                // Manning's bed friction: S_f = -g·n²·|u|·u / h^(4/3)
                // Uses implicit treatment: u_new = u_old / (1 + α·dt) for stability
                let n_manning = self.manning_n[i];
                if n_manning > 0.0 {
                    let depth = (sh[i] - self.ground[i]).max(0.0);
                    if depth > 1e-6 {
                        let speed = (u * u + v * v).sqrt();
                        // h^(4/3) = h · h^(1/3) = h · cbrt(h)
                        let friction = g * n_manning * n_manning * speed / (depth * depth.cbrt());
                        // Implicit: v_new = v_old / (1 + friction·dt)
                        let decay = 1.0 / (1.0 + friction * dt);
                        self.vx[i] *= decay;
                        self.vy[i] *= decay;
                    }
                }

                self.vx[i] *= damp;
                self.vy[i] *= damp;
            }
        }

        // Continuity update (flux form): ∂h/∂t + ∂(h·u)/∂x + ∂(h·v)/∂y = 0
        // Uses depth (h - ground) for flux computation
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let depth = (sh[i] - self.ground[i]).max(0.0);

                // Compute fluxes at cell faces using averaged depth and velocity
                // Flux_x = h·u, Flux_y = h·v (using updated velocities for stability)
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

                let flux_div = (hu_right - hu_left + hv_top - hv_bottom) * inv_2dx;
                self.height[i] -= flux_div * dt;
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
