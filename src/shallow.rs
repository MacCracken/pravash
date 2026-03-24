//! Shallow water equations — 2D surface wave simulation.
//!
//! Models water surface height and horizontal velocities.
//! Good for oceans, lakes, puddles, and flooding effects.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

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
        })
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Total water depth at a cell (height - ground).
    #[inline]
    pub fn depth_at(&self, x: usize, y: usize) -> f64 {
        let i = self.idx(x, y);
        (self.height[i] - self.ground[i]).max(0.0)
    }

    /// Surface elevation at a cell.
    #[inline]
    pub fn surface_at(&self, x: usize, y: usize) -> f64 {
        self.height[self.idx(x, y)]
    }

    /// Add a circular disturbance (drop/splash).
    pub fn add_disturbance(&mut self, cx: f64, cy: f64, radius: f64, amplitude: f64) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                let dx = px - cx;
                let dy = py - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < radius {
                    let factor = 1.0 - dist / radius;
                    self.height[self.idx(x, y)] += amplitude * factor * factor;
                }
            }
        }
    }

    /// Step the simulation using a simple finite-difference scheme.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        if dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let g = self.gravity;
        let dx = self.dx;

        // Update velocities from height gradient
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let dhdx = (self.height[i + 1] - self.height[i - 1]) / (2.0 * dx);
                let dhdy = (self.height[i + nx] - self.height[i - nx]) / (2.0 * dx);
                self.vx[i] -= g * dhdx * dt;
                self.vy[i] -= g * dhdy * dt;
                self.vx[i] *= self.damping;
                self.vy[i] *= self.damping;
            }
        }

        // Update height from velocity divergence
        let vx_snapshot = self.vx.clone();
        let vy_snapshot = self.vy.clone();
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let depth = (self.height[i] - self.ground[i]).max(0.01);
                let dvx = (vx_snapshot[i + 1] - vx_snapshot[i - 1]) / (2.0 * dx);
                let dvy = (vy_snapshot[i + nx] - vy_snapshot[i - nx]) / (2.0 * dx);
                self.height[i] -= depth * (dvx + dvy) * dt;
            }
        }

        Ok(())
    }

    /// Total water volume (sum of depths × cell area).
    pub fn total_volume(&self) -> f64 {
        let cell_area = self.dx * self.dx;
        self.height
            .iter()
            .zip(self.ground.iter())
            .map(|(&h, &g)| (h - g).max(0.0) * cell_area)
            .sum()
    }

    /// Maximum wave height above rest level.
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
        sw.ground[sw.idx(5, 5)] = 0.5;
        assert!((sw.depth_at(5, 5) - 1.5).abs() < EPS);
    }

    #[test]
    fn test_depth_at_dry() {
        let mut sw = ShallowWater::new(10, 10, 0.1, 1.0).unwrap();
        sw.ground[sw.idx(5, 5)] = 2.0; // ground above water
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
}
