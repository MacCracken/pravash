//! Volume of Fluid (VOF) — grid-based free surface tracking.
//!
//! Tracks the volume fraction α ∈ [0, 1] per cell where 0 = empty, 1 = full.
//! Interface cells have 0 < α < 1. Uses simple advection of the volume
//! fraction field with donor-acceptor flux limiting.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};
use tracing::trace_span;

/// Volume of Fluid state on a 2D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VofField {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    /// Volume fraction per cell: 0 = empty, 1 = full fluid.
    pub alpha: Vec<f64>,
}

impl VofField {
    /// Create a VOF field initialized to empty (α = 0).
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx < 4 || ny < 4 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 || !dx.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
        Ok(Self {
            nx,
            ny,
            dx,
            alpha: vec![0.0; nx * ny],
        })
    }

    /// Whether a cell is a surface cell (partially filled).
    #[inline]
    #[must_use]
    pub fn is_surface(&self, x: usize, y: usize) -> bool {
        let a = self.alpha[y * self.nx + x];
        a > 1e-6 && a < 1.0 - 1e-6
    }

    /// Whether a cell is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self, x: usize, y: usize) -> bool {
        self.alpha[y * self.nx + x] > 1.0 - 1e-6
    }

    /// Whether a cell is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self, x: usize, y: usize) -> bool {
        self.alpha[y * self.nx + x] < 1e-6
    }

    /// Fill a rectangular region with fluid (α = 1).
    pub fn fill_rect(&mut self, x_min: f64, y_min: f64, x_max: f64, y_max: f64) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                if px >= x_min && px <= x_max && py >= y_min && py <= y_max {
                    self.alpha[y * self.nx + x] = 1.0;
                }
            }
        }
    }

    /// Fill a circular region with fluid (α = 1).
    pub fn fill_circle(&mut self, cx: f64, cy: f64, radius: f64) {
        let r2 = radius * radius;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                let dx = px - cx;
                let dy = py - cy;
                if dx * dx + dy * dy < r2 {
                    self.alpha[y * self.nx + x] = 1.0;
                }
            }
        }
    }

    /// Total fluid volume (sum of α × cell area).
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        let cell_area = self.dx * self.dx;
        self.alpha.iter().map(|&a| a * cell_area).sum()
    }

    /// Advect the volume fraction field using donor-acceptor method.
    ///
    /// `vx` and `vy` are velocity fields on the same grid.
    /// Uses a flux-limiting approach to maintain α ∈ [0, 1].
    pub fn advect(&mut self, vx: &[f64], vy: &[f64], dt: f64) -> Result<()> {
        let _span = trace_span!("vof::advect", nx = self.nx, ny = self.ny).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let inv_dx = 1.0 / dx;

        let old_alpha = self.alpha.clone();

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;

                // Donor-acceptor fluxes at each face
                // x-direction
                let flux_right = {
                    let u = 0.5 * (vx[i] + vx[i + 1]);
                    let donor = if u > 0.0 {
                        old_alpha[i]
                    } else {
                        old_alpha[i + 1]
                    };
                    u * donor * dt * inv_dx
                };
                let flux_left = {
                    let u = 0.5 * (vx[i - 1] + vx[i]);
                    let donor = if u > 0.0 {
                        old_alpha[i - 1]
                    } else {
                        old_alpha[i]
                    };
                    u * donor * dt * inv_dx
                };
                // y-direction
                let flux_top = {
                    let v = 0.5 * (vy[i] + vy[i + nx]);
                    let donor = if v > 0.0 {
                        old_alpha[i]
                    } else {
                        old_alpha[i + nx]
                    };
                    v * donor * dt * inv_dx
                };
                let flux_bottom = {
                    let v = 0.5 * (vy[i - nx] + vy[i]);
                    let donor = if v > 0.0 {
                        old_alpha[i - nx]
                    } else {
                        old_alpha[i]
                    };
                    v * donor * dt * inv_dx
                };

                self.alpha[i] -= flux_right - flux_left + flux_top - flux_bottom;
                self.alpha[i] = self.alpha[i].clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Compute approximate interface normal at a surface cell using gradient of α.
    /// Returns (nx, ny) normalized. Returns (0, 0) for non-surface cells.
    #[must_use]
    pub fn interface_normal(&self, x: usize, y: usize) -> (f64, f64) {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 {
            return (0.0, 0.0);
        }
        let i = y * self.nx + x;
        let inv_2dx = 0.5 / self.dx;
        let gx = (self.alpha[i + 1] - self.alpha[i - 1]) * inv_2dx;
        let gy = (self.alpha[i + self.nx] - self.alpha[i - self.nx]) * inv_2dx;
        let mag = (gx * gx + gy * gy).sqrt();
        if mag < 1e-20 {
            (0.0, 0.0)
        } else {
            (gx / mag, gy / mag)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vof_new() {
        let vof = VofField::new(10, 10, 0.1).unwrap();
        assert_eq!(vof.alpha.len(), 100);
        assert!(vof.is_empty(5, 5));
    }

    #[test]
    fn test_vof_fill_rect() {
        let mut vof = VofField::new(10, 10, 0.1).unwrap();
        vof.fill_rect(0.2, 0.2, 0.5, 0.5);
        assert!(vof.is_full(3, 3));
        assert!(vof.is_empty(0, 0));
    }

    #[test]
    fn test_vof_fill_circle() {
        let mut vof = VofField::new(20, 20, 0.1).unwrap();
        vof.fill_circle(1.0, 1.0, 0.3);
        assert!(vof.is_full(10, 10));
        assert!(vof.is_empty(0, 0));
    }

    #[test]
    fn test_vof_total_volume() {
        let mut vof = VofField::new(10, 10, 0.1).unwrap();
        vof.alpha.fill(1.0);
        let expected = 10.0 * 10.0 * 0.1 * 0.1;
        assert!((vof.total_volume() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_vof_advect_stationary() {
        let mut vof = VofField::new(10, 10, 0.1).unwrap();
        vof.fill_rect(0.3, 0.3, 0.6, 0.6);
        let vol_before = vof.total_volume();
        let zero_vel = vec![0.0; 100];
        vof.advect(&zero_vel, &zero_vel, 0.01).unwrap();
        let vol_after = vof.total_volume();
        assert!((vol_before - vol_after).abs() < 1e-10);
    }

    #[test]
    fn test_vof_interface_normal() {
        let mut vof = VofField::new(20, 20, 0.1).unwrap();
        // Left half filled
        for y in 0..20 {
            for x in 0..10 {
                vof.alpha[y * 20 + x] = 1.0;
            }
        }
        let (nx, _ny) = vof.interface_normal(10, 10);
        // Gradient of α points from high→low, so -x at the interface
        assert!(nx.abs() > 0.5, "should have strong x-component: nx={nx}");
    }
}
