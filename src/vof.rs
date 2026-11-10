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
    /// Previous-step α (reused across steps to avoid allocation).
    #[serde(skip)]
    scratch: Vec<f64>,
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
            scratch: vec![0.0; nx * ny],
        })
    }

    /// Whether a cell is a surface cell (partially filled), using the default
    /// threshold of `1e-6`.
    #[inline]
    #[must_use]
    pub fn is_surface(&self, x: usize, y: usize) -> bool {
        self.is_surface_with_threshold(x, y, 1e-6)
    }

    /// Whether a cell is a surface cell (partially filled) using a custom threshold.
    ///
    /// A cell is considered a surface cell when its volume fraction is between
    /// `threshold` and `1.0 - threshold`.
    #[inline]
    #[must_use]
    pub fn is_surface_with_threshold(&self, x: usize, y: usize, threshold: f64) -> bool {
        let a = self.alpha[y * self.nx + x];
        a > threshold && a < 1.0 - threshold
    }

    /// Whether a cell is full, using the default threshold of `1e-6`.
    #[inline]
    #[must_use]
    pub fn is_full(&self, x: usize, y: usize) -> bool {
        self.is_full_with_threshold(x, y, 1e-6)
    }

    /// Whether a cell is full using a custom threshold.
    ///
    /// A cell is considered full when its volume fraction exceeds `1.0 - threshold`.
    #[inline]
    #[must_use]
    pub fn is_full_with_threshold(&self, x: usize, y: usize, threshold: f64) -> bool {
        self.alpha[y * self.nx + x] > 1.0 - threshold
    }

    /// Whether a cell is empty, using the default threshold of `1e-6`.
    #[inline]
    #[must_use]
    pub fn is_empty(&self, x: usize, y: usize) -> bool {
        self.is_empty_with_threshold(x, y, 1e-6)
    }

    /// Whether a cell is empty using a custom threshold.
    ///
    /// A cell is considered empty when its volume fraction is below `threshold`.
    #[inline]
    #[must_use]
    pub fn is_empty_with_threshold(&self, x: usize, y: usize, threshold: f64) -> bool {
        self.alpha[y * self.nx + x] < threshold
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

    /// Advect the volume fraction field using an upwind (donor-cell) flux method.
    ///
    /// `vx` and `vy` are cell-centred velocity fields on the same grid.
    /// Face fluxes are shared by both neighbouring cells, faces on the
    /// boundary ring are closed (no flux through the walls), and the step is
    /// sub-cycled so the per-substep Courant number stays ≤ 0.5. Together
    /// these conserve volume except where a converging flow overfills a cell
    /// (α is clamped to [0, 1]).
    ///
    /// Returns an error if `dt` is invalid, the velocity slices do not match
    /// the grid, or the velocity is non-finite.
    pub fn advect(&mut self, vx: &[f64], vy: &[f64], dt: f64) -> Result<()> {
        let _span = trace_span!("vof::advect", nx = self.nx, ny = self.ny).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let n = nx * ny;
        if nx < 4 || ny < 4 || self.alpha.len() != n || vx.len() != n || vy.len() != n {
            return Err(PravashError::InvalidParameter {
                reason: format!(
                    "VOF advect: velocity/alpha lengths ({}, {}, {}) must equal nx*ny = {n}",
                    vx.len(),
                    vy.len(),
                    self.alpha.len()
                )
                .into(),
            });
        }
        if vx.iter().chain(vy).any(|v| !v.is_finite()) {
            return Err(PravashError::Diverged {
                reason: "VOF advect: non-finite velocity".into(),
            });
        }
        let max_v = vx.iter().chain(vy).fold(0.0f64, |m, v| m.max(v.abs()));
        // Two flux directions per cell: keep (|u|+|v|)·dt/dx ≤ 0.5 per substep.
        let courant = 2.0 * max_v * dt / dx;
        let substeps = (courant / 0.5).ceil().clamp(1.0, 1024.0) as usize;
        let sub_dt = dt / substeps as f64;
        for _ in 0..substeps {
            self.advect_substep(vx, vy, sub_dt);
        }
        Ok(())
    }

    /// One conservative donor-cell substep with closed walls.
    fn advect_substep(&mut self, vx: &[f64], vy: &[f64], dt: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let inv_dx = 1.0 / self.dx;

        let mut old_alpha = std::mem::take(&mut self.scratch);
        old_alpha.resize(nx * ny, 0.0);
        old_alpha.copy_from_slice(&self.alpha);

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;

                // Donor-cell fluxes at each face; faces touching the boundary
                // ring are walls (zero flux).
                // x-direction
                let flux_right = if x + 1 == nx - 1 {
                    0.0
                } else {
                    let u = 0.5 * (vx[i] + vx[i + 1]);
                    let donor = if u > 0.0 {
                        old_alpha[i]
                    } else {
                        old_alpha[i + 1]
                    };
                    u * donor * dt * inv_dx
                };
                let flux_left = if x == 1 {
                    0.0
                } else {
                    let u = 0.5 * (vx[i - 1] + vx[i]);
                    let donor = if u > 0.0 {
                        old_alpha[i - 1]
                    } else {
                        old_alpha[i]
                    };
                    u * donor * dt * inv_dx
                };
                // y-direction
                let flux_top = if y + 1 == ny - 1 {
                    0.0
                } else {
                    let v = 0.5 * (vy[i] + vy[i + nx]);
                    let donor = if v > 0.0 {
                        old_alpha[i]
                    } else {
                        old_alpha[i + nx]
                    };
                    v * donor * dt * inv_dx
                };
                let flux_bottom = if y == 1 {
                    0.0
                } else {
                    let v = 0.5 * (vy[i - nx] + vy[i]);
                    let donor = if v > 0.0 {
                        old_alpha[i - nx]
                    } else {
                        old_alpha[i]
                    };
                    v * donor * dt * inv_dx
                };

                let a = old_alpha[i] - (flux_right - flux_left + flux_top - flux_bottom);
                self.alpha[i] = a.clamp(0.0, 1.0);
            }
        }
        self.scratch = old_alpha;
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

    // ── Audit regressions (1.3.0) ──────────────────────────────────────

    #[test]
    fn test_vof_conserves_volume_cellular_flow() {
        // Divergence-free cellular flow, tangential at the walls of [0, L]².
        let (nx, ny, dx) = (34, 34, 0.1);
        let l = (nx - 2) as f64 * dx;
        let mut vof = VofField::new(nx, ny, dx).unwrap();
        vof.fill_rect(0.1, 0.1, 0.5 * l, l + 0.05);
        let v0 = vof.total_volume();
        let n = nx * ny;
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        let pi = std::f64::consts::PI;
        for y in 0..ny {
            for x in 0..nx {
                let (px, py) = ((x as f64 - 0.5) * dx, (y as f64 - 0.5) * dx);
                vx[y * nx + x] = (pi * px / l).sin() * (pi * py / l).cos();
                vy[y * nx + x] = -(pi * px / l).cos() * (pi * py / l).sin();
            }
        }
        for _ in 0..40 {
            vof.advect(&vx, &vy, 0.05).unwrap();
        }
        let v1 = vof.total_volume();
        assert!((v1 - v0).abs() < 0.01 * v0, "volume {v0} -> {v1}");
        assert!(vof.alpha.iter().all(|&a| (0.0..=1.0).contains(&a)));
    }

    #[test]
    fn test_vof_rejects_short_velocity() {
        let mut vof = VofField::new(8, 8, 0.1).unwrap();
        assert!(vof.advect(&[0.0; 10], &[0.0; 64], 0.01).is_err());
        assert!(vof.advect(&[f64::NAN; 64], &[0.0; 64], 0.01).is_err());
    }
}
