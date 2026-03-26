//! Phase-field interface tracking via Allen-Cahn equation.
//!
//! Tracks fluid interfaces using a smooth order parameter φ ∈ [-1, 1].
//! φ = -1 is phase A, φ = +1 is phase B, and the interface is at φ = 0.
//!
//! The evolution follows the Allen-Cahn equation:
//! ∂φ/∂t + u·∇φ = M·(ε²·∇²φ - (φ³ - φ))
//!
//! where M is mobility (controls relaxation speed) and ε controls interface width.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

use tracing::trace_span;

/// Phase-field state on a 2D grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PhaseField {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    /// Order parameter field: -1 = phase A, +1 = phase B.
    pub phi: Vec<f64>,
    /// Mobility coefficient (relaxation speed). Default: 1.0.
    pub mobility: f64,
    /// Interface width parameter. Controls the thickness of the diffuse
    /// interface. Typical: 1-3 grid cells (epsilon ≈ 1-3 * dx). Default: 2*dx.
    pub epsilon: f64,
    /// Scratch buffer.
    #[serde(skip)]
    scratch: Vec<f64>,
}

impl PhaseField {
    /// Create a phase field initialized to phase A (φ = -1) everywhere.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx < 4 || ny < 4 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        if dx <= 0.0 || !dx.is_finite() {
            return Err(PravashError::InvalidParameter {
                reason: format!("cell size must be positive: {dx}").into(),
            });
        }
        let size = nx * ny;
        Ok(Self {
            nx,
            ny,
            dx,
            phi: vec![-1.0; size],
            mobility: 1.0,
            epsilon: 2.0 * dx,
            scratch: vec![0.0; size],
        })
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Set a circular region to phase B (φ = +1).
    pub fn set_circle(&mut self, cx: f64, cy: f64, radius: f64) {
        let r2 = radius * radius;
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                let dx = px - cx;
                let dy = py - cy;
                if dx * dx + dy * dy < r2 {
                    let i = y * self.nx + x;
                    self.phi[i] = 1.0;
                }
            }
        }
    }

    /// Set a rectangular region to phase B (φ = +1).
    pub fn set_rect(&mut self, x_min: f64, y_min: f64, x_max: f64, y_max: f64) {
        for y in 0..self.ny {
            for x in 0..self.nx {
                let px = x as f64 * self.dx;
                let py = y as f64 * self.dx;
                if px >= x_min && px <= x_max && py >= y_min && py <= y_max {
                    let i = y * self.nx + x;
                    self.phi[i] = 1.0;
                }
            }
        }
    }

    /// Query the phase at a cell: true if phase B (φ > 0).
    #[inline]
    #[must_use]
    pub fn is_phase_b(&self, x: usize, y: usize) -> bool {
        self.phi[self.idx(x, y)] > 0.0
    }

    /// Interface indicator: cells where |φ| < threshold are at the interface.
    #[inline]
    #[must_use]
    pub fn is_interface(&self, x: usize, y: usize, threshold: f64) -> bool {
        self.phi[self.idx(x, y)].abs() < threshold
    }

    /// Step the phase field using Allen-Cahn with velocity advection.
    ///
    /// `vx` and `vy` are the velocity fields (same grid dimensions).
    /// Pass empty slices if no advection is needed.
    pub fn step(&mut self, vx: &[f64], vy: &[f64], dt: f64) -> Result<()> {
        let _span = trace_span!("phase_field::step", nx = self.nx, ny = self.ny).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }

        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let inv_2dx = 0.5 / dx;
        let inv_dx2 = 1.0 / (dx * dx);
        let m = self.mobility;
        let eps2 = self.epsilon * self.epsilon;
        let has_velocity = vx.len() == nx * ny && vy.len() == nx * ny;

        // Ensure scratch buffer
        let size = nx * ny;
        self.scratch.resize(size, 0.0);
        self.scratch.copy_from_slice(&self.phi);
        let ph = &self.scratch;

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let p = ph[i];

                // Laplacian: ∇²φ
                let lap = (ph[i + 1] + ph[i - 1] + ph[i + nx] + ph[i - nx] - 4.0 * p) * inv_dx2;

                // Double-well potential derivative: f'(φ) = φ³ - φ
                let f_prime = p * p * p - p;

                // Allen-Cahn: M · (ε²·∇²φ - f'(φ))
                let reaction = m * (eps2 * lap - f_prime);

                // Advection: -u·∇φ (upwind would be better but central is simpler)
                let advection = if has_velocity {
                    let dpdx = (ph[i + 1] - ph[i - 1]) * inv_2dx;
                    let dpdy = (ph[i + nx] - ph[i - nx]) * inv_2dx;
                    -(vx[i] * dpdx + vy[i] * dpdy)
                } else {
                    0.0
                };

                let new_phi = self.phi[i] + (advection + reaction) * dt;

                // Warn if the phase field is diverging significantly
                if new_phi.abs() > 1.5 {
                    tracing::warn!(x, y, new_phi, "phase field divergence detected (|φ| > 1.5)");
                }

                // Clamp to [-1, 1] for stability
                self.phi[i] = new_phi.clamp(-1.0, 1.0);
            }
        }

        // Neumann boundary (zero gradient)
        for x in 0..nx {
            self.phi[x] = self.phi[nx + x];
            self.phi[(ny - 1) * nx + x] = self.phi[(ny - 2) * nx + x];
        }
        for y in 0..ny {
            self.phi[y * nx] = self.phi[y * nx + 1];
            self.phi[y * nx + nx - 1] = self.phi[y * nx + nx - 2];
        }

        Ok(())
    }

    /// Total "mass" of phase B: ∫(φ + 1)/2 dA.
    #[must_use]
    pub fn phase_b_area(&self) -> f64 {
        let cell_area = self.dx * self.dx;
        self.phi.iter().map(|&p| (p + 1.0) * 0.5 * cell_area).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_field_new() {
        let pf = PhaseField::new(10, 10, 0.1).unwrap();
        assert_eq!(pf.phi.len(), 100);
        assert!((pf.phi[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_phase_field_invalid() {
        assert!(PhaseField::new(2, 10, 0.1).is_err());
        assert!(PhaseField::new(10, 10, 0.0).is_err());
    }

    #[test]
    fn test_set_circle() {
        let mut pf = PhaseField::new(20, 20, 0.1).unwrap();
        pf.set_circle(1.0, 1.0, 0.3);
        // Center should be phase B
        assert!(pf.is_phase_b(10, 10));
        // Corner should be phase A
        assert!(!pf.is_phase_b(0, 0));
    }

    #[test]
    fn test_set_rect() {
        let mut pf = PhaseField::new(20, 20, 0.1).unwrap();
        pf.set_rect(0.5, 0.5, 1.0, 1.0);
        assert!(pf.is_phase_b(7, 7));
        assert!(!pf.is_phase_b(0, 0));
    }

    #[test]
    fn test_uniform_stays_uniform() {
        // φ = -1 everywhere should stay φ = -1 (f'(-1) = -1 - (-1) = 0)
        let mut pf = PhaseField::new(10, 10, 0.1).unwrap();
        let empty: Vec<f64> = vec![];
        pf.step(&empty, &empty, 0.01).unwrap();
        for &p in &pf.phi {
            assert!(
                (p - (-1.0)).abs() < 1e-6,
                "uniform should stay uniform: {p}"
            );
        }
    }

    #[test]
    fn test_interface_sharpens() {
        // A smooth interface should sharpen toward the tanh profile
        let mut pf = PhaseField::new(40, 6, 0.1).unwrap();
        // Linear ramp from -1 to +1 across x
        for y in 0..6 {
            for x in 0..40 {
                let i = pf.idx(x, y);
                pf.phi[i] = (x as f64 / 20.0 - 1.0).clamp(-1.0, 1.0);
            }
        }
        let empty: Vec<f64> = vec![];
        for _ in 0..100 {
            pf.step(&empty, &empty, 0.001).unwrap();
        }
        // Interface should still exist (φ crosses 0 somewhere)
        let has_neg = pf.phi.iter().any(|&p| p < -0.5);
        let has_pos = pf.phi.iter().any(|&p| p > 0.5);
        assert!(has_neg && has_pos, "interface should persist");
    }

    #[test]
    fn test_advection_moves_interface() {
        let mut pf = PhaseField::new(30, 6, 0.1).unwrap();
        pf.set_rect(0.5, 0.0, 1.0, 0.6);
        let n = pf.nx * pf.ny;

        // Rightward velocity
        let vx = vec![1.0; n];
        let vy = vec![0.0; n];

        let initial_b = pf.phase_b_area();
        for _ in 0..50 {
            pf.step(&vx, &vy, 0.001).unwrap();
        }

        // Phase B should still exist and be approximately conserved
        let final_b = pf.phase_b_area();
        assert!(final_b > 0.0, "phase B should still exist after advection");
        assert!(
            (final_b - initial_b).abs() / initial_b < 0.3,
            "phase B area should be roughly conserved: initial={initial_b}, final={final_b}"
        );
    }

    #[test]
    fn test_phase_b_area() {
        let mut pf = PhaseField::new(10, 10, 0.1).unwrap();
        // All phase A: area should be 0
        assert!(pf.phase_b_area() < 1e-10);
        // Set all to phase B
        pf.phi.fill(1.0);
        let total = 10.0 * 10.0 * 0.1 * 0.1;
        assert!((pf.phase_b_area() - total).abs() < 1e-6);
    }

    #[test]
    fn test_is_interface() {
        let mut pf = PhaseField::new(10, 10, 0.1).unwrap();
        let i = 5 * pf.nx + 5;
        pf.phi[i] = 0.1;
        assert!(pf.is_interface(5, 5, 0.5));
        assert!(!pf.is_interface(0, 0, 0.5)); // φ = -1, not near interface
    }
}
