//! Grid-based Euler/Navier-Stokes fluid solver.
//!
//! Represents fluid on a fixed grid with velocity and pressure fields.
//! Uses operator splitting: advection → diffusion → projection.

use serde::{Deserialize, Serialize};

use crate::error::{PravashError, Result};

/// 2D grid-based fluid state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidGrid {
    /// Grid resolution.
    pub nx: usize,
    pub ny: usize,
    /// Cell size.
    pub dx: f64,
    /// Velocity field (x-component), row-major.
    pub vx: Vec<f64>,
    /// Velocity field (y-component), row-major.
    pub vy: Vec<f64>,
    /// Pressure field, row-major.
    pub pressure: Vec<f64>,
    /// Density field (for visualization/transport).
    pub density: Vec<f64>,
}

impl FluidGrid {
    /// Create a new grid initialized to zero.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(PravashError::InvalidGridResolution { nx, ny });
        }
        let size = nx * ny;
        Ok(Self {
            nx,
            ny,
            dx,
            vx: vec![0.0; size],
            vy: vec![0.0; size],
            pressure: vec![0.0; size],
            density: vec![0.0; size],
        })
    }

    /// Total number of cells.
    #[inline]
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny
    }

    /// Index from (x, y) coordinates.
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Get velocity at a cell.
    #[inline]
    pub fn velocity_at(&self, x: usize, y: usize) -> (f64, f64) {
        let i = self.idx(x, y);
        (self.vx[i], self.vy[i])
    }

    /// Speed at a cell.
    #[inline]
    pub fn speed_at(&self, x: usize, y: usize) -> f64 {
        let (vx, vy) = self.velocity_at(x, y);
        (vx * vx + vy * vy).sqrt()
    }

    /// Maximum speed in the grid (for CFL).
    pub fn max_speed(&self) -> f64 {
        self.vx
            .iter()
            .zip(self.vy.iter())
            .map(|(&vx, &vy)| (vx * vx + vy * vy).sqrt())
            .fold(0.0f64, f64::max)
    }

    /// Total kinetic energy.
    pub fn total_kinetic_energy(&self) -> f64 {
        let dx2 = self.dx * self.dx;
        self.vx
            .iter()
            .zip(self.vy.iter())
            .zip(self.density.iter())
            .map(|((&vx, &vy), &rho)| 0.5 * rho * (vx * vx + vy * vy) * dx2)
            .sum()
    }

    /// Diffuse a field using Gauss-Seidel iteration.
    pub fn diffuse(field: &mut [f64], nx: usize, ny: usize, diff_rate: f64, dt: f64, iterations: usize) {
        let a = dt * diff_rate * (nx as f64) * (ny as f64);
        for _ in 0..iterations {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let idx = y * nx + x;
                    let neighbors = field[idx - 1] + field[idx + 1]
                        + field[idx - nx] + field[idx + nx];
                    field[idx] = (field[idx] + a * neighbors) / (1.0 + 4.0 * a);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_new() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert_eq!(g.cell_count(), 100);
        assert_eq!(g.vx.len(), 100);
    }

    #[test]
    fn test_grid_invalid() {
        assert!(FluidGrid::new(0, 10, 0.1).is_err());
        assert!(FluidGrid::new(10, 0, 0.1).is_err());
    }

    #[test]
    fn test_grid_idx() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(5, 3), 35);
    }

    #[test]
    fn test_velocity_at() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let i = g.idx(3, 4);
        g.vx[i] = 1.5;
        g.vy[i] = 2.5;
        let (vx, vy) = g.velocity_at(3, 4);
        assert!((vx - 1.5).abs() < f64::EPSILON);
        assert!((vy - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speed_at() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        let i = g.idx(5, 5);
        g.vx[i] = 3.0;
        g.vy[i] = 4.0;
        assert!((g.speed_at(5, 5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_speed_zero() {
        let g = FluidGrid::new(10, 10, 0.1).unwrap();
        assert!(g.max_speed().abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_speed() {
        let mut g = FluidGrid::new(10, 10, 0.1).unwrap();
        g.vx[50] = 3.0;
        g.vy[50] = 4.0;
        assert!((g.max_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diffuse_preserves_zeros() {
        let mut field = vec![0.0; 100];
        FluidGrid::diffuse(&mut field, 10, 10, 0.01, 0.001, 10);
        assert!(field.iter().all(|&v| v.abs() < f64::EPSILON));
    }

    #[test]
    fn test_diffuse_spreads() {
        let mut field = vec![0.0; 100];
        field[55] = 100.0; // spike in center
        FluidGrid::diffuse(&mut field, 10, 10, 0.1, 0.01, 20);
        // Neighbors should have picked up some value
        assert!(field[54] > 0.0 || field[56] > 0.0);
    }

    #[test]
    fn test_grid_serde() {
        let g = FluidGrid::new(5, 5, 0.2).unwrap();
        let json = serde_json::to_string(&g).unwrap();
        let g2: FluidGrid = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.nx, 5);
        assert_eq!(g2.ny, 5);
    }
}
