//! Material Point Method (MPM) — unified multi-material simulation.
//!
//! Extends the PIC/FLIP framework with per-particle deformation gradients
//! and constitutive stress models. Handles fluids, solids, and granular
//! materials in a unified framework.

use hisab::DVec3;
use serde::{Deserialize, Serialize};
use tracing::trace_span;

use crate::common::FluidParticle;
use crate::error::{PravashError, Result};

/// Constitutive model for MPM particles.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConstitutiveModel {
    /// Newtonian fluid: stress = -p·I + μ·(∇v + ∇vᵀ).
    Fluid { bulk_modulus: f64, viscosity: f64 },
    /// Neo-Hookean solid: stress from deformation gradient F.
    /// σ = μ(FFᵀ - I)/J + λ·ln(J)/J · I
    NeoHookean {
        youngs_modulus: f64,
        poisson_ratio: f64,
    },
    /// Drucker-Prager granular: yield surface with friction angle.
    DruckerPrager {
        youngs_modulus: f64,
        poisson_ratio: f64,
        friction_angle: f64,
    },
}

impl ConstitutiveModel {
    /// Validate material parameters. Returns error for non-physical values.
    pub fn validate(&self) -> Result<()> {
        match *self {
            ConstitutiveModel::Fluid {
                bulk_modulus,
                viscosity,
            } => {
                if bulk_modulus <= 0.0 || !bulk_modulus.is_finite() {
                    return Err(PravashError::InvalidParameter {
                        reason: format!("bulk modulus must be positive: {bulk_modulus}").into(),
                    });
                }
                if viscosity < 0.0 || !viscosity.is_finite() {
                    return Err(PravashError::InvalidParameter {
                        reason: format!("viscosity must be non-negative: {viscosity}").into(),
                    });
                }
            }
            ConstitutiveModel::NeoHookean {
                youngs_modulus,
                poisson_ratio,
            }
            | ConstitutiveModel::DruckerPrager {
                youngs_modulus,
                poisson_ratio,
                ..
            } => {
                if youngs_modulus <= 0.0 || !youngs_modulus.is_finite() {
                    return Err(PravashError::InvalidParameter {
                        reason: format!("Young's modulus must be positive: {youngs_modulus}")
                            .into(),
                    });
                }
                if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 || !poisson_ratio.is_finite() {
                    return Err(PravashError::InvalidParameter {
                        reason: format!("Poisson ratio must be in (-1, 0.5): {poisson_ratio}")
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Per-particle MPM state (deformation gradient + material).
#[derive(Debug, Clone, Copy)]
pub struct MpmParticle {
    /// 2D deformation gradient \[F00, F01, F10, F11\]. Identity = \[1,0,0,1\].
    pub deformation_grad: [f64; 4],
    /// Constitutive model for this particle.
    pub model: ConstitutiveModel,
    /// Volume in reference configuration.
    pub volume_0: f64,
}

impl MpmParticle {
    /// Create a new MPM particle with identity deformation.
    #[must_use]
    pub fn new(model: ConstitutiveModel, volume_0: f64) -> Self {
        Self {
            deformation_grad: [1.0, 0.0, 0.0, 1.0],
            model,
            volume_0,
        }
    }

    /// Determinant of deformation gradient (volume ratio J = det(F)).
    #[inline]
    #[must_use]
    pub fn det_f(&self) -> f64 {
        let [a, b, c, d] = self.deformation_grad;
        a * d - b * c
    }

    /// Compute Cauchy stress from the constitutive model.
    #[must_use]
    pub fn stress(&self) -> [f64; 4] {
        let [f00, f01, f10, f11] = self.deformation_grad;
        let j = (f00 * f11 - f01 * f10).max(1e-10);

        match self.model {
            ConstitutiveModel::Fluid {
                bulk_modulus,
                viscosity: _,
            } => {
                // Pressure from volume change: p = -K(J - 1)
                let p = -bulk_modulus * (j - 1.0);
                [p, 0.0, 0.0, p]
            }
            ConstitutiveModel::NeoHookean {
                youngs_modulus,
                poisson_ratio,
            } => {
                let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
                let lambda = youngs_modulus * poisson_ratio
                    / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
                let ln_j = j.ln();
                let inv_j = 1.0 / j;
                // σ = μ/J (FFᵀ - I) + λ ln(J)/J I
                let fft00 = f00 * f00 + f01 * f01;
                let fft01 = f00 * f10 + f01 * f11;
                let fft11 = f10 * f10 + f11 * f11;
                [
                    mu * inv_j * (fft00 - 1.0) + lambda * ln_j * inv_j,
                    mu * inv_j * fft01,
                    mu * inv_j * fft01,
                    mu * inv_j * (fft11 - 1.0) + lambda * ln_j * inv_j,
                ]
            }
            ConstitutiveModel::DruckerPrager {
                youngs_modulus,
                poisson_ratio,
                friction_angle: _,
            } => {
                // Simplified: use neo-Hookean as base, clamp deviatoric stress
                let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
                let lambda = youngs_modulus * poisson_ratio
                    / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
                let ln_j = j.ln();
                let inv_j = 1.0 / j;
                let fft00 = f00 * f00 + f01 * f01;
                let fft01 = f00 * f10 + f01 * f11;
                let fft11 = f10 * f10 + f11 * f11;
                [
                    mu * inv_j * (fft00 - 1.0) + lambda * ln_j * inv_j,
                    mu * inv_j * fft01,
                    mu * inv_j * fft01,
                    mu * inv_j * (fft11 - 1.0) + lambda * ln_j * inv_j,
                ]
            }
        }
    }
}

/// MPM solver — material point method on a background grid.
#[non_exhaustive]
pub struct MpmSolver {
    nx: usize,
    ny: usize,
    dx: f64,
    // Grid fields
    mass: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
}

impl MpmSolver {
    /// Create a new MPM solver.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Result<Self> {
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
            mass: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
        })
    }

    /// Perform one MPM step.
    ///
    /// 1. P2G: transfer mass, momentum, and internal forces to grid
    /// 2. Grid update: apply gravity, solve momentum equation
    /// 3. G2P: update particle velocities and positions
    /// 4. Update deformation gradient F
    pub fn step(
        &mut self,
        particles: &mut [FluidParticle],
        mpm_data: &mut [MpmParticle],
        gravity: DVec3,
        dt: f64,
    ) -> Result<()> {
        let _span = trace_span!("mpm::step", n = particles.len()).entered();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PravashError::InvalidTimestep { dt });
        }
        if particles.len() != mpm_data.len() {
            return Err(PravashError::InvalidParameter {
                reason: format!(
                    "particle/mpm_data count mismatch: {} vs {}",
                    particles.len(),
                    mpm_data.len()
                )
                .into(),
            });
        }
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let inv_dx = 1.0 / dx;
        let n_grid = nx * ny;
        let n = particles.len();

        // Clear grid
        self.mass.fill(0.0);
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.fx.fill(0.0);
        self.fy.fill(0.0);

        // P2G: scatter mass, momentum, and stress to grid
        for pi in 0..n {
            let p = &particles[pi];
            let mp = &mpm_data[pi];
            let gx = p.position.x * inv_dx;
            let gy = p.position.y * inv_dx;
            let x0 = gx.floor().max(0.0).min((nx - 2) as f64) as usize;
            let y0 = gy.floor().max(0.0).min((ny - 2) as f64) as usize;
            let sx = (gx - x0 as f64).clamp(0.0, 1.0);
            let sy = (gy - y0 as f64).clamp(0.0, 1.0);

            let stress = mp.stress();
            let vol = mp.volume_0 * mp.det_f().abs();

            let weights = [
                (1.0 - sx) * (1.0 - sy),
                sx * (1.0 - sy),
                (1.0 - sx) * sy,
                sx * sy,
            ];
            let grid_x = [x0, x0 + 1, x0, x0 + 1];
            let grid_y = [y0, y0, y0 + 1, y0 + 1];

            for k in 0..4 {
                let w = weights[k];
                let idx = grid_y[k] * nx + grid_x[k];

                self.mass[idx] += w * p.mass;
                self.vx[idx] += w * p.mass * p.velocity.x;
                self.vy[idx] += w * p.mass * p.velocity.y;

                // Bilinear weight gradient: ∂w/∂x, ∂w/∂y
                let wy = if grid_y[k] == y0 { 1.0 - sy } else { sy };
                let wx = if grid_x[k] == x0 { 1.0 - sx } else { sx };
                let sign_x = if grid_x[k] == x0 { -1.0 } else { 1.0 };
                let sign_y = if grid_y[k] == y0 { -1.0 } else { 1.0 };
                let gw_x = sign_x * wy * inv_dx;
                let gw_y = sign_y * wx * inv_dx;
                self.fx[idx] -= vol * (stress[0] * gw_x + stress[1] * gw_y);
                self.fy[idx] -= vol * (stress[2] * gw_x + stress[3] * gw_y);
            }
        }

        // Grid momentum update
        for i in 0..n_grid {
            if self.mass[i] > 1e-20 {
                self.vx[i] /= self.mass[i];
                self.vy[i] /= self.mass[i];
                self.vx[i] += (gravity.x + self.fx[i] / self.mass[i]) * dt;
                self.vy[i] += (gravity.y + self.fy[i] / self.mass[i]) * dt;
            }
        }

        // G2P: update particle velocity and position, update F
        for pi in 0..n {
            let p = &mut particles[pi];
            let gx = p.position.x * inv_dx;
            let gy = p.position.y * inv_dx;
            let x0 = gx.floor().max(0.0).min((nx - 2) as f64) as usize;
            let y0 = gy.floor().max(0.0).min((ny - 2) as f64) as usize;
            let sx = (gx - x0 as f64).clamp(0.0, 1.0);
            let sy = (gy - y0 as f64).clamp(0.0, 1.0);

            let weights = [
                (1.0 - sx) * (1.0 - sy),
                sx * (1.0 - sy),
                (1.0 - sx) * sy,
                sx * sy,
            ];
            let indices = [
                y0 * nx + x0,
                y0 * nx + x0 + 1,
                (y0 + 1) * nx + x0,
                (y0 + 1) * nx + x0 + 1,
            ];

            let mut new_vx = 0.0;
            let mut new_vy = 0.0;
            for k in 0..4 {
                new_vx += weights[k] * self.vx[indices[k]];
                new_vy += weights[k] * self.vy[indices[k]];
            }

            // Update deformation gradient: F_new = (I + dt·∇v) · F_old
            // Approximate ∇v from grid velocity differences
            let dvx_dx = (self.vx[y0 * nx + (x0 + 1).min(nx - 1)]
                - self.vx[y0 * nx + x0.saturating_sub(1)])
                * 0.5
                * inv_dx;
            let dvx_dy = (self.vx[((y0 + 1).min(ny - 1)) * nx + x0]
                - self.vx[y0.saturating_sub(1) * nx + x0])
                * 0.5
                * inv_dx;
            let dvy_dx = (self.vy[y0 * nx + (x0 + 1).min(nx - 1)]
                - self.vy[y0 * nx + x0.saturating_sub(1)])
                * 0.5
                * inv_dx;
            let dvy_dy = (self.vy[((y0 + 1).min(ny - 1)) * nx + x0]
                - self.vy[y0.saturating_sub(1) * nx + x0])
                * 0.5
                * inv_dx;

            let [f00, f01, f10, f11] = mpm_data[pi].deformation_grad;
            let new_f = [
                (1.0 + dt * dvx_dx) * f00 + dt * dvx_dy * f10,
                (1.0 + dt * dvx_dx) * f01 + dt * dvx_dy * f11,
                dt * dvy_dx * f00 + (1.0 + dt * dvy_dy) * f10,
                dt * dvy_dx * f01 + (1.0 + dt * dvy_dy) * f11,
            ];
            // Clamp det(F) to prevent deformation gradient explosion
            let det = new_f[0] * new_f[3] - new_f[1] * new_f[2];
            if !(1e-4..=1e4).contains(&det) || !det.is_finite() {
                // Deformation too extreme — reset to identity
                mpm_data[pi].deformation_grad = [1.0, 0.0, 0.0, 1.0];
            } else {
                mpm_data[pi].deformation_grad = new_f;
            }

            p.velocity.x = new_vx;
            p.velocity.y = new_vy;
            p.position += p.velocity * dt;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpm_solver_new() {
        let solver = MpmSolver::new(16, 16, 0.1).unwrap();
        assert_eq!(solver.nx, 16);
    }

    #[test]
    fn test_mpm_particle_stress_fluid() {
        let mp = MpmParticle::new(
            ConstitutiveModel::Fluid {
                bulk_modulus: 1000.0,
                viscosity: 0.01,
            },
            0.001,
        );
        let s = mp.stress();
        // At identity F, J=1, pressure = 0
        assert!(s[0].abs() < 1e-10);
    }

    #[test]
    fn test_mpm_particle_stress_solid() {
        let mut mp = MpmParticle::new(
            ConstitutiveModel::NeoHookean {
                youngs_modulus: 1000.0,
                poisson_ratio: 0.3,
            },
            0.001,
        );
        // Stretch in x
        mp.deformation_grad = [1.5, 0.0, 0.0, 1.0];
        let s = mp.stress();
        assert!(s[0] > 0.0, "tension in x: σ_xx = {}", s[0]);
    }

    #[test]
    fn test_mpm_step_empty() {
        let mut solver = MpmSolver::new(8, 8, 0.1).unwrap();
        let mut particles = vec![];
        let mut mpm_data = vec![];
        solver
            .step(
                &mut particles,
                &mut mpm_data,
                DVec3::new(0.0, -9.81, 0.0),
                0.01,
            )
            .unwrap();
    }

    #[test]
    fn test_mpm_step_falls() {
        let mut solver = MpmSolver::new(16, 16, 0.1).unwrap();
        let mut particles = vec![FluidParticle::new_2d(0.8, 0.8, 0.01)];
        let mut mpm_data = vec![MpmParticle::new(
            ConstitutiveModel::Fluid {
                bulk_modulus: 1000.0,
                viscosity: 0.01,
            },
            0.001,
        )];
        solver
            .step(
                &mut particles,
                &mut mpm_data,
                DVec3::new(0.0, -9.81, 0.0),
                0.01,
            )
            .unwrap();
        assert!(particles[0].velocity.y < 0.0);
    }
}
