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
    /// Weakly compressible fluid: σ = −p·I with p = −K·(J − 1).
    /// `viscosity` is validated but not yet applied (see roadmap).
    Fluid { bulk_modulus: f64, viscosity: f64 },
    /// Neo-Hookean solid: stress from deformation gradient F.
    /// σ = μ(FFᵀ - I)/J + λ·ln(J)/J · I
    NeoHookean {
        youngs_modulus: f64,
        poisson_ratio: f64,
    },
    /// Drucker-Prager granular material (sand plasticity, Klár et al. 2016).
    ///
    /// Elasticity is St. Venant-Kirchhoff in Hencky (log) strain; after each
    /// deformation update the elastic strain is projected back onto the
    /// Drucker-Prager cone. `friction_angle` is in **degrees**, `[0, 90)`.
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
            } => validate_elastic(youngs_modulus, poisson_ratio)?,
            ConstitutiveModel::DruckerPrager {
                youngs_modulus,
                poisson_ratio,
                friction_angle,
            } => {
                validate_elastic(youngs_modulus, poisson_ratio)?;
                if !(0.0..90.0).contains(&friction_angle) {
                    return Err(PravashError::InvalidParameter {
                        reason: format!(
                            "friction angle must be in [0, 90) degrees: {friction_angle}"
                        )
                        .into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Validate Young's modulus and Poisson ratio for an elastic model.
fn validate_elastic(youngs_modulus: f64, poisson_ratio: f64) -> Result<()> {
    if youngs_modulus <= 0.0 || !youngs_modulus.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: format!("Young's modulus must be positive: {youngs_modulus}").into(),
        });
    }
    if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 || !poisson_ratio.is_finite() {
        return Err(PravashError::InvalidParameter {
            reason: format!("Poisson ratio must be in (-1, 0.5): {poisson_ratio}").into(),
        });
    }
    Ok(())
}

/// Lamé parameters (μ, λ) from Young's modulus and Poisson ratio.
#[inline]
fn lame(youngs_modulus: f64, poisson_ratio: f64) -> (f64, f64) {
    let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    let lambda =
        youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    (mu, lambda)
}

/// Closed-form SVD of a row-major 2×2 matrix `[a00, a01, a10, a11]`.
///
/// Returns `(u, sigma, vt)` with `u` and `vt` proper rotations stored as
/// `(cos, sin)` pairs and `A = R(u) · diag(sigma) · R(vt)`, where
/// `R(c, s) = [[c, -s], [s, c]]`. `sigma[1]` carries the sign of `det(A)`.
#[inline]
fn svd2(a: [f64; 4]) -> ((f64, f64), [f64; 2], (f64, f64)) {
    let [a00, a01, a10, a11] = a;
    let e = 0.5 * (a00 + a11);
    let f = 0.5 * (a00 - a11);
    let g = 0.5 * (a10 + a01);
    let h = 0.5 * (a10 - a01);
    let q = e.hypot(h);
    let r = f.hypot(g);
    let a1 = g.atan2(f);
    let a2 = h.atan2(e);
    let theta = 0.5 * (a2 - a1);
    let phi = 0.5 * (a2 + a1);
    let (s_phi, c_phi) = phi.sin_cos();
    let (s_theta, c_theta) = theta.sin_cos();
    ((c_phi, s_phi), [q + r, q - r], (c_theta, s_theta))
}

/// Recompose `R(u) · diag(sigma) · R(vt)` into a row-major 2×2 matrix.
#[inline]
fn compose2(u: (f64, f64), sigma: [f64; 2], vt: (f64, f64)) -> [f64; 4] {
    let (cu, su) = u;
    let (cv, sv) = vt;
    // R(u)·diag(σ) = [[cu·σ0, -su·σ1], [su·σ0, cu·σ1]]
    let m00 = cu * sigma[0];
    let m01 = -su * sigma[1];
    let m10 = su * sigma[0];
    let m11 = cu * sigma[1];
    // · R(vt) = · [[cv, -sv], [sv, cv]]
    [
        m00 * cv + m01 * sv,
        -m00 * sv + m01 * cv,
        m10 * cv + m11 * sv,
        -m10 * sv + m11 * cv,
    ]
}

/// Smallest singular value allowed in the Hencky strain (keeps `ln σ` finite).
const MIN_SINGULAR_VALUE: f64 = 1e-6;

/// Project a deformation gradient onto the Drucker-Prager yield cone
/// (Klár et al. 2016, "Drucker-Prager Elastoplasticity for Sand Animation").
#[must_use]
fn drucker_prager_project(
    f: [f64; 4],
    youngs_modulus: f64,
    poisson_ratio: f64,
    friction_angle_deg: f64,
) -> [f64; 4] {
    let (mu, lambda) = lame(youngs_modulus, poisson_ratio);
    let (u, sigma, vt) = svd2(f);
    let e0 = sigma[0].max(MIN_SINGULAR_VALUE).ln();
    let e1 = sigma[1].max(MIN_SINGULAR_VALUE).ln();
    let tr = e0 + e1;
    // Expansion: the material separates — return to the cone tip (stress-free).
    if tr > 0.0 {
        return compose2(u, [1.0, 1.0], vt);
    }
    let dev0 = e0 - 0.5 * tr;
    let dev1 = e1 - 0.5 * tr;
    let dev_norm = dev0.hypot(dev1);
    let sin_phi = friction_angle_deg.to_radians().sin();
    let alpha = (2.0_f64 / 3.0).sqrt() * 2.0 * sin_phi / (3.0 - sin_phi);
    // d = 2 spatial dimensions.
    let delta_gamma = dev_norm + (2.0 * lambda + 2.0 * mu) / (2.0 * mu) * tr * alpha;
    if delta_gamma <= 0.0 || dev_norm <= 0.0 {
        // Inside the yield cone: elastic, no projection.
        return f;
    }
    let scale = delta_gamma / dev_norm;
    let h0 = e0 - scale * dev0;
    let h1 = e1 - scale * dev1;
    compose2(u, [h0.exp(), h1.exp()], vt)
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

    /// Compute Cauchy stress from the constitutive model (tension positive).
    #[inline]
    #[must_use]
    pub fn stress(&self) -> [f64; 4] {
        let [f00, f01, f10, f11] = self.deformation_grad;
        let j = (f00 * f11 - f01 * f10).max(1e-10);

        match self.model {
            ConstitutiveModel::Fluid {
                bulk_modulus,
                viscosity: _,
            } => {
                // Pressure from volume change p = -K(J - 1); Cauchy stress σ = -p·I.
                // Compression (J < 1) gives negative (compressive) stress.
                let s = bulk_modulus * (j - 1.0);
                [s, 0.0, 0.0, s]
            }
            ConstitutiveModel::NeoHookean {
                youngs_modulus,
                poisson_ratio,
            } => {
                let (mu, lambda) = lame(youngs_modulus, poisson_ratio);
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
                // St. Venant-Kirchhoff in Hencky strain: F = U Σ Vᵀ, ε = ln Σ,
                // Kirchhoff τ = U (2μ ε + λ tr(ε) I) Uᵀ, Cauchy σ = τ / J.
                // The friction angle acts in the plastic projection (see `step`).
                let (mu, lambda) = lame(youngs_modulus, poisson_ratio);
                let ((cu, su), sigma, _) = svd2(self.deformation_grad);
                let e0 = sigma[0].max(MIN_SINGULAR_VALUE).ln();
                let e1 = sigma[1].max(MIN_SINGULAR_VALUE).ln();
                let tr = e0 + e1;
                let t0 = (2.0 * mu * e0 + lambda * tr) / j;
                let t1 = (2.0 * mu * e1 + lambda * tr) / j;
                // U diag(t0, t1) Uᵀ with U = [[cu, -su], [su, cu]]
                let off = (t0 - t1) * cu * su;
                [
                    t0 * cu * cu + t1 * su * su,
                    off,
                    off,
                    t0 * su * su + t1 * cu * cu,
                ]
            }
        }
    }
}

/// MPM solver — material point method on a background grid.
///
/// The domain is `[0, (nx-1)·dx] × [0, (ny-1)·dx]`. Grid walls are
/// separating (free-slip): outward velocity is removed at boundary nodes
/// and particles are kept inside the domain.
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
        let n = nx
            .checked_mul(ny)
            .ok_or(PravashError::InvalidGridResolution { nx, ny })?;
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
    /// 4. Update deformation gradient F (with plastic projection for
    ///    Drucker-Prager particles)
    ///
    /// Returns an error for an invalid timestep, mismatched slice lengths, or
    /// any particle whose material parameters or reference volume are invalid.
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
        for mp in mpm_data.iter() {
            mp.model.validate()?;
            if !mp.volume_0.is_finite() || mp.volume_0 <= 0.0 {
                return Err(PravashError::InvalidParameter {
                    reason: format!("reference volume must be positive: {}", mp.volume_0).into(),
                });
            }
        }
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let inv_dx = 1.0 / dx;
        let n = particles.len();
        let x_max = (nx - 1) as f64 * dx;
        let y_max = (ny - 1) as f64 * dx;

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

        // Grid momentum update + separating (free-slip) walls
        for gy in 0..ny {
            for gx in 0..nx {
                let i = gy * nx + gx;
                if self.mass[i] <= 1e-20 {
                    continue;
                }
                let inv_m = 1.0 / self.mass[i];
                let mut vx = self.vx[i] * inv_m + (gravity.x + self.fx[i] * inv_m) * dt;
                let mut vy = self.vy[i] * inv_m + (gravity.y + self.fy[i] * inv_m) * dt;
                if (gx == 0 && vx < 0.0) || (gx == nx - 1 && vx > 0.0) {
                    vx = 0.0;
                }
                if (gy == 0 && vy < 0.0) || (gy == ny - 1 && vy > 0.0) {
                    vy = 0.0;
                }
                self.vx[i] = vx;
                self.vy[i] = vy;
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

            // Weight gradients ∂w/∂x, ∂w/∂y (same stencil as P2G)
            let grad_x = [-(1.0 - sy), 1.0 - sy, -sy, sy].map(|g| g * inv_dx);
            let grad_y = [-(1.0 - sx), -sx, 1.0 - sx, sx].map(|g| g * inv_dx);

            let mut new_vx = 0.0;
            let mut new_vy = 0.0;
            // Velocity gradient L = Σᵢ vᵢ ⊗ ∇wᵢ(xₚ)
            let (mut dvx_dx, mut dvx_dy, mut dvy_dx, mut dvy_dy) = (0.0, 0.0, 0.0, 0.0);
            for k in 0..4 {
                let vx = self.vx[indices[k]];
                let vy = self.vy[indices[k]];
                new_vx += weights[k] * vx;
                new_vy += weights[k] * vy;
                dvx_dx += vx * grad_x[k];
                dvx_dy += vx * grad_y[k];
                dvy_dx += vy * grad_x[k];
                dvy_dy += vy * grad_y[k];
            }

            // Update deformation gradient: F_new = (I + dt·L) · F_old
            let [f00, f01, f10, f11] = mpm_data[pi].deformation_grad;
            let mut new_f = [
                (1.0 + dt * dvx_dx) * f00 + dt * dvx_dy * f10,
                (1.0 + dt * dvx_dx) * f01 + dt * dvx_dy * f11,
                dt * dvy_dx * f00 + (1.0 + dt * dvy_dy) * f10,
                dt * dvy_dx * f01 + (1.0 + dt * dvy_dy) * f11,
            ];
            if let ConstitutiveModel::DruckerPrager {
                youngs_modulus,
                poisson_ratio,
                friction_angle,
            } = mpm_data[pi].model
            {
                new_f =
                    drucker_prager_project(new_f, youngs_modulus, poisson_ratio, friction_angle);
            }
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
            // Keep particles inside the grid domain (separating walls).
            if p.position.x < 0.0 {
                p.position.x = 0.0;
                p.velocity.x = p.velocity.x.max(0.0);
            } else if p.position.x > x_max {
                p.position.x = x_max;
                p.velocity.x = p.velocity.x.min(0.0);
            }
            if p.position.y < 0.0 {
                p.position.y = 0.0;
                p.velocity.y = p.velocity.y.max(0.0);
            } else if p.position.y > y_max {
                p.position.y = y_max;
                p.velocity.y = p.velocity.y.min(0.0);
            }
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

    // ── Audit regressions (1.3.0) ──────────────────────────────────────

    #[test]
    fn test_fluid_stress_compressive_under_compression() {
        let mut mp = MpmParticle::new(
            ConstitutiveModel::Fluid {
                bulk_modulus: 1000.0,
                viscosity: 0.0,
            },
            0.001,
        );
        mp.deformation_grad = [0.9, 0.0, 0.0, 1.0]; // J = 0.9 (compressed)
        let s = mp.stress();
        assert!(
            s[0] < 0.0 && s[3] < 0.0,
            "compression must give σ < 0: {s:?}"
        );
        assert!((s[0] + 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_svd2_reconstructs() {
        for a in [
            [1.5, 0.2, -0.3, 0.8],
            [1.0, 0.0, 0.0, 1.0],
            [0.7, -0.4, 0.9, 1.1],
            [2.0, 0.0, 0.0, 0.5],
        ] {
            let (u, sigma, vt) = svd2(a);
            assert!(sigma[0] >= sigma[1].abs());
            let r = compose2(u, sigma, vt);
            for k in 0..4 {
                assert!((r[k] - a[k]).abs() < 1e-12, "{a:?} -> {r:?}");
            }
        }
    }

    #[test]
    fn test_drucker_prager_rejects_bad_friction_angle() {
        for phi in [-1.0, 90.0, f64::NAN] {
            let m = ConstitutiveModel::DruckerPrager {
                youngs_modulus: 1e4,
                poisson_ratio: 0.3,
                friction_angle: phi,
            };
            assert!(m.validate().is_err(), "phi={phi}");
        }
    }

    #[test]
    fn test_drucker_prager_projection() {
        let (e, nu, phi) = (1e4, 0.3, 30.0);
        // Expansion → stress-free tip.
        let f = drucker_prager_project([1.2, 0.0, 0.0, 1.1], e, nu, phi);
        let (_, s, _) = svd2(f);
        assert!((s[0] - 1.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
        // Pure volumetric compression is inside the cone → unchanged.
        let fc = [0.95, 0.0, 0.0, 0.95];
        assert_eq!(drucker_prager_project(fc, e, nu, phi), fc);
        // Large shear under mild compression is projected (deviatoric strain shrinks).
        let fs = [1.3, 0.0, 0.0, 0.7];
        let fp = drucker_prager_project(fs, e, nu, phi);
        let (_, s0, _) = svd2(fs);
        let (_, s1, _) = svd2(fp);
        let dev = |s: [f64; 2]| (s[0].ln() - s[1].ln()).abs();
        assert!(dev(s1) < dev(s0), "{s0:?} -> {s1:?}");
    }

    #[test]
    fn test_drucker_prager_stress_zero_at_rest() {
        let mp = MpmParticle::new(
            ConstitutiveModel::DruckerPrager {
                youngs_modulus: 1e4,
                poisson_ratio: 0.3,
                friction_angle: 30.0,
            },
            0.001,
        );
        // Identity F → zero stress.
        assert!(mp.stress().iter().all(|s| s.abs() < 1e-12));
    }

    #[test]
    fn test_step_rejects_invalid_material() {
        let mut solver = MpmSolver::new(8, 8, 0.1).unwrap();
        let mut particles = vec![FluidParticle::new_2d(0.4, 0.4, 0.01)];
        let mut mpm_data = vec![MpmParticle::new(
            ConstitutiveModel::NeoHookean {
                youngs_modulus: 1e4,
                poisson_ratio: 0.5, // λ = ∞
            },
            0.001,
        )];
        assert!(
            solver
                .step(&mut particles, &mut mpm_data, DVec3::ZERO, 0.01)
                .is_err()
        );
        mpm_data[0] = MpmParticle::new(
            ConstitutiveModel::Fluid {
                bulk_modulus: 1e3,
                viscosity: 0.0,
            },
            -1.0,
        );
        assert!(
            solver
                .step(&mut particles, &mut mpm_data, DVec3::ZERO, 0.01)
                .is_err()
        );
    }

    #[test]
    fn test_particles_stay_in_domain() {
        let mut solver = MpmSolver::new(8, 8, 0.1).unwrap();
        let mut particles = vec![FluidParticle::new_2d(0.35, 0.05, 0.01)];
        let mut mpm_data = vec![MpmParticle::new(
            ConstitutiveModel::Fluid {
                bulk_modulus: 1e3,
                viscosity: 0.0,
            },
            0.001,
        )];
        for _ in 0..200 {
            solver
                .step(
                    &mut particles,
                    &mut mpm_data,
                    DVec3::new(0.0, -9.81, 0.0),
                    0.005,
                )
                .unwrap();
        }
        let p = particles[0].position;
        assert!(
            p.y >= 0.0 && p.y <= 0.7 && p.x >= 0.0 && p.x <= 0.7,
            "{p:?}"
        );
    }

    #[test]
    fn test_fluid_block_resists_compression() {
        // A fluid block squeezed from both sides must push back (expand) rather
        // than collapse further — fails with an inverted pressure sign.
        let mut solver = MpmSolver::new(16, 16, 0.05).unwrap();
        let mut particles = Vec::new();
        let mut mpm_data = Vec::new();
        for j in 0..8 {
            for i in 0..8 {
                particles.push(FluidParticle::new_2d(
                    0.25 + i as f64 * 0.025,
                    0.25 + j as f64 * 0.025,
                    0.001,
                ));
                let mut mp = MpmParticle::new(
                    ConstitutiveModel::Fluid {
                        bulk_modulus: 50.0,
                        viscosity: 0.0,
                    },
                    0.025 * 0.025,
                );
                mp.deformation_grad = [0.8, 0.0, 0.0, 0.8]; // pre-compressed
                mpm_data.push(mp);
            }
        }
        let extent = |ps: &[FluidParticle]| {
            let (lo, hi) = ps.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
                (lo.min(p.position.x), hi.max(p.position.x))
            });
            hi - lo
        };
        let w0 = extent(&particles);
        for _ in 0..20 {
            solver
                .step(&mut particles, &mut mpm_data, DVec3::ZERO, 1e-3)
                .unwrap();
        }
        assert!(
            extent(&particles) > w0,
            "compressed fluid should expand: {w0} -> {}",
            extent(&particles)
        );
    }
}
