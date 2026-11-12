//! Oracle dumps and pressure-solver hooks for `grid` (see `crate::oracle`).
//! Read-only: each hook runs one of `FluidGrid`'s private pressure solvers on
//! buffers the caller owns, exactly as `FluidGrid::step` calls it.

use super::{FluidGrid, KEpsilon, MacGrid};
use crate::error::Result;
use crate::oracle::{FluidGridDump, KEpsilonDump, MacGridDump, MgLevelDump};

impl FluidGrid {
    /// The grid's private scratch buffers. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> FluidGridDump {
        FluidGridDump {
            scratch_a: self.scratch_a.clone(),
            scratch_b: self.scratch_b.clone(),
            scratch_c: self.scratch_c.clone(),
            diffuse_rhs: self.diffuse_rhs.clone(),
            mg_levels: self
                .mg_levels
                .iter()
                .map(|(residual, coarse_rhs, coarse_p)| MgLevelDump {
                    residual: residual.clone(),
                    coarse_rhs: coarse_rhs.clone(),
                    coarse_p: coarse_p.clone(),
                })
                .collect(),
        }
    }
}

impl KEpsilon {
    /// The model's private scratch buffers. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> KEpsilonDump {
        KEpsilonDump {
            scratch_k: self.scratch_k.clone(),
            scratch_eps: self.scratch_eps.clone(),
        }
    }
}

impl MacGrid {
    /// The grid's private scratch buffers. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> MacGridDump {
        MacGridDump {
            scratch_div: self.scratch_div.clone(),
            scratch_density: self.scratch_density.clone(),
        }
    }
}

/// `FluidGrid`'s Gauss–Seidel pressure solve on one system (`4p − Σp = rhs`).
/// Oracle hook; not public API.
///
/// # Errors
/// As the private solver.
pub fn pressure_solve_gauss_seidel(
    pressure: &mut [f64],
    div: &[f64],
    nx: usize,
    ny: usize,
    iterations: usize,
) -> Result<()> {
    FluidGrid::pressure_solve(pressure, div, nx, ny, iterations)
}

/// `FluidGrid`'s DST pressure solve on one system (hisab 1.4.0 `dst`/`idst`
/// as pravash calls them). Oracle hook; not public API.
///
/// # Errors
/// As the private solver.
pub fn pressure_solve_dst(pressure: &mut [f64], div: &[f64], nx: usize, ny: usize) -> Result<()> {
    FluidGrid::pressure_solve_dst(pressure, div, nx, ny)
}

/// `FluidGrid`'s CG pressure solve on one system (hisab 1.4.0
/// `conjugate_gradient` as pravash calls it). Oracle hook; not public API.
///
/// # Errors
/// As the private solver.
pub fn pressure_solve_cg(
    pressure: &mut [f64],
    div: &[f64],
    nx: usize,
    ny: usize,
    iterations: usize,
) -> Result<()> {
    FluidGrid::pressure_solve_cg(pressure, div, nx, ny, iterations)
}

/// `FluidGrid`'s multigrid pressure solve on one system, starting from empty
/// level buffers (what a fresh grid has). Returns the level buffers it leaves,
/// finest first. Oracle hook; not public API.
///
/// # Errors
/// As the private solver.
pub fn pressure_solve_multigrid(
    pressure: &mut [f64],
    div: &[f64],
    nx: usize,
    ny: usize,
    nu_smooth: usize,
    cycles: usize,
) -> Result<Vec<MgLevelDump>> {
    let mut levels = Vec::new();
    FluidGrid::pressure_solve_multigrid(pressure, div, nx, ny, nu_smooth, cycles, &mut levels)?;
    Ok(levels
        .into_iter()
        .map(|(residual, coarse_rhs, coarse_p)| MgLevelDump {
            residual,
            coarse_rhs,
            coarse_p,
        })
        .collect())
}
