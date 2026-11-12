//! Oracle hooks for the Cyrius port's golden-value generator.
//!
//! **Not public API.** This module exists only behind the non-default `oracle`
//! feature, is hidden from the docs, and carries no stability promise. It lets
//! `scripts/oracle/` (see `docs/development/pravash-port-plan.md`, M0 step 0)
//! read state that the public API does not expose:
//! - the private fields of every stateful type, for single-step parity rows
//!   (public fields are read directly);
//! - the SPH neighbour hash: cell keys and candidate lists, in summation order;
//! - the grid pressure solvers on one given system;
//! - the MLS 3×3 inverse, which is hisab 1.4.0's `matrix_inverse` as pravash
//!   calls it.
//!
//! Everything here is read-only. Enabling the feature changes no simulation
//! behaviour; `tests/oracle.rs` proves that dumping every step leaves each
//! solver's run bit-identical.
//!
//! The dumps are filled by child modules (`src/<module>/oracle.rs`), which can
//! read their parent's private fields, so the audited sources only gain a
//! `mod oracle;` line.

use hisab::DVec3;

use crate::common::FluidParticle;

#[cfg(feature = "grid")]
pub use crate::grid::oracle::{
    pressure_solve_cg, pressure_solve_dst, pressure_solve_gauss_seidel, pressure_solve_multigrid,
};
#[cfg(feature = "sph")]
pub use crate::sph::oracle::{invert_3x3, sph_cell_key};

// ---------------------------------------------------------------------------
// common
// ---------------------------------------------------------------------------

/// An [`ArenaHandle`](crate::ArenaHandle)'s private fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaHandleDump {
    /// First slot of the block.
    pub start: usize,
    /// Number of slots.
    pub len: usize,
    /// The arena generation the handle was issued in.
    pub generation: u64,
}

/// A [`ParticleArena`](crate::ParticleArena)'s private state, which decides
/// what `alloc`, `free`, `get` and `compact` do next.
#[derive(Debug, Clone)]
pub struct ParticleArenaDump {
    /// Every slot, live or free, in storage order.
    pub particles: Vec<FluidParticle>,
    /// Maximum number of slots.
    pub capacity: usize,
    /// One past the highest slot ever allocated.
    pub watermark: usize,
    /// Free blocks as `(start, len)`, sorted by start.
    pub free_list: Vec<(usize, usize)>,
    /// Current generation (bumped by `compact`).
    pub generation: u64,
}

// ---------------------------------------------------------------------------
// shallow
// ---------------------------------------------------------------------------

/// A `ShallowWater`'s private scratch buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct ShallowWaterDump {
    /// `scratch_vx`.
    pub scratch_vx: Vec<f64>,
    /// `scratch_vy`.
    pub scratch_vy: Vec<f64>,
    /// `scratch_h`.
    pub scratch_h: Vec<f64>,
    /// `scratch_lap`.
    pub scratch_lap: Vec<f64>,
    /// `scratch_line`.
    pub scratch_line: Vec<f64>,
}

// ---------------------------------------------------------------------------
// grid
// ---------------------------------------------------------------------------

/// One multigrid level as `FluidGrid` keeps it between steps.
#[derive(Debug, Clone, PartialEq)]
pub struct MgLevelDump {
    /// The level's residual buffer.
    pub residual: Vec<f64>,
    /// The coarse right-hand side.
    pub coarse_rhs: Vec<f64>,
    /// The coarse correction.
    pub coarse_p: Vec<f64>,
}

/// A `FluidGrid`'s private scratch buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidGridDump {
    /// `scratch_a`.
    pub scratch_a: Vec<f64>,
    /// `scratch_b`.
    pub scratch_b: Vec<f64>,
    /// `scratch_c`.
    pub scratch_c: Vec<f64>,
    /// `diffuse_rhs`.
    pub diffuse_rhs: Vec<f64>,
    /// `mg_levels`, from the finest level down.
    pub mg_levels: Vec<MgLevelDump>,
}

/// A `KEpsilon`'s private scratch buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct KEpsilonDump {
    /// `scratch_k`.
    pub scratch_k: Vec<f64>,
    /// `scratch_eps`.
    pub scratch_eps: Vec<f64>,
}

/// A `MacGrid`'s private scratch buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct MacGridDump {
    /// `scratch_div`.
    pub scratch_div: Vec<f64>,
    /// `scratch_density`.
    pub scratch_density: Vec<f64>,
}

/// A `VofField`'s or
/// `PhaseField`'s private scratch buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ScratchDump {
    /// `scratch`.
    pub scratch: Vec<f64>,
}

// ---------------------------------------------------------------------------
// coupling, mpm
// ---------------------------------------------------------------------------

/// A `FlipSolver`'s private state.
#[derive(Debug, Clone, PartialEq)]
pub struct FlipSolverDump {
    /// Grid width in cells.
    pub nx: usize,
    /// Grid height in cells.
    pub ny: usize,
    /// Cell size.
    pub dx: f64,
    /// Grid x-velocity.
    pub vx: Vec<f64>,
    /// Grid y-velocity.
    pub vy: Vec<f64>,
    /// Grid x-velocity before the pressure solve.
    pub vx_old: Vec<f64>,
    /// Grid y-velocity before the pressure solve.
    pub vy_old: Vec<f64>,
    /// P2G weights.
    pub weight: Vec<f64>,
    /// Pressure.
    pub pressure: Vec<f64>,
    /// Divergence.
    pub div: Vec<f64>,
    /// Per-particle APIC affine matrices, which carry from step to step.
    pub apic_c: Vec<[f64; 4]>,
}

/// A `MpmSolver`'s private state.
#[derive(Debug, Clone, PartialEq)]
pub struct MpmSolverDump {
    /// Grid width in cells.
    pub nx: usize,
    /// Grid height in cells.
    pub ny: usize,
    /// Cell size.
    pub dx: f64,
    /// Grid mass.
    pub mass: Vec<f64>,
    /// Grid x-velocity.
    pub vx: Vec<f64>,
    /// Grid y-velocity.
    pub vy: Vec<f64>,
    /// Grid x-force.
    pub fx: Vec<f64>,
    /// Grid y-force.
    pub fy: Vec<f64>,
}

// ---------------------------------------------------------------------------
// sph
// ---------------------------------------------------------------------------

/// A `SphSolver`'s private state.
///
/// The hisab `SpatialHash` itself is rebuilt from the particles at the start
/// of every neighbour build, so only its size is recorded. Each particle's
/// cell key is `sph_cell_key` of its `positions_f32` entry and the step's
/// `smoothing_radius as f32`.
#[derive(Debug, Clone)]
pub struct SphSolverDump {
    /// The f32 cell size the hash was last built with.
    pub cell_size: f32,
    /// Number of points in the hash.
    pub grid_len: usize,
    /// Number of non-empty cells in the hash.
    pub grid_cells: usize,
    /// `densities`.
    pub densities: Vec<f64>,
    /// `snapshot`.
    pub snapshot: Vec<FluidParticle>,
    /// The f32 positions the hash was built from, in particle order.
    pub positions_f32: Vec<[f32; 3]>,
    /// Neighbour candidates, concatenated in the order the solver sums them.
    pub neighbor_indices: Vec<usize>,
    /// CSR offsets into `neighbor_indices` (`len = n + 1`).
    pub neighbor_offsets: Vec<u32>,
    /// PCISPH accelerations.
    pub pcisph_accel: Vec<DVec3>,
    /// PCISPH pressures.
    pub pcisph_pressures: Vec<f64>,
    /// PCISPH predicted positions.
    pub pcisph_pred_pos: Vec<DVec3>,
    /// PCISPH predicted velocities.
    pub pcisph_pred_vel: Vec<DVec3>,
    /// Velocity Verlet's previous accelerations.
    pub prev_accel: Vec<DVec3>,
    /// DFSPH α factors.
    pub dfsph_alpha: Vec<f64>,
    /// DFSPH κ values.
    pub dfsph_kappa: Vec<f64>,
}
