//! Oracle dump and hooks for `sph` (see `crate::oracle`). Read-only.

use super::SphSolver;
use crate::oracle::SphSolverDump;

impl SphSolver {
    /// The solver's private state, including the neighbour lists in the
    /// order the solver sums them. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> SphSolverDump {
        SphSolverDump {
            cell_size: self.cell_size,
            grid_len: self.grid.len(),
            grid_cells: self.grid.cell_count(),
            densities: self.densities.clone(),
            snapshot: self.snapshot.clone(),
            positions_f32: self.positions_f32.iter().map(|p| [p.x, p.y, p.z]).collect(),
            neighbor_indices: self.neighbor_indices.clone(),
            neighbor_offsets: self.neighbor_offsets.clone(),
            pcisph_accel: self.pcisph_accel.clone(),
            pcisph_pressures: self.pcisph_pressures.clone(),
            pcisph_pred_pos: self.pcisph_pred_pos.clone(),
            pcisph_pred_vel: self.pcisph_pred_vel.clone(),
            prev_accel: self.prev_accel.clone(),
            dfsph_alpha: self.dfsph_alpha.clone(),
            dfsph_kappa: self.dfsph_kappa.clone(),
        }
    }
}

/// The cell key `SphSolver`'s neighbour build computes for an f32 position:
/// `floor(x · (1 / h)) as i32` on each axis, in f32, with `h_f32` the step's
/// `smoothing_radius as f32`. The same expression as `build_neighbors`.
/// Oracle hook; not public API.
#[must_use]
pub fn sph_cell_key(pos: [f32; 3], h_f32: f32) -> [i32; 3] {
    let inv_cs = 1.0 / h_f32;
    [
        (pos[0] * inv_cs).floor() as i32,
        (pos[1] * inv_cs).floor() as i32,
        (pos[2] * inv_cs).floor() as i32,
    ]
}

/// The MLS gradient correction's 3×3 inverse: hisab 1.4.0's `matrix_inverse`
/// exactly as pravash calls it. Oracle hook; not public API.
#[must_use]
pub fn invert_3x3(m: &[f64; 9]) -> Option<[f64; 9]> {
    super::invert_3x3(m)
}
