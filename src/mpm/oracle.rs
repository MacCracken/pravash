//! Oracle dump for `MpmSolver` (see `crate::oracle`). Read-only.

use super::MpmSolver;
use crate::oracle::MpmSolverDump;

impl MpmSolver {
    /// The solver's private grid state. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> MpmSolverDump {
        MpmSolverDump {
            nx: self.nx,
            ny: self.ny,
            dx: self.dx,
            mass: self.mass.clone(),
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            fx: self.fx.clone(),
            fy: self.fy.clone(),
        }
    }
}
