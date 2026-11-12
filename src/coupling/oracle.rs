//! Oracle dump for `FlipSolver` (see `crate::oracle`). Read-only.

use super::FlipSolver;
use crate::oracle::FlipSolverDump;

impl FlipSolver {
    /// The solver's private state. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> FlipSolverDump {
        FlipSolverDump {
            nx: self.nx,
            ny: self.ny,
            dx: self.dx,
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            vx_old: self.vx_old.clone(),
            vy_old: self.vy_old.clone(),
            weight: self.weight.clone(),
            pressure: self.pressure.clone(),
            div: self.div.clone(),
            apic_c: self.apic_c.clone(),
        }
    }
}
