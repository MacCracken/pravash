//! Oracle dump for `ShallowWater` (see `crate::oracle`). Read-only.

use super::ShallowWater;
use crate::oracle::ShallowWaterDump;

impl ShallowWater {
    /// The solver's private scratch buffers. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> ShallowWaterDump {
        ShallowWaterDump {
            scratch_vx: self.scratch_vx.clone(),
            scratch_vy: self.scratch_vy.clone(),
            scratch_h: self.scratch_h.clone(),
            scratch_lap: self.scratch_lap.clone(),
            scratch_line: self.scratch_line.clone(),
        }
    }
}
