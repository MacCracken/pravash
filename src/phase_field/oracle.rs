//! Oracle dump for `PhaseField` (see `crate::oracle`). Read-only.

use super::PhaseField;
use crate::oracle::ScratchDump;

impl PhaseField {
    /// The field's private scratch buffer. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> ScratchDump {
        ScratchDump {
            scratch: self.scratch.clone(),
        }
    }
}
