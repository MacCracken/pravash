//! Oracle dump for `VofField` (see `crate::oracle`). Read-only.

use super::VofField;
use crate::oracle::ScratchDump;

impl VofField {
    /// The field's private scratch buffer. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> ScratchDump {
        ScratchDump {
            scratch: self.scratch.clone(),
        }
    }
}
