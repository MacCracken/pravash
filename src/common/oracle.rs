//! Oracle dumps for `common`'s arena types (see `crate::oracle`). Read-only.

use super::{ArenaHandle, ParticleArena};
use crate::oracle::{ArenaHandleDump, ParticleArenaDump};

impl ArenaHandle {
    /// The handle's private fields. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> ArenaHandleDump {
        ArenaHandleDump {
            start: self.start,
            len: self.len,
            generation: self.generation,
        }
    }
}

impl ParticleArena {
    /// The arena's private state. Oracle hook; not public API.
    #[doc(hidden)]
    #[must_use]
    pub fn oracle_dump(&self) -> ParticleArenaDump {
        ParticleArenaDump {
            particles: self.particles.clone(),
            capacity: self.capacity,
            watermark: self.watermark,
            free_list: self.free_list.clone(),
            generation: self.generation,
        }
    }
}
