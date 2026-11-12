//! pravash's public constants: the list shared by `constants.tsv` (`m_constants.rs`) and, from M0
//! step 6, the `src/constants.cyr` generator, so the two can't drift apart.
//!
//! pravash 1.3.0 has 14 `pub const` items:
//! - the drag coefficients `buoyancy::DragCoefficient::{SPHERE, CUBE, CYLINDER, STREAMLINED, FLAT_PLATE}`;
//! - the added-mass coefficients `coupling::AddedMassCoefficient::{SPHERE, CUBE, CYLINDER, FLAT_PLATE}`;
//! - the presets `FluidMaterial::{WATER, OIL, HONEY, AIR, LAVA}`, structs of four f64 fields.
//!
//! Values come from the oracle's API, never from the source text. `complete` proves the list
//! against `rust-old/src`, so a public constant can't be missed. Private constants (`MIN_DENSITY`,
//! the solver limits, …) are not public API: they are extracted and proven at the milestone whose
//! module uses them, as impetus did (port plan §4 step 6).

use std::path::Path;

use pravash::FluidMaterial;
use pravash::buoyancy::DragCoefficient;
use pravash::coupling::AddedMassCoefficient;

/// One public const of the oracle.
pub struct Constant {
    /// `buoyancy::DragCoefficient::SPHERE`, the golden rows' input.
    pub path: &'static str,
    /// The Rust file under `rust-old/src` and the item line (its doc comment sits above it).
    pub file: &'static str,
    pub item: &'static str,
    /// `(field, value)`: one entry with an empty field for a scalar, one per field for a struct.
    pub values: Vec<(&'static str, f64)>,
}

fn scalar(path: &'static str, file: &'static str, item: &'static str, value: f64) -> Constant {
    Constant {
        path,
        file,
        item,
        values: vec![("", value)],
    }
}

fn material(path: &'static str, item: &'static str, m: FluidMaterial) -> Constant {
    Constant {
        path,
        file: "common.rs",
        item,
        values: vec![
            ("density", m.density),
            ("viscosity", m.viscosity),
            ("surface_tension", m.surface_tension),
            ("speed_of_sound", m.speed_of_sound),
        ],
    }
}

/// Every public const of pravash 1.3.0, in source order.
pub fn all() -> Vec<Constant> {
    vec![
        material(
            "common::FluidMaterial::WATER",
            "pub const WATER",
            FluidMaterial::WATER,
        ),
        material(
            "common::FluidMaterial::OIL",
            "pub const OIL",
            FluidMaterial::OIL,
        ),
        material(
            "common::FluidMaterial::HONEY",
            "pub const HONEY",
            FluidMaterial::HONEY,
        ),
        material(
            "common::FluidMaterial::AIR",
            "pub const AIR",
            FluidMaterial::AIR,
        ),
        material(
            "common::FluidMaterial::LAVA",
            "pub const LAVA",
            FluidMaterial::LAVA,
        ),
        scalar(
            "buoyancy::DragCoefficient::SPHERE",
            "buoyancy.rs",
            "pub const SPHERE",
            DragCoefficient::SPHERE,
        ),
        scalar(
            "buoyancy::DragCoefficient::CUBE",
            "buoyancy.rs",
            "pub const CUBE",
            DragCoefficient::CUBE,
        ),
        scalar(
            "buoyancy::DragCoefficient::CYLINDER",
            "buoyancy.rs",
            "pub const CYLINDER",
            DragCoefficient::CYLINDER,
        ),
        scalar(
            "buoyancy::DragCoefficient::STREAMLINED",
            "buoyancy.rs",
            "pub const STREAMLINED",
            DragCoefficient::STREAMLINED,
        ),
        scalar(
            "buoyancy::DragCoefficient::FLAT_PLATE",
            "buoyancy.rs",
            "pub const FLAT_PLATE",
            DragCoefficient::FLAT_PLATE,
        ),
        scalar(
            "coupling::AddedMassCoefficient::SPHERE",
            "coupling.rs",
            "pub const SPHERE",
            AddedMassCoefficient::SPHERE,
        ),
        scalar(
            "coupling::AddedMassCoefficient::CUBE",
            "coupling.rs",
            "pub const CUBE",
            AddedMassCoefficient::CUBE,
        ),
        scalar(
            "coupling::AddedMassCoefficient::CYLINDER",
            "coupling.rs",
            "pub const CYLINDER",
            AddedMassCoefficient::CYLINDER,
        ),
        scalar(
            "coupling::AddedMassCoefficient::FLAT_PLATE",
            "coupling.rs",
            "pub const FLAT_PLATE",
            AddedMassCoefficient::FLAT_PLATE,
        ),
    ]
}

/// The non-test part of a Rust source file: everything before its `#[cfg(test)] mod tests`.
fn library_part(src: &str) -> &str {
    src.find("#[cfg(test)]\nmod tests")
        .map_or(src, |at| &src[..at])
}

/// Every `pub const NAME` item in `rust-old/src` outside test code must be in `all()` (as a
/// `(file, NAME)` pair), and every listed one must exist: a new public constant can't be missed.
pub fn complete(root: &Path) -> Result<(), String> {
    let mut want: Vec<(String, String)> = all()
        .iter()
        .map(|c| {
            (
                c.file.to_string(),
                c.item.trim_start_matches("pub const ").to_string(),
            )
        })
        .collect();
    let mut found = Vec::new();
    let mut stack = vec![root.join("rust-old/src")];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            let rel = path
                .strip_prefix(root.join("rust-old/src"))
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .into_owned();
            let src =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for line in library_part(&src).lines() {
                if let Some(rest) = line.trim_start().strip_prefix("pub const ")
                    && let Some((ident, _)) = rest.split_once(':')
                {
                    found.push((rel.clone(), ident.trim().to_string()));
                }
            }
        }
    }
    found.sort();
    want.sort();
    if found == want {
        Ok(())
    } else {
        Err(format!(
            "t_constants::all() lists {want:?}, but rust-old/src has public consts {found:?}"
        ))
    }
}
