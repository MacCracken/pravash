//! `src/constants.cyr` — pravash's public constants, and the Rust std float constants its code
//! uses, as `PV_` globals. The list is shared with `constants.tsv` (`m_constants.rs`), so the two
//! can't drift apart.
//!
//! pravash 1.3.0 has 14 `pub const` items:
//! - the drag coefficients `buoyancy::DragCoefficient::{SPHERE, CUBE, CYLINDER, STREAMLINED, FLAT_PLATE}`;
//! - the added-mass coefficients `coupling::AddedMassCoefficient::{SPHERE, CUBE, CYLINDER, FLAT_PLATE}`;
//! - the presets `FluidMaterial::{WATER, OIL, HONEY, AIR, LAVA}`, structs of four f64 fields.
//!
//! It also uses six Rust std float constants (`std_all`): `f64::MAX`, `f64::INFINITY`,
//! `f64::NEG_INFINITY`, `f64::NAN`, `f64::consts::PI` and `f32::EPSILON`. The Cyrius stdlib has
//! no typed names for them (port-parity §1), so they are generated here too.
//!
//! Values come from the oracle's API (or Rust's std), never from the source text. `complete` and
//! `std_complete` prove both lists against `rust-old/src`, so a constant can't be missed. Private
//! constants (`MIN_DENSITY`, the solver limits, …) are not public API: they are extracted and
//! proven at the milestone whose module uses them, as impetus did (port plan §4 step 6).
//!
//! Cyrius names are derived from the Rust path: `FluidMaterial::WATER.density` is
//! `PV_FLUID_MATERIAL_WATER_DENSITY`, `DragCoefficient::SPHERE` is `PV_DRAG_COEFFICIENT_SPHERE`.

use std::fmt::Write as _;
use std::path::Path;

use crate::cyr::{doc_comment, f64_literal, header};

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

/// `FluidMaterial` → `FLUID_MATERIAL`.
fn upper_snake(camel: &str) -> String {
    let mut out = String::new();
    for (k, c) in camel.chars().enumerate() {
        if c.is_ascii_uppercase() && k > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

/// The Cyrius name of one value: `PV_<TYPE>_<CONST>[_<FIELD>]`.
pub fn cyr_name(path: &str, field: &str) -> String {
    let mut parts = path.rsplit("::");
    let konst = parts.next().unwrap_or_default();
    let ty = parts.next().unwrap_or_default();
    let mut name = format!("PV_{}_{konst}", upper_snake(ty));
    if !field.is_empty() {
        name.push('_');
        name.push_str(&field.to_ascii_uppercase());
    }
    name
}

/// How a Rust std constant is written in Cyrius.
pub enum StdForm {
    /// `const NAME = <decimal>;`, an f64 constant.
    F64Const,
    /// `var NAME: f64 = 0x<bits>;`: no literal exists, or it is unwieldy.
    F64Bits,
    /// `var NAME: f32 = <decimal>;`, rounded to f32 on store (exactly, for these values).
    F32Var,
}

/// One Rust std float constant pravash's library code uses.
pub struct StdConstant {
    /// `f64::MAX`, the golden row's input and the source spelling.
    pub path: &'static str,
    pub cyr: &'static str,
    /// The value, widened to f64 for an f32 constant (exactly).
    pub value: f64,
    pub form: StdForm,
}

/// Every Rust std float constant pravash 1.3.0's library code uses.
pub fn std_all() -> Vec<StdConstant> {
    vec![
        StdConstant {
            path: "f64::MAX",
            cyr: "PV_F64_MAX",
            value: f64::MAX,
            form: StdForm::F64Bits,
        },
        StdConstant {
            path: "f64::INFINITY",
            cyr: "PV_F64_INFINITY",
            value: f64::INFINITY,
            form: StdForm::F64Bits,
        },
        StdConstant {
            path: "f64::NEG_INFINITY",
            cyr: "PV_F64_NEG_INFINITY",
            value: f64::NEG_INFINITY,
            form: StdForm::F64Bits,
        },
        StdConstant {
            path: "f64::NAN",
            cyr: "PV_F64_NAN",
            value: f64::NAN,
            form: StdForm::F64Bits,
        },
        StdConstant {
            path: "f64::consts::PI",
            cyr: "PV_F64_PI",
            value: core::f64::consts::PI,
            form: StdForm::F64Const,
        },
        StdConstant {
            path: "f32::EPSILON",
            cyr: "PV_F32_EPSILON",
            value: f64::from(f32::EPSILON),
            form: StdForm::F32Var,
        },
    ]
}

/// The Rust source files under `rust-old/src`, as `(relative path, text)`.
fn sources(root: &Path) -> Result<Vec<(String, String)>, String> {
    let base = root.join("rust-old/src");
    let mut out = Vec::new();
    let mut stack = vec![base.clone()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                let rel = path
                    .strip_prefix(&base)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .into_owned();
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                out.push((rel, src));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Every `f64::NAME` / `f32::NAME` / `f64::consts::NAME` the library code (outside tests, comments
/// and doc comments) names must be in `std_all()`, and every listed one must be used.
pub fn std_complete(root: &Path) -> Result<(), String> {
    let mut found = std::collections::BTreeSet::new();
    for (_, src) in sources(root)? {
        for line in library_part(&src).lines() {
            let code = line.split("//").next().unwrap_or_default();
            for ty in ["f64::", "f32::"] {
                let mut rest = code;
                while let Some(at) = rest.find(ty) {
                    let after = &rest[at + ty.len()..];
                    let name: String = after
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
                        .collect();
                    let first = name.split("::").next().unwrap_or_default();
                    // Constants are SHOUTING_CASE (or `consts::X`); methods (`f64::from`) are not.
                    if first == "consts"
                        || (!first.is_empty()
                            && first.chars().all(|c| c.is_ascii_uppercase() || c == '_'))
                    {
                        found.insert(format!("{ty}{name}"));
                    }
                    rest = &after[name.len().min(after.len())..];
                }
            }
        }
    }
    let want: std::collections::BTreeSet<String> =
        std_all().iter().map(|c| c.path.to_string()).collect();
    if found == want {
        Ok(())
    } else {
        Err(format!(
            "t_constants::std_all() lists {want:?}, but rust-old/src names {found:?}"
        ))
    }
}

/// `src/constants.cyr`.
pub fn emit(root: &Path, provenance: &str) -> Result<String, String> {
    complete(root)?;
    std_complete(root)?;
    let mut out = header(
        "public constants",
        provenance,
        &[
            "Every public constant of the Rust crate, and every Rust std float constant its code\n\
             uses. t_constants.rs proves both lists complete against rust-old/src, and\n\
             tests/golden/constants.tsv pins each value.",
            "A float constant is a `const` with a decimal literal (an f64). A value with no literal\n\
             (±inf, NaN, f64::MAX) is a typed global holding its bit pattern: a hex `const` would be\n\
             an integer. Private constants are extracted and proven at the milestone whose module\n\
             uses them.",
        ],
    );
    for c in all() {
        let file = root.join("rust-old/src").join(c.file);
        let src = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        out.push_str(&doc_comment(&src, c.item)?);
        let _ = writeln!(out, "# Rust: pravash::{}", c.path);
        for (field, value) in &c.values {
            let _ = writeln!(
                out,
                "const {} = {};",
                cyr_name(c.path, field),
                f64_literal(*value)?
            );
        }
        out.push('\n');
    }
    out.push_str("# Rust std float constants pravash uses (port-parity §1: the stdlib has no typed names).\n\n");
    for c in std_all() {
        let _ = writeln!(out, "# Rust: {}", c.path);
        match c.form {
            StdForm::F64Const => {
                let _ = writeln!(out, "const {} = {};", c.cyr, f64_literal(c.value)?);
            }
            StdForm::F64Bits => {
                let _ = writeln!(out, "var {}: f64 = 0x{:016X};", c.cyr, c.value.to_bits());
            }
            StdForm::F32Var => {
                let _ = writeln!(out, "var {}: f32 = {};", c.cyr, f64_literal(c.value)?);
            }
        }
        out.push('\n');
    }
    // One trailing newline, not a blank line, at the end of the file.
    while out.ends_with("\n\n") {
        out.pop();
    }
    Ok(out)
}
