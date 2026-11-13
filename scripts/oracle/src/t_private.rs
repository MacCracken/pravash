//! `src/constants_private.cyr` — the private constants of the ported modules, extracted from the Rust source (port
//! plan §4 step 6, as impetus's `t_private.rs` and tanmatra did).
//!
//! A private constant isn't reachable through the oracle's API, so its value comes from its literal in
//! `rust-old/src`, parsed with Rust's own `f64::from_str` (correctly rounded) and written positionally
//! (`cyr::f64_literal`, which checks the round trip). It is proven by the golden rows that depend on it, and
//! `scripts/oracle/probe_tables.py` compiles each one and compares its bits with the literal's. `pub const` items
//! are public API and come from the oracle (`t_constants.rs`). `emit` proves that every other `const` item in each
//! ported module, at any depth (module, `impl` or function), is emitted or excused, so one added to a ported module
//! can't be missed.
//!
//! Names: `_PV_<MODULE>_<NAME>` (`MIN_DENSITY` in `sph.rs` is `_PV_SPH_MIN_DENSITY`). An f64 is a typed global
//! (`var _PV_…: f64 = 0.0000000001;`), an integer a `const`.

use std::fmt::Write as _;
use std::path::Path;

use crate::cyr::{doc_comment, f64_literal, header};

/// A ported module: its file under `rust-old/src`, and the consts not carried (name, reason).
struct Source {
    file: &'static str,
    excused: &'static [(&'static str, &'static str)],
}

/// The ported modules, in port order. Each is scanned; most have no private constants.
fn sources() -> Vec<Source> {
    vec![
        Source {
            file: "error.rs",
            excused: &[],
        },
        Source {
            file: "common.rs",
            excused: &[],
        },
        Source {
            file: "buoyancy.rs",
            excused: &[],
        },
        Source {
            file: "vortex.rs",
            excused: &[],
        },
        Source {
            file: "bridge.rs",
            excused: &[],
        },
        Source {
            file: "shallow.rs",
            excused: &[],
        },
        Source {
            file: "grid.rs",
            excused: &[],
        },
        Source {
            file: "vof.rs",
            excused: &[],
        },
        Source {
            file: "phase_field.rs",
            excused: &[],
        },
        Source {
            file: "sph.rs",
            excused: &[(
                "FALLBACK",
                "the `PhaseProperties` literal local to `MultiPhaseConfig::get`, written in place",
            )],
        },
        Source {
            file: "coupling.rs",
            excused: &[],
        },
        Source {
            file: "mpm.rs",
            excused: &[],
        },
    ]
}

/// One `const NAME: TYPE = LITERAL;` item.
struct Item {
    name: String,
    ty: String,
    literal: String,
    /// The item's line prefix, for `doc_comment`.
    head: String,
}

/// The non-test part of a Rust source file.
fn library_part(src: &str) -> &str {
    src.find("#[cfg(test)]\nmod tests")
        .map_or(src, |at| &src[..at])
}

/// Every non-`pub` `const` item in `src`'s library part, at any indentation.
fn items(src: &str) -> Result<Vec<Item>, String> {
    let mut out = Vec::new();
    for line in library_part(src).lines() {
        let t = line.trim_start();
        let found = ["pub(super) const ", "pub(crate) const ", "const "]
            .iter()
            .find_map(|p| t.strip_prefix(p).map(|r| (*p, r)));
        let Some((vis, rest)) = found else { continue };
        let (name, after) = rest
            .split_once(':')
            .ok_or_else(|| format!("unparsed const line: {line}"))?;
        let (ty, lit) = after
            .split_once('=')
            .ok_or_else(|| format!("unparsed const line: {line}"))?;
        out.push(Item {
            name: name.trim().to_string(),
            ty: ty.trim().to_string(),
            literal: lit.trim().trim_end_matches(';').trim().to_string(),
            head: format!("{vis}{}", name.trim()),
        });
    }
    Ok(out)
}

/// The module's name prefix: `sph.rs` → `_PV_SPH_`.
fn prefix(file: &str) -> String {
    format!("_PV_{}_", file.trim_end_matches(".rs").to_uppercase())
}

/// `src/constants_private.cyr`.
pub fn emit(root: &Path, provenance: &str) -> Result<String, String> {
    let mut out = header(
        "private constants of the ported modules",
        provenance,
        &[
            "Extracted from rust-old/src by t_private.rs: each literal parsed as Rust parses it and written\n\
             positionally. Proven by the golden rows that depend on them; probe_tables.py compiles each and\n\
             compares its bits with the literal's.",
        ],
    );
    for s in sources() {
        let path = root.join("rust-old/src").join(s.file);
        let src = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let found = items(&src)?;
        for (name, _) in s.excused {
            if !found.iter().any(|i| i.name == *name) {
                return Err(format!(
                    "t_private: {} excuses {name}, which isn't in it",
                    s.file
                ));
            }
        }
        if found.is_empty() {
            continue;
        }
        let _ = writeln!(out, "# ── rust-old/src/{} ──", s.file);
        out.push('\n');
        for it in &found {
            if let Some((_, why)) = s.excused.iter().find(|(n, _)| *n == it.name) {
                let _ = writeln!(out, "# {} is not carried: {why}.", it.name);
                out.push('\n');
                continue;
            }
            out.push_str(&doc_comment(&src, &it.head)?);
            let _ = writeln!(out, "# Rust: `{}: {} = {}`", it.name, it.ty, it.literal);
            let cyr = format!("{}{}", prefix(s.file), it.name);
            match it.ty.as_str() {
                "f64" => {
                    let clean: String = it.literal.chars().filter(|c| *c != '_').collect();
                    let v: f64 = clean
                        .parse()
                        .map_err(|e| format!("{}: f64 literal `{}`: {e}", s.file, it.literal))?;
                    let _ = writeln!(out, "var {cyr}: f64 = {};", f64_literal(v)?);
                }
                "usize" | "u32" | "i32" | "u64" | "i64" => {
                    let v: i64 = it.literal.replace('_', "").parse().map_err(|e| {
                        format!("{}: integer literal `{}`: {e}", s.file, it.literal)
                    })?;
                    let _ = writeln!(out, "const {cyr} = {v};");
                }
                other => {
                    return Err(format!(
                        "t_private: {} has a const {} of type {other}: carry or excuse it",
                        s.file, it.name
                    ));
                }
            }
            out.push('\n');
        }
    }
    while out.ends_with("\n\n") {
        out.pop();
    }
    Ok(out)
}
