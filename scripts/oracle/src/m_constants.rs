//! `constants.tsv` — every public const of pravash 1.3.0, class A.
//!
//! One row per value: `const A s:<path> f:<bits>` for a scalar, and one row per field,
//! `s:<path>.<field>`, for a `FluidMaterial` preset. The list is `t_constants::all`, proven
//! complete against `rust-old/src` before any row is written.

use std::path::Path;

use crate::t_constants;
use crate::tsv::{Class, Module, f, s};

pub fn emit(root: &Path) -> Result<Module, String> {
    t_constants::complete(root)?;
    let mut m = Module::new("constants");
    for c in t_constants::all() {
        for (field, value) in &c.values {
            let path = if field.is_empty() {
                c.path.to_string()
            } else {
                format!("{}.{field}", c.path)
            };
            m.row("const", Class::A, &[s(path)], &[f(*value)]);
        }
    }
    Ok(m)
}
