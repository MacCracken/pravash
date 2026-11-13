//! `src/str_debug_data.cyr` — the code points Rust's `<str as Debug>` writes as `\u{…}` (M7 part b), for the port's
//! `Debug` of `DaimonConfig` and `HooshConfig` (`ai.rs`).
//!
//! `<str as Debug>` writes each `char` through `escape_debug_ext` with grapheme extenders escaped: the named escapes
//! `\0 \t \n \r \" \\`, then `\u{…}` (lowercase hex, no leading zeros) for every grapheme extender and every char
//! core's `is_printable` rejects, then the char itself. Which chars those are is core's own Unicode data (rustc 1.99.1,
//! `char::UNICODE_VERSION`), so the set is read from the oracle's toolchain char by char, never typed. A char's escape
//! doesn't depend on its neighbours (checked here: each is classified alone and after `a`, and both must agree).
//!
//! The table is inclusive ranges [lo, hi] in code point order. The surrogates, which no `str` holds, count as escaped,
//! which merges their neighbours' ranges.

use std::fmt::Write as _;

use crate::cyr::header;

/// Whether `<str as Debug>` writes code point `u` as `\u{…}` (a surrogate counts as escaped).
fn escaped(u: u32) -> Result<bool, String> {
    let Some(c) = char::from_u32(u) else {
        return Ok((0xD800..=0xDFFF).contains(&u));
    };
    let alone = format!("{:?}", c.to_string()).contains("\\u{");
    let after = format!("{:?}", format!("a{c}")).contains("\\u{");
    if alone != after {
        return Err(format!(
            "U+{u:04X}: escaped alone {alone}, after `a` {after}"
        ));
    }
    Ok(alone)
}

/// The escaped ranges, in order.
pub fn ranges() -> Result<Vec<(u32, u32)>, String> {
    let mut out = Vec::new();
    let mut start: Option<u32> = None;
    for u in 0..=0x10_FFFF_u32 {
        let e = escaped(u)?;
        match (e, start) {
            (true, None) => start = Some(u),
            (false, Some(s)) => {
                out.push((s, u - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, 0x10_FFFF));
    }
    Ok(out)
}

pub fn emit(prov: &str) -> Result<String, String> {
    let r = ranges()?;
    let (ma, mi, up) = char::UNICODE_VERSION;
    let mut out = header(
        "the code points Rust's str Debug escapes",
        prov,
        &[&format!(
            "From rustc 1.99.1's core, Unicode {ma}.{mi}.{up}, read char by char by t_str_debug.rs: the code points\n\
             `<str as Debug>` writes as \\u{{…}} beside its named escapes (\\0 \\t \\n \\r \\\" \\\\), the grapheme\n\
             extenders and the chars core's is_printable rejects. Inclusive ranges [lo, hi] in code point order; the\n\
             surrogates, which no string holds, are folded in. src/ai.cyr searches it (_pv_str_debug_escapes)."
        )],
    );
    let _ = writeln!(out, "# The number of ranges.");
    let _ = writeln!(out, "const _PV_STR_DEBUG_RANGES = {};", r.len());
    let _ = writeln!(out);
    let _ = writeln!(out, "# The ranges, two words each: lo, hi.");
    let _ = writeln!(out, "var _PV_STR_DEBUG_ESC: i64[{}] = {{", 2 * r.len());
    for chunk in r.chunks(6) {
        let line: Vec<String> = chunk
            .iter()
            .map(|(lo, hi)| format!("0x{lo:X}, 0x{hi:X}"))
            .collect();
        let _ = writeln!(out, "    {},", line.join(", "));
    }
    out.push_str("};\n");
    Ok(out)
}
