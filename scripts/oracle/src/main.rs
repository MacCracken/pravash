//! `oracle` — writes everything the Cyrius port derives from the frozen Rust pravash 1.3.0 in
//! `rust-old/` (with its hidden `oracle` hooks):
//! - the golden corpus, `tests/golden/<module>.tsv` (port plan §4 step 5);
//! - the generated Cyrius sources, `src/constants.cyr` and later data modules (§4 step 6).
//!
//! ```sh
//! cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml             # write
//! cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml -- --check  # verify
//! cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml --features parallel -- --check
//! ```
//!
//! The serial build writes. The `parallel` build (rayon on in the oracle) may only `--check`:
//! every row it computes must be byte-identical to the serial build's, which is how the corpus
//! proves that thread count is not a parity variable (port plan §5, ADR 0002 #4).
//!
//! `--check` regenerates every output in memory and compares it byte for byte with the committed
//! file. It exits 1 if any output differs or is missing, if a `.tsv` has no emitter, or if a file
//! was recorded on another glibc platform (`glibc.rs`): its transcendental rows are glibc's bits,
//! so they can only be checked where they were made. A `<module>.cr.tsv` is the corrected oracle
//! (ADR 0004) and a `<module>.d3.tsv` the f64-key oracle (ADR 0002 #2), which `validate.py --write`
//! writes and checks: here each is stale only when its module has no emitter.
//!
//! Adapted from impetus's `scripts/oracle/` (same format, same rules), which took it from kimiya's
//! and tanmatra's. Modules are added at the start of their milestone, with the golden grid from
//! the port plan §6: `constants` and `math` (M0); …

mod cyr;
mod glibc;
mod grids;
mod m_bridge;
mod m_buoyancy;
mod m_common;
mod m_constants;
mod m_coupling;
mod m_grid;
mod m_grid_c;
mod m_json;
mod m_math;
mod m_mpm;
mod m_phase_field;
mod m_shallow;
mod m_sph;
mod m_vof;
mod m_vortex;
mod t_constants;
mod t_private;
mod tsv;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Whether this build runs the oracle with rayon (check-only).
const PARALLEL: bool = cfg!(feature = "parallel");

/// Every golden emitter, in file-name order.
fn golden_modules(root: &Path) -> Result<Vec<tsv::Module>, String> {
    Ok(vec![
        m_bridge::emit(),
        m_buoyancy::emit(),
        m_common::emit(),
        m_constants::emit(root)?,
        m_coupling::emit(),
        m_grid::emit(),
        m_json::emit(),
        m_math::emit(),
        m_mpm::emit(),
        m_phase_field::emit(),
        m_shallow::emit(),
        m_sph::emit(),
        m_vof::emit(),
        m_vortex::emit(),
    ])
}

/// Every generated Cyrius source, in file-name order.
fn cyrius_outputs(root: &Path, prov: &str) -> Result<Vec<Output>, String> {
    let files = vec![
        ("src/constants.cyr", t_constants::emit(root, prov)?),
        ("src/constants_private.cyr", t_private::emit(root, prov)?),
        ("tests/constants_rows.cyr", t_constants::emit_rows(prov)),
    ];
    Ok(files
        .into_iter()
        .map(|(rel, text)| {
            let lines = text.lines().count();
            Output {
                rel: rel.to_string(),
                text,
                summary: format!("{lines} lines"),
            }
        })
        .collect())
}

/// One generated file: its path from the repo root, its contents, and a summary.
struct Output {
    rel: String,
    text: String,
    summary: String,
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// `version = "x"` from the first line that starts with it (the `[package]` one).
fn toml_version(text: &str) -> Option<&str> {
    let line = text.lines().find(|l| l.starts_with("version = \""))?;
    line.strip_prefix("version = \"")?.strip_suffix('"')
}

/// The locked version of `crate_name` in a Cargo.lock (its first entry).
fn lock_version<'a>(lock: &'a str, crate_name: &str) -> Option<&'a str> {
    let needle = format!("name = \"{crate_name}\"\n");
    let at = lock.find(&needle)? + needle.len();
    let line = lock[at..].lines().next()?;
    line.strip_prefix("version = \"")?.strip_suffix('"')
}

/// `pravash 1.3.0 (rust-old/), hisab 1.4.0, glam 0.29.3`: the oracle and its math.
fn oracle_provenance(root: &Path) -> Result<String, String> {
    let cargo = read(&root.join("rust-old/Cargo.toml"))?;
    let lock = read(&root.join("scripts/oracle/Cargo.lock"))?;
    let pv = toml_version(&cargo).ok_or("rust-old/Cargo.toml: no version")?;
    let hs = lock_version(&lock, "hisab").ok_or("scripts/oracle/Cargo.lock: no hisab")?;
    let glam = lock_version(&lock, "glam").ok_or("scripts/oracle/Cargo.lock: no glam")?;
    Ok(format!("pravash {pv} (rust-old/), hisab {hs}, glam {glam}"))
}

/// Write `out` to `root/rel`, or in check mode compare it. Returns false on a mismatch.
fn emit(root: &Path, check: bool, out: &Output) -> Result<bool, String> {
    let path: PathBuf = root.join(&out.rel);
    if !check {
        fs::write(&path, &out.text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("wrote    {} ({})", out.rel, out.summary);
        return Ok(true);
    }
    match fs::read_to_string(&path) {
        Ok(old) if old == out.text => {
            println!("ok       {} ({})", out.rel, out.summary);
            Ok(true)
        }
        Ok(_) => {
            println!("DIFFERS  {}", out.rel);
            Ok(false)
        }
        Err(_) => {
            println!("MISSING  {}", out.rel);
            Ok(false)
        }
    }
}

/// The libm platform in a TSV's `# oracle:` header line: the text from `libm: ` to the next
/// `. `, e.g. `glibc 2.44 x86_64 fma`.
fn recorded_platform(text: &str) -> Option<String> {
    let line = text
        .lines()
        .take(5)
        .find_map(|l| l.strip_prefix("# oracle: "))?;
    let (_, rest) = line.split_once("libm: ")?;
    Some(rest.split_once(". ").map_or(rest, |(p, _)| p).to_string())
}

/// The `.tsv` names in `dir`, sorted.
fn tsv_names(dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tsv"))
        .collect();
    names.sort();
    names
}

/// In check mode: every `.tsv` must have been recorded on the running platform. Its `# oracle:`
/// header line names the glibc, so a corpus made elsewhere is refused with a reason rather than
/// as an anonymous DIFFERS.
fn check_platform(dir: &Path, platform: &str) -> bool {
    let mut ok = true;
    for name in tsv_names(dir) {
        let Ok(text) = fs::read_to_string(dir.join(&name)) else {
            continue;
        };
        match recorded_platform(&text) {
            Some(p) if p == platform => {}
            Some(p) => {
                println!(
                    "PLATFORM tests/golden/{name}: recorded on {p}, running {platform}. Its \
                     transcendental rows are glibc's bits: check it on the recorded platform, \
                     or regenerate deliberately"
                );
                ok = false;
            }
            None => {
                println!("PLATFORM tests/golden/{name}: no recorded libm platform in its header");
                ok = false;
            }
        }
    }
    ok
}

fn run() -> Result<bool, String> {
    let check = std::env::args().skip(1).any(|a| a == "--check");
    if PARALLEL && !check {
        return Err("the parallel build only checks: run it with -- --check \
                    (the serial build writes the corpus)"
            .to_string());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = root
        .canonicalize()
        .map_err(|e| format!("{}: {e}", root.display()))?;
    let oracle = oracle_provenance(&root)?;
    let platform = glibc::platform();
    // Every TSV row may hold glibc's bits. The header names the oracle and the platform, but
    // not the build: the parallel build must reproduce the serial build's files exactly.
    let tsv_prov = format!("{oracle}; libm: {platform}");
    let dir = root.join("tests/golden");
    if !check {
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    if PARALLEL {
        println!("parallel build: every output must equal the serial build's");
    }

    let mut ok = true;
    if check {
        ok &= check_platform(&dir, &platform);
    }

    let mut outputs = Vec::new();
    let mut emitted = BTreeSet::new();
    for module in golden_modules(&root)? {
        let name = format!("{}.tsv", module.name);
        emitted.insert(name.clone());
        outputs.push(Output {
            rel: format!("tests/golden/{name}"),
            text: module.render(&tsv_prov),
            summary: format!("{} rows", module.len()),
        });
    }
    // The harness reads this list, so a module added here can't go untested: without a
    // dispatcher it is reported as pending, row by row.
    let mut index = String::from(
        "# pravash golden corpus: one module per line; tests/parity.tcyr replays each\n",
    );
    for name in &emitted {
        index.push_str(name.trim_end_matches(".tsv"));
        index.push('\n');
    }
    outputs.push(Output {
        rel: "tests/golden/modules.txt".to_string(),
        summary: format!("{} modules", emitted.len()),
        text: index,
    });
    // The generated .cyr files hold only the oracle's constants, so they name the oracle alone
    // and don't change with the platform.
    outputs.extend(cyrius_outputs(&root, &oracle)?);

    for out in &outputs {
        ok &= emit(&root, check, out)?;
    }

    // A .tsv with no emitter is stale: the harness would test rows nobody regenerates. A
    // `<module>.cr.tsv` (the corrected oracle, ADR 0004) or `<module>.d3.tsv` (the f64-key
    // oracle, ADR 0002 #2), both written by validate.py, belongs to its module's emitter.
    for name in tsv_names(&dir) {
        let module_file = name
            .strip_suffix(".cr.tsv")
            .or_else(|| name.strip_suffix(".d3.tsv"))
            .map_or_else(|| name.clone(), |m| format!("{m}.tsv"));
        if !emitted.contains(&module_file) {
            println!("STALE    tests/golden/{name} (no emitter)");
            ok = false;
        }
    }
    Ok(ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("oracle: {e}");
            ExitCode::from(2)
        }
    }
}
