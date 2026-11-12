# scripts/oracle: what the port derives from the Rust oracle

`oracle` drives the frozen Rust pravash 1.3.0 in [`rust-old/`](../../rust-old/), with its hidden `oracle` hooks
(port plan §4 step 0), and writes two kinds of output:
- **The golden corpus, `tests/golden/<module>.tsv`** (port plan §4 step 5). It has one row per call, with every
  float as its IEEE-754 bit pattern. The Cyrius parity harness (`tests/parity.tcyr`, from M0 step 7) replays every
  row against the port and checks it within its tolerance class.
- **The generated Cyrius sources**: `src/constants.cyr` from step 6, and data modules as milestones need them. No
  value the port reads is typed by hand.

```sh
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml                        # write
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml -- --check             # verify; exit 1 on any difference
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml --features parallel -- --check   # rayon on: must match
python3 -I -B scripts/oracle/validate.py .                                                    # independent re-check
python3 -I -B scripts/oracle/probe_tables.py .                                                # compile every generated value, compare bits
```

The format, the comparison rules and the generator's structure are impetus's
(`/home/macro/Repos/impetus/scripts/oracle/`), which are kimiya's and tanmatra's: every port's harness reads the
same files. This README records what is specific to pravash.

## Rules

- **`rust-old/` is never built in place or edited.** This crate depends on it by path, with every solver feature
  and `oracle` on. `Cargo.lock` was seeded from `rust-old/Cargo.lock`, so every shared dependency is the version
  the oracle was tested with (hisab 1.4.0, which links glam 0.29.3).
- **Two builds, one corpus.** The serial build writes. The `parallel` build (rayon on in the oracle) may only
  `--check`: it refuses to write, and every file it computes must equal the serial build's byte for byte. That is
  how the corpus proves that thread count is not a parity variable (port plan §5, ADR 0002 #4).
- **The output is deterministic.** Two runs give byte-identical files, and `--check` passes on any machine with the
  recorded glibc. Grids use only + − × ÷, `floor`, decimal parsing, bit construction and seeded xorshift; never a
  transcendental std method, never a timestamp or a hash-map order.
- **Where the transcendentals come from.**
  - **glibc:** `sin`, `cos` (and `sincos`, which LLVM lowers `sin_cos` to), `atan2`, `hypot`, `exp`, `ln`, `log10`
    and `powf` (`nm -D` on the binary lists them as imported from it). Every TSV header records the glibc and the CPU
    class (`glibc 2.44 x86_64 fma`), and `--check` refuses a corpus from another one.
  - **compiler_builtins:** `cbrt`, linked into the binary from its `libm_math`, so those rows are platform-free.
    `validate.py` proves each one correctly rounded (port-parity §3).
- **The corpus is regenerated only from the oracle.** A wrong oracle value is fixed in Rust as 1.3.x, then
  regenerated. It is never patched in a `.tsv`.
- **Modules arrive with their milestone.** Each one starts by adding its golden emitter (`src/m_<module>.rs`, with
  the grid from the port plan §6) and, where it has data, its Cyrius emitter (`src/t_<module>.rs`), registered in
  `src/main.rs`.

| Module | Milestone | Golden rows | Golden emitter | Generated Cyrius |
|---|---|---|---|---|
| `constants` | M0 | 35 | `src/m_constants.rs` (the lists and their completeness proofs: `src/t_constants.rs`) | `src/constants.cyr`: 14 public constants and the 6 std float constants pravash uses |
| `math` | M0 | 33,239 | `src/m_math.rs` | — |

## Format 1

Five header lines, then one row per call:

```text
fn <TAB> class <TAB> inputs <TAB> outputs <TAB> note
```

- **Header:** `# pravash golden corpus, format 1: <module>`; `# oracle: <oracle>; libm: <platform>. …`; the
  regenerate command; the column legend; `# rows: <n>`.
- **Tokens** (space-separated; `-` when a list is empty):
  - `f:<16 hex>`: an f64 as its bit pattern, lowercase;
  - `i:<decimal>`: an integer, or a boolean as 0 / 1. A 64-bit word ≥ 2⁶³ (`as usize` of a huge value) reads
    negative;
  - `e:<Enum>.<Variant>`: an enum variant;
  - `s:<percent-encoded UTF-8>`: a string (every byte outside printable ASCII, and `%`, as `%XX`);
  - `n:`: `None`;
  - `E:<Variant>`: `Err(PravashError::<Variant>)`, with the payload not compared;
  - `v:<len>`: a sequence, whose element tokens follow.
- **Classes** (port plan §5): A exact (NaN matches any NaN), B ≤ 2 ulp, C an ulp bound per function set by the
  step 8 baselines, D relative ≤ 1e-12 or a bound set before the first comparison. Single-step (S) and reference (R)
  rows arrive with the solvers.
- **The note** is a human rendering of the same call, for review only; the harness ignores it.

## validate.py

Standard library only. It never runs the Rust oracle:
- it parses every file (header, row count, five fields, classes, tokens) and checks that `modules.txt` lists
  exactly the files present;
- it checks `constants.tsv` against the literals in `rust-old/src`;
- it recomputes every class-A/B `math` row in Python, in Rust's operation order;
- it recomputes every glibc class-C row by calling the same glibc functions through ctypes;
- it proves every `cbrt` row correctly rounded, with exact rationals.

A tampered row fails it, and fails `--check`; a header naming another glibc is refused by `--check` with the reason.

## probe_tables.py

The step between the generator and the harness. For every `constants.tsv` row it derives the Cyrius name
(`t_constants::cyr_name`'s rule), writes `build/probe_tables.cyr`, which includes `src/constants.cyr` and prints each
value's bits, builds it with the project's `[deps]`, runs it, and compares with the row. A row with no generated
value, a value with no row, a build warning, or one differing bit fails it.
