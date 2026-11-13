# scripts/oracle: what the port derives from the Rust oracle

`oracle` drives the frozen Rust pravash 1.3.0 in [`rust-old/`](../../rust-old/), with its hidden `oracle` hooks
(port plan §4 step 0), and writes two kinds of output:
- **The golden corpus, `tests/golden/<module>.tsv`** (port plan §4 step 5). It has one row per call, with every
  float as its IEEE-754 bit pattern. The Cyrius parity harness (`tests/parity.tcyr`, from M0 step 7) replays every
  row against the port and checks it within its tolerance class.
- **The generated Cyrius sources**: `src/constants.cyr` from step 6, `src/constants_private.cyr` (the ported modules'
  private constants, from their Rust literals; M4 part a), and data modules as milestones need them. No value the
  port reads is typed by hand.

```sh
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml                        # write
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml -- --check             # verify; exit 1 on any difference
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml --features parallel -- --check   # rayon on: must match
python3 -I -B scripts/oracle/validate.py .                                                    # independent re-check
python3 -I -B scripts/oracle/validate.py . --write                                            # regenerate the corrected oracle, the f64-key oracle and the cr tables
python3 -I -B scripts/oracle/probe_tables.py .                                                # compile every generated value, compare bits
```

The generator depends on the oracle by path, on `hisab =1.4.0` (for glam's `DVec3`, to build `common`'s inputs), and
on `serde =1.0.229` and `serde_json =1.0.154` with `float_roundtrip` (the JSON rows: the oracle's own versions; the
default float parse is best-effort). All resolve through `Cargo.lock`, seeded from `rust-old/Cargo.lock`.

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
| `constants` | M0 | 35 | `src/m_constants.rs` (the lists and their completeness proofs: `src/t_constants.rs`) | `src/constants.cyr`: 14 public constants and the 6 std float constants pravash uses; `tests/constants_rows.cyr`: the harness's lookup from each golden path to its value |
| `math` | M0 | 33,239 | `src/m_math.rs` | — |
| (private constants) | M4 part a | — | — | `src/constants_private.cyr` (`src/t_private.rs`): each ported module's private `const`, parsed from its Rust literal; every other one fails the generator unless excused |

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
  step 8 baselines, D relative ≤ 1e-12 or a bound set before the first comparison, S a solver step from a state
  (per field the bound of its path: the harness's `_par_c_bound` gives it per function, 0 for the shallow-water
  step and for the grid step on every projection, the DST's against the corrected oracle where glibc misrounds a
  sine or cosine). Reference (R) checks are the ported tests' own tolerances, not golden rows.
- **The note** is a human rendering of the same call, for review only; the harness ignores it.

## The corrected oracle (ADR 0004)

`tests/golden/<module>.cr.tsv` holds the rows whose outputs change when every exp, ln, log10, sin, cos, hypot,
pow and atan2 result in the oracle's operations is correctly rounded. It is impetus's format (its ADR 0011), written
by `validate.py --write` from MPFR (`mpfr_ref.py`, libmpfr.so.6 through ctypes). It has five header lines, then:

```text
fn <TAB> inputs <TAB> corrected outputs <TAB> note
```

- **The key:** fn and inputs are the golden row's fields, byte for byte. The harness compares that row against
  the corrected outputs, and a pass is *accepted (ADR 4)*.
- **The note:** the oracle's own outputs, then each call glibc misrounded:
  `<fn>(<args>) glibc f:… correctly rounded f:…`.
- **A module with no such row** has no `.cr.tsv`, and `modules.txt` never lists one.

## The f64-key oracle (ADR 0002 #2)

`tests/golden/<module>.d3.tsv` holds the rows whose outputs change on f64 neighbour keys, the D3 switch of M5 part e:
every `oracle::sph_cell_key` row of `math`, and the `SphSolver::neighbor_lists` and solver step rows of `sph` whose
lists change. The oracle hashes f32 positions (hisab 1.4.0); the port keys f64 positions by floor(x · fl(1/h)) as i64,
hisab 3.4.1's cell, and lists the cells at a particle's key ± 1 per axis. `validate.py --write` writes the file from
its own transcription of that grid (`nb_build64`: a dict of cells and integer offsets, where the port sorts runs and
searches them), with the oracle's operations otherwise unchanged. The format is the `.cr.tsv` one:

- **The key:** fn and inputs are the golden row's fields, byte for byte. The harness compares that row against these
  outputs, and a pass is *accepted (ADR 2)*.
- **The outputs** of a `neighbor_lists` row are the f64 grid's dump: its cell size (h), the point and cell counts,
  each point's key and the lists. The oracle's dump also held the f32 positions, so every such row is here.
- **The note:** `oracle (f32 keys)` and the start of the oracle's own outputs.
- **`coupling.d3.tsv`** (M6 part a, ADR 0002 #23) holds the `drag_from_particles_indexed` rows whose sum the f64 keys
  reorder. The oracle fills a hisab `SpatialHash` with f32 positions. The port's cells-only grid keys them in f64
  and walks the block in the same (x, y, z) cell order. validate.py's `cp_query64` transcribes that grid from its
  definition: a dict of cells, and the block's occupied keys in sorted order.
- **A module the switch doesn't change** has no `.d3.tsv`.

The same `--write` writes the tables of the correctly rounded primitives:
- `src/crmath_data.cyr`, by `crtables.py`, a copy of impetus's: π and ln 2 come from exact integer series, and are
  checked against MPFR to 2,048 bits;
- `src/crlog_data.cyr`, by `lntables.py`, the ln part of kimiya's `mathx_tables.py`: MPFR values, with the error
  analysis's properties asserted.

## validate.py

Standard library plus MPFR (for the corrected oracle and the tables), through ctypes. It never runs the Rust
oracle:
- it parses every file (header, row count, five fields, classes, tokens) and checks that `modules.txt` lists
  exactly the files present;
- it checks `constants.tsv` against the literals in `rust-old/src`;
- it recomputes every class-A/B `math` row in Python, in Rust's operation order;
- it recomputes every `buoyancy`, `vortex` and `bridge` row, the class C ones through glibc (ctypes) for the record
  and MPFR for their corrected oracle (`vortex.cr.tsv`, `bridge.cr.tsv`);
- it decodes every `json` document again with Python's `json` (a correctly rounded float parse) under serde's
  derive semantics, and compares its verdict and values with serde_json's (the generator uses serde_json's
  `float_roundtrip` feature);
- it recomputes every `common` row in Python, in Rust's and glam's order: the validation rules, the defaults, the
  vector arithmetic, and the particle arena through a transcription of its free-list logic (the token layouts
  and the arena's op codes are in `src/m_common.rs`'s module comment);
- it recomputes every `shallow`, `grid`, `vof` and `phase_field` row through Python transcriptions of the solvers:
  the shallow-water step, the VOF advection and the phase-field step,
  the grid's samplers, advections, diffusion, forces, walls and four pressure solves (hisab 1.4.0's DST, IDST
  and CG among them), and the k-ε model, the ghost fluid and the MAC grid, with glibc's transcendentals for the
  record and MPFR's for the corrected oracle
  (`grid.cr.tsv`: the rows whose DST uses a sine or cosine glibc misrounds);
- it recomputes every `sph` row (part a: the kernels, the equations of state, the viscosity models), with glibc's ln and
  pow for the record and MPFR's for the corrected oracle (`sph.cr.tsv`); and part b's neighbour lists, from a
  transcription of `build_neighbors` on a dict of cells (f32 keys and probes, the kept grid), and the z-order sort
  (the Morton codes, stable on input index); and part c's brute-force sums, free step, block, split (glibc's sin and
  cos for the record, MPFR's for `sph.cr.tsv`) and merge; and part d's physics extensions, the 3×3 inverse
  (hisab 1.4.0's LU) and the contact angle (its cosine from glibc and MPFR); and M5's solver step, on the grid
  transcription, each scene's trajectory cached across its frames, and its PCISPH, DFSPH and multiphase steps; and,
  for the f64-key oracle, every list-dependent row again on `nb_build64` (M5 part e);
- it recomputes every `coupling` row (M6 part a):
  - the shapes, the bodies' signed distances and normals, the two-way coupling, both integrations, the brute-force
    drag and the level set, in Rust's and glam's order;
  - the indexed drag on a transcription of hisab 1.4.0's f32 `SpatialHash` and its `query_radius`, and again for
    `coupling.d3.tsv` on the port's f64 grid;
  - the two decoders under serde's externally tagged struct-variant semantics;
  - (M6 part b) the `FlipSolver`: P2G, the projection's Gauss–Seidel sweeps with their early stop, both G2P forms and
    the advection, each trajectory cached across its frames, and the scenarios' segments;
- it recomputes every `mpm` row (M6 part c): validation, the 2 × 2 SVD, the three stresses, the Drucker–Prager
  projection and the step. The transcendentals (hypot, atan2, sin, cos, ln, exp) come from glibc for the record and
  from MPFR for the corrected oracle, `mpm.cr.tsv`: the stress and step rows whose outputs change when every call is
  correctly rounded, each with the first misrounded calls in its note. The step trajectories are cached across their
  frames, per source of transcendentals;
- it recomputes every `compute` row (M7 part a): the f32 packing and unpacks (ctypes' f32), the kernel-parameter
  decoders under serde_json's f32 and u32 reads (each decimal rounded to f32 once, in exact rationals, refused when that
  is ±∞), the closed-form derivatives, and hisab 1.4.0's reverse-mode tape op for op on glibc's functions. The Tait
  derivative's pow comes from MPFR for the corrected oracle too, and changes no row, so there is no `compute.cr.tsv`;
- it recomputes every glibc class-C row by calling the same glibc functions through ctypes;
- it proves every `cbrt` row correctly rounded, with exact rationals;
- it recomputes the corrected oracle, the f64-key oracle and the two table files, and fails if any differs from what
  is committed;
- it checks crmath's π and ln 2 against MPFR, every table's head against MPFR's correctly rounded value, and the ln
  tables' ln 2 and 1/ln 10 against integer series MPFR doesn't share.

A tampered row fails it, and fails `--check`; a header naming another glibc is refused by `--check` with the reason.

## probe_tables.py

The step between the generator and the harness. For every `constants.tsv` row it derives the Cyrius name
(`t_constants::cyr_name`'s rule), writes `build/probe_tables.cyr`, which includes `src/constants.cyr` and prints each
value's bits, builds it with the project's `[deps]`, runs it, and compares with the row. It does the same for each
private constant in `src/constants_private.cyr`, against the Rust literal in the `# Rust:` line above it. A row with
no generated value, a value with no row, a build warning, or one differing bit fails it.
