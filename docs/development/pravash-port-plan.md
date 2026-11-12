# pravash Rust → Cyrius port plan

> **Plan of record** for porting pravash to Cyrius. Rust **1.3.0** is the final Rust release and the
> **reference implementation** (the oracle). The Cyrius port ships as **2.0.0**. The method is tanmatra's
> ([its ADR 0006](../../../tanmatra/docs/adr/0006-cyrius-port-with-rust-oracle.md)), as adapted by kimiya
> ([its port plan](../../../kimiya/docs/development/kimiya-port-plan.md), in progress) and impetus
> ([its port plan](../../../impetus/docs/development/impetus-port-plan.md), shipped 2.0.0 on 2026-11-10).
> impetus is the closest model: it is also a stateful simulation built on hisab. Where this plan says
> "as impetus", "as kimiya" or "as tanmatra", follow that document. §2 lists what is different here.
>
> ▶ **Start on Cyrius 6.7.18 with hisab 3.4.1**, once the oracle hooks are in and 1.3.0 is tagged (M0 step 0).
>
> ⏳ **Status: planning** (2026-11-11). Nothing has moved yet. §0.2 lists the decisions the maintainer
> makes before M0 starts.

| Thing | Value | Source |
|---|---|---|
| Rust source (oracle) | pravash **1.3.0**, committed (`f481c25`) but **not tagged**: the last tag is 1.2.0, and crates.io has 1.2.0. 6,882 code lines in `src/` (no tests, comments or blank lines), 16 modules, 193 `pub fn`s, 8 public enums | `Cargo.toml`, `VERSION`, `git tag` |
| Features | default `sph`, `grid`, `shallow`; optional `buoyancy`, `vortex`, `coupling` (coupling + mpm), `compute` (needs `sph`), `parallel` (rayon), `ai`, `logging` (tracing-subscriber, tracing-chrome) | `Cargo.toml` |
| Rust health, 2026-11-11 on rustc 1.99.1 | `cargo test --all-features`: 351 unit and 33 integration tests pass; all 7 doc tests are `ignore`d. fmt, clippy, audit, deny and rustdoc were clean at the 1.3.0 audit (2026-11-09) and are re-run at M0 step 0 | local run |
| Correctness baseline | The 1.3.0 full-module audit. Every fix has a regression test; the projection, solver-sign, delta-SPH, DFSPH, implicit-viscosity, MPM-stress and wave-speed fixes are mutation-checked | `CHANGELOG.md` 1.3.0 |
| Cyrius toolchain | **6.7.18** (installed and current) | `~/.cyrius/current` |
| hisab | **3.4.1**, Cyrius, tag `3.4.1`, bundle `dist/hisab.cyr` | `/home/macro/Repos/hisab` |
| ganita | **1.2.15**, bundled in the 6.7.18 stdlib | `~/.cyrius/versions/6.7.18/lib/ganita.cyr` |
| Oracle's dependencies | hisab **1.4.0** (features `num`, `geo`, `autodiff`), which links glam **0.29.3**; serde, thiserror, tracing; rayon behind `parallel`. `Cargo.lock` is **untracked** (the library-crate rule) | `Cargo.lock` |
| Oracle's math | Rust std → host **glibc 2.44** for `ln` (8 calls), `powf` (6), `exp` (5), `cos` (4), `hypot` (3), `sin` (2), `sin_cos` (2), `cbrt` (2), `atan2` (2), `log10` (1). glam's `DVec3` (f64) is used everywhere; its `Vec3` (**f32**) is used only for the SPH and coupling neighbour hash | `grep` of `src/` |
| Published dependents (crates.io) | badal, pavan, khanij, soorat, kiran: all Rust, all `^1`. soorat is local and Rust; its optional `fluids` feature pulls pravash `sph` + `shallow` | crates.io reverse deps |
| Cyrius consumers | **None.** impetus 2.0 (Cyrius) declares no pravash dependency, and kiran and joshua have no local repo | `~/Repos` |
| Target | pravash **2.0.0**, Cyrius | this document |

## 0. Decisions

### 0.1 Carried from the sibling ports

- **The oracle is frozen.** Rust 1.3.0 moves to `rust-old/` and is never edited. A generator crate in
  `scripts/oracle/` drives it by path. If an oracle value is wrong, it is fixed in Rust as 1.3.x and the
  corpus is regenerated (as tanmatra).
- **No Rust 2.0** (as kimiya and impetus). API shape changes land in the Cyrius API only, accepted by
  ADR 0002 (§3.3). New physics stays out of the port.
- **hisab-first, but off the step path until an allocator lands.** cyrius S1 (a scoped allocator) and hisab
  H9 (allocator-scoped calls) are both still **open** (checked 2026-11-11). Until one of them lands, every hisab
  call that returns a value leaks from the global bump heap. So per-step code is pravash-owned: the neighbour
  search, the pressure solves and the vector math in loops (as impetus [ADR 0012](../../../impetus/docs/adr/0012-impetus-owned-3d-math-stopgap.md)).
  hisab serves the calls that are not per-step. Where hisab 3.4.1 lacks a routine the oracle used, the port
  transcribes Rust hisab 1.4.0's routine and files the gap
  (kimiya [ADR 0003](../../../kimiya/docs/adr/0003-transcribe-missing-hisab-routines.md)).
- **Performance is measured, not gated** (as impetus). Every Rust benchmark gets a Cyrius twin, and both
  columns are recorded per milestone in `docs/benchmarks-rust-v-cyrius.md`. A slow row is a finding to
  file, with its numbers.
- **Parity is functional; accuracy beats glibc.** This is the maintainer's policy from kimiya
  (2026-11-09), also applied in impetus [ADR 0011](../../../impetus/docs/adr/0011-correctly-rounded-exp-sin-cos.md).
  The port uses correctly rounded primitives, never ones that copy glibc's last-ulp errors. Where glibc
  misrounds, the golden expectation is the oracle's formula evaluated on the correctly rounded primitive
  (`*.cr.tsv`), reported as accepted. §5 sets pravash's scope for this.

### 0.2 Open: the maintainer decides before M0 (recommendation first)

| # | Question | Recommended | Alternative |
|---|---|---|---|
| D1 | **Tag and publish 1.3.0.** `release.yml` runs `cargo publish` on any version tag. Tagging therefore publishes 1.3.0 to crates.io, and its five Rust dependents on `^1` get the audit's behaviour changes | Land the `oracle` hooks (M0 step 0b) first. Then tag 1.3.0 on that commit and let it publish. That gives one frozen point, and Rust users get the audited physics | Tag without publishing. Or tag the current commit and add the hooks afterwards as unreleased tooling, as impetus did |
| D2 | **Where the correctly rounded math comes from.** pravash needs exp, ln, log10, pow, sin, cos, sin_cos, hypot and atan2. impetus has its own copies (`src/crmath.cyr`, ADR 0011), and so does kimiya (ln, log10, pow, exp, sin, cos, acos, atan2). The ganita issue asking for a correctly rounded tier is open. pravash would be the **third** crate to need these, which is CLAUDE.md's cue to extract an abstraction | Extract them into ganita as a separate task, seeded from impetus's crmath and kimiya's ln/log10. Vendor a stopgap copy into `src/crmath.cyr` (ADR 0004) only if ganita hasn't landed when M1 needs them | Vendor a third copy now and extract later |
| D3 | **The SPH neighbour hash.** The oracle hashes f32 positions (hisab 1.4.0 `SpatialHash`, with f32 cells), and the roadmap lists the resulting precision loss at large coordinates as a defect. The candidate order (by cell, then by insertion) fixes the order of every SPH sum, so it decides whether SPH steps can be compared bit for bit | Port the f32 keys bit-faithfully (probe 3). Neighbour lists are then class A and SPH steps class B/C through M5. Then switch to f64 cell keys before 2.0 as an ADR 0002 divergence, with the f32 rows remapped by name (kimiya's "noted rows mapped") | Use f64 keys from the start. This is simpler, but every SPH step row becomes class D, because the summation order changes |
| D4 | **serde.** There are 34 `Deserialize` derives, including `FluidGrid` and `ShallowWater`, which consumers can use to save and restore state | Don't carry it in 2.0 (as kimiya): there is no Cyrius consumer and no behaviour depends on it, so it waits for demand. If it is wanted later, impetus's serde-compatible JSON is the model (ADR 0010: `fmtf.cyr`, and `json.cyr` on bayan) | Carry serde-compatible JSON now, as impetus did. That costs about one milestone part per solver |
| D5 | **The items the 1.3.0 audit deferred.** The roadmap lists Neumann and periodic pressure, SPH 2D kernel normalisation, an external-force input to `step()`, neighbour filtering by h, quadratic B-spline MPM and others | None of them in 2.0: they are new physics with no oracle. They start after 2.0, checked by reference tests only (as impetus). The f32 hash fix (D3) is the one exception | Pull the API-shape items (the external-force input, a dimension on the SPH config) into ADR 0002 |
| D6 | **Rust-only features** | `bridge`: deferred until a Cyrius consumer exists (as tanmatra and impetus). `parallel`: not in 2.0. The serial step is the oracle; a `thread.cyr` path is a 2.x item and must give the serial bits. `ai` (two endpoint-config structs, no behaviour): not carried. `logging`: sakshi's init replaces tracing-subscriber and tracing-chrome | Port `bridge` (10 functions, 17 tests) in 2.0 as a small parity-tested file |

## 1. Porting models and references

- **impetus 2.0.0** (`/home/macro/Repos/impetus`), the template. Copy these and adapt them:
  - `scripts/oracle/`: the generator crate, with the glibc header, the step format and the dump flattening, plus `validate.py`;
  - `tests/parity_core.cyr`, `tests/ulp.cyr`, the state loaders and the step comparator;
  - `scripts/surface_parity.py`, `scripts/rust_test_ledger.py`, `scripts/bench_record.py` and `scripts/doc_examples.py`;
  - `src/mathx.cyr`'s Rust-semantics helpers;
  - the CLAUDE.md translation of the Rust principles;
  - on the Rust side, the `oracle` feature pattern (`src/oracle.rs`, plus `tests/oracle.rs` proving the hooks are read-only).
- **kimiya** (in progress), the model for:
  - ADR 0003, transcribing missing hisab routines;
  - ADR 0002's oracle ↔ `Err` mapping and its "noted rows mapped" accounting;
  - the accuracy policy and its working scope;
  - correctly rounded ln and log10.
- **tanmatra 2.0.1**: the parity classes, the port-parity rule (K_F = 2·S·calls + 2) and the origin of the Rust-test ledger.
- **hisab 3.4.1**, the dependency. Read `CLAUDE.md`, `docs/guides/migration-3.4.md` (the Result convention)
  and the doc comment of every function pravash calls.
- **Language and toolchain** (`/home/macro/Repos/cyrius`): `docs/guides/cyrius-guide.md`,
  `docs/stdlib-reference.md`, `docs/development/ecosystem-migration.md` (the 6.7.x entries),
  `cyrius help manifest`, `BENCHMARKS.md`.
- **Open upstream issues this port inherits:**
  - `cyrius/docs/development/issues/2026-11-08-cyrius-port-clobbers-project-docs-and-omits-bench.md`: still open, so M0 step 2 repairs what the scaffold overwrites;
  - `cyrius/…/2026-11-09-no-scoped-allocator-for-alloc.md` (S1);
  - `hisab/docs/development/issues/2026-11-09-allocator-scoped-calls.md` (H9);
  - `ganita/docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`;
  - cyrius's `memcpy` byte loop (`2026-11-02`, impetus S5).

---

## 2. What is different from impetus

impetus is one world with two exclusive backends. pravash is a toolbox of independent solvers and free
functions, so six things change.

1. **Eleven stateful types instead of one world:**
   - `SphSolver`, with four step variants: `step` (WCSPH), `step_pcisph`, `step_dfsph` and `step_multiphase`;
   - `FluidGrid`, `MacGrid`, `KEpsilon`, `ShallowWater`, `VofField`, `PhaseField`, `FlipSolver`, `MpmSolver` and `ParticleArena`.

   Each needs its own oracle dump, state loader and step-row format. Their private fields are invisible to
   the public API: `SphSolver` 13, `FlipSolver` 11, `MpmSolver` 8, `FluidGrid` 5, `ShallowWater` 5,
   `ParticleArena` 5, `ArenaHandle` 3, `MacGrid` 2, `KEpsilon` 2, `VofField` 1, `PhaseField` 1. Exposing them
   through the oracle hooks is M0 step 0's Rust work.
2. **Iterative solvers inside a step.** These include Gauss–Seidel, CG (relative tolerance 1e-6·‖b‖, floor 1e-14),
   multigrid V-cycles (at most 10 levels), the DST, PCISPH and DFSPH iterations, implicit viscosity and the FLIP
   pressure loop. An iteration count is class A, but a 1-ulp difference at a tolerance edge can change it. So
   every convergence test is a branch point (§5), and the iteration count is part of every step row.
3. **f32 in two places:** the SPH and coupling neighbour hash (D3), and `compute`'s GPU packing
   (`PackedParticles`, 41 f32 sites). Cyrius has f32 SIMD lanes and `f32_to_f64` (in `mabda`). Whether scalar
   f64 → f32 rounding and f32 arithmetic match Rust is probe 3.
4. **Particle counts.** SPH and FLIP run 10³–10⁵ particles a step. The neighbour lists keep all 27 cells
   unfiltered, about 6× the true neighbours in 3D. Memory per step (probe 1) and the particle layout (probe 2)
   decide most of the speed.
5. **An unstable sort with ties.** `sort_by_zorder` uses `sort_unstable_by_key` on a Morton code. Particles in
   one cell share a key, so their order after the sort depends on Rust's sort algorithm, and every later step's
   sums follow that order (probe 5).
6. **No snapshot format to design**, unless D4 carries serde. The state loader that the parity harness needs
   is test-only.

---

## 3. Scope

### 3.1 In scope for 2.0.0

| Milestone | Rust module | Code lines | `pub fn` | Tests | Cyrius module(s) | hisab use |
|---|---|---|---|---|---|---|
| M1 | `error.rs` | 26 | — | 6 | `src/error.cyr` | — |
| M1 | *(public and private constants, presets)* | — | — | — | `src/constants.cyr` (generated) | — |
| M1 | *(std f64 methods)* | — | — | — | `src/mathx.cyr`; `src/crmath.cyr` or ganita (D2) | — |
| M1 | `common.rs` | 611 | 35 | 49 | `src/vec3.cyr` (the `DVec3` operations pravash uses), `src/common.cyr` | none: a pravash-owned glam transcription, as impetus's `math3.cyr` |
| M1 | `buoyancy.rs` | 105 | 5 | 16 | `src/buoyancy.cyr` | — |
| M1 | `vortex.rs` | 47 | 5 | 12 | `src/vortex.cyr` | — |
| M2 | `shallow.rs` | 692 | 12 | 44 | `src/shallow.cyr` | — |
| M3 | `grid.rs` | 1,464 | 21 | 49 | `src/grid_solve.cyr` (DST, IDST and CG, §7.2), `src/grid.cyr`, `src/mac_grid.cyr`, `src/k_epsilon.cyr` | `num_dst` / `num_idst` if probe 8 allows; otherwise transcribed |
| M3 | `vof.rs`, `phase_field.rs` | 368 | 19 | 19 | `src/vof.cyr`, `src/phase_field.cyr` | — |
| M4 | `sph.rs`: kernels, equation of state, neighbour hash, free functions, configs | ≈ 900 | ≈ 30 | ≈ 35 | `src/sph_kernels.cyr`, `src/neighbor.cyr`, `src/sph_fns.cyr` | none on the step path; the MLS 3×3 inverse is transcribed (§7.2) |
| M5 | `sph.rs`: `SphSolver` and its four steps, phase change, secondary particles | ≈ 1,270 | ≈ 19 | ≈ 44 | `src/sph.cyr`, `src/sph_pcisph.cyr`, `src/sph_dfsph.cyr`, `src/sph_multiphase.cyr` | — |
| M6 | `coupling.rs` | 590 | 19 | 33 | `src/coupling.cyr` | the M4 neighbour hash |
| M6 | `mpm.rs` | 424 | 6 | 13 | `src/mpm.cyr` | — |
| M7 | `compute.rs` | 201 | 9 | 11 | `src/compute.cyr` | `ad_tape_*` for `autodiff_gradient` (not per-step) |
| M7 | `logging.rs` | 20 | 3 | 1 | `src/logging.cyr` (sakshi) | — |

The `sph.rs` split between M4 and M5 is an estimate; `scripts/surface_parity.py` fixes it at M4's start. The
surface-parity check covers every `pub fn` and every `pub` method on the re-exported types. In scope: 6,718
lines, which is the 6,882 minus `bridge`, `ai` and `lib.rs`.

### 3.2 Deferred or not carried

| Rust surface | Decision | Why |
|---|---|---|
| `bridge.rs` (10 fns, 17 tests): pavan, ushma, goonj | **deferred** until a Cyrius consumer exists (D6) | thin primitive-value wrappers; a small parity-tested file later |
| `parallel` (rayon: 5 `par_iter`, 2 `par_chunks`) | **not in 2.0** (D6) | probe 6 proves that Rust's parallel and serial builds give the same bits, so the serial port answers for both |
| `ai.rs` (`DaimonConfig`, `HooshConfig`) | **not carried** (D6) | endpoint strings with no behaviour |
| serde derives | **not carried** (D4) | no Cyrius consumer |
| `tracing` spans and warnings (16 sites) | **sakshi** spans behind a level | the warn-once-per-step rules are kept |
| `thiserror` text, `PravashError::Diverged { reason }` | **`PV_ERR_*`** append-only codes with no detail string (as tanmatra) | §3.3 #1 |
| `Send + Sync`, `#[inline]`, `Cow<'static, str>` | **not carried** | Rust-only mechanics |

### 3.3 API changes in the Cyrius version (accepted divergences, ADR 0002)

This is the seed list; it is finalised at M0 step 2. Each row says how the harness still compares the
function, so nothing leaves parity entirely.

| # | Rust 1.3.0 | Cyrius 2.0.0 | Golden comparison |
|---|---|---|---|
| 1 | `Result<T, PravashError>` carrying detail strings | value-form `Result` plus a `PV_ERR_*` code | class A on the variant |
| 2 | closures (`autodiff_gradient(f: impl Fn(&mut Tape, &[Var]) -> Var, …)`) | a function pointer plus a context word | the gradient's own class |
| 3 | f32 neighbour-hash keys | f64 keys before 2.0 (D3) | f32 rows mapped by name |
| 4 | `sort_by_zorder`'s tie order (Rust's unstable sort) | a stable sort on (key, input index), if probe 5 shows Rust's order can't be reproduced cheaply | class A on the key sequence and on the multiset; later steps start from the oracle's order |
| 5 | features select modules at compile time | one `[lib]` bundle, or profiles if probe 7 says so | — |
| 6 | bit-identical only on one platform | **bit-identical across targets** for pravash-owned arithmetic (as impetus) | measured on x86_64 ELF, x86_64 PE and aarch64 |

---

## 4. M0: the prep, then initialize with `cyrius port`

0. **Freeze a green oracle (Rust, before the move).** This is maintainer work and ends with D1.
   1. **Re-run the P(-1) gates** on HEAD:
      - fmt;
      - clippy `-D warnings` with `--all-features`, with the defaults, with `--no-default-features`, and with each
        feature alone (1.2 shipped a `compute` that didn't build on its own);
      - rustdoc `-D warnings`, `cargo audit`, `cargo deny`, and the tests.
   2. **The `oracle` feature** (non-default, `#[doc(hidden)] pub mod oracle`, as impetus's `src/oracle.rs`). It is
      read-only and provides:
      - `oracle_dump()` for each stateful type in §2 item 1. A dump covers every private field, the scratch
        buffers whose contents carry from one step to the next (Verlet's previous acceleration, the PCISPH and
        DFSPH buffers, the FLIP and MPM grids, ghost cells), and the last step's iteration counts and residuals;
      - the SPH neighbour hash: each particle's cell key and its candidate list, in the order the solver sums them;
      - each pressure solver (Gauss–Seidel, DST, CG, multigrid) on one given system, returning the solution,
        the iteration count and the residual;
      - the hisab 1.4.0 calls exactly as pravash makes them (`dst`, `idst`, `conjugate_gradient`,
        `matrix_inverse`, `reverse_gradient`), so every transcription is compared against the oracle's own bits.

      `tests/oracle.rs` proves that dumping every frame leaves a 300-step run of each solver bit-identical. The
      gates are those of step 0.1, plus clippy on `oracle` together with each solver feature.
   3. **Track the lockfile for the freeze.** `Cargo.lock` is gitignored under the library-crate rule, but the
      oracle's bits depend on hisab 1.4.0 and glam 0.29.3. Commit it; it moves to `rust-old/Cargo.lock`. ADR 0001
      records the exception. The generator crate's own lock is seeded from it (as kimiya).
   4. **Record glibc 2.44** (`ldd --version`). Every TSV header carries it, and `--check` refuses a corpus from another glibc.
   5. **Commit, then tag 1.3.0** as D1 decides. Nothing in `src/` changes after this point.
1. **Scaffold.** Run `cargo clean` first (`target/` is 3.3 GB). Then run `cyrius port --dry-run .`, then `cyrius port .`.
   - Expect these to move into `rust-old/`: `Cargo.toml`, `Cargo.lock`, `src/`, `tests/`, `benches/`, `examples/`,
     `Makefile`, `deny.toml`, `codecov.yml`, `rust-toolchain.toml`. `--dry-run` doesn't list them by name.
   - Expect these to be written: `src/main.cyr`, `cyrius.cyml`, `tests/pravash.{tcyr,bcyr,fcyr}`,
     `docs/{adr,architecture,guides,examples}/` and `docs/development/state.md`.
2. **Repair what the scaffold overwrites.** The clobber issue is still open on 6.7.18; impetus's §4 step 2 is the checklist.
   - **CLAUDE.md**: merge `git show HEAD:CLAUDE.md` with the scaffold's Cyrius rules, keeping the identity, the
     process and the DO NOT list, including "NEVER use `gh`". Translate the principles as impetus did:
     - `#[non_exhaustive]` → append-only enum values;
     - `#[must_use]` → `#must_use` / `#pure`;
     - no `unwrap`/`panic!` → no `+?`, `-?`, `*?`, `assert_fatal`, `result_unwrap` or `panic` in `src/`;
     - `#[inline]` → `#inline`, which takes at most two parameters;
     - "Vec arena over HashMap" → typed stride arenas and sorted arrays, never `hashmap` (its seed is random per process);
     - "`write!` over `format!`" and "Cow over clone" → `str_builder` writes, borrow by pointer, copy only into owned storage;
     - "tracing on all operations" → sakshi spans, behind a level;
     - "feature-gate optional deps" → `[features]` selects optional dependencies only.

     Also replace the consumer list with the real one (header table) and replace the MSRV with the Cyrius pin.
   - **Roadmap**: restored from git, with this plan's section kept. **`state.md`**: rewritten for 1.3.0 → 2.0.0-dev.
   - **Workflows**: keep the scaffold's Cyrius CI, and add `test --bounds`, `deps --locked` and `audit`. Move the
     Rust workflows to `rust-old/.github/workflows/`, where the publish job can no longer fire.
   - **`.gitignore`**: append `/rust-old/target/` and `/build/`, and add `!rust-old/Cargo.lock` so the frozen lock stays tracked.
   - **Move Rust-only material** to `rust-old/`:
     - `scripts/bench-history.sh`, `scripts/bench-regression.sh` and `scripts/version-bump.sh`;
     - `bench-history.csv` (untracked, but it is the Rust benchmark record);
     - `docs/architecture/overview.md`;
     - `docs/development/migration.md` (the 1.2 → 1.3 guide).

     `docs/development/history.md` stays.
   - **ADRs** (pravash has none yet):
     - 0001, the port with a Rust oracle, including the lockfile exception;
     - 0002, the §3.3 API divergences;
     - 0003, transcribing missing hisab routines (§7.2);
     - 0004, the correctly rounded primitives (per D2).
   - **Checks**: `cyrius.cyml` reads `version = "${file:VERSION}"`, and `bench` is in `[test] stdlib`. Document the
     scaffold's stub functions so `cyrius audit` passes. Rewrite CONTRIBUTING, SECURITY and the README for Cyrius.
3. **`cyrius.cyml`.** There is no `[lib]` until M1, because `modules = []` is invalid. Confirm the keys with `cyrius help manifest`.
   ```toml
   [package]
   name = "pravash"
   version = "${file:VERSION}"
   license = "GPL-3.0-only"
   language = "cyrius"
   cyrius = "6.7.18"

   [deps]
   stdlib = ["syscalls", "string", "alloc", "str", "fmt", "io", "vec", "math", "ganita",
             "tagged", "result", "freelist", "fnptr"]   # sakshi arrives via hisab

   [deps.hisab]
   git = "https://github.com/MacCracken/hisab.git"
   tag = "3.4.1"
   modules = ["dist/hisab.cyr"]
   # scope = "test" if no library module ends up calling hisab (impetus ADR 0007, second amendment);
   # sakshi then moves into [deps] stdlib

   [test]
   stdlib = ["assert", "bench"]

   [lib]
   modules = ["src/error.cyr", ...]
   ```
   Module order: error → constants → mathx → crmath → vec3 → common → buoyancy → vortex → shallow → grid_solve →
   grid → mac_grid → k_epsilon → vof → phase_field → sph_kernels → neighbor → sph_fns → sph → sph_pcisph →
   sph_dfsph → sph_multiphase → coupling → mpm → compute → logging.
4. **Toolchain and math check.** Confirm the counterpart of every Rust f64 method in §7.1 on 6.7.18. impetus
   probed the following, and its results carry over:
   - `f64_min` / `f64_max` don't follow Rust's NaN rules;
   - the stdlib has no public `is_nan`, `is_finite`, `copysign`, `sin_cos`, `EPSILON`, `INFINITY` or `NAN`;
   - `f64_exp` and the native `f64_atan` give different bits on x86_64 and aarch64.

   pravash adds:
   - `ganita_f64_cbrt`;
   - `f64::MAX` (`cfl_dt`'s value when there is no flow);
   - saturating `as i32` from an f32 floor;
   - `saturating_sub` on `usize`.
5. **Golden-value generator** (`scripts/oracle/`), copied from impetus's crate:
   - TSV format 1 with f64 values as bit patterns, deterministic output, a `--check` mode, and the glibc header;
   - `validate.py` recomputes the class A/B rows in Python;
   - one build with `--features full,oracle` but without `parallel`, and a second build with `parallel` for probe 6;
   - the step format as impetus: each row names a scenario and a frame. Its inputs are the dump at frame k plus the
     call's arguments; its outputs are the dump at k + 1 plus the iteration counts.
6. **Table and constant generator** (the same crate), producing `src/constants.cyr` through the oracle's API, never
   by retyping. It covers:
   - the public constants: the `FluidMaterial` presets (`WATER`, `OIL`, `HONEY`, `AIR`, `LAVA`) and the drag
     coefficients in `buoyancy` and `coupling`;
   - the private ones: `MIN_DENSITY`, `MIN_SINGULAR_VALUE`, the CG and multigrid limits, the DST eigenvalue
     threshold (1e-12), the Riemann ε (1e-8), and so on.

   The generator proves the list is every `const` in `rust-old/src`. Write decimal literals out positionally (the
   lexer has no exponent notation), and never put a hex pattern in a `const`.
7. **Parity harness**, from impetus: `tests/parity.tcyr`, `tests/parity_core.cyr` and `tests/ulp.cyr`, plus:
   - a state loader and a step comparator per solver, with per-field classes;
   - the divergence table, which names each ADR 0002 row and reports it by name.
8. **Baseline measurements**, recorded in `docs/development/port-parity.md`:
   - the math sweep through every Cyrius counterpart against glibc, on x86_64 ELF, x86_64 PE (wine) and aarch64 (qemu);
   - K set by the port-parity rule;
   - the corrected oracle (`*.cr.tsv`) for the correctly rounded primitives.
9. **Probes.** Each one is a small program, answered with numbers before M1 writes any code. The programs go in
   `scripts/probes/` and the results in `port-parity.md`.
   1. **Memory model.** Solver-owned storage (freelist or a solver arena) plus a per-step scratch arena, with hisab
      off the step path. Gate: 10⁴ steps of a 10⁴-particle SPH dam break, and 10⁴ steps of a 128² `FluidGrid` on
      CG, each reach a flat heap high-water mark after warm-up.
   2. **Particle layout and vector representation.** Time the SPH density and force loops two ways: over a typed
      stride array of `PvParticle` records (array-of-structs, like Rust's `FluidParticle`) and over column arrays
      (like `ParticleSoa`). Run each with f64 locals and with struct values; impetus measured struct values at 2.5×
      slower than f64 locals.
   3. **f32 fidelity.** Check f64 → f32 rounding (to nearest), f32 multiply and floor, and saturating `as i32`. The
      test: reproduce hisab 1.4.0's `SpatialHash` cell key for 10⁶ seeded positions, including positions on cell
      boundaries and |x| up to 10⁷. Also reproduce the bits of `PackedParticles`' f32 packing.
   4. **The oracle's neighbour candidate order.** Confirm that hisab 1.4.0's query visits cells in a fixed offset
      order, and each cell in insertion order, so that no `HashMap` iteration order reaches a sum.
   5. **Z-order sort ties.** Can the port reproduce the order Rust's `sort_unstable_by_key` gives on 1.99, or does
      it sort on (key, index) instead (ADR 0002 #4)?
   6. **rayon.** The `parallel` build's dumps equal the serial build's, bit for bit, on every scenario in the step
      corpus. Check the reductions as well as the ordered `par_iter` collects.
   7. **Bundle shape.** Compare one `[lib]` holding every module with per-feature profiles (`[lib.sph]`, …). Measure
      the compile time and binary size for a consumer that uses only `shallow`, soorat's case.
   8. **hisab 3.4.1 against 1.4.0** for `dst`, `idst` and the autodiff tape: bits, allocations, and whether the
      algorithm is the oracle's. The answer decides between "use" and "transcribe" in §7.2.

**Gate:**
- `cyrius build` and `cyrius bench` pass;
- `cyrius test` runs the harness (0 modules) with no warnings;
- `cyrius lint` and `fmt --check` are clean;
- committed: the oracle hooks (now in `rust-old/`), the generator, `validate.py`, `tests/ulp.cyr`, ADRs 0001–0004,
  the probe results and the upstream issues (§8).

---

## 5. Parity policy

tanmatra's classes apply, with impetus's S and R classes for simulations:

| Class | Tolerance | pravash examples |
|---|---|---|
| **A — exact** | bit-identical / equal integers | error variants, config validation verdicts, iteration counts, neighbour candidate lists (D3's f32 keys), z-order key sequences, arena handles and free-list order, `classify_flow` regimes, wet/dry masks, FLIP cell marks, constants and presets |
| **B — arithmetic** | ≤ 2 ulp, expected 0 | the polynomial kernels (poly6, spiky, viscosity, cubic spline, Wendland) and their gradients, buoyancy and drag, Gauss–Seidel sweeps, advection interpolation, the shallow-water HLL fluxes (`sqrt` is exact), MPM transfers, `powi` sites |
| **C — transcendental** | ulp ≤ K by the port-parity rule; **0** on correctly rounded primitives, against the corrected oracle | Tait EOS and non-Newtonian `powf`, the shallow-water damping `powf` and Manning `cbrt`, Lamb–Oseen `exp`, the Kolmogorov scale (`ln`, `exp`), the DST eigenvalues (`cos`), MPM's 2×2 SVD (`hypot`, `atan2`, `sin_cos`), its Hencky strain (`ln`, `exp`), Drucker–Prager's friction angle (`to_radians`, `sin`), the contact angle |
| **D — reordered or hisab-backed** | relative ≤ 1e-12, or a per-function bound set before its first comparison | sums reordered by an accepted divergence (D3's f64 keys after the switch, ADR 0002 #4); any hisab 3.x path that isn't a transcription |
| **S — single step** | per field, the class of the path that produced it | one step from an oracle dump. Iteration counts, masks and neighbour lists are class A |
| **R — reference** | the physical tolerance stated in the test | the 33 integration tests, the 1.3.0 regression tests and the unit tests that assert physics: hydrostatic rest, wave speed √(gh), divergence after projection, VOF and shallow-water mass, Lamb–Oseen decay, terminal velocity, dam-break front position |

Rules:
- **Never widen a class to make a row pass** (as tanmatra). A failure is one of three things: a port defect; a
  Cyrius, hisab or ganita defect, which is filed upstream; or an oracle defect, which is fixed in Rust 1.3.x and
  regenerated. The one exception is a divergence accepted by an ADR, which the harness still reports by name.
- **Bit-level trajectories only where the scene isn't chaotic**: a hydrostatic tank, a lake at rest, a linear
  shallow-water wave, a single droplet, Lamb–Oseen decay. They are compared over the whole horizon. Dam breaks,
  splashes and plumes get class S rows at frames {1, 2, 10, 60, 300}, plus class R.
- **Branch points, tested on both sides:**
  - the solver tolerances (CG's 1e-6·‖b‖ and its 1e-14 floor, PCISPH's compression-only convergence with at least one iteration, the implicit-viscosity and FLIP checks);
  - the multigrid depth cap (10 levels);
  - the DST eigenvalue threshold (1e-12);
  - `MIN_DENSITY`;
  - the surface-tension gate |∇c| > 0.1/h;
  - the non-Newtonian saturation cap;
  - shallow water's `dry_thr`, its Riemann ε (1e-8) and its `breaking_threshold`;
  - the Reynolds thresholds in `classify_flow`;
  - MPM's det(F) clamp and the Drucker–Prager yield;
  - the phase-field |φ| > 1.5 warning;
  - VOF's Courant ≤ 0.5 sub-cycling.
- **The accuracy policy's scope for pravash** (§0.1):
  - correctly rounded primitives and the corrected oracle, everywhere;
  - exact-rational and range-edge fixes, in kimiya's working scope, only in the pure functions: `buoyancy`, `vortex`, the kernels, the EOS, and `bridge` when it ports. Each one is flagged to the maintainer;
  - inside the solver chains, Rust's sequence of operations is the definition.

---

## 6. Milestones

Every milestone ends with impetus's gates:
- `cyrius build`, then `cyrius test` and `test --bounds` **with no warnings** (the f64 type diagnostics appear only there);
- `cyrius bench`, recorded against the Rust numbers;
- `vet`, `lint` and `fmt --check` on the milestone's files;
- `distlib --all`, then `distlib --check`, then `deps --locked`;
- surface parity;
- the soak test (probe 1) once a solver steps;
- the CHANGELOG and `state.md`.

Large milestones go in parts, each part gated (CLAUDE.md's task sizing).

| Milestone | Modules | Key work | Golden grid (minimum) |
|---|---|---|---|
| **M1 Foundation** | error, constants, mathx, crmath, vec3, common, buoyancy, vortex | `PravashError` → `PV_ERR_*`. `FluidParticle` as a typed stride record. `FluidConfig` and `FluidMaterial` with validation. `ParticleArena`: the free list sorted by start, adjacent runs merged, stale handles after `compact`. `ParticleSoa`. buoyancy: Archimedes force, drag, Reynolds number, terminal velocity, `classify_flow`. vortex: point vortex, Lamb–Oseen, the Kolmogorov scale | every config field at and past its limit; seeded arena alloc/free/compact sequences; buoyancy and drag across the flow regimes, including NaN and ±∞ inputs; Re on both sides of each regime threshold; Lamb–Oseen at ν·t = 0 and r → 0 |
| **M2 Shallow water** | shallow | The first stateful solver, where the state loader and step rows are built. HLL fluxes from snapshot velocities, well-balancing, wet/dry handling, Manning friction, sediment, breaking, the implicit Peregrine dispersion filter, closed walls, CFL | lake at rest; 1D and 2D dam breaks; a linear wave (speed √(gh)); a wet/dry shoreline; a disturbance; class S at frames {1, 2, 10, 60, 300}; class R from the 44 tests |
| **M3 Grid** (parts a–d) | grid_solve, grid, mac_grid, k_epsilon, vof, phase_field | a: `FluidGrid`'s ghost cells per `BoundaryCondition`; semi-Lagrangian, MacCormack and BFECC advection (the backtrace limiter); diffusion; buoyancy; vorticity confinement; Smagorinsky; Gauss–Seidel and multigrid projection. b: DST, IDST and CG, transcribed from hisab 1.4.0 (ADR 0003). c: `MacGrid` (Neumann pressure) and `KEpsilon`. d: VOF (sub-cycled) and the phase field (upwind) | every pressure solver on the same systems: the solution class B/C, the iterations class A; ghost-cell rules per boundary condition; step corpus: a still tank, a buoyant plume, an inflow, a vortex; divergence after projection (class R); VOF mass over 300 steps |
| **M4 SPH foundations** | sph_kernels, neighbor, sph_fns | The kernels and their gradients and Laplacians, returning 0 for h ≤ 0. Tait EOS, including γ → 0. The neighbour hash (D3) and the lists. The free functions: `step`, `create_particle_block`, `split_particle` and merge, `sort_by_zorder`, `adaptive_dt`, the surface-tension colour field, the MLS gradient corrections (a 3×3 inverse, ADR 0003), heat, viscoelastic Oldroyd-B, delta-SPH, implicit viscosity, contact angle, non-Newtonian viscosity | kernels over r/h ∈ [0, 1.2], including 0, 1 and h ≤ 0; the EOS over γ including 0; neighbour lists for seeded clouds, including cell boundaries and large coordinates; sort ties; each free function fed the oracle's neighbour lists |
| **M5 SPH solvers** (parts per step variant) | sph, sph_pcisph, sph_dfsph, sph_multiphase | `SphSolver`: WCSPH with symplectic Euler or velocity Verlet (`reset_integrator`); PCISPH (at least one iteration, compression only); DFSPH (Bender & Koschier's order, κ/ρ); multiphase (one colour field per phase); phase change; secondary particles (foam, spray, bubbles) | step corpus: a hydrostatic tank, a dam break, a droplet, a two-phase layer, each run through every solver variant; class S at frames {1, 2, 10, 60, 300}; class R from the 79 tests and the SPH integration tests. Then the D3 switch to f64 keys, with the rows remapped |
| **M6 Coupling and MPM** | coupling, mpm | `FlipSolver`: the FLIP/APIC ratio, mass-weighted P2G, projection, `apic_c` auto-resize, `particle_level_set`. Rigid-body coupling: `BodyShape`, quadratic drag, added mass, nearby particles through the neighbour hash. `MpmSolver`: each `ConstitutiveModel` (fluid, elastic, Drucker–Prager in degrees), `validate`, the det(F) clamp, separating walls | FLIP: a sloshing tank and a falling block; coupling: a sphere and a box in a flow; MPM: an elastic block, a sand column and a fluid column; the refusals |
| **M7 Compute and logging** | compute, logging | `ComputeOp`; `PackedParticles` with f32 packing (probe 3); `KernelDerivatives`, including `autodiff_gradient` on hisab's tape through a function pointer (ADR 0002 #2); the logging init on sakshi | packing round trips; derivatives against the oracle (class C) and against finite differences (class R) |
| **M8 Release 2.0.0** | all | The Rust-test ledger: each of the 351 unit tests, 33 integration tests and 7 ignored doc tests is ported, folded into golden rows, or marked n/a or deferred. The full surface-parity diff. `docs/benchmarks-rust-v-cyrius.md` with the 23 Rust benchmarks in both columns. The README, `docs/guides/migration-2.0.md`, the CHANGELOG's Breaking section, the docs' code examples compiled and run. VERSION 2.0.0 | the full corpus green on x86_64 ELF, x86_64 PE and aarch64 |

Order: M0 → M1 → … → M8.
- Shallow water is self-contained (no hisab, no f32, no particles), so it is the cheapest place to build the
  state loader and the step rows.
- The grid comes next, because its hisab transcriptions are small function rows.
- SPH, the largest module at 2,170 lines, comes after probes 2–5 have settled.
- Coupling needs both particles and grids.
- If a consumer needs SPH first (soorat uses `sph` + `shallow`), M4 and M5 can move ahead of M3.

---

## 7. Rust → Cyrius mapping (pravash specifics)

### 7.1 Math functions

These are every std f64 method in `src/` outside tests, as of 1.3.0.

| Rust (calls) | Cyrius | Parity |
|---|---|---|
| `sqrt` (46), `abs` (34), `floor` (21), `ceil` (3) | builtins | exact on every target |
| `max` (128), `min` (37), `clamp` (22) | `_pv_max`, `_pv_min` and `_pv_clamp` with Rust's NaN-ignoring semantics (as impetus's `mathx.cyr`) | exact |
| `is_finite` (101), `is_nan` (3) | `x - x == 0`; `x != x` | exact |
| `exp` (5), `ln` (8), `log10` (1), `powf` (6), `sin` (2), `cos` (4), `sin_cos` (2), `atan2` (2), `hypot` (3) | correctly rounded (D2) | class C, exact against the corrected oracle |
| `cbrt` (2, Manning friction) | `ganita_f64_cbrt`; its comments describe an exact rounding decision, which step 8 measures | class C |
| `powi` (6) | explicit products in LLVM's square-and-multiply order (as tanmatra) | class B |
| `to_radians` (1) | `x * (PI / 180.0)` in Rust's order | class B |
| `as f32`, f32 `*` and `floor`, `as i32` (the neighbour hash and the packing) | probe 3 | class A |

**Typing traps** (from impetus §7.1, probed on 6.7.18):
- An operator is typed by its left operand, so bind builtin and ganita results to a `: f64` local first, and write `-x`, not `(0 - x)`.
- There is no exponent notation.
- A hex pattern in a `const` is an integer.
- The `F64_*` stdlib constants are untyped `var`s.
- Multi-return values are untyped words.

### 7.2 hisab 1.4.0 calls → hisab 3.4.1

| Rust (hisab 1.4.0) | hisab 3.4.1 | Plan |
|---|---|---|
| glam `DVec3` arithmetic | `hvec3_*`, which allocate on every operation | pravash-owned `src/vec3.cyr` on f64 locals (probe 2) until H1/H9 |
| glam `Vec3` (f32) and `SpatialHash` (f32 cells, a `HashMap` of cells) | `spatial_hash_*` (f64 cells; allocates on insert and query) | a pravash-owned neighbour grid (D3) |
| `num::dst`, `num::idst` | `num_dst`, `num_idst` | use them if probe 8 shows the same bits and no per-step leak; otherwise transcribe |
| `num::conjugate_gradient` (matrix-free linear CG, operator as a closure) | **none**: `opt_conjugate_gradient` is the nonlinear optimiser, and `solve_bicgstab_checked` / `solve_gmres_checked` / `solve_pgs_checked` take CSR matrices | transcribe 1.4.0's (ADR 0003); file P1 |
| `num::matrix_inverse` (n×n, called on a 3×3) | `m3_inverse_checked`, which decides singularity exactly (class D in impetus) | transcribe 1.4.0's for bit parity (ADR 0003); file P2 |
| `autodiff::{Tape, Var, reverse_gradient}` | `ad_tape_*` | use it (not per-step); probe 8 checks the bits |

### 7.3 Types and idioms

| Rust | Cyrius |
|---|---|
| `Vec<FluidParticle>`, `&mut [FluidParticle]` | a typed stride array plus a count, per probe 2. Storage comes from freelist or the solver, never `vec` (8-byte words only, and growth never frees) |
| `ParticleSoa` | column arrays |
| `Vec<f64>` grid fields and scratch | solver-owned f64 arrays, allocated once and reused; nothing allocates on the step path |
| payload enums (`BoundaryCondition`, `ConstitutiveModel`, `NonNewtonianViscosity`, `BodyShape`, …) | a kind enum (append-only) plus a payload struct with the union of the fields |
| `Result<T, PravashError>` | value-form `Result` plus a `PV_ERR_*` code. No `Result` through `callptr` |
| `Option<T>` | an out-pointer plus 1/0 |
| closures | function pointers through `fncall*` with a context word |
| `sort_unstable_by_key` | per probe 5 |
| `tracing::trace_span!`, `warn!` | sakshi spans and events behind a level |
| rayon | serial (D6) |

### 7.4 Names

- **Public structs** are `Pv<Name>` with `impl` blocks, so methods keep their Rust names (`s.step_pcisph(…)`).
  Constructors are `PvSphSolver_new` and the like.
- **Globals and enum members** carry `PV_`; private helpers carry `_pv_`.
- None of the three prefixes collides with the 6.7.18 stdlib or hisab 3.4.1 (checked 2026-11-11). Re-check at
  M1, because "last definition wins" across files is only a warning.

### 7.5 Memory

Probe 1 settles this:
- **Solver lifetime**: particles, grid fields, neighbour structures and persistent buffers live in storage the
  solver owns, freed by its `_free`.
- **Step lifetime**: candidate lists, the `accels` arrays (which 1.3 allocates each step) and solver scratch
  live in a step arena that is reset at the start of every step.
- **Never `alloc_reset`**: it would wipe the consumer's heap.

---

## 8. Gaps to file upstream (at M0, then as found)

Each gap gets an issue in the owning repo, with the probe or benchmark that shows it.

**hisab**
- **P1. A matrix-free linear CG**: an operator through a function pointer and context, a relative tolerance,
  the iteration count returned. 1.4.0's `num::conjugate_gradient` has no 3.x counterpart.
- **P2. A general n×n `matrix_inverse`**, or a documented statement that only `m3` / `m4` exist.
- **P3. A spatial-hash rebuild without per-insert allocation, and a neighbour query into a caller's buffer**:
  the per-step SPH pattern. It joins impetus's H7 and H9.
- **A fast DST**, if probe 8 shows `num_dst` is the direct O(N²) form. The roadmap already flags the Rust
  pressure solve's cost.

**Add pravash as a consumer, with its own numbers**, on:
- cyrius S1 (the scoped allocator);
- hisab H9 (allocator-scoped calls) and H1 (allocation-free vector math);
- the ganita correctly rounded tier (D2);
- impetus's S5 (`memcpy` byte loops: every particle-record copy);
- the stable-sort request (`sort_by_zorder`).

**cyrius**
- Add pravash's case to the `cyrius port` clobber issue: a project whose `.gitignore` ignores `Cargo.lock`
  ends up with a frozen oracle that has no lockfile.

---

## 9. After 2.0.0

The roadmap plans the work that follows 2.0.0:
- the bridges;
- the 1.3.0 audit's deferred physics (D5);
- the `parallel` path on `thread.cyr`;
- serde, if wanted (D4);
- the switch to hisab on the step path once S1 or H9 lands;
- retiring `rust-old/`.

---

## 10. Tracking

| Milestone | Status | Golden rows | Reference tests | Benches | Notes |
|---|---|---|---|---|---|
| M0 Initialize | ⏳ planned | — | — | — | waits on D1–D6, then step 0 (Rust oracle hooks, lockfile, tag 1.3.0) |
| M1 Foundation | — | — | — | — | |
| M2 Shallow water | — | — | — | — | |
| M3 Grid | — | — | — | — | |
| M4 SPH foundations | — | — | — | — | |
| M5 SPH solvers | — | — | — | — | |
| M6 Coupling and MPM | — | — | — | — | |
| M7 Compute and logging | — | — | — | — | |
| M8 Release 2.0.0 | — | — | — | — | |
