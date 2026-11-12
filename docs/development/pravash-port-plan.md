# pravash Rust → Cyrius port plan

> **Plan of record** for porting pravash to Cyrius. Rust **1.3.0** is the final Rust release and the
> **reference implementation** (the oracle). The Cyrius port ships as **2.0.0**. The method is tanmatra's
> ([its ADR 0006](../../../tanmatra/docs/adr/0006-cyrius-port-with-rust-oracle.md)), as adapted by kimiya
> ([its port plan](../../../kimiya/docs/development/kimiya-port-plan.md), in progress) and impetus
> ([its port plan](../../../impetus/docs/development/impetus-port-plan.md), shipped 2.0.0 on 2026-11-10).
> impetus is the closest model: it is also a stateful simulation built on hisab. Where this plan says
> "as impetus", "as kimiya" or "as tanmatra", follow that document. §1.1 is the language review that every
> scope decision rests on, and §2 lists what is different here.
>
> ▶ **Start on Cyrius 6.7.18 with hisab 3.4.1**, once the oracle hooks are in and 1.3.0 is tagged (M0 step 0).
>
> ⏳ **Status: M1 complete** (2026-11-11); M2 (shallow water) is next. M0 is complete (§4). M1 delivered, in four
> parts:
> - `src/mathx.cyr`, the first `[lib]` and the bundle;
> - the correctly rounded primitives (`src/crmath.cyr`, `src/crlog.cyr`), gated at 0 ulp against the MPFR-corrected
>   oracle (ADR 0004);
> - `src/error.cyr`, `src/vec3.cyr`, `src/column.cyr` and `src/common.cyr`;
> - `src/buoyancy.cyr`, `src/vortex.cyr`, `src/bridge.cyr` and `src/json.cyr`.
>
> The result: fourteen `[lib]` modules and 45,232 golden rows in seven modules, identical on three targets:
> - 42,697 rows ok and 72 accepted (ADR 4);
> - pending: the M4 cell key's 2,460 and three bayan-policy JSON rows.
>
> All 100 Rust unit tests of the M1 modules are ported, plus 11 integration tests, and surface parity is in CI.

| Thing | Value | Source |
|---|---|---|
| Rust source (oracle) | pravash **1.3.0**, committed (`f481c25`) but **not tagged**: the last tag is 1.2.0, and crates.io has 1.2.0. 6,882 code lines in `src/` (no tests, comments or blank lines), 16 modules, 193 `pub fn`s, 8 public enums, 22 serde-derived types (none in `sph.rs`) | `Cargo.toml`, `VERSION`, `git tag` |
| Features | default `sph`, `grid`, `shallow`; optional `buoyancy`, `vortex`, `coupling` (coupling + mpm), `compute` (needs `sph`), `parallel` (rayon), `ai`, `logging` (tracing-subscriber, tracing-chrome) | `Cargo.toml` |
| Rust health, 2026-11-11 on rustc 1.99.1 | `cargo test --all-features`: 351 unit and 33 integration tests pass; all 7 doc tests are `ignore`d. fmt, clippy, audit, deny and rustdoc were clean at the 1.3.0 audit (2026-11-09) and are re-run at M0 step 0 | local run |
| Correctness baseline | The 1.3.0 full-module audit. Every fix has a regression test; the projection, solver-sign, delta-SPH, DFSPH, implicit-viscosity, MPM-stress and wave-speed fixes are mutation-checked | `CHANGELOG.md` 1.3.0 |
| Cyrius toolchain | **6.7.18** (installed and current) | `~/.cyrius/current` |
| hisab | **3.4.1**, Cyrius, tag `3.4.1`, bundle `dist/hisab.cyr` | `/home/macro/Repos/hisab` |
| ganita | **1.2.15**, bundled in the 6.7.18 stdlib | `~/.cyrius/versions/6.7.18/lib/ganita.cyr` |
| Oracle's dependencies | hisab **1.4.0** (features `num`, `geo`, `autodiff`), which links glam **0.29.3**; serde, thiserror, tracing; rayon behind `parallel`. `Cargo.lock` is **untracked** (the library-crate rule) | `Cargo.lock` |
| Oracle's math | Rust std → host **glibc 2.44** for `ln` (8 calls), `powf` (6), `exp` (5), `cos` (4), `hypot` (3), `sin` (2), `sin_cos` (2), `cbrt` (2), `atan2` (2), `log10` (1). glam's `DVec3` (f64) is used everywhere; its `Vec3` (**f32**) is used only for the SPH and coupling neighbour hash | `grep` of `src/` |
| Published dependents (crates.io) | badal, pavan, khanij, soorat, kiran: all Rust, all `^1`. soorat is local and Rust; its optional `fluids` feature pulls pravash `sph` + `shallow` | crates.io reverse deps |
| Cyrius neighbours | None declares a pravash dependency yet. Bridge partners: **goonj is Cyrius 2.3.1**; ushma is Rust 1.4.0; pavan has no local repo. The `ai` configs point at **daimon (Cyrius 2.5.6)** and **hoosh (Cyrius 2.12.3)**. impetus 2.0 (Cyrius) has no pravash dependency. kiran and joshua have no local repo | `~/Repos` |
| Target | pravash **2.0.0**, Cyrius, carrying **every** module and feature (§3) | this document |

## 0. Decisions

### 0.1 Carried from the sibling ports

- **No deferral without a language check.** A Rust feature leaves the 2.0 scope only for a reason written
  down against §1.1: a capability Cyrius 6.7.18 lacks (named with its guide or stdlib section), a measured
  cost, or the lack of an oracle. The review found no language reason to drop any pravash feature. The only
  items outside 2.0 are the 1.3.0 audit's deferred physics (D5), and the reason for those is that they have
  no oracle.
- **The oracle is frozen.** Rust 1.3.0 moves to `rust-old/` and is never edited. A generator crate in
  `scripts/oracle/` drives it by path. If an oracle value is wrong, it is fixed in Rust as 1.3.x and the
  corpus is regenerated (as tanmatra).
- **No Rust 2.0** (as kimiya and impetus). API shape changes land in the Cyrius API only, accepted by
  ADR 0002 (§3.3).
- **Upstream requests never block the port** (the maintainer, 2026-11-11). Every gap found in hisab, ganita or
  the Cyrius stdlib is filed as a request in that project, with pravash's numbers. Meanwhile pravash ships a
  local stand-in. Each stand-in carries an `UPSTREAM(Ln)` note naming its request and the library call that
  will replace it, and has a row in the conversion ledger (§8). When the library ships, pravash converts the
  stand-in, whether before or after 2.0, and re-runs its parity rows. No milestone waits on an upstream release.
- **hisab-first, but off the step path until an allocator lands.** cyrius S1 (a scoped allocator) and hisab
  H9 (allocator-scoped calls) are both still **open** (checked 2026-11-11). Until one of them lands, every hisab
  call that returns a value leaks from the global bump heap. So per-step code is pravash-owned: the neighbour
  search, the pressure solves and the vector math in loops (as impetus [ADR 0012](../../../impetus/docs/adr/0012-impetus-owned-3d-math-stopgap.md)).
  These are stand-ins under the rule above. hisab and ganita serve the calls that are not per-step. Where neither
  has a routine the oracle used, the port transcribes Rust hisab 1.4.0's routine as a stand-in
  (kimiya [ADR 0003](../../../kimiya/docs/adr/0003-transcribe-missing-hisab-routines.md)).
- **Performance is measured, not gated** (as impetus). Every Rust benchmark gets a Cyrius twin, and both
  columns are recorded per milestone in `docs/benchmarks-rust-v-cyrius.md`. A slow row is a finding to
  file, with its numbers.
- **Parity is functional; accuracy beats glibc.** This is the maintainer's policy from kimiya
  (2026-11-09), also applied in impetus [ADR 0011](../../../impetus/docs/adr/0011-correctly-rounded-exp-sin-cos.md).
  The port uses correctly rounded primitives, never ones that copy glibc's last-ulp errors. Where glibc
  misrounds, the golden expectation is the oracle's formula evaluated on the correctly rounded primitive
  (`*.cr.tsv`), reported as accepted. §5 sets pravash's scope for this.

### 0.2 Decided 2026-11-11 (the maintainer took every recommendation)

| # | Question | Decision | Not chosen |
|---|---|---|---|
| D1 | ✅ **Tag and publish 1.3.0.** `release.yml` runs `cargo publish` on any version tag. Tagging therefore publishes 1.3.0 to crates.io, and its five Rust dependents on `^1` get the audit's behaviour changes | Land the `oracle` hooks (M0 step 0b) first. Then tag 1.3.0 on that commit and let it publish. That gives one frozen point, and Rust users get the audited physics | Tag without publishing. Or tag the current commit and add the hooks afterwards as unreleased tooling, as impetus did |
| D2 | ✅ **Where the correctly rounded math comes from.** pravash needs exp, ln, log10, pow, sin, cos, sin_cos, hypot and atan2. impetus has its own copies (`src/crmath.cyr`, ADR 0011), and so does kimiya (ln, log10, pow, exp, sin, cos, acos, atan2). The ganita issue asking for a correctly rounded tier is open, and pravash is the **third** crate to need these | pravash vendors them into `src/crmath.cyr` now, from impetus's crmath and kimiya's ln and log10 (ledger L1). It joins the open ganita request as the third consumer, which is the extraction cue in CLAUDE.md. The port converts to ganita's functions when they ship | — |
| D3 | ✅ **When to fix the f32 neighbour hash.** The oracle hashes f32 positions (hisab 1.4.0 `SpatialHash`), and the roadmap lists the resulting precision loss at large coordinates as a defect. Cyrius has a native `f32` type (§1.1), so a bit-faithful port costs nothing extra. The question is only when the defect gets fixed | Port the f32 keys bit-faithfully, so neighbour lists are class A and SPH steps class B/C through M5. Then switch to f64 cell keys before 2.0 as an ADR 0002 divergence, with the f32 rows remapped by name (kimiya's "noted rows mapped") | Port f32 keys and fix after 2.0; or use f64 keys from the start, which makes every SPH step row class D, because the summation order changes |
| D4 | ✅ **How strict JSON compatibility is.** The port carries serde on all 22 types through `#derive(Serialize)` / `#derive(Deserialize)` (§1.1). bayan prints f64 with Grisu2, which round-trips exactly but doesn't always give the same digits as serde_json's shortest form | Require decode compatibility: every document serde_json writes from the oracle decodes to the same bits, and every Cyrius document round-trips. Text identity is not required | Byte-identical text, by reusing impetus's shortest-digit printer (`fmtf.cyr`, ADR 0010) as the second instance |
| D5 | ✅ **The items the 1.3.0 audit deferred.** The roadmap lists Neumann and periodic pressure, SPH 2D kernel normalisation, an external-force input to `step()`, neighbour filtering by h, quadratic B-spline MPM and others. The language can express all of them; what they lack is an oracle | None of them in 2.0: they are new physics, checked by reference tests only, so they come after parity (as impetus). The f32 hash fix (D3) is the one exception | Pull the API-shape items (the external-force input, a dimension on the SPH config) into ADR 0002 |
| D6 | ✅ **The shape of `parallel`.** Rust selects rayon with a compile-time feature. Cyrius `[features]` only switch optional dependencies, and `#define` holds 16 names including the target's (§1.1) | Make it a runtime thread count on each solver: 1 (the default) is serial, and any count gives the same bits. `THREADS_CONCURRENT == 0` targets (agnos, cx) run the work inline | A `[lib.parallel]` profile that `#define`s a name, giving two bundles |

---

## 1. Porting models, references and the language

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
- **szal** (Cyrius), the precedent for porting Rust parallelism onto `thread.cyr`. Its `src/engine_parallel.cyr`
  uses worker threads that each write a per-worker result slot, which the joiner reads in spawn order.
- **tanmatra 2.0.1**: the parity classes, the port-parity rule (K_F = 2·S·calls + 2) and the origin of the Rust-test ledger.
- **hisab 3.4.1**, the dependency. Read `CLAUDE.md`, `docs/guides/migration-3.4.md` (the Result convention)
  and the doc comment of every function pravash calls. Its roadmap flags process-global tables that are not
  thread-safe (`K005`), which matters only if a worker thread ever calls hisab.
- **Language and toolchain** (`/home/macro/Repos/cyrius`): `docs/guides/cyrius-guide.md`,
  `docs/stdlib-reference.md`, `docs/development/ecosystem-migration.md` (the 6.7.x entries),
  `cyrius help manifest`, `BENCHMARKS.md`. §1.1 summarises what pravash needs from them.
- **Open upstream issues this port inherits:**
  - `cyrius/docs/development/issues/2026-11-08-cyrius-port-clobbers-project-docs-and-omits-bench.md`: still open, so M0 step 2 repairs what the scaffold overwrites;
  - `cyrius/…/2026-11-09-no-scoped-allocator-for-alloc.md` (S1);
  - `hisab/docs/development/issues/2026-11-09-allocator-scoped-calls.md` (H9);
  - `ganita/docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`;
  - cyrius's `memcpy` byte loop (`2026-11-02`, impetus S5).

### 1.1 Language review: Cyrius 6.7.18 against pravash's Rust surface

Read from the 6.7.18 guide and stdlib reference, and from the shipped `lib/` sources, on 2026-11-11. Each row
is a Rust construct pravash uses, what Cyrius provides for it, and the verdict. Every verdict is "port".

| Rust in pravash | Cyrius 6.7.18 | Where | Verdict |
|---|---|---|---|
| f64 arithmetic and std f64 methods | `f64` scalars (operators typed by the left operand). Builtins `f64_sqrt`, `f64_floor`, `f64_ceil`, `f64_trunc`, `f64_exp`, `f64_ln`, `f64_sin`, `f64_cos` and `f64_atan`, each within 1 ulp; the `_f64_*_polyfill` forms give the same bits on every target. ganita has pow, hypot, atan2 and cbrt | guide *Types*, *Math Builtins*; stdlib *math.cyr* | port, correctly rounded per D2 (§7.1) |
| f32 (the neighbour hash's `as f32` and f32 multiply; `PackedParticles`) | a **native `f32` scalar type**. `f32_from` rounds f64 → f32 to nearest, ties to even; `f32_to` widens. Writes into an `f32` round. f32 arithmetic exists, but an f32 left operand reads its right operand's low 32 bits. There are also `f32v4` / `f32v8` SIMD vectors | guide *Types* (6.7.6), *SIMD Vectors* | port bit-faithfully (probe 3 confirms) |
| float literals like `1e-9` | no exponent form, but decimal literals are correctly rounded | guide *Number Literals* | write constants positionally |
| structs, `impl`, methods, operator traits | structs with typed fields (`f64`, `f32`, narrow ints), `impl` blocks, traits, `dyn`, generic fns and structs, operator overloads | guide *Structs*, *Traits and impl blocks*, *Generic Functions* | port |
| data-carrying enums (`ConstitutiveModel`, `BodyShape`, `NonNewtonianViscosity`, …) | sum types. `enum …: stack` allocates nothing but carries 0–1 field per variant; a multi-field variant boxes, which allocates from the bump heap | guide *Sum Types & Tagged Unions* | a kind enum plus a payload struct, so nothing allocates per step |
| `Result<T, PravashError>`, `?` | value-form `Result`, `?` propagation, no try/catch | guide *`?` Propagation*, *Typed errors* | port, with `PV_ERR_*` codes |
| closures (`autodiff_gradient(f: impl Fn(&mut Tape, &[Var]) -> Var, …)`) | closure literals `\|a, b\| …`, capture by value, called through `fncallN` / `callptr`. A capturing closure takes at most 5 parameters (3 on Windows) and needs `lib/alloc.cyr` | guide *Closures*, *Function Pointers* | port as a closure |
| serde derives (22 types) | `#derive(Serialize)` / `#derive(Deserialize)` on structs and enums, through bayan. They generate `T_to_json`, `T_from_json_str` and `T_from_json`. Fields can be integers, `f64`, `f32`, `bool`, `Str`, `cstring`, `Vec<T>` of integers, `f64` or derived structs, and nested derived structs; enums serialize as variant-name strings (serde's unit-variant form). **Not covered:** `u64` fields, `Vec<Str>`, data-carrying variants, and glam's `[x, y, z]` array form for a vector. f64 is printed with Grisu2: round-trip exact, but not always shortest | guide *Derive Serialize on a struct / an enum*; `lib/bayan.cyr` (dtoa) | port: derive where it fits, hand codecs for `DVec3` fields and data-carrying enums (D4) |
| rayon (`parallel`: 5 `par_iter`, 2 `par_chunks`; sites in `grid.rs` and `sph.rs`) | `thread.cyr`: real OS threads on Linux, macOS and Windows (`thread_create` / `thread_join`, MPMC `chan_*`, mutexes), plus `atomic.cyr` and `thread_local.cyr`. `THREADS_CONCURRENT` is 0 on agnos and cx, where the bodies run inline. `alloc()` is thread-safe (one global lock, single-threaded fast path) | stdlib *Concurrency*; `lib/alloc.cyr` header | port as a worker pool over index ranges, each index writing its own slot, so any thread count gives the serial bits (D6, probe 6) |
| `tracing` spans and warnings (16 sites) | sakshi: `sakshi_span_enter` / `_exit`, levels, `sakshi_warn` / `sakshi_trace`, trace ids | `lib/sakshi.cyr` | port, keeping the warn-once-per-step rules |
| `logging` (tracing-subscriber, tracing-chrome init) | sakshi's level and sink setup | `lib/sakshi.cyr` | port the init. A Chrome-trace export is checked at M7, and filed if sakshi has none |
| `HashMap` inside hisab 1.4.0's `SpatialHash` | `hashmap.cyr` (its seed is random per process), `hashseed.cyr`; no ordered map | stdlib *hashmap*, *hashseed* | pravash's own code needs none: the neighbour grid is sorted arrays |
| `sort_unstable_by_key` (z-order) | `vec_sort_by`, which is introsort (unstable); there is no stable sort in `vec` or `slice` | `lib/vec.cyr` | probe 5 |
| `Vec<T>` growth, per-step `Vec` allocation | `vec` (8-byte words, never frees); `freelist` (individual free); arenas with reset and a growth policy; the allocator vtable (`alloc_via`) | guide *Allocators & Arenas*; stdlib *freelist* | solver-owned storage plus a step arena (§7.5) |
| Cargo features (`sph`, `grid`, `shallow`, …) | `[features]` switches optional dependencies only; `#ifdef` with at most 16 `#define` names, the target's included; `[lib.X]` bundle profiles; `[build] dce = true` dead-code elimination | guide *Manifest keys*, *Preprocessor* | one bundle, with dead-code elimination removing what a consumer doesn't call (probe 7 measures it) |
| `hisab::num::matrix_inverse` (n×n; hisab 1.4.0 does it as LU with partial pivoting plus a per-column `lu_solve`) | ganita `mat_inv` (LU with partial pivoting), `mat_lu` and `mat_lu_solve` | stdlib *linalg.cyr* (in ganita) | use ganita if probe 8 shows the same bits on the MLS moment matrices; otherwise transcribe |
| `hisab::num::conjugate_gradient` (matrix-free linear CG) | nothing in ganita or hisab 3.4.1: `opt_conjugate_gradient` is the nonlinear optimiser, and BiCGSTAB, GMRES and PGS take CSR matrices | hisab `dist/hisab.cyr` | a transcribed stand-in (ADR 0003, ledger L5), with hisab request P1 |
| `#[inline]`, `#[must_use]`, `#[non_exhaustive]` | `#inline` (never applied to a fn with defaults or a `defer`), `#must_use`, `#pure` / `#io` / `#alloc`; enums are append-only by convention | guide *Attributes* | translate |
| `Cow<'static, str>` error detail, `String` config fields | `Str`, `cstring` | guide *Strings* | errors carry codes only; config strings are `Str` |
| `Send + Sync` bounds | — | — | Rust-only mechanics, nothing to port |
| `ai`: endpoint configs, no I/O | struct fields of type `Str`, with the API key as `Str` or 0; `http.cyr` exists if a client is ever wanted | stdlib *http.cyr* | port the two configs, including the redacted debug print |

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
3. **f32 in two places:** the neighbour hash (D3) and `compute`'s GPU packing (`PackedParticles`, 41 f32 sites).
   Both map onto Cyrius's native `f32` (§1.1). The trap is the operator rule: an f32 left operand reads its
   right operand's low 32 bits, so every mixed expression goes through `f32_from` explicitly.
4. **Particle counts and threads.** SPH and FLIP run 10³–10⁵ particles a step. The neighbour lists keep all 27
   cells unfiltered, about 6× the true neighbours in 3D. Memory per step (probe 1), the particle layout
   (probe 2) and the worker pool (probe 6) decide most of the speed.
5. **An unstable sort with ties.** `sort_by_zorder` uses `sort_unstable_by_key` on a Morton code. Particles in
   one cell share a key, so their order after the sort depends on Rust's sort algorithm, and every later step's
   sums follow that order (probe 5).
6. **JSON on simulation state.** serde reaches `FluidGrid`, `ShallowWater`, `MacGrid`, `KEpsilon`, `VofField`
   and `PhaseField`, including their private fields, so consumers can save and restore them. The oracle writes
   serde_json documents for every derived type, and the port has to decode them exactly (D4).

---

## 3. Scope

### 3.1 In scope for 2.0.0: everything

| Milestone | Rust module | Code lines | `pub fn` | Tests | Cyrius module(s) | Library use |
|---|---|---|---|---|---|---|
| M1 | `error.rs` | 26 | — | 6 | `src/error.cyr` | — |
| M1 | *(public and private constants, presets)* | — | — | — | `src/constants.cyr` (generated) | — |
| M1 | *(std f64 methods)* | — | — | — | `src/mathx.cyr` (L2), `src/crmath.cyr` and `src/crlog.cyr` (D2, L1) | ganita where it is already exact; stand-ins otherwise |
| M1 | `common.rs` | 611 | 35 | 49 | `src/vec3.cyr` (the `DVec3` operations pravash uses), `src/common.cyr`, codecs in `src/json.cyr` | none on the step path: a pravash-owned glam transcription, as impetus's `math3.cyr` |
| M1 | `buoyancy.rs` | 105 | 5 | 16 | `src/buoyancy.cyr` | — |
| M1 | `vortex.rs` | 47 | 5 | 12 | `src/vortex.cyr` | — |
| M1 | `bridge.rs` (pavan, ushma, goonj) | 98 | 10 | 17 | `src/bridge.cyr` | — |
| M2 | `shallow.rs` | 692 | 12 | 44 | `src/shallow.cyr` | — |
| M3 | `grid.rs` | 1,464 | 21 | 49 | `src/grid_solve.cyr` (DST, IDST and CG, §7.2), `src/grid.cyr`, `src/mac_grid.cyr`, `src/k_epsilon.cyr` | `num_dst` / `num_idst` if probe 8 allows; otherwise transcribed |
| M3 | `vof.rs`, `phase_field.rs` | 368 | 19 | 19 | `src/vof.cyr`, `src/phase_field.cyr` | — |
| M3 | `parallel` in `grid.rs` (2 row loops, 1 `par_chunks_mut`) | — | — | — | `src/par.cyr` (the worker pool), used by `grid.cyr` | `thread.cyr` |
| M4 | `sph.rs`: kernels, equation of state, neighbour hash, free functions, configs | ≈ 900 | ≈ 30 | ≈ 35 | `src/sph_kernels.cyr`, `src/neighbor.cyr`, `src/sph_fns.cyr` | ganita `mat_inv` or a transcription for the MLS 3×3 inverse (§7.2) |
| M5 | `sph.rs`: `SphSolver` and its four steps, phase change, secondary particles | ≈ 1,270 | ≈ 19 | ≈ 44 | `src/sph.cyr`, `src/sph_pcisph.cyr`, `src/sph_dfsph.cyr`, `src/sph_multiphase.cyr` | — |
| M5 | `parallel` in `sph.rs` (`par_iter_mut`, 2 `into_par_iter` maps) | — | — | — | `src/par.cyr` | `thread.cyr` |
| M6 | `coupling.rs` | 590 | 19 | 33 | `src/coupling.cyr` | the M4 neighbour hash |
| M6 | `mpm.rs` | 424 | 6 | 13 | `src/mpm.cyr` | — |
| M7 | `compute.rs` | 201 | 9 | 11 | `src/compute.cyr` | hisab `ad_tape_*` for `autodiff_gradient` (not per-step) |
| M7 | `logging.rs` | 20 | 3 | 1 | `src/logging.cyr` | sakshi |
| M7 | `ai.rs` (`DaimonConfig`, `HooshConfig`) | 35 | — | 2 | `src/ai.cyr` | bayan (JSON) |

- The `sph.rs` split between M4 and M5 is an estimate; `scripts/surface_parity.py` fixes it at M4's start.
- Each milestone also ports the serde codecs of its own types (D4).
- The surface-parity check covers every `pub fn` and every `pub` method on the re-exported types.
- In scope: all 6,851 library lines (the 6,882 minus `lib.rs`).

### 3.2 Not carried

| Rust surface | Why |
|---|---|
| `Send + Sync` bounds, `#[inline]` hints as Rust applies them, `Cow` lifetimes | Rust-only mechanics with no Cyrius counterpart to port. `#inline`, `#must_use` and `Str` cover the intent (§1.1) |
| `thiserror` message text and `PravashError::Diverged { reason }`'s string | errors become `PV_ERR_*` append-only codes (as tanmatra); the code identifies the failure, and §3.3 #1 records the change |

### 3.3 API changes in the Cyrius version (accepted divergences, ADR 0002)

This is the seed list; it is finalised at M0 step 2. Each row says how the harness still compares the
function, so nothing leaves parity entirely.

| # | Rust 1.3.0 | Cyrius 2.0.0 | Golden comparison |
|---|---|---|---|
| 1 | `Result<T, PravashError>` carrying detail strings | value-form `Result` plus a `PV_ERR_*` code | class A on the variant |
| 2 | f32 neighbour-hash keys | f64 keys before 2.0 (D3) | f32 rows mapped by name |
| 3 | `sort_by_zorder`'s tie order (Rust's unstable sort) | a stable order on (key, input index), if probe 5 shows Rust's order can't be reproduced cheaply | class A on the key sequence and on the multiset; later steps start from the oracle's order |
| 4 | the compile-time `parallel` feature | a runtime thread count per solver, 1 by default (D6) | the same bits for every thread count, checked against the oracle's serial and parallel builds |
| 5 | features select modules at compile time | one `[lib]` bundle with dead-code elimination (or profiles, if probe 7 says so) | — |
| 6 | serde_json's shortest f64 digits | bayan's Grisu2 digits (D4) | every oracle document decodes to the same bits, and every Cyrius document round-trips |
| 7 | bit-identical only on one platform | **bit-identical across targets** for pravash-owned arithmetic (as impetus) | measured on x86_64 ELF, x86_64 PE and aarch64 |

---

## 4. M0: the prep, then initialize with `cyrius port`

0. **Freeze a green oracle (Rust, before the move).** Steps 0.1–0.4 were done 2026-11-11. Step 0.5, the commit
   and the tag, is the maintainer's (D1).
   1. ✅ **Re-run the P(-1) gates** on HEAD (`f481c25` + the plan commits), all green on rustc 1.99.1:
      - fmt;
      - clippy `-D warnings` with `--all-features`, with the defaults, with `--no-default-features`, and with each
        of the 10 features alone;
      - rustdoc `-D warnings`, `cargo audit`, `cargo deny`;
      - tests: 351 unit and 33 integration (the 7 doc tests are `ignore`d).
   2. ✅ **The `oracle` feature** (non-default, kept out of `full`, `#[doc(hidden)] pub mod oracle`, as impetus's
      `src/oracle.rs`). It is read-only. The dump types live in `src/oracle.rs`, and child modules
      (`src/<module>/oracle.rs`) fill them, because a child module can read its parent's private fields and call
      its private functions. So each of the eight audited sources gains only a four-line `mod oracle;`
      declaration. It provides:
      - `oracle_dump()` on each stateful type in §2 item 1, returning its **private** fields; public fields are
        read directly. That covers the scratch buffers whose contents can carry from one step to the next:
        Verlet's previous acceleration, the PCISPH and DFSPH buffers, FLIP's grid and APIC matrices, MPM's grid,
        multigrid's level buffers;
      - the SPH neighbour hash: the f32 positions, and the candidate lists in the order the solver sums them
        (CSR). `oracle::sph_cell_key(pos, h as f32)` gives each particle's cell key by `build_neighbors`'
        expression. hisab's `SpatialHash` is rebuilt at every neighbour build, so only its size is recorded;
      - `oracle::pressure_solve_{gauss_seidel,dst,cg,multigrid}`: `FluidGrid`'s private pressure solvers on a
        caller's system. They cover hisab 1.4.0's `dst`, `idst` and `conjugate_gradient` exactly as pravash calls
        them, and the multigrid hook also returns its level buffers;
      - `oracle::invert_3x3`, which is hisab 1.4.0's `matrix_inverse` as the MLS correction calls it.
        `reverse_gradient` is already public through `KernelDerivatives::autodiff_gradient`.

      **Iteration counts are not dumped** (a deviation from the first version of this step). Recording them would
      mean writes inside the audited solver loops, and the oracle stays read-only. A bit-identical solution
      implies the same iteration count. PCISPH's `sph::pcisph_iter` tracing span lets the generator count its
      iterations through a tracing subscriber, with no code change.

      `tests/oracle.rs` (14 tests) proves the hooks are read-only. Every solver variant runs 300 steps plain and
      300 steps fingerprinting its state after each step, and the final states must be identical:
      - variants: SPH ×5, the grid ×6 (each pressure solver, MacCormack, BFECC), shallow water ×3, FLIP ×2,
        MPM ×3, MacGrid, KEpsilon, VOF, the phase field, the arena;
      - a fingerprint is the `Debug` text, which is exact for every non-NaN value, and the harness refuses any
        state that holds a NaN.

      The file also checks:
      - that `sph_cell_key` names the cell hisab stores each particle in;
      - that the four pressure hooks are deterministic and agree with the DST solution;
      - the 3×3 inverse.

      Its harness was **mutation-checked**: one extra step in the dumped run fails all 10 solver checks. It passes
      with `parallel` (`--all-features`) and without it (CI's `oracle,coupling` matrix entry, `make test-oracle`).
      Gates on the change: fmt; clippy on every set in step 0.1, plus `oracle` alone and with `sph`, `grid`,
      `shallow`, `coupling` and `compute`; rustdoc; audit; deny; all tests.
   3. ✅ **Track the lockfile for the freeze.** `.gitignore` no longer ignores `Cargo.lock`, because the oracle's
      bits depend on hisab 1.4.0 and glam 0.29.3. It moves to `rust-old/Cargo.lock` at step 1, and ADR 0001 records
      the exception. The generator crate's own lock is seeded from it (as kimiya).
   4. ✅ **glibc 2.44** recorded (`ldd --version`, 2026-11-11). Every TSV header carries it, and `--check` refuses
      a corpus from another glibc.
   5. ⏳ **Commit, then tag 1.3.0** (D1). The tag runs `release.yml`, which publishes 1.3.0 to crates.io. Nothing
      in `src/` changes after this point.
1. ✅ **Scaffold. Done 2026-11-11.** Run `cargo clean` first (`target/` was 4.1 GB by then). Then run `cyrius port --dry-run .`, then `cyrius port .`.
   The result matched the expectation below. `rust-old/` stays tracked, because the scaffold appends `/rust-old/target/`
   (not `/rust-old/` as it reports). The frozen crate still passes there: 351 unit, 33 integration and 14 oracle tests.
   The scaffold overwrote CLAUDE.md, the roadmap and both workflows, wrote `version = "0.1.0"`, and left `bench` out of
   `[test] stdlib`: the open clobber issue's items 1–3, confirmed again on 6.7.18.
   - Expect these to move into `rust-old/`: `Cargo.toml`, `Cargo.lock`, `src/`, `tests/`, `benches/`, `examples/`,
     `Makefile`, `deny.toml`, `codecov.yml`, `rust-toolchain.toml`. `--dry-run` doesn't list them by name.
   - Expect these to be written: `src/main.cyr`, `cyrius.cyml`, `tests/pravash.{tcyr,bcyr,fcyr}`,
     `docs/{adr,architecture,guides,examples}/` and `docs/development/state.md`.
2. ✅ **Repair what the scaffold overwrites. Done 2026-11-11.**
   - CLAUDE.md merged, with the principles translated against §1.1 and the §0.1 rules.
   - The roadmap restored from git, and `state.md` rewritten.
   - The Cyrius CI gained `test --bounds`, `deps --locked`, `audit` and a frozen-oracle job. The Rust workflows are in `rust-old/.github/workflows/`.
   - The Rust scripts, `docs/architecture/overview.md` and `docs/development/migration.md` are in `rust-old/`, and so is `rust-old/README.md` (the manifest names it).
   - ADRs 0001–0003 are written.
   - The six scaffold stubs are documented.
   - CONTRIBUTING and SECURITY are rewritten. The README gets a port banner now; its full rewrite is M8's, as in impetus.

   The clobber issue is still open on 6.7.18; impetus's §4 step 2 is the checklist.
   - **CLAUDE.md**: merge `git show HEAD:CLAUDE.md` with the scaffold's Cyrius rules, keeping the identity, the
     process and the DO NOT list, including "NEVER use `gh`". Translate the principles against §1.1:
     - `#[non_exhaustive]` → append-only enum values;
     - `#[must_use]` → `#must_use` / `#pure`;
     - no `unwrap`/`panic!` → no `+?`, `-?`, `*?`, `assert_fatal`, `result_unwrap` or `panic` in `src/`;
     - `#[inline]` → `#inline` on hot paths;
     - "Vec arena over HashMap" → typed stride arenas and sorted arrays, never `hashmap` (its seed is random per process);
     - "`write!` over `format!`" and "Cow over clone" → `str_builder` writes, borrow by pointer, copy only into owned storage;
     - "tracing on all operations" → sakshi spans, behind a level;
     - "feature-gate optional deps" → `[features]` selects optional dependencies, and dead-code elimination drops unused code.

     Also add a rule: no feature is deferred without a check against the language (§0.1). Replace the consumer
     list with the real one (header table) and the MSRV with the Cyrius pin.
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
     - 0001, the port with a Rust oracle, including the lockfile exception and the §0.1 deferral rule;
     - 0002, the §3.3 API divergences;
     - 0003, local stand-ins for library gaps and how they are converted (§8). It covers the hisab 1.4.0
       transcriptions, the ganita substitutes and `crmath` (D2).
   - **Checks**: `cyrius.cyml` reads `version = "${file:VERSION}"`, and `bench` is in `[test] stdlib`. Document the
     scaffold's stub functions so `cyrius audit` passes. Rewrite CONTRIBUTING, SECURITY and the README for Cyrius.
3. ✅ **`cyrius.cyml`. Done 2026-11-11.** It is written as below, with one change: hisab is `scope = "test"` until a library
   module calls it. As a production dependency it would reach every consumer's compile (impetus ADR 0007, second amendment).
   `cyrius deps` locked 52 files, plus 2 commit pins (hisab 3.4.1 `d6703c5`, and sakshi 2.5.10 through hisab). These gates pass,
   with no warnings: `build`, `test`, `test --bounds`, `bench`, `fuzz`, `lint`, `fmt --check`, `vet`, `deps --locked`, `audit`.
   There is no `[lib]` until M1, because `modules = []` is invalid. Confirm the keys with `cyrius help manifest`.
   ```toml
   [package]
   name = "pravash"
   version = "${file:VERSION}"
   license = "GPL-3.0-only"
   language = "cyrius"
   cyrius = "6.7.18"

   [build]
   dce = true

   [deps]
   stdlib = ["syscalls", "string", "alloc", "str", "fmt", "io", "vec", "math", "ganita",
             "tagged", "result", "freelist", "fnptr", "atomic", "thread", "bayan"]
             # sakshi arrives via hisab

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
   Module order: error → constants → crmath → crlog → mathx → vec3 → common → json → buoyancy → vortex → bridge → par →
   shallow → grid_solve → grid → mac_grid → k_epsilon → vof → phase_field → sph_kernels → neighbor → sph_fns → sph →
   sph_pcisph → sph_dfsph → sph_multiphase → coupling → mpm → compute → logging → ai.
4. ✅ **Toolchain and math check. Done 2026-11-11** ([`port-parity.md`](port-parity.md) §1, programs in `scripts/probes/`).
   `p0_semantics.cyr` and its Rust twin `p0_semantics.rs` compare the bits of every operation, and the probe compiles
   warning-free. Exact on every case:
   - the f32 SPH cell key, including saturation and NaN;
   - `floor`, `ceil`, `sqrt` and `abs`;
   - ln, sin and cos, hypot and cbrt on the probed points;
   - `to_radians`, and `powi` (2 and 8) as square-and-multiply.

   Needs a helper (ledger L2):
   - `f64_min` / `f64_max` return NaN for a NaN *second* operand (165 call sites);
   - `f64_to` doesn't saturate as Rust's `as i32`, `as u32` and `as usize` do (3, 2 and 19 sites):
     negatives, huge values and NaN differ.

   Off by 1–2 ulp: there is no `log10` (ln/ln 10 is 1–2 ulp off), and ganita's `pow` and `atan2` and the exp
   polyfill are each 1 ulp off somewhere. All of these are covered by L1.
   Confirm the counterpart of every Rust f64 method in §7.1 on 6.7.18. impetus
   probed the following, and its results carry over:
   - `f64_min` / `f64_max` don't follow Rust's NaN rules;
   - the stdlib has no public `is_nan`, `is_finite`, `copysign`, `sin_cos`, `EPSILON`, `INFINITY` or `NAN`;
   - `f64_exp` and the native `f64_atan` give different bits on x86_64 and aarch64.

   pravash adds:
   - `ganita_f64_cbrt`;
   - `f64::MAX` (`cfl_dt`'s value when there is no flow);
   - saturating `as i32` from an f32 floor;
   - `saturating_sub` on `usize`.
5. ✅ **Golden-value generator. Done 2026-11-11** (`scripts/oracle/`: `README.md`, `validate.py`).
   - **Output:** `constants.tsv` 29 rows (proven complete against `rust-old/src`) and `math.tsv` 33,239 rows. The
     math rows cover every std f64 method pravash calls, plus the saturating casts, `as f32`, and the SPH cell key
     through the oracle hook.
   - **Determinism:** two runs and the `parallel` build's `--check` give the same bytes. The `parallel` build
     refuses to write.
   - **Independent check:** `validate.py` recomputes all 33,268 rows: class A/B in Python, glibc's class C through
     ctypes, and `cbrt` proven correctly rounded.
   - **Tamper checks:** a one-bit change fails both `validate.py` and `--check`, and a header naming another glibc
     is refused by name.
   - **Finding** (port-parity §3): the oracle's `cbrt` is compiler_builtins', not glibc's, and correctly rounded.
   - **Not yet:** the JSON table and the step-row format arrive with the first types and solvers (M1, M2), as
     impetus added its modules.

   As built, copied from impetus's crate:
   - TSV format 1 with f64 values as bit patterns, deterministic output, a `--check` mode, and the glibc header;
   - `validate.py` recomputes the class A/B rows in Python;
   - two builds: `--features full,oracle` without `parallel`, and the same with `parallel`, whose rows must match the
     serial build's (probe 6);
   - the step format as impetus: each row names a scenario and a frame. Its inputs are the dump at frame k plus the
     call's arguments; its outputs are the dump at k + 1 plus the iteration counts;
   - a JSON table: serde_json documents for each of the 22 derived types, from seeded values, including NaN-free
     edge values and every enum variant.
6. ✅ **Table and constant generator. Done 2026-11-11** (`scripts/oracle/src/t_constants.rs`, `cyr.rs`).
   - **`src/constants.cyr`** (94 lines) holds the 14 public constants (the 5 `FluidMaterial` presets as 20 field
     consts, 5 drag and 4 added-mass coefficients) and the 6 Rust std float constants the code names: `f64::MAX`,
     `INFINITY`, `NEG_INFINITY`, `NAN`, `consts::PI` and `f32::EPSILON`.
   - **Proofs:** both lists are proven complete against `rust-old/src`, and `constants.tsv` grew to 35 rows.
   - **Form:** an f64 is a `const` with Rust's `Display` digits (positional, round-trip). ±∞, NaN and `f64::MAX`
     are typed `f64` globals holding their bits, and `f32::EPSILON` is an `f32` global with an exact decimal.
   - **Check:** `scripts/oracle/probe_tables.py` compiles every value and compares its bits with the golden rows:
     35 of 35. A 2-ulp change to one literal fails both it and `--check`.
   - **Not yet:** the **private** constants (`MIN_DENSITY`, `MIN_SINGULAR_VALUE`, the solver limits, …) are
     extracted and proven at the milestone whose module uses them, as impetus did. Their rows come with that
     module's golden grid.

   As planned, producing `src/constants.cyr` through the oracle's API, never
   by retyping. It covers:
   - the public constants: the `FluidMaterial` presets (`WATER`, `OIL`, `HONEY`, `AIR`, `LAVA`) and the drag
     coefficients in `buoyancy` and `coupling`;
   - the private ones: `MIN_DENSITY`, `MIN_SINGULAR_VALUE`, the CG and multigrid limits, the DST eigenvalue
     threshold (1e-12), the Riemann ε (1e-8), and so on.

   The generator proves the list is every `const` in `rust-old/src`. Write decimal literals out positionally (the
   lexer has no exponent notation), and never put a hex pattern in a `const`.
7. ✅ **Parity harness. Done 2026-11-11** ([`port-parity.md`](port-parity.md) §2.1).
   - **Files:** `tests/parity.tcyr` (pravash's hooks and dispatchers), with `tests/parity_core.cyr` and
     `tests/ulp.cyr` copied unchanged from impetus. The constants rows go through `tests/constants_rows.cyr`, a
     lookup the generator writes from the same list as `src/constants.cyr`.
   - **First run:** 2 modules, 0 failing; 5,149 ok, 0 failed, 18,493 measured, 9,632 pending. It is warning-free,
     and the same under `--bounds`.
   - **cbrt:** ganita's cbrt equals the oracle's on every row, so it needs no stand-in.
   - **Mutation check:** a golden bit, a 1-ulp constant, and a mis-wired dispatcher each fail their row.
   - **Not yet:** the state loaders, the step comparator and the JSON comparator arrive with the first solver and
     the first derived types (M2, M1). The divergence table fills as ADR 0002's rows land.

   As planned, from impetus: `tests/parity.tcyr`, `tests/parity_core.cyr` and `tests/ulp.cyr`, plus:
   - a state loader and a step comparator per solver, with per-field classes;
   - a JSON comparator (decode the oracle's document, compare bits, re-encode, decode again);
   - the divergence table, which names each ADR 0002 row and reports it by name.
8. ✅ **Baseline measurements. Done 2026-11-11** ([`port-parity.md`](port-parity.md) §2.2).
   - **Cross-target:** the replay runs warning-free on x86_64 ELF, x86_64 PE (wine) and aarch64 (qemu). Every
     dispatched function's digest is **identical on all three**. The native x87 shadows differ on aarch64, and
     their aarch64 digests equal the polyfills', which confirms the polyfill choice.
   - **Bounds:** K = 2 for sin, cos, sin_cos, atan2, exp, ln and hypot (measured max 1, hypot 2); K = 0 for cbrt
     (correctly rounded on both sides). powf stays measured until its M1 counterpart, the correctly rounded pow:
     its only outlier is ganita's open `pow(-1, huge)` defect.
   - **Gated result:** 0 failing; 21,852 ok, 1,790 measured, 9,632 pending, on all three targets. Both bounds are
     mutation-checked at their edges.
   - **Moved to M1:** the corrected oracle (`*.cr.tsv`, MPFR) comes with the correctly rounded stand-ins (L1),
     which it gates at K = 0, as impetus's ADR 0011. Done in M1 part 2 (ADR 0004, port-parity §6): every bound
     above is now K = 0, powf and log10 included.

   As planned, recorded in `docs/development/port-parity.md`:
   - the math sweep through every Cyrius counterpart against glibc, on x86_64 ELF, x86_64 PE (wine) and aarch64 (qemu);
   - K set by the port-parity rule;
   - the corrected oracle (`*.cr.tsv`) for the correctly rounded primitives.
9. ✅ **Probes. Done 2026-11-11** ([`port-parity.md`](port-parity.md) §4; `scripts/probes/`). The answers:
   1. **Memory.** hisab's hash rebuild leaks 4.1 MB per 1,000-particle step; `num_dst` 10–20 KB per call;
      ganita's 3×3 `mat_inv` 248 B; `hvec3_add` 24 B. The autodiff tape takes 0 B once reset. Freelist and the step
      arena are flat. L3, L4, L6 and L7 are needed.
   2. **Layout.** All four forms give the same bits. AoS with `f64` locals is 1.83 ms a pass; SoA 1.1×; struct
      values 2.5×; `f64v4` 3.5×. Particles are `PvParticle` stride records, and hot loops use `f64` locals.
   3. **f32 keys.** 10⁶ of 10⁶ match Rust on x86_64 ELF, PE and aarch64.
   4. **Neighbour order.** It is (cell offset, insertion): every dumped list equals a no-hash-map build. At
      |x| ≈ 10⁶, 89% of the probe points land in another cell: the defect D3 fixes.
   5. **Sort ties.** Rust's ipnsort differs from the stable order at 75–94% of positions on tie-heavy inputs,
      and not at all on sorted ones. ADR 0002 #3's stable (key, index) order applies, and L9 is definite.
   6. **Parallel.** The oracle's serial and parallel builds give the same digests on 14 scenarios. A Cyrius
      persistent pool gives the serial bits at T = 1–8 and allocates nothing, at 2× for T = 2 and 3–3.9× for T = 8.
   7. **Bundle.** Dead-code elimination takes a full-hisab consumer from 3.06 MB to 677 KB. One `[lib]`, no
      profiles.
   8. **Libraries vs 1.4.0.** hisab's DST differs from 1.4.0's: up to 67,000 ulp on its FFT path, so L6
      transcribes 1.4.0's. ganita's `mat_inv` gives the oracle's bits on all 300 matrices, so L7 is memory only. The
      autodiff tape is within 3 ulp, so it stays on hisab as class D.

   As planned, each one is a small program, answered with numbers before M1 writes any code. The programs go in
   `scripts/probes/` and the results in `port-parity.md`.
   1. **Memory model.** Solver-owned storage (freelist or a solver arena) plus a per-step scratch arena, with hisab
      off the step path. Gate: 10⁴ steps of a 10⁴-particle SPH dam break, and 10⁴ steps of a 128² `FluidGrid` on
      CG, each reach a flat heap high-water mark after warm-up.
   2. **Particle layout and vector representation.** Time the SPH density and force loops two ways: over a typed
      stride array of `PvParticle` records (array-of-structs, like Rust's `FluidParticle`) and over column arrays
      (like `ParticleSoa`). Run each with f64 locals and with struct values (impetus measured struct values at
      2.5× slower than f64 locals), and with `f64v4` lanes where the loop allows it.
   3. **f32 keys.** Reproduce hisab 1.4.0's `SpatialHash` cell key (`floor(x as f32 · inv_cell) as i32`) through
      `f32_from`, f32 `*`, `f32_to`, `f64_floor` and a saturating convert. Check 10⁶ seeded positions, including
      positions on cell boundaries and |x| up to 10⁷, plus the bits of `PackedParticles`' f32 packing. Also pin
      the operator trap: an f32 left operand with an f64 right operand must warn.
   4. **The oracle's neighbour candidate order.** Confirm that hisab 1.4.0's query visits cells in a fixed offset
      order, and each cell in insertion order, so that no `HashMap` iteration order reaches a sum.
   5. **Z-order sort ties.** Can the port reproduce the order Rust's `sort_unstable_by_key` gives on 1.99, or does
      it sort on (key, index) instead (ADR 0002 #3)?
   6. **Parallelism.**
      - Rust's `parallel` build's dumps equal the serial build's, bit for bit, on every scenario in the step corpus
        (the ordered `par_iter` collects, and any reductions).
      - A Cyrius worker pool in `src/par.cyr` (persistent `thread_create` workers fed by `chan_*`, each index range
        writing its own output slots) gives the serial bits at 1, 2, 4 and 8 threads.
      - Measure its speedup at 10³, 10⁴ and 10⁵ particles, and its per-step cost against spawning threads for
        each step.
      - Confirm that workers allocate nothing: step scratch is per worker, and `alloc()`'s global lock stays off
        the loop.
   7. **Bundle shape.** Build one `[lib]` with `dce = true` for a consumer that uses only `shallow` (soorat's case),
      and compare its size and compile time with a `[lib.shallow]` profile.
   8. **hisab and ganita against 1.4.0:** `num_dst` / `num_idst` against 1.4.0's `dst` / `idst`; ganita `mat_inv`
      against 1.4.0's `matrix_inverse` on the MLS moment matrices; the autodiff tape. Record bits and allocations.
      The answer decides between "use" and "transcribe" in §7.2.
   9. **JSON.** `#derive(Serialize)` on each derived type's Cyrius struct, fed the oracle's serde_json documents:
      decoded bits, round trips, and how often bayan's Grisu2 digits differ from serde_json's. Write hand codecs
      for `DVec3` fields (serde writes `[x, y, z]`) and for the data-carrying enums.
10. ✅ **File the upstream requests. Done 2026-11-11.** Nine new requests, each with a self-proving repro
    verified from its repo's root where the gap can be shown:
    - **hisab:** the spatial-hash query leak, a matrix-free linear CG, the per-call DST allocation;
    - **ganita:** `mat_inv_into`;
    - **cyrius:** min / max NaN plus the Rust f64 surface, a stable sort, the `#derive` array and tagged shapes,
      and the struct-value and `f64v4` costs;
    - **sakshi:** a chrome://tracing export.

    pravash's numbers went into four open issues: ganita's correctly rounded tier (as its third consumer, adding
    ln and log10, and reporting that ganita's cbrt is already exact), hisab H9, cyrius S1, and the `cyrius port`
    clobber issue (three new findings). All of these are new or untracked issue files; nothing else in those repos
    was touched. The open memcpy issue is being edited elsewhere, so L12 stays conditional on M1's numbers. The
    ledger (§8) is seeded: 11 open rows and 1 conditional, each naming its request file. `scripts/upstream_ledger.py
    --check` passes, is mutation-checked three ways, and runs in CI. The generator now writes the `UPSTREAM(L2)`
    note above `src/constants.cyr`'s std constants, which are a stand-in themselves.

    As planned, file the upstream requests (§8), with the probes' numbers:
    - new requests where none is open;
    - pravash's numbers added to the open ones.

    Then write `scripts/upstream_ledger.py` and seed the ledger: one row per stand-in, the conditional rows
    settled by probes 5 and 8. No later step waits on any request.

**Gate:**
- `cyrius build` and `cyrius bench` pass;
- `cyrius test` runs the harness (0 modules) with no warnings;
- `cyrius lint` and `fmt --check` are clean;
- `scripts/upstream_ledger.py --check` passes;
- committed: the oracle hooks (now in `rust-old/`), the generator, `validate.py`, `tests/ulp.cyr`, ADRs 0001–0003,
  the probe results, the ledger, and links to the upstream requests.

---

## 5. Parity policy

tanmatra's classes apply, with impetus's S and R classes for simulations:

| Class | Tolerance | pravash examples |
|---|---|---|
| **A — exact** | bit-identical / equal integers | error variants, config validation verdicts, iteration counts, neighbour candidate lists (D3's f32 keys), z-order key sequences, arena handles and free-list order, `classify_flow` regimes, wet/dry masks, FLIP cell marks, constants and presets, decoded JSON, results at every thread count |
| **B — arithmetic** | ≤ 2 ulp, expected 0 | the polynomial kernels (poly6, spiky, viscosity, cubic spline, Wendland) and their gradients, buoyancy and drag, Gauss–Seidel sweeps, advection interpolation, the shallow-water HLL fluxes (`sqrt` is exact), MPM transfers, `powi` sites |
| **C — transcendental** | ulp ≤ K by the port-parity rule; **0** on correctly rounded primitives, against the corrected oracle | Tait EOS and non-Newtonian `powf`, the shallow-water damping `powf` and Manning `cbrt`, Lamb–Oseen `exp`, the Kolmogorov scale (`ln`, `exp`), the DST eigenvalues (`cos`), MPM's 2×2 SVD (`hypot`, `atan2`, `sin_cos`), its Hencky strain (`ln`, `exp`), Drucker–Prager's friction angle (`to_radians`, `sin`), the contact angle, the bridge's `powf` and `log10` |
| **D — reordered or library-backed** | relative ≤ 1e-12, or a per-function bound set before its first comparison | sums reordered by an accepted divergence (D3's f64 keys after the switch, ADR 0002 #3); any hisab 3.x or ganita path that probe 8 shows isn't bit-identical |
| **S — single step** | per field, the class of the path that produced it | one step from an oracle dump. Iteration counts, masks and neighbour lists are class A |
| **R — reference** | the physical tolerance stated in the test | the 33 integration tests, the 1.3.0 regression tests and the unit tests that assert physics: hydrostatic rest, wave speed √(gh), divergence after projection, VOF and shallow-water mass, Lamb–Oseen decay, terminal velocity, dam-break front position |

Rules:
- **Never widen a class to make a row pass** (as tanmatra). A failure is one of three things: a port defect; a
  Cyrius, hisab or ganita defect, which is filed upstream; or an oracle defect, which is fixed in Rust 1.3.x and
  regenerated. The one exception is a divergence accepted by an ADR, which the harness still reports by name.
- **Bit-level trajectories only where the scene isn't chaotic**: a hydrostatic tank, a lake at rest, a linear
  shallow-water wave, a single droplet, Lamb–Oseen decay. They are compared over the whole horizon. Dam breaks,
  splashes and plumes get class S rows at frames {1, 2, 10, 60, 300}, plus class R.
- **Thread count is never a parity variable.** Every step row runs at 1 thread and at the probe-6 counts, and all
  of them must give the same bits.
- **Branch points, tested on both sides:**
  - the solver tolerances (CG's 1e-6·‖b‖ and its 1e-14 floor, PCISPH's compression-only convergence with at least one iteration, the implicit-viscosity and FLIP checks);
  - the multigrid depth cap (10 levels);
  - the DST eigenvalue threshold (1e-12);
  - `MIN_DENSITY`;
  - the surface-tension gate |∇c| > 0.1/h;
  - the non-Newtonian saturation cap;
  - shallow water's `dry_thr`, its Riemann ε (1e-8) and its `breaking_threshold`;
  - the SPH hash's cell-size reuse: `build_neighbors` keeps the old hash when |h_f32 − cell_size| ≤ `f32::EPSILON`
    (absolute, 1.19e-7), but computes its probe keys from the new h_f32. So within that band the hash's cells and
    the keys use different sizes (found in step 0.2; the port reproduces it, and it is a candidate for D3's switch);
  - the Reynolds thresholds in `classify_flow`;
  - MPM's det(F) clamp and the Drucker–Prager yield;
  - the phase-field |φ| > 1.5 warning;
  - VOF's Courant ≤ 0.5 sub-cycling.
- **The accuracy policy's scope for pravash** (§0.1):
  - correctly rounded primitives and the corrected oracle, everywhere;
  - exact-rational and range-edge fixes, in kimiya's working scope, only in the pure functions: `buoyancy`, `vortex`, `bridge`, the kernels and the EOS. Each one is flagged to the maintainer;
  - inside the solver chains, Rust's sequence of operations is the definition.

---

## 6. Milestones

Every milestone ends with impetus's gates:
- `cyrius build`, then `cyrius test` and `test --bounds` **with no warnings** (the f64 and f32 type diagnostics appear only there);
- `cyrius bench`, recorded against the Rust numbers;
- `vet`, `lint` and `fmt --check` on the milestone's files;
- `distlib --all`, then `distlib --check`, then `deps --locked`;
- surface parity;
- the soak test (probe 1) once a solver steps;
- the JSON rows of the milestone's types;
- `scripts/upstream_ledger.py --check`. Requests for the gaps the milestone found are filed, and any stand-in whose
  library has shipped since the last milestone is converted (§8);
- the CHANGELOG and `state.md`.

Large milestones go in parts, each part gated (CLAUDE.md's task sizing).

| Milestone | Modules | Key work | Golden grid (minimum) |
|---|---|---|---|
| **M1 Foundation** | error, constants, mathx, crmath, vec3, common, json, buoyancy, vortex, bridge | `PravashError` → `PV_ERR_*`. `FluidParticle` as a typed stride record. `FluidConfig` and `FluidMaterial` with validation. `ParticleArena`: the free list sorted by start, adjacent runs merged, stale handles after `compact`. `ParticleSoa`. buoyancy: Archimedes force, drag, Reynolds number, terminal velocity, `classify_flow`. vortex: point vortex, Lamb–Oseen, the Kolmogorov scale. bridge: the 10 pavan / ushma / goonj conversions. JSON for the common and buoyancy types, with the `DVec3` hand codec | every config field at and past its limit; seeded arena alloc/free/compact sequences; buoyancy and drag across the flow regimes, including NaN and ±∞ inputs; Re on both sides of each regime threshold; Lamb–Oseen at ν·t = 0 and r → 0; the bridge's inputs over their physical ranges and edges; the JSON table |
| **M2 Shallow water** | shallow | The first stateful solver, where the state loader and step rows are built. HLL fluxes from snapshot velocities, well-balancing, wet/dry handling, Manning friction, sediment, breaking, the implicit Peregrine dispersion filter, closed walls, CFL. `ShallowWater` JSON | lake at rest; 1D and 2D dam breaks; a linear wave (speed √(gh)); a wet/dry shoreline; a disturbance; class S at frames {1, 2, 10, 60, 300}; class R from the 44 tests |
| **M3 Grid** (parts a–e) | grid_solve, grid, mac_grid, k_epsilon, vof, phase_field, par | a: `FluidGrid`'s ghost cells per `BoundaryCondition`; semi-Lagrangian, MacCormack and BFECC advection (the backtrace limiter); diffusion; buoyancy; vorticity confinement; Smagorinsky; Gauss–Seidel and multigrid projection. b: DST, IDST and CG (ADR 0003). c: `MacGrid` (Neumann pressure) and `KEpsilon`. d: VOF (sub-cycled) and the phase field (upwind). e: `src/par.cyr` and the grid's three parallel loops. JSON for every grid type, private fields included | every pressure solver on the same systems: the solution class B/C, the iterations class A; ghost-cell rules per boundary condition; step corpus: a still tank, a buoyant plume, an inflow, a vortex; divergence after projection (class R); VOF mass over 300 steps; every row at each thread count |
| **M4 SPH foundations** | sph_kernels, neighbor, sph_fns | The kernels and their gradients and Laplacians, returning 0 for h ≤ 0. Tait EOS, including γ → 0. The neighbour hash with bit-faithful f32 keys (D3) and the lists. The free functions: `step`, `create_particle_block`, `split_particle` and merge, `sort_by_zorder`, `adaptive_dt`, the surface-tension colour field, the MLS gradient corrections (a 3×3 inverse, §7.2), heat, viscoelastic Oldroyd-B, delta-SPH, implicit viscosity, contact angle, non-Newtonian viscosity | kernels over r/h ∈ [0, 1.2], including 0, 1 and h ≤ 0; the EOS over γ including 0; neighbour lists for seeded clouds, including cell boundaries and large coordinates; sort ties; each free function fed the oracle's neighbour lists |
| **M5 SPH solvers** (parts per step variant, then parallel) | sph, sph_pcisph, sph_dfsph, sph_multiphase | `SphSolver`: WCSPH with symplectic Euler or velocity Verlet (`reset_integrator`); PCISPH (at least one iteration, compression only); DFSPH (Bender & Koschier's order, κ/ρ); multiphase (one colour field per phase); phase change; secondary particles (foam, spray, bubbles). The last part covers the three parallel SPH loops on `src/par.cyr`, then the D3 switch to f64 keys, with the rows remapped | step corpus: a hydrostatic tank, a dam break, a droplet, a two-phase layer, each run through every solver variant; class S at frames {1, 2, 10, 60, 300}; class R from the 79 tests and the SPH integration tests; every row at each thread count |
| **M6 Coupling and MPM** | coupling, mpm | `FlipSolver`: the FLIP/APIC ratio, mass-weighted P2G, projection, `apic_c` auto-resize, `particle_level_set`. Rigid-body coupling: `BodyShape`, quadratic drag, added mass, nearby particles through the neighbour hash. `MpmSolver`: each `ConstitutiveModel` (fluid, elastic, Drucker–Prager in degrees), `validate`, the det(F) clamp, separating walls. Hand codecs for `BodyShape` and `ConstitutiveModel` (data-carrying), derived JSON for `RigidBody` | FLIP: a sloshing tank and a falling block; coupling: a sphere and a box in a flow; MPM: an elastic block, a sand column and a fluid column; the refusals |
| **M7 Compute, logging and ai** | compute, logging, ai | `ComputeOp`; `PackedParticles` with f32 packing (probe 3); `KernelDerivatives`, including `autodiff_gradient` on hisab's tape with a closure; JSON for the three kernel-parameter types. logging: the sakshi init (and the Chrome-trace check, §1.1). ai: `PvDaimonConfig` and `PvHooshConfig` with their defaults (`http://localhost:8090`, `http://localhost:8088`), their JSON, and a debug print that redacts the API key | packing round trips; derivatives against the oracle (class C) and against finite differences (class R); config defaults and JSON (class A); the redaction |
| **M8 Release 2.0.0** | all | The Rust-test ledger: each of the 351 unit tests, 33 integration tests and 7 ignored doc tests is ported, folded into golden rows, or marked n/a with a reason. The full surface-parity diff. `docs/benchmarks-rust-v-cyrius.md` with the 23 Rust benchmarks in both columns, plus thread-count rows. The README, `docs/guides/migration-2.0.md`, the CHANGELOG's Breaking section, the docs' code examples compiled and run. VERSION 2.0.0 | the full corpus green on x86_64 ELF, x86_64 PE and aarch64 |

Order: M0 → M1 → … → M8.
- Shallow water is self-contained (no hisab, no f32, no particles), so it is the cheapest place to build the
  state loader and the step rows.
- The grid comes next, because its library questions are small function rows, and its row loops are the
  simplest first user of the worker pool.
- SPH, the largest module at 2,170 lines, comes after probes 2–6 have settled.
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
| `exp` (5), `ln` (8), `log10` (1), `powf` (6), `sin` (2), `cos` (4), `sin_cos` (2), `atan2` (2), `hypot` (3) | correctly rounded: `src/crmath.cyr` and `src/crlog.cyr` (D2, ledger L1, ADR 0004) until ganita ships its tier; `cbrt` is ganita's, already exact | class C, exact against the corrected oracle |
| `cbrt` (2, Manning friction) | `ganita_f64_cbrt`; its comments describe an exact rounding decision, which step 8 measures | class C |
| `powi` (6) | explicit products in LLVM's square-and-multiply order (as tanmatra) | class B |
| `to_radians` (1) | `x * (PI / 180.0)` in Rust's order | class B |
| `as f32`, f32 `*`, `floor`, `as i32` (the neighbour hash and the packing) | native `f32`: `f32_from` (round to nearest, ties to even), f32 `*`, `f32_to`, `f64_floor`, a saturating convert (probe 3) | class A |

**Typing traps** (guide *Types*):
- An operator is typed by its left operand, so bind builtin and ganita results to a `: f64` local first, and write `-x`, not `(0 - x)`.
- An `f32` left operand reads the low 32 bits of its right operand, so write `x * f32_from(2.0)`.
- Comparisons on builtin results compare bit patterns; use `f64_lt` and its siblings.
- There is no exponent notation.
- A hex pattern in a `const` is an integer.
- The `F64_*` stdlib constants are untyped `var`s.

### 7.2 hisab 1.4.0 calls → Cyrius

| Rust (hisab 1.4.0) | Cyrius 6.7.18 / hisab 3.4.1 | Plan |
|---|---|---|
| glam `DVec3` arithmetic | `hvec3_*`, which allocate on every operation | stand-in `src/vec3.cyr` on f64 locals (probe 2), ledger L3 |
| glam `Vec3` (f32) and `SpatialHash` (f32 cells, a `HashMap` of cells) | `spatial_hash_*` (f64 cells; allocates on insert and query) | stand-in neighbour grid on native `f32` keys (D3): sorted arrays, no per-step allocation, ledger L4 |
| `num::dst`, `num::idst` | `num_dst`, `num_idst`: another algorithm, 10–20 KB per call (port-parity §4.8) | a transcribed stand-in of 1.4.0's direct DST, ledger L6 |
| `num::conjugate_gradient` (matrix-free linear CG, operator as a closure) | **none** in hisab or ganita | stand-in transcribed from 1.4.0 (ADR 0003), with the operator as a Cyrius closure, ledger L5 |
| `num::matrix_inverse` (n×n by LU with partial pivoting, called on a 3×3) | ganita `mat_inv`: **the oracle's bits on all 300 probe matrices**, but 248 B per call | an allocation-free copy of ganita's algorithm, ledger L7, until `mat_inv_into` |
| `autodiff::{Tape, Var, reverse_gradient}` | `ad_tape_*`: 0 B per evaluation once reset; within 3 ulp of 1.4.0's (a different reverse-pass order) | use it, with the user's function as a closure; its rows are class D (library-backed), set at probe 8 |

### 7.3 Types and idioms

| Rust | Cyrius |
|---|---|
| `Vec<FluidParticle>`, `&mut [FluidParticle]` | a typed stride array plus a count, per probe 2. Storage comes from freelist or the solver, never `vec` (8-byte words only, and growth never frees) |
| `ParticleSoa` | column arrays |
| `Vec<f64>` grid fields and scratch | solver-owned f64 arrays, allocated once and reused; nothing allocates on the step path |
| data-carrying enums (`BoundaryCondition`, `ConstitutiveModel`, `NonNewtonianViscosity`, `BodyShape`, …) | a kind enum (append-only) plus a payload struct with the union of the fields, because a multi-field variant boxes (§1.1) |
| `Result<T, PravashError>` | value-form `Result` plus a `PV_ERR_*` code. No `Result` through `callptr` |
| `Option<T>` | the value-form `Option` for one word, otherwise an out-pointer plus 1/0 |
| `impl Fn` parameters | closures, called through `fncallN` / `callptr` |
| serde derives | `#derive(Serialize)` / `#derive(Deserialize)`, plus hand codecs in `src/json.cyr` for `DVec3` (`[x, y, z]`) and the data-carrying enums |
| rayon `par_iter` / `par_chunks_mut` | `src/par.cyr`: a persistent worker pool on `thread_create` and `chan_*`, splitting index ranges, each index writing its own slot; a runtime thread count (D6). The pool calls `_pv_cr_load()` before it starts workers: crmath's tables load lazily behind a plain flag (ADR 0004, port-parity §6) |
| `sort_unstable_by_key` | per probe 5 |
| `tracing::trace_span!`, `warn!` | sakshi spans and events behind a level |
| `String` config fields | `Str` |

### 7.4 Names

- **Public structs** are `Pv<Name>` with `impl` blocks, so methods keep their Rust names (`s.step_pcisph(…)`).
  Constructors are `PvSphSolver_new` and the like.
- **Globals and enum members** carry `PV_`; private helpers carry `_pv_`.
- None of the three prefixes collides with the 6.7.18 stdlib or hisab 3.4.1 (checked 2026-11-11). Re-check at
  M1, because "last definition wins" across files is only a warning. The derive generates `T_to_json`-style names,
  so check those too.

### 7.5 Memory

Probe 1 settles this:
- **Solver lifetime**: particles, grid fields, neighbour structures and persistent buffers live in storage the
  solver owns, freed by its `_drop` (`PvSphSolver_drop`; not `_free`, which `ParticleArena::free(handle)` already
  names: ADR 0002 #9).
- **Step lifetime**: candidate lists, the `accels` arrays (which 1.3 allocates each step) and solver scratch
  live in a step arena that is reset at the start of every step. Each worker thread gets its own slice of it.
- **JSON** decoding allocates (the derive returns a fresh struct), which is fine: it is never on the step path.
- **Never `alloc_reset`**: it would wipe the consumer's heap.

---

## 8. Upstream requests and the conversion ledger

**The rule** (§0.1): a gap in hisab, ganita or the stdlib becomes a request in that project, and pravash ships
a local stand-in. Nothing waits on the request.

**Filing.** Requests go in the owning repo's `docs/development/issues/`, as the siblings file them: a
self-proving repro where one is possible, and the probe or benchmark numbers that show the gap. Where a request
is already open (impetus or kimiya filed it), pravash adds itself as a consumer with its own numbers instead of
filing a second one. The first batch went in at M0 step 10 (2026-11-11). Later gaps are filed in the milestone
that finds them, and a request found already being edited elsewhere waits for its milestone's numbers (L12).

**Marking.** Every stand-in function, or the head of every stand-in file, carries a note of this form:

```cyrius
# UPSTREAM(L5): hisab/docs/development/issues/<date>-linear-conjugate-gradient.md
#   replace with hisab's linear CG once it ships; parity: CG rows (iterations A, solution B)
```

`scripts/upstream_ledger.py --check`, which is written at M0, checks two things: that every `UPSTREAM(Ln)` note
in `src/` has a row below, and that every open row has at least one note. It runs in every milestone's gate.

**Converting.** When a library ships the replacement, before or after 2.0:
1. bump the dependency pin (hisab's tag, or the Cyrius pin for ganita and the stdlib);
2. swap the call and delete the stand-in;
3. re-run the parity rows the ledger names. They must keep their class. If the bits change, the cause is one of
   two things. Either the library is more accurate, in which case it is accepted under the accuracy policy against
   the corrected oracle. Or it is a defect, which is filed upstream, and the stand-in stays;
4. close the row here, remove the notes, and add a CHANGELOG entry.

**The ledger.** Seeded at M0 step 10 (2026-11-11), when every request below was filed with pravash's numbers.
`scripts/upstream_ledger.py` reads this table, so keep its shape:
- the status is `open`, `conditional` (no request until its milestone measures a need) or `closed`;
- each stand-in file is a backticked `src/…` path;
- each request is a backticked `<repo>:<path>` (from `/home/macro/Repos`).

| # | Status | pravash stand-in | Gap | Upstream request | Converts to | Parity rows to re-run |
|---|---|---|---|---|---|---|
| L1 | open | `src/crmath.cyr` (exp, pow, sin, cos, sin_cos, hypot, atan2: impetus's crmath) and `src/crlog.cyr` (ln, log10: kimiya's), with their generated tables `src/crmath_data.cyr` and `src/crlog_data.cyr`: the correctly rounded primitives (D2, ADR 0004). Not cbrt: ganita's already equals the oracle's on every row (port-parity §2.1, §3) | ganita has no correctly rounded tier, and no log10 at all | `ganita:docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md` (pravash joined as the third consumer, adding ln and log10) | ganita's correctly rounded functions | every class C row, which must stay 0 ulp against the corrected oracle |
| L2 | open | `src/mathx.cyr`: Rust-semantics min / max / clamp, `is_nan`, `is_finite`, `sin_cos`, and the saturating casts `_pv_as_i32` / `_pv_as_u32` / `_pv_as_usize`. The typed std float constants are generated into `src/constants.cyr` | `f64_min` / `f64_max` ignore a NaN only as the first operand; no public predicates, typed constants or saturating converts (port-parity §1) | `cyrius:docs/development/issues/2026-11-11-f64-min-max-drop-nan-only-as-first-operand.md` | the stdlib's `f64_*` | the M0 math sweep, class A |
| L3 | open | `src/vec3.cyr`: `DVec3` arithmetic on `f64` locals | every `hvec3_*` result allocates, 24 B (port-parity §4.1) | `hisab:docs/development/issues/2026-11-09-allocator-scoped-calls.md`; `cyrius:docs/development/issues/2026-11-09-no-scoped-allocator-for-alloc.md` (pravash's numbers added to both) | hisab's vector forms, once a call's allocation can be scoped | the class B rows of every module that uses vectors |
| L4 | open | `src/neighbor.cyr`: the neighbour grid on sorted arrays (f32 keys until the D3 switch) | `spatial_hash_query_cell` allocates a vec per query: 4.1 MB per 1,000-particle step (port-parity §4.1) | `hisab:docs/development/issues/2026-11-11-spatial-hash-per-step-rebuild-allocates.md` | `spatial_hash_*` with a non-allocating query | neighbour lists (class A) and every SPH and coupling step row |
| L5 | open | `src/grid_solve.cyr`: matrix-free linear CG, transcribed from hisab 1.4.0 | no linear CG in hisab 3.4.1 or ganita | `hisab:docs/development/issues/2026-11-11-no-matrix-free-linear-cg.md` | hisab's linear CG | the CG pressure rows (iterations class A, solution class B) |
| L6 | open | `src/grid_solve.cyr`: 1.4.0's direct DST and IDST, transcribed, on the correctly rounded `sin` (L1) | `num_dst` / `num_idst` allocate 10–20 KB per call, and are another algorithm (up to 67,000 ulp from 1.4.0's) (port-parity §4.1, §4.8) | `hisab:docs/development/issues/2026-11-11-dst-allocates-per-call.md` | `num_dst` / `num_idst` into caller scratch, with the DST rows re-bounded for their algorithm | the DST pressure rows |
| L7 | open | `src/sph_fns.cyr`: the MLS 3×3 inverse, an allocation-free copy of ganita's `mat_inv` algorithm (the oracle's bits on all 300 probe matrices) | ganita `mat_inv` allocates 248 B per call, once per particle per step (port-parity §4.1, §4.8) | `ganita:docs/development/issues/2026-11-11-mat-inv-allocates-per-call.md` | ganita `mat_inv_into` | the MLS gradient-correction rows |
| L8 | open | the step-path rule itself: solver-owned storage on freelist, step scratch in an arena, hisab off the step | any library call that returns a value leaks on the bump heap | `hisab:docs/development/issues/2026-11-09-allocator-scoped-calls.md`; `cyrius:docs/development/issues/2026-11-09-no-scoped-allocator-for-alloc.md` | hisab back on the step path: L3, L4 and L6 convert together | every step row, plus the soak test |
| L9 | open | `src/sph_fns.cyr`: a stable merge sort on (key, input index) (ADR 0002 #3; port-parity §4.5) | `vec_sort_by` is introsort; no stable sort exists | `cyrius:docs/development/issues/2026-11-11-no-stable-sort.md` | the stdlib's stable sort | the z-order rows (class A) |
| L10 | open | `src/json.cyr`: hand codecs for `DVec3` fields (`[x, y, z]`) and the data-carrying enums | `#derive` has no array form for a vector struct and no externally tagged data variants | `cyrius:docs/development/issues/2026-11-11-derive-json-no-vector-array-or-tagged-variant-shapes.md` | `#derive(Serialize)` / `#derive(Deserialize)` | the JSON table (class A) |
| L11 | open | `src/logging.cyr`: a chrome://tracing writer over sakshi's spans (M7) | sakshi 2.5.10 has spans and text outputs, but no Trace Event export (checked at M0 step 10) | `sakshi:docs/development/issues/2026-11-11-no-chrome-trace-export.md` | sakshi's exporter | the logging test |
| L12 | closed | (none needed) an 8-byte-chunk record copy for particles, had M1's benchmarks shown `memcpy` dominating. Closed at M1 part 3: struct assignment `*dst = *src` copies a 144-byte record word-wise in 11.7 ns, against `memcpy`'s 245 ns, and every particle move is an assignment (benchmarks-rust-v-cyrius, M1 part 3) | `memcpy` copies a byte at a time | the open `cyrius:docs/development/issues/2026-11-02-memcpy-memset-byte-loops-dominate-frame-copies.md` (pravash's numbers added 2026-11-11) | stdlib `memcpy`, where raw bytes are copied | none |
| L13 | open | `src/column.cyr`: a growable column of 8-byte slots on the freelist (push doubling and freeing the outgrown buffer, `drop`, NaN / 0 on a bad index), what `ParticleSoa`'s `Vec<f64>` / `Vec<u8>` fields become | `vec_push_a` never frees the outgrown buffer, even through a freeing Allocator, there is no `vec_free`, and `vec_get` / `vec_set` abort on a bad index | `cyrius:docs/development/issues/2026-11-11-vec-keeps-outgrown-buffer-and-aborts.md` | the stdlib `vec` with a freeing grow, `vec_free_a` and non-aborting access | the ParticleSoa rows of `tests/golden/common.tsv` (class A) |

**Also filed at M0, outside the ledger:** pravash's case on the `cyrius port` clobber issue. A project whose
`.gitignore` ignores `Cargo.lock` ends up with a frozen oracle that has no lockfile.

**Also filed at M1 part 4, outside the ledger:** `bayan:docs/development/issues/2026-11-11-json-integer-negative-zero-loses-its-sign.md` (the integer `-0` reads as +0, where serde_json reads −0.0; one pending JSON row, with impetus's S3 and S4).

**Also filed at M1, outside the ledger** (a finding with no stand-in): `cyrius:docs/development/issues/2026-11-11-freelist-maps-every-block-over-4k.md`. `fl_alloc` maps and unmaps every block over 4 KB, so building and dropping an arena or an SoA store pays an mmap and a page fault per page. Solvers allocate once, so the step path is untouched.

---

## 9. After 2.0.0

The roadmap plans the work that follows 2.0.0:
- the 1.3.0 audit's deferred physics (D5), each with reference tests;
- converting whatever ledger rows (§8) are still open at 2.0 as their libraries ship. L8, which puts hisab back
  on the step path, is the largest of these;
- retiring `rust-old/`.

A stand-in is never a reason to hold the 2.0 release, and an upstream release is never a reason to hold a milestone.

---

## 10. Tracking

| Milestone | Status | Golden rows | Reference tests | Benches | Notes |
|---|---|---|---|---|---|
| M0 Initialize | ✅ **done 2026-11-11** | constants 35 + math 33,239. Gated: **21,852 ok, 0 failed, 1,790 measured (powf, until M1), 9,632 pending (the M1 helpers)**, identical on x86_64 ELF, x86_64 PE and aarch64 | — | 3 probe benches (layout, pool, memory) | Steps 0–10, recorded in §4 and port-parity §1–§4:
  - the oracle frozen at `a88bc4e` (1.3.0 published), with the scaffold and repairs, ADRs 0001–0003 and `cyrius.cyml`;
  - the semantics probe;
  - `scripts/oracle/` (validated, deterministic, parallel-checked), `src/constants.cyr` and `tests/parity.tcyr`;
  - the cross-target digests and K bounds;
  - the eight probes;
  - nine upstream requests filed, four open issues extended, and the ledger seeded and checked in CI.

  Next: M1 |
| M1 Foundation | ✅ **done 2026-11-11** | bridge 3,964 + buoyancy 1,716 + common 3,683 + json 302 + vortex 2,293 + math 33,239: **42,697 ok, 72 accepted (ADR 4), 0 failed, 0 measured, 2,463 pending** (the M4 cell key 2,460; bayan S3, S4, −0) on x86_64 ELF, x86_64 PE and aarch64 | all 100 Rust unit tests of the M1 modules and 11 integration tests: `tests/error`, `common`, `buoyancy`, `vortex`, `bridge`, `integration` `.tcyr`, plus `tests/crmath.tcyr` (85) | `math/*` (10 primitives against glibc); `common/*`, `soa/*`, `arena/*` (9); `m1/*` (7) and `json/*` (4) against Rust 1.3.0 | **Part 1 done 2026-11-11** (port-parity §5): `src/mathx.cyr`, the Rust-semantics helpers (`UPSTREAM(L2)`); the first `[lib]` (constants, mathx), `dist/pravash.cyr` and `tests/lib.cyr`; 6,371 rows newly gated, all bit-exact, identical on three targets.

**Part 2 done 2026-11-11** (port-parity §6, ADR 0004): the correctly rounded primitives (`UPSTREAM(L1)`):
- `src/crmath.cyr` (impetus's crmath: exp, sin, cos, sin_cos, hypot, pow, atan2) and `src/crlog.cyr` (kimiya's ln, log10);
- their tables, generated: `crtables.py` (impetus's) and `lntables.py` (kimiya's), checked against MPFR 4.2.2 and integer series;
- the corrected oracle `tests/golden/math.cr.tsv`: the 43 rows glibc 2.44 misrounds;
- `src/mathx.cyr`'s wrappers;
- K = 0 for every class C math function, powf and log10 included: all 43 corrected rows accepted, identical on three targets;
- `tests/crmath.tcyr` and the first benchmark record (`docs/benchmarks-rust-v-cyrius.md`).

**Part 3 done 2026-11-11** (port-parity §7): `src/error.cyr` (`PV_ERR_*`), `src/vec3.cyr` (`UPSTREAM(L3)`), `src/column.cyr` (`UPSTREAM(L13)`, filed) and `src/common.cyr`:
- the golden module `common.tsv`, 3,683 rows (59 arena scenarios among them), recomputed by `validate.py` and bit-exact on the first replay on three targets;
- the ported Rust tests (the 2 serde tests wait for part 4);
- `scripts/surface_parity.py` in CI, 98 items ok;
- ADR 0002 rows 8 (no aborts) and 9 (out pointers, views, `_drop`);
- L12 closed: struct assignment copies records word-wise, 21× faster than `memcpy`;
- the freelist's large-block cost filed upstream.

**Part 4 done 2026-11-11** (port-parity §8): `src/buoyancy.cyr`, `src/vortex.cyr`, `src/bridge.cyr` and `src/json.cyr` (`UPSTREAM(L10)`):
- the golden modules `buoyancy`, `vortex`, `bridge` (7,973 rows) and `json` (302 documents), every one recomputed by `validate.py`;
- the corrected oracle extended to `vortex.cr.tsv` (3) and `bridge.cr.tsv` (26);
- serde_json (`float_roundtrip`) and serde in the generator, at the oracle's versions;
- `PV_ERR_DESERIALIZE` and ADR 0002 #10;
- the bayan `-0` request filed;
- 45 + 2 + 11 Rust tests ported;
- surface parity 128.

**Next:** M2, shallow water |

| M2 Shallow water | — | — | — | — | |
| M3 Grid | — | — | — | — | |
| M4 SPH foundations | — | — | — | — | |
| M5 SPH solvers | — | — | — | — | |
| M6 Coupling and MPM | — | — | — | — | |
| M7 Compute, logging and ai | — | — | — | — | |
| M8 Release 2.0.0 | — | — | — | — | |
