# pravash — Current State

> Refreshed every release and at every port milestone. `CLAUDE.md` is
> preferences/process/procedures (durable); this file is **state** (volatile).

## Version

**2.0.0** (`VERSION`), the Cyrius library, ready to tag (M8 part e, 2026-11-13). `cyrius.cyml` reads the version
through `version = "${file:VERSION}"`, and the release workflow checks both against the tag and publishes the
CHANGELOG's `## [2.0.0]` section.

The last Rust release is **1.3.0**: commit `a88bc4e` (the audited 1.3.0 plus the hidden `oracle` hooks), frozen
as the oracle in `rust-old/`. It is tagged `1.3.0` and published to crates.io (2026-11-11, port plan D1).

## Port progress

Plan of record: [`pravash-port-plan.md`](pravash-port-plan.md).

| Milestone | Status |
|---|---|
| M0 Initialize | ✅ **done 2026-11-11.** Step 0 (Rust, 2026-11-11): P(-1) gates green; the hidden `oracle` feature (dumps for all eleven stateful types, the SPH neighbour lists, the four pressure-solver hooks, the MLS inverse); `tests/oracle.rs` 14 tests, mutation-checked, green with and without `parallel`; `Cargo.lock` tracked; glibc 2.44. Step 1 (2026-11-11): `cyrius port` scaffold, with the Rust crate in `rust-old/`, which still passes there (351 unit, 33 integration, 14 oracle). Step 2: repairs done (CLAUDE.md, CI, docs, ADRs 0001–0003). Step 3: `cyrius.cyml`, with hisab 3.4.1 test-scoped; `build`, `test`, `test --bounds`, `bench`, `fuzz`, `lint`, `fmt`, `vet`, `deps --locked` and `audit` all green. Step 4: the semantics probe ([`port-parity.md`](port-parity.md) §1), with the helpers it calls for in ledger L2. Step 5: the golden-value generator (`scripts/oracle/`): constants 29 + math 33,239 rows, deterministic, reproduced by the parallel build, all recomputed by `validate.py`; the oracle's cbrt is compiler_builtins' (port-parity §3). Step 6: `src/constants.cyr` (14 public + 6 std float constants, both lists proven complete), every value compiled and compared by `probe_tables.py`. Step 7: the parity harness (`tests/parity.tcyr`): 5,149 ok, 0 failed, 18,493 measured, 9,632 pending; ganita's cbrt equals the oracle's. Step 8: the cross-target baselines: digests identical on x86_64 ELF, PE and aarch64; K = 2 for the primitives, 0 for cbrt; gated **21,852 ok, 0 failed** (port-parity §2.2). Step 9: the eight probes (port-parity §4): every hisab/ganita return leaks on the step path (the hash rebuild 4.1 MB per 1,000-particle step); AoS + f64 locals fastest, struct values 2.5×; f32 keys 10⁶/10⁶ on 3 targets; the neighbour order is (cell offset, insertion); a stable sort replaces ipnsort's tie order; serial = parallel in Rust, and a Cyrius persistent pool gives the serial bits at T = 1–8 (3.9× at 8); one bundle with DCE; hisab's DST is another algorithm (L6 transcribes), ganita's `mat_inv` gives the oracle's bits (L7 for memory only). Step 10: nine upstream requests filed (hisab ×3, ganita, cyrius ×4, sakshi) with verified repros; pravash's numbers in four open issues; the conversion ledger seeded (11 open, 1 conditional) and checked by `scripts/upstream_ledger.py` in CI |
| M1 Foundation | ✅ **done 2026-11-11.** Part 1: `src/mathx.cyr`, the first `[lib]`. Part 2 (ADR 0004): the correctly rounded primitives (`src/crmath.cyr`, `src/crlog.cyr`, L1) and the MPFR-corrected oracle. Part 3: `src/error.cyr`, `src/vec3.cyr` (L3), `src/column.cyr` (L13), `src/common.cyr`; surface parity in CI. Part 4: `src/buoyancy.cyr`, `src/vortex.cyr`, `src/bridge.cyr`, `src/json.cyr` (L10; `PV_ERR_DESERIALIZE`, ADR 0002 #10). 45,232 golden rows in seven modules, identical on three targets: 42,697 ok, 72 accepted (ADR 4), 2,463 pending (the M4 cell key; three bayan JSON policies). All 100 Rust unit tests of the M1 modules and 11 integration tests ported |
| M2 Shallow water | ✅ **done 2026-11-11.** `src/shallow.cyr`: construction, the cell reads, volumes, CFL, the disturbance, sediment, JSON and the step (default, HLL, dispersion, friction, breaking, walls). 888 golden rows bit-exact on three targets, 142 of them class S step rows (the new class), every one recomputed by a Python transcription of the step; all 44 + 3 Rust tests ported. The step runs at 7.2× Rust (call overhead `#inline` declines: filed) |
| M3 Grid | ✅ **done 2026-11-12.** Part a ✅ 2026-11-11: `src/grid_solve.cyr` (the DST, IDST and CG transcriptions, L5 and L6) and `src/grid.cyr` (`PvFluidGrid`, `PvGridConfig`, `PvBoundaryCondition`: construction, the reads, diffuse, all four advections, Smagorinsky, buoyancy, vorticity confinement, the four projections, the wall rules, JSON); `_pv_fmod` (L14). 569 golden rows bit-exact on three targets and with rayon on, 188 of them class S step rows on the periodic and multigrid projections, every one recomputed by a Python transcription; 44 of 49 Rust tests and the 5 grid integration tests ported. Part b ✅ 2026-11-12: the DST and CG projections under parity (21 DST and 49 CG oracle-hook rows, 188 more step rows; 827 grid rows in all, 20 of them bit-exact against the corrected oracle `grid.cr.tsv`, where glibc misrounds a DST sine or cosine), and the DST plans (`dst_x`, `dst_y`: each transform length's sines and eigenvalue terms, kept across steps; the same bits), which take the 63 × 63 DST step from 719 ms to 8.6 ms (Rust 10.1 ms). Part c ✅ 2026-11-12: `src/k_epsilon.cyr` (`PvKEpsilon`), `apply_ghost_fluid` and `src/mac_grid.cyr` (`PvMacGrid`), with JSON; 293 more rows (1,120 grid rows), all bit-exact on the first replay, among them 135 class S step rows; all 49 Rust tests of `grid.rs` ported; surface parity covers all of `grid`. Part d ✅ 2026-11-12: `src/vof.cyr` (`PvVofField`) and `src/phase_field.cyr` (`PvPhaseField`), with JSON; two new golden modules, 384 rows, bit-exact on the first replay, among them 138 class S step rows; all 8 + 11 Rust tests. Part e ✅ 2026-11-12: `src/par.cyr`, the persistent worker pool, and `PvFluidGrid.set_threads` (D6) on Rust's three `parallel` loops; every `FluidGrid::step` row replayed at 2, 3, 4 and 8 threads gives its one-thread words, on three targets |
| M4 SPH foundations | ✅ **done 2026-11-12.** Part d: the physics extensions in `src/sph_fns.cyr` (the list functions, phase change, the `dyn PvReactionProvider`, the contact angle, the secondary particles), 318 rows, 1 against the corrected oracle; L7's 3×3 inverse a transcription of hisab 1.4.0's `matrix_inverse` (ganita's `mat_inv` differs on 12 of 60 rows); 11 Rust tests; ADR 0002 #18. Part c: `src/sph_fns.cyr`, the brute-force density and forces, the free step (`sph_step`, ADR 0002 #17), the particle block, split and merge; 198 rows (5 split rows against the corrected oracle); 10 Rust tests and the 4 free-function SPH integration tests. A cyrius miscompile (`s.f = g(t.e)`, open upstream) caught by the merge rows and worked around. Part b ✅ 2026-11-12: `src/neighbor.cyr`, the neighbour grid on bit-faithful f32 keys (L4) and `sort_by_zorder` on a stable merge sort (L9); 50 rows bit-exact (30 neighbour lists, including the D3 probe drift at large offsets and a kept grid; 20 z-order), and the 2,460 cell-key rows pending since M0; ADR 0002 #16. Part a ✅ 2026-11-12: `src/sph_kernels.cyr` (the seven kernels, `_PvKernelCoeffs`, the linear and Tait equations of state, the non-Newtonian models, the batch and whole-system helpers); private constants generated (`src/constants_private.cyr`, `t_private.rs`); a new golden module `sph.tsv`, 804 rows bit-exact on the first replay, 16 of them against the corrected oracle `sph.cr.tsv`; 22 Rust tests |
| M5 SPH solvers | ✅ **done 2026-11-12.** Part e: the solver's thread count (`set_threads`; Rust's three `parallel` loops on the pool, every step row at 2, 3, 4 and 8 threads giving the serial bits; ADR 0002 #4) and the D3 switch to f64 neighbour keys (hisab 3.4.1's cell, integer neighbour cells; 596 rows mapped by name to `math.d3.tsv` and `sph.d3.tsv`, all passing against validate.py's f64 transcription; ADR 0002 #2). Part d ✅ 2026-11-12: `src/sph_multiphase.cyr`, the multiphase step with `PvPhaseProperties` and `PvMultiPhaseConfig`; 26 rows, the first 25 bit-exact on the first replay and the 26th closing the mutation tests' one gap; the last 5 Rust tests of `sph.rs`; ADR 0002 #22. Part c ✅ 2026-11-12: `src/sph_dfsph.cyr`, the DFSPH step; 35 rows, bit-exact on the first replay; 6 Rust tests; ADR 0002 #21. Part b ✅ 2026-11-12: `src/sph_pcisph.cyr`, the PCISPH step; 37 rows, bit-exact on the first replay; 4 Rust tests; ADR 0002 #20. Part a ✅ 2026-11-12: `src/sph.cyr`, `PvSphSolver` and its WCSPH step (Euler and Verlet, CSF surface tension, the f32 lists of M4 part b) and `adaptive_dt`; 66 rows, every one bit-exact on the first replay; 19 Rust tests and 5 integration tests; ADR 0002 #19. |
| M6 Coupling and MPM | ✅ **done 2026-11-12.** Part a ✅ 2026-11-12: `src/coupling.cyr`, the rigid bodies (`PvBodyShape` / `PvShape`, `PvRigidBody`, the two-way coupling, integration with and without added mass, both drags, the particle level set; JSON for `BodyShape` and `RigidBody`). `drag_from_particles_indexed` takes the caller's cells-only `PvNeighborGrid` (`PvNeighborGrid_build`, f64 keys; ADR 0002 #23). A new golden module, `coupling.tsv`: 4,798 rows, every one bit-exact on the first replay but an indexed drag at 10⁶, which `coupling.d3.tsv` maps (ADR 2); 22 of the 33 Rust tests. Part b ✅ 2026-11-12: `src/flip.cyr`, the `FlipSolver` (P2G, the projection, FLIP/PIC and APIC G2P, the APIC resize; ADR 0002 #24): 245 rows, every one bit-exact on the first replay, among them 210 class S trajectories and scenarios; all 33 Rust tests of `coupling.rs` and all 33 integration tests. Part c ✅ 2026-11-12: `src/mpm.cyr` (`PvConstitutiveModel` / `PvConstitutiveLaw` and its JSON, `PvMat2`, `PvMpmParticle`, `PvMpmSolver`; the 2 × 2 SVD and Drucker–Prager on the correctly rounded primitives; ADR 0002 #25): a new golden module, `mpm.tsv`, 1,336 rows, 1,319 bit-exact against the oracle and 17 against the corrected oracle `mpm.cr.tsv` where glibc misrounds; all 13 Rust tests |
| M7 Compute, logging and ai | ✅ **done 2026-11-12.** Part b ✅ 2026-11-12: `src/ai.cyr` (`PvDaimonConfig`, `PvHooshConfig`: the defaults, the setters with their ownership bits, JSON, Rust's `{:?}` and `{:#?}` with the key redacted, `<str as Debug>`'s escapes from `src/str_debug_data.cyr`, generated from rustc 1.99.1's core; ADR 0002 #27) and `src/logging.cyr` (the three inits, the PRAVASH_LOG filter as tracing-subscriber's EnvFilter decides it, and the Trace Event file through a sakshi hook, ledger L11; ADR 0002 #28), with every one of Rust's 60 spans and 12 warnings at its point in every module. Two new golden modules, `ai.tsv` 579 rows and `logging.tsv` 251 (99 filter rows, 152 trace rows over 37 calls), all matching on the first replay; the instrumentation changed no other row; all 3 Rust tests. Part a ✅ 2026-11-12: `src/compute.cyr` (`PvF32Column`, `PvPackedParticles` and its packing, the three kernel-parameter records and their JSON, `PvComputeOp`, the traits `PvComputeBackend` and `PvNeuralCorrector`, the `KernelDerivatives` functions) and `src/compute_ad.cyr` (`autodiff_gradient` on hisab's tape, outside the bundle; ADR 0002 #26); the f32 JSON reads as serde_json's `float_roundtrip` (ledger L15). A new golden module, `compute.tsv`: 1,228 rows, every one matching on the first replay but three f32 reads, which the stand-in fixed; all 11 Rust tests |
| M8 Release 2.0.0 | ✅ **done 2026-11-13; 2.0.0 ready to tag.** Part e ✅ 2026-11-13: VERSION 2.0.0 and the CHANGELOG's `## [2.0.0]`; the release gate (22 gates: every suite on x86_64 ELF, PE and aarch64, the corpus on each, the generator on the pinned toolchain, validate.py, the docs, the fuzz harness); the fuzz harness (`tests/pravash.fcyr`, in CI); CI and the release independent of Rust: the job that built and tested `rust-old/` and the two steps that read it are gone, and the local tools run `cargo +1.99.1`; the two trace-file tests no longer require `chdir`, which the stdlib's Windows build doesn't wire. Part d ✅ 2026-11-13: Rust's derived `Debug` on the 35 public types (`src/fmtf.cyr`, Rust's shortest float text, impetus's printer with the f32 case; `src/debug.cyr`, the builders and a `debug(sb, alternate)` each; ADR 0002 #29; ledger L16, `#derive(Debug)` and the printer requested at cyrius). A new golden module, `debug.tsv`: 9,602 rows (6,871 f64 and 2,565 f32 printings, 156 values, 10 states after their own steps, which leave Rust's scratch), all matching, all recomputed by validate.py; 14 of 15 mutations caught, the 15th equivalent. Part c ✅ 2026-11-13: the full surface diff (`scripts/surface_parity.py` adds Rust's derives, its trait methods and the reverse direction: 625 items ok, the 36 `Debug` derives pending part d; 466 public Cyrius names, 351 Rust items and 115 Cyrius-only by 18 rules citing their ADR rows). It found six missing clones, ported with a test group each, and three Rust-private types left public, now `_Pv` names. The README, `docs/guides/migration-2.0.md` and the CHANGELOG's Breaking section; Rust's three examples ported (`docs/examples/`) with outputs derived from the oracle (`scripts/oracle/examples.py`: shallow water prints Rust's output exactly, and the SPH examples leave it at the second step by ADR 0002 #2 alone); `scripts/doc_examples.py` runs the 14 blocks and programs, in CI. Part b ✅ 2026-11-12: the release benchmark record (`scripts/bench_record.py`): all 57 rows of Rust's 21 criterion groups in both columns, three pinned interleaved rounds (median 4.4× Rust's time, 0.76× to 37×), and the thread-count rows (the SPH step 2.2–3.2× faster at 8 threads, Rust's rayon 1.05–1.95×). Part a ✅ 2026-11-12: the Rust-test ledger (`scripts/rust_test_ledger.py`, gated in CI; [`rust-test-ledger.md`](rust-test-ledger.md)): all 405 Rust tests accounted for, 391 ported and 14 n/a (the oracle hooks' own); Rust's 7 `ignore`d doc examples written against the API and run as `doc_*` groups |

## Toolchain

- **Cyrius pin**: 6.7.18 (`cyrius.cyml` `[package].cyrius`)
- **hisab**: 3.4.1 (Cyrius), commit `d6703c5`, `scope = "test"`: the bundle calls none, and its one caller, `src/compute_ad.cyr`, stays outside `[lib]` (ADR 0002 #26). It also declares a sakshi package, so `cyrius deps` warns that it refuses to overwrite the stdlib leaf; the two copies are byte-identical
- **sakshi**: 2.5.10, a stdlib leaf since M7 part b (pravash's tracing)
- **ganita**: 1.2.15, bundled in the 6.7.18 stdlib
- **Rust 1.3.0** (`rust-old/`, reference only; nothing in the build, the tests, CI or the release uses it): the corpus
  was recorded on rustc 1.99.1, hisab 1.4.0, glam 0.29.3 (`rust-old/Cargo.lock`) and glibc 2.44. The local porting
  tools run it with `cargo +1.99.1`. On stable 1.100.0 (2026-11-09) the generator writes another
  `src/str_debug_data.cyr` (Unicode 18: 916 ranges for 1.99.1's 910), and its `--check` fails.

## Source

- **Rust oracle**: 6,882 library code lines in `rust-old/src/` (16 modules, 193 `pub fn`s) plus the `oracle`
  hooks. Frozen; never edited.
- **Cyrius**: `[lib]` = `src/error.cyr`, `src/constants.cyr` (generated), `src/crmath_data.cyr` (generated),
  `src/constants_private.cyr` (generated), `src/crmath.cyr`, `src/crlog_data.cyr` (generated), `src/crlog.cyr`,
  `src/mathx.cyr`, `src/logging.cyr`, `src/vec3.cyr`,
  `src/column.cyr`, `src/common.cyr`, `src/buoyancy.cyr`, `src/vortex.cyr`, `src/bridge.cyr`, `src/shallow.cyr`,
  `src/par.cyr`,
  `src/grid_solve.cyr`, `src/grid.cyr`, `src/mac_grid.cyr`, `src/k_epsilon.cyr`, `src/vof.cyr`, `src/phase_field.cyr`,
  `src/sph_kernels.cyr`, `src/neighbor.cyr`, `src/sph_fns.cyr`, `src/sph.cyr`, `src/sph_pcisph.cyr`,
  `src/sph_dfsph.cyr`, `src/sph_multiphase.cyr`, `src/coupling.cyr`, `src/flip.cyr`, `src/mpm.cyr`, `src/compute.cyr`,
  `src/str_debug_data.cyr` (generated), `src/ai.cyr`, `src/json.cyr`, `src/fmtf.cyr` and `src/debug.cyr`, bundled as
  `dist/pravash.cyr`; `src/compute_ad.cyr` outside the bundle (a consumer includes it
  and declares hisab); the scaffold's `src/main.cyr` stub.

## Tests

- **Rust oracle** (`cargo test --manifest-path rust-old/Cargo.toml --all-features`): 351 unit, 33 integration,
  14 oracle, 7 doc tests `ignore`d.
- **Cyrius**:
  - `tests/parity.tcyr` replays the corpus: 18 modules, 0 failing; 67,276 ok, 728 accepted (ADR 4: 131 against the
    corrected oracle, ADR 2: 597 against the f64-key oracle), 0 failed, 0 measured, 3 pending (three JSON rows on
    bayan's policies). Every `FluidGrid::step` row (376) and every `SphSolver` step row (147) also runs at 2, 3, 4
    and 8 threads, 0 differing from one. Bits are identical on x86_64 ELF, x86_64 PE and aarch64.
  - `tests/crmath.tcyr`: 85 checks in 18 groups.
  - The ported Rust tests:
    - `tests/common.tcyr`: all 49 of `common.rs`, the SoA clone (M8 part c), and the Debug of kinds outside their
      enums and of a builder that holds text (M8 part d): 128 checks;
    - `tests/error.tcyr`: the 6 (17 checks);
    - `tests/buoyancy.tcyr` (16), `vortex.tcyr` (12), `bridge.tcyr` (17);
    - `tests/shallow.tcyr`: all 44 of `shallow.rs` (155 checks);
    - `tests/grid.tcyr`: all 49 of `grid.rs`, and two Cyrius-only groups (the layouts, the DST plans): 218 checks;
    - `tests/vof.tcyr`: all 8 of `vof.rs` (27 checks); `tests/phase_field.tcyr`: all 11 of `phase_field.rs`
      (38 checks);
    - `tests/par.tcyr` (Cyrius-only): the worker pool, the grid's thread count and the solver's (50 checks);
    - `tests/sph.tcyr`: all 79 of `sph.rs`, and ten Cyrius-only groups (the merge sort, the lists against a
      brute-force search, at 10⁶ and 10¹², and at saturated keys, the grid's and the solver's layouts, part c's
      no-panic paths, `reset_integrator`, the multiphase config's clone), and Rust's two `ignore`d doc examples, run
      (M8 part a): 202 checks;
    - `tests/coupling.tcyr`: all 33 of `coupling.rs`, and nine Cyrius-only groups (the layouts, the grid's
      cell-size rule, the block walk, the indexed drag against the brute force at 10⁶ and 10¹², the level set's
      refusals, the FLIP solver's construction, APIC resizing, divergence path and flat heap): 106 checks;
    - `tests/mpm.tcyr`: all 13 of `mpm.rs`, and four Cyrius-only groups (the layouts, the refusals with nothing
      changed, the det(F) clamp, the flat heap): 42 checks;
    - `tests/compute.tcyr`: all 11 of `compute.rs`, and seven Cyrius-only groups (the layouts, f32 rounding, the op
      names, a backend through `dyn`, autodiff against analytical gradients and on a growing tape, finite differences,
      the L15 f32 reads, the two clones), and Rust's four `ignore`d doc examples, run (M8 part a): 109 checks;
    - `tests/ai.tcyr`: both of `ai.rs`, and Cyrius-only groups (the layouts, ownership, the escape table's edges,
      every malformed UTF-8 form, both Debug shapes, the JSON text checks, the two clones): 56 checks;
    - `tests/logging.tcyr`: the 1 of `logging.rs`, and Cyrius-only groups (the span and event texts through a sakshi
      hook, the level gates, a full 16-deep stack, the parser's corners, the trace file), and Rust's `ignore`d doc
      example of `init_profiling`, run (M8 part a): 49 checks (the two `chdir` asserts went at M8 part e);
    - `tests/integration.tcyr`: all 33 of `rust-old/tests/integration.rs` (79 checks).
  - `scripts/surface_parity.py` (the full diff since M8 part c): error, common, buoyancy, vortex, bridge, shallow,
    grid, vof, phase_field, sph, coupling, mpm, compute, ai and logging. 625 items ok, 0 missing, 0 pending: the
    public items, the 7 trait methods and all 157 derives (M8 part d added the 35 types' Debug; `PravashError`'s is an
    exception). The reverse direction: 501 public Cyrius names, 386 Rust items and 115 Cyrius-only by 18 rules.
  - `scripts/doc_examples.py` (M8 part c, in CI): 15 blocks and programs, all running (the README's Quick Start, 11
    in the migration guide, Rust's three examples against their `.out`). `scripts/oracle/examples.py` derives the
    `.out` files from the Rust examples and validate.py's transcription.
  - `scripts/rust_test_ledger.py --check` ([`rust-test-ledger.md`](rust-test-ledger.md), M8 part a): all 405 Rust
    tests accounted for. 391 are ported to same-named groups (351 unit, 33 integration, and the 7 `ignore`d doc
    tests written against the API and run). The 14 oracle-hook tests are n/a.
  - `tests/pravash.fcyr` (M8 part e, `cyrius fuzz`, in CI): the 22 JSON readers on 13,200 mutated documents (1,582
    accepted, each written and read back to the same text, each small solver stepped) and the `PRAVASH_LOG` parser on
    40,000 strings; no invariant broken. The scaffold's `tests/pravash.tcyr` smoke test is gone.
  - Every suite passes on x86_64 ELF, x86_64 PE and aarch64 (1,437 checks on each; M8 part e's release gate).
- **Golden corpus** (`tests/golden/`): ai 579, bridge 3,964, buoyancy 1,716, common 3,683, compute 1,228, constants 35, coupling 5,043 (210 class S), debug 9,602, grid 1,120 (511 class
  S step rows), json 302, logging 251, math 34,049, mpm 1,336 (192 class S), phase_field 160 (75 class S), shallow 888 (142 class S step rows), sph 1,534 (190 class S), vof
  224 (63 class S) and vortex 2,293 rows, and the corrected oracle (MPFR 4.2.2): `math.cr.tsv` 43, `vortex.cr.tsv`
  3, `bridge.cr.tsv` 26, `grid.cr.tsv` 20, `sph.cr.tsv` 22 and `mpm.cr.tsv` 17 rows; and the f64-key oracle (ADR 0002 #2):
  `math.d3.tsv` 558, `sph.d3.tsv` 38 and `coupling.d3.tsv` 1 rows.
  `validate.py` recomputes all 68,007 rows, every solver step by transcription, and the list-dependent rows again on
  f64 keys, and every Debug text by its own printer: 0 failures (in about 100 s);
  `probe_tables.py` 41 of 41.
- **Benchmarks** (`tests/pravash.bcyr`, [`../benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md)).
  - **Debug text** (M8 part d, `debug_d/*` against a twin, two pinned interleaved rounds): eight f64s 54× Rust's
    time (32 µs, the exact bignum digit search), a particle 6.4× and its pretty form 3.8×, an 8 × 8 grid after a step
    32×. Off the step path; filed with L16.
  - **The 2.0.0 record** (M8 part b, written by `scripts/bench_record.py`): all 57 rows of Rust's 21 criterion
    groups, three pinned interleaved rounds. The port runs at a median of 4.4× Rust's time (geometric mean 3.9×),
    from 0.76× to 37×:
    - level or faster: the SPH solver steps up to 64 particles and the DST steps at 30 and 63 cells a side;
    - 26–37×: `create_particle_block`, which pays the freelist's large-block maps.
  - **Thread counts:** the SPH step gains 2.2–3.2× at 8 threads (Rust's rayon 1.05–1.95×); the grid steps gain
    1.03–1.14× on both sides.
  - **By milestone** (the per-part twins): the
  10 primitives against Rust's glibc, on shared seeded arguments. The correctly rounded functions run 58 ns (ln) to
  1.41 µs (atan2, whose mean comes from a rare slow path), 11–103× glibc. The `common` bodies run 4–13× Rust 1.3.0
  for validation, energy and the SoA / arena (36 µs to 206 µs per 1,000 particles), and 86× for building one
  particle (a call and a 144-byte struct return, against rustc's inlining). The part 4 functions run 4–64× Rust
  (pow's cost dominates the water viscosity), and JSON about 20× serde_json. The shallow-water step runs at 7.2×
  Rust's criterion numbers at every size, its volume 6.1–6.5× and the disturbance 1.4–1.6×. The grid's periodic
  step runs 2.7–4.0× Rust, diffuse 2.5–3.7× and max_speed 16–17×. The DST step, with part b's plans, runs 6.0×,
  1.03× and 0.85× Rust at 15, 30 and 63 cells a side when each step builds its plans (Rust's bench shape), and
  1.5×, 0.90× and 0.56× once a grid has stepped. The first 15 × 15 step pays 25 sines on the correctly rounded slow
  path (L1). Against Rust twins of part c's bodies (Rust has no criterion bench for them): the MAC step 2.2–2.9×,
  the ghost fluid 7.0–7.4×, the k-ε step 13–14× (a loop Rust vectorises); part d's VOF advect 8.1–8.7× and
  phase-field step 8.7–8.9×. The worker pool runs the periodic advection 4.1× faster at 8 threads (255 × 255); a
  whole grid step gains 1.07–1.15×, its serial Gauss–Seidel solves dominating, as Rust's rayon build gains nothing.
  The SPH kernels run 8–10.5× Rust, Tait 52× and the power law 27× (the correctly rounded pow). The 1,000-particle
  neighbour build runs 1.5× (2D) and 1.9× (3D) Rust on hisab's `HashMap`, and `sort_by_zorder` 2.0×. The
  brute-force sums run 4.9–9.5× Rust's criterion numbers and the free step 5.2–8.1×. The solver's WCSPH step runs
  1.24–2.3× Rust's criterion `solver_step` and `solver_step_st`, and 3.2–3.6× on a dense 625-particle block; PCISPH
  1.34–1.75× its criterion `pcisph_step`, and 4.7–6.3× on the dense block; DFSPH 1.46× (sparse) to 3.8–5.35× (dense)
  against a twin. `split_particle` into 8 pays
  the correctly rounded sin and cos's slow path at fl(k·π/2) (L1): 138 µs, against 3.5 µs into 7. M6 part a, against
  Rust's criterion groups: `couple_sph_bodies` 5.5–5.8×, `drag_from_particles` 7.9×, `particle_level_set` 4.2–4.3×;
  against a twin, the indexed drag 0.70× (625 particles; Rust's query walks a 3D block of cells) and 2.2× (10,000), the
  cells-only grid build 3.4–5.1× hisab's `HashMap` inserts. M6 part b: criterion `flip_step` 3.5–4.2× (after one cell
  lookup a particle and the helpers written out, 1.46–1.6× faster; `_pv_bits` is now `#inline`, which also takes the
  shallow-water step 5–7% faster). M6 part c, against a twin: an MPM step of the elastic block 8.7–9.1×, the fluid
  column 6.8×, the sand column 45× (about 22 correctly rounded calls a particle, L1). M7 part a, against a twin:
  packing 1,000 particles 5.2× (37 of its 52.6 µs the freelist's three large-block cycles), the unpacks 18×, the
  closed-form derivatives 8× and `dtait_drho` 50× (the correctly rounded pow), the autodiff gradient 4.1×, and an
  `SphKernelParams` read 25× serde_json (28× with an f32 midpoint, the L15 text path). M7 part b, against a twin: a
  disabled span 12 ns (Rust 1.56 ns), an enabled one 285 ns (17.3), a silenced warning 10 ns, the filter parse 3.4×,
  the `DaimonConfig` read 27×, write 19× and Debug 4.2×; the instrumentation leaves the median step at 0.995× the
  uninstrumented time (worst 1.02×; `merge_particles` +12 ns). A `defer` closing each span had cost the loops 8–11%,
  so every span closes explicitly.

## Upstream requests and stand-ins

The conversion ledger is in the port plan, §8: 15 rows, 14 open and 1 closed (L12: struct assignment already copies
records word-wise, so no record copy of pravash's own is needed). Each open row names its filed request. The
stand-ins so far:
- L1: `src/crmath.cyr`, `src/crlog.cyr` and their two table files;
- L2: `src/mathx.cyr` and `src/constants.cyr`'s std float constants;
- L3: `src/vec3.cyr`;
- L4 and L9: `src/neighbor.cyr` (the neighbour grid, and the cells-only grid `drag_from_particles_indexed` takes;
  the stable merge sort);
- L7: `_pv_invert_3x3` in `src/sph_fns.cyr` (the MLS 3×3 inverse);
- L5 and L6: `src/grid_solve.cyr` (the CG and the DST / IDST);
- L10: `src/json.cyr`;
- L11: `src/logging.cyr`'s Trace Event writer (a sakshi emit hook);
- L15: `_pv_jr_f32` in `src/json.cyr` (the f32 reads: RN32 of the f64, the decimal compared exactly at a midpoint);
- L13: `src/column.cyr`;
- L14: `_pv_fmod` in `src/mathx.cyr`;
- L16: `src/fmtf.cyr` and `src/debug.cyr` (Rust's float text and the derive's builders).

Each is marked. `scripts/upstream_ledger.py --check`: 19 notes, 0 failures.

- **M8 part d:** cyrius `2026-11-13-derive-debug-and-shortest-float-digits.md` (L16), with pravash's rows and the
  `debug_d` numbers (Cyrius 3.8–54× Rust's time).

Filed 2026-11-11:
- **hisab:** `2026-11-11-spatial-hash-per-step-rebuild-allocates.md`, `2026-11-11-no-matrix-free-linear-cg.md`,
  `2026-11-11-dst-allocates-per-call.md`;
- **ganita:** `2026-11-11-mat-inv-allocates-per-call.md`;
- **cyrius:** `2026-11-11-f64-min-max-drop-nan-only-as-first-operand.md`, `2026-11-11-no-stable-sort.md`,
  `2026-11-11-derive-json-no-vector-array-or-tagged-variant-shapes.md`,
  `2026-11-11-struct-values-and-f64v4-lanes-cost-in-pair-loops.md`;
- **sakshi:** `2026-11-11-no-chrome-trace-export.md`;
- **M7 part b:** pravash's stand-in notes appended to sakshi `2026-11-11-no-chrome-trace-export.md` (L11), and its
  note appended to hisab `2026-11-11-deps-sakshi-package-warns-every-consumer.md` (the deps warning);
- **M7 part a:** cyrius `2026-11-12-no-f32-decimal-parse.md` and bayan
  `2026-11-12-json-tree-number-keeps-no-source-text.md` (L15, each with a repro), and pravash's packing numbers
  appended to cyrius `2026-11-11-freelist-maps-every-block-over-4k.md`;
- **M6 part c, ganita:** pravash's MPM numbers appended to `2026-11-09-correctly-rounded-exp-sin-cos-hypot.md` (the
  fast paths' costs; Drucker–Prager sand 45× Rust on them);
- **M6 part a:** pravash's numbers appended to hisab `2026-11-11-spatial-hash-per-step-rebuild-allocates.md` (L4:
  the radius query's order differs between its two plans, which a sum over the result can't take; the stand-in's
  occupied-cell walk) and cyrius `2026-11-11-inline-refuses-small-numeric-helpers.md` (the distance helper,
  declined at seven parameters and at 33 tokens);
- **M4 part d, ganita:** a correction appended to `2026-11-11-mat-inv-allocates-per-call.md` (L7): `mat_inv` differs
  from the oracle on 12 of 60 inverse rows, so the stand-in transcribes hisab 1.4.0's `matrix_inverse`;
- **M4 part c, cyrius:** pravash's numbers appended to impetus's open
  `2026-11-09-field-store-of-struct-call-copies-a-field-argument.md` (30 of 30 merge rows; no stand-in, a local
  binding at the site, and surface parity's scan);
- **M4 part b:** the stand-ins' numbers appended to hisab `2026-11-11-spatial-hash-per-step-rebuild-allocates.md`
  (L4: the build against hisab 1.4.0's `HashMap`) and cyrius `2026-11-11-no-stable-sort.md` (L9: the merge sort's
  cost per comparison);
- **M3 part b, hisab:** pravash's DST-plan numbers appended to `2026-11-11-dst-allocates-per-call.md` (L6: a DST
  into caller scratch that takes a caller-kept plan);
- **M3 part a, ganita:** `2026-11-11-no-fmod.md` (L14, with a repro), and pravash's sin / cos slow-path numbers
  added to `2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`;
- **M2, cyrius:** `2026-11-11-inline-refuses-small-numeric-helpers.md` (the step's call overhead);
- **M1 part 4, bayan:** `2026-11-11-json-integer-negative-zero-loses-its-sign.md` (with a repro);
- **M1 part 3, cyrius:** `2026-11-11-vec-keeps-outgrown-buffer-and-aborts.md` (L13, with a repro) and
  `2026-11-11-freelist-maps-every-block-over-4k.md` (a performance finding, no stand-in); pravash's numbers on
  `2026-11-02-memcpy-memset-byte-loops-dominate-frame-copies.md`;
- **pravash's numbers added to:** ganita `2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`, hisab
  `2026-11-09-allocator-scoped-calls.md`, cyrius `2026-11-09-no-scoped-allocator-for-alloc.md`, and cyrius
  `2026-11-08-cyrius-port-clobbers-project-docs-and-omits-bench.md`.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib: syscalls, string, alloc, str, fmt, io, vec, math, ganita, tagged, result, freelist, fnptr, atomic, thread, bayan,
  chrono, sakshi
- hisab 3.4.1 (test scope), which also declares sakshi 2.5.10
- test stdlib (`[test] stdlib`, test / bench / fuzz builds only): assert, bench, overflow, simd
- local tools only (not the library, not CI): MPFR 4.x (`libmpfr.so.6`) for `scripts/oracle/validate.py`
- the generator (`scripts/oracle/`): the oracle by path, hisab 1.4.0, serde 1.0.229, serde_json 1.0.154
  (`float_roundtrip`), tracing 0.1.44 and tracing-subscriber 0.3.23, at the oracle's own versions

## Consumers

- **Cyrius:** none declares a pravash dependency yet.
  - goonj (Cyrius 2.3.1) is a bridge partner; ushma (Rust 1.4.0) and pavan (no local repo) are the others.
  - The `ai` configs point at daimon (Cyrius 2.5.6) and hoosh (Cyrius 2.12.3).
- **Rust (crates.io, `^1`):** badal, pavan, khanij, soorat, kiran. soorat is local; its optional `fluids` feature
  uses `sph` + `shallow`.
- **Intended:** kiran (water/smoke/fire), joshua (fluid simulation) and impetus (fluid–body interaction).

## Next

The port is done: 2.0.0 is ready to tag. What remains is the maintainer's:
- commit, then tag `2.0.0`: the release workflow checks VERSION and `cyrius.cyml` against the tag, runs CI, and
  publishes the CHANGELOG's `## [2.0.0]` section with the source tarball and checksums `cyrius deps` verifies;
- the 1.3.x deprecation notice on crates.io (port plan D1), pointing at the Cyrius library;
- the consumers' first Cyrius builds (kiran, joshua, impetus), on `[deps.pravash]` with `tag = "2.0.0"`.

After the release: the stand-ins convert as their libraries ship (port plan §8, 16 rows), and the 1.3.0 audit's
deferred physics items (roadmap) are 2.x work.
