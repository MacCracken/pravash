# Changelog

## [Unreleased]

### Added (Cyrius)
- **buoyancy, vortex, bridge, json** (M1 part 4, which completes M1).
  - **`src/buoyancy.cyr`:**
    - `buoyancy_force` and `drag_force`;
    - `terminal_velocity` and `reynolds_number`, which return an f64 `Result`;
    - `PvFlowRegime` with `classify_flow`.
  - **`src/vortex.cyr`:** `vorticity_2d`, `lamb_oseen_velocity`, `rankine_velocity`, `enstrophy` (Rust's `Sum`:
    an empty field is −0) and `kolmogorov_scale`.
  - **`src/bridge.cyr`:** the ten pavan / ushma / goonj conversions.
  - **Exact arithmetic:** each function follows Rust's operation order, and the four that call a transcendental use
    the correctly rounded primitives (ADR 0004).
  - **`src/json.cyr`:** serde-compatible JSON for `FluidMaterial`, `FluidParticle`, `FluidConfig`, `ParticleSoa`,
    `FlowRegime` and `DragCoefficient`.
    - `x.to_json(sb)` writes, and `PvX_from_json(buf, len, out)` reads with serde's derive semantics (both struct
      forms, unknown fields ignored, the u8 range, `null` refused for an f64).
    - Every serde_json document decodes to the same bits, and every Cyrius document round-trips. The digits are
      bayan's Grisu2, not serde_json's text (D4).
    - It is a marked stand-in (`UPSTREAM(L10)`). The new Cyrius-only code `PV_ERR_DESERIALIZE` (10) comes with ADR
      0002 #10.
  - **Parity:** 8,275 new golden rows in `buoyancy.tsv` (1,716), `vortex.tsv` (2,293), `bridge.tsv` (3,964) and
    `json.tsv` (302).
    - Every row is recomputed by `validate.py`. The class C functions use glibc for the record and MPFR for the
      corrected oracle, which gains `vortex.cr.tsv` (3) and `bridge.cr.tsv` (26).
    - All rows pass on the first replay, on three targets.
    - Three JSON documents that bayan's policies read otherwise than serde_json (S3, S4, the integer `-0`) are rows
      of their own, pending until bayan changes.
  - **Tests:**
    - all 45 Rust tests of the three modules;
    - `common.rs`'s two serde tests, so all 49 of its tests are ported;
    - 11 of `rust-old/tests/integration.rs`, in the new `tests/integration.tcyr`.
  - **Surface parity:** 128 items.

### Performance
- **The M1 part 4 benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md)), against
  Rust 1.3.0:
  - `terminal_velocity` 50 ns (Rust 3.6), `lamb_oseen_velocity` 136 ns (7.4), `enstrophy` of 1,000 values 3.0 µs
    (759 ns);
  - JSON about 20× serde_json: a config read 7.8 µs (329 ns), a 100-particle SoA read 572 µs (31.9 µs).

  The class C rows cost their correctly rounded primitive (ledger L1).

- **error, vec3, common** (M1 part 3): the shared fluid types, the port of `rust-old/src/error.rs` and
  `common.rs`.
  - **`src/error.cyr`:** the `PV_ERR_*` codes in Rust's variant order, append-only, with `pv_error_name` and
    `pv_error_message` (ADR 0002 #1).
  - **`src/vec3.cyr`:** `PvVec3` with glam 0.29.3's `dot`, `length`, `distance` and `is_finite`, in glam's order. It
    is a marked stand-in (`UPSTREAM(L3)`).
  - **`src/common.cyr`:**
    - `PvFluidMaterial`, with the five presets and `custom`;
    - `PvFluidParticle`, a 144-byte record laid out as Rust's;
    - `PvFluidConfig`, with `water_2d`, `validate` and `check_cfl`;
    - `PvParticleSoa`, 18 columns;
    - `PvArenaHandle` and `PvParticleArena`, whose free list is sorted and merged, whose handles go stale after
      `compact`, and which allocates nothing after `new`.
  - **`src/column.cyr`:** the freelist column the SoA's `Vec` fields become. It frees on growth and never aborts. It
    is a marked stand-in (`UPSTREAM(L13)`), because the stdlib `vec` keeps outgrown buffers and aborts on a bad
    index (filed).
  - **API shapes:** ADR 0002 rows 8 and 9.
    - Nothing aborts: a Rust panic becomes NaN, 0 or an empty view.
    - `Option<ArenaHandle>` is an out pointer plus 1 / 0, and a slice is (pointer, count).
    - Destructors are `_drop`.
  - **Parity:** the golden module `tests/golden/common.tsv`, 3,683 rows, 59 of them arena scenarios.
    - `validate.py` recomputes every row, with a Python transcription of the arena.
    - All rows are bit-exact on the first replay, on x86_64 ELF, x86_64 PE and aarch64.
    - The replay is 34,454 ok and 43 accepted (ADR 4).
  - **Tests:** `tests/common.tcyr` ports 47 of `common.rs`'s 49 Rust tests; the two serde tests come with part 4's
    JSON. `tests/error.tcyr` ports the six error tests.
  - **Surface parity:** `scripts/surface_parity.py` (impetus's) checks every public Rust item of the ported modules
    against its Cyrius name (98 items), and now runs in CI.

### Performance
- **The `common` benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M1 part 3),
  against a Rust 1.3.0 twin (`scripts/probes/rust/src/bin/bench_common.rs`).

  | Bench | Rust 1.3.0 | Cyrius |
  |---|---|---|
  | `soa/from_aos_1000` | 16.9 µs | 206 µs |
  | `soa/write_to_aos_1000` | 6.5 µs | 36 µs |
  | `arena/cycle_1000` | 8.3 µs | 110 µs |
  | `common/config_validate` | 12.5 ns | 64–100 ns |

  Two changes, measured A/B (pinned, interleaved):
  - **Record resets** copy one default record: `arena/cycle_1000` from 163–183 µs to 109–126 µs.
  - **The SoA conversions** move words straight between records and columns: `from_aos` 1.8×, `to_aos` 1.9×,
    `write_to_aos` 3.6× faster.

  The rest is the freelist mapping each block over 4 KB, which is filed upstream.
- **Ledger L12 closed, no stand-in needed.** A struct assignment copies a 144-byte particle in 11.7 ns, against
  `memcpy`'s 245 ns. Every particle move is an assignment. The numbers are on the open cyrius `memcpy` issue.

- **crmath and crlog** (M1 part 2, [ADR 0004](docs/adr/0004-correct-rounding-is-the-accuracy-target.md)): every
  std transcendental pravash calls is now correctly rounded, the same bits on every target. Each file is a marked
  stand-in (`UPSTREAM(L1)`) until ganita's correctly rounded tier ships.
  - **The code:**
    - `src/crmath.cyr`: impetus's exp, sin, cos, sin_cos, hypot, pow and atan2;
    - `src/crlog.cyr`: kimiya's ln and log10;
    - their tables `src/crmath_data.cyr` and `src/crlog_data.cyr`, generated by `scripts/oracle/crtables.py` and
      `lntables.py` and checked against MPFR;
    - `src/mathx.cyr` wraps them as `_pv_exp`, `_pv_ln`, `_pv_log10`, `_pv_powf`, `_pv_sin`, `_pv_cos`,
      `_pv_sin_cos`, `_pv_hypot` and `_pv_atan2`, plus `_pv_cbrt` (ganita's, already exact).
  - **The corrected oracle:** `tests/golden/math.cr.tsv`, the 43 math rows glibc 2.44 misrounds (21 log10,
    11 hypot, 3 sin, 3 sin_cos, 2 atan2, 1 each of exp, ln and powf), with MPFR's values.
    - `scripts/oracle/validate.py --write` writes it and the two table files; a plain run checks all three.
    - MPFR comes through `scripts/oracle/mpfr_ref.py`.
  - **Parity:** K = 0 for every class C math function, powf and log10 included.
    - The replay is 30,771 ok and 43 accepted (ADR 4), with 0 failed and 0 measured, identical on x86_64 ELF,
      x86_64 PE and aarch64.
    - The harness core is now impetus's current version, which compares corrected rows.
    - Mutation-checked: a 1-ulp exp fails 164 rows; a one-bit table edit fails rows and `validate.py`.
  - **Tests:** `tests/crmath.tcyr`, 85 checks covering the specials, the exact cases and ties, the hard-to-round
    arguments, and the agreement between fast and slow paths, on all three targets.

### Performance
- **The first benchmark record**, [`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md):
  `tests/pravash.bcyr`'s `math/*` rows against a Rust twin (`scripts/probes/rust/src/bin/bench_libm.rs`), on the same
  1,024 seeded arguments per function, pinned.

  | Function | Correctly rounded (Cyrius) | Counterpart it replaced (Cyrius) | Rust (glibc) |
  |---|---|---|---|
  | ln | 58 ns | 62 ns | 4.64 ns |
  | log10 | 94 ns | — (none existed) | 8.54 ns |
  | exp | 110 ns | 55 ns | 4.85 ns |
  | sin_cos | 228 ns | 82 ns | 12.3 ns |
  | hypot | 173 ns | 33 ns | 6.23 ns |
  | powf | 757 ns | 212 ns | 12.2 ns |
  | atan2 | 1.41 µs | 63 ns | 13.6 ns |

  atan2's mean comes from its slow path: 302 µs a call, on 0.4 % of arguments. Phase 1 alone is 349 ns.

- **mathx** (M1 part 1): `src/mathx.cyr`, the library's first module after the generated constants. It holds
  pravash's Rust-semantics f64 helpers: `_pv_min` / `_pv_max` / `_pv_clamp` (NaN-ignoring, Rust's ±0 rule),
  `_pv_is_nan` / `_pv_is_finite`, `_pv_to_radians`, `_pv_powi2` / `_pv_powi8` in LLVM's order, and the saturating
  `_pv_as_i32` / `_pv_as_u32` / `_pv_as_usize`. It is a marked stand-in (`UPSTREAM(L2)`).
  - **Build:** the first `[lib]` in `cyrius.cyml` (constants, mathx), the bundle `dist/pravash.cyr`, and
    `tests/lib.cyr` for test compiles.
  - **Parity:** 6,371 golden rows newly gated, all bit-exact. The replay is 28,223 ok, 0 failed, identical on
    x86_64 ELF, x86_64 PE and aarch64. Mutation checks: the ±0 rule fails 2 rows, unsaturated negatives 111.

### Changed
- **The Cyrius port begins** (port plan M0 steps 1–3; ADRs 0001–0003). `cyrius port` moved the Rust 1.3.0 crate,
  with its `oracle` hooks and `Cargo.lock`, into `rust-old/` as the frozen parity oracle (commit `a88bc4e`); it
  still passes there. The repairs:
  - the Cyrius manifest: `cyrius.cyml` reads `VERSION`, `[build] dce`, the stdlib leaves, hisab 3.4.1 test-scoped;
  - CI: the Cyrius workflow runs build, test, `test --bounds`, `deps --locked` and `audit`, plus a job that keeps
    the frozen oracle green with and without `parallel`;
  - docs: CLAUDE.md merged for Cyrius; `state.md`; ADR 0001 (port with a Rust oracle), 0002 (API differences)
    and 0003 (local stand-ins for library gaps); CONTRIBUTING and SECURITY for Cyrius; a port banner in the README;
  - archived in `rust-old/`: the Rust workflows, scripts, architecture overview and 1.2 → 1.3 migration guide.

  The library itself is still the Rust 1.3.0 crate: no Cyrius module exists yet.

### Added
- **Semantics probe** (port plan M0 step 4): `scripts/probes/p0_semantics.cyr`, its Rust twin `p0_semantics.rs`
  and `p0_compare.py` compare the bits of the Cyrius counterpart of every f64 / f32 operation pravash uses
  (`docs/development/port-parity.md` §1).
  - **Bit-identical:** the f32 SPH cell keys (including saturation and NaN), floor / ceil / sqrt / abs, ln,
    sin / cos, hypot, cbrt, `to_radians` and `powi`.
  - **Need helpers:** `f64_min` / `f64_max` (asymmetric NaN), and `f64_to`, which doesn't saturate as Rust's
    `as i32` / `as u32` / `as usize` do.
  - **Correctly rounded stand-ins needed:** log10 (there is none), and ganita's pow and atan2 and the exp
    polyfill (1 ulp off).
- **Golden-value generator** (port plan M0 step 5): `scripts/oracle/`, adapted from impetus's. It drives the frozen
  oracle and writes `tests/golden/`: `constants.tsv` (29 rows) and `math.tsv` (33,239 rows).
  - **Format and determinism:** TSV format 1, every float as its bits. Two runs give the same bytes, and the
    `parallel` build reproduces the corpus but may only `--check` it.
  - **`--check`:** refuses a corpus recorded on another glibc.
  - **`validate.py`:** recomputes every row without the oracle. Class A/B in Python; glibc's class C through
    ctypes; `cbrt` proven correctly rounded.
  - **Finding:** the oracle's `cbrt` is compiler_builtins', not glibc's (port-parity §3).
- **`src/constants.cyr`** (port plan M0 step 6), generated by `scripts/oracle` (never edited by hand). It holds
  the 14 public constants (`PV_FLUID_MATERIAL_*`, `PV_DRAG_COEFFICIENT_*`, `PV_ADDED_MASS_COEFFICIENT_*`) and the
  6 Rust std float constants the code uses (`PV_F64_MAX`, `PV_F64_INFINITY`, `PV_F64_NEG_INFINITY`, `PV_F64_NAN`,
  `PV_F64_PI`, `PV_F32_EPSILON`).
  - **Proofs:** both lists are proven complete against `rust-old/src`, and `constants.tsv` pins all 35 values.
  - **`scripts/oracle/probe_tables.py`:** compiles each value and confirms its bits; a 2-ulp change to one literal
    fails it.
- **Parity harness** (port plan M0 step 7): `tests/parity.tcyr` replays `tests/golden/` against the port, with
  impetus's `tests/parity_core.cyr` and `tests/ulp.cyr`. The constants rows go through the generated
  `tests/constants_rows.cyr`.
  - **First run:** 0 failing; 5,149 ok, 0 failed, 18,493 measured (class C, ≤ 2 ulp, bounds at step 8), and
    9,632 pending (the M1 helpers). It is warning-free, and the same under `--bounds`.
  - **Finding:** ganita's cbrt equals the oracle's on all 1,025 rows, so cbrt needs no stand-in.
  - **Mutation-checked:** a flipped golden bit, a 1-ulp constant, and a mis-wired dispatcher each fail their row.
- **Cross-target baselines** (port plan M0 step 8): per-function digests in `tests/parity.tcyr`. Every dispatched
  function gives identical bits on x86_64 ELF, x86_64 PE (wine) and aarch64 (qemu); only the native x87 shadows
  differ, by design.
  - **Bounds:** K = 2 for sin, cos, sin_cos, atan2, exp, ln and hypot (measured max 1, hypot 2), and K = 0 for
    cbrt. powf stays measured until its correctly rounded M1 counterpart.
  - **Gated:** 21,852 ok, 0 failed on all three targets. Both bounds are mutation-checked.
- **Probes** (port plan M0 step 9; `scripts/probes/`, `docs/development/port-parity.md` §4):
  - **Memory:** every hisab or ganita return leaks on the step path. The neighbour-hash rebuild leaks 4.1 MB per
    1,000-particle step, `num_dst` 10–20 KB per call, and the 3×3 `mat_inv` 248 B. Freelist and a step arena
    are flat.
  - **Layout:** all four forms give the same bits. AoS with f64 locals is fastest (12.9 ns a pair); struct
    values are 2.5× slower and f64v4 3.5×.
  - **f32 keys:** 10⁶ of 10⁶ match Rust on x86_64 ELF, x86_64 PE and aarch64.
  - **Neighbour order:** (cell offset, insertion); at |x| ≈ 10⁶, 89% of the probe points miss their cell, the
    defect D3 fixes.
  - **Sort ties:** Rust's ipnsort leaves a different order at 75–94% of positions on tie-heavy inputs, so the
    stable (key, index) order applies (ADR 0002 #3).
  - **Parallel:** the Rust oracle's serial and parallel builds give the same digests on 14 scenarios. A Cyrius
    persistent pool gives the serial bits at 1–8 threads, allocates nothing, and is 3.9× at 8 for 10⁵ particles.
  - **Bundle:** DCE takes a full-hisab consumer from 3.06 MB to 677 KB, so there is one bundle (ADR 0002 #5).
  - **Libraries vs hisab 1.4.0:** hisab 3.4.1's DST is another algorithm, so ledger L6 transcribes 1.4.0's.
    ganita's `mat_inv` gives the oracle's bits on all 300 matrices, so L7 is needed for memory only. The
    autodiff tape is within 3 ulp, so it stays on hisab as class D.
- **Upstream requests and the conversion ledger** (port plan M0 step 10; ADR 0003). Each request was filed with
  pravash's numbers and, where the gap can be shown, a self-proving repro verified from the owning repo's root:
  - **hisab:** a non-allocating spatial-hash query (4.1 MB per 1,000-particle step today), a matrix-free linear CG,
    and an allocation-free DST;
  - **ganita:** `mat_inv_into`;
  - **cyrius:** NaN-ignoring min / max with the Rust f64 surface, a stable sort, `#derive` array and tagged
    shapes, and the struct-value / `f64v4` costs;
  - **sakshi:** a chrome://tracing export;
  - **existing issues:** pravash's numbers added to ganita's correctly rounded tier, hisab H9, cyrius S1 and the
    `cyrius port` clobber issue.

  The ledger (plan §8) is seeded with 11 open rows and 1 conditional. `scripts/upstream_ledger.py --check`, run in
  CI, ties every `UPSTREAM(Ln)` note in `src/` to an open row, every existing stand-in to its note, and every row
  to its filed request. **M0 is complete.**

## [1.3.0] - 2026-11-09

Toolchain/dependency refresh plus a full correctness audit of every module.
Most fixes change numerical results — read **Behavior changes** before
upgrading. No public signatures were removed or changed.

### Behavior changes (migration)
- **grid**: pressure projection now actually removes divergence for any `dx` (it amplified divergence ~1/dx² for dx < 1). Simulations at dx ≠ 1 behave very differently — re-tune forces tuned against the old behaviour.
- **grid**: `FreeSlip` ghost cells now reflect the normal velocity (`ghost = −interior`, zero at the wall face) like `NoSlip`, instead of setting the ghost to 0.
- **sph**: surface tension is now Müller's `−σ·∇²c·n̂/ρ` (was divided by an extra `|n|`) and only acts where `|∇c| > 0.1/h`; coefficients tuned for 1.2 are ~1/h too strong now.
- **sph**: `update_heat` no longer divides by ρ a second time — conduction is ~ρ× faster; use a physical diffusivity (water ≈ 1.4e-7 m²/s).
- **sph**: `update_phase_change` now accumulates latent heat across steps in `fuel` (progress toward the other phase, reset on transition) — don't combine with combustion.
- **sph**: `apply_contact_angle` applies `σ·cos θ/h` toward the wall for wetting (θ < 90°), none at 90°, away for θ > 90°; no longer divides by mass and never acts on degenerate (2D) axes.
- **sph**: PCISPH always runs ≥ 1 iteration and converges on *compression* only.
- **shallow**: continuity flux was halved (waves ran at √(gh/2)); waves now travel at √(gh).
- **shallow**: `breaking_threshold` is now a dimensionless surface slope `|∇η|` (was `|∇η|/depth`, units 1/m).
- **shallow**: `dispersion_coeff` now drives an implicit Peregrine-form filter (`ω² = ghk²/(1+c·h²k²)`, stable for any dt) instead of an explicit 4th-order term that grew exponentially; `use_green_naghdi` selects the depth-weighted operator.
- **shallow**: domain edges are closed walls (no mass leaks into ghost cells).
- **coupling**: drag is quadratic (`½ρ|v|²CdA`, was linear in |v|); added-mass integration applies gravity to the body mass only; FLIP projection is dx-correct; P2G is mass-weighted.
- **mpm**: the `Fluid` stress sign is fixed (compression now pushes back); `DruckerPrager` now has real plasticity with `friction_angle` in **degrees**; `step()` validates every particle's material and volume and returns `Err` instead of producing NaN; grid walls are separating and particles stay in the domain.
- **vortex**: `lamb_oseen_velocity` with `ν·t = 0` returns the point-vortex speed Γ/(2πr) (was 0); `kolmogorov_scale(0, ε)` returns 0 (was ∞).
- **buoyancy**: `reynolds_number` uses |v| and rejects NaN viscosity; `classify_flow` uses |Re|; `terminal_velocity` rejects non-positive ρ, Cd or A individually.

### Changed
- **toolchain**: MSRV 1.89 → 1.99 (edition 2024, latest stable); CI MSRV job, README and CLAUDE.md updated.
- **deps**: hisab 1.1 → 1.4, serde 1.0.229, thiserror 2.0.21, tracing 0.1.44, tracing-subscriber 0.3.23, tracing-chrome 0.7.2, rayon 1.12, serde_json 1.0.154 (dev), criterion 0.5 → 0.8 (dev; benches use `std::hint::black_box`).
- **ai**: the feature no longer pulls `reqwest`, `tokio` (`full`) or `serde_json` — none were used. Lockfile: 214 → 93 packages, no OpenSSL/native-tls.
- **compute**: feature now enables `sph` (it calls the SPH kernels; `--no-default-features --features compute` failed to build).
- **ci**: actions/checkout v4 → v7, codecov-action v4 → v7, action-gh-release v2 → v3.
- **scripts**: `bench-history.sh` now records µs-scale results and names printed on their own line (it silently dropped them — 19 of 57 benchmarks were recorded).
- **deny**: dropped the unused BSD-3-Clause allowance.
- **docs**: docs.rs builds with all features; crate-level module list completed; stale doc claims fixed (Periodic "FFT solve", MacCormack "fallback", k-ε "transport", MPM fluid viscosity).

### Fixed
#### grid
- Projection RHS is now `−dx²·∇·u` for the unit-stencil Poisson system (all four solvers).
- DST and CG solved the opposite-sign system to Gauss-Seidel; all solvers now solve `4p − Σp = rhs` (CG on the SPD operator, relative tolerance).
- Multigrid: residual sign, missing ×4 coarse RHS, and one-sided prolongation fixed; V-cycles now converge, and `pressure_iterations/10` cycles are run.
- MacCormack/BFECC limiter clamped to the destination cell's neighbours (erased features moving > 1 cell/step); now clamps to the backtraced stencil. "BFECC" was MacCormack — it now advects the error-compensated field forward.
- `MacGrid::step`: closed walls, buoyancy only on interior faces (no accumulation on wall faces), Neumann pressure over every cell, density advection (was documented but missing), sizes validated; `cfl_dt` returns `f64::MAX` without flow instead of a length.
- `FluidGrid::step` re-validates sizes (Deserialize/pub fields), rejects NaN `smagorinsky_cs`/`vorticity_confinement`/buoyancy, and errors on non-finite velocity after projection; `cfl_dt` saturates instead of returning ∞.
- `diffuse`, `KEpsilon::step`, `apply_ghost_fluid` ignore mismatched sizes (warning) instead of panicking.
- VOF: closed walls (no leak into the ring), sub-cycled to Courant ≤ 0.5, persistent scratch buffer, slice lengths and non-finite velocity checked.
- Phase field: upwind advection (central + forward Euler is unconditionally unstable), one divergence warning per step (was per cell), wrong-length velocity slices are an error.
#### sph
- DFSPH rewritten in Bender & Koschier order (density solver → advect → divergence solver) with the correct sign of Dρ/Dt, `κ/ρ` scaling, κ computed for all particles first, compression-only correction, and persistent buffers.
- Delta-SPH diffusion had the wrong sign (amplified density noise).
- PCISPH: `max_iterations = 0` teleported particles to the origin; the final integration used one-iteration-old pressures; δ came from an average dominated by sparse particles (now a full-neighbourhood prototype).
- `apply_implicit_viscosity` was K explicit steps; it is now a backward-Euler Gauss-Seidel solve.
- Viscoelastic Oldroyd-B used ∇uᵀ in one term (C_yy drifted in simple shear).
- `use_verlet` was first-order (lagged acceleration, zero first step); it is now velocity Verlet (exact under constant acceleration). New `SphSolver::reset_integrator()` for after reordering.
- Colour-field Laplacian includes the self term; multiphase interface tension uses its own colour field and only acts between phases; out-of-range phases warn once per step; empty `phases` is an error (was a panic).
- `adaptive_dt` with inverted bounds panicked in `f64::clamp`.
- Public kernels return 0 for `h <= 0` (NaN in release); `equation_of_state_tait` handles γ → 0; `create_particle_block` with spacing ≤ 0 returned an endless allocation; `split_particle(.., 1, ..)` moved the centre of mass; `sort_by_zorder` mis-sorted negative coordinates.
- Free functions taking neighbour lists validate them (warning) instead of panicking.
#### shallow
- Riemann path is a proper finite-volume HLL update of (h, hu, hv) with Audusse hydrostatic reconstruction (well-balanced) and mirrored-wall states; it no longer double-counts the pressure gradient and advection. HLL uses dry-bed wave speeds.
- Convective terms are upwinded (centred + forward Euler was unconditionally unstable).
- Breaking detection ignores dry neighbours.
- Sediment: suspended volume `C·h` is conserved through erosion/deposition (bed can no longer rise above the surface); uses `dry_threshold`; validates `dt`/lengths.
- `new()` validates `dx` and height; `step()` validates sizes/gravity/dispersion and errors on non-finite state.
#### coupling
- FLIP pressure solve missing `dx²` (over-corrected ~100× at dx = 0.1).
- `drag_from_particles_indexed` queried only `sample_radius` around the centre and missed bodies larger than the radius; stale hash indices are skipped instead of panicking.
- P2G weights could go negative for particles outside the grid.
- APIC `C` used the quadratic-B-spline factor with bilinear weights; it is now `Σ vᵢ∇wᵢᵀ` (exact for linear fields).
- `flip_ratio` NaN is an error; `particle_level_set` ignores a mismatched buffer.
#### common / compute / misc
- `ParticleArena`: watermark allocations returned stale particles; double free and stale (post-`compact`) handles corrupted the free list or aliased blocks; foreign handles panicked; `alloc(usize::MAX)` overflowed. Handles now carry a generation.
- `ParticleSoa::to_aos`/`write_to_aos` and `PackedParticles::from_soa`/`unpack_*` panicked on mismatched lengths.
- `FluidMaterial::custom`, `FluidConfig::validate`, `check_cfl` accepted NaN/Inf (and negative velocities).
- `KernelDerivatives::dtait_drho` NaN at γ = 0; `dpoly6_dr2(0, 0)` NaN.
- `bridge::heat_to_buoyancy_forcing` NaN at density 0 (ρ cancels).

### Added
- `SphSolver::neighbor_lists()` (the free neighbour-list functions were unusable without it), `SphSolver::reset_integrator()`.
- Constructors for `#[non_exhaustive]` configs that external crates could not build: `HeatConfig::new`, `ViscoelasticConfig::new`, `PhaseChangeConfig::new` / `water_ice`, and `Default` for `SphKernelParams`/`GridKernelParams`/`ShallowKernelParams`.
- `ShallowWater::interior_volume()` (the conserved quantity), `ParticleSoa::is_consistent`/`consistent_len`, `BodyShape::bounding_radius`/`projected_area`.
- **oracle** (Cyrius port tooling, not public API): a hidden, non-default `oracle` feature, kept out of `full`, gives `scripts/oracle/` read-only access to state the public API doesn't expose (port plan M0 step 0). It covers:
  - `oracle_dump()` on every stateful type: `ParticleArena`, `ArenaHandle`, `ShallowWater`, `FluidGrid`, `KEpsilon`, `MacGrid`, `VofField`, `PhaseField`, `FlipSolver`, `MpmSolver` and `SphSolver`. The `SphSolver` dump includes the f32 positions and the neighbour lists in summation order;
  - `oracle::sph_cell_key`;
  - the four grid pressure solvers on a caller's system (`pressure_solve_gauss_seidel`, `_dst`, `_cg`, `_multigrid`);
  - `oracle::invert_3x3`, which is hisab 1.4.0's `matrix_inverse` as the MLS correction calls it.

  The dumps live in child modules (`src/<module>/oracle.rs`), so each audited source gains only a `mod oracle;` declaration. `tests/oracle.rs` (14 tests) runs every solver variant 300 steps plain and 300 steps dumping after each step, and requires bit-identical final states. Its harness was mutation-checked: one extra step in the dumped run fails all 10 solver checks. CI runs it with and without `parallel`. Simulation behaviour is unchanged.

### Changed (port tooling)
- `Cargo.lock` is tracked for the Cyrius port's freeze: the oracle's bits depend on hisab 1.4.0 and glam 0.29.3. This is an exception to the library-crate rule, to be recorded in the port's ADR 0001. The oracle host's glibc is 2.44.

### Security
- `DaimonConfig`'s `Debug` redacts `api_key`.

### Performance
- No speedups are claimed. The full suite (`bench-history.sh`, labels `baseline-1.2.0` → `post-audit-1.3.0`, 57/57 benchmarks) ran on a shared, loaded host and moved ±100% even on unchanged code (`shallow_disturbance`), so it is not used as evidence.
- Pinned A/B (`taskset -c 15`, old/new interleaved, 2 runs each, 2 s measurement) shows parity within run-to-run noise (±20%): `grid_step/63x63` 10.8/13.2 → 13.3/10.5 ms, `grid_step_maccormack/63x63` 12.3/12.9 → 13.8/10.7 ms, `pcisph_step/64` 64.3/68.0 → 80.1/61.5 µs, `solver_step/256` 344/363 → 365/287 µs, `solver_step_st/169` 206/204 → 194/197 µs, `shallow_step/128x128` 546/543 → 567/519 µs, `flip_step/225` 25.0/19.8 → 22.8/28.3 µs.
- Standalone kernels: the new `h > 0` guard first cost ~0.4 ns (+15–30%); reduced to a single comparison, now at parity — `sph_kernels/poly6` 2.35/2.10 → 2.00/2.21 ns, `spiky_grad` 2.11/1.85 → 1.57/1.65 ns, `viscosity_laplacian` 1.53/1.72 → 1.48/1.61 ns.

## [1.2.0]

### Added
- **bridge** — cross-crate primitive-value bridges for pavan (Cp to pressure, dynamic pressure, aero forces, freestream velocity), ushma (temperature to water viscosity, heat to buoyancy forcing, TKE to eddy diffusivity), goonj (velocity to acoustic source strength, TKE to broadband noise, wave height to acoustic pressure)

### Updated
- hisab 1.1.0 -> 1.3.0, zerocopy 0.8.47 -> 0.8.48, iri-string 0.7.11 -> 0.7.12

## [Unreleased-pre-1.2]

### Fixed
- sph: Surface tension div-by-zero in solver step, PCISPH, and multiphase — added threshold floor
- sph: Phase change energy conservation — added `heat_capacity` to `PhaseChangeConfig`, proper J/kg↔K conversion
- sph: Verlet integration used zero prev_accel on every step — fixed resize guard to preserve existing values
- sph: PCISPH delta computed from particle 0 only — now averages over all particles with neighbors
- sph: Implicit viscosity solver ran fixed iterations — added convergence check with early exit
- sph: Contact angle force missing length scale — added smoothing radius divisor
- sph: Non-Newtonian viscosity (Bingham/Herschel-Bulkley) could produce extreme values — added saturation cap
- sph: Multi-phase surface tension early exit used wrong condition — fixed to check `st > 0 || it > 0`
- grid: DST eigenvalue division threshold too small (1e-20 → 1e-12)
- grid: NaN from `sample()` propagated silently — added post-advection divergence check
- grid: Multigrid recursion had no depth limit — capped at 10 levels
- shallow: Riemann solver epsilon too small (1e-10 → 1e-8)
- shallow: HLL flux mixed old/new time levels — now uses consistent snapshot velocities
- shallow: Friction calculation NaN risk near dry threshold — added depth guard
- shallow: Well-balanced pressure gradient used hardcoded 1e-3 — now uses dry threshold
- shallow: Grid minimum size was 1 — raised to 3
- shallow: Sediment array bounds only checked in debug — added runtime check
- coupling: FLIP pressure solver ran fixed 40 iterations — added convergence check with early exit
- coupling: APIC silently degraded to PIC on first step — auto-resize apic_c
- coupling: NaN could propagate through FLIP step — added post-pressure-solve check
- mpm: Deformation gradient could explode — added det(F) clamping
- mpm: Particle/mpm_data count mismatch silently truncated — now returns error
- common: `ParticleArena::active_count` could underflow — switched to `saturating_sub`
- vortex: `lamb_oseen_velocity` div-by-zero when viscosity=0 — added guard
- buoyancy: `reynolds_number` and `terminal_velocity` accepted NaN/Inf inputs — added validation

### Added
- mpm: `ConstitutiveModel::validate()` — validates material parameters (Young's modulus, Poisson ratio, bulk modulus)
- grid: `GridConfig` validation — smagorinsky_cs, vorticity_confinement, iteration counts
- grid: BFECC advection wired up via `use_bfecc` flag
- vof: `is_surface_with_threshold()`, `is_full_with_threshold()`, `is_empty_with_threshold()` — configurable threshold
- grid: Conjugate gradient pressure solver via hisab (`GridConfig.use_cg`) — matrix-free, SPD-optimized
- coupling: `drag_from_particles_indexed()` — spatial-hash-accelerated drag using hisab `SpatialHash`

### Changed
- sph: MLS 3×3 matrix inversion now delegates to `hisab::num::matrix_inverse` (LU-based)
- sph: Density threshold standardized via `MIN_DENSITY` constant across all solvers
- sph: `MultiPhaseConfig::get()` now logs warning on invalid phase index fallback
- grid: All pressure solvers now return `Result<()>` consistently
- grid: k-epsilon solver uses scratch buffer swap instead of cloning
- vortex: `kolmogorov_scale` uses log-form for numeric stability
- phase_field: Phase field clamping now warns via tracing when |φ| > 1.5
- shallow: CFL condition accounts for dispersive stability limit

## [1.1.0] - 2026-03-26

### Added
- sph: Wendland C2/C4 kernels (`kernel_wendland_c2`, `kernel_wendland_c4`) — no tensile instability
- sph: Delta-SPH density diffusion (`apply_delta_sph`) — pressure noise reduction
- sph: Velocity Verlet integration (`SphSolver.use_verlet`) — 2nd-order symplectic
- sph: Non-Newtonian viscosity (`NonNewtonianViscosity`) — power-law, Bingham, Herschel-Bulkley
- sph: Z-order particle sorting (`sort_by_zorder`) — Morton code for cache-friendly access
- sph: Enhanced `adaptive_dt` with force and viscous CFL constraints
- grid: `FluidGrid::cfl_dt()` — advective + viscous CFL timestep
- grid: Smagorinsky SGS turbulence model (`GridConfig.smagorinsky_cs`)
- sph: `ReactionProvider` trait for pluggable chemistry backends (kimiya, etc.)
- sph: `update_reaction()` — generic reaction step using any `ReactionProvider`
- sph: `CombustionConfig` implements `ReactionProvider` for backward compatibility
- sph: DFSPH solver (`step_dfsph`) — divergence-free SPH, 2-stage correction
- sph: MLS gradient correction (`compute_gradient_corrections`, `apply_gradient_correction`)
- grid: Multigrid V-cycle pressure solver (`GridConfig.use_multigrid`)
- shallow: HLL Riemann solver for flux computation (`ShallowWater.use_riemann`)
- coupling: APIC transfers (`FlipSolver.use_apic`) — affine particle-in-cell, angular momentum conserving
- All vector types migrated from `[f64; 3]` to `hisab::DVec3` (SIMD-accelerated)
- grid: Staggered MAC grid (`MacGrid`) — velocity at faces, pressure at centers, no checkerboard
- grid: Ghost fluid method (`apply_ghost_fluid`) — pressure jump at multi-phase interfaces
- vof: Volume of Fluid (`VofField`) — free surface tracking with donor-acceptor advection
- sph: Contact angle / wetting (`apply_contact_angle`) — Young's equation at solid boundaries
- sph: Phase change (`PhaseChangeConfig`, `update_phase_change`) — melting, solidification, evaporation
- shallow: Sediment transport (`SedimentConfig`, `update_sediment`) — Shields criterion, erosion/deposition
- mpm: Material Point Method (`MpmSolver`, `MpmParticle`) — neo-Hookean, fluid, Drucker-Prager constitutive models
- sph: Tait equation of state (`equation_of_state_tait`) for compressible flow
- sph: Implicit viscosity solver (`apply_implicit_viscosity`) — removes viscous CFL restriction
- sph: Particle splitting (`split_particle`) and merging (`merge_particles`) with mass/momentum conservation
- sph: Batch kernel evaluation (`batch_poly6`, `batch_distance_squared`) — SIMD-friendly patterns
- coupling: Narrow-band FLIP option (`FlipSolver.narrow_band_cells`)
- compute: `NeuralCorrector` trait for learned simulation correction (FNO, GNS backends)
- compute: `KernelDerivatives` — analytical Jacobians for differentiable simulation (dPoly6/dr², dEOS/dρ, dTait/dρ)
- grid: k-epsilon turbulence model (`KEpsilon`) — two-equation RANS with production/dissipation transport
- grid: BFECC advection (`advect_bfecc`) — 2nd-order error-compensating advection
- shallow: Green-Naghdi fully nonlinear dispersion (`use_green_naghdi`)
- sph: Foam/spray/bubble generation (`SecondaryParticle`, `generate_secondary_particles`, `update_secondary_particles`)
- compute: `KernelDerivatives::autodiff_gradient()` — reverse-mode autodiff via hisab's `Tape`/`Var`
- hisab dependency updated to 1.1 (reverse-mode autodiff, stiff ODE solvers, SDFs, k-epsilon eigendecomposition)
- shallow: `ShallowWater::cfl_dt()` — wave speed CFL timestep

### Previously Added
- shallow: Non-linear shallow water equations with convective acceleration (u·∇u)
- shallow: Flux-form continuity equation (∂(hu)/∂x + ∂(hv)/∂y) for proper mass transport
- shallow: Neumann boundary enforcement (zero-gradient height, reflected normal velocity)
- shallow: Manning's bed friction (`manning_n` per-cell roughness) with implicit treatment for stability
- shallow: Wetting/drying transitions (`dry_threshold`, `is_wet()`) with velocity zeroing, depth clamping
- shallow: Wave breaking detection (`breaking_threshold`, `is_breaking()`) with turbulent bore dissipation
- shallow: Boussinesq dispersive correction (`dispersion_coeff`) for short-wavelength accuracy
- shallow: Well-balanced hydrostatic reconstruction for terrain-following pressure gradient (lake-at-rest preserved over sloped bathymetry)
- sph: Rayon parallelism for SphSolver density and force loops (`parallel` feature gate)
- grid: Rayon parallelism for advection and pressure projection (`parallel` feature gate)
- common: `ParticleSoa` — SIMD-friendly Structure-of-Arrays layout with AOS↔SOA conversion
- compute: GPU-agnostic `ComputeBackend` trait, `PackedParticles` f32 buffer packing, kernel parameter structs for SPH/grid/shallow
- common: `ParticleArena` — pre-allocated pool with alloc/free/compact, `ArenaHandle` for block references
- logging: `init_profiling()` — chrome://tracing JSON output for flame graph analysis via tracing-chrome
- sph: Multi-phase SPH (`step_multiphase`, `MultiPhaseConfig`, `PhaseProperties`) with per-phase EOS, viscosity, and interface tension
- common: `FluidParticle.phase` field (u8) for multi-phase particle identification
- phase_field: `PhaseField` — Allen-Cahn interface tracking with advection, double-well potential, Neumann BCs
- sph: Viscoelastic fluids (`ViscoelasticConfig`, `update_viscoelastic`) — Oldroyd-B conformation tensor evolution with elastic stress
- sph: Heat transfer (`HeatConfig`, `update_heat`) — SPH Laplacian-based conduction with implicit advection
- common: `FluidParticle.temperature` field (default 293.15 K)
- sph: Combustion reaction (`CombustionConfig`, `update_combustion`) — fuel depletion with exothermic heat release
- common: `FluidParticle.fuel` field for combustion concentration

### Changed
- `#[non_exhaustive]` added to 18 public structs for forward-compatible API evolution
- Cargo.toml: license corrected from deprecated `GPL-3.0` to `GPL-3.0-only`
- shallow: NaN dt now rejected (consistent with grid.rs)
- sph: `SpatialHash::new()` error properly propagated (hisab API change)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- grid: Persistent scratch buffers (zero allocation after first step)
- grid: DST pressure solver now propagates transform errors instead of silent ignore
- common: `FluidConfig::validate()` rejects NaN/Inf for dt, smoothing_radius, density, gas_constant
- common: `FluidConfig::validate()` rejects inverted bounds (min > max)
- sph: PCISPH divergence detection (NaN/Inf check on acceleration and predicted positions)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- coupling: FLIP solver particle boundary clamping to grid domain
- coupling: Directional drag cross-section for box shapes (projected onto velocity normal)
- deny.toml for cargo-deny license and advisory checking

## [0.24.3] - 2026-03-24

### Added
- sph: PCISPH pressure solver (`step_pcisph`) with iterative density-error correction
- sph: Adaptive timestep (`SphSolver::adaptive_dt`) — CFL-driven with NaN guards
- grid: MacCormack advection (forward-backward correction with neighbor clamping)
- grid: Periodic boundary conditions (wrapping edges, periodic sampling, `advect_periodic`)
- grid: DST Poisson solver (exact solve via hisab `dst`/`idst` for wall boundaries)
- coupling: `RigidBody` with `BodyShape` (sphere, box) signed distance functions
- coupling: Two-way SPH-rigid body coupling (`couple_sph_bodies`) with penalty forces
- coupling: `FlipSolver` — FLIP/PIC hybrid (P2G, pressure projection, G2P blend)
- coupling: Particle-level set surface tracking (`particle_level_set`)
- coupling: Added mass computation (`effective_mass`, `integrate_bodies_with_added_mass`)
- coupling: Drag from velocity fields (`drag_from_particles`)

### Changed
- grid: Pressure projection uses DST for wall BCs (exact), GS for periodic
- grid: Correct dx scaling in advection, diffusion, divergence, projection
- grid: Diffusion uses correct implicit equation with preserved RHS
- grid: Warm-start pressure solve from previous step
- grid: Minimum grid size 4x4 enforced
- grid: `GridConfig` is `#[non_exhaustive]` with `Serialize`/`Deserialize`/`Default`
- grid: NaN propagation in `sample()`, NaN/Inf dt validation
- coupling: Central-difference surface normals with degenerate fallback
- coupling: Correct dt scaling in FLIP pressure projection
- coupling: `FlipSolver` grid dims are private with getters
- hisab: Switched to local path dependency for DST access

## [0.2.0] - 2026-03-24

### Added
- sph: `SphSolver` — stateful solver with hisab `SpatialHash` for O(n·k) neighbor queries
- sph: Neighbor list caching via `query_cell` (zero per-particle allocation)
- sph: Surface tension via CSF model with poly6 gradient/laplacian kernels
- docs: Roadmap at `docs/development/roadmap.md`

### Changed
- sph: Symmetric pressure formula (`p_i/ρ_i² + p_j/ρ_j²`) for momentum conservation
- sph: Persistent scratch buffers — zero per-step allocations after first call
- common: `FluidParticle` and `FluidConfig` derive `Copy`
- common: Added `speed_squared()`, `distance_squared_to()`
- common: `FluidConfig::validate()` checks gas_constant, boundary_damping
- common: `FluidMaterial::custom()` validates speed_of_sound
- grid: `FluidGrid::new()` validates dx, checks overflow
- grid: `max_speed()` avoids per-cell sqrt
- buoyancy: `terminal_velocity` and `reynolds_number` return `Result`
- buoyancy: `FlowRegime` is `#[non_exhaustive]`
- error: `Diverged` and `InvalidParameter` use `Cow<'static, str>`

### Fixed
- sph: NaN/Inf divergence detection folded into force loop
- sph: `create_particle_block` pre-allocates with `Vec::with_capacity`
- shallow: Scratch buffers resize on deserialization to prevent panic

## [0.1.0] - 2026-03-23

### Added
- common: FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig with CFL validation
- sph: Poly6/Spiky/Viscosity kernels, density/pressure, forces, particle stepping, block creation
- grid: FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion, kinetic energy
- shallow: ShallowWater heightfield, circular disturbances, wave propagation, volume conservation
- buoyancy: Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime
- vortex: 2D vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale
- error: PravashError with #[non_exhaustive], CFL violation detection
