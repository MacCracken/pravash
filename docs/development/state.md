# pravash — Current State

> Refreshed every release and at every port milestone. `CLAUDE.md` is
> preferences/process/procedures (durable); this file is **state** (volatile).

## Version

**1.3.0** (`VERSION`), on its way to **2.0.0**, the Cyrius library. `cyrius.cyml` reads the version through
`version = "${file:VERSION}"`, and the release workflow checks both against the tag.

The last Rust release is **1.3.0**: commit `a88bc4e` (the audited 1.3.0 plus the hidden `oracle` hooks), frozen
as the oracle in `rust-old/`. It is tagged `1.3.0` and published to crates.io (2026-11-11, port plan D1).

## Port progress

Plan of record: [`pravash-port-plan.md`](pravash-port-plan.md).

| Milestone | Status |
|---|---|
| M0 Initialize | ✅ **done 2026-11-11.** Step 0 (Rust, 2026-11-11): P(-1) gates green; the hidden `oracle` feature (dumps for all eleven stateful types, the SPH neighbour lists, the four pressure-solver hooks, the MLS inverse); `tests/oracle.rs` 14 tests, mutation-checked, green with and without `parallel`; `Cargo.lock` tracked; glibc 2.44. Step 1 (2026-11-11): `cyrius port` scaffold, with the Rust crate in `rust-old/`, which still passes there (351 unit, 33 integration, 14 oracle). Step 2: repairs done (CLAUDE.md, CI, docs, ADRs 0001–0003). Step 3: `cyrius.cyml`, with hisab 3.4.1 test-scoped; `build`, `test`, `test --bounds`, `bench`, `fuzz`, `lint`, `fmt`, `vet`, `deps --locked` and `audit` all green. Step 4: the semantics probe ([`port-parity.md`](port-parity.md) §1), with the helpers it calls for in ledger L2. Step 5: the golden-value generator (`scripts/oracle/`): constants 29 + math 33,239 rows, deterministic, reproduced by the parallel build, all recomputed by `validate.py`; the oracle's cbrt is compiler_builtins' (port-parity §3). Step 6: `src/constants.cyr` (14 public + 6 std float constants, both lists proven complete), every value compiled and compared by `probe_tables.py`. Step 7: the parity harness (`tests/parity.tcyr`): 5,149 ok, 0 failed, 18,493 measured, 9,632 pending; ganita's cbrt equals the oracle's. Step 8: the cross-target baselines: digests identical on x86_64 ELF, PE and aarch64; K = 2 for the primitives, 0 for cbrt; gated **21,852 ok, 0 failed** (port-parity §2.2). Step 9: the eight probes (port-parity §4): every hisab/ganita return leaks on the step path (the hash rebuild 4.1 MB per 1,000-particle step); AoS + f64 locals fastest, struct values 2.5×; f32 keys 10⁶/10⁶ on 3 targets; the neighbour order is (cell offset, insertion); a stable sort replaces ipnsort's tie order; serial = parallel in Rust, and a Cyrius persistent pool gives the serial bits at T = 1–8 (3.9× at 8); one bundle with DCE; hisab's DST is another algorithm (L6 transcribes), ganita's `mat_inv` gives the oracle's bits (L7 for memory only). Step 10: nine upstream requests filed (hisab ×3, ganita, cyrius ×4, sakshi) with verified repros; pravash's numbers in four open issues; the conversion ledger seeded (11 open, 1 conditional) and checked by `scripts/upstream_ledger.py` in CI |
| M1 Foundation | ⏳ **in progress.** Part 1 (2026-11-11): `src/mathx.cyr` (Rust-semantics min / max / clamp, predicates, to_radians, powi, saturating casts; `UPSTREAM(L2)`), the first `[lib]` and `dist/pravash.cyr`; 6,371 rows newly gated, bit-exact on three targets. Part 2 (2026-11-11, ADR 0004): the correctly rounded primitives, `src/crmath.cyr` (impetus's exp, sin, cos, sin_cos, hypot, pow, atan2) and `src/crlog.cyr` (kimiya's ln, log10), with generated tables (`UPSTREAM(L1)`); the MPFR-corrected oracle `tests/golden/math.cr.tsv` (the 43 rows glibc misrounds); K = 0 for every class C math function, all 43 corrected rows accepted, identical on three targets; `tests/crmath.tcyr`; the first benchmark record. Part 3 next: error, vec3, common |
| M2–M8 | not started |

## Toolchain

- **Cyrius pin**: 6.7.18 (`cyrius.cyml` `[package].cyrius`)
- **hisab**: 3.4.1 (Cyrius), commit `d6703c5`, `scope = "test"` until a library module calls it; sakshi 2.5.10 through it
- **ganita**: 1.2.15, bundled in the 6.7.18 stdlib
- **Rust oracle**: rustc 1.99.1, hisab 1.4.0, glam 0.29.3 (`rust-old/Cargo.lock`), glibc 2.44

## Source

- **Rust oracle**: 6,882 library code lines in `rust-old/src/` (16 modules, 193 `pub fn`s) plus the `oracle`
  hooks. Frozen; never edited.
- **Cyrius**: `[lib]` = `src/constants.cyr` (generated), `src/crmath_data.cyr` (generated), `src/crmath.cyr`,
  `src/crlog_data.cyr` (generated), `src/crlog.cyr` and `src/mathx.cyr`, bundled as `dist/pravash.cyr` (4,009
  lines); the scaffold's `src/main.cyr` stub.

## Tests

- **Rust oracle** (`cargo test --manifest-path rust-old/Cargo.toml --all-features`): 351 unit, 33 integration,
  14 oracle, 7 doc tests `ignore`d.
- **Cyrius**:
  - `tests/parity.tcyr` replays the corpus: 2 modules, 0 failing; 30,771 ok, 43 accepted (ADR 4), 0 failed,
    0 measured, 2,460 pending (the SPH cell key until M4). Bits are identical on x86_64 ELF, x86_64 PE and aarch64.
  - `tests/crmath.tcyr`: 85 checks in 18 groups.
  - The scaffold's `tests/pravash.{tcyr,fcyr}` stubs.
- **Golden corpus** (`tests/golden/`): constants 35, math 33,239 rows, and the corrected oracle `math.cr.tsv`
  (43 rows, MPFR 4.2.2). `validate.py` 0 failures; `probe_tables.py` 35 of 35.
- **Benchmarks** (`tests/pravash.bcyr`, [`../benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md)): the
  10 primitives against Rust's glibc, on shared seeded arguments. The correctly rounded functions run 58 ns (ln) to
  1.41 µs (atan2, whose mean comes from a rare slow path), 11–103× glibc.

## Upstream requests and stand-ins

The conversion ledger is in the port plan, §8: 12 rows, 11 open and 1 conditional (L12, memcpy, waiting on M1's
numbers). Each open row names its filed request. The stand-ins so far:
- L1: `src/crmath.cyr`, `src/crlog.cyr` and their two table files;
- L2: `src/mathx.cyr` and `src/constants.cyr`'s std float constants.

Each is marked. `scripts/upstream_ledger.py --check`: 6 notes, 0 failures.

Filed 2026-11-11:
- **hisab:** `2026-11-11-spatial-hash-per-step-rebuild-allocates.md`, `2026-11-11-no-matrix-free-linear-cg.md`,
  `2026-11-11-dst-allocates-per-call.md`;
- **ganita:** `2026-11-11-mat-inv-allocates-per-call.md`;
- **cyrius:** `2026-11-11-f64-min-max-drop-nan-only-as-first-operand.md`, `2026-11-11-no-stable-sort.md`,
  `2026-11-11-derive-json-no-vector-array-or-tagged-variant-shapes.md`,
  `2026-11-11-struct-values-and-f64v4-lanes-cost-in-pair-loops.md`;
- **sakshi:** `2026-11-11-no-chrome-trace-export.md`;
- **pravash's numbers added to:** ganita `2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`, hisab
  `2026-11-09-allocator-scoped-calls.md`, cyrius `2026-11-09-no-scoped-allocator-for-alloc.md`, and cyrius
  `2026-11-08-cyrius-port-clobbers-project-docs-and-omits-bench.md`.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib: syscalls, string, alloc, str, fmt, io, vec, math, ganita, tagged, result, freelist, fnptr, atomic, thread, bayan
- hisab 3.4.1 (test scope), which brings sakshi 2.5.10
- test stdlib (`[test] stdlib`, test / bench / fuzz builds only): assert, bench, overflow, simd
- local tools only (not the library, not CI): MPFR 4.x (`libmpfr.so.6`) for `scripts/oracle/validate.py`

## Consumers

- **Cyrius:** none declares a pravash dependency yet.
  - goonj (Cyrius 2.3.1) is a bridge partner; ushma (Rust 1.4.0) and pavan (no local repo) are the others.
  - The `ai` configs point at daimon (Cyrius 2.5.6) and hoosh (Cyrius 2.12.3).
- **Rust (crates.io, `^1`):** badal, pavan, khanij, soorat, kiran. soorat is local; its optional `fluids` feature
  uses `sph` + `shallow`.
- **Intended:** kiran (water/smoke/fire), joshua (fluid simulation) and impetus (fluid–body interaction).

## Next

M1 part 3: `src/error.cyr` (`PV_ERR_*`), `src/vec3.cyr` and `src/common.cyr` (`FluidParticle`, `FluidConfig`,
`FluidMaterial`, `ParticleArena`, `ParticleSoa`), with surface parity against `rust-old/src/` from here on.
