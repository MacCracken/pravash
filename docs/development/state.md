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
| M0 Initialize | ⏳ **in progress.** Step 0 (Rust, 2026-11-11): P(-1) gates green; the hidden `oracle` feature (dumps for all eleven stateful types, the SPH neighbour lists, the four pressure-solver hooks, the MLS inverse); `tests/oracle.rs` 14 tests, mutation-checked, green with and without `parallel`; `Cargo.lock` tracked; glibc 2.44. Step 1 (2026-11-11): `cyrius port` scaffold, with the Rust crate in `rust-old/`, which still passes there (351 unit, 33 integration, 14 oracle). Step 2: repairs done (CLAUDE.md, CI, docs, ADRs 0001–0003). Step 3: `cyrius.cyml`, with hisab 3.4.1 test-scoped; `build`, `test`, `test --bounds`, `bench`, `fuzz`, `lint`, `fmt`, `vet`, `deps --locked` and `audit` all green. Step 4: the semantics probe ([`port-parity.md`](port-parity.md) §1), with the helpers it calls for in ledger L2. Step 5: the golden-value generator (`scripts/oracle/`): constants 29 + math 33,239 rows, deterministic, reproduced by the parallel build, all recomputed by `validate.py`; the oracle's cbrt is compiler_builtins' (port-parity §3). Step 6: `src/constants.cyr` (14 public + 6 std float constants, both lists proven complete), every value compiled and compared by `probe_tables.py`. Step 7: the parity harness (`tests/parity.tcyr`): 5,149 ok, 0 failed, 18,493 measured, 9,632 pending; ganita's cbrt equals the oracle's. Steps 8–10: pending |
| M1–M8 | not started |

## Toolchain

- **Cyrius pin**: 6.7.18 (`cyrius.cyml` `[package].cyrius`)
- **hisab**: 3.4.1 (Cyrius), commit `d6703c5`, `scope = "test"` until a library module calls it; sakshi 2.5.10 through it
- **ganita**: 1.2.15, bundled in the 6.7.18 stdlib
- **Rust oracle**: rustc 1.99.1, hisab 1.4.0, glam 0.29.3 (`rust-old/Cargo.lock`), glibc 2.44

## Source

- **Rust oracle**: 6,882 library code lines in `rust-old/src/` (16 modules, 193 `pub fn`s) plus the `oracle`
  hooks. Frozen; never edited.
- **Cyrius**: the scaffold's `src/main.cyr` stub and the generated `src/constants.cyr`. No library modules yet (`[lib]` arrives with M1).

## Tests

- **Rust oracle** (`cargo test --manifest-path rust-old/Cargo.toml --all-features`): 351 unit, 33 integration,
  14 oracle, 7 doc tests `ignore`d.
- **Cyrius**: `tests/parity.tcyr` replays the corpus: 2 modules, 0 failing; 5,149 ok, 0 failed, 18,493 measured (class C, bounds at step 8), 9,632 pending (M1 helpers); plus the scaffold's `tests/pravash.{tcyr,bcyr,fcyr}` stubs.
- **Golden corpus** (`tests/golden/`): constants 35, math 33,239 rows; `validate.py` 0 failures; `probe_tables.py` 35 of 35.

## Upstream requests and stand-ins

The conversion ledger is in the port plan, §8. Requests are filed at M0 step 10; none are filed yet.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib: syscalls, string, alloc, str, fmt, io, vec, math, ganita, tagged, result, freelist, fnptr, atomic, thread, bayan
- hisab 3.4.1 (test scope), which brings sakshi 2.5.10
- test stdlib (`[test] stdlib`, test / bench / fuzz builds only): assert, bench

## Consumers

- **Cyrius:** none declares a pravash dependency yet.
  - goonj (Cyrius 2.3.1) is a bridge partner; ushma (Rust 1.4.0) and pavan (no local repo) are the others.
  - The `ai` configs point at daimon (Cyrius 2.5.6) and hoosh (Cyrius 2.12.3).
- **Rust (crates.io, `^1`):** badal, pavan, khanij, soorat, kiran. soorat is local; its optional `fluids` feature
  uses `sph` + `shallow`.
- **Intended:** kiran (water/smoke/fire), joshua (fluid simulation) and impetus (fluid–body interaction).

## Next

M0 steps 8–10 (port plan §4): the parity harness, the baselines, the probes, and filing the upstream requests.
