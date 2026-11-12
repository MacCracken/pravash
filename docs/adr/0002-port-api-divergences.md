# 0002 — API differences between Rust 1.3.0 and Cyrius 2.0.0

**Status**: Accepted
**Date**: 2026-11-11

## Context

There is no Rust 2.0 (port plan §0.1). Changes of API shape land in the Cyrius API only, so the Cyrius library
and its oracle differ on purpose in a few places. The harness has to know each one, so that a planned
difference is reported by name and never mistaken for, or used to hide, a defect. The maintainer decided D1–D6
of the port plan on 2026-11-11, taking each recommendation. Rows 2, 4 and 6 come from D3, D6 and D4.

## Decision

These differences are accepted. Each row says how the harness still compares the function, so none leaves
parity. The list is extended by amendment, one row per accepted difference.

| # | Rust 1.3.0 | Cyrius 2.0.0 | Golden comparison |
|---|---|---|---|
| 1 | `Result<T, PravashError>`, with detail strings (`Diverged { reason }`, thiserror text) | a value-form `Result` with a `PV_ERR_*` code; no detail string | class A on the variant |
| 2 | f32 neighbour-hash keys (hisab 1.4.0 `SpatialHash`) | f32 keys, ported bit-faithfully, through M5; then f64 keys before 2.0, which fixes the precision loss at large coordinates (D3) | before the switch, class A on the lists; after it, the f32 rows mapped by name |
| 3 | `sort_by_zorder`'s order among equal keys, which depends on Rust's unstable sort | a stable order on (key, input index). Probe 5 (2026-11-11, port-parity §4.5): ipnsort's order differs at 75–94% of positions on tie-heavy inputs and can't be reproduced without transcribing core's sort; already-sorted inputs give the same order | class A on the key sequence and on the multiset; later steps start from the oracle's order |
| 4 | the compile-time `parallel` feature (rayon) | a runtime thread count per solver: 1 by default, and every count gives the same bits (D6) | the oracle's serial and parallel builds give the same bits, and so does every Cyrius thread count |
| 5 | features select modules at compile time | one `[lib]` bundle, with dead-code elimination: probe 7 measured a full-hisab consumer at 677 KB with it and 3.06 MB without (port-parity §4.7) | — |
| 6 | serde_json's shortest f64 digits | bayan's Grisu2 digits, which round-trip exactly (D4) | every oracle document decodes to the same bits, and every Cyrius document round-trips; the text is not compared |
| 7 | results bit-identical only on one platform | bit-identical across targets for pravash-owned arithmetic | measured on x86_64 ELF, x86_64 PE and aarch64 |
| 8 | a panic (an index out of range, an `expect`) or an abort on allocation failure | the call returns a neutral value and never aborts (CLAUDE.md: no aborts in library code): NaN for an f64, 0 for a word, (0, 0) for a view, 0 for "failed" from a constructor or a growth, each named in the function's comment (amendment 2026-11-11, M1 part 3); a constructor or step that returns a `Result` reports an allocation failure as `PV_ERR_OUT_OF_MEMORY` (amendment 2026-11-11, M2: row 11) | no rows: a panic has no outputs. Each such path has a reference test (`tests/common.tcyr`: `ParticleSoa::distance_squared` out of range, `alloc(usize::MAX)`) |
| 9 | `Option<ArenaHandle>`, slices (`&[FluidParticle]`), `Vec<FluidParticle>` returns, struct-building `Result<Self>`, and `Drop` | an `out` pointer plus 1 / 0 (`alloc`); (pointer, count) for a slice; a freelist buffer the caller frees for a returned Vec (`to_aos`); an `out` pointer plus `Result` (`PvFluidMaterial_custom`); an explicit `PvX_drop` (named so because Rust's `ParticleArena::free(handle)` takes `free`) (amendment 2026-11-11, M1 part 3) | class A on the same values: the harness reads the pointer, count and flag |
| 10 | serde_json for the serde types (`serde_json::to_string`, `from_str`), with `serde_json::Error` | `x.to_json(sb)` (0, or -1 when the builder can't grow) and `PvX_from_json(buf, len, out)` returning a `Result` whose Err is the Cyrius-only `PV_ERR_DESERIALIZE` (appended as 10: serde_json's error is no `PravashError`). Three documents serde_json reads, the port refuses or reads otherwise, by bayan's policies: an integer above `i64::MAX`, a duplicated unknown field (impetus's S3, S4, filed at bayan), and the integer `-0` read as +0 (filed 2026-11-11) (amendment 2026-11-11, M1 part 4) | `tests/golden/json.tsv`, class A on the decoded values (row 6), every decode round-tripped through the port's writer; the three bayan cases are rows of their own (`*_bayan_*`), pending until bayan changes |
| 11 | `ShallowWater`'s public `Vec<f64>` fields, its derived `Clone`, and an allocation failure in `new` or `step` | `PvColumn` fields (resizable, as Rust's public Vecs; `step` refuses inconsistent lengths as `validate_state` does), `PvShallowWater_clone(out, src)` (1 / 0), and the Cyrius-only `PV_ERR_OUT_OF_MEMORY` (appended as 11) where Rust aborts (amendment 2026-11-11, M2) | class A on the same values; no rows for an allocation failure (Rust has no output) |
| 12 | `FluidGrid` (its public `Vec<f64>` fields, private scratch, `mg_levels`, the per-call `Vec`s of the DST and CG solves), the closed `BoundaryCondition`, and the unit-returning `FluidGrid::diffuse` | as row 11 for the fields, `PvFluidGrid_clone` and `PV_ERR_OUT_OF_MEMORY`; the multigrid levels a column of `*PvMgLevel` records; three Cyrius-only private buffers `solve_a` … `solve_c` that the DST and CG solves reuse across steps instead of allocating per call, and two Cyrius-only DST plans `dst_x`, `dst_y`: each transform length's sines and eigenvalue terms, which Rust recomputes inside every sum, built by the first DST step and kept while the interior keeps its size (the same values, so the same bits; amendment 2026-11-12, M3 part b). `step` refuses a `boundary` value outside `PvBoundaryCondition` with `PV_ERR_INVALID_PARAMETER`, after Rust's own config checks (Rust can't hold one). `PvFluidGrid_diffuse(field, len, …)` returns 1, or 0 where Rust warns and returns (and for no memory). A `velocity_at` / `speed_at` index outside a field reads NaN where Rust panics (amendment 2026-11-11, M3 part a) | `tests/golden/grid.tsv`, class A on construction and the reads, B / S on the arithmetic, C on the DST solve (with `grid.cr.tsv`); the row inputs are fresh grids, so the private buffers' contents never reach a row (only their lengths, in the scratch dump); the trajectories reuse a plan for up to 300 steps and the single steps build theirs, and both give the oracle's bits |

## Consequences

- **Positive**
  - The Cyrius API uses the language's idioms: codes, runtime settings, native JSON.
  - It fixes one known defect (row 2) without a Rust release.
- **Negative**
  - The functions in rows 1, 3 and 6 cannot reuse the Rust tests unchanged: their assertions are re-expressed
    against the new shape.
  - The migration guide (M8) has to describe every row.
- **Neutral**
  - New rows need an amendment here, before the code that relies on them.

## Alternatives considered

- **A Rust 2.0 with the same changes, ported 1:1**: rejected (D1 and the sibling ports). It is a breaking
  release of a crate with five published dependents, made only to be retired.
- **No differences at all**: rejected. It would carry the f32 precision defect, Rust-only error strings and a
  compile-time switch Cyrius can't express cleanly into the Cyrius API.
