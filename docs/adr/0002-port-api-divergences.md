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
