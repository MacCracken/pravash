# 0001 — Port Rust 1.3.0's functionality to Cyrius

**Status**: Accepted
**Date**: 2026-11-11

## Context

AGNOS libraries are moving from Rust to Cyrius. pravash's math dependency, hisab, is already Cyrius (3.4.1).
tanmatra and impetus have finished the same kind of port, and kimiya is in progress, all by the method of
tanmatra's ADR 0006: the Rust crate frozen, golden values generated from it, and a Cyrius harness that replays them.
impetus is the closest model, because it is also a stateful simulation on hisab.

pravash 1.3.0 is the result of a full-module audit (CHANGELOG 1.3.0). Every fix is pinned by a regression test,
and the numerically subtle ones are mutation-checked; 384 tests are green on rustc 1.99.1. A translation needs
recorded results to check value by value, and only a repaired crate can give them.

## Decision

pravash 2.0.0 is a Cyrius library. The Rust 1.3.0 crate at commit `a88bc4e` is frozen in `rust-old/`,
and nothing in it is edited. The commit is the audited 1.3.0 plus the hidden `oracle` hooks.

- **Initialization.** The port was started with `cyrius port` on Cyrius 6.7.18 (2026-11-11).
- **Dump hooks.** A non-default, hidden `oracle` feature gives the generator read-only access to state the
  public API does not expose: the private fields of every stateful type, the SPH neighbour lists, the grid
  pressure solvers, and the MLS inverse. `rust-old/tests/oracle.rs` proves the hooks are read-only, with and
  without `parallel`.
- **Generator mechanics.** Generators live in `scripts/golden/` and depend on `rust-old/` by path.
  - **`rust-old/Cargo.lock` is committed**, an exception to the library-crate rule, so Rust 1.3.0 always builds
    against hisab 1.4.0 and glam 0.29.3.
  - 1.3.0's transcendentals come from the host glibc (2.44 at freeze). Every corpus header records it.
- **Scope.** Every module and feature is ported. No Rust feature is deferred without a check against the
  language (port plan §1.1). The only work outside 2.0 is the 1.3.0 audit's deferred physics, which has no
  recorded results.
- **Parity classes** (port plan §5). Each golden value has a class:
  - A exact;
  - B arithmetic;
  - C transcendental, against the correctly rounded rows where glibc misrounds;
  - D reordered or library-backed;
  - S single step from a state dump;
  - R reference (physical outcomes).
- **Divergences.** A difference from Rust 1.3.0 beyond its class is a defect, unless an ADR accepts it.
  ADR 0002 accepts the planned API changes.
- **Accuracy.** Parity is functional. The port uses correctly rounded primitives, never ones that reproduce
  glibc's rounding errors.

## Consequences

- **Positive**
  - Correctness is measured, not asserted: every public function, and every solver step, is compared with the
    audited Rust crate.
  - The audit's regression tests travel with the code as reference tests.
- **Negative**
  - Two toolchains are needed until the port completes: cargo for Rust 1.3.0 and its generators, cyrius for the
    library.
  - The golden corpus is tied to a glibc version, and class C needs per-target measurement.
  - `rust-old/` stays in the tree until the port no longer needs it.
- **Neutral**
  - The 1.3.0 Rust CI, release workflow, scripts, architecture overview and 1.2 → 1.3 migration guide are kept
    under `rust-old/`, along with its README (the crate's manifest names it).
  - The 1.3.0 tag goes on `a88bc4e`, whose release workflow is still the Rust one.

## Update (2026-11-13): Rust leaves the tree

The port no longer needs `rust-old/`, so it is gone, along with the Rust generator and every tool that ran or read
Rust. Both are at the tag 2.0.0 (commit `0f2b054`); the 1.3.0 crate is at the tag `1.3.0`. What stays is frozen:
the golden corpus, the generated sources, the examples' `.out` files and the records (the Rust-test ledger, the
benchmark record, port parity). `validate.py` still recomputes every row, without Rust. The second toolchain and
the third negative consequence above are therefore retired. A recorded value found wrong in Rust is fixed as Rust
1.3.x and recorded again from that tag. It is never patched in the tree.

## Alternatives considered

- **Port without recorded results** (translate and re-test by hand): rejected. The 1.3 audit showed how easily
  hand-checked values drift, and the sibling ports' corpora caught defects their test suites did not.
- **Keep Rust primary and expose it to Cyrius through FFI**: rejected. The migration's goal is a sovereign Cyrius
  stack, and hisab is already Cyrius.
- **Instrument the Rust solvers to record iteration counts**: rejected. It would mean writes inside the audited
  loops. A bit-identical solution implies the same iteration count.
