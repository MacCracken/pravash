# 0001 — Port to Cyrius, with Rust 1.3.0 frozen as the parity oracle

**Status**: Accepted
**Date**: 2026-11-11

## Context

AGNOS libraries are moving from Rust to Cyrius. pravash's math dependency, hisab, is already Cyrius (3.4.1).
tanmatra and impetus have finished the same kind of port, and kimiya is in progress, all by the method of
tanmatra's ADR 0006: a frozen Rust oracle, generated golden values, and a Cyrius harness that replays them.
impetus is the closest model, because it is also a stateful simulation on hisab.

pravash 1.3.0 is the result of a full-module audit (CHANGELOG 1.3.0). Every fix is pinned by a regression test,
and the numerically subtle ones are mutation-checked; 384 tests are green on rustc 1.99.1. A translation needs
an oracle it can check value by value, and only a repaired crate can be one.

## Decision

pravash 2.0.0 is a Cyrius library. The Rust 1.3.0 crate at commit `a88bc4e` is frozen in `rust-old/` as the
parity oracle, and nothing in it is edited. The commit is the audited 1.3.0 plus the hidden `oracle` hooks.

- **Initialization.** The port was started with `cyrius port` on Cyrius 6.7.18 (2026-11-11).
- **Oracle hooks.** A non-default, hidden `oracle` feature gives the generator read-only access to state the
  public API does not expose: the private fields of every stateful type, the SPH neighbour lists, the grid
  pressure solvers, and the MLS inverse. `rust-old/tests/oracle.rs` proves the hooks are read-only, with and
  without `parallel`.
- **Oracle mechanics.** Generators live in `scripts/oracle/` and depend on `rust-old/` by path.
  - **`rust-old/Cargo.lock` is committed**, an exception to the library-crate rule, so the oracle always builds
    against hisab 1.4.0 and glam 0.29.3.
  - The oracle's transcendentals come from the host glibc (2.44 at freeze). Every corpus header records it.
- **Scope.** Every module and feature is ported. No Rust feature is deferred without a check against the
  language (port plan §1.1). The only work outside 2.0 is the 1.3.0 audit's deferred physics, which has no
  oracle.
- **Parity classes** (port plan §5). Each golden value has a class:
  - A exact;
  - B arithmetic;
  - C transcendental, against the correctly rounded oracle where glibc misrounds;
  - D reordered or library-backed;
  - S single step from a state dump;
  - R reference (physical outcomes).
- **Divergences.** A difference from the oracle beyond its class is a defect, unless an ADR accepts it.
  ADR 0002 accepts the planned API changes.
- **Accuracy.** Parity is functional. The port uses correctly rounded primitives, never ones that reproduce
  glibc's rounding errors.

## Consequences

- **Positive**
  - Correctness is measured, not asserted: every public function, and every solver step, is compared with the
    audited Rust crate.
  - The audit's regression tests travel with the code as reference tests.
- **Negative**
  - Two toolchains are needed until the port completes: cargo for the oracle and its generators, cyrius for the
    library.
  - The golden corpus is tied to a glibc version, and class C needs per-target measurement.
  - `rust-old/` stays in the tree until the port no longer needs it.
- **Neutral**
  - The 1.3.0 Rust CI, release workflow, scripts, architecture overview and 1.2 → 1.3 migration guide are kept
    under `rust-old/`, along with its README (the crate's manifest names it).
  - The 1.3.0 tag goes on `a88bc4e`, whose release workflow is still the Rust one.

## Alternatives considered

- **Port without an oracle** (translate and re-test by hand): rejected. The 1.3 audit showed how easily
  hand-checked values drift, and the sibling ports' corpora caught defects their test suites did not.
- **Keep Rust primary and expose it to Cyrius through FFI**: rejected. The migration's goal is a sovereign Cyrius
  stack, and hisab is already Cyrius.
- **Instrument the Rust solvers to record iteration counts**: rejected. It would mean writes inside the audited
  loops. A bit-identical solution implies the same iteration count.
