# Contributing to pravash

Thank you for considering a contribution! This document covers the development
workflow, coding standards, and review process.

pravash 2.0 is a Cyrius library, the port of the Rust crate 1.3.0. The plan of record is
[`docs/development/pravash-port-plan.md`](docs/development/pravash-port-plan.md), and
[`docs/guides/getting-started.md`](docs/guides/getting-started.md) is the guide to working on it. The Rust 1.3.0 crate
in `rust-old/` is kept for reference: don't change it. Nothing in the build, the tests, CI or the release depends on
it; the local porting tools run it with `cargo +1.99.1`.

## Development Setup

```sh
# Clone the repository
git clone https://github.com/MacCracken/pravash.git
cd pravash

# The Cyrius toolchain version is pinned in cyrius.cyml ([package].cyrius)
cyrius --version

# Resolve dependencies (vendors lib/), build, test
cyrius deps
cyrius build src/main.cyr build/pravash
cyrius test

# The Rust oracle (read-only; needs cargo)
cargo test --manifest-path rust-old/Cargo.toml --all-features
```

## Pull Request Process

1. **Fork and branch** — create a feature branch from `main`.
2. **Keep commits focused** — one logical change per commit; one module at a time during the port.
3. **Write tests** — a ported function needs its golden parity rows and its ported Rust tests; a bug fix needs
   a regression test.
4. **Run the cleanliness check** before pushing (CLAUDE.md lists it): `cyrius fmt --check`, `cyrius lint`,
   `cyrius vet`, `cyrius build`, `cyrius test` (warning-free), `cyrius test --bounds`, `cyrius bench`,
   `cyrius deps --locked`, `cyrius audit`.
5. **Open a PR** against `main` with a clear description of the change.
6. **Address review feedback** — maintainers may request changes before merging.

## Code Style

- `cyrius fmt` formatting and a clean `cyrius lint` (enforced by CI).
- `cyrius test` and `cyrius build` print no float type warnings. Each warning is a bug.
- Public functions have doc comments, and physics functions document their units.
- `#must_use` on pure functions, `#inline` on hot paths, append-only enum values.
- No aborts in `src/`: no `+?`, `-?`, `*?`, `assert_fatal`, `panic` or `result_unwrap`.
- Names: `Pv<Name>` structs, `PV_` globals, `_pv_` private helpers.
- A stand-in for a missing library routine carries an `UPSTREAM(Ln)` note and a ledger row (ADR 0003).

## Testing Requirements

- Every ported public function is compared against the oracle's golden values, within the parity class the
  port plan (§5) assigns. Never widen a class to make a row pass.
- Every Rust test is ported, folded into golden rows, or recorded with a reason.
- Benchmarks go in `tests/*.bcyr`, beside their Rust twins, and are run before and after performance-sensitive
  changes.

```sh
cyrius test            # unit, parity and reference tests
cyrius test --bounds   # the same, bounds-checked
cyrius bench           # benchmarks
cyrius fuzz            # fuzz targets
```

## Commit Messages

Use clear, imperative-mood commit messages:

```
port shallow water's HLL fluxes (M2 part 1)

fix NaN handling in FluidConfig validation
```

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project (see [LICENSE](LICENSE)).
