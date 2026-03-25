# Contributing to pravash

Thank you for considering a contribution! This document covers the development
workflow, coding standards, and review process.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/MacCracken/pravash.git
cd pravash

# Ensure you have the correct toolchain (see rust-toolchain.toml)
rustup show

# Run the full local CI suite
make check
```

## Pull Request Process

1. **Fork and branch** — create a feature branch from `main`.
2. **Keep commits focused** — one logical change per commit.
3. **Write tests** — new features require tests; bug fixes require a regression
   test.
4. **Run CI locally** before pushing:
   ```bash
   make check        # fmt + clippy + test + audit
   ```
5. **Open a PR** against `main` with a clear description of the change.
6. **Address review feedback** — maintainers may request changes before merging.

## Code Style

- Follow `rustfmt` defaults (enforced by CI).
- Zero clippy warnings (`cargo clippy --all-features --all-targets -- -D warnings`).
- Public API items must have doc comments.
- Use `#[inline]` on small, hot-path functions.
- Use `#[non_exhaustive]` on public enums and extensible structs.
- Physics functions must document units in their doc comments.

## Testing Requirements

- All public API changes must include unit tests.
- Integration tests go in `tests/`.
- Benchmarks go in `benches/` and should be run before/after performance-sensitive
  changes.
- Target: maintain or improve code coverage on every PR.

```bash
# Run tests
cargo test --all-features

# Run benchmarks
make bench

# Generate coverage report
make coverage
```

## Commit Messages

Use clear, imperative-mood commit messages:

```
add Manning's bed friction to shallow water solver

fix NaN handling in FluidConfig validation
```

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project (see [LICENSE](LICENSE)).
