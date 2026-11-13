# Working on pravash

This guide is for changing pravash itself. To use the library, see the [README](../../README.md) and the
[migration guide](migration-2.0.md). The rules (the work loop, the cleanliness check, what never to edit) are in
[`CLAUDE.md`](../../CLAUDE.md); the method is [ADR 0001](../adr/0001-cyrius-port-with-rust-oracle.md).

## Build and test

```sh
cyrius deps                                 # vendor the stdlib and hisab (tests only) into lib/
cyrius build src/main.cyr build/pravash     # compile
cyrius test                                 # every tests/*.tcyr; it must also be warning-free
cyrius test --bounds
cyrius bench                                # tests/pravash.bcyr
cyrius distlib && cyrius distlib --check    # dist/pravash.cyr, the bundle consumers include
```

## Layout

- **`src/`**: the library, one file per Rust module or part of one. `cyrius.cyml`'s `[lib]` list is the bundle's
  concatenation order, and it is load-bearing: no file includes another, so each resolves against what precedes it.
  `tests/lib.cyr` mirrors the list for test compiles. `src/main.cyr` is the scaffold's entry point and holds nothing
  of the library.
- **Generated files** (never edited by hand): `src/constants.cyr`, `src/constants_private.cyr`,
  `src/str_debug_data.cyr` (from the oracle, `scripts/oracle/`), and `src/crmath_data.cyr`, `src/crlog_data.cyr`
  (`validate.py --write`, from MPFR).
- **`tests/`**: `parity.tcyr` replays the golden corpus (`tests/golden/<module>.tsv`, with `.cr.tsv` for the
  corrected oracle and `.d3.tsv` for the f64 neighbour keys); `<module>.tcyr` holds the ported Rust tests under
  their Rust names, then Cyrius-only groups; `pravash.bcyr` the benchmarks; `pravash.fcyr` the fuzz target.
- **`rust-old/`**: the Rust 1.3.0 crate, the oracle. Never edited; tools drive it from `scripts/oracle/`.
- **`scripts/`**: the generator and the independent checks (below), the surface and test ledgers, the benchmark
  record, the upstream ledger and the doc examples.
- **`docs/`**: the port plan and live state (`development/`), the ADRs, these guides and the ported examples.

## The checks against the oracle

The generator and `validate.py` are tied to the recorded glibc (2.44), so they run locally, not in CI.

```sh
cargo test --manifest-path rust-old/Cargo.toml --all-features                          # the oracle itself
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml -- --check      # the corpus is current
python3 -I -B scripts/oracle/validate.py .            # every row recomputed in Python, without Rust
python3 -I -B scripts/oracle/probe_tables.py .        # each generated constant compiles to the oracle's bits
python3 -I -B scripts/oracle/examples.py .            # docs/examples/*.out are the oracle's
python3 -I -B scripts/surface_parity.py .             # every Rust item has its Cyrius name, and back
python3 -I -B scripts/rust_test_ledger.py . --check   # every Rust test is ported or recorded
python3 -I -B scripts/upstream_ledger.py --check      # every stand-in has its upstream request
python3 -I -B scripts/doc_examples.py .               # the docs' code blocks and the examples run
```

## Porting or changing a function

1. Read its Rust source in `rust-old/src/`, and its tests.
2. Add its golden rows to the generator (`scripts/oracle/src/m_<module>.rs`), with inputs that reach every branch,
   the edges and the refusals, and regenerate the corpus. Add its recomputation to `validate.py`.
3. Write the Cyrius function under the Rust name (port plan §7.4) and run `tests/parity.tcyr` until every row
   matches within its class. Never widen a tolerance to make a row pass.
4. Port its Rust tests to `tests/<module>.tcyr` under their Rust names.
5. A deliberate difference from Rust needs an ADR 0002 row first; a missing library capability needs a stand-in with
   an `UPSTREAM(Ln)` note and a request filed upstream (ADR 0003).
6. Run the cleanliness check in `CLAUDE.md`, the benchmarks, and update the CHANGELOG and `state.md`.

See [`../adr/template.md`](../adr/template.md) when a design choice deserves an ADR.
