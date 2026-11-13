# Working on pravash

This guide is for changing pravash itself. To use the library, see the [README](../../README.md) and the
[migration guide](migration-2.0.md). The rules (the work loop, the cleanliness check, what never to edit) are in
[`CLAUDE.md`](../../CLAUDE.md); the method is [ADR 0001](../adr/0001-cyrius-port-of-rust-1.3.0.md).

## Build and test

```sh
cyrius deps                                 # vendor the stdlib and hisab (tests only) into lib/
cyrius build src/main.cyr build/pravash     # compile
cyrius test                                 # every tests/*.tcyr; it must also be warning-free
cyrius test --bounds
cyrius bench                                # tests/pravash.bcyr
cyrius fuzz                                 # tests/pravash.fcyr
cyrius distlib && cyrius distlib --check    # dist/pravash.cyr, the bundle consumers include
```

## Layout

- **`src/`**: the library, one file per Rust module or part of one. `cyrius.cyml`'s `[lib]` list is the bundle's
  concatenation order, and it is load-bearing: no file includes another, so each resolves against what precedes it.
  `tests/lib.cyr` mirrors the list for test compiles. `src/main.cyr` is the scaffold's entry point and holds nothing
  of the library.
- **Generated files** (never edited by hand): `src/constants.cyr`, `src/constants_private.cyr`,
  `src/str_debug_data.cyr` (recorded from Rust 1.3.0 at 2.0.0, frozen), and `src/crmath_data.cyr`, `src/crlog_data.cyr`
  (`validate.py --write`, from MPFR).
- **`tests/`**: `parity.tcyr` replays the golden corpus (`tests/golden/<module>.tsv`, with `.cr.tsv` for the
  correctly rounded rows and `.d3.tsv` for the f64 neighbour keys); `<module>.tcyr` holds the ported Rust tests under
  their Rust names, then Cyrius-only groups; `pravash.bcyr` the benchmarks; `pravash.fcyr` the fuzz harness.
- **`scripts/`**: the corpus checks (`golden/`, below), the upstream ledger, the doc examples and the M0 probes
  (`probes/`).
- **`docs/`**: the port plan and live state (`development/`), the ADRs, these guides and the ported examples.

## The frozen corpus and its checks

The golden corpus is frozen at 2.0.0. The Rust generator that recorded it, and `rust-old/` (Rust 1.3.0 with dump
hooks), left the tree once the port was done; both are at the tag 2.0.0 (commit 0f2b054). The tree holds no Rust.
`validate.py` calls the recorded glibc (2.44) for its class-C rows, so it runs locally; CI runs the last two.

```sh
python3 -I -B scripts/golden/validate.py .            # every row recomputed in Python
python3 -I -B scripts/golden/probe_tables.py .        # each generated constant compiles to 1.3.0's bits
python3 -I -B scripts/upstream_ledger.py --check      # every stand-in has its upstream request
python3 -I -B scripts/doc_examples.py .               # the docs' code blocks and the examples run
```

## Changing a function

1. Read its tests (`tests/<module>.tcyr`), its golden rows (`tests/golden/<module>.tsv`) and the ADRs that name it.
2. Make the change and run `tests/parity.tcyr` until every row still matches within its class. Never widen a
   tolerance or patch a row to make one pass.
3. A result that should differ from a recorded row needs an ADR first. Where accuracy is the reason, the row's
   correctly rounded value goes in its `.cr.tsv` (ADR 0004), which `validate.py --write` writes from its
   transcription.
4. Add reference tests (analytic values, or MPFR through `scripts/golden/mpfr_ref.py`), and a regression test for a
   fix, to `tests/<module>.tcyr`.
5. A missing library capability needs a stand-in with an `UPSTREAM(Ln)` note and a request filed upstream
   (ADR 0003).
6. Run the cleanliness check in `CLAUDE.md`, the benchmarks, and update the CHANGELOG and `state.md`.

See [`../adr/template.md`](../adr/template.md) when a design choice deserves an ADR.
