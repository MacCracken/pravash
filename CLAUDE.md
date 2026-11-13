# Pravash — Claude Code Instructions

> **Core rule**: this file is **preferences, process, and procedures** —
> durable rules that change rarely. Volatile state (current version,
> module line counts, port progress, test counts, consumers) lives in
> [`docs/development/state.md`](docs/development/state.md).
> Do not inline state here.

## Project Identity

**Pravash** (Sanskrit: journey/flow) — fluid dynamics simulation for AGNOS: SPH, Euler/Navier–Stokes grids,
shallow water, buoyancy and drag, vortex dynamics, interface tracking, fluid–body coupling and MPM

- **Type**: Flat library (Cyrius), ported from the Rust crate 1.3.0; the port is 2.0.0
- **License**: GPL-3.0-only
- **Language**: Cyrius. The toolchain is pinned in `cyrius.cyml` `[package].cyrius`
- **Version**: `VERSION` at the project root is the source of truth. Don't inline the number here
- **Rust 1.3.0**: the source of the ported functionality. The tree holds no Rust. The crate is at the tag `1.3.0`;
  the copy the corpus was recorded from (`rust-old/`, with dump hooks) and the Rust generator are at the tag `2.0.0`
- **Plan of record**: [`docs/development/pravash-port-plan.md`](docs/development/pravash-port-plan.md), with
  [ADR 0001](docs/adr/0001-cyrius-port-of-rust-1.3.0.md) (the method),
  [ADR 0002](docs/adr/0002-port-api-divergences.md) (2.0 API changes) and
  [ADR 0003](docs/adr/0003-local-stand-ins-for-library-gaps.md) (stand-ins for library gaps)
- **Standards**: [First-Party Standards](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-standards.md) · [First-Party Documentation](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-documentation.md)

## Goal

pravash owns fluid simulation for AGNOS: particle fluids (SPH and its pressure solvers), grid fluids
(Euler/Navier–Stokes, MAC, k-ε, VOF, phase field), shallow water, fluid–body coupling (FLIP/APIC), MPM,
buoyancy and drag. Built on hisab for math. Deterministic stepping is part of the contract: the same inputs
give the same bits, on every target and at every thread count.

## Consumers

kiran (water/smoke/fire effects), joshua (fluid simulation) and impetus (fluid–body interaction) are the
intended consumers. The current list, including the crates.io dependents of the Rust crate, is in `state.md`.

## Quick Start

```sh
cyrius deps                                    # resolve dependencies (vendors lib/)
cyrius build src/main.cyr build/pravash        # compile
cyrius test                                    # run tests/*.tcyr (parity + reference tests)
cyrius bench                                   # run tests/*.bcyr
python3 -I -B scripts/golden/validate.py .           # recompute every row of the frozen corpus
python3 -I -B scripts/golden/validate.py . --write   # regenerate the correctly rounded rows, the f64-key rows and cr tables (MPFR)
python3 -I -B scripts/golden/probe_tables.py .       # generated constants compile to 1.3.0's bits
python3 -I -B scripts/golden/cr_sweep.py .           # the correctly rounded math vs MPFR beyond the corpus
```

The golden corpus (`tests/golden/`) is frozen at 2.0.0 ([`scripts/golden/README.md`](scripts/golden/README.md)).
`validate.py` calls the recorded glibc (2.44) for its class-C rows, so it runs locally, not in CI (the runner's glibc
differs).

## Development Process

### P(-1): Scaffold Hardening (before any new features)

0. Read the roadmap, the port plan, the CHANGELOG and the open issues — know what was intended before auditing what was built
1. Test + benchmark sweep of the existing code
2. Cleanliness check (below)
3. Get baseline benchmarks
4. Internal deep review — gaps, optimizations, security, logging/errors, docs
5. External research — domain completeness, missing capabilities, best practices, world-class accuracy
6. Cleanliness check — must be clean after the review
7. Additional tests/benchmarks from the findings
8. Post-review benchmarks — prove the wins
9. Repeat if heavy

### Cleanliness check

```sh
cyrius fmt --check <files>
cyrius lint <files>
cyrius vet src/main.cyr
cyrius build src/main.cyr build/pravash
cyrius test                 # must be warning-free, not just passing
cyrius test --bounds
cyrius bench
cyrius distlib --all && cyrius distlib --check   # once [lib] exists (M1)
cyrius deps --locked
cyrius audit
python3 -I scripts/upstream_ledger.py --check    # once the ledger exists (M0 step 10)
python3 -I -B scripts/doc_examples.py .          # from M8 part c: the docs' code blocks and the examples run
cyrius fuzz tests/pravash.fcyr                   # from M8 part e: the JSON readers and the PRAVASH_LOG parser
```

`cyrius test` must be **warning-free**. The float type diagnostics come only from the compiles that `test`
and `build` drive, never from `cyrius lint`. Each one is a bug:
- *integer arithmetic with an f64 right operand*
- *f64 arithmetic with a non-f64 right operand*
- *f32 arithmetic with a non-f32 right operand*
- *an integer stored into an f64/f32 slot*
- *comparison mixes f64 and integer*

### Work Loop (continuous)

1. Work phase — one module at a time, in the order the port plan gives
2. Cleanliness check
3. Test + benchmark additions: reference tests (analytic values, or MPFR through `scripts/golden/mpfr_ref.py`),
   regression tests, benches
4. Run benchmarks against the previous Cyrius numbers (the 2.0.0 record beside Rust 1.3.0's is
   `docs/benchmarks-rust-v-cyrius.md`)
5. Internal review — performance, memory (flat heap per step), throughput, correctness
6. Cleanliness check — must be clean after the review
7. Deeper tests/benchmarks from the review
8. Benchmarks again — prove the wins
9. If the review was heavy, return to step 5
10. Documentation — CHANGELOG, `state.md`, roadmap, the port plan's tracking table, the conversion ledger
11. Version check — `VERSION` and `cyrius.cyml` (`${file:VERSION}`) in sync
12. Return to step 1

### Task Sizing

- **Low/Medium effort**: Batch freely — multiple items per work loop cycle
- **Large effort**: Small bites only — break into sub-tasks, verify each before moving to the next. Never batch large items together
- **If unsure**: Treat it as large. Smaller bites are always safer than overcommitting

### Refactoring

- Refactor when the code tells you to — duplication, unclear boundaries, performance bottlenecks
- Never refactor speculatively. Wait for the third instance before extracting an abstraction
- Refactoring is part of the work loop, not a separate phase
- Every refactor must pass the same cleanliness + benchmark gates as new code

## Key Principles

- **The frozen corpus is the parity bar.** `tests/golden/` records what Rust 1.3.0 did. The port matches each row
  within the tolerance class the port plan (§5) assigns, or its correctly rounded row where accuracy wins
  (ADR 0004). Diverge from a row only with an ADR in `docs/adr/`.
- **Accuracy beats glibc.** Parity is functional. Use correctly rounded primitives, never ones that copy
  glibc's last-ulp errors (port plan §0.1, §5).
- **No deferral without a language check.** A Rust feature leaves the scope only for a reason written down
  against the port plan's language review (§1.1): a missing capability named with its guide or stdlib
  section, a measured cost, or nothing to test against. Re-check before inheriting another port's deferral.
- **Upstream requests never block.** A gap in hisab, ganita or the stdlib is filed in that project with
  pravash's numbers, and pravash ships a local stand-in meanwhile. Each stand-in carries an
  `UPSTREAM(Ln)` note and a row in the conversion ledger (port plan §8), and is converted when the library
  ships, with its parity rows re-run (ADR 0003).
- **hisab-first, off the step path** while cyrius S1 / hisab H9 are open: hisab returns allocate, so per-step
  code uses stand-ins (port plan §0.1).
- **Correctness over cleverness.** If the Cyrius behavior diverges silently from Rust, the bugs win.
- Test after every change, not after the feature is "done". ONE change at a time.
- **Never skip benchmarks.** Numbers don't lie. Speed is measured, not gated: a slow row is a finding to
  file, not a reason to change the port. The dev box is shared: compare with pinned, interleaved runs.
- **Tests + benchmarks are the way.** Minimum 80%+ coverage target.
- **Own the stack.** If an AGNOS library wraps an external capability, depend on the AGNOS library.
- **No magic.** Every operation is measurable, auditable, traceable.
- **Deterministic.** Never iterate a `hashmap` (its seed is random per process); use sorted arrays and typed
  stride arenas. Every sort key is total, or the sort is stable. Any thread count gives the serial bits.
- **Flat heap per step.** Never `alloc_reset` (it wipes the consumer's heap). Solver storage is solver-owned;
  per-step scratch lives in a step arena (port plan §7.5).
- **Enums.** Public enums have explicit, append-only numeric values that are never renumbered (the Cyrius
  form of `#[non_exhaustive]`). Property functions `match` with an `_ =>` arm.
- **`#must_use` on all pure functions**, plus `#pure` where it applies; `#inline` on hot paths.
- **No aborts in library code.** `src/` never uses `+?`, `-?` or `*?` (they exit the process),
  `assert_fatal`, `panic` or `result_unwrap`. Errors return a value-form `Result` with a `PV_ERR_*` code.
- **Strings.** Write into a `str_builder` instead of building temporaries; borrow by pointer and copy only
  into owned storage (the Cyrius form of "`write!` over `format!`" and "Cow over clone").
- **Float hygiene.**
  - Bind every call result to a `: f64` (or `: f32`) local before using it in an operator.
  - An `f32` left operand reads its right operand's low 32 bits: write `x * f32_from(2.0)`.
  - Never put hex in a `const`: it is an integer.
  - Never write `(0 - x)` on a float: use `-x`.
  - No exponent notation in literals: the lexer has none. Generated files write decimals positionally.
  - Use the canonical `ganita_f64_*` names, not the deprecated short aliases.
  - For cross-target determinism use the `_polyfill` transcendentals or the correctly rounded stand-ins
    (port plan §7.1).
- **Constants are generated, never typed by hand.** `src/constants.cyr`, `src/constants_private.cyr`,
  `src/str_debug_data.cyr` and `tests/constants_rows.cyr` were written from Rust 1.3.0 at 2.0.0 and are frozen.
  `validate.py --write` writes the MPFR ones (`src/crmath_data.cyr`, `src/crlog_data.cyr`, the `.cr.tsv` and
  `.d3.tsv` rows). Never edit any of them by hand.
- **Names.** Every public global carries the `PV_` prefix (constants, enum members, tables); private helpers
  use `_pv_`; public structs are `Pv<Name>`. A bundle shares one namespace with its consumers.
- **Function names follow the Rust API**, except where ADR 0002 changes them.
- **Optional dependencies** go in `[features]`, so consumers pull only what they need; `[build] dce` drops
  unused code.
- **tracing on all operations** → sakshi spans, behind a level.
- Build with `cyrius build`, not raw `cat file | cc5` — the manifest auto-resolves deps.
- Source files only need project includes — stdlib auto-resolves from `cyrius.cyml`.
- `var buf[N]` = N **bytes**, not N entries.

## DO NOT

- **Do not commit or push** — the user handles all git operations (commit, push, tag)
- **NEVER use `gh` CLI** — use `curl` to the GitHub API only
- **Do not bring Rust back.** No Rust source, toolchain file, cargo step or Rust-dependent check goes in the tree, CI
  or the release. Rust 1.3.0 is at the tag `1.3.0`; `rust-old/` and the generator are at the tag `2.0.0`
- Do not patch the frozen corpus (`tests/golden/*.tsv`): the port's own answers go in the derived `.cr.tsv` and
  `.d3.tsv` rows, with an ADR
- Do not modify `lib/` files (vendored stdlib / dep symlinks)
- Do not hardcode toolchain versions in CI YAML — `cyrius = "X.Y.Z"` in `cyrius.cyml` is the source of truth
- Do not widen a parity tolerance class to make a row pass. Class-C bounds are set by the rule in
  `docs/development/port-parity.md` before a function's first comparison
- Do not wait on an upstream release: write the stand-in and file the request
- Do not add unnecessary dependencies — keep it lean
- Do not `panic` or abort in library code
- Do not break backward compatibility without a major version bump
- Do not skip benchmarks before claiming performance improvements
- Do not skip tests before claiming changes work
- Do not commit `build/`

## Documentation Structure

```
Root files (required):
  README.md, CHANGELOG.md, CLAUDE.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, LICENSE

docs/:
  development/pravash-port-plan.md — the port plan of record (includes the conversion ledger, §8)
  development/state.md             — live state
  development/roadmap.md           — planned work only; done work goes in the CHANGELOG
  development/history.md           — milestone history of the Rust line
  development/port-parity.md       — measured parity baselines and probe results (M0 steps 8–9)
  development/rust-test-ledger.md  — every Rust test's port or reason (the 2.0.0 record, frozen)
  guides/migration-2.0.md          — moving from the Rust crate 1.3 to the Cyrius library 2.0
  examples/                        — Rust's examples ported, each with the output it must print
  benchmarks-rust-v-cyrius.md      — Rust vs Cyrius benchmark record
  adr/NNNN-title.md                — architecture decision records (template.md)
  architecture/                    — non-obvious constraints
  guides/                          — task-oriented how-tos
```

ADR format: `# NNNN — Title`, then Status, Context, Decision, Consequences (see `docs/adr/template.md`).

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/): `## [Unreleased]`, then per version
`### Added / Changed / Fixed / Removed / Security / Performance`.

- Every change that alters behavior gets a CHANGELOG entry
- Performance claims MUST include benchmark numbers (Rust vs Cyrius where relevant)
- Breaking changes get a **Breaking** section with a migration guide
- Group by module when multiple changes land in one release
- Link the ADR when a change was driven by an architectural decision
- A stand-in's conversion to its library gets an entry naming the ledger row
