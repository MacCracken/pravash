# Architecture Decision Records

Decisions about pravash — what we chose, the context, and the consequences we accept. Use these when a future reader would reasonably ask *"why did we do it this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never renumber.
- **One decision per ADR.** If a decision supersedes a prior one, add a new ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or `Deprecated`.
- Use [`template.md`](template.md) as the starting point.

## ADR vs. architecture note vs. guide

| Kind | Lives in | Answers |
|---|---|---|
| ADR | `docs/adr/` | *Why did we choose X over Y?* |
| Architecture note | `docs/architecture/` | *What non-obvious constraint is true about the code?* |
| Guide | `docs/guides/` | *How do I do X?* |

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-cyrius-port-with-rust-oracle.md) | Port to Cyrius, with Rust 1.3.0 frozen as the parity oracle | Accepted |
| [0002](0002-port-api-divergences.md) | API differences between Rust 1.3.0 and Cyrius 2.0.0 | Accepted |
| [0003](0003-local-stand-ins-for-library-gaps.md) | Local stand-ins for library gaps, converted when the library ships | Accepted |
| [0004](0004-correct-rounding-is-the-accuracy-target.md) | Correct rounding is the accuracy target, gated exactly against a corrected oracle | Accepted |
