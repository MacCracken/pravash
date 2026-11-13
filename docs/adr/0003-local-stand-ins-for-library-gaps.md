# 0003 — Local stand-ins for library gaps, converted when the library ships

**Status**: Accepted
**Date**: 2026-11-11

## Context

The port needs things the libraries below it don't yet provide, or provide in a form a per-step solver can't
use:
- correctly rounded transcendentals (ganita);
- a matrix-free linear CG and an allocation-free spatial hash (hisab);
- a stable sort and some `#derive` shapes (the stdlib);
- an allocator scope, without which hisab's returned values leak on every step (cyrius S1, hisab H9).

The siblings handled these gaps in two ways. impetus owned copies until hisab caught up (its ADR 0012), and
kimiya transcribed hisab 1.4.0 routines (its ADR 0003). On 2026-11-11 the maintainer set the rule for pravash:
"we will make the upstream requests in projects like hisab and stdlib ganita but not let it block the port,
with notes to convert to lib when available."

## Decision

Every gap is filed as a request in the owning project, and pravash ships a local stand-in meanwhile. Nothing in
the port waits on an upstream release.

- **Filing.** In the owning repo's `docs/development/issues/`, with a self-proving repro where possible and the
  probe or benchmark numbers that show the gap. Where a request is already open, pravash joins it with its own
  numbers instead of filing a second.
- **Stand-ins.** A stand-in is one of three kinds:
  - a transcription of the routine Rust 1.3.0 used (hisab 1.4.0), which keeps bit parity;
  - a copy of a sibling's code (`crmath`, from impetus and kimiya; port plan D2);
  - a pravash-owned implementation.
- **Marking.** Each stand-in carries an `UPSTREAM(Ln)` note naming its request, the library call that replaces
  it, and the parity rows to re-run. Each one has a row in the conversion ledger (port plan §8).
  `scripts/upstream_ledger.py --check` ties notes and rows together and runs in every milestone's gate.
- **Converting.** When the library ships, before or after 2.0:
  1. bump the pin;
  2. swap the call and delete the stand-in;
  3. re-run the named parity rows. They keep their class. A change of bits is either the library being more
     accurate, which is accepted against the correctly rounded rows, or a defect, which is filed while the stand-in
     stays;
  4. close the ledger row, and add a CHANGELOG entry.

## Consequences

- **Positive**
  - The port proceeds at its own pace, and each upstream project gets a consumer's concrete case and numbers.
  - A transcription keeps 1.3.0's bits, so it is a stronger parity check than a substitute would be.
- **Negative**
  - pravash temporarily owns code that belongs elsewhere: the correctly rounded math, a CG, a neighbour grid,
    3D vector math.
  - Each conversion is work, with its own parity run.
- **Neutral**
  - The ledger is the single list of what pravash still owns on a library's behalf. Rows still open at 2.0
    carry over to the 2.x roadmap.

## Alternatives considered

- **Wait for the libraries**: rejected by the maintainer. It would tie the port's schedule to several other
  projects' releases.
- **Stand-ins with no tracking**: rejected. Without the notes and the ledger, temporary code becomes permanent,
  as impetus's three ADR 0012 stand-ins show.
- **Write the missing pieces upstream first, then use them**: not excluded. An upstream fix can land before the
  stand-in is written, and then no stand-in is needed. But the port doesn't wait on it.
