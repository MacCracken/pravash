#!/usr/bin/env python3
"""Check the conversion ledger against the code (port plan §8, ADR 0003).

    python3 -I -B scripts/upstream_ledger.py [--check]

The ledger is the table in docs/development/pravash-port-plan.md §8: one row per local stand-in for a gap in
hisab, ganita or the Cyrius stdlib. The table's status column is `open`, `conditional` or `closed`, each stand-in
file is a backticked `src/…` path, and each request is a backticked `<repo>:<path>`. This checks:

1. every `UPSTREAM(Ln)` note in `src/` names a row of the ledger that is not closed;
2. every open row whose stand-in file exists carries at least one `UPSTREAM(Ln)` note in that file, so temporary
   code is always marked;
3. every request an open row names exists, at `<parent>/<repo>/<path>` (the upstream repos are siblings of this
   one; where a sibling isn't checked out, as on CI, its requests are reported as unverified, not failed);
4. no row is listed twice, and a closed row has no notes left in `src/`.

It prints the ledger's state either way; with `--check` (the gate) it exits 1 on any failure.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = ROOT.parent
PLAN = ROOT / "docs" / "development" / "pravash-port-plan.md"
NOTE = re.compile(r"UPSTREAM\((L\d+)\)")


def ledger_rows():
    """{id: {status, standins, requests}} from the §8 table."""
    rows = {}
    dupes = []
    text = PLAN.read_text(encoding="utf-8")
    at = text.find("**The ledger.**")
    if at < 0:
        return rows, ["the plan has no '**The ledger.**' table"]
    for line in text[at:].splitlines():
        m = re.match(r"^\| (L\d+) \| (open|conditional|closed) \| (.*)$", line)
        if not m:
            if line.startswith("**Also filed"):
                break
            continue
        rid, status, rest = m.groups()
        if rid in rows:
            dupes.append(f"{rid} is listed twice")
        ticks = re.findall(r"`([^`]+)`", rest)
        standins = sorted({t for t in ticks if t.startswith("src/") and t.endswith(".cyr")})
        requests = sorted({t for t in ticks if re.match(r"^[a-z0-9_-]+:docs/.+\.md$", t)})
        rows[rid] = {"status": status, "standins": standins, "requests": requests}
    return rows, dupes


def notes_in_src():
    """{id: [file:line, ...]} for every UPSTREAM(Ln) note under src/."""
    found = {}
    for path in sorted((ROOT / "src").rglob("*.cyr")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for rid in NOTE.findall(line):
                found.setdefault(rid, []).append(f"{path.relative_to(ROOT)}:{n}")
    return found


def main():
    check = "--check" in sys.argv[1:]
    rows, fails = ledger_rows()
    notes = notes_in_src()
    unverifiable = set()
    if not rows:
        fails.append("no ledger rows found")

    for rid, where in sorted(notes.items()):
        if rid not in rows:
            fails.append(f"{where[0]}: UPSTREAM({rid}) names no ledger row")
        elif rows[rid]["status"] == "closed":
            fails.append(f"{where[0]}: UPSTREAM({rid}) is left over: the row is closed")

    for rid, row in sorted(rows.items(), key=lambda kv: int(kv[0][1:])):
        if row["status"] != "open":
            continue
        for f in row["standins"]:
            path = ROOT / f
            if path.exists() and f"UPSTREAM({rid})" not in path.read_text(encoding="utf-8"):
                fails.append(f"{f} exists but carries no UPSTREAM({rid}) note")
        if not row["requests"]:
            fails.append(f"{rid} is open but names no upstream request")
        for req in row["requests"]:
            repo, rel = req.split(":", 1)
            if not (REPOS / repo).is_dir():
                unverifiable.add(repo)  # CI has no sibling checkouts; checked locally
            elif not (REPOS / repo / rel).exists():
                fails.append(f"{rid}: the request {req} does not exist")

    for rid, row in sorted(rows.items(), key=lambda kv: int(kv[0][1:])):
        present = [f for f in row["standins"] if (ROOT / f).exists()]
        marked = len(notes.get(rid, []))
        print(
            f"{rid:>4}  {row['status']:<11} stand-ins {len(row['standins'])} "
            f"(present {len(present)}), notes {marked}, requests {len(row['requests'])}"
        )
    if unverifiable:
        print(f"note: no local checkout of {', '.join(sorted(unverifiable))}: those requests were not verified")
    for f in fails:
        print("FAIL", f)
    total_notes = sum(len(v) for v in notes.values())
    print(f"{len(rows)} ledger rows, {total_notes} UPSTREAM notes in src/, {len(fails)} failures")
    return 1 if (check and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
