"""The Rust-test ledger: every test of the frozen Rust oracle is ported, or recorded with a reason (port plan M8).

After impetus's scripts/rust_test_ledger.py (the mela pattern). Usage, from the repo root:
    python3 -I -B scripts/rust_test_ledger.py .           # write docs/development/rust-test-ledger.md
    python3 -I -B scripts/rust_test_ledger.py . --check   # exit 1 if the ledger is stale or a test is unaccounted

It lists each `#[test]` in rust-old/src/**.rs and rust-old/tests/*.rs, and each doc test: a Rust code fence in a doc
comment (rustdoc lists an `ignore`d one as an ignored test), or in rust-old/README.md when lib.rs takes it as the
crate's docs. A unit or integration test is ported when one of the Cyrius files mapped to its Rust file holds a
`test_group("<its name>")`; a doc test is ported by its DOC_TESTS entry, a `doc_*` group. Otherwise it needs a REASONS
entry (or its file one in WHOLE_FILE). Errors, besides an unaccounted test:
- a Rust file with tests and no FILES entry;
- a REASONS entry for a test that is ported or doesn't exist;
- a doc fence with no DOC_TESTS entry, or an entry naming no fence or a group its file doesn't hold;
- a group name defined twice in a Cyrius file, or across one Rust file's Cyrius files;
- a Cyrius test function holding a `test_group` that nothing in its file calls, so the group never runs.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
check = "--check" in sys.argv[2:]
out_path = root / "docs/development/rust-test-ledger.md"

# Rust file -> the Cyrius test files its tests are ported into ([] when none is).
FILES = {
    "src/ai.rs": ["tests/ai.tcyr"],
    "src/bridge.rs": ["tests/bridge.tcyr"],
    "src/buoyancy.rs": ["tests/buoyancy.tcyr"],
    "src/common.rs": ["tests/common.tcyr"],
    "src/compute.rs": ["tests/compute.tcyr"],
    "src/coupling.rs": ["tests/coupling.tcyr"],
    "src/error.rs": ["tests/error.tcyr"],
    "src/grid.rs": ["tests/grid.tcyr"],
    "src/logging.rs": ["tests/logging.tcyr"],
    "src/mpm.rs": ["tests/mpm.tcyr"],
    "src/phase_field.rs": ["tests/phase_field.tcyr"],
    "src/shallow.rs": ["tests/shallow.tcyr"],
    "src/sph.rs": ["tests/sph.tcyr"],
    "src/vof.rs": ["tests/vof.tcyr"],
    "src/vortex.rs": ["tests/vortex.tcyr"],
    "tests/integration.rs": ["tests/integration.tcyr"],
    "tests/oracle.rs": [],
}

# The doc tests: (Rust file, the fence's line) -> (what it shows, Cyrius file, test group). Rust marks all seven
# `ignore`: they sketch a consumer's code and don't compile. The port writes each against the API and runs it.
DOC_TESTS = {
    ("src/compute.rs", 12): ("the module doc's GPU backend", "tests/compute.tcyr", "doc_compute_module_backend"),
    ("src/compute.rs", 217): ("`ComputeBackend`: compiled shaders, upload, dispatch, read back", "tests/compute.tcyr",
                              "doc_compute_backend"),
    ("src/compute.rs", 275): ("`NeuralCorrector`: an FNO corrector", "tests/compute.tcyr", "doc_neural_corrector"),
    ("src/compute.rs", 364): ("`KernelDerivatives::autodiff_gradient` of r²", "tests/compute.tcyr",
                              "doc_autodiff_gradient"),
    ("src/logging.rs", 39): ("`init_profiling` and its guard", "tests/logging.tcyr", "doc_init_profiling"),
    ("src/sph.rs", 12): ("the module doc's `SphSolver::step`", "tests/sph.tcyr", "doc_sph_module"),
    ("src/sph.rs", 700): ("`ReactionProvider`: an Arrhenius reactor (kimiya's)", "tests/sph.tcyr",
                          "doc_reaction_provider"),
}

ORACLE = ("n/a: the tests of the `oracle` hooks added to the frozen crate for this port (port plan §4 step 0), not "
          "of pravash 1.3.0's API. `scripts/oracle/` drives the hooks, and the golden corpus checks what they dump")

# (Rust file, test) -> why it has no port.
REASONS = {}
# Rust file -> the reason for each of its tests (the whole file has no port).
WHOLE_FILE = {"tests/oracle.rs": ORACLE}

# Code fence info strings rustdoc compiles or lists as a test (its attributes); any other tag is another language.
RUST_TAGS = {"rust", "ignore", "no_run", "should_panic", "compile_fail", "edition2015", "edition2018",
             "edition2021", "edition2024", "test_harness", "standalone_crate"}


def rust_tests(rel):
    """The (name, ignored) of each #[test] in rust-old/<rel>, in file order."""
    lines = (root / "rust-old" / rel).read_text().split("\n")
    found = []
    for i, line in enumerate(lines):
        if line.strip().startswith("#[test]"):
            ignored = False
            for nxt in lines[i + 1:i + 8]:
                if nxt.strip().startswith("#[ignore"):
                    ignored = True
                m = re.match(r"\s*(?:pub )?fn (\w+)", nxt)
                if m:
                    found.append((m.group(1), ignored))
                    break
    return found


def group_counts(rel):
    """Each test_group name in a Cyrius file, with how often it is defined."""
    out = {}
    for g in re.findall(r'test_group\("([^"]+)"', (root / rel).read_text()):
        out[g] = out.get(g, 0) + 1
    return out


def uncalled(rel):
    """The functions of a Cyrius file that hold a test_group and are called nowhere in it."""
    text = (root / rel).read_text()
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^fn (\w+)\(", text, re.M)]
    out = []
    for at, name in starts:
        close = re.compile(r"^}", re.M).search(text, at)
        end = close.end() if close else len(text)
        if "test_group(" not in text[at:end]:
            continue
        rest = text[:at] + text[end:]
        if not re.search(rf"(?<![\w.]){name}\(", rest):
            out.append(name)
    return out


def doc_fences(rel, text, prefixes):
    """The (line, info) of each Rust code fence in text, inside comments starting with one of prefixes ("" = any)."""
    out = []
    inside = False
    for n, line in enumerate(text.split("\n"), 1):
        body = line.strip()
        if prefixes:
            hit = next((p for p in prefixes if body.startswith(p)), None)
            if hit is None:
                continue
            body = body[len(hit):].strip()
        if body.startswith("```"):
            if inside:
                inside = False
                continue
            inside = True
            info = body[3:].strip()
            tags = {t.strip() for t in info.split(",") if t.strip()}
            if tags <= RUST_TAGS:
                out.append((rel, n, info))
    return out


errors = []
rust_files = sorted(str(p.relative_to(root / "rust-old")) for d in ("src", "tests")
                    for p in (root / "rust-old" / d).rglob("*.rs"))
for rel in rust_files:
    if rust_tests(rel) and rel not in FILES:
        errors.append(f"{rel} has tests and no FILES entry")

rows = []      # (rust file, cyrius files, tests, counts)
notes = []     # (rust file, test, reason)
ignored = []   # (rust file, test, where it is ported)
for rel, cys in FILES.items():
    tests = rust_tests(rel)
    have = {}
    for cy in cys:
        for g, c in group_counts(cy).items():
            have[g] = have.get(g, 0) + c
            if c > 1:
                errors.append(f"{cy}: test_group(\"{g}\") is defined {c} times")
        for name in uncalled(cy):
            errors.append(f"{cy}: {name} holds a test_group and nothing calls it")
    counts = {"ported": 0, "n/a": 0, "deferred": 0}
    for name, ign in tests:
        if name in have:
            counts["ported"] += 1
            if have[name] > 1 and len(cys) > 1:
                errors.append(f"{rel}: {name} is defined {have[name]} times across {cys}")
            if (rel, name) in REASONS:
                errors.append(f"{rel}: {name} is ported but has a REASONS entry")
            if ign:
                ignored.append((rel, name, [cy for cy in cys if name in group_counts(cy)]))
            continue
        reason = REASONS.get((rel, name)) or WHOLE_FILE.get(rel)
        if reason is None:
            errors.append(f"{rel}: {name} is neither ported nor recorded")
            continue
        counts[reason.split(":")[0]] += 1
        if rel not in WHOLE_FILE:
            notes.append((rel, name, reason))
        if ign:
            ignored.append((rel, name, None))
    for (r, t) in REASONS:
        if r == rel and t not in [n for n, _ in tests]:
            errors.append(f"{rel}: REASONS names {t}, which is not a test")
    rows.append((rel, cys, len(tests), counts))
for r, _ in REASONS:
    if r not in FILES:
        errors.append(f"REASONS names {r}, which has no FILES entry")

# Doc tests: every Rust fence in a doc comment (and the README's, when it is the crate's docs) has a DOC_TESTS entry.
fences = []
for p in sorted((root / "rust-old/src").rglob("*.rs")):
    fences += doc_fences(str(p.relative_to(root / "rust-old")), p.read_text(), ("//!", "///"))
if '#[doc = include_str!("../README.md")]' in (root / "rust-old/src/lib.rs").read_text():
    fences += doc_fences("README.md", (root / "rust-old/README.md").read_text(), ())
found = {(rel, n): info for rel, n, info in fences}
for key, info in sorted(found.items()):
    if key not in DOC_TESTS:
        errors.append(f"doc fence {key[0]}:{key[1]} (```{info}) has no DOC_TESTS entry")
doc_rows = []
for (rel, line), (what, cy, name) in DOC_TESTS.items():
    if (rel, line) not in found:
        errors.append(f"DOC_TESTS names {rel}:{line}, which is not a Rust doc fence")
        continue
    c = group_counts(cy).get(name, 0)
    if c != 1:
        errors.append(f"doc test {rel}:{line}: {cy} defines test_group(\"{name}\") {c} times")
    doc_rows.append((rel, line, found[(rel, line)], what, cy, name))
for cy in sorted({cy for _, cy, _ in DOC_TESTS.values()} - {c for cys in FILES.values() for c in cys}):
    for name in uncalled(cy):
        errors.append(f"{cy}: {name} holds a test_group and nothing calls it")

unit = sum(r[2] for r in rows if r[0].startswith("src/"))
integ = sum(r[2] for r in rows if r[0].startswith("tests/") and r[0] not in WHOLE_FILE)
hooks = sum(r[2] for r in rows if r[0] in WHOLE_FILE)
total = sum(r[2] for r in rows) + len(DOC_TESTS)
sums = {k: sum(r[3][k] for r in rows) for k in ("ported", "n/a", "deferred")}
sums["ported"] += len(DOC_TESTS)


def cell(cys):
    return "<br>".join(f"`{c}`" for c in cys) if cys else "—"


md = []
md.append("# Rust-test ledger")
md.append("")
md.append("> **Generated** by `python3 -I -B scripts/rust_test_ledger.py .`; `--check` gates it in CI (port plan M8,")
md.append("> after impetus's ledger). Every test of the frozen Rust oracle (`rust-old/`, pravash 1.3.0 with the `oracle`")
md.append("> hooks) is ported to a Cyrius test group of the same name, or recorded here with its reason. The golden corpus")
md.append("> (`tests/golden/`) checks every ported function on its grid besides; this ledger is about the oracle's own")
md.append("> assertions.")
md.append("")
md.append(f"**{total} Rust tests** ({unit} unit, {integ} integration, {hooks} of the oracle hooks, {len(DOC_TESTS)} doc "
          "tests):")
md.append(f"{sums['ported']} ported, {sums['n/a']} not applicable, {sums['deferred']} deferred.")
md.append("")
md.append("| Rust file | Cyrius tests | Rust tests | Ported | N/A | Deferred |")
md.append("|---|---|---:|---:|---:|---:|")
for rel, cys, n, c in rows:
    md.append(f"| `{rel}` | {cell(cys)} | {n} | {c['ported']} | {c['n/a']} | {c['deferred']} |")
doc_files = sorted({cy for *_, cy, _ in doc_rows})
md.append(f"| doc tests | {cell(doc_files)} | {len(DOC_TESTS)} | {len(DOC_TESTS)} | 0 | 0 |")
md.append(f"| **Total** | | **{total}** | **{sums['ported']}** | **{sums['n/a']}** | **{sums['deferred']}** |")
md.append("")
md.append("## Doc tests")
md.append("")
md.append("Rust's doc comments hold seven code fences, all marked `ignore`: each sketches a consumer's code (a GPU")
md.append("backend, a learned corrector, a chemistry provider) and doesn't compile, so rustdoc lists them as ignored. The")
md.append("port writes each against the 2.0 API and runs it. `rust-old/src/lib.rs` doesn't take `README.md` as the")
md.append("crate's docs, so the README's examples aren't doc tests.")
md.append("")
md.append("| Rust fence | What it shows | Cyrius test |")
md.append("|---|---|---|")
for rel, line, info, what, cy, name in doc_rows:
    md.append(f"| `rust-old/{rel}:{line}` (`{info}`) | {what} | `{cy}`, `{name}` |")
md.append("")
md.append("## Rust's `#[ignore]`d tests")
md.append("")
if ignored:
    for rel, name, where in ignored:
        md.append(f"- `{rel}` `{name}`: " + (f"ported, live, in {cell(where)}." if where else "not ported."))
else:
    md.append("None: pravash 1.3.0 ignores no unit or integration test.")
md.append("")
md.append("## Tests without a port")
md.append("")
md.append("| Rust file | Test | Status |")
md.append("|---|---|---|")
for rel, t, reason in notes:
    md.append(f"| `{rel}` | `{t}` | {reason} |")
for rel, reason in WHOLE_FILE.items():
    md.append(f"| `{rel}` | all {len(rust_tests(rel))} | {reason} |")
md.append("")
text = "\n".join(md)

for e in errors:
    print("LEDGER", e)
if check:
    old = out_path.read_text() if out_path.exists() else ""
    if old != text:
        print(f"STALE    {out_path.relative_to(root)} (regenerate: python3 -I -B scripts/rust_test_ledger.py .)")
        errors.append("stale")
elif not errors:
    out_path.write_text(text)
print(f"rust-test ledger: {total} tests — {sums['ported']} ported, {sums['n/a']} n/a, {sums['deferred']} deferred; "
      f"{len(errors)} problem(s)")
sys.exit(1 if errors else 0)
