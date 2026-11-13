"""Compiles and runs every `cyrius` code block in the README and the guides, and the ported examples, against the
shipped bundle (port plan M8 part c; impetus's scripts/doc_examples.py with pravash's names).

The Cyrius form of Rust's doc tests and `cargo run --example`. Usage, from the repo root, after `cyrius distlib`:
    python3 -I -B scripts/doc_examples.py .

A block that defines `fn main` is a whole consumer program: its `include "lib/pravash.cyr"` is pointed at `dist/`,
where a consumer's `cyrius deps` would have vendored the bundle. Any other block is a fragment: it goes into
`fn main(): i64 { alloc_init(); <block> return 0; }` after the bundle's include. Each docs/examples/<name>.cyr is a
whole program too, and must print exactly docs/examples/<name>.out (scripts/oracle/examples.py derives those from
the oracle), be formatted, and lint clean (lint runs on the built copy, whose include points at dist/). Each program
is built with `cyrius build` into build/doc_examples/ and run. A compile warning, a failed build, a nonzero exit, an
example's other output, formatting or lint fails the check: exit 1, naming the block.
"""
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
DOCS = ["README.md"] + sorted(str(p.relative_to(root)) for p in (root / "docs/guides").glob("*.md"))
EXAMPLES = sorted(str(p.relative_to(root)) for p in (root / "docs/examples").glob("*.cyr"))
BUNDLE = "dist/pravash.cyr"
TRAILER = "\nvar r = main();\nsys_exit_group(r);\n"
# The stdlib resolver's notice on every build while hisab 3.4.1 (test scope) declares a sakshi package: the two are
# the same file, and only hisab can drop its copy (port-parity §28).
SAKSHI_LEAF_NOTICE = re.compile(r"refusing to overwrite stdlib leaf 'sakshi'")


def blocks(text):
    """(line number, code) of each ```cyrius block."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip() == "```cyrius":
            start = i + 1
            j = start
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            out.append((start + 1, "\n".join(lines[start:j])))
            i = j
        i += 1
    return out


def program(code):
    """The block as a whole program, or None with a reason."""
    if re.search(r"^fn main\(", code, re.M):
        src = code.replace('include "lib/pravash.cyr"', f'include "{BUNDLE}"')
        if BUNDLE not in src:
            return None, 'a whole program must include "lib/pravash.cyr"'
        if "sys_exit_group(" not in src:
            src += TRAILER
        return src, None
    body = "\n".join("    " + ln if ln else "" for ln in code.split("\n"))
    return f'include "{BUNDLE}"\n\nfn main(): i64 {{\n    alloc_init();\n{body}\n    return 0;\n}}\n' + TRAILER, None


def build_and_run(label, src, out_dir):
    """Builds and runs one program: (stdout, None), or (None, why it failed)."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", label)
    cyr, exe = out_dir / f"{stem}.cyr", out_dir / stem
    cyr.write_text(src)
    b = subprocess.run(["cyrius", "build", str(cyr.relative_to(root)), str(exe.relative_to(root))],
                       cwd=root, capture_output=True, text=True)
    log = [ln for ln in (b.stdout + b.stderr).split("\n") if not SAKSHI_LEAF_NOTICE.search(ln)]
    warn = [ln.strip() for ln in log if "warning" in ln.lower()]
    if b.returncode != 0 or warn:
        errs = [ln.strip() for ln in log if "error" in ln.lower()]
        detail = warn if b.returncode == 0 else (errs or [ln for ln in log if ln.strip()][-3:])
        what = "warned" if b.returncode == 0 else "failed"
        return None, f"build {what}: " + " | ".join(detail[:3])
    r = subprocess.run([str(exe)], cwd=root, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None, f"exit {r.returncode}"
    return r.stdout, None


def style(ex, out_dir):
    """None when the example is formatted (`cyrius fmt --check` on the file as written) and lint-clean (`cyrius lint`
    on the built copy, since lint compiles the file and `lib/pravash.cyr` exists only in a consumer); else why not."""
    f = subprocess.run(["cyrius", "fmt", "--check", ex], cwd=root, capture_output=True, text=True)
    if f.returncode != 0:
        return "not formatted (cyrius fmt --check)"
    copy = out_dir / (re.sub(r"[^A-Za-z0-9]+", "_", ex) + ".cyr")
    lint = subprocess.run(["cyrius", "lint", str(copy.relative_to(root))], cwd=root, capture_output=True, text=True)
    log = [ln for ln in (lint.stdout + lint.stderr).split("\n") if not SAKSHI_LEAF_NOTICE.search(ln)]
    if lint.returncode != 0 or not any(ln.strip() == "0 warnings" for ln in log):
        return "lint: " + " | ".join(ln.strip() for ln in log if ln.strip())[-200:]
    return None


def main():
    out_dir = root / "build/doc_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    total, failed = 0, []
    for doc in DOCS:
        path = root / doc
        if not path.exists():
            continue
        for line, code in blocks(path.read_text()):
            total += 1
            label = f"{doc}:{line}"
            src, why = program(code)
            if src is None:
                failed.append(f"{label}: {why}")
                continue
            out, why = build_and_run(label, src, out_dir)
            if why is not None:
                failed.append(f"{label}: {why}")
    for ex in EXAMPLES:
        total += 1
        src, why = program((root / ex).read_text())
        if src is None:
            failed.append(f"{ex}: {why}")
            continue
        out, why = build_and_run(ex, src, out_dir)
        want = root / ex.replace(".cyr", ".out")
        if why is None:
            why = style(ex, out_dir)
        if why is not None:
            failed.append(f"{ex}: {why}")
        elif not want.exists():
            failed.append(f"{ex}: no {want.relative_to(root)} to compare with")
        elif out != want.read_text():
            got = out.split("\n")
            exp = want.read_text().split("\n")
            k = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), min(len(got), len(exp)))
            failed.append(f"{ex}: prints otherwise than {want.relative_to(root)} at line {k + 1}")
    for f in failed:
        print("DOC", f)
    print(f"doc examples: {total} blocks and programs ({len(DOCS)} docs, {len(EXAMPLES)} examples), "
          f"{total - len(failed)} ran, {len(failed)} failed")
    return 1 if failed else 0


sys.exit(main())
