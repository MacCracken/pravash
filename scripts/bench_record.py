"""The release benchmark record: Rust 1.3.0's criterion suite beside the Cyrius bench suite (port plan M8 part b).

Usage, from the repo root, on the saved outputs of interleaved runs (each Rust run beside a Cyrius one):
    python3 -I -B scripts/bench_record.py --rust rust_1.txt rust_2.txt rust_3.txt --cyrius cy_1.txt cy_2.txt cy_3.txt

A Rust file is the stdout of
    taskset -c 15 cargo bench --locked --manifest-path rust-old/Cargo.toml --features "sph grid shallow coupling" \
        --bench benchmarks -- --noplot
and a Cyrius file that of `taskset -c 15 cyrius bench`. It prints the markdown table of every criterion row, in
`rust-old/benches/benchmarks.rs`'s order, with each side's range over the runs (criterion's point estimate; the
Cyrius mean) and the ratio of the medians. It fails when a criterion row is missing from any run of either side, so
the table always covers the whole suite.

    python3 -I -B scripts/bench_record.py --threads <dir> <rounds>

prints the thread-count table from a directory of runs on cores 8–15, round r of each named
- `rust_grid_t<T>_<r>.txt`: `RAYON_NUM_THREADS=<T> taskset -c 8-15 cargo bench ... --features "sph grid shallow
  coupling parallel" --bench benchmarks -- --noplot grid_step`;
- `rust_sph_t<T>_<r>.txt`: `RAYON_NUM_THREADS=<T> taskset -c 8-15 cargo run --release --locked --features parallel
  --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_m5e`;
- `cy_par_<r>.txt`: `taskset -c 8-15 cyrius bench`, whose `*_t<T>` rows are the solvers at T threads;

for T in 1, 2, 4, 8.
"""
import re
import statistics
import sys

UNITS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}


def rust_rows(path):
    """{name: ns} from a criterion run: the middle of `time: [lo mid hi]`, the name on the same line or the one
    before (criterion wraps a long name)."""
    out = {}
    order = []
    prev = ""
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        m = re.match(r"^(\S*)\s+time:\s+\[(\S+) (\S+) (\S+) (\S+) (\S+) (\S+)\]", line)
        if m:
            name = m.group(1) or prev
            out[name] = float(m.group(4)) * UNITS[m.group(5)]
            order.append(name)
        elif line and not line[0].isspace():
            prev = line.strip()
    return out, order


def cyrius_rows(path):
    """{name: ns} from a Cyrius bench run: `  name: <mean><unit> avg ...`."""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s+(\S+): ([\d.]+)(ns|us|ms|s) avg", line)
        if m:
            out[m.group(1)] = float(m.group(2)) * UNITS[m.group(3)]
    return out


def twin_rows(path):
    """{name: ns} from a Rust twin run: `  name   <ns> ns`."""
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s+(\S+/\S+)\s+([\d.]+) ns$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def fmt(ns):
    """A time with three significant digits in the unit criterion would print."""
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)):
        if ns >= scale or unit == "ns":
            v = ns / scale
            digits = 2 if v < 10 else 1 if v < 100 else 0
            return f"{v:.{digits}f} {unit}"
    return f"{ns:.2f} ns"


def span(values):
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo.split()[0]}–{hi}" if lo.split()[1] == hi.split()[1] else f"{lo}–{hi}"


THREADS = (1, 2, 4, 8)

# (row, Rust source, Rust name, Cyrius name with {t} for the thread count); a None Rust name is a Cyrius-only row.
THREAD_ROWS = [
    ("`grid_step/63x63` (DST)", "grid", "grid_step/63x63", "grid_step_t{t}/63x63"),
    ("`grid_step_periodic/63x63`", "grid", "grid_step_periodic/63x63", "grid_step_periodic_t{t}/63x63"),
    ("`grid_step_periodic/255x255` (Cyrius only)", "grid", None, "grid_step_periodic_t{t}/255x255"),
    ("`sph_m5e/step_625`", "sph", "sph_m5e/step_625", "sph_m5e/step_625_t{t}"),
    ("`sph_m5e/multiphase_625`", "sph", "sph_m5e/multiphase_625", "sph_m5e/multiphase_625_t{t}"),
    ("`sph_m5e/step_2500`", "sph", "sph_m5e/step_2500", "sph_m5e/step_2500_t{t}"),
]


def threads_table(d, rounds):
    rust = {(src, th): [] for src in ("grid", "sph") for th in THREADS}
    cy = []
    for r in range(1, rounds + 1):
        for th in THREADS:
            rust[("grid", th)].append(rust_rows(f"{d}/rust_grid_t{th}_{r}.txt")[0])
            rust[("sph", th)].append(twin_rows(f"{d}/rust_sph_t{th}_{r}.txt"))
        cy.append(cyrius_rows(f"{d}/cy_par_{r}.txt"))
    errors = []
    print("| Bench | | T = 1 | T = 2 | T = 4 | T = 8 | Speed-up at 8 |")
    print("|---|---|---|---|---|---|---|")
    for label, src, rname, cname in THREAD_ROWS:
        lines = []
        if rname is not None:
            vals = []
            for th in THREADS:
                got = [runs[rname] for runs in rust[(src, th)] if rname in runs]
                if len(got) != rounds:
                    errors.append(f"Rust {rname} at T = {th}: {len(got)} of {rounds} runs")
                    got = got or [float("nan")]
                vals.append(got)
            lines.append(("Rust", vals))
        vals = []
        for th in THREADS:
            name = cname.format(t=th)
            got = [runs[name] for runs in cy if name in runs]
            if len(got) != rounds:
                errors.append(f"Cyrius {name}: {len(got)} of {rounds} runs")
                got = got or [float("nan")]
            vals.append(got)
        lines.append(("Cyrius", vals))
        for k, (side, v) in enumerate(lines):
            gain = statistics.median(v[0]) / statistics.median(v[-1])
            cells = " | ".join(span(x) for x in v)
            print(f"| {label if k == 0 else ''} | {side} | {cells} | {gain:.2f}× |")
    for e in errors:
        print("RECORD", e, file=sys.stderr)
    return 1 if errors else 0


def main(argv):
    if argv[:1] == ["--threads"] and len(argv) == 3:
        return threads_table(argv[1], int(argv[2]))
    if "--rust" not in argv or "--cyrius" not in argv:
        print(__doc__)
        return 2
    r_at, c_at = argv.index("--rust"), argv.index("--cyrius")
    rust_files = argv[r_at + 1:c_at] if r_at < c_at else argv[r_at + 1:]
    cy_files = argv[c_at + 1:] if r_at < c_at else argv[c_at + 1:r_at]
    rust = [rust_rows(f) for f in rust_files]
    cy = [cyrius_rows(f) for f in cy_files]
    order = rust[0][1]
    errors = []
    for k, (rows, _) in enumerate(rust):
        if set(rows) != set(order):
            errors.append(f"Rust run {rust_files[k]} has rows {sorted(set(rows) ^ set(order))} unlike the first")
    for k, rows in enumerate(cy):
        missing = [n for n in order if n not in rows]
        if missing:
            errors.append(f"Cyrius run {cy_files[k]} lacks {missing}")
    if errors:
        for e in errors:
            print("RECORD", e, file=sys.stderr)
        return 1
    print("| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |")
    print("|---|---|---|---|")
    ratios = []
    for name in order:
        rv = [rows[name] for rows, _ in rust]
        cv = [rows[name] for rows in cy]
        ratio = statistics.median(cv) / statistics.median(rv)
        ratios.append(ratio)
        shown = f"{ratio:.2f}×" if ratio < 10 else f"{ratio:.1f}×" if ratio < 100 else f"{ratio:.0f}×"
        print(f"| `{name}` | {span(rv)} | {span(cv)} | {shown} |")
    geo = statistics.geometric_mean(ratios)
    print()
    print(f"{len(order)} rows over {len(rust_files)} Rust and {len(cy_files)} Cyrius runs. Cyrius / Rust: "
          f"median {statistics.median(ratios):.2f}×, geometric mean {geo:.2f}×, from {min(ratios):.2f}× to "
          f"{max(ratios):.1f}×.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
