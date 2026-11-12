#!/usr/bin/env python3
"""Independent check of the pravash golden corpus (port plan §4 step 5).

Python, standard library only; run with `python3 -I scripts/oracle/validate.py .`. It never calls
the Rust oracle, so it catches a generator that records the wrong thing:

- every `tests/golden/*.tsv` parses: the five-line header, the row count, five tab fields per
  row, a class in A–D, well-formed tokens; `modules.txt` lists exactly the files present;
- `constants.tsv` matches the literals in `rust-old/src` (the `pub const` items, and the fields of
  the `FluidMaterial` presets) and Rust's std float constants (`f64::MAX`, …, `f32::EPSILON`);
- every class-A and class-B row of `math.tsv` is recomputed in Python, in Rust's operation order:
  sqrt, abs, floor, ceil, min, max, clamp, is_nan, is_finite, the saturating casts (`as usize`,
  `as u32`), `as f32`, the SPH cell key, to_radians, powi2, powi8;
- every class-C row of `math.tsv` from glibc is recomputed by calling the same glibc functions the
  oracle imports (`sin`, `cos`, `atan2`, `hypot`, `exp`, `log`, `log10`, `pow`) through ctypes, so
  the generator's plumbing is checked against libm directly. This is an independent recomputation
  of what was recorded, not a check of glibc itself: that comparison is the step 8 baselines';
- `f64::cbrt` is not glibc's (rustc links compiler_builtins' own into the binary, port-parity §3),
  so each recorded value is proven correctly rounded instead, with exact rational arithmetic.

The correctly rounded math (ADR 0004) adds three generated outputs, which `--write` writes and a plain
run checks are current:
- `src/crmath_data.cyr`, the tables of src/crmath.cyr (scripts/oracle/crtables.py), whose π and ln 2
  and every table's head are checked against MPFR;
- `src/crlog_data.cyr`, the tables of src/crlog.cyr (scripts/oracle/lntables.py, which asserts the
  properties the ln error analysis relies on);
- the corrected oracle, `tests/golden/<module>.cr.tsv`: each row whose outputs change when every
  exp, ln, log10, sin, cos, hypot, pow and atan2 result is correctly rounded, with the oracle's
  outputs and each misrounded call in its note.
MPFR (libmpfr.so.6, 4.x) is the one system library this needs besides libm, through ctypes
(scripts/oracle/mpfr_ref.py).

Exit status 1 on any failure. Adapted from impetus's validate.py.
"""

import ctypes
import ctypes.util
import importlib.util
from fractions import Fraction
import math
import re
import struct
import sys
from pathlib import Path

WRITE = "--write" in sys.argv[2:]
sys.dont_write_bytecode = True   # the modules loaded below leave no __pycache__ in the repo


def _load(name):
    """A module beside this script, by path: `-I` keeps the script's directory off sys.path."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MPFR = _load("mpfr_ref")
CRTABLES = _load("crtables")
LNTABLES = _load("lntables")

FAILS = []


def fail(msg):
    FAILS.append(msg)


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def same(a, b):
    """Class A: identical bits, or both NaN."""
    if math.isnan(a) and math.isnan(b):
        return True
    return bits(a) == bits(b)


TOKEN = re.compile(r"^(f:[0-9a-f]{16}|i:-?[0-9]+|e:\S+|s:\S*|n:|E:\S+|v:[0-9]+)$")


def parse_tokens(field, where):
    if field == "-":
        return []
    toks = field.split(" ")
    for t in toks:
        if not TOKEN.match(t):
            fail(f"{where}: malformed token {t!r}")
    return toks


def tok_f(t):
    return from_bits(int(t[2:], 16))


def tok_i(t):
    return int(t[2:])


def unpct(s):
    return re.sub(r"%([0-9A-F]{2})", lambda m: chr(int(m.group(1), 16)), s)


def read_module(path):
    """The rows of one golden file, as (fn, class, ins, outs, note) tuples."""
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    head = lines[:5]
    name = path.stem
    if len(head) < 5 or not head[0].startswith(f"# pravash golden corpus, format 1: {name}"):
        fail(f"{path.name}: bad header line 1")
    if len(head) < 5 or "libm: " not in head[1]:
        fail(f"{path.name}: the oracle line names no libm platform")
    m = re.match(r"^# rows: ([0-9]+)$", head[4]) if len(head) == 5 else None
    rows = []
    for n, line in enumerate(lines[5:], start=6):
        cols = line.split("\t")
        where = f"{path.name}:{n}"
        if len(cols) != 5:
            fail(f"{where}: {len(cols)} fields, not 5")
            continue
        func, cls, ins, outs, note = cols
        if cls not in "ABCD" or len(cls) != 1:
            fail(f"{where}: class {cls!r}")
        rows.append((func, cls, parse_tokens(ins, where), parse_tokens(outs, where), where))
    if not m:
        fail(f"{path.name}: no row-count line")
    elif int(m.group(1)) != len(rows):
        fail(f"{path.name}: header says {m.group(1)} rows, file has {len(rows)}")
    return rows


# ── constants ──────────────────────────────────────────────────────────────────────────────

SCALAR_HOMES = {
    "buoyancy.rs": "buoyancy::DragCoefficient",
    "coupling.rs": "coupling::AddedMassCoefficient",
}


def library_part(src):
    at = src.find("#[cfg(test)]\nmod tests")
    return src if at < 0 else src[:at]


def rust_constants(root):
    """path -> value, read from the Rust literals."""
    out = {}
    src_dir = root / "rust-old" / "src"
    for file, home in SCALAR_HOMES.items():
        src = library_part((src_dir / file).read_text())
        for name, lit in re.findall(r"pub const (\w+): f64 = ([0-9_.eE+-]+);", src):
            out[f"{home}::{name}"] = float(lit.replace("_", ""))
    src = library_part((src_dir / "common.rs").read_text())
    for name, body in re.findall(r"pub const (\w+): Self = Self \{(.*?)\};", src, re.S):
        for field, lit in re.findall(r"(\w+): ([0-9_.eE+-]+),", body):
            out[f"common::FluidMaterial::{name}.{field}"] = float(lit.replace("_", ""))
    return out


# The Rust std float constants pravash's code uses, as Rust defines them.
STD_CONSTANTS = {
    "f64::MAX": sys.float_info.max,
    "f64::INFINITY": math.inf,
    "f64::NEG_INFINITY": -math.inf,
    "f64::NAN": math.nan,
    "f64::consts::PI": math.pi,
    "f32::EPSILON": 2.0**-23,
}


def check_constants(root, rows):
    want = {**rust_constants(root), **STD_CONSTANTS}
    seen = set()
    for func, cls, ins, outs, where in rows:
        path = unpct(ins[0][2:])
        seen.add(path)
        if func != "const" or cls != "A":
            fail(f"{where}: not a class-A const row")
        if path not in want:
            fail(f"{where}: {path} is not a pub const in rust-old/src")
        elif not same(tok_f(outs[0]), want[path]):
            fail(f"{where}: {path} = {tok_f(outs[0])!r}, the Rust literal is {want[path]!r}")
    for path in sorted(set(want) - seen):
        fail(f"constants.tsv: no row for {path}")
    return len(rows)


# ── math ───────────────────────────────────────────────────────────────────────────────────

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
for _name, _arity in [("sin", 1), ("cos", 1), ("exp", 1), ("log", 1), ("log10", 1),
                      ("floor", 1), ("ceil", 1), ("atan2", 2), ("hypot", 2),
                      ("pow", 2)]:
    _fn = getattr(_libm, _name)
    _fn.restype = ctypes.c_double
    _fn.argtypes = [ctypes.c_double] * _arity


def r_min(x, y):
    """Rust's f64::min as rustc 1.99.1 lowers it: a NaN operand is ignored; equal → y."""
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return x if x < y else y


def r_max(x, y):
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return x if x > y else y


def r_clamp(x, lo, hi):
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x


def r_sqrt(x):
    if math.isnan(x) or x < 0:
        return math.nan
    return math.sqrt(x)


def r_trunc_int(x):
    """The integer toward zero of a finite x."""
    return math.trunc(x)


def r_as_usize(x):
    """`x as usize` as the 64-bit word, read as i64 (the generator's `i:` token)."""
    if math.isnan(x):
        return 0
    if x == math.inf:
        w = 2**64 - 1
    elif x == -math.inf:
        w = 0
    else:
        t = r_trunc_int(x)
        w = 0 if t < 0 else min(t, 2**64 - 1)
    return w - 2**64 if w >= 2**63 else w


def r_as_u32(x):
    if math.isnan(x) or x == -math.inf:
        return 0
    if x == math.inf:
        return 2**32 - 1
    t = r_trunc_int(x)
    return 0 if t < 0 else min(t, 2**32 - 1)


def f32(x):
    """f64 → f32, round to nearest even, overflow to ±∞ (the C conversion), widened back."""
    return ctypes.c_float(x).value


def r_as_i32(x):
    if math.isnan(x):
        return 0
    if x == math.inf:
        return 2**31 - 1
    if x == -math.inf:
        return -(2**31)
    t = r_trunc_int(x)
    return max(-(2**31), min(t, 2**31 - 1))


def r_cell_key(p, h):
    """floor(p_f32 · (1 / h_f32)) as i32, all in f32. Each f32 operation is the f64 operation
    rounded once to f32, which is exact for ÷ and × of f32 operands (53 ≥ 2·24 + 2)."""
    h32 = f32(h)
    inv = f32(1.0 / h32)
    p32 = f32(p)
    prod = f32(p32 * inv)
    return r_as_i32(_libm.floor(prod))


def cbrt_correctly_rounded(x, y):
    """Whether y is cbrt(x) correctly rounded to nearest (ties to even), decided exactly.

    The oracle's `f64::cbrt` is not glibc's: rustc links compiler_builtins' own `cbrt` into the
    binary (port-parity §3), so this can't be a ctypes call. It is checked against the definition
    instead: |x| must lie between the cubes of the midpoints to y's neighbours."""

    if math.isnan(x):
        return math.isnan(y)
    if x == 0 or math.isinf(x):
        return bits(x) == bits(y)
    if math.isnan(y) or math.isinf(y) or y == 0 or (x < 0) != (y < 0):
        return False
    ax, ay = abs(x), abs(y)
    lo = (Fraction(math.nextafter(ay, 0.0)) + Fraction(ay)) / 2
    hi = (Fraction(ay) + Fraction(math.nextafter(ay, math.inf))) / 2
    fx = Fraction(ax)
    if lo**3 < fx < hi**3:
        return True
    if fx in (lo**3, hi**3):  # an exact tie: the even neighbour wins
        return bits(ay) % 2 == 0
    return False


def r_powi8(x):
    x2 = x * x
    x4 = x2 * x2
    return x4 * x4


UNARY_F = {
    "f64::sqrt": r_sqrt,
    "f64::abs": math.fabs,
    "f64::floor": _libm.floor,
    "f64::ceil": _libm.ceil,
    "f64::as_f32": f32,
    "f64::to_radians": lambda x: x * (math.pi / 180.0),
    "f64::powi2": lambda x: x * x,
    "f64::powi8": r_powi8,
    "f64::sin": _libm.sin,
    "f64::cos": _libm.cos,
    "f64::exp": _libm.exp,
    "f64::ln": _libm.log,
    "f64::log10": _libm.log10,
}
BINARY_F = {
    "f64::max": r_max,
    "f64::min": r_min,
    "f64::atan2": _libm.atan2,
    "f64::hypot": _libm.hypot,
    "f64::powf": _libm.pow,
}
UNARY_I = {
    "f64::is_finite": lambda x: int(math.isfinite(x)),
    "f64::is_nan": lambda x: int(math.isnan(x)),
    "f64::as_usize": r_as_usize,
    "f64::as_u32": r_as_u32,
}
CLASS = {
    **dict.fromkeys(["f64::sqrt", "f64::abs", "f64::floor", "f64::ceil", "f64::as_f32",
                     "f64::max", "f64::min", "f64::clamp", "f64::is_finite", "f64::is_nan",
                     "f64::as_usize", "f64::as_u32", "oracle::sph_cell_key"], "A"),
    **dict.fromkeys(["f64::to_radians", "f64::powi2", "f64::powi8"], "B"),
    **dict.fromkeys(["f64::sin", "f64::cos", "f64::sin_cos", "f64::exp", "f64::ln",
                     "f64::log10", "f64::cbrt", "f64::atan2", "f64::hypot", "f64::powf"], "C"),
}


def check_math(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        if CLASS.get(func) != cls:
            fail(f"{where}: {func} is class {cls}, expected {CLASS.get(func)}")
            continue
        xs = [tok_f(t) for t in ins]
        if func in UNARY_F:
            want = UNARY_F[func](xs[0])
            got = tok_f(outs[0])
            ok = same(got, want)
        elif func in BINARY_F:
            want = BINARY_F[func](xs[0], xs[1])
            got = tok_f(outs[0])
            ok = same(got, want)
        elif func in UNARY_I:
            want = UNARY_I[func](xs[0])
            got = tok_i(outs[0])
            ok = got == want
        elif func == "f64::clamp":
            want = r_clamp(*xs)
            got = tok_f(outs[0])
            ok = same(got, want)
        elif func == "f64::sin_cos":
            want = (_libm.sin(xs[0]), _libm.cos(xs[0]))
            got = (tok_f(outs[0]), tok_f(outs[1]))
            ok = same(got[0], want[0]) and same(got[1], want[1])
        elif func == "f64::cbrt":
            want = "correctly rounded"
            got = tok_f(outs[0])
            ok = cbrt_correctly_rounded(xs[0], got)
        elif func == "oracle::sph_cell_key":
            want = r_cell_key(xs[0], xs[1])
            got = tok_i(outs[0])
            ok = got == want
        else:
            fail(f"{where}: no check for {func}")
            continue
        if not ok:
            fail(f"{where}: {func}{tuple(xs)} recorded {got!r}, recomputed {want!r}")
    return counts


# ── the correctly rounded math (ADR 0004) ─────────────────────────────────────────────────────
# The corrected oracle: the oracle's own operations with every exp, ln, log10, sin, cos, hypot, pow and
# atan2 result correctly rounded (MPFR). Where glibc rounded every call correctly the row is unchanged
# and nothing is written; where it didn't, tests/golden/<module>.cr.tsv carries the corrected outputs.
# cbrt is not here: the oracle's (compiler_builtins') is already correctly rounded, checked above.

CR_UNARY = {"f64::exp": "exp", "f64::ln": "log", "f64::log10": "log10", "f64::sin": "sin",
            "f64::cos": "cos"}
CR_BINARY = {"f64::hypot": "hypot", "f64::powf": "pow", "f64::atan2": "atan2"}


def corrections(module, rows):
    """(fn, inputs text, corrected outputs text, note) for every row of `module` whose outputs change
    when every transcendental result is correctly rounded."""
    out = []
    for fn, cls, ins, outs, where in rows:
        log = []
        if module != "math":
            continue
        if fn in CR_BINARY:
            nm = CR_BINARY[fn]
            x, y = tok_f(ins[0]), tok_f(ins[1])
            want = [MPFR.cr2(nm, x, y)]
            calls = [(f"{nm}(f:{bits(x):016x}, f:{bits(y):016x})", tok_f(outs[0]), want[0])]
        elif fn in CR_UNARY:
            nm = CR_UNARY[fn]
            x = tok_f(ins[0])
            want = [MPFR.cr(nm, x)]
            calls = [(f"{nm}(f:{bits(x):016x})", tok_f(outs[0]), want[0])]
        elif fn == "f64::sin_cos":
            x = tok_f(ins[0])
            want = [MPFR.cr("sin", x), MPFR.cr("cos", x)]
            calls = [(f"sin(f:{bits(x):016x})", tok_f(outs[0]), want[0]),
                     (f"cos(f:{bits(x):016x})", tok_f(outs[1]), want[1])]
        else:
            continue
        for call, glibc, cr in calls:
            if not same(glibc, cr):
                log.append(f"{call} glibc f:{bits(glibc):016x} correctly rounded f:{bits(cr):016x}")
        got = [tok_f(t) for t in outs]
        if len(got) == len(want) and all(same(a, b) for a, b in zip(got, want)):
            continue
        new = " ".join(f"f:{bits(w):016x}" for w in want)
        out.append((fn, " ".join(ins), new, "oracle " + " ".join(outs) + "; " + "; ".join(log)))
    return out


def cr_text(module, golden_header, rows):
    """The text of tests/golden/<module>.cr.tsv, or None when no row needs a correction."""
    fixes = corrections(module, rows)
    if not fixes:
        return None
    oracle = next(l for l in golden_header if l.startswith("# oracle:"))
    oracle = oracle.replace(" Generated by scripts/oracle; do not edit.", "")
    head = [
        f"# pravash corrected oracle (ADR 0004), format 1: {module}",
        f"{oracle} Correctly rounded: MPFR {MPFR.VERSION}. Generated by scripts/oracle/validate.py; do not edit.",
        "# regenerate: python3 -I scripts/oracle/validate.py . --write",
        "# columns: fn<TAB>inputs<TAB>outputs<TAB>note (fn, inputs: the golden row's, byte for byte; outputs: "
        "the oracle's operations with every exp, ln, log10, sin, cos, hypot, pow and atan2 result correctly "
        "rounded; note: the oracle's outputs, then each call glibc misrounded)",
        f"# rows: {len(fixes)}",
    ]
    return "\n".join(head + ["\t".join(f) for f in fixes]) + "\n"


def check_corrections(root, module, path_tsv, rows):
    """tests/golden/<module>.cr.tsv is current (written with --write)."""
    header = path_tsv.read_text().splitlines()[:5]
    text = cr_text(module, header, rows)
    path = path_tsv.with_name(f"{module}.cr.tsv")
    rel = f"tests/golden/{module}.cr.tsv"
    if WRITE:
        if text is None:
            if path.exists():
                path.unlink()
                print(f"removed {rel}")
        else:
            path.write_text(text)
            print(f"wrote {rel} ({text.count(chr(10)) - 5} rows)")
        return
    if text is None:
        if path.exists():
            fail(f"{rel}: no row needs a correction, but the file exists (regenerate with --write)")
        return
    if not path.exists() or path.read_text() != text:
        fail(f"{rel}: differs from the corrected oracle (regenerate with --write)")
        return
    print(f"{module}.cr.tsv: current, {text.count(chr(10)) - 5} corrected rows")


def _write_or_check(root, rel, text):
    path = root / rel
    if WRITE:
        path.write_text(text)
        print(f"wrote {rel} ({len(text.splitlines())} lines)")
        return False
    if not path.exists() or path.read_text() != text:
        fail(f"{rel} is not its generator's output (regenerate with --write)")
        return False
    return True


def _atanh_inv(n, prec):
    """atanh(1/n) to `prec` fraction bits (Σ 1/((2k + 1)·n^(2k + 1)), each term truncated), as a Fraction."""
    one = 1 << (prec + 64)
    x = one // n
    total = x
    k = 1
    while x:
        x //= n * n
        total += x // (2 * k + 1)
        k += 1
    return Fraction(total >> 64, 1 << prec)


def check_tables(root):
    """src/crmath_data.cyr and src/crlog_data.cyr are their generators' renderings, and crmath's π and ln 2
    and every table's head are MPFR's."""
    n = CRTABLES.PREC
    for name, fn in (("pi", CRTABLES.pi_bits), ("log2", CRTABLES.ln2_bits)):
        if fn(n) != MPFR.const_bits(name, n):
            fail(f"crtables: {name} to {n} bits differs from MPFR's")
    t = CRTABLES.tables()
    for j, (hi, lo) in enumerate(t["exp2"]):
        if hi != MPFR.bits(MPFR.cr("exp2", j / 64)):
            fail(f"crtables: 2^({j}/64) hi is not MPFR's correctly rounded value")
    for i, (sh, sl, ch, cl) in enumerate(t["sincos"]):
        if sh != MPFR.bits(MPFR.cr("sin", i / 64)) or ch != MPFR.bits(MPFR.cr("cos", i / 64)):
            fail(f"crtables: sin or cos of {i}/64 hi is not MPFR's correctly rounded value")
    for i, (hi, lo) in enumerate(t["atan"]):
        if hi != MPFR.bits(MPFR.cr("atan", i / 64)):
            fail(f"crtables: atan({i}/64) hi is not MPFR's correctly rounded value")
    for name, want in (("PI_HI", MPFR.atan2(0.0, -1.0)), ("PIO2_HI", MPFR.atan2(1.0, 0.0)),
                       ("PIO4", MPFR.atan2(1.0, 1.0)), ("PI3O4", MPFR.atan2(1.0, -1.0))):
        if t[name] != MPFR.bits(want):
            fail(f"crtables: {name} is not MPFR's correctly rounded value")
    for j, (c, lh, lm, ll) in enumerate(LNTABLES.ln_table()):
        if j != 256 and LNTABLES.bits(lh) != MPFR.bits(-MPFR.cr("log", c)):
            fail(f"lntables: −ln c_{j} hi is not MPFR's correctly rounded value")
    # ln 2 and 1/ln 10 to 2^-150 relative, against integer series (ln 10 = 3·ln 2 + 2·atanh(1/9)): MPFR's
    # constants, as the ln tables use them, checked by arithmetic MPFR doesn't share.
    k = LNTABLES.ln_consts()
    ln2 = Fraction(CRTABLES.ln2_bits(), 1 << CRTABLES.PREC)
    ln10 = 3 * ln2 + 2 * _atanh_inv(9, CRTABLES.PREC)
    for name, parts, exact in (("ln 2", ("LN2H", "LN2M", "LN2L"), ln2), ("1/ln 10", ("IVH", "IVM", "IVL"), 1 / ln10)):
        approx = sum((Fraction(k[p]) for p in parts), Fraction(0))
        if abs(approx - exact) > exact / (1 << 150):
            fail(f"lntables: {name} as {' + '.join(parts)} is off by more than 2^-150 relative")
    ok = _write_or_check(root, "src/crmath_data.cyr", CRTABLES.render())
    ok = _write_or_check(root, "src/crlog_data.cyr", LNTABLES.render()) and ok
    if ok:
        print(f"crmath_data, crlog_data: current; π and ln 2 to {n} bits, and every table's head, equal "
              f"MPFR {MPFR.VERSION}'s")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    golden = root / "tests" / "golden"
    files = sorted(p for p in golden.glob("*.tsv") if not p.name.endswith(".cr.tsv"))
    listed = [l for l in (golden / "modules.txt").read_text().split("\n") if l and not l.startswith("#")]
    if sorted(listed) != sorted(p.stem for p in files):
        fail(f"modules.txt lists {listed}, the files are {[p.stem for p in files]}")
    total = 0
    for path in files:
        rows = read_module(path)
        total += len(rows)
        if path.stem == "constants":
            n = check_constants(root, rows)
            print(f"constants: {n} rows checked against the Rust literals and std")
        elif path.stem == "math":
            counts = check_math(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"math: {len(rows)} rows recomputed ({summary})")
            check_corrections(root, "math", path, rows)
        else:
            print(f"{path.stem}: {len(rows)} rows parsed (no recomputation yet)")
    for path in sorted(golden.glob("*.cr.tsv")):
        if not (golden / path.name.replace(".cr.tsv", ".tsv")).exists():
            fail(f"{path.name}: no golden module of that name")
    check_tables(root)
    for msg in FAILS[:40]:
        print("FAIL", msg)
    if len(FAILS) > 40:
        print(f"... and {len(FAILS) - 40} more")
    print(f"{total} rows, {len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
