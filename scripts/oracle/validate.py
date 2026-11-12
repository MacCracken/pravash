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
import json
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


# ── common (M1 part 3) ─────────────────────────────────────────────────────────────────────
# Each row recomputed from rust-old/src/common.rs and glam 0.29.3, in Rust's operation order: the validation
# rules, the particle and config defaults, the vector arithmetic (no fused operations), and the arena, whose
# free-list logic is transcribed below. Token layouts: scripts/oracle/src/m_common.rs.

def fin(x):
    return math.isfinite(x)


def r_custom(d, v, s, c):
    if not fin(d) or d <= 0.0:
        return "E:InvalidDensity"
    if not fin(v) or v < 0.0:
        return "E:InvalidViscosity"
    if not fin(s) or s < 0.0:
        return "E:InvalidParameter"
    if not fin(c) or c <= 0.0:
        return "E:InvalidParameter"
    return [d, v, s, c]


def r_particle(pos, mass):
    return [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mass, 0, 1.0, 0.0, 1.0, 293.15, 0.0]


def dot3(x, y, z):
    return ((x * x) + (y * y)) + (z * z)


WATER_2D = [0.001, 0.0, -9.81, 0.0, 0.05, 2000.0, 1000.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5]


def r_validate(c):
    dt, gx, gy, gz, h, k, rho, lx, ly, lz, hx, hy, hz, bd = c
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    if not fin(h) or h <= 0.0:
        return "E:InvalidSmoothingRadius"
    if not fin(rho) or rho <= 0.0:
        return "E:InvalidDensity"
    if not fin(k) or k <= 0.0:
        return "E:InvalidParameter"
    if not (0.0 <= bd <= 1.0):
        return "E:InvalidParameter"
    if not (fin(gx) and fin(gy) and fin(gz)):
        return "E:InvalidParameter"
    if not all(fin(x) for x in (lx, ly, lz, hx, hy, hz)):
        return "E:InvalidParameter"
    if lx > hx or ly > hy or lz > hz:
        return "E:InvalidParameter"
    return None


def r_check_cfl(c, v):
    cfl = (abs(v) * c[0]) / c[4]
    if not fin(cfl) or cfl > 1.0:
        return "E:CflViolation"
    return None


class Arena:
    """rust-old/src/common.rs's ParticleArena, transcribed: slots hold only the mass the scenario writes."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.mass = [0.0] * capacity
        self.watermark = 0
        self.free_list = []
        self.generation = 0

    def active_count(self):
        return max(self.watermark - sum(l for _, l in self.free_list), 0)

    def alloc(self, n):
        if n == 0:
            return (0, 0, self.generation)
        if n < 0 or n >= 1 << 63:
            return None
        start = None
        for i, (s, l) in enumerate(self.free_list):
            if l >= n:
                start = s
                if l == n:
                    del self.free_list[i]
                else:
                    self.free_list[i] = (s + n, l - n)
                break
        if start is None:
            if n <= self.capacity - self.watermark:
                start = self.watermark
                self.watermark += n
            else:
                return None
        for j in range(start, start + n):
            self.mass[j] = 0.0
        return (start, n, self.generation)

    def in_range(self, h):
        s, n, g = h
        return g == self.generation and s + n <= self.watermark

    def free(self, h):
        s, n, g = h
        if n == 0 or not self.in_range(h):
            return
        end = s + n
        pos = 0
        while pos < len(self.free_list) and self.free_list[pos][0] < s:
            pos += 1
        if pos > 0 and sum(self.free_list[pos - 1]) > s:
            return
        if pos < len(self.free_list) and self.free_list[pos][0] < end:
            return
        self.free_list.insert(pos, (s, n))
        if pos + 1 < len(self.free_list):
            (s1, l1), (s2, l2) = self.free_list[pos], self.free_list[pos + 1]
            if s1 + l1 == s2:
                self.free_list[pos] = (s1, l1 + l2)
                del self.free_list[pos + 1]
        if pos > 0:
            (s0, l0), (s1, l1) = self.free_list[pos - 1], self.free_list[pos]
            if s0 + l0 == s1:
                self.free_list[pos - 1] = (s0, l0 + l1)
                del self.free_list[pos]
        if self.free_list and sum(self.free_list[-1]) == self.watermark:
            self.watermark = self.free_list.pop()[0]

    def get_len(self, h):
        return h[1] if self.in_range(h) else 0

    def compact(self):
        if not self.free_list:
            return self.watermark
        write = read = fi = 0
        while read < self.watermark:
            if fi < len(self.free_list) and read == self.free_list[fi][0]:
                read += self.free_list[fi][1]
                fi += 1
                continue
            self.mass[write] = self.mass[read]
            write += 1
            read += 1
        self.watermark = write
        self.free_list = []
        self.generation = (self.generation + 1) % (1 << 64)
        return write


def r_scenario(ins):
    a = Arena(ins[0])
    issued = []
    out = []
    ops = ins[1:]
    for k in range(len(ops) // 4):
        code, x, y, z = ops[4 * k: 4 * k + 4]
        if code == 0:
            h = a.alloc(x)
            if h is None:
                out += [0, 0, 0, 0]
            else:
                out += [1, h[0], h[1], h[2]]
                for j in range(h[0], h[0] + a.get_len(h)):
                    a.mass[j] = float(k + 1)
                issued.append(h)
        elif code in (1, 3):
            if not issued:
                out.append(-1)
                continue
            h = issued[x % len(issued)]
            if code == 1:
                a.free(h)
                out.append(a.active_count())
            else:
                out.append(a.get_len(h))
        elif code == 2:
            out.append(a.compact())
        else:
            h = (x, y, z)
            if code == 4:
                a.free(h)
                out.append(a.active_count())
            else:
                out.append(a.get_len(h))
    out += [a.watermark, a.generation, ("v", 2 * len(a.free_list))]
    for s, l in a.free_list:
        out += [s, l]
    out.append(("v", a.watermark))
    out += [("f", m) for m in a.mass[: a.watermark]]
    return out


def tok_val(t):
    """A token's value: floats and integers, `v:` as ("v", n), errors as their text."""
    if t.startswith("f:"):
        return tok_f(t)
    if t.startswith("i:"):
        return tok_i(t)
    if t.startswith("v:"):
        return ("v", int(t[2:]))
    return t


def same_list(got, want):
    """Token lists equal: floats by bits (NaN matches NaN), the rest exactly."""
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if isinstance(w, tuple) and w[0] == "f":
            w = w[1]
        if isinstance(w, float) or isinstance(g, float):
            if not (isinstance(g, float) and isinstance(w, float) and same(g, w)):
                return False
        elif g != w:
            return False
    return True


def check_common(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        if func == "FluidMaterial::custom":
            r = r_custom(*x)
            want = [r] if isinstance(r, str) else r
        elif func == "FluidParticle::new":
            want = r_particle(x[0:3], x[3])
        elif func == "FluidParticle::new_2d":
            want = r_particle([x[0], x[1], 0.0], x[2])
        elif func in ("FluidParticle::speed_squared", "FluidParticle::speed", "FluidParticle::kinetic_energy"):
            v2 = dot3(x[0], x[1], x[2])
            want = [{"FluidParticle::speed_squared": v2, "FluidParticle::speed": r_sqrt(v2),
                     "FluidParticle::kinetic_energy": (0.5 * x[3]) * v2}[func]]
        elif func in ("FluidParticle::distance_squared_to", "ParticleSoa::distance_squared",
                      "FluidParticle::distance_to"):
            d2 = dot3(x[0] - x[3], x[1] - x[4], x[2] - x[5])
            want = [r_sqrt(d2) if func == "FluidParticle::distance_to" else d2]
        elif func == "FluidConfig::water_2d":
            want = WATER_2D
        elif func == "FluidConfig::validate":
            r = r_validate(x)
            want = [] if r is None else [r]
        elif func == "FluidConfig::check_cfl":
            r = r_check_cfl(x[:14], x[14])
            want = [] if r is None else [r]
        elif func == "ParticleSoa::roundtrip":
            want = x
        elif func == "ParticleSoa::consistent_len":
            want = [x[0], min(x), int(all(l == x[0] for l in x)), min(x)]
        elif func == "ParticleArena::scenario":
            want = r_scenario(x)
        else:
            fail(f"{where}: no check for {func}")
            continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


# ── buoyancy, vortex, bridge (M1 part 4) ───────────────────────────────────────────────────
# Each function transcribed from rust-old/src/{buoyancy,vortex,bridge}.rs in Rust's operation order. A
# transcendental goes through `P`, a dict of primitives: glibc's (ctypes, the oracle's own libm) reproduces the
# golden row bit for bit, and MPFR's gives the corrected oracle (ADR 0004).

PI = math.pi           # std::f64::consts::PI


def fdiv(a, b):
    """IEEE a / b: Python raises on b = 0, where IEEE gives ±∞ (by the signs) or NaN (0/0, NaN/0)."""
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0.0:
            return math.nan
        return math.copysign(math.inf, math.copysign(1.0, a) * math.copysign(1.0, b))
GLIBC = {"exp": _libm.exp, "log": _libm.log, "log10": _libm.log10, "pow": _libm.pow}


def r_terminal(m, g, rho, cd, a):
    if not fin(g) or not fin(m):
        return "E:InvalidParameter"
    if not (fin(rho) and rho > 0.0 and fin(cd) and cd > 0.0 and fin(a) and a > 0.0):
        return "E:InvalidParameter"
    arg = fdiv((2.0 * m) * g, (rho * cd) * a)
    if arg < 0.0:
        return "E:InvalidParameter"
    return r_sqrt(arg)


def r_reynolds(rho, v, length, mu):
    if not fin(mu) or mu <= 0.0:
        return "E:InvalidViscosity"
    if not fin(rho) or not fin(v) or not fin(length):
        return "E:InvalidParameter"
    if rho < 0.0 or length < 0.0:
        return "E:InvalidParameter"
    return fdiv((rho * abs(v)) * length, mu)


def r_classify(re):
    a = abs(re)
    if a < 2300.0:
        return "e:FlowRegime.Laminar"
    if a < 4000.0:
        return "e:FlowRegime.Transitional"
    return "e:FlowRegime.Turbulent"


def r_lamb_oseen(gam, r, nu, t, P):
    if r <= 0.0:
        return 0.0
    pv = fdiv(gam, (2.0 * PI) * r)
    core = (4.0 * nu) * t
    if core <= 0.0:
        return pv
    return pv * (1.0 - P["exp"](fdiv(-(r * r), core)))


def r_rankine(gam, r, core):
    if r <= 0.0:
        return 0.0
    if r <= core:
        return fdiv(gam * r, ((2.0 * PI) * core) * core)
    return fdiv(gam, (2.0 * PI) * r)


def r_enstrophy(field, dx):
    dx2 = dx * dx
    s = -0.0           # Rust's f64 Sum folds from -0.0
    for w in field:
        s = s + (w * w) * dx2
    return s


def r_kolmogorov(nu, eps, P):
    if eps <= 0.0:
        return math.inf
    if nu <= 0.0:
        return 0.0
    return P["exp"]((0.75 * P["log"](nu)) - (0.25 * P["log"](eps)))


def r_water_viscosity(temp, P):
    tc = r_clamp(temp, 273.0, 373.0)
    return 2.414e-5 * P["pow"](10.0, fdiv(247.8, tc - 140.0))


def r_heat(q, vol, rho, cp, beta, dt):
    if not (vol > 0.0 and cp > 0.0 and rho > 0.0):
        return 0.0
    return fdiv(((9.81 * beta) * q) * dt, cp * vol)


def r_eddy(k, length):
    if k <= 0.0:
        return 0.0
    return fdiv((0.5477 * r_sqrt(k)) * length, 0.85)


def r_acoustic(v, c):
    if c <= 0.0:
        return 0.0
    return r_powi8(fdiv(abs(v), c))


def r_noise_db(k, P):
    if k <= 0.0:
        return 0.0
    u = r_sqrt((2.0 / 3.0) * k)
    if u <= 5e-8:
        return 0.0
    return 20.0 * P["log10"](fdiv(u, 5e-8))


def r_wave(h, freq, c):
    if c <= 0.0 or freq <= 0.0:
        return 0.0
    k = fdiv((2.0 * PI) * freq, c)
    return (((1.225 * 9.81) * abs(h)) * k) * abs(h)


# fn -> (inputs -> outputs). The class C ones take the primitives.
M1B = {
    "buoyancy_force": lambda x: [(x[0] * x[1]) * x[2]],
    "drag_force": lambda x: [((((0.5 * x[0]) * x[1]) * x[1]) * x[2]) * x[3]],
    "terminal_velocity": lambda x: [r_terminal(*x)],
    "reynolds_number": lambda x: [r_reynolds(*x)],
    "classify_flow": lambda x: [r_classify(x[0])],
    "vorticity_2d": lambda x: [x[0] - x[1]],
    "rankine_velocity": lambda x: [r_rankine(*x)],
    "enstrophy": lambda x: [r_enstrophy(x[1:1 + x[0][1]], x[1 + x[0][1]])],
    "cp_to_surface_pressure": lambda x: [x[2] + (x[0] * x[1])],
    "freestream_dynamic_pressure": lambda x: [((0.5 * x[1]) * x[0]) * x[0]],
    "aero_coefficients_to_forces": lambda x: [(x[0] * x[2]) * x[3], (x[1] * x[2]) * x[3]],
    "wind_to_freestream": lambda x: [r_clamp(x[0], -340.0, 340.0)],
    "heat_to_buoyancy_forcing": lambda x: [r_heat(*x)],
    "tke_to_eddy_diffusivity": lambda x: [r_eddy(*x)],
    "velocity_to_acoustic_source_strength": lambda x: [r_acoustic(*x)],
    "wave_height_to_acoustic_pressure": lambda x: [r_wave(*x)],
}
M1B_C = {
    "lamb_oseen_velocity": lambda x, P: [r_lamb_oseen(*x, P)],
    "kolmogorov_scale": lambda x, P: [r_kolmogorov(*x, P)],
    "temperature_to_water_viscosity": lambda x, P: [r_water_viscosity(x[0], P)],
    "tke_to_broadband_noise_db": lambda x, P: [r_noise_db(x[0], P)],
}


def check_m1b(rows):
    """buoyancy, vortex and bridge rows, recomputed (class C on glibc, as the oracle ran)."""
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        if func in M1B:
            want = M1B[func](x)
        elif func in M1B_C:
            want = M1B_C[func](x, GLIBC)
        else:
            fail(f"{where}: no check for {func}")
            continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


# ── json (M1 part 4) ───────────────────────────────────────────────────────────────────────
# Each document decoded again by Python's json module (its float parse is correctly rounded, as serde_json's
# `float_roundtrip` and bayan's are) under serde's derive semantics for these types: an object by field name
# (unknown fields ignored, a duplicated known field refused) or an array of exactly the fields; an f64 from a finite
# number (an integer `-0` is −0.0, as serde_json reads it), never null; a u8 from an integer in 0 … 255; a vector
# of exactly 3; a unit variant as "Name" or {"Name": null}; the unit struct as null.

class JsonRefused(Exception):
    pass


def _j_obj(pairs):
    return ("obj", pairs)


def _j_const(name):
    raise JsonRefused(name)


def j_load(doc):
    try:
        return json.loads(doc, object_pairs_hook=_j_obj, parse_int=lambda s: ("int", s),
                          parse_float=lambda s: ("float", s), parse_constant=_j_const)
    except (ValueError, JsonRefused):
        raise JsonRefused("not JSON")


def j_f64(v):
    if isinstance(v, tuple) and v[0] == "int":
        x = -0.0 if v[1] == "-0" else float(int(v[1]))
    elif isinstance(v, tuple) and v[0] == "float":
        x = float(v[1])
    else:
        raise JsonRefused("not a number")
    if not math.isfinite(x):
        raise JsonRefused("out of range")
    return x


def j_u8(v):
    if not (isinstance(v, tuple) and v[0] == "int"):
        raise JsonRefused("not an integer")
    n = int(v[1])
    if not 0 <= n <= 255:
        raise JsonRefused("out of range")
    return n


def j_arr(v, n=None):
    if not isinstance(v, list) or (n is not None and len(v) != n):
        raise JsonRefused("not an array of the length")
    return v


def j_struct(v, names):
    """The field values of struct node v, in declaration order."""
    if isinstance(v, list):
        return j_arr(v, len(names))
    if not (isinstance(v, tuple) and v[0] == "obj"):
        raise JsonRefused("not a struct")
    out = []
    for name in names:
        hits = [val for key, val in v[1] if key == name]
        if len(hits) != 1:
            raise JsonRefused("missing or duplicated field")
        out.append(hits[0])
    return out


MATERIAL_FIELDS = ["density", "viscosity", "surface_tension", "speed_of_sound"]
PARTICLE_FIELDS = ["position", "velocity", "acceleration", "density", "pressure", "mass", "phase",
                   "conformation", "temperature", "fuel"]
CONFIG_FIELDS = ["dt", "gravity", "smoothing_radius", "gas_constant", "rest_density", "bounds_min",
                 "bounds_max", "boundary_damping"]
SOA_FIELDS = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "accel_x", "accel_y", "accel_z",
              "density", "pressure", "mass", "phase", "conf_xx", "conf_xy", "conf_yy", "temperature", "fuel"]
REGIMES = ["Laminar", "Transitional", "Turbulent"]


def j_decode(func, doc):
    """The tokens serde_json reads from doc as func's type, or ["E:Deserialize"]."""
    try:
        v = j_load(doc)
        base = func.split("::")[0]
        if base == "FluidMaterial":
            return [j_f64(x) for x in j_struct(v, MATERIAL_FIELDS)]
        if base == "FluidParticle":
            f = j_struct(v, PARTICLE_FIELDS)
            out = []
            for k in range(3):
                out += [j_f64(x) for x in j_arr(f[k], 3)]
            out += [j_f64(f[3]), j_f64(f[4]), j_f64(f[5]), j_u8(f[6])]
            out += [j_f64(x) for x in j_arr(f[7], 3)]
            return out + [j_f64(f[8]), j_f64(f[9])]
        if base == "FluidConfig":
            f = j_struct(v, CONFIG_FIELDS)
            out = [j_f64(f[0])] + [j_f64(x) for x in j_arr(f[1], 3)]
            out += [j_f64(f[2]), j_f64(f[3]), j_f64(f[4])]
            out += [j_f64(x) for x in j_arr(f[5], 3)] + [j_f64(x) for x in j_arr(f[6], 3)]
            return out + [j_f64(f[7])]
        if base == "ParticleSoa":
            out = []
            for k, col in enumerate(j_struct(v, SOA_FIELDS)):
                vals = [j_u8(x) if k == 12 else j_f64(x) for x in j_arr(col)]
                out += [("v", len(vals))] + vals
            return out
        if base == "FlowRegime":
            if isinstance(v, str) and v in REGIMES:
                return [f"e:FlowRegime.{v}"]
            if isinstance(v, tuple) and v[0] == "obj" and len(v[1]) == 1:
                key, val = v[1][0]
                if key in REGIMES and val is None:
                    return [f"e:FlowRegime.{key}"]
            raise JsonRefused("no variant")
        if base == "DragCoefficient":
            if v is None:
                return []
            raise JsonRefused("not null")
    except JsonRefused:
        return ["E:Deserialize"]
    raise ValueError(func)


def check_json(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        doc = unpct(ins[0][2:])
        got = [tok_val(t) for t in outs]
        want = j_decode(func, doc)
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, decoded {want!r}")
    return counts


# ── the correctly rounded math (ADR 0004) ─────────────────────────────────────────────────────
# The corrected oracle: the oracle's own operations with every exp, ln, log10, sin, cos, hypot, pow and
# atan2 result correctly rounded (MPFR). Where glibc rounded every call correctly the row is unchanged
# and nothing is written; where it didn't, tests/golden/<module>.cr.tsv carries the corrected outputs.
# cbrt is not here: the oracle's (compiler_builtins') is already correctly rounded, checked above.

CR_UNARY = {"f64::exp": "exp", "f64::ln": "log", "f64::log10": "log10", "f64::sin": "sin",
            "f64::cos": "cos"}
CR_BINARY = {"f64::hypot": "hypot", "f64::powf": "pow", "f64::atan2": "atan2"}


def mpfr_logged(log):
    """MPFR's correctly rounded primitives, each call glibc rounds otherwise recorded into `log`."""
    def unary(name, mp):
        def f(x):
            want = MPFR.cr(mp, x)
            got = GLIBC[name](x)
            if not same(got, want):
                log.append(f"{name}(f:{bits(x):016x}) glibc f:{bits(got):016x} correctly rounded f:{bits(want):016x}")
            return want
        return f

    def pw(x, y):
        want = MPFR.cr2("pow", x, y)
        got = GLIBC["pow"](x, y)
        if not same(got, want):
            log.append(f"pow(f:{bits(x):016x}, f:{bits(y):016x}) glibc f:{bits(got):016x} "
                       f"correctly rounded f:{bits(want):016x}")
        return want
    return {"exp": unary("exp", "exp"), "log": unary("log", "log"), "log10": unary("log10", "log10"), "pow": pw}


def corrections(module, rows):
    """(fn, inputs text, corrected outputs text, note) for every row of `module` whose outputs change
    when every transcendental result is correctly rounded."""
    out = []
    for fn, cls, ins, outs, where in rows:
        log = []
        if module in ("vortex", "bridge") and fn in M1B_C:
            x = [tok_val(t) for t in ins]
            want = M1B_C[fn](x, mpfr_logged(log))
            got = [tok_f(t) for t in outs]
            if len(got) == len(want) and all(same(a, b) for a, b in zip(got, want)):
                continue
            new = " ".join(f"f:{bits(w):016x}" for w in want)
            seen = list(dict.fromkeys(log))
            out.append((fn, " ".join(ins), new, "oracle " + " ".join(outs) + "; " + "; ".join(seen)))
            continue
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
        elif path.stem == "common":
            counts = check_common(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"common: {len(rows)} rows recomputed ({summary})")
        elif path.stem in ("buoyancy", "vortex", "bridge"):
            counts = check_m1b(rows)
            summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            print(f"{path.stem}: {len(rows)} rows recomputed ({summary})")
            if path.stem != "buoyancy":
                check_corrections(root, path.stem, path, rows)
        elif path.stem == "json":
            counts = check_json(rows)
            summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            print(f"json: {len(rows)} documents decoded again ({summary})")
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
