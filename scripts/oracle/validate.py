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
        if cls not in "ABCDS" or len(cls) != 1:
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
                      ("pow", 2), ("fmod", 2)]:
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
    "f64::rem": _libm.fmod,
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
                     "f64::as_usize", "f64::as_u32", "f64::rem", "oracle::sph_cell_key"], "A"),
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
GLIBC = {"exp": _libm.exp, "log": _libm.log, "log10": _libm.log10, "pow": _libm.pow, "sin": _libm.sin,
         "cos": _libm.cos}


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


# ── shallow (M2) ───────────────────────────────────────────────────────────────────────────
# rust-old/src/shallow.rs transcribed, in Rust's order, on a state dict (scripts/oracle/src/m_shallow.rs's token
# layout). cbrt is MPFR's correctly rounded one, which the oracle's compiler_builtins cbrt equals (port-parity §3).

SW_FIELDS = ["height", "vx", "vy", "ground", "manning_n"]
SW_SCALARS = ["nx", "ny", "dx", "gravity", "damping", "dry_threshold", "breaking_threshold",
              "breaking_dissipation", "dispersion_coeff", "use_green_naghdi", "use_riemann"]


def sw_read(x, k=0):
    """(state dict, the index after it) from the token values at k."""
    sw = dict(zip(SW_SCALARS, x[k:k + 11]))
    at = k + 11
    for name in SW_FIELDS:
        n = x[at][1]
        sw[name] = list(x[at + 1:at + 1 + n])
        at += 1 + n
    return sw, at


def sw_tokens(sw):
    out = [sw[s] for s in SW_SCALARS]
    for name in SW_FIELDS:
        out += [("v", len(sw[name]))] + list(sw[name])
    return out


def r_max(x, y):
    """Rust's f64::max (a NaN operand ignored)."""
    if x != x:
        return y
    if y != y:
        return x
    return x if x > y else y


def r_minf(x, y):
    if x != x:
        return y
    if y != y:
        return x
    return x if x < y else y


def sw_new(nx, ny, dx, h):
    if nx < 3 or ny < 3 or nx >= 1 << 63 or ny >= 1 << 63:
        return ["E:InvalidGridResolution"]
    if not fin(dx) or dx <= 0.0:
        return ["E:InvalidParameter"]
    if not fin(h):
        return ["E:InvalidParameter"]
    if nx * ny >= 1 << 64:
        return ["E:InvalidGridResolution"]
    n = nx * ny
    out = [nx, ny, dx, 9.81, 0.999, 1e-4, 0.5, 5.0, 0.0, 0, 0]
    for v in (h, 0.0, 0.0, 0.0, 0.0):
        out += [n, v]
    return out + [n, n, n, n, 0]


def sw_depth(sw, i):
    return r_max(sw["height"][i] - sw["ground"][i], 0.0)


def sw_e(sw, k, own):
    return sw["height"][k] if sw["height"][k] - sw["ground"][k] >= sw["dry_threshold"] else own


def sw_breaking(sw, x, y):
    nx, ny = sw["nx"], sw["ny"]
    if x == 0 or x + 1 >= nx or y == 0 or y + 1 >= ny:
        return 0
    i = y * nx + x
    if sw["breaking_threshold"] <= 0.0 or i + nx >= len(sw["height"]):
        return 0
    eta, g = sw["height"], sw["ground"]
    if eta[i] - g[i] < sw["dry_threshold"]:
        return 0
    inv_2dx = 0.5 / sw["dx"]
    own = eta[i]
    sx = (sw_e(sw, i + 1, own) - sw_e(sw, i - 1, own)) * inv_2dx
    sy = (sw_e(sw, i + nx, own) - sw_e(sw, i - nx, own)) * inv_2dx
    return int(r_sqrt((sx * sx) + (sy * sy)) > sw["breaking_threshold"])


def sw_total_volume(sw):
    area = sw["dx"] * sw["dx"]
    s = -0.0
    for h, g in zip(sw["height"], sw["ground"]):
        s = s + r_max(h - g, 0.0) * area
    return s


def sw_interior_volume(sw):
    area = sw["dx"] * sw["dx"]
    nx, ny = sw["nx"], sw["ny"]
    if nx < 3 or ny < 3 or len(sw["height"]) < nx * ny or len(sw["ground"]) < nx * ny:
        return 0.0
    s = 0.0
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            s += r_max(sw["height"][i] - sw["ground"][i], 0.0)
    return s * area


def sw_max_wave(sw, rest):
    m = 0.0
    for h in sw["height"]:
        m = r_max(m, abs(h - rest))
    return m


def sw_cfl(sw, cfl):
    g = sw["gravity"]
    m = 0.0
    for i in range(len(sw["height"])):
        d = r_max(sw["height"][i] - sw["ground"][i], 0.0)
        u, v = sw["vx"][i], sw["vy"][i]
        m = r_max(m, r_sqrt((u * u) + (v * v)) + r_sqrt(g * d))
    if m < 1e-20:
        return sys.float_info.max
    return fdiv(cfl * sw["dx"], m)


def sw_disturb(sw, cx, cy, r, a):
    h = list(sw["height"])
    if r <= 0.0:
        return h
    r2 = r * r
    inv = 1.0 / r
    for y in range(sw["ny"]):
        for x in range(sw["nx"]):
            px = float(x) * sw["dx"]
            py = float(y) * sw["dx"]
            ddx, ddy = px - cx, py - cy
            d2 = (ddx * ddx) + (ddy * ddy)
            if d2 < r2:
                fac = 1.0 - (r_sqrt(d2) * inv)
                i = y * sw["nx"] + x
                h[i] = h[i] + ((a * fac) * fac)
    return h


def sw_sediment(sw, conc, cfg, dt):
    nx, ny = sw["nx"], sw["ny"]
    ground = list(sw["ground"])
    conc = list(conc)
    n = nx * ny
    if (not fin(dt) or dt <= 0.0 or nx < 3 or ny < 3 or len(conc) < n
            or any(len(sw[f]) < n for f in SW_FIELDS)):
        return ground, conc
    shields, d, rho_s, rho_w, er, dr = cfg
    g = sw["gravity"]
    tau_cr = ((shields * (rho_s - rho_w)) * g) * d
    dry = r_max(sw["dry_threshold"], 1e-9)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            depth = r_max(sw["height"][i] - ground[i], 0.0)
            if depth < dry:
                continue
            susp = r_max(conc[i], 0.0) * depth
            u, v = sw["vx"][i], sw["vy"][i]
            nm = r_max(sw["manning_n"][i], 0.01)
            tau_b = fdiv(((((rho_w * g) * nm) * nm) * ((u * u) + (v * v))), MPFR.cr("cbrt", depth))
            if tau_b > tau_cr:
                ero = (er * (tau_b - tau_cr)) * dt
                ground[i] -= ero
                depth += ero
                susp += ero
            if susp > 0.0:
                c = fdiv(susp, depth)
                depo = r_minf((dr * c) * dt, susp)
                ground[i] += depo
                depth -= depo
                susp -= depo
            conc[i] = r_max(fdiv(susp, depth), 0.0) if depth > 0.0 else 0.0
    return ground, conc


def sw_hll(h_l, u_l, h_r, u_r, g):
    h_l, h_r = r_max(h_l, 0.0), r_max(h_r, 0.0)
    if h_l <= 0.0 and h_r <= 0.0:
        return 0.0, 0.0
    c_l, c_r = r_sqrt(g * h_l), r_sqrt(g * h_r)
    if h_l <= 0.0:
        s_l, s_r = u_r - 2.0 * c_r, u_r + c_r
    elif h_r <= 0.0:
        s_l, s_r = u_l - c_l, u_l + 2.0 * c_l
    else:
        s_l, s_r = r_minf(u_l - c_l, u_r - c_r), r_max(u_l + c_l, u_r + c_r)
    f_h_l = h_l * u_l
    f_hu_l = ((h_l * u_l) * u_l) + (((0.5 * g) * h_l) * h_l)
    f_h_r = h_r * u_r
    f_hu_r = ((h_r * u_r) * u_r) + (((0.5 * g) * h_r) * h_r)
    if s_l >= 0.0:
        return f_h_l, f_hu_l
    if s_r <= 0.0:
        return f_h_r, f_hu_r
    den = fdiv(1.0, r_max(s_r - s_l, 1e-8))
    f_h = (((s_r * f_h_l) - (s_l * f_h_r)) + ((s_l * s_r) * (h_r - h_l))) * den
    f_hu = (((s_r * f_hu_l) - (s_l * f_hu_r)) + ((s_l * s_r) * ((h_r * u_r) - (h_l * u_l)))) * den
    return f_h, f_hu


def sw_face(left, right, g):
    eta_l, z_l, un_l, ut_l = left
    eta_r, z_r, un_r, ut_r = right
    h_l, h_r = r_max(eta_l - z_l, 0.0), r_max(eta_r - z_r, 0.0)
    z_f = r_max(z_l, z_r)
    hs_l = r_minf(r_max(eta_l - z_f, 0.0), h_l)
    hs_r = r_minf(r_max(eta_r - z_f, 0.0), h_r)
    f_h, f_hn = sw_hll(hs_l, un_l, hs_r, un_r, g)
    f_ht = f_h * (ut_l if f_h >= 0.0 else ut_r)
    return (f_h, f_hn + (0.5 * g) * ((h_l * h_l) - (hs_l * hs_l)),
            f_hn + (0.5 * g) * ((h_r * h_r) - (hs_r * hs_r)), f_ht)


def sw_slope(eta, ground, nx, i, dry, inv_2dx):
    if eta[i] - ground[i] < dry:
        return None
    e = lambda k: eta[k] if eta[k] - ground[k] >= dry else eta[i]
    sx = (e(i + 1) - e(i - 1)) * inv_2dx
    sy = (e(i + nx) - e(i - nx)) * inv_2dx
    return r_sqrt((sx * sx) + (sy * sy))


def sw_validate(sw):
    nx, ny = sw["nx"], sw["ny"]
    n = nx * ny
    if nx < 3 or ny < 3 or n >= 1 << 64 or any(len(sw[f]) != n for f in SW_FIELDS):
        return "E:InvalidGridResolution"
    for key, strict in (("dx", True), ("gravity", True), ("dispersion_coeff", False)):
        v = sw[key]
        if not fin(v) or (v <= 0.0 if strict else v < 0.0):
            return "E:InvalidParameter"
    return None


def sw_step(sw, dt, P, scratch):
    """One step of rust-old/src/shallow.rs in place on the state dict; None, or the error token."""
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    bad = sw_validate(sw)
    if bad:
        return bad
    nx, ny, g, dx = sw["nx"], sw["ny"], sw["gravity"], sw["dx"]
    inv_dx = fdiv(1.0, dx)
    inv_2dx = 0.5 * inv_dx
    damp = P["pow"](sw["damping"], dt / 0.001)
    dry = sw["dry_threshold"]
    H, U, V, G, M = sw["height"], sw["vx"], sw["vy"], sw["ground"], sw["manning_n"]
    size = nx * ny
    for i in range(size):
        if H[i] - G[i] < dry:
            U[i] = 0.0
            V[i] = 0.0
    svx, svy, sh = list(U), list(V), list(H)
    scratch["vx"], scratch["vy"], scratch["h"] = svx, svy, sh
    riemann = sw["use_riemann"] != 0
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            u, v = svx[i], svy[i]
            hi = r_max(sh[i] - G[i], 0.0)
            if not riemann:
                if hi < dry:
                    dhdx = dhdy = 0.0
                else:
                    gx = G[i]
                    def press(j_hi, j_lo):
                        g_hi, g_lo = G[j_hi], G[j_lo]
                        if abs(g_hi - gx) > 1e-12 or abs(g_lo - gx) > 1e-12:
                            z_hi, z_lo = r_max(gx, g_hi), r_max(gx, g_lo)
                            h_ih, h_jh = r_max(sh[i] - z_hi, 0.0), r_max(sh[j_hi] - z_hi, 0.0)
                            h_il, h_jl = r_max(sh[i] - z_lo, 0.0), r_max(sh[j_lo] - z_lo, 0.0)
                            sq = (((h_jh * h_jh) - (h_ih * h_ih)) + (h_il * h_il)) - (h_jl * h_jl)
                            return fdiv((0.5 * g) * sq, (2.0 * dx) * r_max(hi, dry))
                        return (g * (sh[j_hi] - sh[j_lo])) * inv_2dx
                    dhdx = press(i + 1, i - 1)
                    dhdy = press(i + nx, i - nx)
                def up(f, vel, s):
                    return (f[i] - f[i - s]) * inv_dx if vel > 0.0 else (f[i + s] - f[i]) * inv_dx
                dudx, dudy = up(svx, u, 1), up(svx, v, nx)
                dvdx, dvdy = up(svy, u, 1), up(svy, v, nx)
                U[i] = U[i] - (((dhdx + (u * dudx)) + (v * dudy)) * dt)
                V[i] = V[i] - (((dhdy + (u * dvdx)) + (v * dvdy)) * dt)
            nm = M[i]
            if nm > 0.0 and hi > dry:
                speed = r_sqrt((u * u) + (v * v))
                ds = r_max(hi, dry)
                fr = fdiv(((g * nm) * nm) * speed, ds * MPFR.cr("cbrt", ds))
                decay = fdiv(1.0, 1.0 + (fr * dt))
                U[i] *= decay
                V[i] *= decay
            bt, bd = sw["breaking_threshold"], sw["breaking_dissipation"]
            if bt > 0.0 and bd > 0.0:
                sl = sw_slope(sh, G, nx, i, dry, inv_2dx)
                if sl is not None and sl > bt:
                    decay = fdiv(1.0, 1.0 + (bd * dt))
                    U[i] *= decay
                    V[i] *= decay
            U[i] *= damp
            V[i] *= damp
    disp = sw["dispersion_coeff"]
    if riemann:
        sw_riemann(sw, dt, svx, svy, sh)
        if disp > 0.0:
            sw_dispersion(sw, svx, svy, sh)
    else:
        if disp > 0.0:
            sw_dispersion(sw, svx, svy, sh)
        depth = lambda k: r_max(sh[k] - G[k], 0.0)
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                d_i = depth(i)
                hr = 0.0 if x + 2 == nx else 0.5 * ((d_i * U[i]) + (depth(i + 1) * U[i + 1]))
                hl = 0.0 if x == 1 else 0.5 * ((d_i * U[i]) + (depth(i - 1) * U[i - 1]))
                ht = 0.0 if y + 2 == ny else 0.5 * ((d_i * V[i]) + (depth(i + nx) * V[i + nx]))
                hb = 0.0 if y == 1 else 0.5 * ((d_i * V[i]) + (depth(i - nx) * V[i - nx]))
                fd = (((hr - hl) + ht) - hb) * inv_dx
                H[i] = r_max(H[i] - (fd * dt), G[i])
    for x in range(nx):
        top, below = (ny - 1) * nx + x, (ny - 2) * nx + x
        H[x], U[x], V[x] = H[nx + x], U[nx + x], 0.0
        H[top], U[top], V[top] = H[below], U[below], 0.0
    for y in range(ny):
        l, r = y * nx, y * nx + nx - 1
        H[l], U[l], V[l] = H[l + 1], 0.0, V[l + 1]
        H[r], U[r], V[r] = H[r - 1], 0.0, V[r - 1]
    if not all(fin(a) for a in H + U + V):
        return "E:Diverged"
    return None


def sw_riemann(sw, dt, svx, svy, sh):
    nx, ny, g, dry = sw["nx"], sw["ny"], sw["gravity"], sw["dry_threshold"]
    inv_dx = fdiv(1.0, sw["dx"])
    H, U, V, G = sw["height"], sw["vx"], sw["vy"], sw["ground"]
    st = lambda k: (sh[k], G[k], svx[k], svy[k])
    k_ = dt * inv_dx
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            eta, z, u, v = st(i)
            h_i = r_max(eta - z, 0.0)
            own_x, mir_x = (eta, z, u, v), (eta, z, -u, v)
            right = mir_x if x + 2 == nx else st(i + 1)
            left = mir_x if x == 1 else st(i - 1)
            fr, fl = sw_face(own_x, right, g), sw_face(left, own_x, g)
            own_y, mir_y = (eta, z, v, u), (eta, z, -v, u)
            def swap(s):
                return (s[0], s[1], s[3], s[2])
            top = mir_y if y + 2 == ny else swap(st(i + nx))
            bot = mir_y if y == 1 else swap(st(i - nx))
            gt, gb = sw_face(own_y, top, g), sw_face(bot, own_y, g)
            new_h = r_max(h_i - (k_ * (((fr[0] - fl[0]) + gt[0]) - gb[0])), 0.0)
            hu = (h_i * U[i]) - (k_ * (((fr[1] - fl[2]) + gt[3]) - gb[3]))
            hv = (h_i * V[i]) - (k_ * (((fr[3] - fl[3]) + gt[1]) - gb[2]))
            H[i] = z + new_h
            if new_h > dry:
                U[i], V[i] = fdiv(hu, new_h), fdiv(hv, new_h)
            else:
                U[i], V[i] = 0.0, 0.0


def sw_dispersion(sw, svx, svy, sh):
    nx, ny, c = sw["nx"], sw["ny"], sw["dispersion_coeff"]
    inv_dx2 = fdiv(1.0, sw["dx"] * sw["dx"])
    dry, dw, G = sw["dry_threshold"], sw["use_green_naghdi"] != 0, sw["ground"]
    m = max(nx, ny)
    def depth(k):
        d = sh[k] - G[k]
        return d if d >= dry else 0.0
    def beta(a, b):
        h = 0.5 * (depth(a) + depth(b)) if dw else depth(a)
        return ((c * h) * h) * inv_dx2
    for field, old in ((sw["vx"], svx), (sw["vy"], svy)):
        delta = [0.0] * (nx * ny)
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                delta[i] = field[i] - old[i]
        for outer, inner, so, si in ((ny, nx, nx, 1), (nx, ny, 1, nx)):
            ln = inner - 2
            cp, dp = [0.0] * m, [0.0] * m
            for o in range(1, outer - 1):
                base = o * so
                for k in range(ln):
                    i = base + (k + 1) * si
                    b_lo, b_hi = beta(i, i - si), beta(i, i + si)
                    a_k, b_k, c_k = -b_lo, (1.0 + b_lo) + b_hi, -b_hi
                    cpp, dpp = (cp[k - 1], dp[k - 1]) if k > 0 else (0.0, 0.0)
                    den = b_k - (a_k * cpp)
                    cp[k] = fdiv(c_k, den)
                    dp[k] = fdiv(delta[i] - (a_k * dpp), den)
                nxt = 0.0
                for k in range(ln - 1, -1, -1):
                    i = base + (k + 1) * si
                    val = dp[k] - (cp[k] * nxt)
                    delta[i] = val
                    nxt = val
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                field[i] = old[i] + delta[i]


def sw_run(x, P):
    """A step row's outputs: the state after its nsteps steps, or the first failure."""
    sw, k = sw_read(x)
    dt, steps = x[k], x[k + 1]
    for _ in range(steps):
        bad = sw_step(sw, dt, P, {})
        if bad:
            return [bad]
    return sw_tokens(sw)


def sw_json(doc):
    """serde_json's reading of a ShallowWater document (16 fields; the scratch buffers skipped)."""
    names = ["nx", "ny", "dx", "height", "vx", "vy", "ground", "gravity", "damping", "manning_n",
             "dry_threshold", "breaking_threshold", "breaking_dissipation", "dispersion_coeff",
             "use_green_naghdi", "use_riemann"]
    try:
        f = j_struct(j_load(doc), names)
        def us(v):
            if not (isinstance(v, tuple) and v[0] == "int") or int(v[1]) < 0:
                raise JsonRefused("not a usize")
            return int(v[1])
        def bl(v):
            if not isinstance(v, bool):
                raise JsonRefused("not a bool")
            return int(v)
        sw = {"nx": us(f[0]), "ny": us(f[1]), "dx": j_f64(f[2]), "gravity": j_f64(f[7]),
              "damping": j_f64(f[8]), "dry_threshold": j_f64(f[10]), "breaking_threshold": j_f64(f[11]),
              "breaking_dissipation": j_f64(f[12]), "dispersion_coeff": j_f64(f[13]),
              "use_green_naghdi": bl(f[14]), "use_riemann": bl(f[15])}
        for name, k in zip(SW_FIELDS, (3, 4, 5, 6, 9)):
            sw[name] = [j_f64(e) for e in j_arr(f[k])]
        return sw_tokens(sw)
    except JsonRefused:
        return ["E:Deserialize"]


def check_shallow(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        short = func.split("::")[-1]
        if func == "ShallowWater::new":
            nx, ny = (v % (1 << 64) for v in x[:2])
            want = sw_new(nx, ny, x[2], x[3])
        elif func == "SedimentConfig::default":
            want = [0.047, 0.001, 2650.0, 1000.0, 0.01, 0.02]
        elif func == "ShallowWater::from_json":
            want = sw_json(unpct(ins[0][2:]))
        else:
            sw, k = sw_read(x)
            n = sw["nx"] * sw["ny"]
            cells = [(i % sw["nx"], i // sw["nx"]) for i in range(n)]
            if short == "depth_at":
                want = [("v", n)] + [sw_depth(sw, i) for i in range(n)]
            elif short == "surface_at":
                want = [("v", n)] + [sw["height"][i] for i in range(n)]
            elif short == "is_wet":
                want = [("v", n)] + [int(sw_depth(sw, i) >= sw["dry_threshold"]) for i in range(n)]
            elif short == "is_breaking":
                want = [("v", n)] + [sw_breaking(sw, cx, cy) for cx, cy in cells]
            elif short == "total_volume":
                want = [sw_total_volume(sw)]
            elif short == "interior_volume":
                want = [sw_interior_volume(sw)]
            elif short == "max_wave_height":
                want = [sw_max_wave(sw, x[k])]
            elif short == "cfl_dt":
                want = [sw_cfl(sw, x[k])]
            elif short == "add_disturbance":
                h = sw_disturb(sw, *x[k:k + 4])
                want = [("v", len(h))] + h
            elif short == "step":
                want = sw_run(x, GLIBC)
            elif func == "update_sediment":
                nc = x[k][1]
                conc = x[k + 1:k + 1 + nc]
                cfg = x[k + 1 + nc:k + 7 + nc]
                g, c = sw_sediment(sw, conc, cfg, x[k + 7 + nc])
                want = [("v", len(g))] + g + [("v", len(c))] + c
            else:
                fail(f"{where}: no check for {func}")
                continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


# ── grid (M3 part a) ───────────────────────────────────────────────────────────────────────
# rust-old/src/grid.rs's FluidGrid transcribed, in Rust's order, on a state dict (scripts/oracle/src/m_grid.rs's
# token layout). Python's + − × ÷, math.sqrt, math.floor and math.fmod are IEEE's (fmod is exact), and Python's
# operator precedence and left associativity are Rust's, so each formula is written as Rust writes it. Part a's
# rows use the periodic Gauss–Seidel and multigrid projections; the DST and CG paths arrive with part b.

FG_FIELDS = ["vx", "vy", "pressure", "density"]
BCS = ["NoSlip", "FreeSlip", "Periodic"]
GC_FIELDS = ["dt", "viscosity", "diffusion_iterations", "pressure_iterations", "boundary",
             "vorticity_confinement", "buoyancy_alpha", "ambient_density", "use_maccormack", "use_bfecc",
             "smagorinsky_cs", "use_multigrid", "use_cg"]
GC_SMOKE = {"dt": 0.1, "viscosity": 0.0001, "diffusion_iterations": 20, "pressure_iterations": 40,
            "boundary": "NoSlip", "vorticity_confinement": 0.5, "buoyancy_alpha": 0.1, "ambient_density": 0.0,
            "use_maccormack": 0, "use_bfecc": 0, "smagorinsky_cs": 0.0, "use_multigrid": 0, "use_cg": 0}
NAN = float("nan")
USIZE = 1 << 64


def fg_read(x, k=0):
    """(state dict, the index after it) from the token values at k; nx and ny as usizes."""
    g = {"nx": x[k] % USIZE, "ny": x[k + 1] % USIZE, "dx": x[k + 2]}
    at = k + 3
    for name in FG_FIELDS:
        n = x[at][1]
        g[name] = list(x[at + 1:at + 1 + n])
        at += 1 + n
    return g, at


def word(n):
    """A usize as the signed word its token holds."""
    return n - USIZE if n >= 1 << 63 else n


def fg_tokens(g):
    out = [word(g["nx"]), word(g["ny"]), g["dx"]]
    for name in FG_FIELDS:
        out += [("v", len(g[name]))] + list(g[name])
    return out


def gc_read(x, k):
    c = dict(zip(GC_FIELDS, x[k:k + 13]))
    c["boundary"] = c["boundary"].split(".")[-1]
    return c


def gc_tokens(c):
    return [c[f] if f != "boundary" else f"e:BoundaryCondition.{c[f]}" for f in GC_FIELDS]


def fg_new(nx, ny, dx):
    if nx < 4 or ny < 4:
        return ["E:InvalidGridResolution"]
    if dx <= 0.0 or not fin(dx):
        return ["E:InvalidParameter"]
    size = nx * ny
    if size >= USIZE:
        return ["E:InvalidParameter"]
    return [word(nx), word(ny), dx] + [size, 0.0] * 4 + [size] * 4 + [0]


def fg_sample(f, nx, ny, fx, fy):
    if not fin(fx) or not fin(fy):
        return NAN
    x0 = int(r_minf(r_max(float(math.floor(fx)), 0.0), float(nx - 2)))
    y0 = int(r_minf(r_max(float(math.floor(fy)), 0.0), float(ny - 2)))
    x1 = x0 + 1
    y1 = y0 + 1
    sx = r_clamp(fx - float(x0), 0.0, 1.0)
    sy = r_clamp(fy - float(y0), 0.0, 1.0)
    v00 = f[y0 * nx + x0]
    v10 = f[y0 * nx + x1]
    v01 = f[y1 * nx + x0]
    v11 = f[y1 * nx + x1]
    return v00 * (1.0 - sx) * (1.0 - sy) + v10 * sx * (1.0 - sy) + v01 * (1.0 - sx) * sy + v11 * sx * sy


def fg_bounds(f, nx, ny, fx, fy):
    if not fin(fx) or not fin(fy):
        return (-math.inf, math.inf)
    x0 = int(r_minf(r_max(float(math.floor(fx)), 0.0), float(nx - 2)))
    y0 = int(r_minf(r_max(float(math.floor(fy)), 0.0), float(ny - 2)))
    v00 = f[y0 * nx + x0]
    v10 = f[y0 * nx + x0 + 1]
    v01 = f[(y0 + 1) * nx + x0]
    v11 = f[(y0 + 1) * nx + x0 + 1]
    return (r_minf(r_minf(r_minf(v00, v10), v01), v11), r_max(r_max(r_max(v00, v10), v01), v11))


def fg_clamp_safe(v, lo, hi):
    return r_clamp(v, lo, hi) if lo <= hi else v


def fg_copy_ring(dst, src, nx, ny):
    dst[:nx] = src[:nx]
    top = (ny - 1) * nx
    dst[top:top + nx] = src[top:top + nx]
    for y in range(1, ny - 1):
        dst[y * nx] = src[y * nx]
        dst[y * nx + nx - 1] = src[y * nx + nx - 1]


def fg_advect(dst, src, vx, vy, nx, ny, dt, inv_dx):
    k = dt * inv_dx
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            dst[i] = fg_sample(src, nx, ny, float(x) - k * vx[i], float(y) - k * vy[i])


def fg_sample_periodic(f, nx, ny, fx, fy):
    if not fin(fx) or not fin(fy):
        return NAN
    nxf = float(nx)
    nyf = float(ny)
    fx = math.fmod(math.fmod(fx, nxf) + nxf, nxf)
    fy = math.fmod(math.fmod(fy, nyf) + nyf, nyf)
    x0 = int(math.floor(fx)) % nx
    y0 = int(math.floor(fy)) % ny
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    sx = fx - float(math.floor(fx))
    sy = fy - float(math.floor(fy))
    v00 = f[y0 * nx + x0]
    v10 = f[y0 * nx + x1]
    v01 = f[y1 * nx + x0]
    v11 = f[y1 * nx + x1]
    return v00 * (1.0 - sx) * (1.0 - sy) + v10 * sx * (1.0 - sy) + v01 * (1.0 - sx) * sy + v11 * sx * sy


def fg_advect_periodic(dst, src, vx, vy, nx, ny, dt, inv_dx):
    k = dt * inv_dx
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            dst[i] = fg_sample_periodic(src, nx, ny, float(x) - k * vx[i], float(y) - k * vy[i])


def fg_maccormack(dst, src, vx, vy, nx, ny, dt, inv_dx, temp):
    k = dt * inv_dx
    fg_advect(dst, src, vx, vy, nx, ny, dt, inv_dx)
    fg_copy_ring(dst, src, nx, ny)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            temp[i] = fg_sample(dst, nx, ny, float(x) + k * vx[i], float(y) + k * vy[i])
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            corrected = dst[i] + 0.5 * (src[i] - temp[i])
            fx = float(x) - k * vx[i]
            fy = float(y) - k * vy[i]
            lo, hi = fg_bounds(src, nx, ny, fx, fy)
            dst[i] = fg_clamp_safe(corrected, lo, hi)


def fg_bfecc(dst, src, vx, vy, nx, ny, dt, inv_dx, t1, t2):
    k = dt * inv_dx
    fg_advect(t1, src, vx, vy, nx, ny, dt, inv_dx)
    fg_copy_ring(t1, src, nx, ny)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            t2[i] = fg_sample(t1, nx, ny, float(x) + k * vx[i], float(y) + k * vy[i])
    fg_copy_ring(t2, src, nx, ny)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            t2[i] = src[i] + 0.5 * (src[i] - t2[i])
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            fx = float(x) - k * vx[i]
            fy = float(y) - k * vy[i]
            v = fg_sample(t2, nx, ny, fx, fy)
            lo, hi = fg_bounds(src, nx, ny, fx, fy)
            dst[i] = fg_clamp_safe(v, lo, hi)


def fg_diffuse_buf(field, nx, ny, rate, dt, dx, iters):
    a = dt * rate / (dx * dx)
    inv_denom = 1.0 / (1.0 + 4.0 * a)
    rhs = list(field)
    for _ in range(iters):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                nb = field[i - 1] + field[i + 1] + field[i - nx] + field[i + nx]
                field[i] = (rhs[i] + a * nb) * inv_denom


def fg_diffuse(field, nx, ny, rate, dt, dx, iters):
    if nx < 3 or ny < 3 or nx * ny >= USIZE or nx * ny != len(field):
        return field
    fg_diffuse_buf(field, nx, ny, rate, dt, dx, iters)
    return field


def fg_divergence(div, vx, vy, nx, ny, scale):
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            div[i] = -((vx[i + 1] - vx[i - 1]) + (vy[i + nx] - vy[i - nx])) * scale


def fg_gs(p, div, nx, ny, iters):
    for _ in range(iters):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                p[i] = (div[i] + (p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx])) * 0.25


def fg_vcycle(p, rhs, nx, ny, nu, levels, depth):
    if nx <= 6 or ny <= 6 or depth >= 10:
        fg_gs(p, rhs, nx, ny, 50)
        return
    fg_gs(p, rhs, nx, ny, nu)
    n = nx * ny
    cnx = -(-nx // 2)
    cny = -(-ny // 2)
    if cnx < 4 or cny < 4:
        fg_gs(p, rhs, nx, ny, nu * 4)
        return
    cn = cnx * cny
    while len(levels) <= depth:
        levels.append([[], [], []])
    lv = levels[depth]
    residual = lv[0] = [0.0] * n
    c_rhs = lv[1] = [0.0] * cn
    c_p = lv[2] = [0.0] * cn
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            a_p = 4.0 * p[i] - (p[i - 1] + p[i + 1] + p[i - nx] + p[i + nx])
            residual[i] = rhs[i] - a_p
    for cy in range(1, cny - 1):
        for cx in range(1, cnx - 1):
            fi = cy * 2 * nx + cx * 2
            r = residual
            c_rhs[cy * cnx + cx] = (4.0 * r[fi] + 2.0 * (r[fi - 1] + r[fi + 1] + r[fi - nx] + r[fi + nx])
                                    + r[fi - nx - 1] + r[fi - nx + 1] + r[fi + nx - 1] + r[fi + nx + 1]) * 0.25
    fg_vcycle(c_p, c_rhs, cnx, cny, nu, levels, depth + 1)

    def c_at(cx, cy):
        return c_p[cy * cnx + cx] if cx < cnx and cy < cny else 0.0

    for y in range(1, ny - 1):
        cy = y // 2
        odd_y = y % 2 == 1
        for x in range(1, nx - 1):
            cx = x // 2
            odd_x = x % 2 == 1
            if not odd_x and not odd_y:
                corr = c_at(cx, cy)
            elif odd_x and not odd_y:
                corr = 0.5 * (c_at(cx, cy) + c_at(cx + 1, cy))
            elif not odd_x and odd_y:
                corr = 0.5 * (c_at(cx, cy) + c_at(cx, cy + 1))
            else:
                corr = 0.25 * (c_at(cx, cy) + c_at(cx + 1, cy) + c_at(cx, cy + 1) + c_at(cx + 1, cy + 1))
            p[y * nx + x] += corr
    fg_gs(p, rhs, nx, ny, nu)


def fg_multigrid(p, div, nx, ny, nu, cycles, levels):
    for _ in range(max(cycles, 1)):
        fg_vcycle(p, div, nx, ny, nu, levels, 0)


# The DST's transcendentals, per transform length n: the n² sines sin(((π·(i+1))·(k+1))/(n+1)) (row k, entry i,
# hisab 1.4.0 `dst`'s argument) and the n eigenvalue terms 2·cos((π·(k+1))/(n+1)) − 2 of `pressure_solve_dst`.
# Each depends only on (i, k, n), so one table per length gives the same values as the solver's per-call ones.
# glibc's (the oracle's own calls) and MPFR's correctly rounded ones, each cached; for MPFR's, the calls glibc
# rounds otherwise, by argument.
_DST_GLIBC = {}
_DST_CR = {}


def _dst_args(n):
    np1 = float(n + 1)
    sines = [[((PI * float(i + 1)) * float(k + 1)) / np1 for i in range(n)] for k in range(n)]
    coses = [(PI * float(k + 1)) / np1 for k in range(n)]
    return sines, coses


def dst_glibc(n):
    """(sines, eigenvalue terms) for length n from glibc's sin and cos, as the oracle calls them."""
    if n not in _DST_GLIBC:
        sa, ca = _dst_args(n)
        _DST_GLIBC[n] = ([[_libm.sin(a) for a in row] for row in sa], [2.0 * _libm.cos(a) - 2.0 for a in ca])
    return _DST_GLIBC[n]


def dst_cr(n):
    """(sines, eigenvalue terms, the misrounded calls) for length n from MPFR's correctly rounded sin and cos."""
    if n not in _DST_CR:
        sa, ca = _dst_args(n)
        miss = {}
        sines = []
        for row in sa:
            out = []
            for a in row:
                want = MPFR.cr("sin", a)
                got = _libm.sin(a)
                if not same(got, want):
                    miss[f"sin(f:{bits(a):016x})"] = f"glibc f:{bits(got):016x} correctly rounded f:{bits(want):016x}"
                out.append(want)
            sines.append(out)
        terms = []
        for a in ca:
            want = MPFR.cr("cos", a)
            got = _libm.cos(a)
            if not same(got, want):
                miss[f"cos(f:{bits(a):016x})"] = f"glibc f:{bits(got):016x} correctly rounded f:{bits(want):016x}"
            terms.append(2.0 * want - 2.0)
        _DST_CR[n] = (sines, terms, [f"{k} {v}" for k, v in miss.items()])
    return _DST_CR[n]


def dst_cr_logged(log):
    """MPFR's tables, each length's misrounded calls recorded into `log` when the length is used."""
    def tables(n):
        sines, terms, miss = dst_cr(n)
        log.extend(miss)
        return sines, terms
    return tables


def fg_dst(x, n, sines):
    """hisab 1.4.0 `dst` of x (length n): each sum from +0 in index order."""
    out = []
    for row in sines:
        s = 0.0
        for xi, si in zip(x, row):
            s += xi * si
        out.append(s)
    return out


def fg_idst(x, n, sines):
    """hisab 1.4.0 `idst`: the DST, each value then × 2/(n+1)."""
    scale = 2.0 / float(n + 1)
    return [v * scale for v in fg_dst(x, n, sines)]


def fg_pressure_dst(p, div, nx, ny, tables):
    """The private `pressure_solve_dst` (`tables(n)` gives a length's sines and eigenvalue terms)."""
    inx, iny = nx - 2, ny - 2
    if inx == 0 or iny == 0:
        return
    sx, lx = tables(inx)
    sy, ly = tables(iny)
    rhs = [div[(iy + 1) * nx + ix + 1] for iy in range(iny) for ix in range(inx)]

    def rows(f, s):
        for iy in range(iny):
            rhs[iy * inx:(iy + 1) * inx] = f(rhs[iy * inx:(iy + 1) * inx], inx, s)

    def cols(f, s):
        for ix in range(inx):
            t = f([rhs[iy * inx + ix] for iy in range(iny)], iny, s)
            for iy in range(iny):
                rhs[iy * inx + ix] = t[iy]

    rows(fg_dst, sx)
    cols(fg_dst, sy)
    for ky in range(iny):
        for kx in range(inx):
            eig = -(lx[kx] + ly[ky])
            if abs(eig) > 1e-12:
                rhs[ky * inx + kx] /= eig
    rows(fg_idst, sx)
    cols(fg_idst, sy)
    p[:] = [0.0] * len(p)
    for iy in range(iny):
        for ix in range(inx):
            p[(iy + 1) * nx + ix + 1] = rhs[iy * inx + ix]


def fg_cg(p, div, nx, ny, iters):
    """The private `pressure_solve_cg`, on hisab 1.4.0 `conjugate_gradient`: None, or the failure's token."""
    n = nx * ny

    def dot(v, w):
        s = -0.0
        for a, b in zip(v, w):
            s += a * b
        return s

    def a_mul(v):
        out = [0.0] * n
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                out[i] = 4.0 * v[i] - (((v[i - 1] + v[i + 1]) + v[i - nx]) + v[i + nx])
        return out

    tol = r_max(1e-6 * math.sqrt(dot(div, div)), 1e-14)
    nb = len(div)
    if nb == 0 or len(p) != nb:
        return "E:Diverged"
    x = list(p)
    ax = a_mul(x)
    r = [div[i] - ax[i] for i in range(nb)]
    q = list(r)
    rs_old = dot(r, r)
    tol_sq = tol * tol
    if rs_old >= tol_sq:
        for _ in range(iters):
            ap = a_mul(q)
            pap = dot(q, ap)
            if abs(pap) < 1e-12:
                break
            alpha = fdiv(rs_old, pap)
            for i in range(nb):
                x[i] += alpha * q[i]
                r[i] -= alpha * ap[i]
            rs_new = dot(r, r)
            if rs_new < tol_sq:
                break
            beta = fdiv(rs_new, rs_old)
            q = [r[i] + beta * q[i] for i in range(nb)]
            rs_old = rs_new
    p[:] = x
    return None


def fg_project(vx, vy, p, nx, ny, inv_2dx):
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            vx[i] -= (p[i + 1] - p[i - 1]) * inv_2dx
            vy[i] -= (p[i + nx] - p[i - nx]) * inv_2dx


def fg_vorticity(vx, vy, nx, ny, dx, dt, eps, w):
    inv_2dx = 0.5 / dx
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            w[i] = (vy[i + 1] - vy[i - 1]) * inv_2dx - (vx[i + nx] - vx[i - nx]) * inv_2dx
    for y in range(2, max(ny - 2, 0)):
        for x in range(2, max(nx - 2, 0)):
            i = y * nx + x
            if abs(w[i]) < 1e-10:
                continue
            eta_x = (abs(w[i + 1]) - abs(w[i - 1])) * inv_2dx
            eta_y = (abs(w[i + nx]) - abs(w[i - nx])) * inv_2dx
            eta_mag = math.sqrt(eta_x * eta_x + eta_y * eta_y)
            if eta_mag < 1e-10:
                continue
            nx_v = eta_x / eta_mag
            ny_v = eta_y / eta_mag
            vx[i] += eps * dx * dt * ny_v * w[i]
            vy[i] -= eps * dx * dt * nx_v * w[i]


def fg_enforce(vx, vy, nx, ny, bc):
    if bc == "NoSlip":
        for x in range(nx):
            vx[x] = -vx[nx + x]
            vy[x] = -vy[nx + x]
            top = (ny - 1) * nx + x
            below = (ny - 2) * nx + x
            vx[top] = -vx[below]
            vy[top] = -vy[below]
        for y in range(ny):
            l = y * nx
            vx[l] = -vx[l + 1]
            vy[l] = -vy[l + 1]
            r = y * nx + nx - 1
            vx[r] = -vx[r - 1]
            vy[r] = -vy[r - 1]
    elif bc == "FreeSlip":
        for x in range(nx):
            vx[x] = vx[nx + x]
            vy[x] = -vy[nx + x]
            top = (ny - 1) * nx + x
            below = (ny - 2) * nx + x
            vx[top] = vx[below]
            vy[top] = -vy[below]
        for y in range(ny):
            l = y * nx
            vx[l] = -vx[l + 1]
            vy[l] = vy[l + 1]
            r = y * nx + nx - 1
            vx[r] = -vx[r - 1]
            vy[r] = vy[r - 1]
    else:
        for x in range(nx):
            vx[x] = vx[(ny - 2) * nx + x]
            vy[x] = vy[(ny - 2) * nx + x]
            vx[(ny - 1) * nx + x] = vx[nx + x]
            vy[(ny - 1) * nx + x] = vy[nx + x]
        for y in range(ny):
            vx[y * nx] = vx[y * nx + nx - 2]
            vy[y * nx] = vy[y * nx + nx - 2]
            vx[y * nx + nx - 1] = vx[y * nx + 1]
            vy[y * nx + nx - 1] = vy[y * nx + 1]
        c = (ny - 2) * nx + nx - 2
        vx[0] = vx[c]
        vy[0] = vy[c]
        vx[nx - 1] = vx[(ny - 2) * nx + 1]
        vy[nx - 1] = vy[(ny - 2) * nx + 1]
        vx[(ny - 1) * nx] = vx[nx + nx - 2]
        vy[(ny - 1) * nx] = vy[nx + nx - 2]
        vx[(ny - 1) * nx + nx - 1] = vx[nx + 1]
        vy[(ny - 1) * nx + nx - 1] = vy[nx + 1]


def fg_enforce_scalar(f, nx, ny):
    for x in range(nx):
        f[x] = f[nx + x]
        f[(ny - 1) * nx + x] = f[(ny - 2) * nx + x]
    for y in range(ny):
        f[y * nx] = f[y * nx + 1]
        f[y * nx + nx - 1] = f[y * nx + nx - 2]


def fg_advect_by(dst, src, g, c, sc, drhs):
    nx, ny = g["nx"], g["ny"]
    inv_dx = 1.0 / g["dx"]
    periodic = c["boundary"] == "Periodic"
    if c["use_bfecc"] and not periodic:
        fg_bfecc(dst, src, g["vx"], g["vy"], nx, ny, c["dt"], inv_dx, sc, drhs)
    elif c["use_maccormack"] and not periodic:
        fg_maccormack(dst, src, g["vx"], g["vy"], nx, ny, c["dt"], inv_dx, sc)
    elif periodic:
        fg_advect_periodic(dst, src, g["vx"], g["vy"], nx, ny, c["dt"], inv_dx)
    else:
        fg_advect(dst, src, g["vx"], g["vy"], nx, ny, c["dt"], inv_dx)


def fg_step(g, c, tables=dst_glibc):
    """One FluidGrid::step: None, or the failure's token. The multigrid levels persist in g["mg"]; `tables` gives
    the DST's sines and eigenvalue terms per length (glibc's, or MPFR's for the corrected oracle)."""
    dt = c["dt"]
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    if c["viscosity"] < 0.0 or not fin(c["viscosity"]):
        return "E:InvalidViscosity"
    for key in ("smagorinsky_cs", "vorticity_confinement"):
        if not fin(c[key]) or c[key] < 0.0:
            return "E:InvalidParameter"
    if not fin(c["buoyancy_alpha"]) or not fin(c["ambient_density"]):
        return "E:InvalidParameter"
    if c["pressure_iterations"] == 0 or c["diffusion_iterations"] == 0:
        return "E:InvalidParameter"
    nx, ny, dx = g["nx"], g["ny"], g["dx"]
    n = nx * ny
    if nx < 4 or ny < 4 or n >= USIZE or any(len(g[f]) != n for f in FG_FIELDS):
        return "E:InvalidGridResolution"
    if not fin(dx) or dx <= 0.0:
        return "E:InvalidParameter"
    inv_dx = 1.0 / dx
    inv_2dx = 0.5 * inv_dx
    bc = c["boundary"]
    if bc not in BCS:
        raise ValueError(f"no boundary condition {bc}")
    sa, sb, sc, drhs = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    vx, vy, rho = g["vx"], g["vy"], g["density"]
    fg_advect_by(sa, vx, g, c, sc, drhs)
    fg_advect_by(sb, vy, g, c, sc, drhs)
    vx[:] = sa
    vy[:] = sb
    fg_enforce(vx, vy, nx, ny, bc)
    if any(v != v for v in vx) or any(v != v for v in vy):
        return "E:Diverged"
    eff = c["viscosity"]
    if c["smagorinsky_cs"] > 0.0:
        cs_dx = c["smagorinsky_cs"] * dx
        cs_dx2 = cs_dx * cs_dx
        sum_nu_t = 0.0
        count = 0.0
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                dudx = (vx[i + 1] - vx[i - 1]) * inv_2dx
                dudy = (vx[i + nx] - vx[i - nx]) * inv_2dx
                dvdx = (vy[i + 1] - vy[i - 1]) * inv_2dx
                dvdy = (vy[i + nx] - vy[i - nx]) * inv_2dx
                s11, s22, s12 = dudx, dvdy, 0.5 * (dudy + dvdx)
                s_mag = math.sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12))
                sum_nu_t += cs_dx2 * s_mag
                count += 1.0
        eff = c["viscosity"] + (sum_nu_t / count if count > 0.0 else 0.0)
    if eff > 0.0:
        fg_diffuse_buf(vx, nx, ny, eff, dt, dx, c["diffusion_iterations"])
        fg_diffuse_buf(vy, nx, ny, eff, dt, dx, c["diffusion_iterations"])
        fg_enforce(vx, vy, nx, ny, bc)
    if c["buoyancy_alpha"] != 0.0:
        for i in range(n):
            vy[i] += c["buoyancy_alpha"] * (rho[i] - c["ambient_density"]) * dt
    if c["vorticity_confinement"] > 0.0:
        sa = [0.0] * n
        fg_vorticity(vx, vy, nx, ny, dx, dt, c["vorticity_confinement"], sa)
    fg_enforce(vx, vy, nx, ny, bc)
    sa = [0.0] * n
    fg_divergence(sa, vx, vy, nx, ny, 0.5 * dx)
    if c["use_cg"]:
        bad = fg_cg(g["pressure"], sa, nx, ny, c["pressure_iterations"])
        if bad:
            return bad
    elif c["use_multigrid"]:
        fg_multigrid(g["pressure"], sa, nx, ny, 4, -(-c["pressure_iterations"] // 10), g["mg"])
    elif bc == "Periodic":
        fg_gs(g["pressure"], sa, nx, ny, c["pressure_iterations"])
    else:
        fg_pressure_dst(g["pressure"], sa, nx, ny, tables)
    fg_project(vx, vy, g["pressure"], nx, ny, inv_2dx)
    fg_enforce(vx, vy, nx, ny, bc)
    if not all(fin(v) for v in vx + vy):
        return "E:Diverged"
    sa = [0.0] * n
    fg_advect_by(sa, rho, g, c, sc, drhs)
    rho[:] = sa
    if bc != "Periodic":
        fg_enforce_scalar(rho, nx, ny)
    return None


def fg_run(x, tables=dst_glibc):
    """A step row's outputs: the state and the scratch dump after its nsteps steps, or the first failure."""
    g, k = fg_read(x)
    c = gc_read(x, k)
    steps = x[k + 13]
    g["mg"] = []
    for _ in range(steps):
        bad = fg_step(g, c, tables)
        if bad:
            return [bad]
    n = g["nx"] * g["ny"]
    out = fg_tokens(g) + [n] * 4 + [len(g["mg"])]
    for lv in g["mg"]:
        out += [len(lv[0]), len(lv[1]), len(lv[2])]
    return out


def fg_json(func, doc):
    """serde_json's reading of a FluidGrid, GridConfig or BoundaryCondition document."""
    def us(v):
        if not (isinstance(v, tuple) and v[0] == "int") or int(v[1]) < 0:
            raise JsonRefused("not a usize")
        return int(v[1])

    def bl(v):
        if not isinstance(v, bool):
            raise JsonRefused("not a bool")
        return int(v)

    def bc(v):
        if isinstance(v, str) and v in BCS:
            return v
        if isinstance(v, tuple) and v[0] == "obj" and len(v[1]) == 1:
            key, val = v[1][0]
            if key in BCS and val is None:
                return key
        raise JsonRefused("no variant")

    try:
        v = j_load(doc)
        if func == "BoundaryCondition::from_json":
            return [f"e:BoundaryCondition.{bc(v)}"]
        if func == "GridConfig::from_json":
            f = j_struct(v, GC_FIELDS)
            c = {"dt": j_f64(f[0]), "viscosity": j_f64(f[1]), "diffusion_iterations": us(f[2]),
                 "pressure_iterations": us(f[3]), "boundary": bc(f[4]), "vorticity_confinement": j_f64(f[5]),
                 "buoyancy_alpha": j_f64(f[6]), "ambient_density": j_f64(f[7]), "use_maccormack": bl(f[8]),
                 "use_bfecc": bl(f[9]), "smagorinsky_cs": j_f64(f[10]), "use_multigrid": bl(f[11]),
                 "use_cg": bl(f[12])}
            return gc_tokens(c)
        f = j_struct(v, ["nx", "ny", "dx"] + FG_FIELDS)
        g = {"nx": us(f[0]), "ny": us(f[1]), "dx": j_f64(f[2])}
        for name, val in zip(FG_FIELDS, f[3:]):
            g[name] = [j_f64(e) for e in j_arr(val)]
        return fg_tokens(g)
    except JsonRefused:
        return ["E:Deserialize"]


def r_cfl_dt(g, visc, cfl):
    big = sys.float_info.max
    dt = big
    max_v = fg_max_speed(g)
    if max_v > 1e-20:
        dt = r_minf(dt, g["dx"] / max_v)
    if visc > 1e-20:
        dt = r_minf(dt, g["dx"] * g["dx"] / (4.0 * visc))
    return r_minf(cfl * dt, big)


def fg_max_speed(g):
    m = 0.0
    for u, v in zip(g["vx"], g["vy"]):
        m = r_max(m, u * u + v * v)
    return math.sqrt(m)


# ── grid, part c: KEpsilon, apply_ghost_fluid, MacGrid (M3 part c) ───────────────────────────

KE_FIELDS = ["k", "epsilon", "c_mu", "c_eps1", "c_eps2"]


def ke_read(x, k=0):
    """(k-ε dict, the index after it) from the token values at k."""
    m = {}
    at = k
    for name in ("k", "epsilon"):
        n = x[at][1]
        m[name] = list(x[at + 1:at + 1 + n])
        at += 1 + n
    m["c_mu"], m["c_eps1"], m["c_eps2"] = x[at:at + 3]
    return m, at + 3


def ke_tokens(m):
    return [("v", len(m["k"]))] + m["k"] + [("v", len(m["epsilon"]))] + m["epsilon"] + [
        m["c_mu"], m["c_eps1"], m["c_eps2"]]


def ke_step(m, vx, vy, nx, ny, dx, dt):
    """One KEpsilon::step: True, or False where Rust warns and returns."""
    n = nx * ny
    if n >= USIZE or nx < 3 or ny < 3 or len(vx) != n or len(vy) != n:
        return False
    if len(m["k"]) != n or len(m["epsilon"]) != n or not (fin(dx) and dx > 0.0 and fin(dt) and dt > 0.0):
        return False
    inv_2dx = 0.5 / dx
    ko, eo = list(m["k"]), list(m["epsilon"])
    k, e = m["k"], m["epsilon"]
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            nu_t = fdiv(m["c_mu"] * ko[i] * ko[i], r_max(eo[i], 1e-20))
            dudx = (vx[i + 1] - vx[i - 1]) * inv_2dx
            dudy = (vx[i + nx] - vx[i - nx]) * inv_2dx
            dvdx = (vy[i + 1] - vy[i - 1]) * inv_2dx
            dvdy = (vy[i + nx] - vy[i - nx]) * inv_2dx
            sh = dudy + dvdx
            s_mag_sq = 2.0 * (dudx * dudx + dvdy * dvdy + 0.5 * (sh * sh))
            prod = nu_t * s_mag_sq
            k[i] = r_max(ko[i] + (prod - eo[i]) * dt, 1e-10)
            eps_over_k = fdiv(eo[i], r_max(ko[i], 1e-10))
            e[i] = r_max(eo[i] + (m["c_eps1"] * prod - m["c_eps2"] * eo[i]) * eps_over_k * dt, 1e-10)
    return True


def ghost(p, ls, nx, ny, dx, sigma):
    """apply_ghost_fluid on p in place (nothing where Rust warns and returns)."""
    n = nx * ny
    if nx < 3 or ny < 3 or n >= USIZE or len(p) != n or len(ls) != n or not (fin(dx) and dx > 0.0):
        return
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            phi, pr, pl, pt, pb = ls[i], ls[i + 1], ls[i - 1], ls[i + nx], ls[i - nx]
            if not (phi * pr < 0.0 or phi * pl < 0.0 or phi * pt < 0.0 or phi * pb < 0.0):
                continue
            gx = (pr - pl) * 0.5 * inv_dx
            gy = (pt - pb) * 0.5 * inv_dx
            g = r_max(math.sqrt(gx * gx + gy * gy), 1e-20)
            lxx = (pr - 2.0 * phi + pl) * inv_dx2
            lyy = (pt - 2.0 * phi + pb) * inv_dx2
            kappa = fdiv(-(lxx + lyy), g)
            if phi > 0.0:
                p[i] -= fdiv(sigma * kappa * phi, g)


def mac_velocity_at(g, x, y):
    nx = g["nx"]
    u = 0.5 * (g["vx"][y * (nx + 1) + x] + g["vx"][y * (nx + 1) + x + 1])
    v = 0.5 * (g["vy"][y * nx + x] + g["vy"][(y + 1) * nx + x])
    return u, v


def mac_max_speed(g):
    m = 0.0
    for y in range(g["ny"]):
        for x in range(g["nx"]):
            u, v = mac_velocity_at(g, x, y)
            m = r_max(m, u * u + v * v)
    return math.sqrt(m)


def mac_new(nx, ny, dx):
    if nx < 4 or ny < 4:
        return ["E:InvalidGridResolution"]
    if dx <= 0.0 or not fin(dx):
        return ["E:InvalidParameter"]
    n = nx * ny
    out = [nx, ny, dx]
    for size in ((nx + 1) * ny, nx * (ny + 1), n, n):
        out += [size, 0.0]
    return out + [n, n]


def mac_step(g, c):
    """One MacGrid::step: None, or the failure's token. Scratch lengths in g["sd"]."""
    dt = c["dt"]
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    if c["pressure_iterations"] == 0:
        return "E:InvalidParameter"
    alpha, amb = c["buoyancy_alpha"], c["ambient_density"]
    if not fin(alpha) or not fin(amb):
        return "E:InvalidParameter"
    nx, ny, dx = g["nx"], g["ny"], g["dx"]
    n = nx * ny
    if (n >= USIZE or nx < 4 or ny < 4 or len(g["vx"]) != (nx + 1) * ny or len(g["vy"]) != nx * (ny + 1)
            or len(g["pressure"]) != n or len(g["density"]) != n):
        return "E:InvalidGridResolution"
    if not fin(dx) or dx <= 0.0:
        return "E:InvalidParameter"
    inv_dx = 1.0 / dx
    u, v, p, rho = g["vx"], g["vy"], g["pressure"], g["density"]
    n1 = nx + 1
    for y in range(ny):
        u[y * n1] = 0.0
        u[y * n1 + nx] = 0.0
    for x in range(nx):
        v[x] = 0.0
        v[ny * nx + x] = 0.0
    if alpha != 0.0:
        for y in range(1, ny):
            for x in range(nx):
                rf = 0.5 * (rho[(y - 1) * nx + x] + rho[y * nx + x])
                v[y * nx + x] += alpha * (rf - amb) * dt
    rhs = [-((u[y * n1 + x + 1] - u[y * n1 + x]) + (v[(y + 1) * nx + x] - v[y * nx + x])) * dx
           for y in range(ny) for x in range(nx)]
    for _ in range(c["pressure_iterations"]):
        for y in range(ny):
            for x in range(nx):
                i = y * nx + x
                sm, cnt = 0.0, 0.0
                if x > 0:
                    sm += p[i - 1]
                    cnt += 1.0
                if x + 1 < nx:
                    sm += p[i + 1]
                    cnt += 1.0
                if y > 0:
                    sm += p[i - nx]
                    cnt += 1.0
                if y + 1 < ny:
                    sm += p[i + nx]
                    cnt += 1.0
                p[i] = fdiv(rhs[i] + sm, cnt)
    total = -0.0
    for q in p:
        total += q
    mean = fdiv(total, float(n))
    p[:] = [q - mean for q in p]
    for y in range(ny):
        for x in range(1, nx):
            u[y * n1 + x] -= (p[y * nx + x] - p[y * nx + x - 1]) * inv_dx
    for y in range(1, ny):
        for x in range(nx):
            v[y * nx + x] -= (p[y * nx + x] - p[(y - 1) * nx + x]) * inv_dx
    dt_inv_dx = dt * inv_dx
    adv = []
    for y in range(ny):
        for x in range(nx):
            cu, cv = mac_velocity_at(g, x, y)
            adv.append(fg_sample(rho, nx, ny, float(x) - dt_inv_dx * cu, float(y) - dt_inv_dx * cv))
    g["density"] = adv
    g["sd"] = n
    if not all(fin(q) for q in u + v + adv):
        return "E:Diverged"
    return None


def grid_c_json(func, doc):
    """serde_json's reading of a MacGrid or KEpsilon document."""
    def us(v):
        if not (isinstance(v, tuple) and v[0] == "int") or int(v[1]) < 0:
            raise JsonRefused("not a usize")
        return int(v[1])

    try:
        v = j_load(doc)
        if func == "KEpsilon::from_json":
            f = j_struct(v, KE_FIELDS)
            m = {"k": [j_f64(e) for e in j_arr(f[0])], "epsilon": [j_f64(e) for e in j_arr(f[1])],
                 "c_mu": j_f64(f[2]), "c_eps1": j_f64(f[3]), "c_eps2": j_f64(f[4])}
            return ke_tokens(m)
        f = j_struct(v, ["nx", "ny", "dx"] + FG_FIELDS)
        g = {"nx": us(f[0]), "ny": us(f[1]), "dx": j_f64(f[2])}
        for name, val in zip(FG_FIELDS, f[3:]):
            g[name] = [j_f64(e) for e in j_arr(val)]
        return fg_tokens(g)
    except JsonRefused:
        return ["E:Deserialize"]


def grid_c_want(func, x, ins):
    """The recomputed outputs of a part c row, or None for another function."""
    if func == "KEpsilon::new":
        nx, ny = x[0] % USIZE, x[1] % USIZE
        n = nx * ny
        return ke_tokens({"k": [1e-4] * n, "epsilon": [1e-5] * n, "c_mu": 0.09, "c_eps1": 1.44,
                          "c_eps2": 1.92}) + [n, n]
    if func == "KEpsilon::turbulent_viscosity":
        m, _ = ke_read(x)
        return [("v", len(m["k"]))] + [fdiv(m["c_mu"] * kk * kk, r_max(m["epsilon"][i], 1e-20))
                                       for i, kk in enumerate(m["k"])]
    if func == "KEpsilon::step":
        m, k = ke_read(x)
        scratch = len(m["k"])
        n1 = x[k][1]
        vx = list(x[k + 1:k + 1 + n1])
        k2 = k + 1 + n1
        n2 = x[k2][1]
        vy = list(x[k2 + 1:k2 + 1 + n2])
        k3 = k2 + 1 + n2
        nx, ny, dx, dt, steps = x[k3] % USIZE, x[k3 + 1] % USIZE, x[k3 + 2], x[k3 + 3], x[k3 + 4]
        for _ in range(steps):
            if ke_step(m, vx, vy, nx, ny, dx, dt):
                scratch = nx * ny
        return ke_tokens(m) + [scratch, scratch]
    if func == "apply_ghost_fluid":
        n = x[0][1]
        pr = list(x[1:1 + n])
        m = x[1 + n][1]
        ls = list(x[2 + n:2 + n + m])
        k = 2 + n + m
        ghost(pr, ls, x[k] % USIZE, x[k + 1] % USIZE, x[k + 2], x[k + 3])
        return [("v", n)] + pr
    if func == "MacGrid::new":
        return mac_new(x[0] % USIZE, x[1] % USIZE, x[2])
    if func in ("MacGrid::from_json", "KEpsilon::from_json"):
        return grid_c_json(func, unpct(ins[0][2:]))
    if not func.startswith("MacGrid::"):
        return None
    g, k = fg_read(x)
    nx, ny = g["nx"], g["ny"]
    if func == "MacGrid::velocity_at":
        out = [("v", 2 * nx * ny)]
        for y in range(ny):
            for xx in range(nx):
                out += list(mac_velocity_at(g, xx, y))
        return out
    if func == "MacGrid::divergence_at":
        inv_dx = 1.0 / g["dx"]
        out = [("v", nx * ny)]
        for y in range(ny):
            for xx in range(nx):
                dvx = g["vx"][y * (nx + 1) + xx + 1] - g["vx"][y * (nx + 1) + xx]
                dvy = g["vy"][(y + 1) * nx + xx] - g["vy"][y * nx + xx]
                out.append((dvx + dvy) * inv_dx)
        return out
    if func == "MacGrid::max_speed":
        return [mac_max_speed(g)]
    if func == "MacGrid::cfl_dt":
        m = x[k][1]
        mv = mac_max_speed(g)
        return [("v", m)] + [sys.float_info.max if mv < 1e-20 else fdiv(cf * g["dx"], mv)
                             for cf in x[k + 1:k + 1 + m]]
    c = gc_read(x, k)
    steps = x[k + 13]
    g["sd"] = None
    for _ in range(steps):
        bad = mac_step(g, c)
        if bad:
            return [bad]
    n = nx * ny
    return fg_tokens(g) + [n, n]


def check_grid(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        short = func.split("::")[-1]
        cw = grid_c_want(func, x, ins)
        if cw is not None:
            want = cw
        elif func == "FluidGrid::new":
            want = fg_new(x[0] % USIZE, x[1] % USIZE, x[2])
        elif func in ("GridConfig::smoke", "GridConfig::default"):
            want = gc_tokens(GC_SMOKE)
        elif short == "from_json":
            want = fg_json(func, unpct(ins[0][2:]))
        elif func == "FluidGrid::diffuse":
            n = x[0][1]
            field = list(x[1:1 + n])
            nx, ny, rate, dt, dx, iters = x[1 + n:7 + n]
            out = fg_diffuse(field, nx % USIZE, ny % USIZE, rate, dt, dx, iters)
            want = [("v", len(out))] + out
        elif func.startswith("oracle::pressure_solve_"):
            n = x[0][1]
            p = list(x[1:1 + n])
            m = x[1 + n][1]
            div = list(x[2 + n:2 + n + m])
            k = 2 + n + m
            nx, ny = x[k], x[k + 1]
            if short == "pressure_solve_gauss_seidel":
                fg_gs(p, div, nx, ny, x[k + 2])
                want = [("v", n)] + p
            elif short == "pressure_solve_dst":
                fg_pressure_dst(p, div, nx, ny, dst_glibc)
                want = [("v", n)] + p
            elif short == "pressure_solve_cg":
                bad = fg_cg(p, div, nx, ny, x[k + 2])
                want = [bad] if bad else [("v", n)] + p
            else:
                levels = []
                fg_multigrid(p, div, nx, ny, x[k + 2], x[k + 3], levels)
                want = [("v", n)] + p + [len(levels)]
                for lv in levels:
                    for col in lv:
                        want += [("v", len(col))] + col
        else:
            g, k = fg_read(x)
            nx, ny = g["nx"], g["ny"]
            n = nx * ny
            if short == "cell_count":
                want = [word(n % USIZE)]
            elif short == "idx":
                want = [("v", n)] + [(i // nx) * nx + i % nx for i in range(n)]
            elif short == "velocity_at":
                want = [("v", 2 * n)]
                for i in range(n):
                    want += [g["vx"][i], g["vy"][i]]
            elif short == "speed_at":
                want = [("v", n)] + [math.sqrt(g["vx"][i] * g["vx"][i] + g["vy"][i] * g["vy"][i])
                                     for i in range(n)]
            elif short == "max_speed":
                want = [fg_max_speed(g)]
            elif short == "total_kinetic_energy":
                dx2 = g["dx"] * g["dx"]
                e = -0.0
                for u, v, rho in zip(g["vx"], g["vy"], g["density"]):
                    e += 0.5 * rho * (u * u + v * v) * dx2
                want = [e]
            elif short == "cfl_dt":
                m = x[k][1]
                visc = x[k + 1:k + 1 + m]
                cfl = x[k + 2 + m:k + 2 + 2 * m]
                want = [("v", m)] + [r_cfl_dt(g, a, b) for a, b in zip(visc, cfl)]
            elif short == "step":
                want = fg_run(x)
            else:
                fail(f"{where}: no check for {func}")
                continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
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
    return {"exp": unary("exp", "exp"), "log": unary("log", "log"), "log10": unary("log10", "log10"), "pow": pw,
            "sin": unary("sin", "sin"), "cos": unary("cos", "cos")}


# ── vof and phase_field (M3 part d) ─────────────────────────────────────────────────────────

def r_clamp(x, lo, hi):
    """Rust's f64::clamp (a NaN x stays NaN)."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def r_as_usize(x):
    """Rust's saturating `as usize`: NaN and negatives 0."""
    if x != x or x <= 0.0:
        return 0
    if x >= 2.0 ** 64:
        return USIZE - 1
    return int(x)


def fill_shape(field, nx, ny, dx, kind, a, b, c, d):
    """VofField::fill_rect / fill_circle and PhaseField::set_rect / set_circle: 1.0 at each matching cell centre."""
    r2 = c * c
    for y in range(ny):
        py = float(y) * dx
        for x in range(nx):
            px = float(x) * dx
            if kind == 0:
                hit = px >= a and px <= c and py >= b and py <= d
            else:
                ddx, ddy = px - a, py - b
                hit = ddx * ddx + ddy * ddy < r2
            if hit:
                field[y * nx + x] = 1.0


def vof_read(x, k=0):
    v = {"nx": x[k] % USIZE, "ny": x[k + 1] % USIZE, "dx": x[k + 2]}
    n = x[k + 3][1]
    v["alpha"] = list(x[k + 4:k + 4 + n])
    return v, k + 4 + n


def vof_tokens(v):
    return [word(v["nx"]), word(v["ny"]), v["dx"], ("v", len(v["alpha"]))] + v["alpha"]


def fresh_scratch(nx, ny, dx):
    """The scratch length of the generator's `fresh` field: new(nx, ny, dx)'s, else new(4, 4, 0.1)'s."""
    ok = nx < 1 << 16 and ny < 1 << 16 and nx >= 4 and ny >= 4 and fin(dx) and dx > 0.0
    return nx * ny if ok else 16


def vof_advect(v, vx, vy, dt):
    """One VofField::advect: None, or the failure's token. The scratch length in v["sl"]."""
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    nx, ny, dx = v["nx"], v["ny"], v["dx"]
    n = nx * ny
    if nx < 4 or ny < 4 or len(v["alpha"]) != n or len(vx) != n or len(vy) != n:
        return "E:InvalidParameter"
    if not all(fin(q) for q in vx + vy):
        return "E:Diverged"
    m = 0.0
    for q in vx + vy:
        m = r_max(m, abs(q))
    courant = fdiv(2.0 * m * dt, dx)
    q = courant / 0.5
    c = math.ceil(q) if fin(q) else q
    sub = r_as_usize(r_clamp(float(c), 1.0, 1024.0))
    if sub == 0:
        return None
    sub_dt = dt / float(sub)
    inv_dx = fdiv(1.0, dx)
    a = v["alpha"]

    def flux(v0, v1, a0, a1):
        u = 0.5 * (v0 + v1)
        donor = a0 if u > 0.0 else a1
        return u * donor * sub_dt * inv_dx

    for _ in range(sub):
        o = list(a)
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                i = y * nx + x
                fr = 0.0 if x + 1 == nx - 1 else flux(vx[i], vx[i + 1], o[i], o[i + 1])
                fl = 0.0 if x == 1 else flux(vx[i - 1], vx[i], o[i - 1], o[i])
                ft = 0.0 if y + 1 == ny - 1 else flux(vy[i], vy[i + nx], o[i], o[i + nx])
                fb = 0.0 if y == 1 else flux(vy[i - nx], vy[i], o[i - nx], o[i])
                a[i] = r_clamp(o[i] - (fr - fl + ft - fb), 0.0, 1.0)
    v["sl"] = n
    return None


def vof_json(doc):
    def us(val):
        if not (isinstance(val, tuple) and val[0] == "int") or int(val[1]) < 0:
            raise JsonRefused("not a usize")
        return int(val[1])

    try:
        f = j_struct(j_load(doc), ["nx", "ny", "dx", "alpha"])
        return vof_tokens({"nx": us(f[0]), "ny": us(f[1]), "dx": j_f64(f[2]),
                           "alpha": [j_f64(e) for e in j_arr(f[3])]})
    except JsonRefused:
        return ["E:Deserialize"]


def check_vof(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        if func == "VofField::new":
            nx, ny, dx = x[0] % USIZE, x[1] % USIZE, x[2]
            if nx < 4 or ny < 4:
                want = ["E:InvalidGridResolution"]
            elif dx <= 0.0 or not fin(dx):
                want = ["E:InvalidParameter"]
            else:
                want = [nx, ny, dx, nx * ny, 0.0, nx * ny]
        elif func == "VofField::from_json":
            want = vof_json(unpct(ins[0][2:]))
        else:
            v, k = vof_read(x)
            nx, ny = v["nx"], v["ny"]
            a = v["alpha"]
            short = func.split("::")[-1]
            queries = {"is_surface": lambda q, t: q > t and q < 1.0 - t,
                       "is_full": lambda q, t: q > 1.0 - t,
                       "is_empty": lambda q, t: q < t}
            base = short.replace("_with_threshold", "")
            if base in queries:
                ths = list(x[k + 1:k + 1 + x[k][1]]) if short.endswith("_with_threshold") else [1e-6]
                want = [("v", len(ths) * nx * ny)] + [int(queries[base](a[y * nx + xx], t))
                                                      for t in ths for y in range(ny) for xx in range(nx)]
            elif short == "interface_normal":
                want = [("v", 2 * nx * ny)]
                for y in range(ny):
                    for xx in range(nx):
                        if xx == 0 or xx >= nx - 1 or y == 0 or y >= ny - 1:
                            want += [0.0, 0.0]
                            continue
                        i = y * nx + xx
                        inv_2dx = 0.5 / v["dx"]
                        gx = (a[i + 1] - a[i - 1]) * inv_2dx
                        gy = (a[i + nx] - a[i - nx]) * inv_2dx
                        mag = math.sqrt(gx * gx + gy * gy) if fin(gx * gx + gy * gy) else gx * gx + gy * gy
                        want += [0.0, 0.0] if mag < 1e-20 else [fdiv(gx, mag), fdiv(gy, mag)]
            elif short == "total_volume":
                area = v["dx"] * v["dx"]
                t = -0.0
                for q in a:
                    t += q * area
                want = [t]
            elif short == "fill_rect":
                fill_shape(a, nx, ny, v["dx"], 0, *x[k:k + 4])
                want = vof_tokens(v)
            elif short == "fill_circle":
                fill_shape(a, nx, ny, v["dx"], 1, x[k], x[k + 1], x[k + 2], 0.0)
                want = vof_tokens(v)
            elif short == "advect":
                n1 = x[k][1]
                vx = list(x[k + 1:k + 1 + n1])
                k2 = k + 1 + n1
                n2 = x[k2][1]
                vy = list(x[k2 + 1:k2 + 1 + n2])
                dt, steps = x[k2 + 1 + n2], x[k2 + 2 + n2]
                v["sl"] = fresh_scratch(nx, ny, v["dx"])
                want = None
                for _ in range(steps):
                    bad = vof_advect(v, vx, vy, dt)
                    if bad:
                        want = [bad]
                        break
                if want is None:
                    want = vof_tokens(v) + [v["sl"]]
            else:
                fail(f"{where}: no check for {func}")
                continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


def pf_read(x, k=0):
    p = {"nx": x[k] % USIZE, "ny": x[k + 1] % USIZE, "dx": x[k + 2]}
    n = x[k + 3][1]
    p["phi"] = list(x[k + 4:k + 4 + n])
    p["mobility"], p["epsilon"] = x[k + 4 + n], x[k + 5 + n]
    return p, k + 6 + n


def pf_tokens(p):
    return [word(p["nx"]), word(p["ny"]), p["dx"], ("v", len(p["phi"]))] + p["phi"] + [p["mobility"], p["epsilon"]]


def pf_step(p, vx, vy, dt):
    """One PhaseField::step: None, or the failure's token."""
    if not fin(dt) or dt <= 0.0:
        return "E:InvalidTimestep"
    nx, ny = p["nx"], p["ny"]
    n = nx * ny
    if len(p["phi"]) != n or nx < 3 or ny < 3:
        return "E:InvalidGridResolution"
    has_v = not (len(vx) == 0 and len(vy) == 0)
    if has_v and (len(vx) != n or len(vy) != n):
        return "E:InvalidParameter"
    inv_dx = fdiv(1.0, p["dx"])
    inv_dx2 = inv_dx * inv_dx
    m, eps2 = p["mobility"], p["epsilon"] * p["epsilon"]
    f = p["phi"]
    ph = list(f)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            i = y * nx + x
            q = ph[i]
            lap = (ph[i + 1] + ph[i - 1] + ph[i + nx] + ph[i - nx] - 4.0 * q) * inv_dx2
            fp = q * q * q - q
            react = m * (eps2 * lap - fp)
            adv = 0.0
            if has_v:
                u, w = vx[i], vy[i]
                dpdx = (q - ph[i - 1] if u > 0.0 else ph[i + 1] - q) * inv_dx
                dpdy = (q - ph[i - nx] if w > 0.0 else ph[i + nx] - q) * inv_dx
                adv = -(u * dpdx + w * dpdy)
            f[i] = r_clamp(f[i] + (adv + react) * dt, -1.0, 1.0)
    for x in range(nx):
        f[x] = f[nx + x]
        f[(ny - 1) * nx + x] = f[(ny - 2) * nx + x]
    for y in range(ny):
        f[y * nx] = f[y * nx + 1]
        f[y * nx + nx - 1] = f[y * nx + nx - 2]
    return None


def pf_json(doc):
    def us(val):
        if not (isinstance(val, tuple) and val[0] == "int") or int(val[1]) < 0:
            raise JsonRefused("not a usize")
        return int(val[1])

    try:
        f = j_struct(j_load(doc), ["nx", "ny", "dx", "phi", "mobility", "epsilon"])
        return pf_tokens({"nx": us(f[0]), "ny": us(f[1]), "dx": j_f64(f[2]),
                          "phi": [j_f64(e) for e in j_arr(f[3])], "mobility": j_f64(f[4]),
                          "epsilon": j_f64(f[5])})
    except JsonRefused:
        return ["E:Deserialize"]


def check_phase_field(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        if func == "PhaseField::new":
            nx, ny, dx = x[0] % USIZE, x[1] % USIZE, x[2]
            if nx < 4 or ny < 4:
                want = ["E:InvalidGridResolution"]
            elif dx <= 0.0 or not fin(dx):
                want = ["E:InvalidParameter"]
            else:
                want = pf_tokens({"nx": nx, "ny": ny, "dx": dx, "phi": [-1.0] * (nx * ny), "mobility": 1.0,
                                  "epsilon": 2.0 * dx}) + [nx * ny]
        elif func == "PhaseField::from_json":
            want = pf_json(unpct(ins[0][2:]))
        else:
            p, k = pf_read(x)
            nx, ny = p["nx"], p["ny"]
            f = p["phi"]
            short = func.split("::")[-1]
            if short == "is_phase_b":
                want = [("v", nx * ny)] + [int(f[i] > 0.0) for i in range(nx * ny)]
            elif short == "is_interface":
                ths = list(x[k + 1:k + 1 + x[k][1]])
                want = [("v", len(ths) * nx * ny)] + [int(abs(f[i]) < t) for t in ths for i in range(nx * ny)]
            elif short == "phase_b_area":
                area = p["dx"] * p["dx"]
                t = -0.0
                for q in f:
                    t += (q + 1.0) * 0.5 * area
                want = [t]
            elif short == "set_circle":
                fill_shape(f, nx, ny, p["dx"], 1, x[k], x[k + 1], x[k + 2], 0.0)
                want = pf_tokens(p)
            elif short == "set_rect":
                fill_shape(f, nx, ny, p["dx"], 0, *x[k:k + 4])
                want = pf_tokens(p)
            elif short == "step":
                n1 = x[k][1]
                vx = list(x[k + 1:k + 1 + n1])
                k2 = k + 1 + n1
                n2 = x[k2][1]
                vy = list(x[k2 + 1:k2 + 1 + n2])
                dt, steps = x[k2 + 1 + n2], x[k2 + 2 + n2]
                want = None
                for _ in range(steps):
                    bad = pf_step(p, vx, vy, dt)
                    if bad:
                        want = [bad]
                        break
                if want is None:
                    want = pf_tokens(p) + [nx * ny]
            else:
                fail(f"{where}: no check for {func}")
                continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


# ── sph (M4) ────────────────────────────────────────────────────────────────────────────────

SPH_MIN_DENSITY = 1e-10   # sph.rs `MIN_DENSITY` (src/constants_private.cyr)


def kc_new(h):
    """sph.rs's private `KernelCoeffs::new`: (h, h², poly6, spiky_grad, visc_lap)."""
    h2 = h * h
    h3 = h2 * h
    h6 = h3 * h3
    h9 = h6 * h3
    return (h, h2, fdiv(315.0, 64.0 * PI * h9), fdiv(-45.0, PI * h6), fdiv(45.0, PI * h6))


def kc_poly6(kc, r2):
    if r2 > kc[1]:
        return 0.0
    d = kc[1] - r2
    return kc[2] * d * d * d


def sph_kernel(name, r, h):
    """The seven public kernels, in Rust's operation order."""
    if name in ("kernel_poly6", "kernel_spiky_grad", "kernel_viscosity_laplacian"):
        if name == "kernel_poly6" and r < 0.0:
            return 0.0
        if not h > 0.0:
            return 0.0
        kc = kc_new(h)
        if name == "kernel_poly6":
            return kc_poly6(kc, r * r)
        if r > kc[0] or r <= 0.0:
            return 0.0
        d = kc[0] - r
        return kc[3] * d * d if name == "kernel_spiky_grad" else kc[4] * d
    q = fdiv(r, h)
    grad = name.endswith("_grad")
    if grad:
        if q >= 2.0 or q <= 0.0:
            return 0.0
    elif not (q >= 0.0 and q < 2.0):
        return 0.0
    t = 1.0 - 0.5 * q
    c2 = name.startswith("kernel_wendland_c2")
    sigma = fdiv(7.0 if c2 else 9.0, 4.0 * PI * h * h)
    if c2:
        if grad:
            return fdiv(sigma * (-5.0 * q) * t * t * t, h)
        return sigma * t * t * t * t * (1.0 + 2.0 * q)
    g = 1.0 + 3.0 * q + 35.0 / 12.0 * q * q
    if grad:
        t5 = t * t * t * t * t
        gp = 3.0 + 35.0 / 6.0 * q
        return fdiv(sigma * (t5 * (-3.0 * g + t * gp)), h)
    t6 = t * t * t * t * t * t
    return sigma * t6 * g


def eos_tait(d, r0, c, g, P):
    """`equation_of_state_tait`, with ln and pow from P (glibc for the record, MPFR for the corrected oracle)."""
    rho0 = r_max(r0, SPH_MIN_DENSITY)
    c2 = c * c
    x = fdiv(d, rho0)
    if abs(g) < 1e-12:
        return r0 * c2 * P["log"](x)
    b = fdiv(r0 * c2, g)
    return b * (P["pow"](x, g) - 1.0)


def nn_viscosity(kind, k, n, ty, mu_p, rate, P):
    """`NonNewtonianViscosity::effective_viscosity`, pow from P."""
    sr = r_max(rate, 1e-10)
    if kind == "PowerLaw":
        return k * P["pow"](sr, n - 1.0)
    if kind == "Bingham":
        return r_min(mu_p + fdiv(ty, sr), 1e6 * mu_p)
    return r_min(k * P["pow"](sr, n - 1.0) + fdiv(ty, sr), 1e6 * k)


def particle_read(x, k):
    """A particle's 18 token values at k as a dict (m_common::particle's layout)."""
    v = x[k:k + 18]
    return {"pos": tuple(v[0:3]), "vel": tuple(v[3:6]), "acc": tuple(v[6:9]), "density": v[9],
            "pressure": v[10], "mass": v[11], "phase": v[12], "conf": tuple(v[13:16]), "temperature": v[16],
            "fuel": v[17]}


def v3_len2(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def nb_key(x32, inv32):
    """hisab's per-axis cell key: floor(x · inv) in f32, as i32 (saturating)."""
    return r_as_i32(_libm.floor(f32(x32 * inv32)))


def nb_build(pos, h):
    """`SphSolver::build_neighbors` on a fresh solver (cell size 1.0): the grid's cell size, its point and cell
    counts, the f32 positions, each particle's own key, and the CSR lists. A dict stands in for hisab's HashMap;
    each cell keeps its points in insertion (index) order, as the Vec does."""
    h32 = f32(h)
    inv_cs = f32(1.0 / h32)
    half = f32(h32 * 0.5)
    cell, inv_cell = 1.0, 1.0
    if abs(f32(cell - h32)) > f32(2.0 ** -23):
        cell, inv_cell = h32, f32(1.0 / h32)
    p32 = [tuple(f32(c) for c in q) for q in pos]
    cells = {}
    for i, q in enumerate(p32):
        cells.setdefault(tuple(nb_key(c, inv_cell) for c in q), []).append(i)
    own = [tuple(nb_key(c, inv_cs) for c in q) for q in p32]

    def wrap(v):
        return (v + 2 ** 31) % 2 ** 32 - 2 ** 31

    offsets, indices = [0], []
    for c in own:
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    # (c + d) as f32 rounds the integer first (above 2^24 it moves), then × h, then + h/2.
                    probe = tuple(f32(f32(f32(float(wrap(k + d))) * h32) + half) for k, d in zip(c, (dx, dy, dz)))
                    indices += cells.get(tuple(nb_key(t, inv_cell) for t in probe), [])
        offsets.append(len(indices))
    return cell, len(pos), len(cells), p32, own, offsets, indices


def spread_bits(v):
    v &= 0xFFFFFFFF
    for sh, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F),
                     (2, 0x3333333333333333), (1, 0x5555555555555555)):
        v = (v | (v << sh)) & mask
    return v


def morton_2d(x, y, inv):
    ix = r_as_u32(r_max(x * inv, 0.0))
    iy = r_as_u32(r_max(y * inv, 0.0))
    return spread_bits(ix) | (spread_bits(iy) << 1)


def sph_want(func, x, P):
    """A part a sph row's recomputed outputs (P: the transcendentals)."""
    if func.startswith("kernel_"):
        h, n = x[0], x[1][1]
        return [("v", n)] + [sph_kernel(func, r, h) for r in x[2:2 + n]]
    if func == "equation_of_state":
        return [x[2] * (x[0] - x[1])]
    if func == "equation_of_state_tait":
        return [eos_tait(x[0], x[1], x[2], x[3], P)]
    if func == "NonNewtonianViscosity::effective_viscosity":
        kind = x[0].split(".")[-1]
        n = x[5][1]
        return [("v", n)] + [nn_viscosity(kind, x[1], x[2], x[3], x[4], r, P) for r in x[6:6 + n]]
    if func == "batch_poly6":
        n = x[0][1]
        kc = kc_new(x[1 + n])
        return [("v", n)] + [kc_poly6(kc, r2) for r2 in x[1:1 + n]]
    if func == "batch_distance_squared":
        pos = x[0:3]
        n = x[3][1] // 3
        out = [("v", n)]
        for j in range(n):
            q = x[4 + 3 * j:7 + 3 * j]
            d = (pos[0] - q[0], pos[1] - q[1], pos[2] - q[2])
            out.append(v3_len2(d))
        return out
    if func == "SphSolver::neighbor_lists":
        h, n = x[0], x[1]
        pos = [tuple(x[2 + 3 * j:5 + 3 * j]) for j in range(n)]
        cell, ln, nc, p32, own, offsets, indices = nb_build(pos, h)
        return ([cell, ln, nc, ("v", 3 * n)] + [c for q in p32 for c in q] + [("v", 3 * n)]
                + [r_cell_key(c, h) for q in p32 for c in q] + [("v", len(offsets))] + offsets
                + [("v", len(indices))] + indices)
    if func == "sort_by_zorder":
        cell, n = x[0], x[1]
        ps = [x[2 + 18 * j:20 + 18 * j] for j in range(n)]
        if n < 2 or not (fin(cell) and cell > 0.0):
            return [("v", 0)] + [t for q in ps for t in q]
        inv = 1.0 / cell
        mx = my = sys.float_info.max
        for q in ps:
            mx, my = r_min(mx, q[0]), r_min(my, q[1])
        codes = [morton_2d(q[0] - mx, q[1] - my, inv) for q in ps]
        order = sorted(range(n), key=lambda j: codes[j])   # Python's sort is stable: (code, input index)
        return ([("v", n)] + [codes[j] - 2 ** 64 if codes[j] >= 2 ** 63 else codes[j] for j in order]
                + [t for j in order for t in ps[j]])
    if func in ("total_kinetic_energy", "max_speed"):
        n = x[0]
        ps = [particle_read(x, 1 + 18 * j) for j in range(n)]
        if func == "total_kinetic_energy":
            t = -0.0
            for q in ps:
                t += 0.5 * q["mass"] * v3_len2(q["vel"])
            return [t]
        m = 0.0
        for q in ps:
            m = r_max(m, v3_len2(q["vel"]))
        return [math.sqrt(m) if m == m and m >= 0.0 else m]
    return sph_c_want(func, x, P)


# ── sph part c (M4): the brute-force sums, the free step, the block, split and merge ──
# A particle is its 18 token values as a list (m_common::particle's order): position 0–2, velocity 3–5,
# acceleration 6–8, density 9, pressure 10, mass 11, phase 12, conformation 13–15, temperature 16, fuel 17.

def kc_spiky(kc, r):
    if r > kc[0] or r <= 0.0:
        return 0.0
    d = kc[0] - r
    return kc[3] * d * d


def kc_visc(kc, r):
    if r > kc[0] or r <= 0.0:
        return 0.0
    return kc[4] * (kc[0] - r)


def pd2(a, b):
    """`distance_squared_to`: glam's (dx² + dy²) + dz², d = a − b."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def r_density(pi, ps, kc):
    d = 0.0
    for q in ps:
        r2 = pd2(pi, q)
        if r2 <= kc[1]:
            d += q[11] * kc_poly6(kc, r2)
    return d


def r_pair_force(i, ps, kc, mu, kind):
    """`pressure_force` (kind "p") or `viscosity_force` (kind "v") on particle i."""
    pi = ps[i]
    f = [0.0, 0.0, 0.0]
    rho_i = r_max(pi[9], SPH_MIN_DENSITY)
    for j, pj in enumerate(ps):
        if j == i:
            continue
        r2 = pd2(pi, pj)
        if r2 > kc[1] or r2 < 1e-20:
            continue
        r = math.sqrt(r2)
        rho_j = r_max(pj[9], SPH_MIN_DENSITY)
        if kind == "p":
            sym = fdiv(pi[10], rho_i * rho_i) + fdiv(pj[10], rho_j * rho_j)
            scale = fdiv(-pi[11] * pj[11] * sym * kc_spiky(kc, r), r)
            for a in range(3):
                f[a] += scale * (pi[a] - pj[a])
        else:
            scale = fdiv(mu * pj[11] * kc_visc(kc, r), rho_j)
            for a in range(3):
                f[a] += scale * (pj[3 + a] - pi[3 + a])
    return f


def r_sph_step(ps, c, mu):
    """sph.rs `step` on ps (lists, changed in place): None, or the error token."""
    e = r_validate(c)
    if e:
        return e
    n = len(ps)
    if n == 0:
        return None
    kc = kc_new(c[4])
    dens = [r_density(p, ps, kc) for p in ps]
    for p, d in zip(ps, dens):
        p[9] = d
        p[10] = c[5] * (d - c[6])
    snap = [list(p) for p in ps]
    for i, pi in enumerate(snap):
        acc = [c[1], c[2], c[3]]
        rho = r_max(pi[9], SPH_MIN_DENSITY)
        for j, pj in enumerate(snap):
            if j == i:
                continue
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            rho_j = r_max(pj[9], SPH_MIN_DENSITY)
            sym = fdiv(pi[10], rho * rho) + fdiv(pj[10], rho_j * rho_j)
            p_scale = fdiv(-pj[11] * sym * kc_spiky(kc, r), r)
            for a in range(3):
                acc[a] += p_scale * (pi[a] - pj[a])
            v_scale = fdiv(fdiv(mu * pj[11] * kc_visc(kc, r), rho_j), rho)
            for a in range(3):
                acc[a] += v_scale * (pj[3 + a] - pi[3 + a])
        if not all(fin(a) for a in acc):
            return "E:Diverged"
        ps[i][6:9] = acc
    dt = c[0]
    for p in ps:
        for a in range(3):
            p[3 + a] = p[3 + a] + p[6 + a] * dt
        for a in range(3):
            p[a] = p[a] + p[3 + a] * dt
    lo, hi, damp = c[7:10], c[10:13], c[13]
    for p in ps:
        for a in range(3):
            if p[a] < lo[a]:
                p[a] = lo[a]
                p[3 + a] *= -damp
            elif p[a] > hi[a] and hi[a] > lo[a]:
                p[a] = hi[a]
                p[3 + a] *= -damp
    return None


def r_block(ox, oy, sx, sy, sp, m):
    """`create_particle_block`'s particles."""
    if not (fin(sp) and sp > 0.0) or not fin(sx) or not fin(sy):
        return []
    nx = r_as_usize(_libm.ceil(fdiv(sx, sp)))
    ny = r_as_usize(_libm.ceil(fdiv(sy, sp)))
    if nx * ny >= 2 ** 64:
        return []
    return [r_particle([ox + float(ix) * sp, oy + float(iy) * sp, 0.0], m) for iy in range(ny) for ix in range(nx)]


def r_split(parent, n, off, P):
    """`split_particle`'s children, cos and sin from P."""
    if n == 0:
        return []
    if n == 1:
        return [list(parent)]
    cm = fdiv(parent[11], float(n))
    out = []
    for k in range(n):
        angle = fdiv(2.0 * PI * float(k), float(n))
        ch = list(parent)
        ch[11] = cm
        ch[0] = ch[0] + off * P["cos"](angle)
        ch[1] = ch[1] + off * P["sin"](angle)
        out.append(ch)
    return out


def r_merge(a, b):
    """`merge_particles`."""
    total = a[11] + b[11]
    inv = fdiv(1.0, r_max(total, 1e-20))
    w = lambda k: (a[k] * a[11] + b[k] * b[11]) * inv
    return ([w(0), w(1), w(2), w(3), w(4), w(5), 0.0, 0.0, 0.0, w(9), w(10), total, a[12]]
            + list(a[13:16]) + [w(16), w(17)])


def sph_c_want(func, x, P):
    """A part c sph row's recomputed outputs, or None for another function."""
    def parts(k):
        n = x[k]
        return [list(x[k + 1 + 18 * j:k + 19 + 18 * j]) for j in range(n)]

    if func in ("compute_density", "pressure_force"):
        h = x[0]
        ps = parts(1)
        kc = kc_new(h)
        if func == "compute_density":
            return [("v", len(ps))] + [r_density(p, ps, kc) for p in ps]
        return [("v", 3 * len(ps))] + [c for i in range(len(ps)) for c in r_pair_force(i, ps, kc, 0.0, "p")]
    if func == "viscosity_force":
        h, mu = x[0], x[1]
        ps = parts(2)
        kc = kc_new(h)
        return [("v", 3 * len(ps))] + [c for i in range(len(ps)) for c in r_pair_force(i, ps, kc, mu, "v")]
    if func == "step":
        c, mu, steps = list(x[0:14]), x[14], x[15]
        ps = parts(16)
        for _ in range(steps):
            e = r_sph_step(ps, c, mu)
            if e:
                return [e]
        return [v for p in ps for v in p]
    if func == "create_particle_block":
        ps = r_block(*x[0:6])
        return [("v", len(ps))] + [v for p in ps for v in p]
    if func == "split_particle":
        ch = r_split(list(x[0:18]), x[18], x[19], P)
        return [("v", len(ch))] + [v for p in ch for v in p]
    if func == "merge_particles":
        return r_merge(list(x[0:18]), list(x[18:36]))
    return sph_d_want(func, x, P)


# ── sph part d (M4): the physics extensions ──

def lists_valid(o, idx, n):
    """sph.rs's private `neighbor_lists_valid`."""
    return (len(o) == n + 1 and all(o[k] <= o[k + 1] for k in range(len(o) - 1)) and o[-1] <= len(idx)
            and all(0 <= j < n for j in idx))


def r_viscoelastic(ps, o, idx, h, g, lam, dt):
    n = len(ps)
    if n == 0 or not lists_valid(o, idx, n):
        return
    kc = kc_new(h)
    inv_l = fdiv(1.0, r_max(lam, 1e-20))
    snap = [list(p) for p in ps]
    for i in range(n):
        pi = snap[i]
        rho = r_max(pi[9], SPH_MIN_DENSITY)
        a, b = o[i], o[i + 1]
        dudx = dudy = dvdx = dvdy = 0.0
        for j in idx[a:b]:
            if j == i:
                continue
            pj = snap[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            scale = fdiv(fdiv(pj[11], r_max(pj[9], SPH_MIN_DENSITY)) * kc_spiky(kc, r), r)
            dx, dy = pi[0] - pj[0], pi[1] - pj[1]
            dvx, dvy = pj[3] - pi[3], pj[4] - pi[4]
            dudx += dvx * scale * dx
            dudy += dvx * scale * dy
            dvdx += dvy * scale * dx
            dvdy += dvy * scale * dy
        cxx, cxy, cyy = pi[13], pi[14], pi[15]
        dcxx = 2.0 * (dudx * cxx + dudy * cxy) - (cxx - 1.0) * inv_l
        dcxy = dudx * cxy + dudy * cyy + dvdx * cxx + dvdy * cxy - cxy * inv_l
        dcyy = 2.0 * (dvdx * cxy + dvdy * cyy) - (cyy - 1.0) * inv_l
        ps[i][13:16] = [cxx + dcxx * dt, cxy + dcxy * dt, cyy + dcyy * dt]
        fx = fy = 0.0
        for j in idx[a:b]:
            if j == i:
                continue
            pj = snap[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            scale = fdiv(fdiv(pj[11], r_max(pj[9], SPH_MIN_DENSITY)) * kc_spiky(kc, r), r)
            dx, dy = pi[0] - pj[0], pi[1] - pj[1]
            dxx, dxy, dyy = pj[13] - cxx, pj[14] - cxy, pj[15] - cyy
            fx += fdiv(g * (dxx * dx + dxy * dy) * scale, rho)
            fy += fdiv(g * (dxy * dx + dyy * dy) * scale, rho)
        ps[i][6] += fx
        ps[i][7] += fy


def r_sph_heat(ps, o, idx, h, kappa, dt):
    n = len(ps)
    if n == 0 or not (fin(kappa) and kappa > 0.0) or not lists_valid(o, idx, n):
        return
    kc = kc_new(h)
    snap = [p[16] for p in ps]
    for i in range(n):
        pi = ps[i]
        acc = 0.0
        for j in idx[o[i]:o[i + 1]]:
            if j == i:
                continue
            pj = ps[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            acc += fdiv(pj[11], r_max(pj[9], SPH_MIN_DENSITY)) * (snap[j] - snap[i]) * kc_visc(kc, r)
        ps[i][16] += kappa * acc * dt


def r_delta(ps, o, idx, h, c0, delta, dt):
    n = len(ps)
    if n == 0 or not (fin(delta) and delta > 0.0) or not lists_valid(o, idx, n):
        return
    kc = kc_new(h)
    eps = 0.01 * h * h
    snap = [(p[9], p[0], p[1], p[2], p[11]) for p in ps]
    for i in range(n):
        rho_i, xi, yi, zi, _ = snap[i]
        corr = 0.0
        for j in idx[o[i]:o[i + 1]]:
            if j == i:
                continue
            rho_j, xj, yj, zj, mj = snap[j]
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            vol = fdiv(mj, r_max(rho_j, SPH_MIN_DENSITY))
            psi = fdiv(2.0 * (rho_j - rho_i), r2 + eps)
            corr -= vol * psi * r * kc_spiky(kc, r)
        ps[i][9] += delta * h * c0 * corr * dt


def r_implicit(ps, o, idx, h, nu, dt, iters):
    n = len(ps)
    pos = lambda x: fin(x) and x > 0.0
    if n == 0 or not pos(nu) or not pos(h) or not pos(dt) or not lists_valid(o, idx, n):
        return
    kc = kc_new(h)
    c = nu * dt
    stop = 1e-8 * 1e-8
    v_old = [tuple(p[3:6]) for p in ps]
    for _ in range(iters):
        mx = 0.0
        for i in range(n):
            pi = ps[i]
            rho = r_max(pi[9], SPH_MIN_DENSITY)
            w = [0.0, 0.0, 0.0]
            diag = 0.0
            for j in idx[o[i]:o[i + 1]]:
                if j == i:
                    continue
                pj = ps[j]
                r2 = pd2(pi, pj)
                if r2 > kc[1] or r2 < 1e-20:
                    continue
                r = math.sqrt(r2)
                a = fdiv(pj[11] * kc_visc(kc, r), r_max(pj[9], SPH_MIN_DENSITY) * rho)
                for k in range(3):
                    w[k] += pj[3 + k] * a
                diag += a
            den = 1.0 + c * diag
            nv = [fdiv(v_old[i][k] + w[k] * c, den) for k in range(3)]
            ch = sum_sq(nv[0] - pi[3], nv[1] - pi[4], nv[2] - pi[5])
            mx = r_max(mx, ch)
            pi[3:6] = nv
        if mx < stop:
            break


def sum_sq(x, y, z):
    return x * x + y * y + z * z


def r_invert3(m):
    """hisab 1.4.0 `matrix_inverse` on a 3×3 (LU with partial pivoting, then a dense solve per unit column)."""
    lu = [list(m[0:3]), list(m[3:6]), list(m[6:9])]
    piv = [0, 1, 2]
    for col in range(3):
        mr, mv = col, abs(lu[col][col])
        for row in range(col + 1, 3):
            v = abs(lu[row][col])
            if v > mv:
                mv, mr = v, row
        if mv < 1e-12:
            return None
        if mr != col:
            lu[col], lu[mr] = lu[mr], lu[col]
            piv[col], piv[mr] = piv[mr], piv[col]
        d = lu[col][col]
        for row in range(col + 1, 3):
            lu[row][col] = fdiv(lu[row][col], d)
            fct = lu[row][col]
            for j in range(col + 1, 3):
                lu[row][j] -= fct * lu[col][j]
    inv = [[0.0] * 3 for _ in range(3)]
    for col in range(3):
        e = [1.0 if k == col else 0.0 for k in range(3)]
        x = [e[i] for i in piv]
        for i in range(1, 3):
            sm = x[i]
            for j in range(i):
                sm -= lu[i][j] * x[j]
            x[i] = sm
        for i in (2, 1, 0):
            sm = x[i]
            for j in range(i + 1, 3):
                sm -= lu[i][j] * x[j]
            x[i] = fdiv(sm, lu[i][i])
        for row in range(3):
            inv[row][col] = x[row]
    return [v for r in inv for v in r]


def r_mls(ps, o, idx, h):
    n = len(ps)
    ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    out = [list(ident) for _ in range(n)]
    if not (fin(h) and h > 0.0) or not lists_valid(o, idx, n):
        return out
    kc = kc_new(h)
    for i in range(n):
        pi = ps[i]
        m = [0.0] * 9
        for j in idx[o[i]:o[i + 1]]:
            if j == i:
                continue
            pj = ps[j]
            d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            vol = fdiv(pj[11], r_max(pj[9], SPH_MIN_DENSITY))
            gr = fdiv(kc_spiky(kc, r), r)
            gw = [(pi[k] - pj[k]) * gr for k in range(3)]
            for row in range(3):
                for col in range(3):
                    m[row * 3 + col] += vol * d[row] * gw[col]
        inv = r_invert3(m)
        if inv is not None:
            out[i] = inv
    return out


def r_phase_change(ps, t, lh, cp, below, above):
    l_heat = r_max(lh, 1e-20)
    c = r_max(cp, 1e-10)
    for p in ps:
        if p[12] == below:
            other, heating = above, True
        elif p[12] == above:
            other, heating = below, False
        else:
            continue
        toward = (p[16] - t) * c if heating else (t - p[16]) * c
        prog = r_clamp(p[17], 0.0, 1.0)
        if toward > 0.0:
            need = (1.0 - prog) * l_heat
            if toward >= need:
                p[12] = other
                p[17] = 0.0
                extra = fdiv(toward - need, c)
                p[16] = t + extra if heating else t - extra
            else:
                p[17] = prog + fdiv(toward, l_heat)
                p[16] = t
        elif prog > 0.0:
            stored = prog * l_heat
            back = -toward
            if back >= stored:
                p[17] = 0.0
                extra = fdiv(back - stored, c)
                p[16] = t - extra if heating else t + extra
            else:
                p[17] = prog - fdiv(back, l_heat)
                p[16] = t


def r_react(ign, rate, heat, t, fuel, dt):
    """`impl ReactionProvider for CombustionConfig`."""
    if t <= ign or fuel <= 0.0:
        return 0.0, 0.0
    cons = fuel - fdiv(fuel, 1.0 + rate * dt)
    return cons, heat * cons


def r_contact(ps, c, theta, sigma, P):
    h = c[4]
    if not (fin(h) and h > 0.0) or not fin(sigma) or not fin(theta):
        return
    accel = fdiv(sigma * P["cos"](theta), h)
    lo, hi = c[7:10], c[10:13]
    for p in ps:
        w = [0.0, 0.0, 0.0]
        for a in range(3):
            if hi[a] <= lo[a]:
                continue
            if p[a] - lo[a] < h:
                w[a] += 1.0
            if hi[a] - p[a] < h:
                w[a] -= 1.0
        l2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
        if l2 < 1e-20:
            continue
        inv = fdiv(1.0, math.sqrt(l2))
        for a in range(3):
            p[6 + a] = p[6 + a] - (w[a] * inv) * accel


SEC_KINDS = ("SecondaryType.Foam", "SecondaryType.Spray", "SecondaryType.Bubble")


def r_gen_secondary(ps, thr, life):
    out = []
    for p in ps:
        am = math.sqrt(sum_sq(p[6], p[7], p[8]))
        if am > thr:
            out.append([p[0], p[1], p[2], p[3] * 1.2, p[4] * 1.2, p[5] * 1.2, life * 0.5, "e:SecondaryType.Spray"])
        elif am > thr * 0.3:
            out.append([p[0], p[1], p[2], p[3] * 0.1, p[4] * 0.1, p[5] * 0.1, life, "e:SecondaryType.Foam"])
    return out


def r_upd_secondary(ss, g, dt):
    out = []
    for s in ss:
        v = s[3:6]
        kind = s[7]
        if kind.endswith("Spray"):
            v = [v[k] + g[k] * dt for k in range(3)]
        elif kind.endswith("Foam"):
            v = [x * 0.98 for x in v]
        elif kind.endswith("Bubble"):
            v = [v[0], v[1] + 0.5 * dt, v[2]]
            v = [x * 0.95 for x in v]
        pos = [s[k] + v[k] * dt for k in range(3)]
        life = s[6] - dt
        if life > 0.0:
            out.append(pos + v + [life, kind])
    return out


def sph_d_want(func, x, P):
    """A part d sph row's recomputed outputs, or None for another function."""
    def parts(k):
        n = x[k]
        return [list(x[k + 1 + 18 * j:k + 19 + 18 * j]) for j in range(n)], k + 1 + 18 * n

    def lists(k):
        no = x[k][1]
        o = list(x[k + 1:k + 1 + no])
        ni = x[k + 1 + no][1]
        return o, list(x[k + 2 + no:k + 2 + no + ni])

    flat = lambda ps: [v for p in ps for v in p]
    lead = {"update_viscoelastic": 4, "update_heat": 3, "apply_delta_sph": 4, "apply_implicit_viscosity": 4,
            "compute_gradient_corrections": 1}
    if func in lead:
        ps, k = parts(lead[func])
        o, idx = lists(k)
        if func == "update_viscoelastic":
            r_viscoelastic(ps, o, idx, x[0], x[1], x[2], x[3])
        elif func == "update_heat":
            r_sph_heat(ps, o, idx, x[0], x[1], x[2])
        elif func == "apply_delta_sph":
            r_delta(ps, o, idx, x[0], x[1], x[2], x[3])
        elif func == "apply_implicit_viscosity":
            r_implicit(ps, o, idx, x[0], x[1], x[2], x[3])
        else:
            mats = r_mls(ps, o, idx, x[0])
            return [("v", 9 * len(mats))] + [v for m in mats for v in m]
        return flat(ps)
    if func == "oracle::invert_3x3":
        inv = r_invert3(list(x[0:9]))
        return ["n:"] if inv is None else inv
    if func == "apply_gradient_correction":
        c, g = x[0:9], x[9:12]
        return [c[0] * g[0] + c[1] * g[1] + c[2] * g[2], c[3] * g[0] + c[4] * g[1] + c[5] * g[2],
                c[6] * g[0] + c[7] * g[1] + c[8] * g[2]]
    if func == "update_phase_change":
        ps, _ = parts(6)
        r_phase_change(ps, x[0], x[1], x[2], x[3], x[4])
        return flat(ps)
    if func in ("update_reaction", "update_combustion"):
        ign, rate, heat, dt = x[0:4]
        ps, _ = parts(4)
        for p in ps:
            if func == "update_reaction":
                if p[17] > 0.0:
                    cons, dh = r_react(ign, rate, heat, p[16], p[17], dt)
                    p[17] = r_max(p[17] - cons, 0.0)
                    p[16] += dh
            elif p[16] > ign and p[17] > 0.0:
                cons = p[17] - fdiv(p[17], 1.0 + rate * dt)
                p[17] = r_max(p[17] - cons, 0.0)
                p[16] += heat * cons
        return flat(ps)
    if func == "CombustionConfig::react":
        return list(r_react(*x[0:6]))
    if func == "apply_contact_angle":
        ps, _ = parts(16)
        r_contact(ps, list(x[0:14]), x[14], x[15], P)
        return flat(ps)
    if func == "generate_secondary_particles":
        ps, _ = parts(2)
        out = r_gen_secondary(ps, x[0], x[1])
        return [("v", len(out))] + [v for s in out for v in s]
    if func == "update_secondary_particles":
        n = x[4][1]
        ss = [list(x[5 + 8 * j:13 + 8 * j]) for j in range(n)]
        out = r_upd_secondary(ss, x[0:3], x[3])
        return [("v", len(out))] + [v for s in out for v in s]
    return sph_m5_want(func, x, P)


# ── sph M5 part a: the solver's WCSPH step ──

def kc_h9(kc):
    h = kc[0]
    h2 = h * h
    h3 = h2 * h
    h6 = h3 * h3
    return h6 * h3


def kc_poly6_grad(kc, r2):
    """`KernelCoeffs::poly6_grad_scalar`: (−945/((32π)·h⁹))·d·d, d = h² − r²."""
    if r2 > kc[1]:
        return 0.0
    d = kc[1] - r2
    return fdiv(-945.0, 32.0 * PI * kc_h9(kc)) * d * d


def kc_poly6_lap(kc, r2):
    """`KernelCoeffs::poly6_laplacian`: (−945/((32π)·h⁹))·d·(3h² − 7r²)."""
    if r2 > kc[1]:
        return 0.0
    d = kc[1] - r2
    return fdiv(-945.0, 32.0 * PI * kc_h9(kc)) * d * (3.0 * kc[1] - 7.0 * r2)


def r_csf(n, lap, sigma, rho, h):
    mag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if not (fin(sigma) and sigma > 0.0) or mag != mag or mag <= fdiv(0.1, h):
        return [0.0, 0.0, 0.0]
    sc = fdiv(-sigma * lap, rho * mag)
    return [n[0] * sc, n[1] * sc, n[2] * sc]


SOLVER_CACHE = {}


def r_solver_step(ps, c, mu, sigma, verlet, prev):
    """One `SphSolver::step` on ps (lists, in place); prev is Verlet's state (a list, or None before a first step).
    Returns (error token or None, prev)."""
    e = r_validate(c)
    if e:
        return e, prev
    n = len(ps)
    if n == 0:
        return None, prev
    h = c[4]
    hw = f32(h)
    if not (fin(hw) and hw > 0.0):
        return "E:InvalidSmoothingRadius", prev
    _, _, _, _, _, offsets, indices = nb_build([tuple(p[0:3]) for p in ps], h)
    kc = kc_new(h)
    dens = []
    for i in range(n):
        d = 0.0
        for j in indices[offsets[i]:offsets[i + 1]]:
            r2 = pd2(ps[i], ps[j])
            if r2 <= kc[1]:
                d += ps[j][11] * kc_poly6(kc, r2)
        dens.append(d)
    for p, d in zip(ps, dens):
        p[9] = d
        p[10] = c[5] * (d - c[6])
    snap = [list(p) for p in ps]
    csf = sigma > 0.0
    lap0 = kc_poly6_lap(kc, 0.0)
    accs = []
    for i, pi in enumerate(snap):
        acc = [c[1], c[2], c[3]]
        rho = r_max(pi[9], SPH_MIN_DENSITY)
        cn = [0.0, 0.0, 0.0]
        lapc = fdiv(pi[11], rho) * lap0 if csf else 0.0
        for j in indices[offsets[i]:offsets[i + 1]]:
            if j == i:
                continue
            pj = snap[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            rho_j = r_max(pj[9], SPH_MIN_DENSITY)
            dd = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]]
            sym = fdiv(pi[10], rho * rho) + fdiv(pj[10], rho_j * rho_j)
            psc = fdiv(-pj[11] * sym * kc_spiky(kc, r), r)
            for a in range(3):
                acc[a] += psc * dd[a]
            vs = fdiv(fdiv(mu * pj[11] * kc_visc(kc, r), rho_j), rho)
            for a in range(3):
                acc[a] += vs * (pj[3 + a] - pi[3 + a])
            if csf:
                mor = fdiv(pj[11], rho_j)
                pg = kc_poly6_grad(kc, r2)
                for a in range(3):
                    cn[a] += mor * pg * dd[a]
                lapc += mor * kc_poly6_lap(kc, r2)
        st = r_csf(cn, lapc, sigma, rho, kc[0])
        accs.append([acc[a] + st[a] for a in range(3)])
    for i, a in enumerate(accs):
        if not all(fin(x) for x in a):
            return "E:Diverged", prev
        ps[i][6:9] = a
    dt = c[0]
    if verlet:
        started = prev is not None and len(prev) == n
        if not started:
            prev = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i, p in enumerate(ps):
            a = list(p[6:9])
            v = list(p[3:6])
            if started:
                v = [v[k] + (0.5 * (a[k] - prev[i][k])) * dt for k in range(3)]
            for k in range(3):
                p[k] = p[k] + ((v[k] * dt) + ((0.5 * a[k]) * dt) * dt)
                p[3 + k] = v[k] + a[k] * dt
            prev[i] = a
    else:
        for p in ps:
            for k in range(3):
                p[3 + k] = p[3 + k] + p[6 + k] * dt
            for k in range(3):
                p[k] = p[k] + p[3 + k] * dt
    lo, hi, damp = c[7:10], c[10:13], c[13]
    for p in ps:
        for a in range(3):
            if p[a] < lo[a]:
                p[a] = lo[a]
                p[3 + a] *= -damp
            elif p[a] > hi[a] and hi[a] > lo[a]:
                p[a] = hi[a]
                p[3 + a] *= -damp
    return None, prev


def _copy_state(st):
    ps, prev, err_ = st
    return [list(p) for p in ps], (None if prev is None else [list(q) for q in prev]), err_


def r_solver_run(c, mu, sigma, verlet, steps, ps0, key):
    """`steps` steps of one fresh solver, continuing from the longest cached state of the same scene (the trajectory
    rows share their inputs but for the step count)."""
    best = None
    for k, st in SOLVER_CACHE.get(key, []):
        if k <= steps and (best is None or k > best[0]):
            best = (k, st)
    if best is None:
        at, ps, prev, err_ = 0, [list(p) for p in ps0], None, None
    else:
        at = best[0]
        ps, prev, err_ = _copy_state(best[1])
    while at < steps and err_ is None:
        err_, prev = r_solver_step(ps, c, mu, sigma, verlet, prev)
        at += 1
    SOLVER_CACHE.setdefault(key, []).append((steps, _copy_state((ps, prev, err_))))
    return [err_] if err_ else [v for p in ps for v in p]


def r_adaptive_dt(ps, h, mu, cfl, lo, hi):
    if not (fin(h) and h > 0.0) or not (fin(cfl) and cfl > 0.0) or not fin(lo) or not fin(hi):
        return r_max(hi, lo)
    dtc = sys.float_info.max
    mv = 0.0
    for p in ps:
        mv = r_max(mv, sum_sq(p[3], p[4], p[5]))
    mv = math.sqrt(mv) if mv == mv and mv >= 0.0 else mv
    if mv > 1e-20:
        dtc = r_min(dtc, fdiv(h, mv))
    ma = 0.0
    for p in ps:
        ma = r_max(ma, p[6] * p[6] + p[7] * p[7] + p[8] * p[8])
    ma = math.sqrt(ma) if ma == ma and ma >= 0.0 else ma
    if ma > 1e-20:
        q = fdiv(h, ma)
        dtc = r_min(dtc, math.sqrt(q))
    if mu > 1e-20:
        dtc = r_min(dtc, fdiv(h * h, 6.0 * mu))
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return r_clamp(cfl * dtc, a, b)


# ── sph M5 part b: PCISPH ──

def r_pcisph_predict(snap, offsets, indices, kc, dt, pres, npa):
    pos, vel = [], []
    for i, pi in enumerate(snap):
        rho = r_max(pi[9], SPH_MIN_DENSITY)
        pa = [0.0, 0.0, 0.0]
        for j in indices[offsets[i]:offsets[i + 1]]:
            if j == i:
                continue
            pj = snap[j]
            d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            rho_j = r_max(pj[9], SPH_MIN_DENSITY)
            sym = fdiv(pres[i], rho * rho) + fdiv(pres[j], rho_j * rho_j)
            sc = fdiv(-pj[11] * sym * kc_spiky(kc, r), r)
            for a in range(3):
                pa[a] += d[a] * sc
        v = [pi[3 + a] + (npa[i][a] + pa[a]) * dt for a in range(3)]
        vel.append(v)
        pos.append([pi[a] + v[a] * dt for a in range(3)])
    return pos, vel


def r_pcisph_step(ps, c, mu, sigma, iters, max_err):
    """One `SphSolver::step_pcisph` on ps (lists, in place): None or the error token."""
    e = r_validate(c)
    if e:
        return e
    n = len(ps)
    if n == 0:
        return None
    h = c[4]
    if not (fin(f32(h)) and f32(h) > 0.0):
        return "E:InvalidSmoothingRadius"
    _, _, _, _, _, offsets, indices = nb_build([tuple(p[0:3]) for p in ps], h)
    kc = kc_new(h)
    dt, rest = c[0], c[6]
    dens = []
    for i in range(n):
        d = 0.0
        for j in indices[offsets[i]:offsets[i + 1]]:
            r2 = pd2(ps[i], ps[j])
            if r2 <= kc[1]:
                d += ps[j][11] * kc_poly6(kc, r2)
        dens.append(d)
    for p, d in zip(ps, dens):
        p[9] = d
        p[10] = 0.0
    snap = [list(p) for p in ps]
    csf = sigma > 0.0
    lap0 = kc_poly6_lap(kc, 0.0)
    npa = []
    for i, pi in enumerate(snap):
        acc = [c[1], c[2], c[3]]
        rho = r_max(pi[9], SPH_MIN_DENSITY)
        cn = [0.0, 0.0, 0.0]
        lapc = fdiv(pi[11], rho) * lap0 if csf else 0.0
        for j in indices[offsets[i]:offsets[i + 1]]:
            if j == i:
                continue
            pj = snap[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            r = math.sqrt(r2)
            rho_j = r_max(pj[9], SPH_MIN_DENSITY)
            vs = fdiv(fdiv(mu * pj[11] * kc_visc(kc, r), rho_j), rho)
            for a in range(3):
                acc[a] += vs * (pj[3 + a] - pi[3 + a])
            if csf:
                dd = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]]
                mor = fdiv(pj[11], rho_j)
                pg = kc_poly6_grad(kc, r2)
                for a in range(3):
                    cn[a] += mor * pg * dd[a]
                lapc += mor * kc_poly6_lap(kc, r2)
        st = r_csf(cn, lapc, sigma, rho, kc[0])
        acc = [acc[a] + st[a] for a in range(3)]
        if not all(fin(x) for x in acc):
            return "E:Diverged"
        npa.append(acc)
    # δ from the prototype with the fullest neighbourhood.
    best = None
    for i, pi in enumerate(snap):
        sg = [0.0, 0.0, 0.0]
        sq = 0.0
        count = 0
        for j in indices[offsets[i]:offsets[i + 1]]:
            if j == i:
                continue
            pj = snap[j]
            r2 = pd2(pi, pj)
            if r2 > kc[1] or r2 < 1e-20:
                continue
            count += 1
            r = math.sqrt(r2)
            gr = fdiv(kc_spiky(kc, r), r)
            gw = [gr * (pi[a] - snap[j][a]) for a in range(3)]
            for a in range(3):
                sg[a] += gw[a]
            sq += gw[0] * gw[0] + gw[1] * gw[1] + gw[2] * gw[2]
        if count == 0:
            continue
        dot = sg[0] * sg[0] + sg[1] * sg[1] + sg[2] * sg[2]
        q = fdiv(dt * pi[11], rest)
        denom = 2.0 * (q * q) * (dot + sq)
        if abs(denom) > 1e-20 and (best is None or count > best[0] or (count == best[0] and denom > best[1])):
            best = (count, denom)
    delta = 0.0 if best is None else fdiv(1.0, best[1])
    pres = [0.0] * n
    converged = False
    pos = vel = None
    for _ in range(max(iters, 1)):
        pos, vel = r_pcisph_predict(snap, offsets, indices, kc, dt, pres, npa)
        mx = 0.0
        errs = []
        for i in range(n):
            pd = 0.0
            for j in indices[offsets[i]:offsets[i + 1]]:
                d = [pos[i][a] - pos[j][a] for a in range(3)]
                r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                if r2 <= kc[1]:
                    pd += snap[j][11] * kc_poly6(kc, r2)
            er = pd - rest
            errs.append(er)
            mx = r_max(mx, fdiv(r_max(er, 0.0), rest))
        if mx < max_err:
            converged = True
            break
        pres = [r_max(pres[i] + errs[i] * delta, 0.0) for i in range(n)]
    if not converged:
        pos, vel = r_pcisph_predict(snap, offsets, indices, kc, dt, pres, npa)
    for i in range(n):
        if not all(fin(x) for x in pos[i]):
            return "E:Diverged"
        ps[i][3:6] = vel[i]
        ps[i][0:3] = pos[i]
        ps[i][10] = pres[i]
    lo, hi, damp = c[7:10], c[10:13], c[13]
    for p in ps:
        for a in range(3):
            if p[a] < lo[a]:
                p[a] = lo[a]
                p[3 + a] *= -damp
            elif p[a] > hi[a] and hi[a] > lo[a]:
                p[a] = hi[a]
                p[3 + a] *= -damp
    return None


PCISPH_CACHE = {}


def r_pcisph_run(c, mu, sigma, iters, max_err, steps, ps0, key):
    """`steps` PCISPH steps of one fresh solver, continuing from the longest cached state of the same scene."""
    best = None
    for k, st in PCISPH_CACHE.get(key, []):
        if k <= steps and (best is None or k > best[0]):
            best = (k, st)
    if best is None:
        at, ps, err_ = 0, [list(p) for p in ps0], None
    else:
        at, ps, err_ = best[0], [list(p) for p in best[1][0]], best[1][1]
    while at < steps and err_ is None:
        err_ = r_pcisph_step(ps, c, mu, sigma, iters, max_err)
        at += 1
    PCISPH_CACHE.setdefault(key, []).append((steps, ([list(p) for p in ps], err_)))
    return [err_] if err_ else [v for p in ps for v in p]


def sph_m5_want(func, x, P):
    """An M5 sph row's recomputed outputs, or None for another function."""
    def parts(k):
        n = x[k]
        return [list(x[k + 1 + 18 * j:k + 19 + 18 * j]) for j in range(n)]

    def frozen(v):
        return tuple(bits(t) if isinstance(t, float) else t for t in v)

    if func == "SphSolver::step":
        c, mu, sigma, verlet, steps = list(x[0:14]), x[14], x[15], x[16], x[17]
        key = (frozen(x[0:17]), frozen(x[18:]))
        return r_solver_run(c, mu, sigma, verlet, steps, parts(18), key)
    if func == "SphSolver::adaptive_dt":
        return [r_adaptive_dt(parts(5), *x[0:5])]
    if func == "SphSolver::step_pcisph":
        c, mu, sigma, iters, max_err, steps = list(x[0:14]), x[14], x[15], x[16], x[17], x[18]
        key = (frozen(x[0:18]), frozen(x[19:]))
        return r_pcisph_run(c, mu, sigma, iters, max_err, steps, parts(19), key)
    return None


def check_sph(rows):
    counts = {}
    for func, cls, ins, outs, where in rows:
        counts[func] = counts.get(func, 0) + 1
        x = [tok_val(t) for t in ins]
        got = [tok_val(t) for t in outs]
        want = sph_want(func, x, GLIBC)
        if want is None:
            fail(f"{where}: no check for {func}")
            continue
        if not same_list(got, want):
            fail(f"{where}: {func} recorded {got!r}, recomputed {want!r}")
    return counts


def tokens_text(vals):
    """Recomputed outputs as token text: floats `f:`, ("v", n) `v:n`, an error's text as it is, integers `i:`."""
    return " ".join(f"f:{bits(w):016x}" if isinstance(w, float) else (f"v:{w[1]}" if isinstance(w, tuple)
                    else (w if isinstance(w, str) else f"i:{w}")) for w in vals)


def grid_misrounds(fn, x):
    """Whether a grid row's DST calls a sine or cosine glibc misrounds: only such rows change under MPFR (the DST's
    are the grid's only transcendentals). A step takes the DST without CG or multigrid on walls, once its checks pass."""
    if fn == "oracle::pressure_solve_dst":
        n = x[0][1]
        k = 2 + n + x[1 + n][1]
        nx, ny = x[k], x[k + 1]
    elif fn == "FluidGrid::step":
        g, k = fg_read(x)
        c = gc_read(x, k)
        if c["use_cg"] or c["use_multigrid"] or c["boundary"] == "Periodic":
            return False
        nx, ny = g["nx"], g["ny"]
        if nx * ny != len(g["vx"]):
            return False   # refused before the projection
    else:
        return False
    if nx < 3 or ny < 3:
        return False
    return bool(dst_cr(nx - 2)[2] or dst_cr(ny - 2)[2])


def corrections(module, rows):
    """(fn, inputs text, corrected outputs text, note) for every row of `module` whose outputs change
    when every transcendental result is correctly rounded."""
    out = []
    for fn, cls, ins, outs, where in rows:
        log = []
        if module == "shallow" and fn == "ShallowWater::step":
            x = [tok_val(t) for t in ins]
            want = sw_run(x, mpfr_logged(log))
            got = [tok_val(t) for t in outs]
            if same_list(got, want):
                continue
            seen = list(dict.fromkeys(log))
            out.append((fn, " ".join(ins), tokens_text(want), "oracle (state); " + "; ".join(seen)))
            continue
        if module == "sph" and fn in ("equation_of_state_tait", "NonNewtonianViscosity::effective_viscosity",
                                      "split_particle", "apply_contact_angle"):
            x = [tok_val(t) for t in ins]
            want = sph_want(fn, x, mpfr_logged(log))
            got = [tok_val(t) for t in outs]
            if same_list(got, want):
                continue
            seen = list(dict.fromkeys(log))
            out.append((fn, " ".join(ins), tokens_text(want), "oracle " + " ".join(outs)[:200] + "; " + "; ".join(seen)))
            continue
        if module == "grid":
            x = [tok_val(t) for t in ins]
            if not grid_misrounds(fn, x):
                continue
            if fn == "FluidGrid::step":
                want = fg_run(x, dst_cr_logged(log))
            else:
                n = x[0][1]
                p = list(x[1:1 + n])
                m = x[1 + n][1]
                div = list(x[2 + n:2 + n + m])
                k = 2 + n + m
                fg_pressure_dst(p, div, x[k], x[k + 1], dst_cr_logged(log))
                want = [("v", n)] + p
            got = [tok_val(t) for t in outs]
            if same_list(got, want):
                continue
            seen = list(dict.fromkeys(log))
            out.append((fn, " ".join(ins), tokens_text(want), "oracle (state); " + "; ".join(seen)))
            continue
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
        elif path.stem == "shallow":
            counts = check_shallow(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"shallow: {len(rows)} rows recomputed ({summary})")
            check_corrections(root, "shallow", path, rows)
        elif path.stem == "grid":
            counts = check_grid(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"grid: {len(rows)} rows recomputed ({summary})")
            check_corrections(root, "grid", path, rows)
        elif path.stem == "sph":
            counts = check_sph(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"sph: {len(rows)} rows recomputed ({summary})")
            check_corrections(root, "sph", path, rows)
        elif path.stem in ("vof", "phase_field"):
            counts = check_vof(rows) if path.stem == "vof" else check_phase_field(rows)
            summary = ", ".join(f"{k.split('::')[-1]} {v}" for k, v in sorted(counts.items()))
            print(f"{path.stem}: {len(rows)} rows recomputed ({summary})")
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
