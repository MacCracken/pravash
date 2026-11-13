"""The tables of src/crlog.cyr, the correctly rounded ln and log10 (ADR 0004).

Extracted from kimiya's scripts/golden/mathx_tables.py (its ADR 0006): the ln part only, with the
`_PV_CR_LN_` prefix. Keep the two in step: the functions below are kimiya's, unchanged.

Every value is computed from MPFR (through scripts/golden/mpfr_ref.py's ctypes binding, at 400 bits) or
from exact rational arithmetic (Python's Fraction), then split into doubles, each the round-to-nearest of
what is left, so a k-word entry is the value to about 2^(-53k) relative. The output is deterministic.

Besides writing the values, the generator checks the properties the code's error analysis relies on
(asserting, so a table that would break a proof is never written): for each of the 512 intervals of m,
r = m·c_j − 1 is exactly representable for every double m in the interval and |r| < 2^-9 (a bit-count
argument on the interval's end points); c = 1 and L = 0 on the interval holding 1; |L_j| ≥ max |r|
elsewhere (the Fast2Sum precondition); LN2H has at most 42 bits and IVH splits 26 + 27.

`render()` gives src/crlog_data.cyr's text. `python3 -I scripts/golden/validate.py . --write` writes
the file; without `--write` validate.py checks it is current.
"""
from fractions import Fraction
import ctypes
import importlib.util
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _load(name):
    """A module beside this one, by path (`-I` keeps the script's directory off sys.path)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mpfr_ref = _load("mpfr_ref")   # the binding: emin / emax already set for binary64

_m = mpfr_ref._m
_P = ctypes.c_void_p
_RNDN = 0
PREC = 400

_m.mpfr_log.argtypes = [_P, _P, ctypes.c_int]
_m.mpfr_const_log2.argtypes = [_P, ctypes.c_int]
_m.mpfr_sub.argtypes = [_P, _P, _P, ctypes.c_int]
_m.mpfr_add.argtypes = [_P, _P, _P, ctypes.c_int]
_m.mpfr_div.argtypes = [_P, _P, _P, ctypes.c_int]
_m.mpfr_set.argtypes = [_P, _P, ctypes.c_int]
_m.mpfr_neg.argtypes = [_P, _P, ctypes.c_int]
_m.mpfr_clear.argtypes = [_P]


class _V:
    """An MPFR number at `prec` bits (400 by default)."""

    def __init__(self, prec=PREC):
        self.t = (ctypes.c_byte * 32)()
        _m.mpfr_init2(ctypes.byref(self.t), prec)

    @property
    def p(self):
        return ctypes.byref(self.t)

    def __del__(self):
        _m.mpfr_clear(ctypes.byref(self.t))


def _d(x):
    v = _V()
    _m.mpfr_set_d(v.p, x, _RNDN)
    return v


def _op(fn, a, b=None):
    r = _V()
    if b is None:
        getattr(_m, fn)(r.p, a.p, _RNDN)
    else:
        getattr(_m, fn)(r.p, a.p, b.p, _RNDN)
    return r


def _ln2():
    r = _V()
    _m.mpfr_const_log2(r.p, _RNDN)
    return r


def _get(v):
    return _m.mpfr_get_d(v.p, _RNDN)


def _round_bits(v, bits):
    """v rounded to nearest at `bits` significant bits, as a double (exact for bits <= 53)."""
    r = _V(bits)
    _m.mpfr_set(r.p, v.p, _RNDN)
    return _m.mpfr_get_d(r.p, _RNDN)


def _split(v, n, first_bits=53):
    """n doubles: the first rounded to `first_bits` bits, each next the nearest double to what is left."""
    out = []
    rest = _op("mpfr_set", v)
    for i in range(n):
        h = _round_bits(rest, first_bits) if i == 0 else _get(rest)
        out.append(h)
        rest = _op("mpfr_sub", rest, _d(h))
    return out


def _fsplit(q, n, first_bits=53):
    """The same split of an exact Fraction."""
    out = []
    rest = q
    for i in range(n):
        if i == 0 and first_bits < 53:
            h = _fround_bits(rest, first_bits)
        else:
            h = float(rest)       # Fraction -> float is correctly rounded
        out.append(h)
        rest = rest - Fraction(h)
    return out


def _fround_bits(q, bits):
    """A Fraction rounded to nearest-even at `bits` significant bits."""
    if q == 0:
        return 0.0
    s = -1 if q < 0 else 1
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    scale = Fraction(2) ** (bits - 1 - e)
    n = a * scale
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return s * float(Fraction(fl) / scale)


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _sigbits(x):
    """The number of significant bits of a double (0 for 0)."""
    if x == 0.0:
        return 0
    f = Fraction(abs(x))
    n = f.numerator
    while n % 2 == 0:
        n //= 2
    return n.bit_length()


# ── ln: m·c_j ≈ 1 over 512 intervals of m in [0.7495, 1.4990), −ln c_j as a triple-double ───────

N_LN = 512
LN_OFF = 0x3FE7FC0000000000   # bits(1) − 256.5·2^43: interval 256 is centred (in bits) on 1


def _m_of_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def ln_table():
    rows = []
    for j in range(N_LN):
        lo_b = LN_OFF + j * (1 << 43)
        hi_b = lo_b + (1 << 43) - 1                 # the last double of the interval
        m_lo, m_hi = _m_of_bits(lo_b), _m_of_bits(hi_b)
        if j == 256:
            c = 1.0
        else:
            mid = (Fraction(m_lo) + Fraction(m_hi)) / 2
            inv = 1 / mid
            # c on the grid 2^-9 (c >= 1, m < 1) or 2^-10 (c < 1, m >= 1): r = m·c − 1 then fits 53 bits.
            g = 9 if m_hi < 1.0 else 10
            c = float(Fraction(round(inv * 2 ** g), 2 ** g))
        cf = Fraction(c)
        r_lo = Fraction(m_lo) * cf - 1
        r_hi = Fraction(m_hi) * cf - 1
        rmax = max(abs(r_lo), abs(r_hi))
        assert rmax < Fraction(1, 2 ** 9), (j, float(rmax))
        # Exactness of r = m·c − 1 for every double m of the interval: r is a multiple of
        # ulp(m)·ulp(c), and |r| < 2^-9, so it has at most log2(2^-9 / (ulp(m)·ulp(c))) bits.
        for mm in (m_lo, m_hi):
            if j == 256:
                continue
            um = Fraction(2) ** (_exp2(mm) - 52)
            uc = _ulp_grid(c)
            nbits = (Fraction(1, 2 ** 9) / (um * uc)).numerator.bit_length() - 1
            assert nbits <= 53, (j, nbits)
            assert _exp2(mm) == _exp2(m_lo), j    # the interval stays in one binade
        if j == 256:
            lh, ll, lt = 0.0, 0.0, 0.0
            assert m_lo < 1.0 < m_hi
        else:
            lv = _op("mpfr_neg", _op("mpfr_log", _d(c)))
            lh, ll, lt = _split(lv, 3)
            assert abs(lh) >= float(rmax), j      # Fast2Sum(L, r) precondition
        rows.append((c, lh, ll, lt))
    return rows


def _exp2(x):
    """The binary exponent of a positive double."""
    return ((bits(x) >> 52) & 0x7FF) - 1023


def _ulp_grid(c):
    """The coarsest power of two c is a multiple of."""
    f = Fraction(c)
    g = Fraction(1)
    while (f / g).denominator != 1:
        g /= 2
    while (f / (g * 2)).denominator == 1:
        g *= 2
    return g


def ln_consts():
    ln2 = _ln2()
    h, mid, lo = _split(ln2, 3, first_bits=42)
    assert _sigbits(h) <= 42
    ln10 = _op("mpfr_log", _d(10.0))
    iv = _op("mpfr_div", _d(1.0), ln10)
    ivh, ivm, ivl = _split(iv, 3)
    i1, i2 = _fsplit(Fraction(ivh), 2, first_bits=26)
    assert _sigbits(i1) <= 26 and _sigbits(i2) <= 27 and i1 + i2 == ivh
    return {
        "LN2H": h, "LN2M": mid, "LN2L": lo,
        "C3": float(Fraction(1, 3)), "C4": -0.25, "C5": float(Fraction(1, 5)),
        "C6": float(Fraction(-1, 6)), "C7": float(Fraction(1, 7)), "C8": -0.125,
        "IVH": ivh, "IVM": ivm, "IVL": ivl, "IVH1": i1, "IVH2": i2,
    }


def ln_acc():
    """(−1)^(k+1)/k for k = 2 ... 16 as triple-doubles: the accurate log1p series."""
    return [_fsplit(Fraction((-1) ** (k + 1), k), 3) for k in range(2, 17)]


# ── output ───────────────────────────────────────────────────────────────────────────────────────

def _hex(x):
    return f"0x{bits(x):016X}"


def _table(name, words, comment, per_line=4):
    out = [f"# {line}" if line else "#" for line in comment]
    out.append(f"var {name}: i64[{len(words)}] = {{")
    for i in range(0, len(words), per_line):
        chunk = words[i:i + per_line]
        sep = "," if i + per_line < len(words) else ""
        out.append("    " + ", ".join(chunk) + sep)
    out.append("};")
    out.append("")
    return out


def _scalars(prefix, d, order, comment):
    """One i64 global per value (`var _PV_CR_<prefix><NAME> = 0x…;`): a scalar global reads in one load, a
    table element in an address computation and a load, which the fast paths feel."""
    lines = [f"# {line}" if line else "#" for line in comment]
    for k in order:
        lines.append(f"var _PV_CR_{prefix}{k} = {_hex(d[k])};   # {d[k]!r}")
    lines.append("")
    return lines


LN_K_ORDER = ["LN2H", "LN2M", "LN2L", "C3", "C4", "C5", "C6", "C7", "C8",
              "IVH", "IVM", "IVL", "IVH1", "IVH2"]


def render():
    """The text of src/crlog_data.cyr."""
    L = [
        "# crlog_data.cyr — the tables of src/crlog.cyr, the correctly rounded ln and log10 (ADR 0004).",
        "# Generated by scripts/golden/lntables.py (python3 -I scripts/golden/validate.py . --write) from MPFR",
        "# (400-bit values) and exact rationals; do not edit.",
        "#",
        "# UPSTREAM(L1): ganita:docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md",
        "#   removed with src/crlog.cyr when ganita's correctly rounded tier ships.",
        "#",
        "# Every value is an IEEE-754 binary64 bit pattern, in an i64 table (read with load64) or an i64 global,",
        "# assigned to an f64 local (a hex literal in an f64 initialiser would be an integer). A multi-word entry",
        "# is a value split into doubles, each the nearest double to what the previous ones leave: a + b + c is",
        "# the value to about 2^(-53·k) relative. Adapted from kimiya's src/mathx_data.cyr (its ln part).",
        "",
    ]
    L += _table("_PV_CR_LN_T", [_hex(v) for row in ln_table() for v in row], [
        "ln: x = 2^E·m, m in [0.74951171875, 1.4990234375), j = the 9 bits of bits(x) − 0x3FE7FC0000000000",
        "below the exponent field. Four words per j at index 4j: c_j ≈ 1/m on interval j (a multiple of",
        "2^-9 if c_j > 1, of 2^-10 if c_j < 1, so r = m·c_j − 1 is exact and |r| < 2^-9), then",
        "−ln(c_j) as a triple-double. Interval 256 holds m = 1: c = 1, −ln c = 0.",
    ], per_line=4)
    L += _scalars("LN_", ln_consts(), LN_K_ORDER, [
        "ln 2 = LN2H + LN2M + LN2L (LN2H at 42 bits, so E·LN2H is exact for |E| < 2^11). C3 … C8: the",
        "Taylor coefficients 1/3, −1/4, 1/5, −1/6, 1/7, −1/8 of log1p(r) − r + r²/2. 1/ln 10 = IVH + IVM +",
        "IVL, and IVH = IVH1 + IVH2 (26 + 27 bits).",
    ])
    L += _table("_PV_CR_LN_ACC", [_hex(v) for row in ln_acc() for v in row], [
        "(−1)^(k+1)/k for k = 2 … 16 as triple-doubles (index 3·(k − 2)): the accurate log1p series.",
    ], per_line=3)
    return "\n".join(L).rstrip("\n") + "\n"
