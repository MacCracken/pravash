"""The tables of src/crmath.cyr, the correctly rounded exp, sin, cos and atan2 (ADR 0004).

π and ln 2 are computed here in exact integer arithmetic (Machin's formula; 2·atanh(1/3)), and
`validate.py` checks them against MPFR. From them:
- 2/π in 28-bit chunks for the Payne–Hanek reduction: chunk n is bits 28n + 1 … 28n + 28 after the
  binary point;
- π/2 and ln 2 as the slow path's 14-limb fixed point: limb 0 the integer part, limbs 1 … 13 the
  fraction, 28 bits each, truncated;
- phase 2's doubles: 1/ln 2 and 2/π rounded to nearest; ln 2 as hi + mid + lo with hi cut to 42
  significant bits, so k·hi is exact for |k| < 2^11; π/2 as p1 + p2 + p3 + p4 with p1 cut to 33 bits,
  so k·p1 is exact for |k| < 2^20; each later part the nearest double to what is left;
- phase 1's: 64/ln 2; ln 2/64 as hi (36 bits) + lo; π/2 as c1 + c2 (33 bits each) + c3; 1/3! … 1/7!;
  the table 2^(j/64) (j < 64, hi + lo, by an exact integer 64th root) and sin and cos of i/64 (i < 52,
  hi + lo, by their Taylor series in exact rationals);
- atan2's: π and π/2 as hi + lo, π/4 and 3π/4 nearest, and atan(i/64) (i ≤ 64, hi + lo, by Euler's series in
  exact integer arithmetic).

Copied from impetus's scripts/golden/crtables.py (its ADR 0011) with the `_PV_CR_` / `_pv_cr_` prefix and
the header's ledger note; keep the two in step.

`render()` gives src/crmath_data.cyr's text. `python3 -I scripts/golden/validate.py . --write`
writes the file; without `--write` validate.py checks it is current.
"""
from fractions import Fraction
import struct

LIMB = 28
LIMBS = 14            # the slow path's fixed point: limb 0 integer, 13 fraction limbs (364 bits)
TWO_OVER_PI_CHUNKS = 56
PREC = 2048           # bits of π and ln 2 computed (more than 28 · 56 + the guard below)
GUARD = 64


def _atan_inv(n, bits):
    """atan(1/n) · 2^bits, truncated at every term (error below the number of terms)."""
    one = 1 << bits
    x = one // n
    n2 = n * n
    total = x
    k = 1
    sign = -1
    while x:
        x //= n2
        total += sign * (x // (2 * k + 1))
        sign = -sign
        k += 1
    return total


def pi_bits(n=PREC):
    """floor(π · 2^n), by Machin's formula π = 16·atan(1/5) − 4·atan(1/239) with GUARD extra bits."""
    b = n + GUARD
    v = 16 * _atan_inv(5, b) - 4 * _atan_inv(239, b)
    return v >> GUARD


def ln2_bits(n=PREC):
    """floor(ln 2 · 2^n), by ln 2 = 2·atanh(1/3) = 2·Σ 1/((2k + 1)·3^(2k + 1))."""
    b = n + GUARD
    one = 1 << b
    x = one // 3
    total = x
    k = 1
    while x:
        x //= 9
        total += x // (2 * k + 1)
        k += 1
    return (2 * total) >> GUARD


def _double_bits(q):
    """The bits of the double nearest the rational q (Python's float(Fraction) is correctly rounded)."""
    return struct.unpack("<Q", struct.pack("<d", float(q)))[0]


def _cut(q, sig):
    """q (positive) truncated to `sig` significant bits, as an exact rational."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    while Fraction(2) ** e > q:
        e -= 1
    while Fraction(2) ** (e + 1) <= q:
        e += 1
    scale = Fraction(2) ** (sig - 1 - e)
    return Fraction(int(q * scale)) / scale


EXP2_N = 64          # phase 1 of exp: 2^(j/64), j = 0 … 63
SINCOS_N = 64        # phase 1 of sin and cos: sin and cos of i/64, i = 0 … SINCOS_ROWS − 1
SINCOS_ROWS = 52     # covers [0, π/4 + 2^-30]: π/4 · 64 ≈ 50.3, rounded up and one more


def _iroot(n, k):
    """floor(n^(1/k)) for a positive integer n (Newton's method on integers)."""
    x = 1 << (n.bit_length() // k + 1)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    while x ** k > n:
        x -= 1
    while (x + 1) ** k <= n:
        x += 1
    return x


def exp2_frac(j, bits=256):
    """2^(j/64) to `bits` fraction bits, truncated, as an exact rational."""
    return Fraction(_iroot(1 << (j + EXP2_N * bits), EXP2_N), 1 << bits)


def _sin_cos_rational(a, terms=40):
    """sin a and cos a for a rational |a| < 1 by their Taylor series (error far below 2^-200)."""
    s = Fraction(0)
    c = Fraction(0)
    t = Fraction(1)
    for n in range(2 * terms):
        if n % 2 == 0:
            c += t if (n // 2) % 2 == 0 else -t
        else:
            s += t if (n // 2) % 2 == 0 else -t
        t = t * a / (n + 1)
    return s, c


ATAN_N = 64          # phase 1 of atan2: atan(i/64), i = 0 … 64


def atan_frac(a, b, bits=400):
    """atan(a/b) for integers 0 ≤ a ≤ b, as an exact rational: Euler's series
    atan x = (x/(1 + x²))·Σ (2^(2n)·(n!)²/(2n + 1)!)·(x²/(1 + x²))^n, whose ratio x²/(1 + x²) ≤ 1/2 converges
    even at x = 1, in fixed point to `bits` fraction bits, truncated at every term (error below the number of
    terms)."""
    s2 = a * a + b * b
    one = 1 << bits
    term = one
    total = term
    n = 1
    while term:
        term = term * (2 * n) * a * a // ((2 * n + 1) * s2)
        total += term
        n += 1
    return Fraction(total * a * b, s2 * one)


def _hi_lo(q):
    """q as hi + lo: the double nearest q, and the double nearest the remainder."""
    hi = Fraction(float(q))
    return _double_bits(hi), _double_bits(q - hi)


def tables():
    """Every table value, as a dict."""
    pi = Fraction(pi_bits(), 1 << PREC)
    ln2 = Fraction(ln2_bits(), 1 << PREC)
    two_over_pi = 2 / pi
    pio2 = pi / 2
    t = {}
    frac_bits = LIMB * TWO_OVER_PI_CHUNKS
    w = int(two_over_pi * (1 << frac_bits))
    t["2opi"] = [(w >> (frac_bits - LIMB * (n + 1))) & ((1 << LIMB) - 1) for n in range(TWO_OVER_PI_CHUNKS)]

    def limbs(c):
        v = int(c * (1 << (LIMB * (LIMBS - 1))))
        return [(v >> (LIMB * (LIMBS - 1 - j))) & ((1 << LIMB) - 1) for j in range(LIMBS)]

    t["pio2"] = limbs(pio2)
    t["ln2"] = limbs(ln2)
    t["INVLN2"] = _double_bits(1 / ln2)
    t["TWO_OVER_PI"] = _double_bits(two_over_pi)
    hi = _cut(ln2, 42)
    mid = Fraction(float(ln2 - hi))
    lo = Fraction(float(ln2 - hi - mid))
    t["LN2_HI"], t["LN2_MID"], t["LN2_LO"] = _double_bits(hi), _double_bits(mid), _double_bits(lo)
    p1 = _cut(pio2, 33)
    p2 = Fraction(float(pio2 - p1))
    p3 = Fraction(float(pio2 - p1 - p2))
    p4 = Fraction(float(pio2 - p1 - p2 - p3))
    t["PIO2_1"], t["PIO2_2"], t["PIO2_3"], t["PIO2_4"] = (_double_bits(p) for p in (p1, p2, p3, p4))
    # Phase 1 of exp: n = round(x·64/ln 2), r = x − n·(ln 2/64) with hi cut to 36 bits (n·hi exact
    # for |n| < 2^17), the table 2^(j/64) as hi + lo, and 1/k! for the polynomial.
    l64 = ln2 / EXP2_N
    t["INV_L64"] = _double_bits(EXP2_N / ln2)
    l64_hi = _cut(l64, 36)
    t["L64_HI"], t["L64_LO"] = _double_bits(l64_hi), _double_bits(l64 - l64_hi)
    t["exp2"] = [_hi_lo(exp2_frac(j)) for j in range(EXP2_N)]
    fact = 1
    for k in range(1, 8):
        fact *= k
        t[f"F{k}"] = _double_bits(Fraction(1, fact))
    # Phase 1 of sin and cos: π/2 as c1 + c2 + c3, c1 and c2 cut to 33 bits (k·c1, k·c2 exact for
    # |k| < 2^20); sin and cos of i/64 as hi + lo.
    c1 = _cut(pio2, 33)
    c2 = _cut(pio2 - c1, 33)
    c3 = Fraction(float(pio2 - c1 - c2))
    t["PIO2_C1"], t["PIO2_C2"], t["PIO2_C3"] = (_double_bits(c) for c in (c1, c2, c3))
    t["sincos"] = []
    for i in range(SINCOS_ROWS):
        sn, cs = _sin_cos_rational(Fraction(i, SINCOS_N))
        t["sincos"].append(_hi_lo(sn) + _hi_lo(cs))
    # atan2: π and π/2 as hi + lo for phase 1's quadrant moves; π/4 and 3π/4 nearest for the specials; atan(i/64)
    # as hi + lo, i = 0 … 64.
    t["PI_HI"], t["PI_LO"] = _hi_lo(pi)
    t["PIO2_HI"], t["PIO2_LO"] = _hi_lo(pio2)
    t["PIO4"] = _double_bits(pi / 4)
    t["PI3O4"] = _double_bits(3 * pi / 4)
    t["atan"] = [_hi_lo(atan_frac(i, ATAN_N)) for i in range(ATAN_N + 1)]
    return t


DOUBLES = [
    ("INVLN2", "1/ln 2, nearest"),
    ("LN2_HI", "ln 2 cut to 42 significant bits"),
    ("LN2_MID", "ln 2 − hi, nearest"),
    ("LN2_LO", "ln 2 − hi − mid, nearest"),
    ("TWO_OVER_PI", "2/π, nearest"),
    ("PIO2_1", "π/2 cut to 33 significant bits"),
    ("PIO2_2", "π/2 − p1, nearest"),
    ("PIO2_3", "π/2 − p1 − p2, nearest"),
    ("PIO2_4", "π/2 − p1 − p2 − p3, nearest"),
    ("INV_L64", "64/ln 2, nearest"),
    ("L64_HI", "ln 2/64 cut to 36 significant bits"),
    ("L64_LO", "ln 2/64 − hi, nearest"),
    ("PIO2_C1", "π/2 cut to 33 significant bits"),
    ("PIO2_C2", "π/2 − c1 cut to 33 significant bits"),
    ("PIO2_C3", "π/2 − c1 − c2, nearest"),
    ("F3", "1/3!, nearest"),
    ("F4", "1/4!, nearest"),
    ("F5", "1/5!, nearest"),
    ("F6", "1/6!, nearest"),
    ("F7", "1/7!, nearest"),
    ("PI_HI", "π, nearest"),
    ("PI_LO", "π − hi, nearest"),
    ("PIO2_HI", "π/2, nearest"),
    ("PIO2_LO", "π/2 − hi, nearest"),
    ("PIO4", "π/4, nearest"),
    ("PI3O4", "3π/4, nearest"),
]


def render():
    """The text of src/crmath_data.cyr."""
    t = tables()
    out = [
        "# crmath_data.cyr — the tables of src/crmath.cyr (ADR 0004). Generated by scripts/golden/crtables.py",
        "# (python3 -I scripts/golden/validate.py . --write) from π and ln 2 computed exactly in integer",
        "# arithmetic and checked against MPFR; do not edit.",
        "#",
        "# UPSTREAM(L1): ganita:docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md",
        "#   removed with src/crmath.cyr when ganita's correctly rounded tier ships.",
        "",
        "# The fast paths' doubles, as bit patterns in typed f64 globals.",
    ]
    for name, what in DOUBLES:
        out.append(f"var _PV_CR_{name}: f64 = 0x{t[name]:016X};  # {what}")
    out += [
        "",
        f"# 2/π in {TWO_OVER_PI_CHUNKS} chunks of {LIMB} bits: chunk n is bits 28n + 1 … 28n + 28 after the binary point.",
        f"var _pv_cr_2opi[{8 * TWO_OVER_PI_CHUNKS}];",
        f"# π/2 and ln 2 in the slow path's fixed point: {LIMBS} limbs of {LIMB} bits, limb 0 the integer part.",
        f"var _pv_cr_pio2[{8 * LIMBS}];",
        f"var _pv_cr_ln2[{8 * LIMBS}];",
        f"# Phase 1 of exp: 2^(j/64) as hi, lo for j = 0 … {EXP2_N - 1} (16 bytes a row).",
        f"var _pv_cr_exp2t[{16 * EXP2_N}];",
        f"# Phase 1 of sin and cos: sin(i/64) hi, lo, cos(i/64) hi, lo for i = 0 … {SINCOS_ROWS - 1} (32 bytes a row).",
        f"var _pv_cr_sct[{32 * SINCOS_ROWS}];",
        f"# Phase 1 of atan2: atan(i/64) hi, lo for i = 0 … {ATAN_N} (16 bytes a row).",
        f"var _pv_cr_atant[{16 * (ATAN_N + 1)}];",
        "var _pv_cr_ready = 0;",
        "",
        "# Fills the tables, once.",
        "fn _pv_cr_load() {",
        "    if (_pv_cr_ready == 1) { return 0; }",
    ]
    for name in ("2opi", "pio2", "ln2"):
        for n, v in enumerate(t[name]):
            out.append(f"    store64(&_pv_cr_{name} + {8 * n}, 0x{v:07X});")
    for j, (hi, lo) in enumerate(t["exp2"]):
        out.append(f"    store64(&_pv_cr_exp2t + {16 * j}, 0x{hi:016X});")
        out.append(f"    store64(&_pv_cr_exp2t + {16 * j + 8}, 0x{lo:016X});")
    for i, row in enumerate(t["sincos"]):
        for c, v in enumerate(row):
            out.append(f"    store64(&_pv_cr_sct + {32 * i + 8 * c}, 0x{v:016X});")
    for i, (hi, lo) in enumerate(t["atan"]):
        out.append(f"    store64(&_pv_cr_atant + {16 * i}, 0x{hi:016X});")
        out.append(f"    store64(&_pv_cr_atant + {16 * i + 8}, 0x{lo:016X});")
    out += [
        "    _pv_cr_ready = 1;",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(out)
