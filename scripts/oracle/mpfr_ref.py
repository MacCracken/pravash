"""Correctly rounded binary64 results from the system MPFR library (ADR 0004).

Copied from impetus's scripts/oracle/mpfr_ref.py (its ADR 0011), itself kimiya's (its ADR 0006) plus
`const_bits` for the correctly rounded math tables (scripts/oracle/crtables.py); keep the copies in step.

Each function returns the IEEE-754 binary64 round-to-nearest-even value of the exact mathematical
result: MPFR at 53 bits of precision with binary64's exponent range (emin = -1073, emax = 1024) and
`mpfr_subnormalize`, so subnormal results, overflow to infinity and exact ties (pow(3*2^-215, 5),
exp10(23)) round exactly as IEEE round-to-nearest does. A decimal reference at fixed precision can't
do that: it misjudges exact ties (it rounds pow(0.5, 1075) = 2^-1075 up instead of to 0).

No Python package is needed: MPFR (libmpfr.so.6, MPFR 4.x) is called through ctypes. Run
`python3 -I scripts/oracle/mpfr_ref.py` for the self-checks.
"""
import ctypes
import ctypes.util
import struct

_lib = ctypes.util.find_library("mpfr") or "libmpfr.so.6"
_m = ctypes.CDLL(_lib)

# mpfr_t on LP64: _mpfr_prec (long), _mpfr_sign (int, padded), _mpfr_exp (long), _mpfr_d (limb *).
_MPFR_T = ctypes.c_byte * 32
_RNDN = 0
_P = ctypes.c_void_p

_m.mpfr_get_version.restype = ctypes.c_char_p
_m.mpfr_set_emin.argtypes = [ctypes.c_long]
_m.mpfr_set_emax.argtypes = [ctypes.c_long]
_m.mpfr_init2.argtypes = [_P, ctypes.c_long]
_m.mpfr_set_d.argtypes = [_P, ctypes.c_double, ctypes.c_int]
_m.mpfr_get_d.argtypes = [_P, ctypes.c_int]
_m.mpfr_get_d.restype = ctypes.c_double
_m.mpfr_subnormalize.argtypes = [_P, ctypes.c_int, ctypes.c_int]

_UNARY = ("exp", "expm1", "exp2", "exp10", "log", "log2", "log10", "log1p",
          "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "cbrt", "sqrt")
_BINARY = ("pow", "atan2", "hypot")
for _name in _UNARY:
    getattr(_m, "mpfr_" + _name).argtypes = [_P, _P, ctypes.c_int]
for _name in _BINARY:
    getattr(_m, "mpfr_" + _name).argtypes = [_P, _P, _P, ctypes.c_int]

_m.mpfr_set_emin(-1073)
_m.mpfr_set_emax(1024)
_a = _MPFR_T()
_b = _MPFR_T()
_r = _MPFR_T()
for _v in (_a, _b, _r):
    _m.mpfr_init2(ctypes.byref(_v), 53)

VERSION = _m.mpfr_get_version().decode()


def _finish(t):
    t = _m.mpfr_subnormalize(ctypes.byref(_r), t, _RNDN)
    return _m.mpfr_get_d(ctypes.byref(_r), _RNDN)


def cr(name, x):
    """The correctly rounded binary64 value of the unary MPFR function `name` at x (a float)."""
    if name not in _UNARY:
        raise ValueError(name)
    _m.mpfr_set_d(ctypes.byref(_a), x, _RNDN)
    t = getattr(_m, "mpfr_" + name)(ctypes.byref(_r), ctypes.byref(_a), _RNDN)
    return _finish(t)


def cr2(name, x, y):
    """The correctly rounded binary64 value of the binary MPFR function `name` (pow, atan2, hypot)."""
    if name not in _BINARY:
        raise ValueError(name)
    _m.mpfr_set_d(ctypes.byref(_a), x, _RNDN)
    _m.mpfr_set_d(ctypes.byref(_b), y, _RNDN)
    t = getattr(_m, "mpfr_" + name)(ctypes.byref(_r), ctypes.byref(_a), ctypes.byref(_b), _RNDN)
    return _finish(t)


def exp(x): return cr("exp", x)
def expm1(x): return cr("expm1", x)
def exp10(x): return cr("exp10", x)
def log(x): return cr("log", x)
def log10(x): return cr("log10", x)
def cos(x): return cr("cos", x)
def sin(x): return cr("sin", x)
def acos(x): return cr("acos", x)
def cbrt(x): return cr("cbrt", x)
def pow(x, y): return cr2("pow", x, y)
def atan2(y, x): return cr2("atan2", y, x)


_m.mpfr_init2.argtypes = [_P, ctypes.c_long]
_m.mpfr_clear.argtypes = [_P]
_m.mpfr_const_pi.argtypes = [_P, ctypes.c_int]
_m.mpfr_const_log2.argtypes = [_P, ctypes.c_int]
_m.mpfr_get_str.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int,
                            ctypes.c_size_t, _P, ctypes.c_int]
_m.mpfr_get_str.restype = ctypes.c_void_p
_m.mpfr_free_str.argtypes = [ctypes.c_void_p]
_RNDZ = 1


def const_bits(name, n):
    """floor(c · 2^n) for c = π ("pi") or ln 2 ("log2"), as a Python int: MPFR's constant at n + 64
    bits, rounded toward zero, read back exactly in hexadecimal and scaled here (scaling inside MPFR
    would overflow the binary64 exponent range set above)."""
    v = (ctypes.c_byte * 32)()
    _m.mpfr_init2(ctypes.byref(v), n + 64)
    getattr(_m, "mpfr_const_" + name)(ctypes.byref(v), _RNDZ)
    e = ctypes.c_long(0)
    sp = _m.mpfr_get_str(None, ctypes.byref(e), 16, 0, ctypes.byref(v), _RNDZ)
    digits = ctypes.cast(sp, ctypes.c_char_p).value.decode()
    _m.mpfr_free_str(sp)
    _m.mpfr_clear(ctypes.byref(v))
    mant = int(digits, 16)
    shift = n + 4 * (e.value - len(digits))     # c = 0.digits × 16^e
    return mant << shift if shift >= 0 else mant >> -shift


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


if __name__ == "__main__":
    import math
    # Exact ties, rounded to even (glibc 2.44's pow returns 121 here: port-parity.md §8).
    assert bits(pow(3 * 2.0 ** -215, 5)) == 122
    assert bits(pow(9 * 2.0 ** -430, 2.5)) == 122
    assert pow(0.5, 1075) == 0.0 and bits(pow(0.5, 1074)) == 1
    assert exp10(23.0) == float("1e23")            # 10^23 is exactly a midpoint; the parser rounds it to even too
    assert exp10(22.0) == 1e22 and log10(1e22) == 22.0
    # Subnormal and overflow ends.
    assert bits(exp(-745.1332191019411)) == 1 and exp(-745.2) == 0.0
    assert exp(709.78) < math.inf and exp(709.8) == math.inf
    # Signed zeros, specials.
    assert bits(expm1(-0.0)) == bits(-0.0) and log(1.0) == 0.0 and math.isnan(log(-1.0))
    assert log(0.0) == -math.inf and exp(-math.inf) == 0.0
    # Agrees with glibc where glibc is correctly rounded.
    assert exp(1.0) == math.e and log(10.0) == math.log(10.0) and acos(-1.0) == math.pi
    print(f"mpfr_ref: MPFR {VERSION}, self-checks ok")
