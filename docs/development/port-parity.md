# pravash port parity — measured baselines

Measurements that set the parity classes (port plan §5). The oracle is Rust pravash 1.3.0 (commit `a88bc4e`,
tag `1.3.0`, with its hidden `oracle` hooks), built by rustc 1.99.1 on **glibc 2.44**, x86_64. Rust std f64
methods call the host libm; vector math is glam 0.29.3 through hisab 1.4.0. The port is Cyrius 6.7.18 with
ganita 1.2.15 and hisab 3.4.1.

## 1. Semantics probe (M0 step 4, 2026-11-11, x86_64 Linux ELF)

A compiled probe of the Cyrius counterpart of every Rust f64 / f32 operation pravash uses (port plan §7.1),
compared bit for bit with Rust's value on the same inputs:
- `scripts/probes/p0_semantics.cyr` is the Cyrius side;
- `scripts/probes/p0_semantics.rs` is its Rust twin (rustc `-O`, every input through `black_box`, so LLVM
  cannot fold a call);
- `scripts/probes/p0_compare.py` joins the two outputs.

The probe compiles with no type warnings.

```sh
cyrius build scripts/probes/p0_semantics.cyr build/p0_semantics && ./build/p0_semantics > build/p0_cyrius.tsv
rustc -O --edition 2024 -o build/p0_rust scripts/probes/p0_semantics.rs && ./build/p0_rust > build/p0_rust.tsv
python3 -I scripts/probes/p0_compare.py build/p0_cyrius.tsv build/p0_rust.tsv
```

### 1.1 Results

| Rust (std) | Cyrius | Result | Consequence |
|---|---|---|---|
| `min(NaN, 1)`, `min(1, NaN)`, and the same for `max` → `1.0` | `f64_min`, `f64_max` | `min(NaN, 1)` = 1, but **`min(1, NaN)` = NaN**, and the same for `max` | asymmetric. Rust semantics need `_pv_min` / `_pv_max` (165 call sites) and `_pv_clamp` (22): ledger L2 |
| `min(+0, −0)`, `max(+0, −0)` → −0 both | `f64_min`, `f64_max` | −0 both | the same on these; the helpers keep it |
| `clamp(NaN, 0, 1)` → NaN; `clamp(2, 0, 1)` → 1 | `f64_clamp` | the same bits | `_pv_clamp` still wraps it, for the NaN-bound rules (Rust panics; no caller passes one) |
| `floor`, `ceil` (incl. −0.5 → −1 / −0, −0 → −0), `sqrt`, `abs(−0)` | builtins | the same bits | class A |
| `exp` at −0.7, 1, 5.3 | `f64_exp` (x87 on x86_64) | the same bits on all three | per-target bits (x87 here, the polyfill on aarch64); not used |
| `exp` | `_f64_exp_polyfill` | **1 ulp** at `exp(1)` | the correctly rounded `exp` (L1) |
| `ln` at 0.3, 2, 10 | `f64_ln`, `_f64_ln_polyfill` | the same bits | still correctly rounded through L1, for every argument |
| `log10` at 0.25, 3 | **none** in the stdlib or ganita; `ln / LN10` | **1 and 2 ulp** | a correctly rounded `log10` (kimiya's, L1) |
| `powf(2, 0.5)`; `(1.5, 2.3)`, `(0.9, 0.333)` | `ganita_f64_pow` | **1 ulp** (`…3bcc` vs √2's `…3bcd`); same on the other two | the correctly rounded `pow` (L1) |
| `atan2(1, −1)`, `(0.3, 0.7)`; `(−2, −0.5)` | `ganita_f64_atan2` | the same; **1 ulp** at `(−2, −0.5)` | the correctly rounded `atan2` (L1); ganita's rides on the native `f64_atan` (per-target, impetus §2) |
| `hypot` at (3, 4), (0.001, 2.5), (1.1, 2.2) | `ganita_f64_hypot` | the same bits | ganita misrounds a third of arguments (impetus, MPFR); the correctly rounded `hypot` (L1) |
| `cbrt` at 2, 0.001, 27 | `ganita_f64_cbrt` | the same bits | its comments describe an exact rounding decision; the step 8 sweep measures it. If it holds, `cbrt` needs no stand-in |
| `sin`, `cos` at 0.5, 1, 3 | `f64_sin` / `_f64_sin_polyfill` (the same code), `f64_cos` | the same bits | same bits on every target; still correctly rounded through L1 |
| `to_radians(30)` | `x * (F64_PI / 180.0)` | the same bits | class B; `F64_PI` is `consts::PI`'s bits |
| `powi(x, 2)`, `powi(x, 8)` (the only exponents, 5 and 1 sites) | `x * x`; `((x²)²)²` | the same bits | LLVM's expansion of a literal exponent; class B |
| `0.1 as f32` | `f32_from(0.1)` | the same bits (`0x3dcccccd`) | f32 rounding is native |
| SPH cell key `floor(p_f32 · (1 / h_f32)) as i32`, at 1.37, 0.475, 12345.678, −0.0001, 0.15, 1e10 (saturates), NaN (→ 0) | `f32_from`, f32 `/` and `*`, `f32_to`, `f64_floor`, a saturating convert | **the same on all seven** | D3's bit-faithful keys work on native f32. The convert is `_pv_as_i32` |
| `x as i32` (saturating), the 3 key sites | `f64_to` | **differs past the range**: `1e10` gives `0x540be400`, not `i32::MAX` | `_pv_as_i32` (L2) |
| `x as usize` (saturating: NaN and negatives → 0, past 2⁶⁴ → `u64::MAX`), 19 sites | `f64_to` | **differs**: `floor(−3.7)` gives `0xfff…fc`, not 0; `1e29` gives `i64::MAX`, not `u64::MAX` | `_pv_as_usize` (L2); the probe's `sat_u64` gives Rust's value on all six cases |
| `x as u32` (saturating), the Morton code's 2 sites | `f64_to` | **differs**: `5e9` gives `0x2a05f200`, not `u32::MAX` | `_pv_as_u32` (L2) |
| `f64::MAX` (8 sites), `INFINITY` (2), `NEG_INFINITY` (1), `NAN` (2), `f32::EPSILON` (1) | no stdlib names; an `f64` / `f32` global initialized with the bit pattern (a hex word ≥ 2⁵² is the legal boxed-float idiom) | used for NaN and +∞ in the probe | step 6 generates them as typed globals, never `const` (a hex `const` is an integer) |
| `is_nan` (3), `is_finite` (101) | none public | `x != x`; `x − x == 0` | `_pv_is_nan`, `_pv_is_finite` (L2) |

### 1.2 Language notes from writing the probe

These are the language as documented, met for the first time here; none is a defect.
- `case` is a reserved word, so it can't name a parameter.
- `print(ptr, len)` takes a length; `println(cstring)` doesn't.
- Array initializer lists are module-scope constants, and they don't take `cstring` elements.
- An `f32` value goes through `f32_from` / `f32_to`, and f32 arithmetic needs both operands `f32` (§1.1's key
  probe).

## 2. Corpus baselines (M0 steps 7–8)

### 2.1 First replay (M0 step 7, 2026-11-11, x86_64 Linux ELF)

`tests/parity.tcyr`, adapted from impetus's harness, replays the corpus through the Cyrius counterparts available
before M1. It runs warning-free, with the same result in bounds-checked mode:
**2 modules, 0 failing | rows: 5,149 ok, 0 failed, 18,493 measured, 9,632 pending, 0 accepted.**

| Function | Cyrius | Rows | Result |
|---|---|---|---|
| `const` | `src/constants.cyr` through `tests/constants_rows.cyr` | 35 | all bit-exact |
| sqrt, abs, floor, ceil | builtins | 3,634 | all bit-exact (class A) |
| `as f32` | `f32_from` then `f32_to` | 1,480 | all bit-exact (class A) |
| cbrt | `ganita_f64_cbrt` | 1,025 | **all 0 ulp**: ganita's cbrt is the oracle's (correctly rounded), so cbrt needs no stand-in |
| sin, cos, sin_cos | `f64_sin` / `f64_cos` (the polyfills) | 8,628 | ≤ 1 ulp (sin 66, cos 71 rows at 1) |
| exp | `_f64_exp_polyfill` (shadow: native `f64_exp`, x87) | 2,125 | ≤ 1 ulp (163 rows at 1; native 42) |
| ln | `_f64_ln_polyfill` (shadow: native `f64_ln`) | 1,610 | ≤ 1 ulp (52 rows; native 1) |
| atan2 | ganita's atan2 on `_f64_atan_polyfill` (shadow: ganita's own, on native `f64_atan`) | 1,600 | ≤ 1 ulp (343 rows; ganita's own 328) |
| hypot | `ganita_f64_hypot` | 1,715 | ≤ 2 ulp (400 at 1, 2 at 2) |
| powf | `ganita_f64_pow` | 1,790 | ≤ 1 ulp (125 at 1); one outlier, ganita's known `pow(-1, DBL_MAX)` = +∞ against Rust's 1 (tanmatra ADR 0007, kimiya ADR 0004). The correctly rounded pow (L1) replaces it, so no ADR is needed |
| min, max | pending (M1 `_pv_min` / `_pv_max`); shadow: `f64_min` / `f64_max` | 760 each | the shadows differ on exactly 69 rows each, all NaN-second-operand cases (§1) |
| clamp, is_nan, is_finite, `as usize`, `as u32`, the SPH cell key, to_radians, powi2, powi8 | pending (M1 helpers, L2) | 7,311 | — |
| log10 | pending (the correctly rounded log10, L1) | 801 | — |

The class-C rows are *measured*: no bound is set until step 8, which repeats this on x86_64 PE and aarch64 with
per-function digests. All the counterparts above except cbrt are 1–2 ulp from glibc, which is what the correctly
rounded stand-ins (L1) are for. The harness was mutation-checked three ways, and each change fails its row with the
row's note: one bit of a golden sqrt row, a 1-ulp change to one generated constant, and floor dispatched as ceil
(813 rows fail).

### 2.2 Cross-target baselines (M0 step 8)

Pending: the same replay on x86_64 PE (wine) and aarch64 (qemu), per-function digests, and the class-C bounds.

## 3. Oracle provenance findings (M0 step 5, 2026-11-11)

What the golden generator (`scripts/oracle/`) found about where the oracle's bits come from. These findings are
about the oracle, not the port.

- **cbrt is not glibc's.** rustc 1.99.1 resolves `f64::cbrt` to `compiler_builtins::math::libm_math::cbrt`, linked
  into the binary. `nm` shows `cbrt` defined there, and not among the symbols imported from glibc: `sin`, `cos`,
  `sincos`, `atan2`, `hypot`, `exp`, `log`, `log10`, `pow`. glibc's own `cbrt` differs: `cbrt(27)` is
  3.0000000000000004 there, and 3 exactly in the oracle. `validate.py` first caught this as 491 mismatches against
  glibc. It now proves all **1,025** `cbrt` rows **correctly rounded**, with exact rationals: the cube of the
  midpoint to each neighbour of the recorded value brackets the input. So the `cbrt` rows are platform-free, and
  ganita's `cbrt`, which matched on the §1 probe points, is the candidate exact counterpart; step 8 measures it on
  the whole sweep.
- **`sin_cos` is glibc's `sincos`.** LLVM lowers `x.sin_cos()` to glibc's `sincos`. On all 2,876 probed angles it
  gives the same bits as separate `sin` and `cos` calls (`validate.py` checks each pair against ctypes `sin` and
  `cos`), so the port can treat `sin_cos` as the pair.
- **Thread count changes no row.** The generator's `parallel` build (rayon on in the oracle) reproduces the serial
  corpus byte for byte; it is check-only. No row in the M0 modules runs through a parallel loop yet; the step rows
  from M2 on do.
