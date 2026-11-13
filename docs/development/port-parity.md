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

### 2.2 Cross-target baselines and the class-C bounds (M0 step 8, 2026-11-11)

**Method.** The same replay runs three ways: `cyrius test tests/parity.tcyr` (x86_64 Linux ELF), with `--win`
(x86_64 PE, under wine), and with `--aarch64` (aarch64 Linux, under qemu). Each run is warning-free. Beside the
histograms, the harness prints a **digest** per function: a 64-bit FNV-1a fold of every output word the port
returns, in row order. Equal digests mean bit-identical results on every row, which histograms alone can't show.

**Result: every dispatched function gives the same bits on all three targets.**

| Digest | x86_64 ELF | x86_64 PE | aarch64 |
|---|---|---|---|
| `const` (35 rows) | `0x30ce491fbf8f5944` | same | same |
| `f64::sin_cos` | `0xad45b202a22e6a31` | same | same |
| `f64::sin` | `0x67180dc38fb16041` | same | same |
| `f64::cos` | `0x4212f8b5d2dcd2c9` | same | same |
| `f64::atan2` (ganita's on `_f64_atan_polyfill`) | `0x3e384187e8288aed` | same | same |
| `f64::hypot` (`ganita_f64_hypot`) | `0xdf1c2d8737693d0c` | same | same |
| `f64::exp` (`_f64_exp_polyfill`) | `0x4f0e5046d7cc0f41` | same | same |
| `f64::ln` (`_f64_ln_polyfill`) | `0xd4604f4385941ec0` | same | same |
| `f64::powf` (`ganita_f64_pow`) | `0xfbb4a8cf2186ac9c` | same | same |
| `f64::cbrt` (`ganita_f64_cbrt`) | `0x40ee50fb449fea82` | same | same |
| class A/B math rows | `0x85c8c0329e69c66f` | same | same |
| shadow: native `f64_exp` | `0xd2acc77fe0ddf658` | same | `0x4f0e5046d7cc0f41` |
| shadow: native `f64_ln` | `0x507bc977672036b1` | same | `0xd4604f4385941ec0` |
| shadow: ganita's own atan2 (native `f64_atan`) | `0x779a1ed33f36f54` | same | `0x3e384187e8288aed` |

The shadows differ on aarch64 by design. On x86_64 the native `f64_exp`, `f64_ln` and `f64_atan` are x87; on
aarch64 they are the polyfills, so there each shadow digest equals the dispatched polyfill form's x86_64 digest.
That cross-match independently confirms the choice: the polyfill forms are the cross-target ones. The
per-function histograms are identical on all three targets for every dispatched function. PE matches ELF in every
line, shadows included, as wine runs the same x87 code.

**The class-C bounds**, set here before each function's first gated comparison (`_par_c_bound` in
`tests/parity.tcyr`):

| Function | K (ulp) | Why | Measured max (all 3 targets) |
|---|---|---|---|
| sin, cos, sin_cos, atan2, exp, ln, hypot | **2** | impetus's rule for primitives: glibc and each Cyrius counterpart are within about 1 ulp of the correctly rounded value, so two implementations can differ by 2 | 1; hypot 2 (2 rows) |
| cbrt | **0** | both the oracle's (compiler_builtins) and ganita's are correctly rounded, so they must agree exactly | 0 |
| powf | — (measured until M1) | the M1 counterpart is the correctly rounded pow (L1, K = 0 against the corrected oracle). Today's, ganita's, is within 1 ulp on 1,789 rows. The one other row is ganita's open `pow(-1, \|y\| > 2^64)` defect (ganita issue 2026-11-08): +∞ against 1. No ADR accepts a counterpart that is being replaced | 1, plus that row |
| log10 | — (pending) | no counterpart until the correctly rounded log10 (L1) | — |

pravash's own class-C functions get K_F = 2·S·c + 2 at their milestones (tanmatra's port-parity rule). If K_F
exceeds 64, the primitive is made exact or an ADR is written.

**Gated result, identical on x86_64 ELF, x86_64 PE and aarch64: 2 modules, 0 failing | rows: 21,852 ok, 0 failed,
1,790 measured, 9,632 pending, 0 accepted.**

**The bounds are mutation-checked:**
- cbrt shifted by 1 ulp fails 1,023 of 1,025 rows (the two left are NaN rows, which match any NaN);
- sin shifted by 3 ulp fails 2,845 of 2,876 rows (the 31 left were 1 ulp below, so they land at 2, within K).

**What M1 changes.** The correctly rounded stand-ins (L1) replace the polyfills and ganita's pow, atan2 and hypot.
They are compared exactly (K = 0) against a corrected oracle, which is glibc's rows re-rounded where glibc
misrounds (`*.cr.tsv`, with MPFR references, as impetus's ADR 0011). Those rows are reported as accepted under the
accuracy policy. The K = 2 bounds above gate M0's interim counterparts only.

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

## 4. Probes (M0 step 9, 2026-11-11, x86_64 Linux ELF unless noted)

Each probe is a small program, answered with numbers before M1 writes code. The Cyrius programs are
`scripts/probes/p*.cyr`, `.tcyr` and `.bcyr`; the Rust sides are the `scripts/probes/rust/` crate (on the frozen
oracle, its lock seeded from `rust-old/Cargo.lock`) and the single-file `scripts/probes/p3_keys.rs`. Timings are
pinned with `taskset` and repeated where noted. The box is shared, so differences under ±20% are noise.

### 4.1 Memory on the step path (`p1_memory.tcyr`)

| Call (as a pravash step would make it) | Global heap per call, never returned |
|---|---|
| `hvec3_add` (glam's DVec3 arithmetic) | 24 B |
| hisab `spatial_hash`: clear, insert 1,000 particles, 27 `query_cell` each, one step | **4,117,081 B** (≈ 41 MB a step at 10⁴ particles) |
| `num_dst` / `num_idst`, n = 62 | 10,208 B (≈ 2.5 MB a step for a 64² DST solve) |
| `num_dst`, n = 126 | 20,448 B |
| ganita `mat_inv`, 3×3 (the MLS correction: one per particle per step) | 248 B (≈ 2.5 MB a step at 10⁴ particles) |
| `ad_tape` gradient: reset, 18 operations, `ad_grad` | **0 B** |
| freelist: 100,000 alloc/free pairs of 64 B | 0 B of global heap growth |
| a growable step arena, 300 steps of 10,000 × 240 B | capacity 2,424,832 B after step 1 and after step 300 |

**Decision.** This confirms §0.1 of the port plan with pravash's own numbers:
- solver storage goes on freelist, and step scratch in a solver-owned arena reset every step;
- no hisab or ganita call that returns a value goes on the step path while cyrius S1 / hisab H9 are open;
- ledger rows L3 (vectors), L4 (the neighbour grid), L6 (the DST) and L7 (the 3×3 inverse) are needed on memory
  grounds alone;
- the autodiff tape allocates nothing once reset, and is not per-step anyway, so it stays on hisab.

### 4.2 Particle layout and vector form (`p2_layout.bcyr`)

One SPH density pass: 4,096 particles, 141,376 candidate pairs, the sum in Rust's order. All four forms give
**the same bits** (digest `0xb7fa6660c11c64c7`). Three pinned runs, load average under 1.6:

| Form | Per pass | Per pair | vs (a) |
|---|---|---|---|
| (a) array-of-structs (`FluidParticle`'s 144 B layout), fields read into `f64` locals | **1.83 ms** | 12.9 ns | 1× |
| (b) structure-of-arrays (`ParticleSoa`'s columns), `f64` locals | 2.01 ms | 14.2 ns | 1.1× |
| (c) array-of-structs, struct values (`PvV3` with `-` overloaded) | 4.6 ms | 32.5 ns | 2.5× |
| (d) the pair difference in an `f64v4`, lanes summed in Rust's order | 6.45 ms | 45.6 ns | 3.5× |

**Decision.** Particles are typed stride arrays of `PvParticle` records in `FluidParticle`'s layout, which is also
the Rust API's shape, and hot loops read fields into `f64` locals. Struct values cost 2.5× here, the same as
impetus measured: they are the readable form off the hot path, and the cost is a compiler finding (filed at
step 10). SIMD can't vectorize a sum without changing Rust's summation order, and within one pair `f64v4_make` and
the lane reads cost more than they save (also filed at step 10).

### 4.3 f32 cell keys at scale (`p3_keys.rs`, `p3_keys.cyr`)

10⁶ positions: 600,000 in a box, every cell boundary ±2 doubles for six smoothing radii, and the rest log-uniform up
to |x| = 10⁷. Rust's key and the Cyrius key (`f32_from`, f32 `/` and `*`, `f32_to`, `f64_floor`, a saturating
convert) agree on **all 10⁶ on x86_64 ELF, x86_64 PE and aarch64**. The operator trap warns as documented: an f32
left operand with an f64 right operand gives *f32 arithmetic with a non-f32 right operand*. The `as f32` packing of
`PackedParticles` is the `f64::as_f32` rows, all bit-exact (§2).

**Decision.** D3's bit-faithful f32 keys port with no risk.

### 4.4 The neighbour candidate order (`p4_neighbor_order`)

The expectation uses no hash map: particle indices grouped by cell key, in index order, concatenated over the 27
offsets in `build_neighbors`' order. Every list the oracle dumped equals it:

| Scenario | Particles | Every list equal | Probe points that land outside their intended cell |
|---|---|---|---|
| 2D block at the origin | 1,600 | yes | 0 of 43,200 |
| 3D cloud at the origin | 2,744 | yes | 0 of 74,088 |
| 2D block at 10³ | 1,600 | yes | 0 of 43,200 |
| 2D block at 10⁵ | 1,600 | yes | 0 of 43,200 |
| 2D block at 10⁶ | 1,600 | yes | **38,400 of 43,200** |

**Decision.** The order is (cell offset, insertion), so the port builds it from sorted arrays with no hash map.

The last column measures the roadmap's precision defect (D3). At |x| ≈ 10⁶ the f32 spacing (0.0625) exceeds
h = 0.05, so the probe point `(cx + dx) · h + h/2` rounds into another cell for 89% of lookups. Each neighbour list
there comes from the wrong cells. That is what D3's switch to f64 keys before 2.0 fixes.

### 4.5 Z-order sort ties (`p5_sort_ties`)

`sort_by_zorder` sorts with `sort_unstable_by_key` (ipnsort on rustc 1.99). Against the stable order on (key, input
index), with the key sequences checked non-decreasing through a replica of the private Morton code:

| Input | Particles | Tie groups (particles) | Positions differing from the stable order |
|---|---|---|---|
| 40 × 40 lattice at h/2 | 1,600 | 400 (1,600) | 1,200 |
| 100 × 100 lattice at h/4 | 10,000 | 625 (10,000) | 9,382 |
| random cloud of 5,000 | 5,000 | 400 (5,000) | 4,577 |
| the 40 × 40 lattice, sorted once already | 1,600 | 400 (1,600) | **0** |

**Decision.**
- **Rust's tie order can't be reproduced cheaply.** It is ipnsort's, so matching it means transcribing core's sort.
  ADR 0002 #3 applies: the port sorts stably on (key, input index). The golden rows compare the key sequence and
  the multiset (class A), and later steps start from the oracle's order.
- **An already-sorted input** (what a solver sees from its second sort on) comes back unchanged from both sorts.
- **Ledger row L9** (a stable sort) is definite.

### 4.6 Parallelism (`p6_parallel`, `p6_pool.bcyr`)

**Rust.** The oracle's serial and `parallel` (rayon) builds give **identical final-state digests on all 14
scenarios**: the five SPH variants, six `FluidGrid` variants, `MacGrid`, `ShallowWater` and `FlipSolver` (APIC),
300 steps each. Thread count is not a parity variable on the oracle side.

**Cyrius.** A pool over `thread_create` and `chan_*`, splitting the density pass into fixed index ranges: each
particle's density is written by one worker, from its own list in Rust's order. Cores 8–15, one run, load
average 3.3:

| Particles | T | Persistent pool | Spawn per pass | Digest | Pool heap growth |
|---|---|---|---|---|---|
| 1,024 | 1 / 2 / 4 / 8 | 488 / 249 / 207 / 181 µs | 565 / 304 / 262 / 303 µs | = serial at every T | 0 B |
| 16,384 | 1 / 2 / 4 / 8 | 7.61 / 3.84 / 2.32 / 2.10 ms | 8.07 / 4.22 / 2.05 / 1.98 ms | = serial at every T | 0 B |
| 102,400 | 1 / 2 / 4 / 8 | 47.6 / 24.0 / 15.7 / 12.2 ms | 47.7 / 24.2 / 12.2 / 11.3 ms | = serial at every T | 0 B |

**Decision** (ADR 0002 #4):
- **Shape:** a persistent pool (`src/par.cyr`). It costs the same as spawning per pass at N ≥ 16k, within noise,
  and much less at small N. Speedup is 2× at T = 2 and 3–3.9× at T = 8, so the loop is memory-bound past four
  threads.
- **Bits:** every T gives the serial bits.
- **Memory:** workers allocate nothing.
- **Targets:** `THREADS_CONCURRENT` is 1 on Linux.

### 4.7 Bundle shape (probe 7)

A consumer built with the project's `[deps]`, with and without dead-code elimination:

| Consumer | `CYRIUS_DCE=0` | `[build] dce = true` | Compile |
|---|---|---|---|
| uses nothing (the stdlib prepend only) | 1,218,896 B | **407,888 B** (−67%) | 0.9 s |
| includes all of hisab (70,654 lines), calls `hvec3_add` | 3,056,544 B | **676,768 B** (−78%) | 1.7 s |

**Decision** (ADR 0002 #5). One `[lib]` bundle, with no profiles: dead-code elimination drops what a consumer doesn't
call. What a large bundle costs is compile time, about 0.8 s per 70k lines, which is small at pravash's size.

### 4.8 hisab and ganita against hisab 1.4.0 (`p8_hisab`, `p8_hisab.tcyr`)

| Candidate | Against (hisab 1.4.0, as the oracle calls it) | Records | Same bits | Largest distance |
|---|---|---|---|---|
| `num_dst`, n ≤ 14 | `dst` | 36 | 20 | 384 ulp |
| `num_idst`, n ≤ 14 | `idst` | 36 | 21 | 410 ulp |
| `num_dst`, n = 30 … 126 | `dst` | 12 | 0 | 66,671 ulp |
| `num_idst`, n = 30 … 126 | `idst` | 12 | 0 | 67,196 ulp |
| ganita `mat_inv` | `matrix_inverse` (through `invert_3x3`), 300 moment matrices, singular ones included | 300 | **300** | 0 |
| `ad_tape` gradient, arithmetic only | `reverse_gradient` | 100 | 84 | 3 ulp |
| `ad_tape` gradient, through sin and exp | `reverse_gradient` | 100 | 93 | 1 ulp |

**Decision.**
- **DST, ledger L6, definite:** hisab 1.4.0's DST is the direct O(N²) sum with a glibc `sin` per term.
  hisab 3.4.1's is a different algorithm (an FFT past small n), and it allocates 10–20 KB a call. The port
  transcribes 1.4.0's operation order on the correctly rounded `sin` (L1).
- **The 3×3 inverse, L7, definite on memory grounds only:** ganita's `mat_inv` gives the oracle's bits on every
  matrix, singular verdicts included. The stand-in is an allocation-free copy of the same algorithm, with the same
  bits, until ganita has a caller-buffer variant.
- **Autodiff, no stand-in:** `KernelDerivatives::autodiff_gradient` uses hisab's `ad_tape`, which takes 0 B per
  evaluation and is not per-step. Its reverse pass accumulates in a different order from 1.4.0's (3 ulp at most on
  arithmetic-only gradients), so its rows are **class D**: plan §5's library-backed path, set here before their
  first comparison.

## 5. M1a part 1: the Rust-semantics helpers (2026-11-11)

`src/mathx.cyr`, the first `[lib]` module after `src/constants.cyr` (bundle `dist/pravash.cyr`; tests include
`tests/lib.cyr`, which mirrors `[lib]`). It holds the helpers the probe (§1) called for:
- `_pv_min`, `_pv_max`, `_pv_clamp`, `_pv_is_nan`, `_pv_is_finite`;
- `_pv_to_radians`, `_pv_powi2`, `_pv_powi8`;
- the saturating `_pv_as_i32`, `_pv_as_u32`, `_pv_as_usize`.

It is a stand-in, marked `UPSTREAM(L2)`.

**Result:** every one of the **6,371** rows newly dispatched passes on the first run, all bit-exact (the class B
rows `to_radians`, `powi2` and `powi8` at 0 ulp on all 2,801). The replay, identical on x86_64 ELF, `--bounds`,
x86_64 PE and aarch64, warning-free, is **2 modules, 0 failing | rows: 28,223 ok, 0 failed, 1,790 measured, 3,261
pending**. The pending rows are now only the SPH cell key (2,460, M4's `src/neighbor.cyr`) and log10 (801, part 2's
`crmath`). The class A/B digest, now covering the helper rows, is `0xeba4c9735da7855` on all three targets.

**Mutation checks:**
- `_pv_max` returning x on equal operands fails exactly the 2 rows that pin Rust's ±0 rule;
- `_pv_as_usize` keeping negatives fails 111 rows.

## 6. M1a part 2: the correctly rounded primitives (2026-11-11, ADR 0004)

Four `[lib]` modules after `src/constants.cyr`, before `src/mathx.cyr`, each marked `UPSTREAM(L1)`:
- `src/crmath_data.cyr` and `src/crmath.cyr`: impetus's crmath (exp, sin, cos, sin_cos, hypot, pow, atan2), with
  the `_pv_cr_` prefix. The tables are generated by a copy of impetus's `crtables.py`, and their bits equal
  impetus's (a diff of the two files, prefix aside);
- `src/crlog_data.cyr` and `src/crlog.cyr`: kimiya's ln and log10, on crmath's TwoSum and TwoProduct, which are the
  same operations as kimiya's. The tables come from `scripts/oracle/lntables.py`, the ln part of kimiya's
  `mathx_tables.py`, and every value equals kimiya's.

`src/mathx.cyr`'s wrappers (`_pv_exp`, `_pv_ln`, `_pv_log10`, `_pv_powf`, `_pv_cbrt`, `_pv_hypot`, `_pv_sin_cos`,
`_pv_sin`, `_pv_cos`, `_pv_atan2`) are what the dispatcher, and later the solvers, call.

**The corrected oracle.** `tests/golden/math.cr.tsv`, written by `validate.py --write` with MPFR 4.2.2, holds the
43 math rows glibc 2.44 misrounds:

| Function | Rows | glibc misrounds | Note |
|---|---|---|---|
| log10 | 801 | 21 | |
| hypot | 1,715 | 11 | |
| sin, sin_cos | 2,876 each | 3 each | the same three sin results |
| atan2 | 1,600 | 2 | |
| exp | 2,125 | 1 | |
| ln | 1,610 | 1 | |
| powf | 1,790 | 1 | |
| cos | 2,876 | 0 | |
| cbrt | 1,025 | — | the oracle's (compiler_builtins') is already correctly rounded, §3 |

`validate.py` also checks:
- the two table files are their generators' output;
- crmath's π and ln 2 equal MPFR's to 2,048 bits, and every table's head is MPFR's correctly rounded value;
- the ln tables' ln 2 and 1/ln 10 are within 2^-150 of integer series that MPFR doesn't share
  (ln 10 = 3·ln 2 + 2·atanh(1/9)).

**Result:** with K = 0 for every class C math function, powf and log10 included, the replay is **2 modules,
0 failing | rows: 30,771 ok, 0 failed, 0 measured, 2,460 pending, 43 accepted (ADR 4: 43)**. It is identical on
x86_64 ELF, x86_64 PE and aarch64, warning-free, and under `--bounds`. On each of the 43 rows the port equals the
corrected outputs and not glibc's. The pending rows are only the SPH cell key (M4).

The digests are identical on all three targets for every dispatched function; the native shadows differ on aarch64
by design (§2.2):

| Digest | All three targets |
|---|---|
| f64::sin_cos | `0x17c86316a0b1d71b` |
| f64::sin | `0x53d90652356a097c` |
| f64::cos | `0x52871fe987f1c18e` |
| f64::atan2 | `0xe4c5252092c030de` |
| f64::hypot | `0xba018fb7078e3e67` |
| f64::exp | `0xbb495cab624cd6a7` |
| f64::ln | `0x507bc977672036b1` |
| f64::log10 | `0xd3237f1c13951705` |
| f64::powf | `0xc10ae7b7dbf204a4` |
| f64::cbrt | `0x40ee50fb449fea82` |
| class A/B math rows | `0xeba4c9735da7855` (unchanged from §5) |
| const | `0x30ce491fbf8f5944` |

On x86_64 the native x87 `f64_ln` shadow gives the dispatched ln's digest: it is correctly rounded on all 1,610
rows. The other natives are not (exp 41 rows at 1 ulp), and on aarch64 the native shadows are the polyfills.

**The replaced counterparts, against the correctly rounded value** (shadows, ungated; the reference is the
corrected oracle where glibc misrounds, glibc elsewhere). These are the counts the ganita request now cites:

| Shadow | 1 ulp | 2 ulp | Other |
|---|---|---|---|
| `_f64_exp_polyfill` | 164 of 2,125 | 0 | |
| `_f64_ln_polyfill` | 53 of 1,610 | 0 | |
| `f64_sin` / `f64_cos` (the polyfills on every target) | 67 / 71 of 2,876 | 0 | |
| ganita's atan2 on the polyfill atan | 345 of 1,600 | 0 | |
| `ganita_f64_atan2` (native atan) | 330 of 1,600 | 0 | |
| `ganita_f64_hypot` | 399 of 1,715 | 2 | |
| `ganita_f64_pow` | 126 of 1,790 | 0 | 1 NaN/∞ row: ganita's open `pow(-1, huge)` defect |

**Cyrius-only tests** (`tests/crmath.tcyr`, 85 checks in 18 groups, passing on all three targets):
- impetus's crmath groups: the IEEE specials, fast = slow on seeded arguments, phase 1's decision rate, the
  extreme reductions, and hypot, pow and atan2's exact cases, ties and values;
- kimiya's ln and log10 groups:
  - the specials, with every row checked by the full function and the accurate path alone;
  - log10(10^k) = k for k = 0 … 22;
  - the Lefèvre–Muller hard cases and the arguments kimiya found, of which the fast path sends exactly 3 to the
    accurate path;
  - fast = accurate on 3,000 seeded arguments across all positive finite doubles.

**Mutation checks** (each restored and re-verified after):
- one corrected output's last bit flipped: that row FAILED, and `validate.py` fails ("differs from the corrected
  oracle");
- `math.cr.tsv` deleted: the 43 rows fail against glibc's outputs, and `validate.py` fails;
- a correction that matches no golden row: "1 matching no golden row" fails the module;
- `_pv_exp` on the exp polyfill: 164 rows FAILED;
- one bit of one `−ln c_j` word in `src/crlog_data.cyr`: 3 ln and log10 rows FAILED, and `validate.py` fails;
- one bit of one `2^(j/64)` word in `src/crmath_data.cyr`: 36 exp and 24 powf rows FAILED, and `validate.py` fails.

The table mutations reach only the rows whose arguments use that table entry, so `validate.py`'s regeneration check
is the gate that covers every table word.

**For M3:** crmath loads its tables lazily behind a plain flag, and counts its slow-path calls in globals.
`src/par.cyr` calls `_pv_cr_load()` before it starts workers, so that on aarch64 a worker never sees the flag set
before the tables are visible. The counters are racy but never feed a result (ADR 0004, Consequences).

## 7. M1a part 3: error, vec3, common (2026-11-11)

Five `[lib]` modules, in order:
- `src/error.cyr` first: the `PV_ERR_*` codes, `pv_error_name` and `pv_error_message`;
- after `src/mathx.cyr`: `src/vec3.cyr` (`PvVec3`, glam's operations in glam's order, `UPSTREAM(L3)`),
  `src/column.cyr` (a freelist column, `UPSTREAM(L13)`) and `src/common.cyr`.

`src/common.cyr` holds:
- `PvFluidMaterial`, its five presets as struct globals, and `custom`;
- `PvFluidParticle`, the 144-byte record of probe 2, with the conformation as an inline `(f64, f64, f64)`;
- `PvFluidConfig`, with `validate` and `check_cfl`;
- `PvParticleSoa`, 18 columns;
- `PvArenaHandle` and `PvParticleArena`, whose free list is allocated once at capacity / 2 + 1 entries, so no
  operation allocates.

The API shapes are ADR 0002 #1, #8 (no aborts) and #9 (out pointers, (pointer, count) views, `_drop`).

**The golden module** `tests/golden/common.tsv` (scripts/oracle/src/m_common.rs) has **3,683 rows**:
- `custom` 297;
- `new` and `new_2d` 90 each;
- `speed_squared`, `speed` and `kinetic_energy` 330 each;
- the two particle distances and the SoA's `distance_squared` 280 each;
- `water_2d` 1, `validate` 644, `check_cfl` 492;
- the SoA round trip 60 and `consistent_len` 120;
- 59 arena scenarios.

Each scenario row replays an op sequence and gives every op's result, then the arena's private state from the
oracle dump: watermark, generation, free list, and every slot's mass, which marks where each block went. The 59 are
the Rust tests' scenarios, foreign and stale handles, `alloc(usize::MAX)`, a zero-capacity arena, and 48 seeded
mixes at six capacities.

`validate.py` recomputes every row without the oracle, in Rust's order:
- the validation rules;
- the defaults;
- glam's dot order;
- a transcription of the arena's free-list logic.

The generator gained a direct `hisab =1.4.0` dependency for `DVec3`; the lock changed by that one edge.

**Result:** on the first replay, all 3,683 rows pass, every one bit-exact (the class B rows at 0 ulp on all 1,830).
The replay is **3 modules, 0 failing | rows: 34,454 ok, 0 failed, 0 measured, 2,460 pending, 43 accepted (ADR 4: 43)**.
It is identical on x86_64 ELF, x86_64 PE and aarch64, warning-free, and under `--bounds`. `tests/common.tcyr` ports
47 of the 49 Rust tests (108 checks); the 2 serde tests wait for part 4's JSON. `tests/error.tcyr` ports the 6
error tests, re-expressed for codes (15 checks).

**Surface parity** (`scripts/surface_parity.py`, impetus's, now in CI): error and common, **98 items ok, 0
missing**. Every other `rust-old/src` file is listed with its milestone. It carries impetus's scan for the open
cyrius 6.7.18 miscompile, where `s.f = g(t.e)` stores `t.e` when `g` returns a struct. That scan flagged three
stores in the SoA's row copy. They were correct in practice: the 60 round-trip rows pass. They are bound to locals
anyway, and the faster column copy replaced them since.

**Mutation checks** (each restored and re-verified after):
- **The corpus**, three rows edited together: a `speed` value moved 1 ulp, a passing `validate` row given
  `E:InvalidTimestep`, an arena scenario's final free-list start changed.
  - `validate.py` fails all three.
  - The replay fails the `validate` and arena rows. The 1-ulp `speed` row passes, as class B's ≤ 2 ulp allows,
    and shows in its histogram as `1:1`. Class B is the plan's arithmetic class (§5). Every bit of these rows is
    still pinned, by `validate.py`'s exact recomputation.
- **The SoA's column mapping**, two columns swapped: all 60 round-trip rows fail.
- **The surface parity script**: a renamed method, field and enum variant are each reported missing.

**The memcpy question (ledger L12)** is answered and closed. A struct assignment moves a 144-byte record in
11.7 ns, `memcpy` in 245 ns, so every particle move is an assignment. Details: benchmarks-rust-v-cyrius, M1 part 3.

## 8. M1a part 4: buoyancy, vortex, bridge, JSON (2026-11-11); M1 complete

Four `[lib]` modules after `src/common.cyr`:
- `src/buoyancy.cyr`: the two forces, `terminal_velocity`, `reynolds_number`, `PvFlowRegime` and `classify_flow`;
- `src/vortex.cyr`: vorticity, Lamb–Oseen, Rankine, enstrophy, the Kolmogorov scale;
- `src/bridge.cyr`: the ten pavan / ushma / goonj conversions;
- `src/json.cyr`: serde-compatible codecs for the six serde types of M1 (`UPSTREAM(L10)`). The Cyrius-only
  `PV_ERR_DESERIALIZE` is appended to the error codes as 10 (ADR 0002 #10).

Every function is Rust's in Rust's order. The four that call a transcendental use the correctly rounded primitives:
`lamb_oseen_velocity`, `kolmogorov_scale`, `temperature_to_water_viscosity` and `tke_to_broadband_noise_db`.

**The golden modules:**
- `buoyancy.tsv`: 1,716 rows;
- `vortex.tsv`: 2,293 rows;
- `bridge.tsv`: 3,964 rows;
- `json.tsv`: 302 documents.

`validate.py` recomputes all of them:
- **The arithmetic** in Python, in Rust's order, with an IEEE division (Python's `/` raises at 0). One finding:
  Rust's `f64` `Sum` folds from −0.0, so an empty enstrophy is −0, which the port matches.
- **The four class C functions**, transcribed with their primitives as parameters. On glibc (ctypes) every one of
  their 3,198 rows reproduces the oracle bit for bit. On MPFR they give the corrected oracle:
  - `vortex.cr.tsv`: 3 rows, 2 kolmogorov and 1 Lamb–Oseen. In the Lamb–Oseen row a 1-ulp exp misround grows to
    7 ulp through 1 − e^(−r²/4νt);
  - `bridge.cr.tsv`: 26 rows, 25 `tke_to_broadband_noise_db` (a log10 misround times 20 gives 1–2 ulp) and 1 water
    viscosity.
- **Every JSON document**, decoded again by Python's `json` (a correctly rounded float parse) under serde's derive
  semantics. Its verdict and values agree with serde_json's on all 302.

**JSON findings:**
- **The float parse.** serde_json's default float parse is best-effort, so the generator uses its `float_roundtrip`
  feature, as impetus's does; serde_json and serde join the generator's dependencies at the oracle's own versions.
- **Where serde_json and bayan agree:**
  - a struct reads from an array of exactly its fields;
  - unknown fields are ignored and a duplicated known field is refused;
  - `null`, a string or a boolean is refused for an f64, and so is `1e400`;
  - `1e-400` reads as 0;
  - the phase refuses `256`, `-1` and `1.0`;
  - a unit variant reads as `"Name"` or `{"Name":null}`, and the unit struct only as `null`.
- **Where bayan's policies differ (three documents),** each a row of its own that the harness leaves pending
  (ADR 0002 #10):
  - an integer above `i64::MAX` into an f64, and a duplicated unknown field: impetus's S3 and S4, already at bayan;
  - the integer `-0`, which serde_json reads as −0.0 and bayan's tree as the integer 0: filed at bayan with a repro
    (`bayan:docs/development/issues/2026-11-11-json-integer-negative-zero-loses-its-sign.md`).

**Result:** on the first replay every new row passes. All 7,973 function rows are bit-exact; the 29 corrected rows
are accepted (ADR 4), each equal to MPFR's correction. All 299 decodes match, and each round-trips through the port's
own writer. The replay is **7 modules, 0 failing | rows: 42,697 ok, 0 failed, 0 measured, 2,463 pending, 72 accepted
(ADR 4: 72)**:
- **pending:** the M4 cell key's 2,460 and the 3 bayan rows;
- **targets:** identical on x86_64 ELF, x86_64 PE and aarch64, warning-free, and under `--bounds`.

**Tests ported:**
- all 45 Rust tests of the three modules: `tests/buoyancy.tcyr`, `vortex.tcyr`, `bridge.tcyr`, 54 checks;
- `common.rs`'s two serde tests, completing its 49;
- the 11 tests of `rust-old/tests/integration.rs` that use only M1 modules: `tests/integration.tcyr`, which later
  milestones extend.

Surface parity covers error, common, buoyancy, vortex and bridge: 128 items.

**Mutation checks** (each restored and re-verified after):
- Lamb–Oseen on the exp polyfill: 16 rows FAILED.
- The phase reader taking 256: 1 JSON row FAILED.
- `null` read as 0 for an f64: 11 JSON rows FAILED, across all four struct types.
- `classify_flow`'s 2300 made inclusive: the 2 threshold rows FAILED.
- A corrected bridge row's last bit: that row FAILED, and `validate.py` fails.
- The particle writer putting mass in `fuel`: the round trip fails 82 rows. The decode alone wouldn't notice.

**M1 is complete:**
- **The library:** fourteen `[lib]` modules.
- **The corpus:** 45,232 golden rows across seven modules. Every one recomputed by `validate.py`, and every one
  gated:
  - all bit-exact or accepted against the corrected oracle;
  - the only pending rows are the M4 cell key's 2,460 and the 3 bayan rows.
- **The Rust tests:** all 100 unit tests of the M1 modules are ported (error 6, common 49, buoyancy 16, vortex
  12, bridge 17), and the 11 integration tests that use only them.

## 9. M2: shallow water (2026-11-11)

`src/shallow.cyr`, the first stateful solver, after `src/bridge.cyr` in `[lib]`. `PvShallowWater` mirrors
`ShallowWater`:
- **The record:** 328 bytes, its layout pinned by a test.
- **Public fields:** the five public Vec fields are `PvColumn`s, resizable as Rust's public Vecs are, and `step`
  refuses inconsistent lengths as `validate_state` does.
- **Scratch:** the five scratch buffers persist across steps, so a step allocates nothing once their sizes are set.
- **Shapes:** ADR 0002 #11 covers the columns, `PvShallowWater_clone` and `PV_ERR_OUT_OF_MEMORY` (appended as 11),
  where Rust aborts.

The code covers construction and validation, the four cell reads, the disturbance, the volumes, the CFL step,
sediment transport (`PvSedimentConfig`, `update_sediment`) and JSON (scratch skipped, as serde's `#[serde(skip)]`).
The step has its three paths:
- **the default forward-backward scheme:** the well-balanced pressure gradient (Audusse-style reconstruction where
  the bed varies), first-order upwind convection, then face-averaged continuity;
- **the finite-volume HLL path** (`use_riemann`), with the hydrostatic reconstruction and mirrored ghost states;
- **Peregrine's implicit dispersion,** by alternating-direction Thomas solves, with the Green–Naghdi face-averaged
  variant.

Manning friction, breaking dissipation, the damping `powf` and the closed walls apply on every path. Every formula is
Rust's in Rust's operation order, with each field's raw storage read into f64 locals in the hot loops.

**The golden module** `shallow.tsv` holds 888 rows:
- 746 for construction, the reads, the disturbance, the reductions, sediment and JSON;
- 142 `ShallowWater::step` rows in the new class S (port plan §5).

The step rows come in two kinds:
- **bit-level trajectories** of the non-chaotic scenes (a flat lake, a lake over a slope and bump on both schemes, a
  linear wave) from the initial state at 1, 2, 10, 60 and 300 steps;
- **single steps** from the oracle's own state at frames {0, 1, 2, 10, 60, 300}: 1D dam breaks onto wet and dry
  beds on both schemes, a 2D dam break on both, a wet/dry beach on both, a drop, a rough bed, a breaking front, the
  three dispersion variants, 24 random wet-and-dry states, and 14 refusals (dt, field lengths, nx, dx, gravity,
  dispersion, a divergence).

Grids are 8 × 8 (1D: 24 × 3), and notes are cut at 240 characters (the harness ignores them), so the file is 3.8 MB.

**The S class.** It compares each field under the path's bound, from the harness's `_par_c_bound` hook. For the
shallow-water step that bound is 0: the damping `powf` is correctly rounded (ADR 0004), and cbrt is exact on both
sides.

`validate.py` transcribes the whole step: both schemes, dispersion, friction, breaking, the walls, the refusals. On
glibc's pow it reproduces all 142 step rows bit for bit; on MPFR's it would write `shallow.cr.tsv`. It wrote none:
glibc's pow is correctly rounded on all 22 (damping, dt) pairs the rows use. The rest of the module is recomputed by
transcription too.

**Result:** every row passes on the first replay. The 746 part-1 rows are bit-exact, and the 142 step rows hold
38,946 field values at 0 ulp, the 300-step trajectories included. The replay is **8 modules, 0 failing | rows:
43,585 ok, 0 failed, 0 measured, 2,463 pending, 72 accepted (ADR 4: 72)**.

**Mutation checks** (each restored and re-verified after):
- the Riemann x-momentum taking the wrong face side: 23 step rows FAILED;
- the Green–Naghdi coupling using one cell's depth: 9 FAILED;
- friction's h·∛h made h·h: 28 FAILED;
- the transcription's breaking decay dropped: `validate.py` fails 40 rows;
- the walls' order swapped (columns before rows): no change, and correctly so. Each corner's height, vx and vy come
  out the same in either order, so it is an equivalent mutation, not a gap.

**Tests ported:**
- all 44 Rust tests of `shallow.rs`: `tests/shallow.tcyr`, 155 checks, plus a group pinning the record's
  328-byte layout that `_pv_sw_col`'s offsets rely on;
- the 3 shallow tests of `integration.rs`, so `tests/integration.tcyr` has 14 tests and 29 checks.

Each Rust `is_err()` became the exact `PV_ERR_*` code, a stronger check. `capacity()` became the scratch
columns' `cap`, and the data pointers are checked too. Surface parity now covers 165 items with `shallow`.

**Speed** (benchmarks-rust-v-cyrius, M2): the step runs at 7.2× Rust's at every grid size, a per-cell cost of
calls to small helpers that `#inline` declines (filed). The volume rewrite on raw loads made `total_volume` 3.1×
faster, with its rows unchanged.

## 10. M3 part a: the grid (2026-11-11)

`src/grid_solve.cyr` and `src/grid.cyr` go after `src/shallow.cyr` in `[lib]`. `PvFluidGrid` mirrors `FluidGrid`:
- **The record:** 312 bytes: nx, ny, dx and twelve columns at 24 + 24·k, pinned by a test.
- **Public fields:** vx, vy, pressure and density are `PvColumn`s, and `step` refuses inconsistent lengths.
- **Scratch:** the four scratch buffers persist across steps. `mg_levels` is a column of `*_PvMgLevel` records,
  each holding three columns. Three Cyrius-only buffers, `solve_a` … `solve_c`, hold what Rust's DST and CG solves
  allocate per call (ADR 0002 #12).
- **Config and walls:** `PvGridConfig` and `PvBoundaryCondition` (NoSlip 0, FreeSlip 1, Periodic 2) come with
  their JSON (`src/json.cyr`).

`grid.cyr` covers:
- construction and the reads (`cell_count`, `idx`, `velocity_at`, `speed_at`, `max_speed`, `cfl_dt`,
  `total_kinetic_energy`) and the public `PvFluidGrid_diffuse`;
- the step's whole pipeline:
  - semi-Lagrangian, periodic, MacCormack and BFECC advection, with the backtrace limiter;
  - Gauss–Seidel diffusion with the Smagorinsky viscosity;
  - buoyancy and vorticity confinement;
  - the four projections: CG, multigrid V-cycles, Gauss–Seidel on periodic walls, DST otherwise;
  - the ghost rules of each wall type and the density's zero-gradient ring.

`grid_solve.cyr` transcribes hisab 1.4.0's direct DST-I and IDST-I (L6) and its matrix-free linear CG (L5, with the
operator as a function pointer) into caller scratch, so nothing on the step path allocates.

Rust's `%` on f64 (the periodic sampler's wrap) is C's fmod, which ganita and the stdlib don't have. It is
`_pv_fmod` in `src/mathx.cyr`: hisab 3.4.1's exact integer-bit fmod, transcribed (ledger L14, requested at ganita).
The math corpus gains 810 `f64::rem` rows for it (every special pair, wide gaps, subnormals, the sampler's shapes),
all bit-exact.

**The golden module** `grid.tsv` holds 569 rows (4.5 MB):
- construction, the configs, the reads (`cfl_dt` takes eight (ν, cfl) pairs a row), `diffuse`;
- the Gauss–Seidel and multigrid oracle hooks on 10 shared systems (the multigrid rows give every level's
  residual, coarse right-hand side and coarse correction);
- JSON for the three types;
- 188 `FluidGrid::step` rows in class S:
  - bit-level trajectories of four still tanks (no-slip, free-slip, periodic, and ambient-density fluid);
  - single steps at frames {0, 1, 2, 10, 60, 300} of a buoyant plume under ten scheme and force variants, of a
    16 × 16 plume (two multigrid levels), of a vortex and of an inflow jet;
  - 24 random states on random configs;
  - 40 refusals: each config check, the state's lengths, nx, ny, dx, and both divergence checks.

Every row's input grid is fresh, built by `new` and then given the public fields, so the private buffers' contents
never reach a row, only their lengths (the scratch dump). Part a's step rows use the periodic and multigrid
projections. The DST and CG rows, and the step rows on those paths, come with part b and its corrected-oracle work:
the DST's sines and cosines are where glibc may misround.

`validate.py` transcribes the grid too: the samplers, all four advections, diffusion, Smagorinsky, the forces, the
walls, Gauss–Seidel and the recursive V-cycle, the step and its refusals, and the JSON readers. It recomputes all
569 rows. Two mutations of the transcription are caught: the restriction weight 0.25 → 0.2501 fails the multigrid
rows, and the BFECC compensation × 1.0000001 fails the step rows.

**Result:** every row passes on the first replay, bit-exact. The 188 step rows hold 60,421 field values at 0 ulp,
the 300-step trajectories included. The replay is **9 modules, 0 failing | rows: 44,964 ok, 0 failed, 0 measured,
2,463 pending, 72 accepted (ADR 4: 72)**. It is identical on x86_64 ELF, x86_64 PE and aarch64 and under `--bounds`,
and the oracle's `--check` passes with rayon on, so the parallel advection and projection loops give the serial bits.

**Mutation checks** (each restored and re-verified after):

| Mutation | Rows failed |
|---|---|
| a sampler weight `sx` → `1 − sx` | 94 step rows |
| free-slip's bottom tangential reflected instead of copied | 20 |
| the prolongation's half weight → a quarter | 32 multigrid rows (108 in all) |
| BFECC's compensation ½ → 0.4 | 23 |
| vorticity confinement's force along N instead of N × ω | 101 |
| Smagorinsky's 2·S₁₂² → S₁₂² | 23 |
| MacCormack's ring copy dropped | 9 |
| the multigrid coarsest size 6 → 7 | 4 multigrid rows |
| fmod's signed zero → +0 | 5 `f64::rem` rows |

One mutation changed nothing, correctly so. Dropping a periodic corner assignment is equivalent: the row copies run
before the column copies, so each corner already holds its diagonal cell when Rust's corner block runs. That block
is redundant in 1.3.0, not a gap in the rows.

**Tests ported:**
- 44 of the 49 Rust tests of `grid.rs`: `tests/grid.tcyr`, 164 checks. The rest wait for part c, where `MacGrid`,
  `KEpsilon` and `apply_ghost_fluid` arrive: the three MacGrid tests, the KEpsilon size-mismatch test, the
  ghost-fluid test, and the MacGrid half of `test_cfl_dt_finite_without_flow`. A Cyrius-only group pins the record
  layouts.
- The 5 grid tests of `integration.rs` and `serde_roundtrip_all_types`, whose four types now all have codecs:
  `tests/integration.tcyr` has 59 checks.

`test_pressure_solvers_agree` and `test_projection_all_solvers_dx_scaled` already show the DST and CG solves agreeing
with Gauss–Seidel to Rust's 1e-3 on the same systems, before part b's rows. Surface parity covers 227 items with
`grid`, the 24 part-c items listed as pending.

**Speed** (benchmarks-rust-v-cyrius, M3 part a):
- **The periodic step is 2.7–4.0× Rust** and diffuse 2.5–3.7×, the codegen's per-cell cost.
- **max_speed is 16–17×** after writing its max out, which made it 1.44× faster with the rows unchanged.
- **The DST steps are 15–231×,** because the correctly rounded sin takes 18.4 µs at fl(k·π/2), where the DST's
  arguments land whenever n + 1 has small factors. Appended to the L1 request. Part b's sine table per transform
  length keeps the bits and removes it.

## 11. M3 part b: the DST and CG projections (2026-11-12)

Part a ported every projection; part b puts the DST and CG ones under parity and adds the DST plans.

**The rows.** Part b's rows come after part a's and from their own seed (0x6_72b), so part a's 569 keep their values.
`grid.tsv` now holds 827 rows (8.8 MB):
- `oracle::pressure_solve_dst`, 21 rows in class C (bound 0):
  - one on each of part a's 10 shared systems;
  - five at the transform lengths where glibc misrounds a sine or a cosine. Over every DST argument of lengths 1–65,
    glibc 2.44 misrounds 157 of 95,810 calls (0.16%), none below length 12. The five cover 12, 18, 25, 36, 38 and
    60 (two eigenvalue cosines at 60), and part a's 32 × 32 system covers 30;
  - the edges: one-cell sides (sin(π/2), cos(π/2)) and empty interiors, which leave the guess.
- `oracle::pressure_solve_cg`, 49 rows in class B:
  - each shared system at 1, 10, 40 and 200 iterations;
  - part b's smaller systems;
  - a zero right-hand side, whose first residual is already under the 1e-14 floor;
  - an empty system, which hisab refuses (Diverged).
- 188 more `FluidGrid::step` rows, on the DST (the smoke default) and CG projections:
  - 300-step trajectories of still tanks (DST, free-slip DST, CG, periodic CG);
  - single steps at frames {0, 1, 2, 10, 60, 300} of the 10 × 10 plume under eight DST and seven CG variants (budgets
    1, 10 and 200; CG over multigrid, since CG comes first);
  - a 14 × 14 plume, whose transform length 12 misrounds;
  - a vortex and a jet;
  - 24 random states;
  - 20 × 27 and 14 × 40 random states;
  - the checks after the solve: an overflowing projection, and a NaN velocity.

**The corrected oracle.** `validate.py` transcribes hisab 1.4.0's `dst`, `idst` and `conjugate_gradient` and the
two solves. It recomputes every row with glibc's sin and cos, as the oracle calls them. It then writes
`tests/golden/grid.cr.tsv` with MPFR's, for every row that uses a length with a misrounded call, and only those can
change. The file has 20 rows: the 6 DST systems at those lengths and 14 steps (the 14 × 14 plume on two DST configs,
the 20 × 27 and 14 × 40 states). Each note lists the distinct calls glibc misrounds, between 1 and 5 a row.

**Result:**

| Rows | Outcome |
|---|---|
| CG, 49 | every row bit-exact against the oracle |
| DST, 21 | 15 bit-exact against the oracle, 6 bit-exact against the corrected oracle and none equal to glibc's |
| step, 376 | 362 bit-exact against the oracle, the other 14 against the corrected one: 141,390 field values at 0 ulp |

The replay is **9 modules, 0 failing | rows: 45,202 ok, 0 failed, 0 measured, 2,463 pending, 92 accepted
(ADR 4: 92)**. Before the corrected oracle existed, the same 20 rows failed and nothing else did.

**The DST plans.** hisab computes sin(((π·(i+1))·(k+1))/(n+1)) inside every sum, which makes 4n³ correctly rounded
sines per 2D solve. Each value depends only on (i, k, n), so the grid now keeps two Cyrius-only columns, `dst_x`
and `dst_y` (ADR 0002 #12, amended). Each holds its transform length's n² sines and n eigenvalue terms
2·cos((π·(k+1))/(n+1)) − 2. The first DST step builds them, and later steps keep them while the interior keeps its
size: a column n² + n words long is length n's plan. The values are the same, so the bits are the same.
- The trajectories reuse a plan for up to 300 steps, and every single-step row builds its own; both give the
  oracle's bits.
- `PvFluidGrid` is 360 bytes.
- `fg_dst_plan_is_kept_per_length` (`tests/grid.tcyr`, now 185 checks) checks that a plan is kept for its own
  length, rebuilt word for word for another, separate per side on a non-square grid, and copied by a clone.

**Mutation checks** (each restored after):

| Mutation | Caught by |
|---|---|
| the sine table read transposed (term (k, i) for (i, k)) | 24 rows |
| IDST's ×(2/(n+1)) as (×2)/(n+1) | 102 |
| one plan for both sides of a non-square grid | 21 |
| the DST's ring left unzeroed | 19 |
| CG's tolerance 1e-6 → 1e-7 | 4 |
| CG's stencil summed as (l + r) + (d + u) | 102 |
| CG's residual norm carried with an ulp's error | 98 |
| a plan kept when the column is longer than the length needs | the plan test (no row changes a grid's size) |

One mutation, (2·c) − 2 → 2·(c − 1) in the eigenvalue term, is equivalent. Multiplying by 2 is exact and commutes
with rounding, so both forms give the same bits for every c.

Three mutations of the Python transcription are caught as well:

| Mutation of `validate.py` | Rows failed |
|---|---|
| the DST argument multiplied in the other order | 25 |
| CG's tolerance | 4 |
| IDST's scale order | 103 |

**Speed** (benchmarks-rust-v-cyrius, M3 part b), against part a's sine-in-the-sum:

| Grid | Part a | Plans, first step | Plans, later steps |
|---|---|---|---|
| 15 × 15 | 28.5 ms | 760 µs | 190 µs |
| 30 × 30 | 15.9 ms | 1.10 ms | 965 µs |
| 63 × 63 | 719 ms | 8.6 ms | 5.64 ms |

Rust's criterion bench steps a fresh clone each time, so its twin builds the plans in every timed step. That step
is 6.0×, 1.03× and 0.85× Rust; a grid's later steps are 1.5×, 0.90× and 0.56×. The first 15 × 15 step still pays
25 sines on the correctly rounded slow path (L1). The plan form is appended to hisab's L6 request.

## 12. M3 part c: KEpsilon, the ghost fluid and MacGrid (2026-11-12)

The rest of `grid.rs`:
- **`src/k_epsilon.cyr`:** `PvKEpsilon`, a 120-byte record: k and ε columns, the three constants, two scratch
  columns. It has `new`, `turbulent_viscosity`, the source-term `step` (Rust swaps the k and ε Vecs with their
  scratch; the port swaps the column records), clone, drop and JSON.
- **`apply_ghost_fluid`,** in `src/grid.cyr`.
- **`src/mac_grid.cyr`:** `PvMacGrid`, 168 bytes: nx, ny, dx, the face and cell columns, two scratch columns. It has
  `new`, `velocity_at`, `max_speed`, `divergence_at`, `cfl_dt`, the step, clone, drop and JSON.
- **The step:** closed walls, face buoyancy, Neumann Gauss–Seidel with the mean pinned, and density advection
  through `FluidGrid`'s sampler.
- **ADR 0002 #13** records the Cyrius shapes:
  - the flags where Rust warns and returns;
  - (pointer, count) slices;
  - a `KEpsilon::new` that can fail;
  - every size product checked where Rust panics on overflow;
  - NaN for an out-of-range read.

**The rows** (`m_grid_c.rs`, from their own seed, after parts a and b): 293 of them, so `grid.tsv` has 1,120.

| Function | Rows | What they cover |
|---|---|---|
| `KEpsilon::new` | 7 | sizes 0 × 0 to 16 × 4 |
| `KEpsilon::turbulent_viscosity` | 12 | log-uniform k and ε (ε down past the 1e-20 floor), the specials, changed constants |
| `KEpsilon::step` (class S) | 46 | 50-step trajectories under shear, rotation and random flows; random models with the specials; non-square grids; a step that hits both 1e-10 floors; 17 refusals, among them Rust's two size-mismatch shapes (the 16 × 4 one steps) |
| `apply_ghost_fluid` | 31 | circles of several radii and cell sizes, a plane, random fields, zeros and specials in the level set, σ zero, NaN and ∞, 9 refusals |
| `MacGrid::new` | 28 | the sizes and every special cell size |
| the four MAC reads | 52 | random states with NaN, ∞ and −0 faces, the no-flow grid (`cfl_dt` f64::MAX), eight CFL factors a row |
| `MacGrid::step` (class S) | 89 | 300-step trajectories of an empty and a full tank; single steps at frames {0, 1, 2, 10, 60, 300} of a buoyant blob under four configs (one setting every field the MAC step ignores) and of a carried density; the projection on arbitrary faces at 400 sweeps; 24 random states; 24 refusals and divergences in Rust's order |
| `MacGrid::from_json`, `KEpsilon::from_json` | 28 | the oracle's documents, seq forms, missing, unknown, mistyped and null fields |

`validate.py` transcribes all three types and both decoders and recomputes every row: 48,050 rows, 0 failures.

**Result:** every row passes on the first replay, all at 0 ulp. The two step functions take class S bound 0, and
the replay is **9 modules, 0 failing | rows: 45,495 ok, 0 failed, 0 measured, 2,463 pending, 92 accepted
(ADR 4: 92)**.

**Mutation checks** (each restored after):

| Mutation | Rows failed |
|---|---|
| ν_t as C_μ·(k·k) | 22 |
| \|S\|²'s sum re-associated | 18 |
| ε/k without k's 1e-10 floor | 8 |
| the k ring left uncopied | 18 |
| the crossing test without the bottom neighbour | 15 |
| the MAC Neumann count wrong at the second row | 61 |
| the buoyancy product re-associated | 11 |
| the top wall left open | 25 |

Four mutations survive the gate:
- **Re-associating the ghost Laplacian** (φ_r − 2φ) + φ_l → φ_r − (2φ − φ_l) shows only as 3 values at 1–2 ulp
  (`max:2` in the histogram). That is inside class B's 2-ulp bound, so the row passes. Every other part c value is
  at 0 ulp, so the histogram exposes it, but the gate doesn't. Part a's and b's class B rows share this property.
- **The other three are equivalent:**
  - φ > 0 → φ ≥ 0: a cell with φ = 0 never counts as crossing;
  - the MAC mean summed from +0: Gauss–Seidel never leaves a −0 pressure, because each sum starts at +0;
  - the density swap written as a copy: it changes only the scratch buffer's contents, which no later read sees.

Three mutations of the Python transcription are caught (8, 11 and 15 rows).

**Tests:** the last five Rust tests of `grid.rs` and the MacGrid half of `test_cfl_dt_finite_without_flow`, so
`tests/grid.tcyr` has all 49, with 218 checks. The layout group pins `PvMacGrid` and `PvKEpsilon`. Surface parity
covers all of `grid`: 227 items ok, 0 pending.

**Speed** (benchmarks-rust-v-cyrius, M3 part c). Rust has no criterion bench for these types, so
`scripts/probes/rust/src/bin/bench_grid_c.rs` times the same bodies on the oracle:

| Bench | Cyrius / Rust |
|---|---|
| MAC step | 2.2–2.9× |
| ghost fluid | 7.0–7.4× |
| k-ε step | 13–14× |

The k-ε step got there by writing its three maxes out. Each is a max against a positive constant floor, so the bits
are the same, and the step became 1.21× faster in a pinned A/B. The rest of the k-ε gap is a loop of independent
cells that Rust vectorises and the Cyrius codegen runs scalar.

## 13. M3 part d: VOF and the phase field (2026-11-12)

Two more grid modules:
- **`src/vof.cyr`:** `PvVofField`, a 72-byte record: nx, ny, dx, the α column, the scratch column. It has `new`,
  the surface, full and empty queries (with and without a threshold), `fill_rect`, `fill_circle`, `total_volume`,
  `interface_normal`, `advect`, clone, drop and JSON.
- **The advect:** donor-cell face fluxes with closed walls, sub-cycled to Courant 0.5 at most (at most 1,024
  sub-steps, and none when the count is NaN).
- **`src/phase_field.cyr`:** `PvPhaseField`, 88 bytes: nx, ny, dx, the φ column, mobility and ε, the scratch column.
  It has `new`, `set_circle`, `set_rect`, `is_phase_b`, `is_interface`, `phase_b_area`, the Allen–Cahn `step` with
  first-order upwind advection and zero-gradient ghosts, clone, drop and JSON.
- **Shared:** the fills (`_pv_fill_shape`).
- **ADR 0002 #14:**
  - (pointer, count) velocities;
  - every size product checked where Rust panics;
  - the scratch grown before any change;
  - a cell outside a field reads NaN where Rust panics.

**The rows:** two new golden modules, 384 rows, so `modules.txt` lists 11.

| Module | Rows | What they cover |
|---|---|---|
| `vof.tsv` | 224 | `new` (sizes, every special dx); the six queries and `interface_normal` over every cell of 13 states (random α with the edge values 0, 1, 1e-6, 1 − 1e-6, −0.1, 1.2, NaN, ∞, and gradients just above and below the normal's 1e-20 floor), eight thresholds each; `total_volume`; eight rectangles and eight circles (inverted, degenerate, NaN, infinite); 63 advect rows (class S) |
| `phase_field.tsv` | 160 | `new`; seven circles and seven rectangles; the queries and `phase_b_area` over 10 states; 75 step rows (class S) |

The VOF advect rows include:
- 300-step trajectories of a block in a rotation;
- single steps at frames {0, 1, 2, 10, 60, 300} of a disc under uniform, vortex and random flows;
- 1, 6, 600 and 1,024 sub-steps;
- 16 random states;
- cell sizes of 0, NaN and −0.1 (advect doesn't check dx);
- faces of exactly zero velocity next to NaN and ∞;
- 11 refusals in Rust's order.

The phase-field step rows include:
- trajectories of a uniform field and a sharpening ramp;
- frames of a relaxing disc, a carried strip, a high-Péclet front with no mobility, a rotation and a random flow;
- 16 random states with random mobility and ε;
- a clamping step and a negative mobility;
- the cell sizes, u = 0 beside ±∞, and the 3 × 3 minimum;
- 12 refusals.

`validate.py` transcribes both modules and recomputes every row: 48,434 rows, 0 failures.

**Result:** every row passes on the first replay, all at 0 ulp. The replay is **11 modules, 0 failing | rows: 45,879
ok, 0 failed, 0 measured, 2,463 pending, 92 accepted (ADR 4: 92)**.

**Mutation checks:**

| Mutation | Rows failed |
|---|---|
| the flux as (u·α)·(dt/dx) | 34 |
| the net flux re-associated | 22 |
| the bottom wall at the second row | 32 |
| the Courant target 0.5 → 0.25 | 21 |
| a rectangle's right edge exclusive | 3 |
| the phase-field Laplacian re-associated | 15 |
| the ghost columns skipping the corners | 39 |

Four mutations first survived, because no row reached what they change:
- the VOF donor at u ≥ 0;
- the 1,024 sub-step cap;
- the normal's 1e-20 floor;
- the phase field's upwind choice at u = 0.

Each now fails one targeted row:
- an 8 × 8 still field with NaN and ∞ beside finite cells (0·∞ shows the donor choice);
- 600 sub-steps with α unsaturated;
- gradients of 1e-12 and 1e-21;
- u = 0 beside ±∞.

Three mutations survive, and none can be caught:
- **Two are equivalent:**
  - (φ·φ)·φ → φ·(φ·φ) multiplies the same two operands;
  - reading the current φ from the field instead of the scratch reads the same value, because that cell isn't
    written yet.
- **The third changes only a signed zero:** starting the no-velocity advection term at −0 instead of +0.
  Classes B, C and S measure ulp distance, where +0 and −0 are the same point (tests/ulp.cyr, shared with impetus
  and tanmatra), so it can't fail a row by design.

Three mutations of the Python transcription are caught (3, 15 and 40 rows).

**Tests:** all 8 Rust tests of `vof.rs` (`tests/vof.tcyr`, 27 checks; the cellular-flow test uses the correctly rounded
sin and cos within its own 1% bound) and all 11 of `phase_field.rs` (`tests/phase_field.tcyr`, 38 checks), each with a
layout group. Surface parity adds both modules: 258 items ok, 0 missing.

**Speed** (benchmarks-rust-v-cyrius, M3 part d). Rust has no criterion bench for these, so
`scripts/probes/rust/src/bin/bench_vof_pf.rs` times the same bodies on the oracle. Both run 8.1–8.9× Rust, after
two changes that keep the bits:
- The VOF sub-step's four flux calls and its clamp are written out, and each x face shared along a row is computed
  once. Rust computes each twice from the same four values, so the left cell's F_r is the right cell's F_l.
- The phase-field step's clamp is written out.

Each made its step 1.17–1.19× faster in a pinned A/B. The rest is a loop of independent cells that Rust vectorises.

## 14. M3 part e: the worker pool and the grid's parallel loops (2026-11-12)

**`src/par.cyr`** is the pool probe 6 chose (§4.6, D6). A `_PvPool` of T threads is the caller plus T − 1 persistent
workers, each parked on its own job channel. `_pv_par_for(pool, lo, hi, fp, ctx)` works like this:
- it splits [lo, hi) into T contiguous slices, [lo + (n·w)/T, lo + (n·(w+1))/T);
- it posts a job token to every worker, runs slice 0 itself, and waits for T − 1 answers;
- every index is written by exactly one thread, so the split changes no bit;
- a pool of 0 runs the range on the caller.

`_pv_pool_new(T)` gives no pool, and the caller then runs serially, in these cases:
- a count of 1 or less;
- a target where `THREADS_CONCURRENT` or `CHAN_BLOCKING` is 0 (agnos, cx);
- a failure to make a channel or a thread (the workers already started are stopped).

It loads crmath's tables first, because they fill lazily behind a plain flag. A pool's channels are not returned
when it stops, since the stdlib has no `chan_free`, but a pool starts once per thread-count change, not per step.

**The grid** takes Rust's three `parallel` loops onto it: `advect` and `advect_periodic` (each row its own slice of
dst) and `project_velocity` (each row its own slice of vx and vy). Each becomes a row kernel. MacCormack's and BFECC's
forward steps go through `advect`, as Rust's do, and their other loops stay serial, as Rust's are. `PvFluidGrid` gains
`threads` (1 by default) and `pool` after its columns, so it is now 376 bytes:
- `set_threads(n)` stops any pool and records the count;
- the first step after it starts the pool, or leaves the grid serial when it can't;
- a clone keeps the count and starts its own pool;
- drop stops it;
- ADR 0002 #4 is amended.

**Parity at every thread count.** `tests/parity.tcyr` runs each of the 376 `FluidGrid::step` rows at one thread,
which is what the golden outputs gate. It then reloads the row and runs it again at 2, 3, 4 and 8 threads, and
fails the row if any output word differs from the one-thread run.
- 3 splits the rows unevenly, and 8 leaves workers without rows on the 4-row grids.
- Result: **376 rows, 0 differing from one thread**, on x86_64 ELF, PE (wine) and aarch64 (qemu) and under `--bounds`.
- The replay is unchanged: 11 modules, 0 failing, 45,879 ok, 92 accepted.

**Tests:** `tests/par.tcyr`, 40 checks.
- No pool below 2 threads, and a pool of 0 runs the whole range on the caller.
- At 2, 3, 4, 8 and 16 threads, the range is covered exactly once on exactly T distinct thread ids (`gettid`).
- Ranges shorter than T, and empty ones, run correctly.
- 500 calls on one pool take no heap.
- The grid's count works through its lifecycle: `set_threads`, the lazy start, the clone, a count below 1, drop.
  One thread and four give the same bits.

**Mutation checks:**

| Mutation | Caught by |
|---|---|
| each slice skipping its first index | 342 step rows at T > 1, 12 pool checks |
| the caller skipping slice 0 | 339 step rows at T > 1, 7 pool checks |

Keeping the projection serial changes nothing, correctly: serial is always a valid schedule. What the pool buys is
speed, which is measured, not gated.

**Speed** (benchmarks-rust-v-cyrius, M3 part e; cores 8–15). The periodic advection alone, on one pool:

| Grid | T = 2 | T = 4 | T = 8 |
|---|---|---|---|
| 255 × 255 | 2.0× | 3.2–3.6× | 4.1× |
| 63 × 63 | 1.85× | 3.3× | 3.4× |

A whole step gains less: 1.07× at 8 threads on 63 × 63, 1.15× on 255 × 255. The two serial Gauss–Seidel solves
(diffusion, then pressure) dominate it, and the three pooled loops are about a fifth of a periodic step. Rust's own
`parallel` build gains nothing at its bench sizes either: grid_step_periodic 2.07 ms at 63 × 63, as serial, and
124 µs against 70.6 µs at 15 × 15.

## 15. M4 part a: the SPH kernels and equations of state (2026-11-12)

M4 splits into four parts:
- **a:** the kernels, the equations of state and the small helpers;
- **b:** the neighbour grid with f32 keys (ledger L4), the cell key, the lists and the z-order sort (L9);
- **c:** density, the forces, the brute-force `step`, the particle blocks, split and merge;
- **d:** the physics extensions, among them the MLS corrections (L7).

**`src/sph_kernels.cyr`:**
- **`_PvKernelCoeffs`:** `_pv_kc_init` and the poly6, spiky-gradient and viscosity-Laplacian evaluations. Also the two
  colour-field terms the solver uses, checked by Rust's `test_poly6_kernels_consistent` until M5's rows reach them.
- **The seven public kernels.** Each computes only its own coefficient, by `_pv_kc_init`'s expression; see Speed.
- **`equation_of_state` and `equation_of_state_tait`,** with the γ → 0 `ln` limit.
- **The non-Newtonian viscosity:** the first data-carrying enum, a kind enum plus a payload struct (§7.3,
  ADR 0002 #15).
- **`batch_poly6`, `batch_distance_squared`, `total_kinetic_energy` and `max_speed`.**

**Private constants are now generated.** `scripts/oracle/src/t_private.rs`, after impetus's, writes
`src/constants_private.cyr`.
- Each private `const` of a ported module is parsed from its Rust literal and written positionally:
  `_PV_SPH_MIN_DENSITY`, `_PV_SPH_CSF_NORMAL_THRESHOLD` and `_PV_SPH_CONVERGENCE_THRESHOLD` (part d), and
  `_PV_GRID_MIN_SIZE`, which replaces the grid's two hand-typed 4s.
- Any other private `const` in a ported module makes the generator fail unless it is excused (`FALLBACK`, a struct
  literal).
- `probe_tables.py` compiles each one against its literal: 39 values, 0 failures.

**The rows** (`m_sph.rs`, a new golden module `sph.tsv`): 804.

| Function | Rows | What they cover |
|---|---|---|
| the seven kernels | 22 radii each, 29 distances a row | the specials, negative h, 1e±40 (h⁶ and h⁹ over- and underflowing), r from −0.1h to 2.4h with 0, h and 2h exactly, NaN and ±∞ |
| `equation_of_state` | 120 | random and special (ρ, ρ₀, k) |
| `equation_of_state_tait` | 471 | γ on both sides of the 1e-12 limit (±1e-13, ±1e-12, 1.1e-12), the usual values, negatives, NaN, ∞; ρ₀ at, below and through the 1e-10 floor; 12 arguments where glibc 2.44 misrounds pow(x, 7), pow(x, 1.4) or ln(x), found by an MPFR search |
| the non-Newtonian models | 22, 19 strain rates each | every variant, zero, NaN and degenerate fields, rates across the 1e-10 floor, 6 rates where glibc misrounds pow(s, n − 1) |
| batch and totals | 37 | the batch functions and the totals over up to 100 particles with special velocities and masses |

`validate.py` transcribes all of part a. It recomputes the rows with glibc's `ln` and `pow`, and writes
`tests/golden/sph.cr.tsv` from MPFR's: 16 rows, the 12 Tait and 4 viscosity rows at the misrounding arguments.

**Result:**
- Every row passes on the first replay. The 16 corrected rows match MPFR's correctly rounded result, and none
  matches glibc's.
- The replay is **12 modules, 0 failing | rows: 46,667 ok, 0 failed, 0 measured, 2,463 pending, 108 accepted
  (ADR 4: 108)**.
- 10 of 11 port mutations fail rows. The eleventh, `max_speed`'s max written as `q > m`, is equivalent: m is never
  NaN, and a squared speed is never −0.
- 3 of 3 mutations of the transcription are caught.

**Tests:** 22 Rust tests in `tests/sph.tcyr` (26 checks). `test_degenerate_inputs_no_panic` waits for part d, whose
functions it also calls. Surface parity adds `sph`: 348 items, 72 of them pending, labelled by part (b, c, d, M5).

**Speed** (benchmarks-rust-v-cyrius, M4 part a):

| Bench | Cyrius / Rust |
|---|---|
| the classic kernels | 10–10.5× |
| the Wendland kernels | 8–9× |
| the Tait EOS (`pow`) | 52× (correctly rounded `pow`, L1) |
| the Tait EOS, γ → 0 (`ln`) | 14× |
| the power law (`pow`) | 27× (correctly rounded `pow`, L1) |
| `batch_poly6` over 1,000 | 54× (about 8 µs of its 18.6 µs is an 8 KB freelist block mapped and unmapped per call, the filed `fl_alloc` finding) |
| the totals over 1,000 | 9–11× |

Each public kernel computing only its own coefficient made the classic kernels 1.30–1.78× faster with the rows
unchanged.

## 16. M4 part b: the neighbour grid and the z-order sort (2026-11-12)

**`src/neighbor.cyr`:**
- **`_pv_msort`,** a stable bottom-up merge sort over index words with a comparator `cmp(ctx, a, b)` (ledger L9).
  Started in index order, it sorts on (key, index), a total order (ADR 0002 #3). Both users below share it.
- **The f32 cell key,** `_pv_nb_key`: floor(x · inv) on the f32 product, cast as a saturating i32. `_pv_sph_cell_key`
  is the oracle hook `sph_cell_key` on one axis.
- **`PvNeighborGrid`** (ledger L4), solver-owned, eight reused columns. hisab 1.4.0's `SpatialHash` keeps a
  `HashMap` of cells, each in insertion order, which is index order. The grid sorts the points on (cell key, index)
  instead, so each cell is one run in that same order, and keeps each cell's key in a column. Nothing is allocated
  once the columns have grown.
- **`_pv_nb_build`** is `SphSolver::build_neighbors`, exactly:
  - the grid is renewed with cell size h as f32 only when that differs from its own by more than f32::EPSILON;
    otherwise it is emptied and keeps its old 1/cell, while the probes still use 1/h;
  - each particle's 27 probes sit at (c + d) as f32 · h + h/2 in f32, z outermost, then y, then x. Each probe's
    cell comes from the grid's key, so a probe that lands in another cell lists that cell, and a cell probed twice
    is listed twice;
  - `c + d` wraps as an i32, as Rust's release builds do. The oracle's build has overflow checks and panics there,
    so no row reaches it (ADR 0002 #16).
- **`sort_by_zorder`** makes each Morton code once, sorts stably and applies the order in place.

**Two findings in the oracle's own behaviour, reproduced bit for bit:**
- **D3 is visible in the rows.** In the clouds at offsets 1e6 and 3e6 (h 0.05), 400 of 720 probe axes land in
  another cell than the one they aim at. In the kept-grid row at h = 1 + f32::EPSILON, every probe axis of the
  4.5e6 cluster does (36 of the row's 612). The lists the solver sums over at those coordinates are not the 27
  cells around the particle.
- **A grid within f32::EPSILON of h is kept.** It keys by its old 1/cell while each particle's own key uses 1/h,
  so a point on a cell edge can sit in a grid cell other than its own key's: at h = 1 + f32::EPSILON, x = 1 has
  key 0 and grid cell 1.

**The rows** (`m_sph.rs`, part b's seed 0x0_5f4b; a fresh `SphSolver` stepped once with `water_2d()` at h, its
lists read through the oracle dump):

| Function | Rows | What they cover |
|---|---|---|
| `SphSolver::neighbor_lists` | 30 | lattices on cell boundaries (h a power of two, so k·h is exact) and off them (h 0.05), plain and jittered, 2D and 3D; random clouds at offsets 0, −0.5, −7.25, 1e3, 1e5, 1e6 and 3e6; h = 1, 1 + EPS, 1 − EPS/2 (kept), 1 + 2·EPS (renewed), 1 + 1e-9; one particle, duplicates, a NaN coordinate and −0, points a hair either side of a boundary; the kept grid on an integer lattice and a cluster at 4.5e6 |
| `sort_by_zorder` | 20 | shuffled blocks at their own spacing (distinct codes) and at coarse cells (ties), negative and all-positive coordinates; random clouds with NaN, ±∞ and 1e12 (saturated codes); cell 0, −1, NaN and ∞; one particle; none |
| `oracle::sph_cell_key` (`math.tsv`) | 2,460 | pending since M0 |

Each z-order row records Rust's codes in its output order, then the particles in the stable order. The particles are
recovered by matching bits, so a row checks the multiset and the codes' sequence (ADR 0002 #3).

`validate.py` transcribes the build (a dict of cells, the f32 keys and probes, the kept grid) and the Morton codes. The
transcription first missed that `(c + d) as f32` rounds the integer above 2²⁴, and the 3e6 row caught it. Result:
49,288 rows, 0 failures.

**Result:**
- Every row passes: 30, 20 and 2,460. The replay is **12 modules, 0 failing | rows: 49,177 ok, 0 failed,
  0 measured, 3 pending, 108 accepted (ADR 4: 108)**. The 3 pending rows are bayan's JSON policies.
- **The mutation tests found two corpus gaps.** On the first version, 11 of 14 port mutations failed rows. Two
  survivors were gaps:
  - keying the z probes by 1/h instead of the grid's 1/cell;
  - folding min x from 0 instead of f64::MAX.

  The transcription's survivor, inserting by 1/h, was the first gap again. Four rows close both: the kept grid on
  integer points and at 4.5e6, at h = 1 + EPS and 1 − EPS/2, and two z-order blocks whose minima are both positive.
  None draws a random number, so no other row moved. All three mutants then fail rows.
- **The third survivor, dropping the i32 wrap, is unreachable.** Every key that would wrap is saturated, and the
  oracle's build panics on it.
- **Mutations of the second (faster) version:** 15 of 15 caught.
  - 13 fail rows: ties, the f64 key, the EPS rule, the probe's h/2, the row search's start, the scan's and the lower
    bound's `<`, the match and the cell order dropping an axis, a cell key from the wrong axis, the cycle's end,
    signed codes, the min fold.
  - A list that never grows crashes.
  - A cycle whose slots aren't marked never ends, and is killed at the test timeout.
- **Mutations of the transcription:** 6 of 6 caught after the new rows: the probe's integer rounding, the EPS rule's
  `>` as `>=`, x as the outer probe loop, the Morton axes swapped, ties reversed, inserting by 1/h.

**Tests** (`tests/sph.tcyr`, 45 checks):
- `test_zorder_negative_coordinates`.
- `test_sort_by_zorder` and `_preserves_data` wait for part c's `create_particle_block`.
- Three Cyrius-only groups:
  - `msort_is_stable`, at every length 0–70;
  - `nb_lists_cover_brute_force`: on 120 jittered points at h 0.1, every pair within h is listed and no index twice,
    and a rebuild gives the same lists;
  - `nb_layout_matches_column_offsets`: 224 bytes.

Surface parity: 348 items ok, 71 pending.

**Speed** (benchmarks-rust-v-cyrius, M4 part b), against a twin of `build_neighbors` on hisab 1.4.0's
`SpatialHash`:

| Bench | Cyrius / Rust |
|---|---|
| the 1,000-particle build, 2D (36 neighbours a particle) | 1.5× |
| the build, 3D (216 at the interior) | 1.9× |
| `sort_by_zorder` of 1,000 | 2.0× |

The first version ran the builds at 4.0× Rust and the sort at 7.6×. These changes, A/B'd with the rows unchanged, made the builds 2.2–2.6× faster
and the sort 3.8× faster:
- each axis's probe keys are made once per particle;
- each cell's run is appended in bulk;
- the cells are ordered z, y, x with their keys in a column, so one binary search and a short scan find a probe row's
  three cells;
- the sort permutes in place by assignment instead of byte-wise `memcpy`.

## 17. M4 part c: the brute-force sums, the free step, the block, split and merge (2026-11-12)

**`src/sph_fns.cyr`:**
- **`compute_density`, `pressure_force` and `viscosity_force`,** the O(n²) sums, in Rust's operation order: the
  symmetric pressure term, the 1e-20 ≤ r² ≤ h² window, and ρ floored at 1e-10 by Rust's NaN-ignoring max. The pair
  distance is written out, (dx² + dy²) + dz² as `distance_squared_to` gives it.
- **`sph_step`,** Rust's free `step` (renamed, ADR 0002 #17):
  - the config validated;
  - every density, then each pressure;
  - the accelerations from a snapshot;
  - symplectic Euler;
  - the walls, axis by axis, where a high corner acts only when it is above the low one. `water_2d`'s z has hi = lo,
    so only its low wall acts.

  Its densities and snapshot are one freelist block, freed before it returns. On `PV_ERR_DIVERGED` the state is
  Rust's: every density and pressure set, and the accelerations before the failing particle written.
- **`create_particle_block`, `split_particle` and `merge_particles`.** The split ring's cosines and sines are
  correctly rounded.

**The rows** (`m_sph.rs`, part c's seed 0x0_5f4c): 198.

| Function | Rows | What they cover |
|---|---|---|
| `compute_density`, `pressure_force`, `viscosity_force` | 20 each | each particle's value in index order, on random clouds (2D and 3D, offsets, μ of both signs), a lattice on the radius exactly (r² = h² inside), coincident pairs and pairs either side of r² = 1e-20, special densities, masses and pressures (at and below the floor, NaN, ±∞), a NaN position, and h of 0, −0.05, NaN, ∞, 1e40 and 1e-40 |
| `step` (class S) | 43 | a dam break whose interior density is ρ₀, its trajectory at 1, 2, 10, 60 and 300 steps, and one step from frames 1, 10 and 60; a 3D cube in a closed box; a splash through every wall, z off the plane; the Rust tests' single particles; n = 0; damping 0 and 1, μ 0 and negative, a box flat on two axes; each config error in its order, with particles and without; three divergences (a NaN position, an ∞ mass beside a neighbour, and an ∞ x velocity, whose x acceleration alone is non-finite) |
| `create_particle_block` | 36 | the Rust tests' shapes, quotients a hair under an integer (0.3/0.1, ceil 3), negative and zero sizes, every early return, ∞ × ∞ counts (`checked_mul`, empty) and ∞ × 0, NaN and huge origins, random blocks |
| `split_particle` (class C) | 29 | 0 to 64 children, special offsets and parents; and n = 19, 27, 39, 52 and 61 from the origin, child counts whose ring angles glibc misrounds (an MPFR search over n ≤ 300 found 89 such counts) |
| `merge_particles` | 30 | random pairs, totals of 0, NaN, ∞ and negative (the 1e-20 floor), a NaN position, an −∞ density |

`validate.py` transcribes part c from the Rust source, with glibc's sin and cos for the record and MPFR's for the
corrected oracle. `sph.cr.tsv` gains the 5 split rows: 21 rows, which the port matches, and none matches glibc's.
Result: 49,486 rows, 0 failures.

**Result:**
- **First replay.** Every density, force, step and block row passed. The 5 corrected split rows waited for
  `sph.cr.tsv`. All 30 merge rows failed, and every scalar field was right.
- **The merge failures are a cyrius miscompile.** `p.position = _pv_mass_weighted(a.position, …, b.position, …)`
  stored `b.position`: a field store of a struct-returning call with struct-field arguments stores an argument
  (cyrius `2026-11-09-field-store-of-struct-call-copies-a-field-argument.md`, open, filed by impetus). Binding each
  call to a local fixed all 30, and pravash's numbers are on the issue. Surface parity's `field_store_scan` flags the
  original line; it hadn't been run before that replay.
- **The replay is 12 modules, 0 failing | rows: 49,370 ok, 0 failed, 0 measured, 3 pending, 113 accepted (ADR 4:
  113)**.
- **The mutation tests found one corpus gap.** 22 of 25 port mutations failed rows on the first run (one more mutant
  was a control). Of the three survivors:
  - the position integrated with the old velocity was written equivalently; its real form fails 18 rows;
  - the divergence check needing all three components non-finite was a gap. Every diverging row was NaN on every
    axis. A row whose ∞ x velocity diverges only x (no random numbers drawn, so no other row moved) now fails it;
  - dropping the block's height-finiteness check changes no output. A non-finite height saturates to a count the port
    refuses, and Rust's check only spares it a 2⁶⁴-iteration loop.
- **Mutations of the transcription:** 6 of 6 caught (the forces' 1e-20 guard, the damping's sign, position before
  velocity, sin and cos swapped, merge's floor, the divergence test).

**Tests** (`tests/sph.tcyr`, 69 checks):
- the 10 part c Rust tests, in Rust's order: `test_compute_density_self`, the three block tests, the four step tests,
  `test_sort_by_zorder` and `_preserves_data`. That makes 33 of the 79;
- Cyrius-only (ADR 0002 #8):
  - `part_c_out_of_range_reads_nan`;
  - `part_c_refused_sizes_are_empty`: an overflowing block, no children, 2⁶² children;
  - `sph_step_diverged_leaves_rusts_state`.

`tests/integration.tcyr` adds the four free-function SPH tests (24 of 33, 66 checks). Surface parity: 348 items ok,
64 pending, `step` renamed `sph_step`.

**Speed** (benchmarks-rust-v-cyrius, M4 part c):

| Bench | Cyrius / Rust |
|---|---|
| the brute-force sums | 4.9–9.5× |
| the free step (9, 25, 64 particles) | 5.2–8.1× |
| `create_particle_block` of 1,000 | 24× (most of it a mapped 144 KB block) |
| `split_particle` into 8 | ≈ 970×: angles at fl(k·π/2), the L1 slow path (3.5 µs into 7) |
| `merge_particles` | 13× |

A template particle made the block 1.6× faster, and writing merge out made it 1.3× faster, with the rows unchanged.

## 18. M4 part d: the physics extensions (2026-11-12); M4 complete

**`src/sph_fns.cyr`, part d:**
- **The list functions:** `update_viscoelastic`, `update_heat`, `apply_delta_sph`, `apply_implicit_viscosity` and
  `compute_gradient_corrections`. They take caller CSR lists as (pointer, count) words, which `_pv_lists_valid`
  checks as Rust's private `neighbor_lists_valid` does. Words Rust's `u32` and `usize` can't hold are refused too.
  Each copies Rust's per-call snapshot into one freelist block, freed before it returns. Each returns 1, or 0 with
  nothing changed where Rust warns (invalid lists) or would abort (no memory).
- **The per-particle functions:** `update_phase_change` (the Stefan condition, with the fuel field as the latent-heat
  accumulator), `update_reaction`, `update_combustion`, `apply_contact_angle`, and the secondary particles.
- **A Rust trait as a Cyrius trait.** `ReactionProvider` is `trait PvReactionProvider`, taken as a `dyn` (Cyrius
  6.7.8's checked trait objects), and `impl PvReactionProvider for PvCombustionConfig` is Rust's impl. A call through
  a `dyn` returns one value, so `react`'s `(f64, f64)` is a `PvReaction` struct (a probe confirmed both: a
  multi-value return through a `dyn` is refused by name).
- **The 3×3 inverse (L7), `_pv_invert_3x3`,** transcribes the oracle's own function, hisab 1.4.0's `matrix_inverse`:
  - LU with partial pivoting; the first largest |pivot| wins, and a NaN pivot stays, since `val > NaN` never holds;
  - an **absolute** singular test, `max_val < 1e-12`. hisab's `EPSILON_F64` is 1e-12, not `f64::EPSILON`; the first
    draft assumed the latter, and checking the constant caught it before any row ran;
  - a dense forward and back substitution of each unit column.

**ganita's `mat_inv` is not the oracle's function.** Probe 8 (§4.8) found it bit-identical on 300 moment matrices,
and L7 was planned as a copy of it. Run against part d's 60 inverse rows, it agrees on the 48 ordinary ones and
differs on 12:
- a 1e-13 diagonal and a pivot of exactly 1e-12: ganita scales its tolerance by the matrix;
- 1e300 entries;
- all 9 matrices with a ±∞ or NaN entry: ganita refuses non-finite input.

So the stand-in follows hisab. The correction is on ganita's request: an `into` form with hisab's pivot test would
keep the rows, or pravash adopts ganita's behaviour as an ADR divergence when it converts.

**The rows** (`m_sph.rs`, part d's seed 0x0_5f4d): 318.

| Function | Rows | What they cover |
|---|---|---|
| the four list functions | 35, 28, 35, 32 | 2D and 3D clouds with index-order and shuffled lists (the sums follow the list order), a lattice, lists whose first offset is past 0, the four invalid shapes (too few offsets, a decreasing pair, a last offset past the indices, an index at n: each row shows the particles unchanged), n = 0, and each scalar's specials (0, −1, NaN, ∞, 1e±30); Rust's test shapes (simple shear, two particles, alternating velocities at 0–200 sweeps) |
| `compute_gradient_corrections` | 17 | 3D clouds and a 4³ cube (invertible moments), a 2D cloud (z has no moment: the identity), shuffled and invalid lists, special h, NaN densities and ∞ masses |
| `oracle::invert_3x3` | 60 | random matrices, pivoting, the identity, a permutation, rank 2, zero, a 1e-13 diagonal, pivots either side of 1e-12, 1e300 entries, ±∞ and NaN in each position |
| `apply_gradient_correction` | 21 | the identity, random matrices and vectors, special values |
| `update_phase_change` | 10 | every branch (below, above, a third phase), temperatures either side of T*, progress from −0.5 to 1.5 and NaN, the floors (L and c of 0, NaN, negative), phase_below = phase_above, and exact ties in binary-exact numbers |
| `update_reaction`, `update_combustion`, `CombustionConfig::react` | 4, 4, 30 | temperatures at and around ignition (NaN: the reaction runs through `react`'s `<=`, and combustion's `>` skips it), fuel 0, negative and NaN, a NaN ignition, negative rates |
| `apply_contact_angle` (class C) | 25 | walls, corners (the normal normalised), 2D (z degenerate) and 3D boxes, angles 0 to π and outside, three angles whose cosine glibc misrounds (an MPFR search over k·π/4096 found 7), the early returns, a NaN bound (Rust's `hi <= lo` keeps that axis's walls), an inverted box, particles exactly h from a wall |
| the secondary particles | 9, 8 | thresholds through 0, negative, NaN and ∞, NaN and ∞ accelerations; lifetimes at, below and around dt, NaN, every kind, dt 0, negative and NaN, a NaN gravity |

`validate.py` transcribes part d from the Rust source; result: 49,804 rows, 0 failures. `sph.cr.tsv` gains one
contact-angle row, 22 rows in all: of the three misrounded angles, only one's ulp survives the arithmetic after it.

**Result:**
- **First replay:** 315 of 316 rows passed. The 316th is that contact-angle row, which matches the corrected oracle.
  All 60 inverse rows passed, the non-finite ones included, which confirms the dense transcription.
- **The replay is 12 modules, 0 failing | rows: 49,687 ok, 0 failed, 0 measured, 3 pending, 114 accepted (ADR 4:
  114)**.
- **The mutation tests found two gaps,** both exact ties: no random row puts the energy toward the other phase exactly
  at the latent heat still needed, or a particle exactly h from a wall. Two rows in binary-exact numbers (T* 0, L 10,
  c 2; h 0.0625) close them, drawing no random numbers. Then:
  - 36 of 36 port mutations fail rows: the list checks, every term of the viscoelastic, heat, delta-SPH and implicit
    updates, the inverse's pivot test, ties, permutation and substitution, the identity default, the phase-change
    branches, the reaction floors and NaN semantics, the contact angle's walls, normalisation and NaN bounds, and the
    secondaries' gains, order and compaction;
  - 5 of 5 against the faster code (a transposed moment, a dropped one, a misplaced store, two field stores);
  - 6 of 6 against the transcription.

**Tests** (`tests/sph.tcyr`, 98 checks): 11 more Rust tests, in Rust's order, through a `brute_lists` helper that
mirrors Rust's test `brute_neighbors`:
- the four combustion tests;
- `test_apply_gradient_correction_identity`, `test_delta_sph_diffuses_toward_neighbor`,
  `test_viscoelastic_simple_shear_upper_convected`, `test_phase_change_accumulates_latent_heat`,
  `test_contact_angle_2d_and_neutral` and `test_implicit_viscosity_large_dt_is_stable`;
- `test_degenerate_inputs_no_panic`.

That makes 44 of the 79. The five that make their lists with `SphSolver::step` (`test_heat_uniform_no_change`,
`test_heat_conducts_from_hot_to_cold`, `test_viscoelastic_relaxes_to_identity`,
`test_viscoelastic_elastic_stress_finite`, `test_mls_correction_interior`) wait for M5, with the solver's own.

Surface parity: 349 items ok, 22 pending, all of them M5's solver types. It now checks Rust traits too: a
`pub trait X` must be a `trait PvX`, and an `impl X for T` an `impl PvX for PvT`.

**M4 is complete:** `sph.tsv` has 1,370 rows, every one passing, 22 of them against the corrected oracle.

## 19. M5 part a: the SPH solver's WCSPH step (2026-11-12)

M5 follows the plan of record: parts by step variant, then parallel.
- **a:** the solver and its WCSPH step;
- **b:** PCISPH;
- **c:** DFSPH;
- **d:** the multiphase step;
- **e:** the three parallel loops on `src/par.cyr`, then the D3 switch to f64 keys, with the rows remapped.

**`src/sph.cyr`:**
- **`PvSphSolver`** owns every buffer and reuses them across steps (port plan §7.5):
  - part b's `PvNeighborGrid`, embedded first. Cyrius has no `&s.f`, so the grid's address is the solver's;
  - the densities and the snapshot, as Rust's;
  - the accelerations, which Rust allocates per step;
  - Verlet's previous accelerations.
- **The step,** in Rust's order:
  - the neighbour lists (part b's f32 grid, so D3's probe drift is in the rows);
  - the densities over the lists, self included;
  - the pressures;
  - the accelerations from the snapshot: the free step's terms, plus the CSF surface tension from the colour field,
    whose Laplacian starts at the self term. The CSF vector is always added, +0 when it is off, as Rust's
    `+ DVec3::ZERO` is;
  - the divergence check;
  - symplectic Euler or velocity Verlet. Verlet restarts from zero when its stored count isn't n;
  - the walls.
- **Errors leave Rust's state.** Every buffer grows before a particle changes. Verlet's buffer is only reserved up
  front, so a step that fails leaves it as Rust's would.
- **`PvSphSolver_adaptive_dt`** is Rust's associated fn. `neighbor_lists()` returns a `PvNeighborLists`, the form part d
  of M4 takes.

**The rows** (`m_sph.rs`, M5 part a's seed 0x0_5f5a): 66.

| Function | Rows | What they cover |
|---|---|---|
| `SphSolver::step` (class S) | 49 | each a fresh solver run from the start to a frame, so Verlet's state never needs to be an input: a hydrostatic tank (frames 1, 2, 10, 60, 300), a dam break by Euler and by Verlet (the same frames), a droplet in zero gravity with surface tension 0.5 by both integrators (to 60), a 3D cube in a closed box with surface tension (to 60), a splash through every wall by both, a block at x ≈ 10⁶ where the f32 keys drift (D3), the Rust tests' single particles, n = 0, σ negative, NaN and ∞, μ 0 and negative, and the errors (dt 0, h whose `as f32` is ∞ or 0, an ∞ mass beside a neighbour, an ∞ x velocity that diverges one axis) |
| `SphSolver::adaptive_dt` | 17 | each term binding (speed, acceleration, viscosity), none binding, inverted bounds, every early return, NaN speeds and accelerations, n = 0 |

A NaN position doesn't diverge here, as it does in the brute-force step. It keys to cell 0, its list holds only
itself, and its acceleration stays gravity. The oracle shows exactly that, and so does the row.

`validate.py` transcribes the step on part b's grid transcription. Trajectory rows share their inputs except the
step count, so it continues each scene from its longest cached state. Result: 49,870 rows, 0 failures, in 26 s.

**Result:**
- Every row passed on the first replay, the 300-step trajectories by both integrators included. The replay is
  **12 modules, 0 failing | rows: 49,753 ok, 0 failed, 0 measured, 3 pending, 114 accepted (ADR 4: 114)**.
- **The mutation tests:** 19 of 22 port mutations failed rows on the first run. Of the three survivors:
  - dividing only on all three axes was the gap part c's free step had: no solver row diverged on one axis. The
    ∞-x-velocity row closes it (no random numbers drawn) and the mutant then fails it;
  - `r² ≤ h²` written `<` in the density is equivalent. At r² = h² the kernel is exactly 0, and a non-finite mass
    there diverges the step either way;
  - the CSF threshold's `≤` written `<` is an untested tie: no row's colour normal is exactly 0.1/h.
- **Mutations of the transcription:** 5 of 5 caught (the CSF sign and self term, the Verlet half-step, the density's
  self term, the adaptive-dt swap).

**Tests** (`tests/sph.tcyr`, 144 checks):
- 19 Rust tests in Rust's order: the nine `test_solver_*`, the five part d tests that make their lists with the solver
  (the heat, viscoelastic and MLS ones), the three adaptive-dt tests, the Verlet test and the interior surface-tension
  test. That makes 63 of the 79;
- two Cyrius-only groups: the solver's layout (336 bytes), and `reset_integrator`, whose next Verlet step must equal a
  fresh solver's, bit for bit.

`tests/integration.tcyr` adds the five solver integration tests (29 of 33, 74 checks). Surface parity: 349 items ok,
12 pending (parts b–d).

**Speed** (benchmarks-rust-v-cyrius, M5 part a):

| Bench | Cyrius / Rust |
|---|---|
| criterion `solver_step`, 9 to 256 particles (sparse) | 1.27–2.3× |
| criterion `solver_step_st`, 25 to 169 | 1.24–2.0× |
| a dense 625-particle block, by Euler or Verlet | 3.2× |
| the dense block with surface tension | 3.6× |

## 20. M5 part b: PCISPH (2026-11-12)

**`src/sph_pcisph.cyr`** is `step_pcisph`, a second `impl PvSphSolver` block. Rust's order throughout:
- the densities over the lists, then the pressures zeroed;
- from a snapshot, the non-pressure accelerations (gravity, viscosity, CSF), checked one particle at a time;
- the pressure factor δ = 1/(β·(|Σ∇W|² + Σ|∇W|²)), β = 2·((dt·m)/ρ₀)², from the prototype particle. That is the
  first one with the most neighbours, and among those the largest denominator. Rust's `powi(2)` is a square;
- at least one correction iteration. Each predicts velocities and positions from the snapshot and the current
  pressures, then measures the predicted compression max(ρ* − ρ₀, 0)/ρ₀; if that isn't under the threshold, it raises
  the pressures, clamped at 0. A last prediction uses the final pressures when the loop didn't converge;
- the predicted velocities, positions and pressures applied, each position checked. The accelerations are not written,
  as in Rust;
- the walls.

The four buffers (non-pressure accelerations, pressures, predicted positions and velocities) are solver columns,
grown with the others before any particle changes.

**The rows** (`m_sph.rs`, M5 part b's seed 0x0_5f5b): 37, class S.
- M5 part a's scenes: the tank and the dam break to 300 steps, the droplet with surface tension, the 3D cube, the
  splash;
- iteration counts 0 (one runs), 1, 5 and 20, and thresholds every prediction meets (1e9), none meets (0, negative),
  and NaN;
- the Rust tests' shapes, among them a single particle with no neighbour, whose δ is 0;
- the errors: the config, an h whose `as f32` is ∞, a non-pressure acceleration diverging on one axis, and a
  predicted position diverging (an ∞ velocity on a lone particle).

`validate.py` transcribes PCISPH from the Rust source on the grid transcription, its trajectories cached as part a's
are. Result: 49,907 rows, 0 failures, in 39 s.

**Result:**
- All 36 rows of the first set passed on the first replay. The replay is **12 modules, 0 failing | rows: 49,790 ok,
  0 failed, 0 measured, 3 pending, 114 accepted (ADR 4: 114)**.
- **The mutation tests:** 13 of 16 port mutations failed rows on the first run. Of the three survivors:
  - running no iteration at max_iterations = 0 was a gap. Every zero-iteration row converged in its one forced
    iteration, which then changes nothing. A zero-iteration row with a threshold of −1, which no iteration meets,
    closes it (no random numbers drawn), and the mutant then fails it;
  - preferring the later of two equal (count, denominator) prototypes is equivalent: δ is the same;
  - counting expansion as error is equivalent: the running maximum starts at 0, so a negative relative error never
    wins. Rust's inner `max(err, 0)` is redundant.
- **Mutations of the transcription:** 4 of 4 caught (β's factor 2, the prototype ties, the convergence test's `<`,
  the pressure clamp).

**Tests** (`tests/sph.tcyr`, 157 checks): the four PCISPH Rust tests in Rust's order (`test_pcisph_empty`,
`_single_particle_falls`, `_density_bounded`, `_zero_and_one_iteration`), so 68 of the 79 (recounted at part c: §21). The layout test pins all
eight solver columns, after a PCISPH step sizes the last four. Surface parity: 349 items ok, 11 pending (parts c–d).

**Speed** (benchmarks-rust-v-cyrius, M5 part b):

| Bench | Cyrius / Rust |
|---|---|
| criterion `pcisph_step`, 9 to 64 particles (sparse) | 1.34–1.75× |
| a dense 625-particle block, at most 5 iterations to 1% | 4.7× |
| the dense block, 5 iterations always (threshold 0) | 6.3× |

## 21. M5 part c: DFSPH (2026-11-12)

**`src/sph_dfsph.cyr`** is `step_dfsph`, a third `impl PvSphSolver` block. It follows Bender & Koschier's order, as
Rust does:
- the neighbour lists, then each density and stiffness factor α = max(ρ, 1e-10)/(|Σgⱼ|² + Σ|gⱼ|²) with
  gⱼ = (xᵢ − xⱼ)·((mⱼ·W')/r), or 0 when that denominator isn't above 1e-20;
- the non-pressure velocity v* from a snapshot (gravity, viscosity);
- the density solver: κᵢ = (max(ρ*ᵢ − ρ₀, 0)/dt²)·αᵢ with ρ* = ρ + dt·Dρ/Dt, stopping once Σerr/(n·ρ₀) is under the
  threshold, each round applied to the velocities;
- the advection, then a second neighbour build and α at the new positions;
- the divergence solver: κᵛᵢ = (max(Dρᵢ/Dt, 0)/dt)·αᵢ, its test on (Σ·dt)/(n·ρ₀);
- the diagnostic pressures by the linear equation of state, the divergence check, the walls.

Unlike PCISPH, `max_iterations` = 0 runs neither solver. α and κ are solver columns, grown before any particle
changes. The second build's lists may need more room after the advection, so an out-of-memory there is the one that
leaves a partial step (ADR 0002 #21).

**The rows** (`m_sph.rs`, M5 part c's seed 0x0_5f5c): 35, class S.
- part a's scenes (the tank and the dam break to 300 steps, the droplet, the cube, the splash);
- Rust's compression test: a block at spacing 0.012 against the rest density of one at 0.02, no gravity, 20 rounds to
  1e-3;
- iteration counts 0, 1, 3 and 20, and thresholds every round meets, none meets, NaN and negative;
- the Rust tests' shapes, among them one particle, whose α is 0;
- masses of 1e-15 against a rest density of 1e-12, so densities sit under α's 1e-10 floor while still compressed;
- the errors: the config, an h whose `as f32` is ∞, and two divergences.

DFSPH advects before its second build. A position that went ±∞ would saturate its key, and the build's `cx + 1` would
overflow, which panics in the oracle's overflow-checked build. The generator met that panic on its first run, so the
diverging rows use NaN velocities, whose key is 0.

`validate.py` transcribes DFSPH from the Rust source, its trajectories cached as before. Result: 49,942 rows,
0 failures, now in about 70 s.

**Result:**
- All 34 rows of the first set passed on the first replay. The replay is **12 modules, 0 failing | rows: 49,825 ok,
  0 failed, 0 measured, 3 pending, 114 accepted (ADR 4: 114)**.
- **The mutation tests found one gap.** On the first run 15 of 18 port mutations failed rows. Of the three survivors:
  - dropping α's density floor was a gap. Every row's densities are far above 1e-10 whenever the particles are
    compressed, the only time α is used. The faint-mass row closes it (no random numbers drawn) and the mutant then
    fails it;
  - letting α's gradient include the particle itself is equivalent: its own r² is 0, under the 1e-20 cut;
  - stopping the density solver on `≤` is equivalent: at the tie, Rust applies an all-zero κ, which changes no
    velocity.
- **Mutations of the transcription:** 5 of 5 caught.

**Tests** (`tests/sph.tcyr`, 171 checks): the four DFSPH Rust tests in Rust's order. Also `test_grid_cfl_dt` and
`test_shallow_cfl_dt`, two tests of the grid's and shallow water's `cfl_dt` that live in `sph.rs` and were missed when
M2 and M3 ported their modules. The layout test pins all ten solver columns.

**The test count, corrected.** A recount from the source shows 74 of the 79 ported, the last five the multiphase
step's. M4 part a ported 23 tests, not the 22 its record says, so each count since then was one short: 68 after M5
part b, not 67.

Surface parity: 349 items ok, 10 pending (part d).

**Speed** (benchmarks-rust-v-cyrius, M5 part c; criterion has no DFSPH group):

| Bench | Cyrius / Rust |
|---|---|
| the `solver_step` bench's 64-particle block (sparse), to 1% | 1.46× |
| a dense 625-particle block, to 1% | 3.8× |
| the dense block, every round (threshold 0) | 5.35× |

## 22. M5 part d: the multiphase step (2026-11-12)

**`src/sph_multiphase.cyr`** holds `PvPhaseProperties`, `PvMultiPhaseConfig` and `step_multiphase`, a fourth
`impl PvSphSolver` block. The configuration's `Vec<PhaseProperties>` is a column of three words a phase (ADR 0002
#22). The step is the WCSPH step with a material per particle, in Rust's order:
- the checks: the config, then n = 0 (Ok), then no phase (`PV_ERR_INVALID_PARAMETER`), then the f32 h;
- the neighbour lists, the densities, and each pressure by its own phase's rest density and gas constant;
- the forces from a snapshot:
  - gravity and the symmetric pressure term;
  - the viscosity at the mean of the two phases', (μᵢ + μⱼ)·½;
  - the free-surface colour field for σ > 0, its Laplacian starting from the self term;
  - the interface colour field over unlike neighbours for an interface tension > 0, with no self term;
  - the two CSF terms added in that order, `(a + a_s) + a_i`;
- the divergence check, symplectic Euler (no Verlet), the walls.

A phase outside the list reads phase 0, and an empty list reads Rust's local fallback (1000, 2000, 0), which the step
never reaches. Rust also warns once a step about an out-of-range phase; that waits for M7's logging.

**The rows** (`m_sph.rs`, M5 part d's seed 0x0_5f5d): 26, class S. The inputs carry the phases as a count and then
three values each.
- the plan's two-phase layer: water at its rest density below a lighter phase, at frames 1, 2, 10, 60 and 300;
- the free-surface tension alone, the interface tension alone, both, and an interface tension of 0, negative and NaN;
- three phases, with indices outside the list;
- the Rust tests' shapes: the single phase, water under air, two particles of two phases, and the two stacked blocks
  with and without interface tension;
- a random splash of three phases, and n = 0;
- the errors in Rust's order: the config, no phase (and the empty list with no particles, which is Ok), an h whose
  `as f32` is ∞ (with and without phases), and a divergence on one axis;
- masses of 1e-15, so every density sits under the 1e-10 floor (the mutation tests' gap, below).

`validate.py` transcribes the step from the Rust source, each scene's trajectory cached across its frames. Result:
49,968 rows, 0 failures, in about 65 s.

**Result:**
- All 25 rows of the first set passed on the first replay. The replay is **12 modules, 0 failing | rows: 49,851 ok,
  0 failed, 0 measured, 3 pending, 114 accepted (ADR 4: 114)**.
- **The mutation tests found one gap.** On the first run 18 of 22 port mutations failed rows. Of the four survivors:
  - dropping the 1e-10 density floor was a gap. No row's density came near it. The faint-mass row closes it (no random
    numbers drawn), and the mutant then fails it;
  - running the free-surface sums for σ ≥ 0, or the interface sums for any nonzero tension, is equivalent: the CSF
    helper returns zero for a σ that isn't positive, NaN included, as Rust's does;
  - a strict density cutoff (r² < h²) is equivalent: poly6 is exactly 0 at r² = h².
- **Mutations of the transcription:** 5 of 5 caught.

**Tests** (`tests/sph.tcyr`, 181 checks): the five multiphase Rust tests in Rust's order. Four follow
`test_solver_large_particle_count`, and the empty-phases test sits among the adaptive-dt tests. That makes all 79 of
`sph.rs`.

Surface parity: 349 items ok, 0 pending, 0 missing.

**Speed** (benchmarks-rust-v-cyrius, M5 part d; criterion has no multiphase group):

| Bench | Cyrius / Rust |
|---|---|
| the dense 625-particle block, one phase | 3.4–3.5× |
| two phases, interface tension 0.05 | 3.5× |
| two phases, and free-surface tension 0.072 | 3.8–3.9× |

## 23. M5 part e: the solver's thread count and the f64 keys (2026-11-12); M5 complete

**The thread count** (ADR 0002 #4, amended). `PvSphSolver` gains `threads` and `pool` after its columns, and
`s.set_threads(n)`, as `PvFluidGrid` has. The first step after it starts a `src/par.cyr` pool. Rust's three `parallel`
loops are `_pv_par_for` kernels, each particle written by one thread:
- `_pv_sph_density_k`: `step`'s densities. The multiphase step runs the same kernel serially, as Rust runs its own;
- `_pv_sph_accel_k`: `step`'s accelerations;
- `_pv_mp_accel_k`: `step_multiphase`'s accelerations.

Struct results return through caller frame storage (the Cyrius guide, "Returning a struct by value"), so the kernels'
`PvVec3` helpers need no locking. A kernel reads only the snapshot, the lists and its context, and writes only its own
words. PCISPH and DFSPH have no `parallel` loops in Rust and run serially at any count.

- **Parity:** every solver step row of all four steps (147) replays at 2, 3, 4 and 8 threads: **0 differ** from one
  thread. The replay shares `_par_threads_agree` with the grid's. The fix that came with it: an unusable solver row
  (-1) fell through `_par_sph_d` and was read as pending.
- **Tests** (`tests/par.tcyr`, 50 checks): the lifecycle (the pool starts on the first step; 20 more steps at 4 threads
  take no heap; `set_threads(0)` is 1 and stops it; drop stops it and keeps the count), and Euler, Verlet and
  multiphase at 2, 3, 4, 8 and 16 threads against one thread's bits.
- **Mutations:** routing the density, a WCSPH acceleration or a multiphase acceleration through a shared global (a
  race at T > 1) is caught by both the replay (18–39 of 147 rows differ) and the test. A pool that never starts is
  caught by the lifecycle test; the replay can't see it.

**The f64 keys** (port plan D3, ADR 0002 #2, amended). `src/neighbor.cyr`:
- each point's cell is floor(x · fl(1/h)) as a saturating i64 (NaN 0), hisab 3.4.1's `_sh_cell_coord`, with
  `_pv_as_i64` (`src/mathx.cyr`);
- a particle's list is the 27 cells at its own key ± 1 per axis in integers, (dz, dy, dx) order, each cell's points in
  index order: per (dz, dy) row, one binary search to x key c − 1 and a scan to c + 1. An offset past i64's range has
  no cell;
- the cell size is h at every build. The f32 grid kept its old size within f32::EPSILON of h while keying its probes
  by the new h (§5's branch point), and that band is gone with the probes;
- the f32 position column is gone: the grid is 200 bytes, the solver 472. The steps keep Rust's `h as f32` check.

**The rows mapped by name.** `validate.py` transcribes the f64 grid from its definition (`nb_build64`: a dict of cells
and integer offsets) and recomputes every list-dependent row on it. Each row whose outputs change goes to
`tests/golden/<module>.d3.tsv` with the f64 outputs; the harness loads it with the `.cr.tsv` rows, tagged ADR 2:

| Function | Rows | Changed (`.d3.tsv`) | Why |
|---|---|---|---|
| `oracle::sph_cell_key` (math) | 2,460 | 558 | the f32 product rounds across a cell boundary, or the key passes i32's range |
| `SphSolver::neighbor_lists` | 30 | 30 | the dump has no f32 positions; and the lists of the clouds at offsets up to 3e6 |
| `SphSolver::step` | 49 | 6 | the block at x ≈ 10⁶ (frames 1 and 10), and the two droplets at frames 10 and 60 |
| `SphSolver::step_pcisph` | 37 | 0 | |
| `SphSolver::step_dfsph` | 35 | 1 | a 70-particle scene at frame 2 |
| `SphSolver::step_multiphase` | 26 | 1 | water under air (Rust's test shape), 10 steps |

Where a particle sits within an f32 rounding of a cell boundary (the oracle rounds the position to f32 and then the
product x · 20 to f32; at h 0.05, 1/h as f32 is exactly 20), the two grids put it in different cells. Its list then
holds another set of candidates or another order, so the sums change bits. Both grids list every particle within h.

**Result:**
- validate.py's transcription changed exactly the rows the port failed on (558 and 38). On replay **all 596 pass
  against it bit for bit: 12 modules, 0 failing | rows: 49,255 ok, 0 failed, 0 measured, 3 pending, 710 accepted
  (ADR 4: 114, ADR 2: 596)**. The other rows keep the oracle's outputs, class A or S as before. The switch reordered
  no sum, so no row needed class D (§5).
- **Mutations of the grid:** 11 of 11 caught. Seven fail rows (floor → ceil, an i32 key, the x run one short at either
  end, a 1/h in f32, y outermost, a cell size other than h). The four overflow guards (x at i64::MAX, x at i64::MIN,
  y, z) fail only `tests/sph.tcyr`: no row can hold a ±∞ position, because the oracle's overflow-checked build panics
  on `cx + 1`.
- **Mutations of the transcription:** 5 of 5 caught, each by a `.d3.tsv` that no longer matches (an f32 1/h in the
  key, x / h in the grid, x outermost, the multiphase rows not recomputed, the dump's keys zeroed).

**Tests** (`tests/sph.tcyr`, 190 checks): two Cyrius-only groups. A jittered lattice at x = 10⁶ and 10¹², where the
f32 probes drifted, lists every particle within h exactly once. Keys saturated at ±∞ on each axis never list each
other, and a NaN coordinate is cell 0. The Rust tests pass unchanged; `test_solver_grid_clear_reuse`'s bound (the cell
size within f32::EPSILON of h as f32) holds for a cell size of h.

**Speed** (benchmarks-rust-v-cyrius, M5 part e):

| Bench | Cyrius / Rust |
|---|---|
| the 1,000-particle neighbour build, 2D (f64 keys; f32: 1.5×) | 0.93× |
| the 1,000-particle neighbour build, 3D (f32: 1.9×) | 1.25–1.34× |
| the dense 625-particle step, σ 0.072, 1 / 8 threads | 2.8–3.0× / 1.2–1.45× |
| the 2,500-particle step, 1 / 8 threads | 5.2× / 2.7–2.9× |
| the two-phase step, 1 / 8 threads | 2.8–3.0× / 1.8–1.9× |

## 24. M6 part a: the rigid bodies of `coupling` (2026-11-12)

**`src/coupling.cyr`** ports the rigid-body half of `coupling.rs`:
- **The shapes.** `BodyShape` is the kind enum `PvBodyShape` and the payload struct `PvShape` (port plan §7.3, as
  ADR 0002 #15). Its signed distance, volume, bounding radius and projected area are free functions by the enum-method
  rule.
- **The bodies.** `PvRigidBody` holds Rust's fields in their order, and has `signed_distance` and `surface_normal`
  (the central difference at ε = 1e-6, with Rust's two fallbacks).
- **The free functions:** the two-way coupling, integration with and without added mass, both drags and the level set.

Every formula is Rust's in its operation order. That covers glam's dot products, the component-wise vector
quotients (`sum / count`, `v_rel / |v_rel|`) and Rust's NaN-ignoring min and max.

The free functions take (pointer, count) slices (ADR 0002 #23). One restructure is bit-neutral (the A/B below):
- **A per-body context:** every distance in the hot loops reads the shape and centre from it, a two-argument call.
- **Bodies outside particles:** the coupling runs its bodies in the outer loop. Each particle's acceleration still
  sums the bodies in order, and each body's force sums the particles in order, and no pair reads what another
  writes.

**The indexed drag's hash** (ADR 0002 #23). Rust's `drag_from_particles_indexed` takes a hisab 1.4.0 `SpatialHash`
that the caller fills with each particle's f32 position. Its `query_radius` walks the (2⌈r/cell⌉ + 1)³ cells about the
body centre's cell, x outermost, each cell's items in insertion order.

The port takes the caller's **cells-only `PvNeighborGrid`**, the L4 stand-in, built by `PvNeighborGrid_build`:
- **The build:** M4 part b's keys and stable sort on the f64 keys of D3, then the cells sorted again into (x, y, z)
  order in an eighth column, `xorder`. `_pv_nb_build` and the new build share `_pv_nb_cells`.
- **The walk** (`_pv_nb_block`) runs through `xorder` from the block's first cell. A binary search skips each row,
  column or z run outside the block, so it visits Rust's cells in Rust's order and costs the occupied cells, never
  the block's volume.
- **The cell-size rule** is hisab 3.4.1's `spatial_hash_new`: finite and > 0 with 1/cell finite.

Rust's walk has no such bound. An infinite radius is i32::MAX cells a side and never returns, and a block past i32's
range panics in the oracle's overflow-checked build. So no row has a radius that is ±∞ as f32; `tests/coupling.tcyr`
covers the infinite radius instead.

Hisab 3.4.1's own `spatial_hash_query_radius` wasn't usable as the counterpart, for the two reasons below. Both are
now appended to L4's request, which the conversion needs answered:
- **The per-call allocation:** it allocates its result vector, L4's open request.
- **The order:** it switches to insertion order whenever the block holds more cells than the hash has entries
  (`full_scan`), which would reorder the drag's sum.

**The rows** (`m_coupling.rs`, seed 0x60_c0_0a): 4,798, class B, with `RigidBody::new` and the two decoders class A.
- the shapes over the specials, degenerate and negative sizes, points at the centre, on faces, edges and corners, and
  random ones about three centres; volumes and bounding radii over sizes k/7 and (k/3, k/7, k/11) (no random numbers
  drawn);
- the bodies' normals at the centre (the direction fallback), a box's centre (zero differences) and near the surface;
- the coupling on the Rust tests' scenes, the benchmark's block, approaching and receding particles, massless, NaN and ∞
  masses, negative damping, the specials as h, k and c, and seeded clouds about one to four bodies in 2D and 3D;
- integration over odd masses, the specials as dt, ρ and C_a, and each `AddedMassCoefficient`;
- both drags on the Rust tests' scenes, 40 seeded clouds at five cell sizes (and half slices for the indexed form), the
  specials, a NaN particle and a NaN body, bodies at 10⁴, 10⁶ and 3·10⁶, and a 9 × 9 × 9 lattice taller and wider than
  the block (no random numbers drawn);
- the level set on the Rust tests' grids, every refusal (a mismatch either way, an overflowing product, sizes a usize
  holds but an i64 doesn't), the specials, NaN and ∞ particles, and seeded clouds;
- the flow scene, a stream at +x past a sphere and a box: the coupling, both integrations and both drags on frames 0,
  1, 5, 20 and 40;
- 103 `BodyShape` and 30 `RigidBody` documents, serde_json's own and hand-written. The hand-written ones cover the
  struct variant as an object and as an array, the bare `"Sphere"`, null, missing, duplicated and unknown fields,
  arrays of the wrong length, two keys and an unknown variant.

`validate.py` transcribes every function from the Rust source:
- the indexed drag on hisab 1.4.0's f32 hash (`cp_query32`), and again on the port's f64 grid (`cp_query64`, from its
  definition: the block's occupied keys in sorted order) for `coupling.d3.tsv`;
- the decoders under serde's externally tagged struct-variant semantics.

It covers 54,766 rows with 0 failures, in about 85 s.

**Result:**
- **On the first replay** every row but one was bit-exact. The exception was an indexed drag about a body at 10⁶ with
  cells of 0.1. There the f32 positions are 0.0625 apart, so the oracle's cells order the particles differently and its
  sum is 3 ulp away. validate.py's f64
  transcription changes exactly that row, `coupling.d3.tsv` holds it, and it passes against it.
- **The full replay:** 13 modules, 0 failing | rows: 54,052 ok, 0 failed, 0 measured, 3 pending, 711 accepted
  (ADR 4: 114, ADR 2: 597). Every coupling value is at 0 ulp.
- **The mutation tests found two gaps.** On the first run 27 of 31 port mutations failed rows. A 32nd, whose pattern
  matched twice, was replaced by a misnamed `Box` key, which fails 47 rows. Of the four survivors:
  - a walk that skips the rest of an x column when z leaves the block was a gap. No cloud had cells beyond the block
    in z, and only `tests/coupling.tcyr` caught it. The 9 × 9 × 9 lattice rows close it: 3 rows now fail;
  - a level set accepted when nx·ny < len was a gap. A row with a longer buffer closes it;
  - skipping a zero-cell block (`cr >= 0` → `cr > 0`) is equivalent. ⌈r·fl(1/cell)⌉ = 0 means r ≤ 0 or NaN, and then
    no particle can have |sd| < sr;
  - computing the sphere volume as c·((r·r)·r) instead of Rust's ((c·r)·r)·r moves 25 of the new volume rows by 1 ulp.
    That is inside class B's 2-ulp bound, so it shows in the histogram (`max:1`) but passes, as M3 part c's ghost
    Laplacian did. validate.py's exact recomputation still pins every bit of the rows.
- **Mutations of the transcription:** 8 of 8 caught (a sphere's + r; the f32 query z outermost; the f64 grid in z-major
  order, caught by `coupling.d3.tsv`; no force reset; the level set's far value; a struct variant taking an array of
  any length; the drag area of the unnormalised velocity; the coupling rows not recomputed).

**Tests** (`tests/coupling.tcyr`, 55 checks):
- the 22 Rust tests of the shapes, the bodies, the coupling, added mass, drag and the level set, in Rust's order (the
  FLIP solver's 11 come with part b). `test_indexed_drag_matches_bruteforce_for_large_body` builds the grid where Rust
  builds its hash;
- five Cyrius-only groups: the record layouts (the shape at byte 56); the cell-size rule (0, −1, NaN, ∞ and a
  subnormal refused, the grid left empty); the block walk against a brute-force enumeration on 40 blocks about a 3D
  lattice with gaps; the indexed drag against the brute force on seeded clouds at the origin, 10⁶ and 10¹² and with an
  infinite radius; the level set's refusals.

`tests/sph.tcyr` pins the grid's new layout (224 bytes, eight columns; the solver 496): 192 checks.
`tests/integration.tcyr` adds the two coupling integration tests (31 of 33, 77 checks). Rust's
`coupling_sphere_falls_through_fluid` couples `&mut [body.clone()]`, a temporary, so its body falls under gravity
alone while the particles feel it; the port couples a copy too.

Surface parity: 386 items ok (the FLIP solver's 9 pending part b), 0 missing.

**Speed** (benchmarks-rust-v-cyrius, M6 part a):

| Bench | Cyrius / Rust |
|---|---|
| criterion `coupling_sph_bodies/625_particles` | 5.5–5.8× |
| criterion `drag_from_particles/625_particles` | 7.9× |
| criterion `particle_level_set`, the three sizes | 4.2–4.3× |
| the indexed drag, 625 / 10,000 particles (a twin) | 0.70–0.72× / 2.2–2.3× |
| the grid build against hisab's hash, 625 / 10,000 | 3.4× / 5.1× |

## 25. M6 part b: the FLIP solver (2026-11-12)

**`src/flip.cyr`** ports `FlipSolver` (ADR 0002 #24):
- **The record.** `PvFlipSolver` has the public `flip_ratio`, `use_apic` and `narrow_band_cells` first. The last is
  reserved, read nowhere, as in Rust. The seven grid fields and the APIC matrices follow as solver-owned columns,
  the matrices four words a particle.
- **The constructor.** `PvFlipSolver_new` refuses what Rust refuses, in Rust's order: below 4 cells, a bad dx, nx·ny
  past a usize. A size Rust's `vec!` would abort on is `PV_ERR_OUT_OF_MEMORY`. The blend is clamped to [0, 1], or
  0.95 when it isn't finite.
- **The step,** in Rust's order:
  - the checks (dt, a finite blend), then the APIC resize to the particle count;
  - P2G: the mass-weighted splat of v + C·(xᵢ − xₚ), with C zero unless APIC, then each node divided by a weight
    above 1e-20;
  - the pre-solve copy, then gravity·dt on every node;
  - the projection: the divergence, at most 40 warm-started Gauss–Seidel sweeps that stop when no cell moves by 1e-6,
    and the gradient;
  - the finite check (`PV_ERR_DIVERGED` leaves the grid as computed and the particles untouched);
  - G2P: APIC's affine transfer, or the FLIP/PIC blend with the blend re-clamped;
  - advection, z included, then the walls.

Each cell lookup is floor(g).max(0).min(n − 2) and (g − x0).clamp(0, 1). The splat's weights are products of the
fractions times the mass; the sampler's are the value times one fraction, then the other. Rust's fn-local
`PRESSURE_ITERATIONS` is generated (`t_private.rs` now scans `coupling.rs`).

**The rows** (`m_coupling.rs` part b, seed 0x60_c0_0b, after part a's rows, which stay byte for byte): 245.
- **`FlipSolver::new`** (35, class A): the Rust tests' sizes; every refusal in order (2 and 3 cells, dx 0, NaN and the
  specials, usize::MAX × 4); the blend's clamp and default over the specials and 1.5, −0.5, 2, −0.0. Each row gives
  the sizes of the eight vectors.
- **`FlipSolver::step`** (197, class S, bound 0): trajectories of one fresh solver, each row giving the particles and
  the private state (`oracle_dump`).
  - The scenes:
    - the Rust tests' (the empty step, a moving particle, gravity under PIC, both blends, the wall, the radial
      divergence by PIC and APIC, the NaN blend, the two integration tests);
    - the plan's sloshing tank (a block moving right in the left of a 16 × 16 tank) by PIC, FLIP 0.95, FLIP 1.0 and
      APIC, and its falling block by FLIP and APIC, each at frames 1, 2, 10, 60 and 300;
    - a pool at rest (one sweep converges) and under gravity (all 40 run).
  - The specials:
    - as dt, as the blend, and as each gravity component;
    - in each of a particle's seven slots among four others;
    - particles past every wall, and in 3D;
    - a corner particle whose vy alone overflows (the mutation tests' gap, below).
  - 40 seeded scenes (grid shapes 4–19 a side, cell sizes, blends past [0, 1], APIC, gravities, time steps, clouds in
    2D and 3D, some outside the walls).
- **`FlipSolver::scenario`** (13, class S): segments that change the blend, APIC and the particle count. They cover
  the matrices truncated, grown, kept across a FLIP segment, every particle dropped and new ones added, and a NaN
  blend in a later segment; eight are seeded.

`validate.py` transcribes the solver on a dict, each trajectory cached across its frames. With the scenarios'
segments it covers 55,011 rows with 0 failures, in about 85 s.

**Result:**
- **On the first replay** all 244 rows of the first set were bit-exact, the 300-step trajectories included. With the
  gap's row: 13 modules, 0 failing | rows: 54,297 ok, 0 failed, 0 measured, 3 pending, 711 accepted (ADR 4: 114,
  ADR 2: 597).
- **The mutation tests found one gap.** On the first run 25 of 27 port mutations failed rows. Of the two survivors:
  - a divergence check that reads only vx was a gap. In every row a non-finite vy also reached vx through the
    pressure. The corner row closes it: a particle on node (0, 0) moving at f64::MAX in y, under gravity f64::MAX at
    dt 1, makes only that node's vy ∞. No interior divergence reads that node, so vx stays finite. The mutant then
    fails it;
  - a G2P that skips its own APIC resize fails only `tests/coupling.tcyr`, which calls it directly (Rust's
    `test_apic_c_exact_for_rigid_rotation`). Through `step` the resize has already happened, so no row can see it.
- **The optimisations' code:** after the restructure below, three mutations rerun on it fail rows (the fraction's
  upper clamp dropped, the finite check on vx only, the sampler's fractions swapped).
- **Mutations of the transcription:** 7 of 7 caught (no clamp, a stale C without APIC, no re-clamp, the sweeps' stop
  at 1e-5, z frozen, the default blend 0.9, a scenario that doesn't truncate).

**Tests** (`tests/coupling.tcyr`, 106 checks):
- the 11 FLIP tests in Rust's order, which makes all 33 of `coupling.rs`. The three that read or set private state
  read the columns and call `_pv_flip_p2g` and `_pv_flip_g2p`;
- four Cyrius-only groups:
  - the layout (240 bytes), the refusals in Rust's order and the out-of-memory size;
  - APIC's resize to six, four and six particles, and a FLIP step leaving the matrices alone;
  - a NaN velocity's divergence leaving the particles as they were, and four refused time steps;
  - 30 steps of a block taking no heap.

`tests/integration.tcyr` adds the two FLIP integration tests: all 33, 79 checks. Surface parity: 386 items ok,
0 pending, 0 missing.

**Speed** (benchmarks-rust-v-cyrius, M6 part b):

| Bench | Cyrius / Rust |
|---|---|
| criterion `flip_step/25_particles` | 4.1–4.2× |
| criterion `flip_step/100_particles` | 3.7–3.8× |
| criterion `flip_step/225_particles` | 3.5× |

The restructure, each step bit-neutral (an A/B, pinned):
- **One cell lookup per particle:** G2P's four samples share it, where Rust's `sample_field` recomputes it per field.
- **Written out:** the lookup's max and min, the sweeps' running maximum and the finite check. Cyrius declines
  `#inline` for each.
- **`_pv_bits` inlined:** `src/mathx.cyr`, the one helper of these that `#inline` accepts.

Together they make the step 1.46× (25 particles) to 1.6× (225) faster than the first version.

## 26. M6 part c: MPM (2026-11-12); M6 complete

**`src/mpm.cyr`** ports `mpm.rs` (ADR 0002 #25):
- **The model.** `ConstitutiveModel` is the kind enum `PvConstitutiveModel` and the payload struct `PvConstitutiveLaw`.
  `constitutive_model_validate` applies Rust's rules in Rust's order: K and the viscosity; E and ν through
  `validate_elastic`; φ in [0, 90).
- **The matrices.** The `[f64; 4]` matrices are `PvMat2`, which a call can return by value (Cyrius returns at most
  three values).
- **The algebra.** The private `svd2`, `compose2` and `drucker_prager_project` work on word pointers. The SVD's
  hypot, atan2, sin and cos, the Hencky strain's ln, and the projection's exp and sin are the correctly rounded
  primitives (ADR 0004).
- **The step,** in Rust's order:
  - every check first (dt, the two counts, then each particle's model and reference volume), so a refusal changes
    nothing;
  - P2G: w·m, (w·m)·v, and −V·(σ·∇w), with ∇w = ±(the other axis's weight)·(1/dx);
  - the grid: nodes of mass ≤ 1e-20 keep their momentum (a NaN mass is not ≤, so its node is updated); the walls
    zero the outward component;
  - G2P: F ← (I + dt·L)·F, the Drucker–Prager projection, and the clamp, which resets F unless 1e-4 ≤ det F ≤ 1e4;
  - advection (z by its own velocity), then the walls.

The cell lookup is the FLIP solver's (`_pv_flip_axis`, the same expression). `MIN_SINGULAR_VALUE` is generated
(`t_private.rs` now scans `mpm.rs`).

**The rows** (`m_mpm.rs`, seed 0x60_c0_0c, a new module): 1,336.
- **`ConstitutiveModel::validate`** (87, class A): each parameter at its edges (ν at −1, 0.5 and just inside; φ at 0,
  −0, 90 and just inside) and as each special.
- **`MpmParticle::stress`** (876, class C) and **`det_f`** (102, class B): eight models over 102 deformation
  gradients, the models' specials at a stretched F.
  - The gradients: the identity; stretches and compressions; shears; 3-4-5 and 5-12-13 rotations alone and stretched
    (rational, so the inputs need no transcendentals); the Rust tests'; an inversion and a singular F; a det outside
    the clamp; each special in each entry; 40 seeded.
- **`MpmSolver::step`** (192, class S): trajectories of one fresh solver.
  - The plan's three scenes (density 1000 at spacing 0.025), at frames 1, 2, 10, 60 and 300: an elastic block falling
    to the floor (Neo-Hookean, E 1e4, ν 0.3), a sand column (Drucker–Prager, φ 30°) and a fluid column (K 1e3)
    collapsing. Also a column of the three materials stacked, and sand at φ 0.
  - The Rust tests' scenes.
  - The refusals in order; the specials as dt, as gravity, in a particle's slots, in F's entries and as V₀.
  - Particles past the walls, moving outward, in 3D.
  - The clamp: determinants on both sides of each bound.
  - 30 seeded mixed scenes.
- **`MpmParticle::new`** (4) and **`MpmSolver::new`** (16): every refusal, class A. **`ConstitutiveModel::from_json`**
  (59): serde_json's documents and hand-written ones, the seq form, fields missing, extra, duplicated and wrong-typed,
  the bare `"Fluid"`.

`validate.py` transcribes the module with the transcendentals as a parameter: glibc's (ctypes) to check the record,
MPFR's to write the corrected oracle `mpm.cr.tsv`. Each step trajectory is cached across its frames, per source. It
covers 56,347 rows with 0 failures, in about 90 s.

**Result:**
- **On the first replay** every row matched but 17: 3 stresses and 14 steps, at up to 2 ulp on the stresses and
  diverging on the steps. validate.py's MPFR transcription changes exactly those 17. With `mpm.cr.tsv` the port
  matches all 17 bit for bit (*accepted (ADR 4)*, port = corrected 17, = glibc's 0).
  - Their notes name the calls glibc misrounds: hypot 18 times among the first three of each row, cos 7, sin 6,
    log 4, atan2 3, exp 1.
  - The long trajectories misround hundreds of calls each (one note ends "… and 652 more").
- **The replay:** 14 modules, 0 failing | rows: 55,616 ok, 0 failed, 0 measured, 3 pending, 728 accepted (ADR 4: 131,
  ADR 2: 597).
- **The mutation tests found one gap.** On the first run 28 of 31 port mutations failed rows. A 32nd, μ's Lamé formula,
  used a variable before its declaration and didn't compile. Rewritten (μ without the 2), it fails 697. Of the three
  survivors:
  - the clamp's lower bound at 1e-3 was a gap, in the port and the transcription alike: no row's determinant lay
    between 1e-4 and 1e-3. A row of soft particles at det 5e-4, 9e3, 5e-5 and 1.1e4 closes it (no random numbers
    drawn), and the mutants fail it, as a 2e4 upper bound does;
  - dropping the projection's |dev| ≤ 0 guard is equivalent. At |dev| = 0, Δγ = ((2λ + 2μ)/(2μ))·tr·α with tr ≤ 0 and
    α ≥ 0. Since 2λ + 2μ = E/((1 + ν)(1 − 2ν)) > 0 for a valid ν, Δγ ≤ 0 and the first test returns;
  - skipping a node of NaN mass is equivalent at the rows' resolution: its momentum is NaN too, so its velocity is NaN
    either way, and the harness matches any NaN.
- **Mutations of the transcription:** 8 of 8 caught with the gap's row:
  - the SVD's q from hypot(f, h);
  - α without the 2;
  - Neo-Hookean without 1/J on ln J;
  - the low wall zeroing vx;
  - the clamp at 1e-3;
  - φ = 90 accepted;
  - Drucker–Prager decoded from two fields;
  - the corrected oracle taking glibc's hypot, which leaves `mpm.cr.tsv` stale.

**Tests** (`tests/mpm.tcyr`, 42 checks):
- all 13 Rust tests in Rust's order. The SVD, compose and projection tests call the private functions;
- four Cyrius-only groups:
  - the layouts (the model 48 bytes, a particle 88 with its model at byte 32, the solver 144);
  - the refusals of `new` and of validation at every edge, and a kind outside the enum (refused, a NaN stress);
  - the step's refusals in Rust's order with nothing changed, and the clamp resetting det 2e4 and 5e-5;
  - 30 steps of three materials taking no heap.

`cyrius lint` found a real fault in the first version of the SVD test: a `var vals[16]` (16 bytes) holding 128.
Fixed before the gates.

Surface parity: 401 items ok, 0 pending, 0 missing. With it every module of M6 is ported.

**Speed** (benchmarks-rust-v-cyrius, M6 part c; Rust's criterion suite has no MPM group):

| Bench (a twin) | Cyrius / Rust |
|---|---|
| one step of the elastic block, 81 particles | 8.7–9.1× |
| one step of the fluid column, 144 | 6.8× |
| one step of the sand column, 80 | 45× |

The sand column's ratio is the correctly rounded primitives' cost, about 22 calls a particle (L1). The numbers are
appended to ganita's request.

## 27. M7 part a: compute (2026-11-12)

**`src/compute.cyr`** ports `compute.rs` (ADR 0002 #26):
- **The buffers.** Rust's `Vec<f32>`s are `PvF32Column`s (`data`, `len`, `cap`; 4 bytes an element): a store through
  the `*f32` view rounds as `as f32` does, and a read widens exactly as `f64::from`.
- **The packing.** `PvPackedParticles_from_soa` packs `consistent_len` particles as [x, y, z, mass],
  [vx, vy, vz, +0] and [density, pressure]. The three unpacks write at most `count`, bounded by the buffer's whole
  chunks and every column they write, as Rust's `take(n)` over zipped iterators is.
- **The records.** The three kernel-parameter structs keep Rust's field order; `gravity` is three f32 fields.
- **The rest.** `PvComputeOp` and the two backend traits, which return status words. The four closed-form
  derivatives. `dtait_drho` on the correctly rounded pow (ADR 0004).

**`src/compute_ad.cyr`** holds `PvKernelDerivatives_autodiff_gradient`, on hisab 3.4.1's tape:
- **Outside the bundle.** hisab stays test-scope, so `dist/pravash.cyr` names no hisab symbol; the file follows
  itihas's `src/mcp.cyr`.
- **Not an optional dep.** `[features]` gates only the root manifest's own optional deps; a consumer can't select a
  dependency's feature.
- **Not a profile.** A `[lib.autodiff]` profile was tried. distlib can't verify it against a test-scope hisab, and
  reports the seven `ad_*` names as consumer-supplied.
- **The tape.** It keeps one tape, grown to twice its size when a function fills it (a root that isn't `ad_ok`, or a
  full tape), up to 2^24 nodes.

**The f32 reads (ledger L15).** serde_json 1.0.154 with `float_roundtrip` reads an f32 field by rounding the decimal
to f32 once (`f64_from_parts` with `single_precision`), and refuses one that rounds to ±∞. Integers go through
`u64 as f32` / `i64 as f32`.
- **The first replay** failed 3 of the 95 then-recorded decoder rows, all on this:
  - `1e300` and `-1e39`, which serde_json refuses and f32 of the f64 makes ±∞;
  - `3.4028235677973366e38`, whose f64 is the overflow midpoint 2^128 − 2^103, so f32 of it ties to +∞. serde_json
    reads f32::MAX.
- **The rule.** Every f32 midpoint is an f64, so a decimal and its nearest f64 never lie on opposite sides of one.
  RN32(RN64(x)) is therefore RN32(x) except where RN64(x) is a midpoint.
- **The stand-in.** `_pv_jr_f32` takes RN32 of bayan's f64 and tests for that case (`_pv_f32_is_tie`). There it
  compares the decimal with the midpoint exactly (`_pv_dec_cmp`):
  - a 32-bit-limb bignum, 170 limbs;
  - 800 kept digits and a sticky flag (an f64 has at most 767 significant digits);
  - orders past 10^±400 decided without the bignum.

  It then nudges the f32 one step toward the decimal's side (`_pv_f32_round`).
- **The text.** bayan's tree keeps no number text. `_pv_jr_num_text` finds the node's position among the tree's
  numbers in pre-order, and scans the document for that number token, skipping strings and their escapes.
- **Integers** round once from the i64, compared exactly at a midpoint.

**The rows** (`m_compute.rs`, seed 0x70_c0_0a, a new module): 1,228.
- **`PackedParticles::from_soa`** (10) and the three **unpacks** (12 each), class A: the empty SoA; 1–13 particles
  with magnitudes to 10^±40 (past f32's range both ways), the specials (±0, the smallest subnormal, f64::MAX, ±∞,
  NaN), 16777217 (a tie) and 1e-46; and mismatched columns. The unpacks run on written
  buffers, short buffers, a count above them, an extra pair, and short SoA columns.
- **The defaults** (3) and **the decoders** (71 + 15 + 13), class A:
  - serde_json's documents, the seq form, missing, extra and wrong-typed fields, the u32 bounds;
  - 38 f32 corner decimals in `dt`: midpoints written exactly (1 + 2^-24, 1 + 3·2^-24, 2^-150, 3·2^-150, the overflow
    threshold, 33554450 and 33554470 with their trailing zeros), the shortest f64 digits beside them, integer
    mantissas with exponents, integers past 2^53 next to a midpoint, 900-digit tails and 60 leading zeros;
  - three documents that place midpoints after a string holding digits and an escaped quote, an unknown field's
    numbers, the gravity array, and the seq form.
- **`dpoly6_dr2`** and **`dwendland_c2_dr`** (340 each, class B) and **`deos_drho`** (11, A): log-spread radii and
  smoothing lengths, both sides of the support, the specials.
- **`dtait_drho`** (245, class C): densities about ρ₀, γ from 0 to 7, the floored ρ₀, the specials. validate.py's MPFR
  pass changes none of them (glibc's pow is correctly rounded on all 245), so there is no `compute.cr.tsv`; the port
  matches every row bit for bit.
- **`autodiff_gradient`** (144, class D): six catalogue functions, 24 seeded inputs each. They are listed in
  `m_compute.rs`; the sixth, x0^1.5 · x1, gives a pow node an upstream gradient that isn't 1.

`validate.py` transcribes the module:
- the packing through ctypes' f32;
- serde's f32 and u32 reads in exact rationals (`f32_nearest`), with serde_json's integer paths (`-0` is −0.0; a
  negative below i64::MIN goes through `-(u as f64)`);
- the derivatives;
- hisab 1.4.0's tape and sweep, op for op, on glibc's functions.

It covers 57,575 rows with 0 failures, in about 100 s.

**Result:**
- **The port.** Every row matches but the autodiff's, which are class D:
  - 310 gradient components agree bit for bit, 34 differ by 1 ulp, 9 by 2, 6 by 3–16 and 1 by 19.
  - All four past 3 ulp are function 2's ∂/∂h, the poly6 kernel, which has no transcendentals. Its h-gradient sums
    terms of opposite sign along three paths (h², h³, h⁹), so the cancellation magnifies the difference between the
    tapes' accumulation orders: 3.4.1's sweep against 1.4.0's.
  - Probe 8's "within 3 ulp" (§4.8) was measured on functions without such a cancellation. The relative errors stay
    below 5e-15, well inside class D's 1e-12.
- **The replay:** 15 modules, 0 failing | rows: 56,844 ok, 0 failed, 0 measured, 3 pending, 728 accepted (ADR 4: 131,
  ADR 2: 597).
- **Mutations of the port.** 26 of 30 fail rows or tests on the first run, among them every mutation of the L15
  pieces but one. Of the four survivors:
  - `compute_op_name`'s strings were unchecked: a test now checks each;
  - no midpoint ended in a zero, so the bignum's 10^e on the decimal side was never applied. 33554450.0 survived by
    chance (its even neighbour is below, where the mutant's "below" also lands); 33554470.0, whose even neighbour is
    above, fails the mutant;
  - a dpoly6 regrouping, (−3·(c·d))·d, moves 22 rows by 1 ulp and 1 by 2: inside class B, visible only in the
    histogram;
  - the tape growing by 64 nodes instead of doubling changes no result.

  Two more survive parity and fail `tests/compute.tcyr` alone: the subnormal exponent q = −1075, which no f32
  midpoint reaches; and an unreset tape.
- **Mutations of the transcription.** 10 of 13 caught on the first run:
  - pow's `(g·n)·x^(n−1)` regrouped survived, because the catalogue's only pow (function 3) had an upstream
    gradient of 1. The sixth function closes it (9 rows fail);
  - 315/(64π) regrouped as (315/64)/π is equivalent: both round to the same double;
  - dropping the negative-u64 integer path leaves every row passing: the path takes an integer below i64::MIN, which
    bayan refuses (S3, ADR 0002 row 10), so the corpus can't hold one.

**Tests** (`tests/compute.tcyr`, 83 checks):
- all 11 Rust tests in Rust's order;
- seven Cyrius-only groups:
  - the layouts (80 / 36 / 28 bytes);
  - f32 rounding to nearest even, −0 kept, the padding +0;
  - the op names;
  - a `MockBackend` through `dyn PvComputeBackend`;
  - autodiff against analytical gradients, on a 1,000-op function that grows the tape, and 50 calls taking no
    heap;
  - the derivatives against central differences (class R);
  - the L15 pieces: the tie test at 1 + 2^-24, 2^-150, the overflow midpoint, f32::MAX and 1e39; the nudged rounding
    at both overflow edges; the decimal comparison (exponents, trailing zeros, the 800-digit sticky tail, the order
    bounds, the smallest subnormal); the integers; and the text locator over strings with escaped quotes.

## 28. M7 part b: ai and logging (2026-11-12); M7 complete

**`src/ai.cyr`** ports `ai.rs` (ADR 0002 #27): `PvDaimonConfig` and `PvHooshConfig`, each string as bytes and a length
(a 0 key is None), with ownership bits, `_default()`, the setters and `_drop`, and `debug(sb, alternate)`, Rust's `{:?}`
and `{:#?}`. `<str as Debug>`'s `\u{…}` set is `src/str_debug_data.cyr`, generated from rustc 1.99.1's core
(`t_str_debug.rs`: every code point classified alone and after `a`, which agree everywhere; 910 ranges, Unicode
17.0.0). Their JSON is in `src/json.cyr`. Two serde_json refusals that bayan passes through are checked on the text
before the parse: a raw byte below 0x20 inside a string, and bytes that aren't UTF-8. bayan already refuses a lone
surrogate either way, as serde_json does.

**`src/logging.cyr`** ports `logging.rs` and carries the crate's tracing (ADR 0002 #28):
- **Spans and events.** `_pv_span(target, name, fields…)` enters a sakshi span named `name{k=v …}` when its target
  enables TRACE, and `_pv_span_end` exits it. `_pv_warn` emits the sakshi WARN line `pravash::<module>: message k=v`.
  Every Rust `trace_span!` (60) and `warn!` (12) has its counterpart at Rust's point:
  - spans in Rust block scopes close explicitly, before an error return inside the block too;
  - the warnings earlier parts left out (phase field's divergence, the multiphase phase fallback, the level set's
    size check, the list functions' invalid lists) are emitted;
  - MacCormack's and BFECC's forward passes take a span-free `_pv_fg_advect_raw`, as Rust's run without one.
- **The filter.** The per-target level the init sets is EnvFilter's for a field-less event outside a span. It is a
  transcription of tracing-subscriber 0.3.23's directive state machine, with its quirk kept: it walks the trimmed
  directive but slices the untrimmed one.
- **The trace file** is tracing-chrome's shape through a sakshi hook (ledger L11).

**A finding on the way: `defer` cost the loops.** The spans were first closed by a `defer` in each instrumented
function. A pinned A/B against the uninstrumented code showed the median step unchanged, but `shallow_disturbance`
8–11%, `shallow_step` about 4% and `flip_step/25` about 8% slower, growing with the work, which a fixed span cost can't
explain. Closing the span explicitly instead restored `add_disturbance` (0.998–1.014× the old time, against
1.08–1.11× with the `defer`). So every instrumented function now closes its span before each `return` (195 of them
over 50 functions), with no `defer` left. A minimal repro showed no cost on a plain loop and about 2% on one calling
an `#inline` helper, so it is likely a side effect of the frame layout and not filed. Separately, the phase field's
divergence count first cost its step 6%; it now runs only past the clamp's ±1, at no cost on the common path.

**The rows**, two new modules:
- **`ai.tsv`** (seed 0x70_c0_0b): 579 rows, class A.
  - Both defaults.
  - 211 decodes: serde_json's text of configs whose strings mix printable ASCII, every control character, Latin-1,
    combining marks, the format, separator, private-use and unassigned classes and astral text (16 picked strings, 40
    seeded), and hand-written documents for each escape serde_json reads and each it refuses (a raw control
    character, a lone surrogate either way, an unknown escape, a short `\u`), the sequence forms, and fields missing,
    null, duplicated, extra and wrong-typed.
  - For every document that reads, serde_json's `to_string` and both Debug forms.
- **`logging.tsv`**: 251 rows, class A.
  - 99 filter rows: 63 PRAVASH_LOG values (names in any case, 0–5 with `+` and leading zeros, targets and prefixes,
    last-wins, empty pieces, the whitespace quirk, invalid, span and field forms) and unset, against 8 defaults
    (lenient: an invalid directive dropped, `error` when none is left; a dynamic-only default leaves no fallback).
  - 152 trace rows over 37 traced calls: each scenario run under a capture subscriber on the oracle (tracing-core's
    `Subscriber`, recording enters, exits and events), from the arena's two warnings to the multigrid's nested
    per-level sweeps and PCISPH's iteration spans, refusals and divergence included.

**`validate.py`**:
- recomputes every `ai` row (Python's json under serde's derive rules, serde_json's escaping, Rust's Debug);
- checks the escape table against the stdlib's Unicode 17.0.0 categories, apart from Rust: all 956,567 code points
  Rust's printable rule and Mn / Me decide are present, and the 65 others are Mc (61) and Lo (4), the Other_Grapheme_Extend
  members the categories can't name;
- recomputes every filter row through its own EnvFilter transcription;
- checks every trace row's form (spans exited in nesting order, each name a pravash span with its declared fields,
  each event a pravash warning under its target).

It covers 58,405 rows with 0 failures.

**Result:**
- **The port.** Every `ai` row, every filter row and every trace row matched on the first replay. Adding the
  instrumentation changed no other row's bits: the whole corpus replays, the thread-count runs included.
- **The replay:** 17 modules, 0 failing | rows: 57,674 ok, 0 failed, 0 measured, 3 pending, 728 accepted (ADR 4: 131,
  ADR 2: 597).
- **Mutations of the port.** `ai`: 14 of 16 on the first run. The two survivors were the UTF-8 decoder's overlong and
  surrogate checks, which no Rust `&str` can reach. Cyrius-only tests of every malformed form close them. `logging`:
  15 of 19 on the first run. The four survivors were gaps in the scenarios, each closed by a row:
  - a dynamic-only lenient default (no `error` fallback);
  - φ₀ = 1.2, between a threshold of 1 and 1.5;
  - a particle whose phase equals the phase count;
  - a diverging SPH step, which shows a forces span left open on the divergence return.
- **Mutations of the transcription.** `ai`: 6 of 7; dropping the missing-endpoint refusal is equivalent (a missing
  endpoint is refused as not a string). `logging`: 2 of 4, the other two equivalent:
  - LevelFilter's `""` is never reached through a directive;
  - slicing the target from the trimmed string can't show: with leading whitespace the level slice always starts at
    or before the `=`, so the directive fails either way.

**Tests:**
- `tests/ai.tcyr` (42 checks): both Rust tests, plus the layouts, ownership (the defaults own nothing; the setters
  copy and release, by the freelist's LIFO reuse), the escape table's edges, every malformed UTF-8 form, both Debug
  shapes, and the JSON text checks.
- `tests/logging.tcyr` (48 checks): Rust's `init_does_not_panic`, plus the span and event texts through a sakshi
  hook, the level gates (per target, and sakshi's level before any init), a full 16-deep stack, the parser's corners,
  and the trace file: written under `build/`, read back, checked entry by entry, removed, and written empty when an
  init came first.
- **Quiet suites.** Once the warnings were emitted, sakshi's default printed 590 of them during `cyrius test`: the
  refusal rows and tests provoke them on purpose. Rust's tests run with no subscriber and print none. So parity
  and the common, coupling, grid and sph tests set sakshi to ERROR at their start. The trace rows set their own
  level and restore it.

## 29. M8 part a: the Rust-test ledger (2026-11-12)

**`scripts/rust_test_ledger.py`** writes [`rust-test-ledger.md`](rust-test-ledger.md), and its `--check` runs in CI.
It is impetus's ledger, adapted. It lists every `#[test]` in `rust-old/src/**.rs` and `rust-old/tests/*.rs`, and every
Rust code fence in a doc comment, and needs each one to be either:
- **ported**: a `test_group` of the same name in the Cyrius file mapped to its Rust file (a doc fence, by its entry:
  the fence's file and line, and its `doc_*` group);
- **recorded**: a reason, per test or for a whole file.

It also fails:
- a test with neither;
- a Rust file with tests and no mapping;
- a reason for a test that is ported or doesn't exist;
- a doc fence with no entry, or an entry naming no fence;
- a group defined twice;
- a test function that holds a group and that nothing in its file calls, which impetus's ledger can't see. The
  check matters here: `tests/sph.tcyr` reaches 10 of its test functions only through other test functions.

**The count.**
- 351 unit tests and 33 integration tests, each ported under its own name.
- The 14 tests of the `oracle` hooks, n/a: they test the hooks this port added to the frozen crate, not 1.3.0's API.
- 7 doc tests. Rust marks every fence `ignore`, because each sketches a consumer's code and doesn't compile; the port
  writes each against the API and runs it:
  - `compute.rs`: a backend that supports everything, a backend whose density shader is a poly6 sum over the packed
    buffers, an FNO corrector applied through `dyn`, and the autodiff gradient of r² (exactly 1 at r = 0.5);
  - `logging.rs`: the profiling guard over three shallow-water steps. Its drop flushes all three steps' end entries
    and the closing `]`;
  - `sph.rs`: a solver step of a block, and an Arrhenius `ReactionProvider` (kimiya's, here written with the
    activation temperature Ea/R, so no gas constant is typed) applied through `update_reaction`.

  That adds 17 checks: compute 83 → 97, sph 192 → 195, logging 48 → 51.
- No unit or integration test is `#[ignore]`d in 1.3.0. `rust-old/src/lib.rs` doesn't take the README as the crate's
  docs, so the README's example is not a doc test. It is part c's, with the docs' examples.

**Total:** 405 tests, 391 ported, 14 n/a, 0 deferred. No test was folded into golden rows instead of being ported.

**Mutations of the ledger** (a scratch copy per mutation, `--check` expected to fail): 17 of 17 caught.
- a ported group renamed, or present only in another module's file;
- a group defined twice (a Rust name, and a Cyrius-only one);
- a test function's call removed (two files);
- a doc fence's entry wrong, a doc group missing, a new doc fence;
- the README taken as the crate's docs;
- a Rust file unmapped, a new Rust test, the oracle file's reason dropped;
- a reason for a ported test or for no test;
- the generated document edited.

The first run of the mutations missed the duplicate case: the script only flagged a duplicate of a Rust test name,
and the mutation duplicated a name no Rust test has. Any group defined twice in a file now fails.

**A correction to the plan.** M8's row named 23 Rust benchmarks. `rust-old/benches/benchmarks.rs` registers 21
criterion groups, which give 57 rows, as `rust-old/bench-history.csv` records. Part b covers those 21.

## 30. M8 part c: the full surface diff, the docs and the examples (2026-11-13)

**The full surface diff.** Through M7, `scripts/surface_parity.py` checked one direction: each public Rust item has
the Cyrius name the naming rules give. Part c adds three checks:
- **Each Rust trait's methods**: the 7 of `ComputeBackend`, `NeuralCorrector` and `ReactionProvider` are declared by
  their `Pv` traits.
- **Each `#[derive(..)]` on a public type**, read from its attributes: 157 derives on 45 types. Copy, Clone on a Copy
  type, and PartialEq or Eq on an enum are plain values; the others are functions (`to_json`, `_from_json`,
  `_default`, `_clone`, `eq`, `debug`, `pv_error_message`). 121 match. Two are exceptions with reasons
  (`DragCoefficient`'s PartialEq, a unit struct the port holds no value of; `RigidBody`'s Clone, copied by
  assignment), and two are renamed (`DragCoefficient`'s codecs).
- **The reverse direction**: each public name the port defines (no `_` prefix) in the bundle and in
  `src/compute_ad.cyr` is a counterpart the forward check found (351 of 466), or matches one of 18 `CYRIUS_ONLY`
  rules, each citing the ADR 0002 row or plan section that admits it (115). A rule that matches nothing fails, as an
  unexplained name does.

It found three things:
- **Six Rust `Clone`s with no counterpart**, on types whose Cyrius form owns storage: `ParticleSoa`,
  `MultiPhaseConfig`, `PackedParticles`, `NeuralCorrection`, `DaimonConfig` and `HooshConfig`. A struct assignment
  would alias their columns or strings, and a second drop would free them twice. Each now has a `PvX_clone(out, src)`
  deep copy (1, or 0 with nothing held), with a test group: bit for bit, independent of the source, outliving it,
  and from an empty source. That adds 41 checks: common 117 → 125, sph 195 → 202, compute 97 → 109, ai 42 → 56.
- **Three Rust-private types that were public names**: `KernelCoeffs`, `FluidGrid`'s multigrid level and the worker
  pool (Rust's rayon has none) were `PvKernelCoeffs`, `PvMgLevel` and `PvPool`, with `PvPool_new` and `PvPool_drop`.
  No public signature used them, and 2.0.0 would have frozen them into the surface, so they are `_PvKernelCoeffs`,
  `_PvMgLevel`, `_PvPool`, `_pv_pool_new` and `_pv_pool_drop`.
- **Rust's derived `Debug` on 36 public types**, which only the ai configs had. Cyrius's derives are Serialize,
  Deserialize, accessors, json and cyml (the guide's *Derive* section), and impetus didn't port Debug. The maintainer
  chose to port it, as M8 part d: a `debug(sb, alternate)` per type, golden rows from the oracle, and a
  `#derive(Debug)` request filed at cyrius. Until then the 36 are listed as pending.

**Mutations of the diff** (a scratch copy per mutation): 17 of 17 caught.
- a clone, a JSON codec, a default, an equality, the hand-written `Debug` or a trait method removed;
- a public fn, struct or constant added, or a private name made public;
- a `CYRIUS_ONLY` rule removed, a stale rule, a stale derive exception or rename;
- a derive added on the Rust side, and a Clone derived on a type that owns storage.

The hand-written Debug's removal was caught on the first run, but with the generic message `impl PvDebug for ..`. It
now names the method it wants, as Default's does.

**The examples.** Rust's `basic_fluid`, `shallow_water` and `multiphase` are ported as `docs/examples/*.cyr` and
print Rust's lines in Rust's format. `scripts/oracle/examples.py` derives the output each must print.
- **`shallow_water`** prints exactly what `cargo run --example shallow_water` prints.
- **The two SPH examples** print Rust's first frame, and different numbers from frame 10. validate.py's
  transcription of the solver, run on the same scenes, prints Rust's frames exactly on the f32 neighbour keys and the
  port's frames exactly on the f64 keys. So ADR 0002 #2 is the whole difference.

  In `basic_fluid` the states first differ after the second step. Before it, the f32 keys put a particle at
  x = 0.3999999999999997 (0.4 rounds up in f32) into cell 8, where the f64 keys put it in cell 7. 49 particles' lists
  then hold the same neighbours within h in another order: 0 of the 49 differ as sets. Their sums run in another
  order and differ in the last bits. The scene, which reaches 6,800 m/s by frame 10, grows that into different
  numbers.

  The README and the migration guide say so. The corpus covers the same effect: the 597 rows ADR 0002 #2 maps are
  checked against validate.py's f64 transcription.

**The docs' code.** `scripts/doc_examples.py` builds and runs every `cyrius` block in the README and the guides, the
fragments wrapped in a `main`, and each example against its `.out`. It counts 14 blocks and programs: the README's
Quick Start (Rust's, ported), 10 in the migration guide, and the three examples, all passing on the first full run.
It runs in CI after `cyrius distlib --check`. Each example must also be formatted and lint-clean. Lint compiles its
file, and an example's `lib/pravash.cyr` exists only in a consumer, so lint runs on the built copy, whose include
points at `dist/`. A first gate that linted the examples as written failed on that include. Its mutations, 7 of 7
caught:
- a float-type warning, a nonzero exit and a compile error;
- an example printing other output;
- a misindented line or a trailing space (fmt), and a line over 120 characters (lint).

An unused local isn't one of lint's rules. `PvColumn::fill_f(x)`, Rust's `vec.fill(x)` on an f64 field, was
added for the shallow-water example, which needed the private `_pv_bits` without it. Rust's
`test_friction_scales_with_roughness` now calls it, through `friction_flow`.

**Not yet scheduled.** `tests/pravash.fcyr` is still the scaffold's stub, whose comment promised the port's targets
at M8. The Rust crate had no fuzz targets, and the plan never scheduled any. It is listed for part e: targets for the
JSON readers and the `PRAVASH_LOG` parser, the inputs a consumer doesn't control.
