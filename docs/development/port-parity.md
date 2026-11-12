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
