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
- **Scratch:** the four scratch buffers persist across steps. `mg_levels` is a column of `*PvMgLevel` records,
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
