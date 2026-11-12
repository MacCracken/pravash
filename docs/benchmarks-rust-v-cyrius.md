# Benchmarks — Rust 1.3.0 vs the Cyrius port

Each row times one benchmark body twice:
- **Rust:** the same work in Rust, either `rust-old/`'s criterion benches or a twin in `scripts/probes/rust/`;
- **Cyrius:** `tests/pravash.bcyr`, run by `cyrius bench`.

Port milestones append rows as their modules land. Numbers are host-specific, so compare within a column set,
never across hosts.

Speed is measured, not gated (CLAUDE.md, port plan §0). These are first numbers on a young language with no speed
work yet. They are recorded so the compiler, ganita and hisab can see where pravash's workload spends its time. A
slow row is a finding to file, with its numbers, not a reason to change the port.

## Method

- **Host:** x86_64 Linux, shared. Every run is pinned to one core (`taskset -c 15`), and Rust and Cyrius runs
  are interleaved. Differences within ±20 % are noise on this box.
- **Toolchains:** rustc 1.99.1 (release, `--locked`); Cyrius 6.7.18 (ganita 1.2.15).
- **Cyrius timing:** the average of one batch of 1,000,000 calls. `lib/bench.cyr` measures its clock-read floor
  and subtracts it from every sample: 1.22 µs per read on this host, while its clocksource is HPET.
- **Rust timing:** `Instant` around 1,000,000 calls, the median of 7 runs, with `black_box` on each argument and a
  folded result.

## M1a part 2: the correctly rounded primitives (2026-11-11, ADR 0004)

- **What is timed.** `math/*` in `tests/pravash.bcyr` against `scripts/probes/rust/src/bin/bench_libm.rs`. Each
  function runs on 1,024 seeded arguments, cycled (xorshift64; the same doubles on both sides, checked by a digest
  of the tables).
- **The Rust column** is glibc 2.44, which Rust's std calls; for cbrt it is compiler_builtins'.
- **The middle column** is the counterpart M1a part 1 dispatched. That counterpart is not correctly rounded, and
  the correctly rounded function replaced it. It stays in the replay as an ungated shadow.

Two pinned runs, within 3 % of each other; the table gives the second.

| Bench | Arguments | Rust (glibc) | Cyrius, replaced counterpart | Cyrius, correctly rounded | CR / Rust |
|---|---|---|---|---|---|
| `math/exp` | x ∈ [−20, 20) | 4.85 ns | 55 ns (`_f64_exp_polyfill`) | 110 ns | 23× |
| `math/ln` | x ∈ [0.001, 1000) | 4.64 ns | 62 ns (`_f64_ln_polyfill`) | 58 ns | 12× |
| `math/log10` | x ∈ [0.001, 1000) | 8.54 ns | — (none existed) | 94 ns | 11× |
| `math/sin` | x ∈ [−10, 10) | 9.41 ns | 45 ns (`f64_sin`) | 172 ns | 18× |
| `math/cos` | x ∈ [−10, 10) | 9.39 ns | — | 180 ns | 19× |
| `math/sin_cos` | x ∈ [−10, 10) | 12.3 ns | 82 ns (`f64_sin` + `f64_cos`) | 228 ns | 19× |
| `math/hypot` | x, y ∈ [−100, 100) | 6.23 ns | 33 ns (`ganita_f64_hypot`) | 173 ns | 28× |
| `math/powf` | x ∈ [0.1, 10), y ∈ [−10, 10) | 12.2 ns | 212 ns (`ganita_f64_pow`) | 757 ns | 62× |
| `math/atan2` | y, x ∈ [−1, 1) | 13.6 ns | 63 ns (`ganita_f64_atan2`) | 1.41 µs | 103× |
| `math/cbrt` | x ∈ [−1000, 1000) | 15.8 ns | — | 121 ns (`ganita_f64_cbrt`, already exact) | 7.6× |
| loop floor | the identity / an empty dispatch | 0.98 ns | — | 8 ns | — |

**Reading it.**
- **ln is faster than the polyfill it replaced** (58 against 62 ns), and correctly rounded. kimiya's fast path is
  table-driven, has a short Estrin polynomial and inlines its core. log10 adds one double-double product and a
  second rounding test.
- **exp, sin, cos and sin_cos cost 2–4× their counterparts.** That matches impetus's measurement (its
  `benchmarks-rust-v-cyrius.md`, M1b part 3: exp 97–104 ns, sin_cos 187–201 ns). Phase 1 decides about 99.7 % of
  calls; the rest go to double-double.
- **hypot is 5× ganita's.** It scales the operands, forms x² + y² in double-double from exact products and
  corrects its square root once by the residual. ganita's is 1–2 ulp off on 23 % of the golden rows (401 of
  1,715).
- **powf at 757 ns** equals impetus's 755 ns on its own arguments. On these arguments phase 1 decides 1,019 of
  the 1,024, and no call reaches the slow path. The cost is phase 1 itself: log x in double-double (a 2·atanh
  series), y·log x by TwoProduct, then exp's table phase.
- **atan2's mean is its slow path's.** The two paths, timed apart on the same arguments (a scratch bench, pinned):
  - **Phase 1**, in double-double, decides all but about 2^-10 of arguments. It takes **349 ns** on the 1,020 it
    decides here.
  - **The slow path**, vectoring CORDIC in the 364-bit fixed point, takes **302 µs** a call. It runs on the
    other 4 (0.4 %), which adds about 1.1 µs to the mean.

  impetus measured 634 ns on its own arguments. This is the row to file for ganita's tier. An intermediate phase
  before the fixed point, as exp, sin, cos and pow have, would remove almost all of the slow path's share.
- **The Cyrius loop floor** (8 ns) is an if-chain dispatch with an indirect argument load. Rust's (0.98 ns) is an
  inlined identity. Neither is subtracted from the rows.

These numbers go to the ganita request (ledger L1,
`ganita:docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`) as the tier's third
consumer's.

## M1a part 3: common — particles, configuration, the SoA layout and the arena (2026-11-11)

`common/*`, `soa/*` and `arena/*` in `tests/pravash.bcyr` against `scripts/probes/rust/src/bin/bench_common.rs`,
the same bodies on the frozen pravash 1.3.0. Pinned and interleaved. The Cyrius column is the last of three runs; the
first two are within 10 % except `config_validate`, which read 64 ns and then 100 ns with no code change, on a box
whose noise is ±20 %.

| Bench | Body | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|---|
| `common/particle_new_2d` | build a 2D particle | 0.8 ns | 69 ns | 86× |
| `common/kinetic_energy` | (0.5·m)·\|v\|² over 1,000 particles | 1.6 ns | 13 ns | 8× |
| `common/config_validate` | `water_2d().validate()` | 12.5 ns | 64–100 ns | 5–8× |
| `common/check_cfl` | the CFL check, 64 speeds | 1.6 ns | 13 ns | 8× |
| `common/material_custom` | a validated material | 7.4 ns | 31 ns | 4.2× |
| `soa/from_aos_1000` | 1,000 particles into 18 columns, then dropped | 16.9 µs | 206 µs | 12× |
| `soa/to_aos_1000` | back to a new particle buffer, then freed | 11.2 µs | 99 µs | 8.8× |
| `soa/write_to_aos_1000` | into an existing buffer | 6.5 µs | 36 µs | 5.6× |
| `arena/cycle_1000` | `new(1000)`, 100 blocks of 10, every other freed, `compact` (500 records moved), drop | 8.3 µs | 110 µs | 13× |

**The A/B that set two of these rows** (pinned and interleaved, two runs each; before → after):
- **Record resets copy one default record**, `*q = d`, instead of storing ten fields and three `PvVec3_zero()`
  results: `arena/cycle_1000` 163–183 µs → 109–126 µs.
- **The SoA conversions move words straight between the records and the reserved columns**, instead of a
  `push_f` / `get_f` call chain per field:
  - `from_aos` 360–372 → 204–208 µs;
  - `to_aos` 195 → 99–109 µs;
  - `write_to_aos` 133–153 → 36–40 µs.

  A particle record's 18 words are the 18 columns' order, so word k of record i is column k's slot i. The
  round-trip rows pin that mapping: swapping two columns fails all 60.

**Reading it.**
- **A struct copy is the fast way to move a record.** Measured on 500 copies of a 144-byte particle:
  - `*dst = *src` takes 11.7 ns a record;
  - an 18-word `load64` / `store64` loop takes 37 ns;
  - `memcpy` takes **245 ns**, because it copies a byte at a time.

  So `compact` and the resets assign records, and the ledger's conditional L12 (a record copy of pravash's own)
  closed unneeded. The numbers are on the open cyrius `memcpy` issue.
- **What's left of `from_aos` and the arena cycle is the freelist's large blocks.**
  - `fl_alloc` maps and unmaps every block over 4 KB, and each page faults on first touch: 60 µs for a 144 KB
    arena, about 9 µs for each of the SoA's 18 columns of 8 KB.
  - glibc's malloc reuses those buffers.
  - Filed, with these numbers (`cyrius:docs/development/issues/2026-11-11-freelist-maps-every-block-over-4k.md`).
    Solvers allocate once, so no step pays it.
- **Small bodies pay for calls and struct returns.** rustc inlines `new_2d` and `kinetic_energy` to a handful of
  instructions. Cyrius calls each function and returns the 144-byte particle through a hidden pointer. Hot loops
  read fields into f64 locals instead (port-parity §4.2).

## M1a part 4: buoyancy, vortex, bridge and JSON (2026-11-11)

`m1/*` and `json/*` in `tests/pravash.bcyr` against the same bodies in `bench_common.rs`. The Rust JSON rows use
serde_json 1.0.154 with the generator's `float_roundtrip` parse. One pinned, interleaved run.

| Bench | Body | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|---|
| `m1/terminal_velocity` | √(2mg / ρCdA), validated | 3.6 ns | 50 ns | 14× |
| `m1/reynolds_number` | ρ·\|v\|·L / μ, validated | 1.4 ns | 28 ns | 20× |
| `m1/lamb_oseen_velocity` | one correctly rounded exp | 7.4 ns | 136 ns | 18× |
| `m1/kolmogorov_scale` | two ln and one exp, correctly rounded | 12.4 ns | 264 ns | 21× |
| `m1/enstrophy_1000` | Σ ω²·dx² over 1,000 values | 759 ns | 3.0 µs | 4.0× |
| `m1/water_viscosity` | one correctly rounded pow | 12.4 ns | 792 ns | 64× |
| `m1/broadband_noise_db` | a sqrt and one correctly rounded log10 | 13.5 ns | 104 ns | 7.7× |
| `json/config_write` | `FluidConfig` to text | 408 ns | 8.3 µs | 20× |
| `json/config_read` | text to `FluidConfig` | 329 ns | 7.8 µs | 24× |
| `json/particle_read` | text to `FluidParticle` | 475 ns | 10.0 µs | 21× |
| `json/soa_read_100` | a 100-particle `ParticleSoa` (18 arrays) | 31.9 µs | 572 µs | 18× |

**Reading it.**
- **The class C rows are their primitive's cost.** `lamb_oseen_velocity` is exp's 110 ns plus the arithmetic;
  `kolmogorov_scale` is two ln (58 ns each) and an exp; `water_viscosity` is pow's 757 ns (M1 part 2). They gain
  when ganita's tier does (ledger L1).
- **`enstrophy` is 4×,** a plain load-multiply-add loop with no calls in it.
- **JSON is about 20× serde_json.**
  - **Writing** makes a builder call per token and a Grisu2 print per float. Each call goes through the
    non-aborting `_a` path that ADR 0002 #8 requires.
  - **Reading** builds bayan's value tree on a reset arena, then looks each field up by name.
  - Neither is on a step path. A faster writer would batch tokens into a local buffer; that waits for a consumer
    that needs it.

## M2: shallow water (2026-11-11)

Rust's own criterion benches (`rust-old/benches/benchmarks.rs`: `shallow_step`, `shallow_volume`,
`shallow_disturbance`), re-run pinned on core 15 for this record (criterion's median; the step and disturbance time
a fresh clone, excluded from the measurement). Most rows agree with the 1.3.0 `bench-history.csv` within the box's
noise; its 128 × 128 step and disturbance ran 2.1× and 1.7× slower than today's, so the table uses today's run.
The Cyrius rows are their twins in `tests/pravash.bcyr`: the same grid (n × n, dx 0.1, depth 1), the same drop,
clones made outside the timed batch.

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `shallow_step/32x32` | 30.7 µs | 221 µs | 7.2× |
| `shallow_step/64x64` | 128.6 µs | 929 µs | 7.2× |
| `shallow_step/128x128` | 526 µs | 3.82 ms | 7.3× |
| `shallow_volume/32x32` | 773 ns | 4.75 µs | 6.1× |
| `shallow_volume/64x64` | 3.12 µs | 20.3 µs | 6.5× |
| `shallow_volume/128x128` | 12.7 µs | 81.3 µs | 6.4× |
| `shallow_disturbance/32x32` | 5.17 µs | 8.45 µs | 1.6× |
| `shallow_disturbance/64x64` | 21.4 µs | 31.5 µs | 1.5× |
| `shallow_disturbance/128x128` | 83.4 µs | 121 µs | 1.4× |

**The A/B that set the volume rows** (pinned, interleaved, two runs each): `total_volume`'s loop on raw loads, with
Rust's `max(d, 0)` written as `d > 0 ? d : 0`, took 14.5–14.6 µs → 4.74–4.77 µs at 32 × 32. That rewrite is the same
value for every d, NaN and ±0 included. The golden rows are unchanged.

**Reading it.**
- **The step scales linearly and holds 7.2× at every size,** so the cost is per cell, not per step: about 30 calls to
  small helpers per cell. The heaviest are `_pv_max`, the upwind difference, the depth and the slope.
- **`#inline` would remove them, but declines each one:** more than 2 parameters, a body over 32 tokens, or a
  pointer parameter. Filed (`cyrius:docs/development/issues/2026-11-11-inline-refuses-small-numeric-helpers.md`)
  with these numbers. Hand-inlining a 300-line kernel in Rust's order is the alternative; it waits for that answer.
- **The disturbance is 1.4–1.6×:** one sqrt per cell inside the radius and few calls.

## M3 part a: the grid (2026-11-11)

Rust's own criterion benches (`rust-old/benches/benchmarks.rs`: `grid_diffuse`, `grid_max_speed`, `grid_step`,
`grid_step_maccormack`, `grid_step_periodic`), re-run pinned on core 15 for this record (criterion's median; the steps
time a fresh clone, excluded from the measurement). The Cyrius rows are their twins in `tests/pravash.bcyr`, on the
same inputs: diffuse's spike, max_speed's sin / cos fields, and the steps' four-cell plume, using the smoke config
(the DST projection), MacCormack at dt 0.01, or periodic walls (Gauss–Seidel).

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `grid_diffuse/32x32` | 86.5 µs | 320 µs | 3.7× |
| `grid_diffuse/64x64` | 456 µs | 1.29 ms | 2.8× |
| `grid_diffuse/128x128` | 2.05 ms | 5.04 ms | 2.5× |
| `grid_max_speed/32x32` | 360 ns | 5.89 µs | 16× |
| `grid_max_speed/64x64` | 1.44 µs | 24.1 µs | 17× |
| `grid_max_speed/128x128` | 5.72 µs | 96.2 µs | 17× |
| `grid_step/15x15` | 127 µs | 29.3 ms | 231× |
| `grid_step/30x30` | 1.07 ms | 16.3 ms | 15× |
| `grid_step/63x63` | 10.1 ms | 736 ms | 73× |
| `grid_step_maccormack/15x15` | 134 µs | 29.2 ms | 218× |
| `grid_step_maccormack/30x30` | 1.11 ms | 16.6 ms | 15× |
| `grid_step_maccormack/63x63` | 10.5 ms | 737 ms | 70× |
| `grid_step_periodic/15x15` | 70.6 µs | 284 µs | 4.0× |
| `grid_step_periodic/30x30` | 377 µs | 1.27 ms | 3.4× |
| `grid_step_periodic/63x63` | 2.07 ms | 5.68 ms | 2.7× |
| `math/fmod` (x in [−200, 200), y in [4, 64)) | 7.26 ns (glibc) | 20 ns (`_pv_fmod`, L14) | 2.8× |

**The A/B behind the max_speed rows** (pinned, interleaved, two runs each, on the grid-only bench): the fold's
NaN-ignoring max written out as `if (q >= m || m != m) m = q`, instead of a `_pv_max` call, took 8.54–8.58 µs →
5.95 µs at 32 × 32 and 140–141 µs → 96.7–98.2 µs at 128 × 128 (1.44×). The rewrite gives the same value for every
pair, NaN and ±0 included, and the golden rows are unchanged. Marking `_pv_bits` `#inline` (it is called on every
f64 store) changed nothing measurable, so it was reverted.

**Reading it.**
- **The DST steps are slow because of one function, the correctly rounded `sin`.** hisab 1.4.0's direct DST
  evaluates sin(((π·a)·b)/(n+1)) for every term. At fl(k·π/2) that sin takes **18.4 µs on average** (cos the same),
  against 165 ns on generic arguments: the result is within about 2⁻⁵³ of 0 or ±1, and the fast path's rounding
  test sends it to the accurate path. When n + 1 has small factors, many DST terms land there. A 15 × 15 grid has
  n + 1 = 14 and one term in 7 lands there: sin averages 3,992 ns over the DST's arguments, so the step is 231× Rust.
  A 30 × 30 grid has n + 1 = 29, a prime: 171 ns, and the step is 15×. A 63 × 63 grid has n + 1 = 62: 778 ns, 73×.
  - Appended to the L1 request (`ganita:docs/development/issues/2026-11-09-correctly-rounded-exp-sin-cos-hypot.md`)
    with these numbers.
  - The fix on pravash's side keeps the bits: the arguments depend only on (a, b, n), so each transform length's
    sine table can be built once and reused for every row, column and step. That comes with the DST's parity rows
    in M3 part b, measured by A/B.
- **The periodic step (no DST) is 2.7–4.0×,** in line with the shallow-water step's per-cell call cost. Its sampler
  calls the fmod stand-in four times a sample, about 1 ms of the 5.7 ms at 63 × 63.
- **diffuse is 2.5–3.7×:** a Gauss–Seidel sweep is a chain of dependent loads and adds, which neither side
  vectorises. The gap is the codegen's, 18 ns a cell update against 4.8 ns.
- **max_speed is 16–17× after the A/B.** Rust vectorises the squared-speed fold; the Cyrius loop is scalar.

## M3 part b: the DST plans (2026-11-12)

The DST projection with each transform length's sines and eigenvalue terms kept in a plan the grid owns, against
part a's sine call inside every term. The bits are unchanged: every DST, CG and step row is bit-exact before and
after. The A/B is pinned on core 15 and interleaved, three runs of each side on a grid-step-only bench (A: part a's
tree; B: this one). The runs agree within 1%. The Rust column is part a's criterion record.

| Bench | Rust 1.3.0 | A: sin in the sum | B: plans | B / Rust |
|---|---|---|---|---|
| `grid_step/15x15` | 127 µs | 28.5 ms | 760 µs | 6.0× |
| `grid_step/30x30` | 1.07 ms | 15.9 ms | 1.10 ms | 1.03× |
| `grid_step/63x63` | 10.1 ms | 719 ms | 8.6 ms | 0.85× |
| `grid_step_maccormack/15x15` | 134 µs | 28.5 ms | 801 µs | 6.0× |
| `grid_step_maccormack/30x30` | 1.11 ms | 16.2 ms | 1.36 ms | 1.23× |
| `grid_step_maccormack/63x63` | 10.5 ms | 724 ms | 9.85 ms | 0.94× |
| `grid_step_warm/15x15` (Cyrius only; against Rust's `grid_step`) | 127 µs | 29.0 ms | 190 µs | 1.5× |
| `grid_step_warm/30x30` | 1.07 ms | 16.0 ms | 965 µs | 0.90× |
| `grid_step_warm/63x63` | 10.1 ms | 721 ms | 5.64 ms | 0.56× |

**Reading it.**
- **Rust's bench steps a fresh clone of an unstepped grid every time**, and so does its twin. Each timed step
  therefore builds its plans: n² sines and n cosines per transform length, against part a's 4n³ sines and 2n²
  cosines. `grid_step_warm` clones a grid that has stepped once, so it carries its plans, as every step after a
  grid's first does.
- **The first 15 × 15 step is still 6× Rust**, and the slow path is why. Its plan build costs about 570 µs (760 − 190),
  and 25 of its 169 sines land on fl(k·π/2), at about 18 µs each on the correctly rounded slow path (ledger L1).
  A 30 × 30 grid has none (n + 1 = 29 is prime), so its first step costs only 135 µs more than a later one.
- **A grid's later steps are 0.56–1.5× Rust.** Rust's step still calls glibc's sin 4n³ times, about 900,000 at
  63 × 63, while the planned DST is a multiply-add per term.
- Appended to hisab's L6 request (`hisab:docs/development/issues/2026-11-11-dst-allocates-per-call.md`): let the
  into-scratch DST take a plan the caller keeps.

## M3 part c: MacGrid, KEpsilon and the ghost fluid (2026-11-12)

rust-old/benches has no bench for these types, so `scripts/probes/rust/src/bin/bench_grid_c.rs` times the same
bodies on the frozen oracle, and `tests/pravash.bcyr`'s `bench_grid_c` is the Cyrius side. Each row is `iters` calls
on one state, which evolves the same way on both sides, bit for bit. Runs were pinned on core 15 and interleaved,
three of each side. The runs agree within 1%, and the table gives medians.

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `mac_step/16x16` | 73.6 µs | 210 µs | 2.9× |
| `mac_step/32x32` | 353 µs | 848 µs | 2.4× |
| `mac_step/64x64` | 1.53 ms | 3.38 ms | 2.2× |
| `kepsilon_step/32x32` | 4.30 µs | 61.1 µs | 14× |
| `kepsilon_step/64x64` | 18.6 µs | 259 µs | 14× |
| `kepsilon_step/128x128` | 80.0 µs | 1.08 ms | 13× |
| `ghost_fluid/32x32` | 1.97 µs | 14.6 µs | 7.4× |
| `ghost_fluid/64x64` | 7.65 µs | 53.9 µs | 7.0× |
| `ghost_fluid/128x128` | 29.3 µs | 205 µs | 7.0× |

The bodies:
- `mac_step`: an n × n MAC grid at dx 0.1, density 1 on a 4 × 2 blob at the floor's centre, the smoke config (40
  Gauss–Seidel sweeps);
- `kepsilon_step`: `new(n, n)` under a solid-body rotation, dx 0.1, dt 0.01;
- `ghost_fluid`: a zero pressure and a circle's signed distance, dx 0.1, σ 0.07.

**The A/B behind the k-ε rows** (pinned, interleaved, three runs each): the step's three maxes are each a max against
a positive constant floor. Written out as `if (x > floor)`, instead of `_pv_max` calls, the step took 73.7–74.2 µs →
61.0–61.1 µs at 32 × 32 and 1.29–1.30 ms → 1.07–1.09 ms at 128 × 128 (1.21×). The rows are unchanged.

**Reading it.**
- **The MAC step is 2.2–2.9×.** Its Gauss–Seidel sweeps are a dependent chain, which neither side vectorises, so
  the gap is the per-operation codegen cost, as for `grid_diffuse`.
- **The k-ε step is 13–14×.** Its cells are independent: Rust vectorises the loop, and the Cyrius codegen runs it
  scalar, about 68 ns an interior cell against 4.8.
- **The ghost fluid is 7.0–7.4×.** Most cells take only the crossing test (five loads and four products), which Rust
  also vectorises.

## M3 part d: VOF and the phase field (2026-11-12)

rust-old/benches has no bench for `VofField` or `PhaseField`, so `scripts/probes/rust/src/bin/bench_vof_pf.rs` times
the same bodies on the frozen oracle, and `tests/pravash.bcyr`'s `bench_m3d` is the Cyrius side. Each row is `iters`
calls on one state, which evolves the same way on both sides. Runs were pinned on core 15 and interleaved, three of
each side, and the table gives medians. The Rust column comes from the runs interleaved with the first Cyrius build
(they agree within 5%). The Cyrius column is the B side of the A/B below (within 1%).

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `vof_advect/32x32` | 6.77 µs | 54.5 µs | 8.1× |
| `vof_advect/64x64` | 48.5 µs | 421 µs | 8.7× |
| `vof_advect/128x128` | 288 µs | 2.52 ms | 8.7× |
| `phase_field_step/32x32` | 4.65 µs | 41.3 µs | 8.9× |
| `phase_field_step/64x64` | 20.0 µs | 176 µs | 8.8× |
| `phase_field_step/128x128` | 82.7 µs | 722 µs | 8.7× |

The bodies:
- `vof_advect`: an n × n field at dx 0.1 with [0.2·L, 0.6·L]² filled (L = n·dx), in the rotation
  u = −0.1·(y − c), v = 0.1·(x − c), dt 0.01;
- `phase_field_step`: a disc of radius 0.25·L at the centre, in the same rotation, dt 0.001.

**The A/B behind both rows** (pinned, interleaved, three runs each, the rows unchanged):

| Bench | Before | After | Speedup |
|---|---|---|---|
| VOF advect 32 / 64 / 128 | 63.6–64.5 µs / 494–497 µs / 2.95–2.97 ms | 54.3–54.5 µs / 420–421 µs / 2.52 ms | 1.17–1.18× |
| phase-field step 32 / 64 / 128 | 48.9–49.0 µs / 209 µs / 859–860 µs | 41.3 µs / 176 µs / 721–724 µs | 1.18–1.19× |

- **The VOF advect:** the sub-step's four `_pv_vof_flux` calls and its `_pv_clamp` call are written out, and each x
  face shared along a row is computed once. Rust computes it twice from the same four values, so the left cell's
  F_r is the right cell's F_l.
- **The phase-field step:** its one `_pv_clamp` call per cell is written out.

**Reading it.** Both loops update independent cells from a copy of the old field, which Rust vectorises and the
Cyrius codegen runs scalar. That puts them at about 9×, between the MAC step (a dependent Gauss–Seidel chain, 2.2–2.9×)
and the k-ε step (13–14×).

## M3 part e: thread counts (2026-11-12)

Pinned to cores 8–15. Each Cyrius row times `iters` steps of one grid, after a first step that starts its pool
(`tests/pravash.bcyr`'s `bench_threads`; a clone per step would time a pool start each time). The bits are the
one-thread ones at every count. Rust is its criterion bench built with `parallel`, rayon at
`RAYON_NUM_THREADS=8`. Three Cyrius runs; the medians are given.

| Bench | Rust serial (part a) | Rust rayon, 8 threads | Cyrius T = 1 | T = 2 | T = 4 | T = 8 |
|---|---|---|---|---|---|---|
| `grid_step/63x63` (DST) | 10.1 ms | 10.7 ms | 5.99 ms | 6.00 ms | 5.74 ms | 5.81 ms |
| `grid_step_periodic/63x63` | 2.07 ms | 2.07 ms | 5.88 ms | 5.78 ms | 5.42 ms | 5.49 ms |
| `grid_step_periodic/255x255` (Cyrius only) | — | — | 95.8 ms | 86.9 ms | 86.3 ms | 83.1 ms |
| `grid_step/15x15` | 127 µs | 198 µs | | | | |
| `grid_step_periodic/15x15` | 70.6 µs | 124 µs | | | | |
| `grid_step_periodic/30x30` | 377 µs | 512 µs | | | | |

**The pooled loop alone** (the periodic advection, 40 or 400 calls on one pool):

| Grid | T = 1 | T = 2 | T = 4 | T = 8 |
|---|---|---|---|---|
| 255 × 255 | 5.65 ms | 2.85 ms | 1.55–1.79 ms | 1.37 ms |
| 63 × 63 | 344 µs | 186–190 µs | 104–105 µs | 101 µs |

**Reading it.**
- **The pool scales.** The loop runs 2.0× faster at T = 2, 3.2–3.6× at T = 4 and 4.1× at T = 8 on 255 × 255. That is
  probe 6's curve: memory-bound past four threads.
- **The step doesn't, and neither does Rust's.** Rust parallelises the same three loops: the two advections and the
  projection. On a periodic step they are about a fifth of the time; the serial Gauss–Seidel diffusion and pressure
  solves are the rest. So eight threads save about 0.4 ms of 5.9 at 63 × 63, and 13 ms of 96 at 255 × 255, which
  is what three advections at 4× predict. Rust's rayon build is no faster than its serial one at 63 × 63, and slower
  on the small grids, where rayon's per-call cost exceeds the work.
- Parallelising the solves would change their order of operations (Gauss–Seidel is a sequential sweep), so no
  thread count could keep the bits. The port stays with Rust's three loops.
