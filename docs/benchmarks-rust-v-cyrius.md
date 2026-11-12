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

## M4 part a: the SPH kernels and equations of state (2026-11-12)

Pinned to core 15.
- **Rust:** `sph_kernels/*` is its criterion group (poly6, spiky_grad, viscosity_laplacian at (0.3, 1.0)), one run.
  `sph_a/*` is its twin, `scripts/probes/rust/src/bin/bench_sph_a.rs`, on the same arguments (bodies in its header),
  three runs.
- **Cyrius:** `tests/pravash.bcyr`'s `bench_sph_a`, three runs interleaved with Rust's. Each loop calls the function
  alone, with its arguments in globals so nothing folds.

The medians:

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `sph_kernels/poly6` | 2.03 ns | 21 ns | 10× |
| `sph_kernels/spiky_grad` | 1.62 ns | 17 ns | 10.5× |
| `sph_kernels/viscosity_laplacian` | 1.48 ns | 15 ns | 10× |
| `sph_a/wendland_c2` | 2.57 ns | 24 ns | 9.3× |
| `sph_a/wendland_c4_grad` | 4.33 ns | 35 ns | 8.1× |
| `sph_a/tait` (γ = 7, pow) | 14.5 ns | 758 ns | 52× |
| `sph_a/tait_gamma0` (ln) | 5.86 ns | 84 ns | 14× |
| `sph_a/power_law` (pow) | 13.3 ns | 365 ns | 27× |
| `sph_a/batch_poly6_1000` | 314 ns | 16.9 µs | 54× |
| `sph_a/total_kinetic_energy_1000` | 1.22 µs | 11.3 µs | 9.3× |
| `sph_a/max_speed_1000` | 974 ns | 11.1 µs | 11× |

**The A/B behind the kernel rows** (pinned, interleaved, three runs each, measured on the first bench layout, which
also tested the kernel's index against seven branches per call): each public kernel computing only its own
coefficient, where it had filled the whole coefficient struct (four divisions), took

| Kernel | Before | After |
|---|---|---|
| poly6 | 34–35 ns | 27 ns |
| spiky_grad | 33 ns | 20–21 ns |
| viscosity_laplacian | 32 ns | 18–19 ns |

That is 1.30–1.78×, with the rows unchanged. Rust's code builds the whole struct too, but rustc inlines it and
keeps only the coefficient used.

**Reading it.**
- **The kernels are 8–10.5×,** the codegen's call and per-operation cost on a function of a dozen operations.
- **The Tait EOS and the power law are pow:** 760 and 365 ns, against about 10 ns for glibc's. That is the
  correctly rounded stand-in's cost (ledger L1, `math/powf` in M1a part 2). The γ → 0 branch's ln costs 84 ns.
- **`batch_poly6` is half allocation.** Its 8 KB result buffer comes from the freelist, which maps and unmaps every
  block over 4 KB (`cyrius:docs/development/issues/2026-11-11-freelist-maps-every-block-over-4k.md`).
  - Measured apart: 500 squared distances (4,000 bytes, under the threshold) take 5.2 µs, about 10 ns each.
  - So the 1,000 evaluations are about 10 µs of the 18.6 µs, and the mapping about 8 µs.
- **The totals are 9–11×,** a method call per particle.

## M4 part b: the neighbour grid and the z-order sort (2026-11-12)

Pinned to core 15.
- **Rust:** `SphSolver::build_neighbors` is private, so the twin, `scripts/probes/rust/src/bin/bench_sph_b.rs`,
  re-states it line for line on hisab 1.4.0's public `SpatialHash` (`new`, `clear`, `insert`, `query_cell`), the
  calls the solver makes. Its grid is kept across builds, which is the solver's `clear` path. `sort_by_zorder` is
  the public function. Medians of 7 runs.
- **Cyrius:** `tests/pravash.bcyr`'s `bench_sph_b`, on the same inputs:
  - a 40-wide 2D lattice of 1,000 and a 10³ lattice, both at spacing 0.025 and h 0.05, built after one build that
    sizes the grid;
  - the 2D lattice in a fixed shuffle (k → k·389 mod 1000), copied in and sorted at cell 0.025 per call. Both
    copy with their fast form: Rust's `copy_from_slice`, and record assignment in Cyrius, since `memcpy` copies a
    byte at a time.

Two runs, interleaved with Rust's:

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `sph_b/neighbor_build_2d_1000` (36 neighbours a particle) | 0.81–0.83 ms | 1.24–1.25 ms | 1.5× |
| `sph_b/neighbor_build_3d_1000` (216 at the interior) | 0.83–0.84 ms | 1.60–1.64 ms | 1.9× |
| `sph_b/zorder_1000` | 95–96 µs | 191–193 µs | 2.0× |

**The A/B behind the rows.** The same bench was run against the first `src/neighbor.cyr` (built in a copy of the
tree), pinned, four runs interleaved:

| Bench | Before | After | |
|---|---|---|---|
| `neighbor_build_2d_1000` | 3.22–3.26 ms | 1.20–1.24 ms | 2.6× |
| `neighbor_build_3d_1000` | 3.38–3.45 ms | 1.56–1.60 ms | 2.2× |
| `zorder_1000` | 722–739 µs | 189–192 µs | 3.8× |

All 50 rows were unchanged.
- **The build.** Before the changes, the phases measured apart came to:
  - 27,000 binary searches over the cells, 1.94 ms (72 ns each, three dependent loads a step);
  - the probe keys, 0.5 ms;
  - 36,000 pushes, 0.22 ms;
  - the merge sort, 0.1 ms.

  The changes:
  - Each axis's three probe keys depend on that axis alone, so they are made once per particle: 9 instead of 39.
  - Each cell's run is appended in one loop instead of a push per index.
  - The cells are ordered z, then y, then x, with their keys in a column of their own. A probe row's three x cells
    then sit together, so one binary search and a scan of at most a few cells replace three searches.
- **The sort.** Before the changes, it allocated a 144 KB block for the permuted copy (mapped and unmapped per call,
  about 60 µs) and moved each record with `memcpy` twice, about 480 µs. Now the sorted order is applied in place,
  cycle by cycle, with each record moved by assignment, and the codes, order and merge buffer share one 24 KB block.

**Reading it.**
- **The build is within 2× of Rust's `HashMap`.** Rust's cost is mostly the 27,000 hash lookups, so its 2D and 3D
  rows are about equal. The stand-in's grows with the lists: 216 neighbours a particle in 3D, against 36 in 2D.
- **The sort's remaining cost is the comparator call.** Measured apart, the merge sort over 1,000 codes takes
  100 µs, about 10 ns a comparison through `fncall3`, and the Morton codes 37 µs. The numbers are on the stable-sort
  request (`cyrius:docs/development/issues/2026-11-11-no-stable-sort.md`, ledger L9). The build's numbers are on
  hisab's spatial-hash request (L4).
- **`memcpy` again.** A 144 KB copy took 232 µs (0.6 GB/s). The fix was M1's: move records by assignment (L12).

## M4 part c: the brute-force sums, the free step, the block, split and merge (2026-11-12)

Pinned to core 15.
- **Rust:**
  - its criterion groups `sph_density/<n>_particles` (particle 0's density in a unit block at spacing 1/√n, h 0.05),
    `sph_step/<count>_particles` (one free step of a 0.5 block at spacing 1/√n for n = 25, 100 and 225, under
    `water_2d` at water's viscosity) and `sph_pressure_force/625_particles`, one run;
  - for the rest, the twin `scripts/probes/rust/src/bin/bench_sph_c.rs` (bodies in its header), two runs.
- **Cyrius:** `tests/pravash.bcyr`'s `bench_sph_c`, two runs interleaved with the twin's. Criterion's `iter_batched`
  clones the step's particles outside its timing. Here each step first copies them in by assignment, about 12 ns a
  particle (under 1% of the step).

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `sph_density/100_particles` | 90.9 ns | 865–867 ns | 9.5× |
| `sph_density/500_particles` | 632 ns | 4.41–4.43 µs | 7.0× |
| `sph_density/1000_particles` | 1.21 µs | 8.53 µs | 7.1× |
| `sph_step/9_particles` | 376 ns | 3.04–3.08 µs | 8.1× |
| `sph_step/25_particles` | 2.13 µs | 15.3–15.5 µs | 7.2× |
| `sph_step/64_particles` | 18.0 µs | 93–95 µs | 5.2× |
| `sph_pressure_force/625_particles` | 1.28 µs | 6.27–6.29 µs | 4.9× |
| `sph_c/viscosity_force_625` | 774–778 ns | 6.20–6.24 µs | 8.0× |
| `sph_c/create_particle_block_1000` | 3.50–3.66 µs | 85–90 µs | 24× |
| `sph_c/split_particle_8` | 142 ns | 138 µs | ≈ 970× (the L1 slow path) |
| `sph_c/merge_particles` | 7.4 ns | 95–100 ns | 13× |

**The A/B behind two rows.** The bench was built against the first `src/sph_fns.cyr` (in a copy of the tree), pinned,
three runs interleaved. All 198 rows were unchanged.

| Bench | Before | After | |
|---|---|---|---|
| `create_particle_block_1000` | 140–154 µs | 85–90 µs | 1.6× |
| `merge_particles` | 126–129 ns | 95–100 ns | 1.3× |

- The block had built each particle with `PvFluidParticle_new_2d`, a 144-byte struct return, about 86 ns each (M1
  part 3). Now one template particle is built and copied whole into each record, then placed.
- Merge's six helper calls are written out.

**Reading it.**
- **The sums and the step are the codegen's per-operation cost.** At spacing 1/√n against h 0.05, almost no pair in
  these blocks is within h, so each pair is a distance and two compares. That costs about 8 ns a pair here and about
  1 ns in Rust. In the step's 64-particle block (spacing 0.067), no pair is within h, and the density and force loops
  take about 11 ns a pair each.
- **The block is mostly its 144 KB buffer.** The freelist maps and unmaps every block over 4 KB, and each page faults
  on first touch: about 60 µs of the 85–90 (the filed `2026-11-11-freelist-maps-every-block-over-4k.md`).
- **Split into 8 children is the correctly rounded sin and cos's slow path.** Its ring angles k·π/4 include fl(π/2),
  fl(π) and fl(3π/2), where cos or sin is nearly zero and the stand-in falls to its multi-precision path: one
  `cos(fl(π/2))` takes 40 µs. Into 7 children, with no such angle, the split takes 3.5 µs. This is the same cost as
  M3 part b's DST sines at fl(k·π/2) (ledger L1).

## M4 part d: the physics extensions (2026-11-12)

Pinned to core 15. Rust's criterion suite times none of part d.
- **Rust:** the twin `scripts/probes/rust/src/bin/bench_sph_d.rs` (bodies in its header). The particles are the
  25 × 25 block at spacing 0.02 (625 particles; Rust's `sph_viscosity_force` bench block) with brute-force lists at
  h 0.05.
- **Cyrius:** `tests/pravash.bcyr`'s `bench_sph_d`. The functions change the particles, so each row starts from a fresh
  copy and both sides make the same sequence of calls. Their states, and so their work, are the same, the implicit
  solve's early exit included.

Two runs, interleaved with the twin's. The gradient-correction and secondary-update rows are after the A/B below.

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `sph_d/viscoelastic_625` | 163–166 µs | 1.48 ms | 9.1× |
| `sph_d/heat_625` | 49.5 µs | 452 µs | 9.1× |
| `sph_d/delta_sph_625` | 64.4–64.9 µs | 578–587 µs | 8.9× |
| `sph_d/implicit_viscosity_625` (5 sweeps) | 186–188 µs | 1.56–1.68 ms | 8.4× |
| `sph_d/gradient_corrections_625` | 157–158 µs | 888–909 µs | 5.7× |
| `sph_d/invert_3x3` | 269–297 ns | 426–428 ns | 1.5× |
| `sph_d/phase_change_625` | 1.03–1.22 µs | 10.3 µs | ≈ 9× |
| `sph_d/combustion_625` | 2.08 µs | 8.9–9.0 µs | 4.3× |
| `sph_d/reaction_625` (through a `dyn`) | 3.33–3.48 µs | 14.7–15.0 µs | 4.4× |
| `sph_d/contact_angle_625` | 2.24–2.30 µs | 16.5–16.7 µs | 7.3× |
| `sph_d/generate_secondary_625` | 3.10–3.38 µs | 50–51 µs | 15× |
| `sph_d/update_secondary_625` | 1.18–1.23 µs | 9.4–9.7 µs | 8× |

**The A/B behind two rows.** The bench was built against the first part d code (in a copy of the tree), pinned,
three runs interleaved. Every row was unchanged.

| Bench | Before | After | |
|---|---|---|---|
| `gradient_corrections_625` | 1.90–1.95 ms | 888–909 µs | 2.1× |
| `update_secondary_625` | 20.9–21.8 µs | 9.4–9.7 µs | 2.2× |

- The moment matrix had been accumulated in nine stack words, a load and a store per term per pair. It is now nine
  f64 locals, stored once per particle for the inverse.
- The secondary update had built two `PvVec3` struct returns per particle. It now stores each field.

**Reading it.**
- **The list functions are the pair loop's cost,** like part c's sums: about 21 neighbours a particle, each a distance,
  two compares and the kernel. Their snapshots (up to 90 KB) are one mapped freelist block per call.
- **The 3×3 inverse is close to Rust's,** which allocates two `Vec<Vec<f64>>` and two vectors per call (L7's reason).
- **Generating secondaries is mostly allocation.** Every particle sheds one here, so its 40 KB result is a block mapped
  per call (the filed freelist finding).
- **A per-component field store beats a `PvVec3` store.** `s.velocity.x = vx` for each component was 2.2× faster than
  `s.velocity = PvVec3_new(..)` on the secondary update. Part c's step and the other per-particle writes use the
  struct form in O(n) loops, where it costs little.

**Two criterion groups part c's record missed.** Part c timed `viscosity_force` and `create_particle_block` with a
twin, but Rust's suite has groups for both. Pinned, criterion one run, Cyrius two:

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `sph_viscosity_force/625_particles` (velocity (sin 0.1i, cos 0.1i, 0)) | 765 ns | 6.23–6.27 µs | 8.2× |
| `create_particle_block/100_particles` | 388 ns | 13.4–13.6 µs | 35× |
| `create_particle_block/500_particles` | 1.81 µs | 48.5–49.1 µs | 27× |
| `create_particle_block/1000_particles` | 3.55 µs | 89.2–89.6 µs | 25× |

They agree with part c's twins (8.0× and 24×). Even the 100-particle block is 14 KB, so it too is a block mapped and
unmapped per call.

## M5 part a: the solver's WCSPH step (2026-11-12)

Pinned to core 15.
- **Rust:** its criterion groups `solver_step/<count>_particles` and `solver_step_st/<count>_particles` (σ 0.072),
  one run. Each is one solver on the block at (0.1, 0.3), size 0.5, spacing 1/√n, under `water_2d` at water's
  viscosity. At spacing 0.03 or more against h 0.05, few pairs interact. The twin
  `scripts/probes/rust/src/bin/bench_sph_m5a.rs` runs a dense 25 × 25 block at spacing 0.02 (about 21 neighbours a
  particle), two runs.
- **Cyrius:** `tests/pravash.bcyr`'s `bench_sph_m5a`, interleaved with the twin. Criterion clones the particles
  outside its timing; here each step copies them in by assignment, about 12 ns a particle.

| Bench | Rust 1.3.0 | Cyrius | Cyrius / Rust |
|---|---|---|---|
| `solver_step/9_particles` | 7.96 µs | 9.95–10.1 µs | 1.27× |
| `solver_step/25_particles` | 22.0 µs | 28.0–28.9 µs | 1.3× |
| `solver_step/64_particles` | 59.7 µs | 89.2–90.0 µs | 1.5× |
| `solver_step/169_particles` | 168 µs | 337–339 µs | 2.0× |
| `solver_step/256_particles` | 271 µs | 634–635 µs | 2.3× |
| `solver_step_st/25_particles` | 22.9 µs | 28.2–28.6 µs | 1.24× |
| `solver_step_st/64_particles` | 59.8 µs | 89.6–89.9 µs | 1.5× |
| `solver_step_st/169_particles` | 177 µs | 355–357 µs | 2.0× |
| `sph_m5a/dense_625` | 798–817 µs | 2.55–2.56 ms | 3.2× |
| `sph_m5a/dense_625_st` | 823–825 µs | 2.93–3.02 ms | 3.6× |
| `sph_m5a/dense_625_verlet` | 808–817 µs | 2.61–2.63 ms | 3.2× |

**Reading it.**
- **On the sparse blocks the two are close,** where little but the neighbour build happens: Rust rebuilds a
  `HashMap` and allocates a `Vec` per cell. The ratio grows with n (1.27× at 9 particles, 2.3× at 256); this record
  doesn't split why.
- **On the dense block the step is a third grid, two thirds sums.** Timed apart, the neighbour build takes 0.86 ms of
  the 2.57 ms. The rest is the density and force loops over about 30,000 list entries, roughly 21 of them within h
  per particle.
- **A measured, declined change.** Writing the force loop's three small helpers out (`_pv_max`, the spiky gradient,
  the viscosity Laplacian), with their guards kept, gained about 6%: 2.53–2.67 → 2.37–2.46 ms, consistent over three
  interleaved runs. That is too little for the duplication, so the helper calls stay; the inlining request is filed
  (`cyrius:docs/development/issues/2026-11-11-inline-refuses-small-numeric-helpers.md`).
