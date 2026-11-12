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
