# Pravash Roadmap

> Last updated: 2026-11-13. Planned work only: what is done is in the [CHANGELOG](../../CHANGELOG.md), for 2.0.0
> and every Rust release before it.

2.0.0, the Cyrius library, was released on 2026-11-13. Its golden corpus (`tests/golden/`, frozen) records what Rust
1.3.0 did.

**How a change that moves a recorded row lands.** Most items below change results the corpus records. Such a change:
- states its divergence first, as an ADR 0002 row or a new ADR;
- adds reference tests: an analytic solution, a conservation law or a convergence order;
- puts its new outputs in derived rows that `validate.py` writes from its own transcription of the fix, as the
  `.d3.tsv` rows were written for the f64 keys. It never patches a `.tsv` row or widens a tolerance class.

An item marked **(API)** adds to the public surface. It ships additively, as a new function or setter. A Cyrius
struct literal is positional, so a new field in an existing public struct would break consumers' literals.

## 2.0.x — the audits' remaining findings

- [ ] **P(-1) on the Cyrius code** (CLAUDE.md), before the first fix: the test and benchmark sweep, the baselines,
  an internal deep review (gaps, memory, security, errors, docs) and external research. Its findings join the lists
  below.

The 1.3.0 audit deferred the items below as larger design changes, and port plan D5 kept them out of 2.0. The three
the 1.1.0 audit deferred are marked.

### grid

- [ ] **Neumann pressure walls** (∂p/∂n = 0) for `FluidGrid`. All four pressure solvers use homogeneous Dirichlet
  walls. Needs a DCT, or ghost updates with the mean removed. `MacGrid` already uses Neumann walls.
- [ ] **A periodic pressure solve.** `BoundaryCondition::Periodic` still solves with Dirichlet pressure, and
  `sample_periodic` wraps with period `nx` where the ghost cells imply `nx − 2`. Needs wrapped indices or an FFT.
- [ ] **Divergence left after the projection.** The collocated compact Laplacian doesn't match the wide divergence
  stencil. `MacGrid` projects exactly; consider making it the default path.
- [ ] **`apply_ghost_fluid`'s sign and units.** σκφ/|∇φ| is a pressure times a length. Add σκ to the Poisson
  right-hand side, with θ-weighting.
- [ ] **Turbulence:** a per-cell Smagorinsky ν_t in place of the domain average, and k-ε's advection and diffusion
  terms (it has only its sources today).
- [ ] **The DST pressure solve's cost.** 2.0 caches the sines and eigenvalue terms per transform length (the DST
  plans: the 63 × 63 step takes 8.6 ms, against 1.3.0's 10.1 ms). The O(N²) sums remain, and an FFT-based DST removes
  them.

### sph

- [ ] **2D kernel normalisation** (API). poly6, spiky and viscosity use their 3D constants on z = 0 particles; the
  Wendland kernels are 2D. Needs a dimension on the solver or its config.
- [ ] **An external-force input** (API). `step` overwrites `acceleration`, so the viscoelastic and contact-angle
  forces are applied after it and integrated again.
- [ ] **Neighbour lists.** A particle lists the 27 cells at its key ± 1 per axis, unfiltered: about 6× its true
  neighbours in 3D. Filter by h, and drop the empty z-layers in 2D. Run PCISPH's and DFSPH's loops on the solver's
  pool too; they are serial, as in Rust. (2.0 already took out the per-step `accels` allocations: a step takes no
  heap.)
- [ ] **The MLS gradient correction in 2D** is a no-op, because the 3×3 moment matrix is singular. Add a 2×2 path.

### mpm

- [ ] **Quadratic B-spline weights and APIC / MLS-MPM transfers.** The linear hat with PIC is dissipative.
- [ ] **Fluid viscosity in the stress** (1.1.0 audit, deferred again at 1.3.0: it may belong in the grid momentum).

### coupling

- [ ] **FLIP's cells:** marked fluid, air or solid, with Neumann solid walls, on the MAC layout.
- [ ] **`particle_level_set` is brute force**, O(cells · N). Put it on the neighbour grid.

### shallow

- [ ] **The height clamping is silent** (1.1.0 audit). Track the mass it removes.
- [ ] **Fully nonlinear Green–Naghdi.** The current dispersive option is a depth-weighted Peregrine operator.
- [ ] **Tracing in the hot loops** (1.1.0 audit, deferred for its cost). Measure a sakshi span inside the step
  first: `logging_b/span_on` takes 288 ns.

### bridge

- [ ] **A calibrated aeroacoustic model** (Proudman) for `tke_to_broadband_noise_db`. Today's scaling is only a
  relative cue.

## 2.1.x — the consumers and the pending rows

- [ ] **The consumers' first Cyrius builds:** kiran, joshua and impetus, on `[deps.pravash]` with `tag = "2.0.0"`.
  None consumes the library yet. Each finding is a 2.1.x issue.
- [ ] **The bridge partners:** goonj (Cyrius 2.3.1) first. ushma (Rust 1.4.0) and pavan (no local repo) follow
  when they port.
- [ ] **The 1.3.x deprecation notice on crates.io** (port plan D1), pointing at the Cyrius library. crates.io still
  shows 1.3.0 with no notice.
- [ ] **The three pending JSON rows**, which wait on bayan's policies. Each is decided with bayan, then replayed:
  - an integer `-0`, which serde_json reads as −0.0 and bayan as +0
    (`bayan:docs/development/issues/2026-11-11-json-integer-negative-zero-loses-its-sign.md`);
  - an integer past the i64 range (`100000000000000000000`), which serde_json reads into an f64 field as 1e20 and
    bayan refuses (`bayan:docs/development/issues/2026-11-09-json-integer-above-i64-max-refuses-the-document.md`);
  - a repeated unknown key (`"x"` twice), which serde_json ignores and bayan refuses
    (`bayan:docs/development/issues/2026-11-09-json-duplicate-key-has-no-opt-in-for-ignored-members.md`).

## 2.2.x – 2.x.x — the stand-ins convert

Each conversion lands when its library ships the replacement (ADR 0003; the conversion ledger, port plan §8). It
re-runs the parity rows its ledger row names, and its CHANGELOG entry names the row. A library that ships early
converts early. Within each library, the order below is by size, smallest first.

### 2.2.x — the stdlib (cyrius)

- [ ] **L2:** the Rust-semantics `min`, `max` and `clamp`, the predicates, the typed float constants and the
  saturating casts → the stdlib's `f64_*`.
- [ ] **L9:** the stable merge sort → the stdlib's stable sort.
- [ ] **L13:** `PvColumn` → a `vec` with a freeing grow, `vec_free_a`, and access that doesn't abort.
- [ ] **L15:** the f32 field read → the stdlib's f32 decimal parse, over bayan's number text.
- [ ] **L10:** the hand JSON codecs → `#derive(Serialize)` and `#derive(Deserialize)`, with the vector-array and
  tagged-variant shapes.
- [ ] **L16:** the Debug builders and the shortest float printer → `#derive(Debug)`, and the stdlib printer with
  Rust's tie rule. The printer is the slowest row: eight f64s take 54× Rust's time.

### 2.3.x — ganita

- [ ] **L14:** the exact fmod → ganita's fmod.
- [ ] **L7:** the MLS 3×3 inverse → `mat_inv_into`.
- [ ] **L1:** crmath and crlog → ganita's correctly rounded functions. Every class C row stays at 0 ulp against the
  correctly rounded rows.

### 2.4.x — hisab on the step path

- [ ] **L5:** the matrix-free CG → hisab's linear CG.
- [ ] **L8, with L3, L4 and L6**, the largest: hisab back on the step path, once a call's allocation can be scoped
  (the cyrius and hisab requests).
  - The vectors, the spatial hash and the DST convert together, and every step row re-runs, with the soak test.
  - L6 brings another DST algorithm, up to 67,000 ulp from 1.4.0's (port-parity §4.8). Its rows get a bound by
    the port-parity rule before their first comparison.

### 2.x.x — tracing and speed

- [ ] **L11:** the chrome://tracing writer → sakshi's Trace Event exporter.
- [ ] **Tracing from the worker pool.** sakshi's span stack belongs to the process, so the pool's threads emit
  nothing, and a span past 16 deep isn't recorded.
  - The exporter request (L11) asks for per-thread tracks.
  - A per-thread span stack is the other half, and is not yet filed.
- [ ] **Speed.** The median benchmark takes 4.4× Rust 1.3.0's time, from 0.76× to 37× (`create_particle_block`, on
  the freelist's large blocks). The slow rows are filed upstream with their numbers. Re-measure against the 2.0.0
  record ([benchmarks](../benchmarks-rust-v-cyrius.md)) as each conversion lands.
