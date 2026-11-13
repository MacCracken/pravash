# Pravash Roadmap

> Last updated: 2026-11-13 (after 2.0.1). Planned work only: what is done is in the [CHANGELOG](../../CHANGELOG.md), for 2.0.0
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

2.0.1's hardening pass (the CHANGELOG) fixed what it found that keeps Rust's behaviour. The items below change it,
so each needs its ADR first. The 1.3.0 audit deferred most of them as larger design changes, and port plan D5 kept
them out of 2.0. The 1.1.0 audit's three deferrals are marked, and so are 2.0.1's findings (its review and its
external research).

### Time steps and solver reports (2.0.1)

- [ ] **`adaptive_dt`'s limits.** It has no acoustic limit Δt ≤ C·h/(c + v_max), no capillary limit for σ > 0, and
  its viscous limit h²/(6ν) reads the dynamic viscosity the step divides by ρᵢρⱼ as if it were kinematic. The
  stability constants are written for the smoothing length, about h/2 of pravash's support radius. Under that
  reading `water_2d`'s dt of 1e-3 is about 4× the acoustic limit.
- [ ] **A stiffness check.** `validate` checks only k > 0. Warn when c < 10·v_max, Monaghan's rule for about 1%
  density error.
- [ ] **Solver reports** (API). PCISPH and DFSPH return Ok whether or not they converged. Report the last solve's
  iterations and residual. DFSPH also lacks a warm start and a minimum iteration count.
- [ ] **Non-Newtonian caps.** The power law is uncapped as the shear rate → 0, and Bingham's cap is 1e6·μ_p, which is
  0 when μ_p = 0. Use Papanastasiou regularisation or an absolute μ_max, and feed it to the viscous time step.
- [ ] **MPM:** a stable-dt helper, Δt ≤ C·dx/(c_p + |v|). F reset to I outside det F ∈ [1e-4, 1e4] discards the
  volume; reset a fluid's F to √J·I instead. Neo-Hookean clamps J at 1e-10, so an inverted element gets a huge λ·ln J.
- [ ] **Phase field:** check dt ≤ dx²/(4Mε²) and ε ≳ dx, where today only a WARN past |φ| = 1.5 fires.

### Non-finite inputs Rust passes through (2.0.1)

- [ ] **MPM:** a NaN mass spreads NaN to its neighbours, and the step still returns Ok. FLIP zeroes a non-finite mass.
- [ ] **FLIP:** after one PV_ERR_DIVERGED the NaN pressure warm-starts the next solve, so every later step diverges.
  Clear it on that path, or document that the solver must be rebuilt.
- [ ] **`terminal_velocity`** returns Ok(NaN) when m·g or the drag term overflows; refuse a non-finite argument.
- [ ] **VOF `advect`** at dx = 0 with any motion runs 1,024 sub-steps and turns the interior NaN. Refuse the cell size.

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
- [ ] **δ-SPH has no lasting effect** (2.0.1). `apply_delta_sph` diffuses ρ, and the next step's summation
  recomputes ρ, which undoes it. The Molteni–Colagrossi form also lacks the hydrostatic correction. It needs a
  continuity-equation density option (2.1+ below); document the limit now.
- [ ] **Negative free-surface pressure** (2.0.1). Summation density with p = k(ρ − ρ₀) goes negative at the free
  surface, and particles clump. Offer a clamp ρ ≥ ρ₀ for WCSPH, as SPlisHSPlasH does (API).
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
- [ ] **Thin layers** (2.0.1). u = hu/h just above the dry threshold can spike and collapse dt. Use Kurganov–Petrova
  desingularisation, and a draining-time-step flux limit in place of the height clamp. That keeps h ≥ 0 and conserves
  mass, which also covers the mass item above.
- [ ] **Tracing in the hot loops** (1.1.0 audit, deferred for its cost). Measure a sakshi span inside the step
  first: `logging_b/span_on` takes 288 ns.

### bridge

- [ ] **A calibrated aeroacoustic model** (Proudman) for `tke_to_broadband_noise_db`. Today's scaling is only a
  relative cue.

## 2.1.x — the consumers and the pending rows

- [ ] **The consumers' first Cyrius builds:** kiran, joshua and impetus, on `[deps.pravash]` with `tag = "2.0.1"`.
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

### 2.1+ — capabilities (2.0.1's external research)

- [ ] **Boundary particles** for walls and rigid bodies (Akinci et al. 2012). Today's clamp-and-damp walls add no
  density, so wall particles are density-deficient.
- [ ] **Particle shifting** (Lind et al. 2012), against the clumping and voids of long runs.
- [ ] **A continuity-equation density** for WCSPH, with Fourtakas or Antuono density diffusion. This is what gives
  δ-SPH its effect.
- [ ] **Geometric (PLIC) VOF**, dimension-split, in place of the donor-cell scheme that smears the interface. Also a
  mass-conserving Allen–Cahn: the plain one shrinks drops.
- [ ] **A fixed-corotated MPM model with snow plasticity** (Stomakhin et al.), the inversion-robust standard.
- [ ] **A calibrated initial lattice:** `create_particle_block` takes the mass independently of the spacing, so
  Σm·W ≠ ρ₀ at t = 0 and the run starts with a pressure transient. Add a helper that sets m = ρ₀ / ΣW.

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
- [ ] **L17:** the worker pool keeps 12.8 KB per start → a `thread_join` that frees each thread's handle and TLS
  block, and a `chan_free`. Also two stdlib requests outside the ledger, with pravash's numbers: the growable arena
  that keeps a chunk per larger need (the JSON decode arena), and the string builder that never frees an outgrown
  buffer.
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
