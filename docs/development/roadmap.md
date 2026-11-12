# Pravash Roadmap

> Last updated: 2026-11-11

### 2.0.0 — Cyrius port (planned)

- [ ] Port to Cyrius with Rust 1.3.0 as the frozen oracle in `rust-old/`: the plan of record is
  [pravash-port-plan.md](pravash-port-plan.md). It follows tanmatra, kimiya and impetus.
- [ ] 2.0 carries every module and feature: bridge, ai, logging, `parallel` (on `thread.cyr`) and serde (on
  `#derive`). The plan's §1.1 language review of Cyrius 6.7.18 found no language reason to defer any of them
- [ ] Upstream requests (hisab, ganita, the stdlib) are filed at M0 and as found, but never block the port.
  pravash ships local stand-ins marked `UPSTREAM(Ln)` and converts them when the library lands (plan §8, the
  conversion ledger)
- [x] Decisions D1–D6 (plan §0.2), made 2026-11-11:
  - **D1:** tag and publish 1.3.0 after the oracle hooks land.
  - **D2:** a local `crmath`, plus the ganita request.
  - **D3:** f32 hash keys ported faithfully, then switched to f64 before 2.0.
  - **D4:** JSON that decodes compatibly, with no text identity required.
  - **D5:** no deferred audit physics in 2.0.
  - **D6:** `parallel` as a runtime thread count.
- [x] M0 steps 0.1–0.4 (Rust, before the freeze), done 2026-11-11:
  - P(-1) gates green;
  - the hidden `oracle` feature, with state dumps for every stateful type, the neighbour lists, the
    pressure-solver hooks and the MLS inverse;
  - `tests/oracle.rs`: 14 tests, mutation-checked, run with and without `parallel`;
  - `Cargo.lock` tracked, glibc 2.44 recorded.
- [x] M0 step 0.5: committed `a88bc4e`, then tagged 1.3.0 and published to crates.io (D1), 2026-11-11
- [x] M0 steps 1–3, done 2026-11-11:
  - the `cyrius port` scaffold, with the frozen crate in `rust-old/`;
  - the repairs (CLAUDE.md, CI, `state.md`, CONTRIBUTING, SECURITY, the README banner);
  - ADRs 0001–0003;
  - `cyrius.cyml`, with every Cyrius gate green.
- [x] M0 step 4: the semantics probe (port-parity §1)
- [x] M0 step 5: the golden-value generator (`scripts/oracle/`: constants 29 + math 33,239 rows, validated)
- [x] M0 step 6: `src/constants.cyr` (public + std float constants, compiled and compared by `probe_tables.py`)
- [x] M0 step 7: the parity harness (`tests/parity.tcyr`: 5,149 ok, 0 failed; the rest measured or pending)
- [x] M0 step 8: the cross-target baselines (identical bits on x86_64 ELF, PE and aarch64; K = 2 for the primitives, 0 for cbrt)
- [ ] M0 steps 9–10: the parity harness, the baselines, the probes, and filing the upstream requests

### 1.3.0

- [x] Toolchain: MSRV 1.99 (latest stable, edition 2024); deps to latest (hisab 1.4, criterion 0.8); unused `ai` network deps removed; CI actions bumped
- [x] Full module audit (sph, grid/vof/phase_field, shallow/coupling, common/mpm/compute/bridge/buoyancy/vortex) — see CHANGELOG 1.3.0 for every fix
- [x] Regression test per fix, mutation-checked for the projection, solver-sign, delta-SPH, DFSPH, implicit-viscosity, MPM-stress and wave-speed fixes
- [x] `bench-history.sh` records all benchmarks (µs results were dropped)

#### Deferred from the 1.3.0 audit (larger design changes)

- [ ] **grid**: Neumann (∂p/∂n = 0) pressure walls for `FluidGrid` — all solvers use homogeneous Dirichlet; needs DCT (hisab `dct`) or ghost updates + mean removal. `MacGrid` already uses Neumann.
- [ ] **grid**: periodic pressure solve (wrapped indices / FFT) — `BoundaryCondition::Periodic` uses Dirichlet pressure; `sample_periodic` wraps with period `nx` while ghosts imply `nx − 2`
- [ ] **grid**: collocated compact Laplacian vs. wide divergence stencil leaves residual divergence after projection — `MacGrid` is exact; consider making MAC the default path
- [ ] **grid**: `apply_ghost_fluid` sign/units (`σκφ/|∇φ|` is pressure × length); should add σκ to the Poisson RHS with θ-weighting
- [ ] **grid**: local (per-cell) Smagorinsky ν_t instead of a domain average; k-ε advection/diffusion terms (only sources today)
- [ ] **grid**: DST pressure solve is O(N²) per line with `sin()` per element — cache a sine table / FFT-based DST (perf)
- [ ] **sph**: 2D kernel normalisation — 3D poly6/spiky/viscosity constants are used on z = 0 particles (Wendland kernels are 2D); needs a dimension on the solver/config
- [ ] **sph**: `step()` overwrites `acceleration`, so external forces (viscoelastic, contact angle) must be applied after and re-integrated — add an external-force input
- [ ] **sph**: neighbour lists keep all 27 cells unfiltered (≈6× true neighbours in 3D); filter by h, drop empty z-layers in 2D; parallelise PCISPH/DFSPH/multiphase loops; remove per-step `accels` allocations
- [ ] **sph**: MLS gradient correction is a no-op in 2D (singular 3×3 moment matrix) — use a 2×2 path
- [ ] **sph**: f32 spatial hash loses precision at large coordinates (≈0.05h at 1e4 with h = 0.01)
- [ ] **mpm**: quadratic B-spline weights + APIC/MLS-MPM transfers (linear hat + PIC is dissipative); fluid viscosity in stress
- [ ] **coupling**: FLIP cell marking (fluid/air/solid), Neumann solid walls, MAC layout; `particle_level_set` is brute force O(cells·N)
- [ ] **bridge**: calibrated aeroacoustic noise model (Proudman) for `tke_to_broadband_noise_db`; current scaling is a relative cue
- [ ] **shallow**: fully nonlinear Green-Naghdi (current option is a depth-weighted Peregrine operator)

### 1.1.0

- [x] P(-1) scaffold hardening — full deep audit
- [x] Bug fixes — from audit findings below

#### Critical — Correctness & Safety

- [x] **sph**: Surface tension div-by-zero risk (3 locations) — added threshold floor
- [x] **sph**: Phase change energy conservation bug — added heat_capacity to PhaseChangeConfig
- [x] **sph**: DFSPH stale density — verified correct (density recomputed before stage 2)
- [x] **sph**: Verlet integration prev_accel clobber — fixed resize guard
- [x] **sph**: MLS gradient direction — verified correct (sign convention consistent)
- [x] **sph**: PCISPH delta robustness — now averages over all particles
- [x] **mpm**: Lambda singularity — added ConstitutiveModel::validate()
- [x] **mpm**: Deformation gradient explosion — added det(F) clamping
- [x] **mpm**: Unvalidated parameters — added validate() with bounds checks
- [x] **coupling**: FLIP pressure solver — added convergence check with early exit
- [x] **shallow**: Sediment array bounds — replaced debug_assert with runtime check
- [x] **shallow**: Grid minimum size — changed from 1 to 3
- [x] **common**: ParticleArena::active_count — switched to saturating_sub
- [x] **vortex**: Lamb-Oseen div-by-zero — added viscosity <= 0 guard

#### High — Numerical Stability & Correctness

- [x] **sph**: Implicit viscosity — added convergence check with early exit
- [x] **sph**: KernelCoeffs::new() — added debug_assert for h > 0
- [x] **sph**: Contact angle force — added h as length scale divisor
- [x] **sph**: Multi-phase surface tension — fixed early exit condition
- [x] **sph**: Non-Newtonian viscosity — added saturation cap
- [x] **grid**: DST eigenvalue threshold — changed 1e-20 to 1e-12
- [x] **grid**: NaN from sample() — added post-advection NaN check
- [x] **grid**: GridConfig validation — added smagorinsky_cs, iterations checks
- [x] **grid**: Multigrid depth limit — capped at 10 levels
- [x] **shallow**: Riemann solver epsilon — changed 1e-10 to 1e-8
- [x] **shallow**: HLL flux time levels — now uses snapshot velocities
- [x] **shallow**: Friction NaN — added depth.max(1e-6) guard
- [x] **shallow**: Well-balanced clamping — uses dry_thr instead of hardcoded 1e-3
- [x] **shallow**: CFL dispersion — added dispersive stability limit
- [x] **coupling**: APIC initialization — auto-resize apic_c in step()
- [x] **coupling**: FLIP NaN check — added post-pressure-solve divergence check
- [x] **mpm**: Particle count mismatch — now returns error instead of silent truncation

#### Medium — Code Quality & API

- [x] **sph**: #[must_use] on split/merge — already present
- [x] **sph**: #[inline] on hot paths — already present
- [x] **sph**: Density threshold — added MIN_DENSITY constant, replaced all occurrences
- [x] **sph**: MultiPhaseConfig::get() — added tracing::warn on fallback
- [x] **grid**: BFECC advection — wired up via existing use_bfecc flag
- [x] **grid**: k-epsilon clones — replaced with scratch buffer + swap
- [x] **grid**: Pressure solver error handling — all return Result<()>
- [x] **grid**: #[must_use] on step() — removed (double_must_use with Result)
- [x] **shallow**: Dry threshold consistency — friction now uses dry_thr
- [x] **shallow**: #[inline] on query functions — already present
- [x] **shallow**: add_disturbance radius — added <= 0 guard
- [ ] **shallow**: Silent height clamping — deferred (tracking mass loss adds complexity)
- [x] **coupling**: #[must_use] on drag_from_particles — already present
- [x] **coupling**: narrow_band_cells — documented as reserved for future use
- [x] **coupling**: Pressure iteration count — extracted to const with convergence check
- [ ] **mpm**: Fluid viscosity in stress — deferred (may be handled in grid momentum)
- [x] **compute**: #[must_use] on KernelDerivatives — already present
- [x] **compute**: PackedParticles SOA check — added debug_assert_eq
- [x] **vortex**: Kolmogorov scale — switched to log form
- [x] **phase_field**: Clamping divergence — added tracing::warn when |φ| > 1.5
- [x] **vof**: Surface threshold — added configurable threshold methods

#### Low — Tracing & Documentation

- [x] **sph**: Added tracing on pressure_force, viscosity_force, split/merge, sort
- [x] **grid**: Added tracing on diffuse, divergence
- [ ] **shallow**: Hot loop tracing — deferred (performance concern in inner loops)
- [x] **buoyancy**: Added NaN/Inf input validation

### Engineering Backlog

- No current items

---

## Cross-Crate Bridges

- [x] `bridge.rs` module — primitive-value conversions for cross-crate fluid dynamics (no deps on sibling crates)
- [x] **pavan bridge**: airfoil pressure distribution → surface flow velocity boundary conditions; wind field → free-stream velocity
- [x] **ushma bridge**: fluid temperature field → viscosity; heat source positions → thermal buoyancy forcing
- [x] **goonj bridge**: fluid velocity field → acoustic source strength (aeroacoustics); turbulent kinetic energy → broadband noise level
