# Pravash Roadmap

> Last updated: 2026-03-26

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
