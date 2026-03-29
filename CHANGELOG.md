# Changelog

## [Unreleased]

## [1.2.0]

### Added
- **bridge** — cross-crate primitive-value bridges for pavan (Cp to pressure, dynamic pressure, aero forces, freestream velocity), ushma (temperature to water viscosity, heat to buoyancy forcing, TKE to eddy diffusivity), goonj (velocity to acoustic source strength, TKE to broadband noise, wave height to acoustic pressure)

### Updated
- hisab 1.1.0 -> 1.3.0, zerocopy 0.8.47 -> 0.8.48, iri-string 0.7.11 -> 0.7.12

## [Unreleased-pre-1.2]

### Fixed
- sph: Surface tension div-by-zero in solver step, PCISPH, and multiphase — added threshold floor
- sph: Phase change energy conservation — added `heat_capacity` to `PhaseChangeConfig`, proper J/kg↔K conversion
- sph: Verlet integration used zero prev_accel on every step — fixed resize guard to preserve existing values
- sph: PCISPH delta computed from particle 0 only — now averages over all particles with neighbors
- sph: Implicit viscosity solver ran fixed iterations — added convergence check with early exit
- sph: Contact angle force missing length scale — added smoothing radius divisor
- sph: Non-Newtonian viscosity (Bingham/Herschel-Bulkley) could produce extreme values — added saturation cap
- sph: Multi-phase surface tension early exit used wrong condition — fixed to check `st > 0 || it > 0`
- grid: DST eigenvalue division threshold too small (1e-20 → 1e-12)
- grid: NaN from `sample()` propagated silently — added post-advection divergence check
- grid: Multigrid recursion had no depth limit — capped at 10 levels
- shallow: Riemann solver epsilon too small (1e-10 → 1e-8)
- shallow: HLL flux mixed old/new time levels — now uses consistent snapshot velocities
- shallow: Friction calculation NaN risk near dry threshold — added depth guard
- shallow: Well-balanced pressure gradient used hardcoded 1e-3 — now uses dry threshold
- shallow: Grid minimum size was 1 — raised to 3
- shallow: Sediment array bounds only checked in debug — added runtime check
- coupling: FLIP pressure solver ran fixed 40 iterations — added convergence check with early exit
- coupling: APIC silently degraded to PIC on first step — auto-resize apic_c
- coupling: NaN could propagate through FLIP step — added post-pressure-solve check
- mpm: Deformation gradient could explode — added det(F) clamping
- mpm: Particle/mpm_data count mismatch silently truncated — now returns error
- common: `ParticleArena::active_count` could underflow — switched to `saturating_sub`
- vortex: `lamb_oseen_velocity` div-by-zero when viscosity=0 — added guard
- buoyancy: `reynolds_number` and `terminal_velocity` accepted NaN/Inf inputs — added validation

### Added
- mpm: `ConstitutiveModel::validate()` — validates material parameters (Young's modulus, Poisson ratio, bulk modulus)
- grid: `GridConfig` validation — smagorinsky_cs, vorticity_confinement, iteration counts
- grid: BFECC advection wired up via `use_bfecc` flag
- vof: `is_surface_with_threshold()`, `is_full_with_threshold()`, `is_empty_with_threshold()` — configurable threshold
- grid: Conjugate gradient pressure solver via hisab (`GridConfig.use_cg`) — matrix-free, SPD-optimized
- coupling: `drag_from_particles_indexed()` — spatial-hash-accelerated drag using hisab `SpatialHash`

### Changed
- sph: MLS 3×3 matrix inversion now delegates to `hisab::num::matrix_inverse` (LU-based)
- sph: Density threshold standardized via `MIN_DENSITY` constant across all solvers
- sph: `MultiPhaseConfig::get()` now logs warning on invalid phase index fallback
- grid: All pressure solvers now return `Result<()>` consistently
- grid: k-epsilon solver uses scratch buffer swap instead of cloning
- vortex: `kolmogorov_scale` uses log-form for numeric stability
- phase_field: Phase field clamping now warns via tracing when |φ| > 1.5
- shallow: CFL condition accounts for dispersive stability limit

## [1.1.0] - 2026-03-26

### Added
- sph: Wendland C2/C4 kernels (`kernel_wendland_c2`, `kernel_wendland_c4`) — no tensile instability
- sph: Delta-SPH density diffusion (`apply_delta_sph`) — pressure noise reduction
- sph: Velocity Verlet integration (`SphSolver.use_verlet`) — 2nd-order symplectic
- sph: Non-Newtonian viscosity (`NonNewtonianViscosity`) — power-law, Bingham, Herschel-Bulkley
- sph: Z-order particle sorting (`sort_by_zorder`) — Morton code for cache-friendly access
- sph: Enhanced `adaptive_dt` with force and viscous CFL constraints
- grid: `FluidGrid::cfl_dt()` — advective + viscous CFL timestep
- grid: Smagorinsky SGS turbulence model (`GridConfig.smagorinsky_cs`)
- sph: `ReactionProvider` trait for pluggable chemistry backends (kimiya, etc.)
- sph: `update_reaction()` — generic reaction step using any `ReactionProvider`
- sph: `CombustionConfig` implements `ReactionProvider` for backward compatibility
- sph: DFSPH solver (`step_dfsph`) — divergence-free SPH, 2-stage correction
- sph: MLS gradient correction (`compute_gradient_corrections`, `apply_gradient_correction`)
- grid: Multigrid V-cycle pressure solver (`GridConfig.use_multigrid`)
- shallow: HLL Riemann solver for flux computation (`ShallowWater.use_riemann`)
- coupling: APIC transfers (`FlipSolver.use_apic`) — affine particle-in-cell, angular momentum conserving
- All vector types migrated from `[f64; 3]` to `hisab::DVec3` (SIMD-accelerated)
- grid: Staggered MAC grid (`MacGrid`) — velocity at faces, pressure at centers, no checkerboard
- grid: Ghost fluid method (`apply_ghost_fluid`) — pressure jump at multi-phase interfaces
- vof: Volume of Fluid (`VofField`) — free surface tracking with donor-acceptor advection
- sph: Contact angle / wetting (`apply_contact_angle`) — Young's equation at solid boundaries
- sph: Phase change (`PhaseChangeConfig`, `update_phase_change`) — melting, solidification, evaporation
- shallow: Sediment transport (`SedimentConfig`, `update_sediment`) — Shields criterion, erosion/deposition
- mpm: Material Point Method (`MpmSolver`, `MpmParticle`) — neo-Hookean, fluid, Drucker-Prager constitutive models
- sph: Tait equation of state (`equation_of_state_tait`) for compressible flow
- sph: Implicit viscosity solver (`apply_implicit_viscosity`) — removes viscous CFL restriction
- sph: Particle splitting (`split_particle`) and merging (`merge_particles`) with mass/momentum conservation
- sph: Batch kernel evaluation (`batch_poly6`, `batch_distance_squared`) — SIMD-friendly patterns
- coupling: Narrow-band FLIP option (`FlipSolver.narrow_band_cells`)
- compute: `NeuralCorrector` trait for learned simulation correction (FNO, GNS backends)
- compute: `KernelDerivatives` — analytical Jacobians for differentiable simulation (dPoly6/dr², dEOS/dρ, dTait/dρ)
- grid: k-epsilon turbulence model (`KEpsilon`) — two-equation RANS with production/dissipation transport
- grid: BFECC advection (`advect_bfecc`) — 2nd-order error-compensating advection
- shallow: Green-Naghdi fully nonlinear dispersion (`use_green_naghdi`)
- sph: Foam/spray/bubble generation (`SecondaryParticle`, `generate_secondary_particles`, `update_secondary_particles`)
- compute: `KernelDerivatives::autodiff_gradient()` — reverse-mode autodiff via hisab's `Tape`/`Var`
- hisab dependency updated to 1.1 (reverse-mode autodiff, stiff ODE solvers, SDFs, k-epsilon eigendecomposition)
- shallow: `ShallowWater::cfl_dt()` — wave speed CFL timestep

### Previously Added
- shallow: Non-linear shallow water equations with convective acceleration (u·∇u)
- shallow: Flux-form continuity equation (∂(hu)/∂x + ∂(hv)/∂y) for proper mass transport
- shallow: Neumann boundary enforcement (zero-gradient height, reflected normal velocity)
- shallow: Manning's bed friction (`manning_n` per-cell roughness) with implicit treatment for stability
- shallow: Wetting/drying transitions (`dry_threshold`, `is_wet()`) with velocity zeroing, depth clamping
- shallow: Wave breaking detection (`breaking_threshold`, `is_breaking()`) with turbulent bore dissipation
- shallow: Boussinesq dispersive correction (`dispersion_coeff`) for short-wavelength accuracy
- shallow: Well-balanced hydrostatic reconstruction for terrain-following pressure gradient (lake-at-rest preserved over sloped bathymetry)
- sph: Rayon parallelism for SphSolver density and force loops (`parallel` feature gate)
- grid: Rayon parallelism for advection and pressure projection (`parallel` feature gate)
- common: `ParticleSoa` — SIMD-friendly Structure-of-Arrays layout with AOS↔SOA conversion
- compute: GPU-agnostic `ComputeBackend` trait, `PackedParticles` f32 buffer packing, kernel parameter structs for SPH/grid/shallow
- common: `ParticleArena` — pre-allocated pool with alloc/free/compact, `ArenaHandle` for block references
- logging: `init_profiling()` — chrome://tracing JSON output for flame graph analysis via tracing-chrome
- sph: Multi-phase SPH (`step_multiphase`, `MultiPhaseConfig`, `PhaseProperties`) with per-phase EOS, viscosity, and interface tension
- common: `FluidParticle.phase` field (u8) for multi-phase particle identification
- phase_field: `PhaseField` — Allen-Cahn interface tracking with advection, double-well potential, Neumann BCs
- sph: Viscoelastic fluids (`ViscoelasticConfig`, `update_viscoelastic`) — Oldroyd-B conformation tensor evolution with elastic stress
- sph: Heat transfer (`HeatConfig`, `update_heat`) — SPH Laplacian-based conduction with implicit advection
- common: `FluidParticle.temperature` field (default 293.15 K)
- sph: Combustion reaction (`CombustionConfig`, `update_combustion`) — fuel depletion with exothermic heat release
- common: `FluidParticle.fuel` field for combustion concentration

### Changed
- `#[non_exhaustive]` added to 18 public structs for forward-compatible API evolution
- Cargo.toml: license corrected from deprecated `GPL-3.0` to `GPL-3.0-only`
- shallow: NaN dt now rejected (consistent with grid.rs)
- sph: `SpatialHash::new()` error properly propagated (hisab API change)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- grid: Persistent scratch buffers (zero allocation after first step)
- grid: DST pressure solver now propagates transform errors instead of silent ignore
- common: `FluidConfig::validate()` rejects NaN/Inf for dt, smoothing_radius, density, gas_constant
- common: `FluidConfig::validate()` rejects inverted bounds (min > max)
- sph: PCISPH divergence detection (NaN/Inf check on acceleration and predicted positions)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- coupling: FLIP solver particle boundary clamping to grid domain
- coupling: Directional drag cross-section for box shapes (projected onto velocity normal)
- deny.toml for cargo-deny license and advisory checking

## [0.24.3] - 2026-03-24

### Added
- sph: PCISPH pressure solver (`step_pcisph`) with iterative density-error correction
- sph: Adaptive timestep (`SphSolver::adaptive_dt`) — CFL-driven with NaN guards
- grid: MacCormack advection (forward-backward correction with neighbor clamping)
- grid: Periodic boundary conditions (wrapping edges, periodic sampling, `advect_periodic`)
- grid: DST Poisson solver (exact solve via hisab `dst`/`idst` for wall boundaries)
- coupling: `RigidBody` with `BodyShape` (sphere, box) signed distance functions
- coupling: Two-way SPH-rigid body coupling (`couple_sph_bodies`) with penalty forces
- coupling: `FlipSolver` — FLIP/PIC hybrid (P2G, pressure projection, G2P blend)
- coupling: Particle-level set surface tracking (`particle_level_set`)
- coupling: Added mass computation (`effective_mass`, `integrate_bodies_with_added_mass`)
- coupling: Drag from velocity fields (`drag_from_particles`)

### Changed
- grid: Pressure projection uses DST for wall BCs (exact), GS for periodic
- grid: Correct dx scaling in advection, diffusion, divergence, projection
- grid: Diffusion uses correct implicit equation with preserved RHS
- grid: Warm-start pressure solve from previous step
- grid: Minimum grid size 4x4 enforced
- grid: `GridConfig` is `#[non_exhaustive]` with `Serialize`/`Deserialize`/`Default`
- grid: NaN propagation in `sample()`, NaN/Inf dt validation
- coupling: Central-difference surface normals with degenerate fallback
- coupling: Correct dt scaling in FLIP pressure projection
- coupling: `FlipSolver` grid dims are private with getters
- hisab: Switched to local path dependency for DST access

## [0.2.0] - 2026-03-24

### Added
- sph: `SphSolver` — stateful solver with hisab `SpatialHash` for O(n·k) neighbor queries
- sph: Neighbor list caching via `query_cell` (zero per-particle allocation)
- sph: Surface tension via CSF model with poly6 gradient/laplacian kernels
- docs: Roadmap at `docs/development/roadmap.md`

### Changed
- sph: Symmetric pressure formula (`p_i/ρ_i² + p_j/ρ_j²`) for momentum conservation
- sph: Persistent scratch buffers — zero per-step allocations after first call
- common: `FluidParticle` and `FluidConfig` derive `Copy`
- common: Added `speed_squared()`, `distance_squared_to()`
- common: `FluidConfig::validate()` checks gas_constant, boundary_damping
- common: `FluidMaterial::custom()` validates speed_of_sound
- grid: `FluidGrid::new()` validates dx, checks overflow
- grid: `max_speed()` avoids per-cell sqrt
- buoyancy: `terminal_velocity` and `reynolds_number` return `Result`
- buoyancy: `FlowRegime` is `#[non_exhaustive]`
- error: `Diverged` and `InvalidParameter` use `Cow<'static, str>`

### Fixed
- sph: NaN/Inf divergence detection folded into force loop
- sph: `create_particle_block` pre-allocates with `Vec::with_capacity`
- shallow: Scratch buffers resize on deserialization to prevent panic

## [0.1.0] - 2026-03-23

### Added
- common: FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig with CFL validation
- sph: Poly6/Spiky/Viscosity kernels, density/pressure, forces, particle stepping, block creation
- grid: FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion, kinetic energy
- shallow: ShallowWater heightfield, circular disturbances, wave propagation, volume conservation
- buoyancy: Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime
- vortex: 2D vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale
- error: PravashError with #[non_exhaustive], CFL violation detection
