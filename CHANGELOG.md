# Changelog

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
