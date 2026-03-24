# Changelog

## [0.2.0] - 2026-03-24

### Added
- sph: `SphSolver` — stateful solver with hisab `SpatialHash` for O(n·k) neighbor queries
- sph: Neighbor list caching (flat buffer + offset table, single query per particle via `query_cell`)
- sph: Surface tension via Continuum Surface Force (CSF) model with poly6 kernel
- sph: `poly6_grad_scalar` and `poly6_laplacian` kernel functions for correct CSF
- docs: Roadmap at `docs/development/roadmap.md`

### Changed
- sph: `SphSolver` uses symmetric pressure formula (`p_i/ρ_i² + p_j/ρ_j²`) for momentum conservation
- sph: Persistent scratch buffers (densities, snapshot, positions_f32) — zero per-step allocations after first call
- sph: Spatial hash reuses HashMap capacity via `clear()` when cell size unchanged
- sph: f32 positions cached to avoid double conversion
- common: `FluidParticle` and `FluidConfig` now derive `Copy`
- common: Added `speed_squared()`, `distance_squared_to()` to `FluidParticle`
- common: `FluidConfig::validate()` checks gas_constant > 0, boundary_damping ∈ [0,1]
- common: `FluidMaterial::custom()` validates speed_of_sound > 0
- grid: `FluidGrid::new()` validates dx > 0, checks integer overflow
- grid: `max_speed()` avoids per-cell sqrt
- grid: `diffuse()` hoists loop-invariant denominator
- buoyancy: `terminal_velocity` and `reynolds_number` return `Result`
- buoyancy: `FlowRegime` is `#[non_exhaustive]`
- error: `Diverged` and `InvalidParameter` use `Cow<'static, str>`
- shallow: Scratch buffers for velocity snapshots, timestep-proportional damping

### Fixed
- sph: NaN/Inf divergence detection folded into force loop
- sph: `create_particle_block` pre-allocates with `Vec::with_capacity`
- grid: `debug_assert` on index bounds
- shallow: Hoisted `radius²` in `add_disturbance`

## [0.1.0] - 2026-03-23

### Added
- common: FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig with CFL validation
- sph: Poly6/Spiky/Viscosity kernels, density/pressure computation, pressure + viscosity forces, particle stepping with boundary enforcement, particle block creation
- grid: FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion, kinetic energy
- shallow: ShallowWater heightfield, circular disturbances, wave propagation, volume conservation
- buoyancy: Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime classification
- vortex: 2D vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale
- error: PravashError with #[non_exhaustive], CFL violation detection
