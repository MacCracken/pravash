# Migration Guide

## From 0.24.x

### Breaking Changes

#### `FluidParticle` — new fields
`FluidParticle` has new fields: `phase` (u8), `conformation` ([f64; 3]), `temperature` (f64), `fuel` (f64). If you construct particles with struct literals, add the new fields. The `new()` / `new_2d()` constructors handle defaults automatically.

```rust
// Before (0.24.x):
let p = FluidParticle { position, velocity, acceleration, density, pressure, mass };

// After:
let p = FluidParticle::new(position, mass); // preferred — handles all defaults
// Or with struct literal (all fields required):
let p = FluidParticle {
    position, velocity, acceleration, density, pressure, mass,
    phase: 0,
    conformation: [1.0, 0.0, 1.0],
    temperature: 293.15,
    fuel: 0.0,
};
```

#### `FluidConfig::validate()` — stricter
Now rejects NaN and Infinity for `dt`, `smoothing_radius`, `rest_density`, `gas_constant`. Also rejects inverted bounds (min > max). Previously these passed validation silently.

#### SPH pressure formula — now symmetric
`pressure_force()` and the brute-force `step()` now use the symmetric momentum-conserving formula `p_i/ρ_i² + p_j/ρ_j²`, matching `SphSolver::step()`. Previously used the non-symmetric form. Results will differ slightly for the same initial conditions.

#### `SpatialHash::new()` returns `Result`
The hisab `SpatialHash::new()` API changed to return `Result`. `SphSolver` handles this internally — no action needed unless you use `SpatialHash` directly.

#### `#[non_exhaustive]` on public structs
Many public structs now have `#[non_exhaustive]`. If you construct them outside the crate via struct literals, you'll need to use constructors or builder patterns instead. Affected: `FluidMaterial`, `FluidGrid`, `ShallowWater`, `PhaseField`, `FluidConfig`, `GridConfig`, `SphSolver`, `FlipSolver`, and compute/AI config structs.

#### `ShallowWater` — new fields
New fields: `manning_n` (Vec), `dry_threshold`, `breaking_threshold`, `breaking_dissipation`, `dispersion_coeff`. All have sensible defaults via the constructor.

#### `ShallowWater::step()` — non-linear
The step function now uses non-linear shallow water equations with convective terms, flux-form continuity, boundary enforcement, wet-dry handling, breaking detection, and dispersive correction. Results will differ from the previous linearized scheme.

### New Features

#### `ParticleSoa` — SOA layout
Structure-of-Arrays alternative to `FluidParticle` slices. Convert with `ParticleSoa::from_aos()` / `.to_aos()`. Better cache behavior for SIMD operations.

#### `ParticleArena` — memory pool
Pre-allocated particle pool with `alloc()` / `free()` / `compact()`. Avoids per-step heap allocation for dynamic particle creation/destruction.

#### `ComputeBackend` trait — GPU-agnostic compute
Implement this trait to accelerate SPH/grid/shallow computations on GPU or distributed backends. Pravash provides `PackedParticles` for f32 buffer packing.

#### `PhaseField` — interface tracking
Allen-Cahn phase-field solver for tracking fluid interfaces. Use with grid-based simulations.

#### Multi-phase SPH
`SphSolver::step_multiphase()` with `MultiPhaseConfig` for immiscible fluid simulation (water-oil, water-air).

#### Viscoelastic fluids
`update_viscoelastic()` with Oldroyd-B conformation tensor for honey/lava effects.

#### Heat transfer + combustion
`update_heat()` for SPH conduction, `update_combustion()` for fire effects.

#### Rayon parallelism
Enable the `parallel` feature for automatic parallelization of SPH density/force loops and grid advection/projection.

#### Profiling
`logging::init_profiling()` produces chrome://tracing JSON for flame graph analysis.

### New Feature Flags
- `parallel` — Rayon parallelism for SPH and grid
- `compute` — GPU-agnostic compute interface

### New Dependencies
- `rayon` (optional, behind `parallel`)
- `tracing-chrome` (optional, behind `logging`)
