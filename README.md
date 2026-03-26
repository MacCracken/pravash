# Pravash

> **Pravash** (Sanskrit: प्रवास — journey, flow) — fluid dynamics simulation for AGNOS

[![CI](https://github.com/MacCracken/pravash/actions/workflows/ci.yml/badge.svg)](https://github.com/MacCracken/pravash/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/pravash.svg)](https://crates.io/crates/pravash)
[![docs.rs](https://docs.rs/pravash/badge.svg)](https://docs.rs/pravash)

Particle-based and grid-based fluid simulation. SPH for real-time effects, Euler/Navier-Stokes for accurate simulation, shallow water for surface waves, multi-phase for immiscible fluids, viscoelastic for honey/lava, heat transfer and combustion for fire effects. Built on [hisab](https://crates.io/crates/hisab) for math foundations.

**Non-linear SWE, HLL Riemann, Green-Naghdi dispersion, sediment transport, multi-phase SPH, DFSPH, MPM, VoF, phase-field, Oldroyd-B viscoelasticity, non-Newtonian viscosity, heat conduction, combustion, foam/spray/bubble, APIC, MAC grid, multigrid, k-epsilon turbulence, autodiff, Rayon parallelism, GPU-agnostic compute interface** — zero `unsafe`, 290+ tests.

## Installation

```toml
[dependencies]
pravash = "1.1"
```

Default features: `sph`, `grid`, `shallow`.

Optional: `buoyancy`, `vortex`, `coupling`, `compute`, `parallel`, `ai`, `logging`.

MSRV: **1.89** (Rust edition 2024).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `sph` | yes | Smoothed Particle Hydrodynamics — particle-based fluids |
| `grid` | yes | Euler/Navier-Stokes grid-based solver |
| `shallow` | yes | Non-linear shallow water equations — 2D surface waves |
| `buoyancy` | no | Buoyancy, drag, Reynolds number, flow regime |
| `vortex` | no | Vorticity, Lamb-Oseen/Rankine vortices, enstrophy, Kolmogorov scale |
| `coupling` | no | Fluid-body interaction, FLIP/PIC hybrid, level set, added mass |
| `compute` | no | GPU-agnostic `ComputeBackend` trait, packed f32 buffers |
| `parallel` | no | Rayon parallelism for SPH and grid operations |
| `ai` | no | Daimon/hoosh integration (network deps) |
| `logging` | no | Structured logging + chrome://tracing profiling via `PRAVASH_LOG` |
| `full` | -- | Enables all features |

## Quick Start

```rust
use pravash::sph::{SphSolver, create_particle_block, total_kinetic_energy};
use pravash::common::{FluidConfig, FluidMaterial};

let mut particles = create_particle_block([0.2, 0.5], [0.3, 0.3], 0.02, 0.001);
let config = FluidConfig::water_2d();
let mut solver = SphSolver::new();

for _ in 0..100 {
    solver.step(&mut particles, &config, FluidMaterial::WATER.viscosity).unwrap();
}

println!("KE: {}", total_kinetic_energy(&particles));
```

## Modules

| Module | Description |
|--------|-------------|
| `sph` | SPH solver with spatial hash, PCISPH, DFSPH, Wendland/Delta-SPH kernels, adaptive timestep, multi-phase, viscoelastic (Oldroyd-B), non-Newtonian, implicit viscosity, heat/combustion, particle splitting/merging, foam/spray/bubble, MLS correction |
| `grid` | Navier-Stokes: MAC grid, semi-Lagrangian + MacCormack + BFECC advection, multigrid/DST/GS pressure, k-epsilon turbulence, Smagorinsky SGS, ghost fluid, periodic BCs. Rayon-parallelized |
| `shallow` | Non-linear SWE, HLL Riemann, Manning friction, wetting/drying, wave breaking, Green-Naghdi dispersion, Boussinesq, sediment transport, well-balanced terrain-following |
| `mpm` | Material Point Method — neo-Hookean, fluid, Drucker-Prager constitutive models |
| `vof` | Volume of Fluid — free surface tracking with donor-acceptor advection |
| `phase_field` | Allen-Cahn phase-field interface tracking with advection and double-well relaxation |
| `buoyancy` | Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime classification |
| `vortex` | Vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale |
| `coupling` | RigidBody interaction (sphere/box SDF), FLIP/PIC/APIC hybrid, narrow-band FLIP, particle-level set, added mass, field drag |
| `compute` | GPU-agnostic `ComputeBackend` trait, `NeuralCorrector`, `KernelDerivatives` with autodiff, `PackedParticles` f32 buffers |
| `common` | `FluidParticle` (phase, temperature, fuel, conformation tensor), `FluidMaterial`, `FluidConfig`, `ParticleSoa` (SOA), `ParticleArena` (memory pool) |

## Examples

| Example | Features | Description |
|---------|----------|-------------|
| [`basic_fluid`](examples/basic_fluid.rs) | `sph` | SPH dam break with SphSolver, kinetic energy tracking |
| [`shallow_water`](examples/shallow_water.rs) | `shallow` | Shallow water splash with Manning friction |
| [`multiphase`](examples/multiphase.rs) | `sph` | Water-oil two-phase simulation with interface tension |

Run an example:

```sh
cargo run --example basic_fluid --features sph
cargo run --example shallow_water --features shallow
cargo run --example multiphase --features sph
```

## Architecture

```
pravash (this crate)
  +-- hisab (math: vectors, spatial hash, FFT/DST, PDE solvers)
```

## Consumer Crates

| Crate | Usage |
|-------|-------|
| kiran | Water, smoke, fire effects |
| joshua | Fluid simulation |
| impetus | Fluid-body interaction |

## Documentation

- [Migration Guide](docs/development/migration.md) — breaking changes from 0.24.x
- [Roadmap](docs/development/roadmap.md) — upcoming features
- [Contributing](CONTRIBUTING.md) — workflow, code style, testing requirements
- [Security Policy](SECURITY.md) — reporting, supported versions

## Building

```sh
cargo build --all-features
cargo test --all-features
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
