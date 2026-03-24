# Pravash

> **Pravash** (Sanskrit: प्रवास — journey, flow) — fluid dynamics simulation for AGNOS

Particle-based and grid-based fluid simulation. SPH for real-time effects, Euler/Navier-Stokes for accurate simulation, shallow water for surface waves. Built on [hisab](https://crates.io/crates/hisab) for math foundations.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `sph` | yes | Smoothed Particle Hydrodynamics — particle-based fluids |
| `grid` | yes | Euler/Navier-Stokes grid-based solver |
| `shallow` | yes | Shallow water equations — 2D surface waves |
| `buoyancy` | no | Buoyancy, drag, Reynolds number, flow regime |
| `vortex` | no | Vorticity, Lamb-Oseen/Rankine vortices, enstrophy, Kolmogorov scale |
| `ai` | no | Daimon/hoosh integration |
| `logging` | no | Structured logging via `PRAVASH_LOG` |

## Modules

| Module | Description |
|--------|-------------|
| `sph` | SPH kernels (Poly6, Spiky, Viscosity), density/pressure computation, force calculation, particle stepping |
| `grid` | FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion |
| `shallow` | ShallowWater heightfield with disturbances, wave propagation, volume conservation |
| `buoyancy` | Archimedes buoyancy, drag force, terminal velocity, Reynolds number |
| `vortex` | Vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale |
| `common` | FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig |

## Quick Start

```rust
use pravash::sph::{create_particle_block, step};
use pravash::common::{FluidConfig, FluidMaterial};

let mut particles = create_particle_block([0.2, 0.5], [0.3, 0.3], 0.02, 0.001);
let config = FluidConfig::water_2d();

for _ in 0..100 {
    step(&mut particles, &config, FluidMaterial::WATER.viscosity).unwrap();
}
```

## Roadmap

See [docs/development/roadmap.md](docs/development/roadmap.md).

## License

GPL-3.0 — see [LICENSE](LICENSE).
