# Architecture Overview

> Pravash — fluid dynamics simulation for AGNOS

## Dependency Stack

```
pravash
  └── hisab (math: DVec3, spatial hash, FFT/DST, PDE solvers, autodiff)
```

External: `serde`, `thiserror`, `tracing` (required); `rayon`, `reqwest`, `tokio`, `serde_json`, `tracing-subscriber`, `tracing-chrome` (optional, feature-gated).

## Module Map

| Module | Feature Gate | Role |
|--------|-------------|------|
| `common` | always | Core types: `FluidParticle`, `FluidMaterial`, `FluidConfig`, `ParticleSoa`, `ParticleArena` |
| `error` | always | `PravashError` enum (CFL violation, divergence, invalid params, grid errors) |
| `sph` | `sph` | Particle-based fluids — SPH, PCISPH, DFSPH, multi-phase, viscoelastic, non-Newtonian, heat, combustion, foam/spray/bubble |
| `grid` | `grid` | Grid-based Navier-Stokes — MAC grid, multigrid/DST/GS pressure, MacCormack/BFECC advection, k-epsilon turbulence |
| `shallow` | `shallow` | Shallow water equations — HLL Riemann, Manning friction, wetting/drying, wave breaking, Green-Naghdi dispersion, sediment transport |
| `phase_field` | `grid` | Allen-Cahn interface tracking |
| `vof` | `grid` | Volume of Fluid surface tracking |
| `buoyancy` | `buoyancy` | Archimedes buoyancy, drag, terminal velocity, Reynolds number |
| `vortex` | `vortex` | Vorticity, Lamb-Oseen/Rankine vortices, enstrophy, Kolmogorov scale |
| `coupling` | `coupling` | Fluid-body interaction, FLIP/PIC/APIC, level set, added mass, drag |
| `mpm` | `coupling` | Material Point Method — neo-Hookean, fluid, Drucker-Prager |
| `compute` | `compute` | GPU-agnostic `ComputeBackend` trait, `NeuralCorrector`, `KernelDerivatives`, autodiff |
| `logging` | `logging` | Structured logging + chrome://tracing profiling |
| `ai` | `ai` | Daimon/hoosh integration |

## Data Flow

```
Particles (FluidParticle / ParticleSoa / ParticleArena)
    │
    ├── sph::SphSolver.step()         ── neighbor query (hisab SpatialHash) → density → pressure → forces → integrate
    ├── sph::step_pcisph()            ── iterative density correction loop
    ├── sph::step_dfsph()             ── divergence-free two-stage correction
    ├── sph::step_multiphase()        ── per-phase EOS/viscosity with interface tension
    │
    ├── grid::FluidGrid.step()        ── advect → diffuse → add forces → project pressure
    │
    ├── shallow::ShallowWater.step()  ── flux computation → update height/velocity → friction → boundaries
    │
    ├── coupling::FlipSolver.step()   ── P2G → pressure solve → G2P blend
    ├── coupling::couple_sph_bodies() ── two-way particle-rigid body interaction
    │
    └── mpm::MpmSolver.step()         ── P2G → grid update → G2P → constitutive model
```

## Consumers

| Crate | Usage |
|-------|-------|
| kiran | Water, smoke, fire rendering effects |
| joshua | Fluid simulation |
| impetus | Fluid-body interaction |

## Design Principles

- **GPU-agnostic**: `ComputeBackend` trait abstracts GPU dispatch; soorat owns wgpu, pravash stays agnostic
- **Feature-gated**: consumers pull only needed modules
- **Zero unsafe**: all safe Rust
- **Arena allocation**: `ParticleArena` for pool-based particle management, `ParticleSoa` for SIMD-friendly SOA layout
- **DVec3 throughout**: all vectors use hisab's SIMD-accelerated `DVec3`
