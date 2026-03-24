# Pravash Roadmap

> Last updated: 2026-03-24

## Version Scheme

`0.D.M` pre-1.0 — D = development milestone, M = minor within milestone.

## Milestones

### 0.1 — Foundation (complete)

Core types, basic solvers, project scaffolding.

- [x] `common`: FluidParticle, FluidMaterial (5 presets), FluidConfig with CFL
- [x] `sph`: Poly6/Spiky/Viscosity kernels, density/pressure, pressure+viscosity forces, symplectic Euler, boundary enforcement
- [x] `grid`: FluidGrid, Gauss-Seidel diffusion, kinetic energy
- [x] `shallow`: ShallowWater heightfield, disturbances, wave propagation, volume conservation
- [x] `buoyancy`: Archimedes force, drag, terminal velocity, Reynolds, flow regime
- [x] `vortex`: 2D vorticity, Lamb-Oseen/Rankine, enstrophy, Kolmogorov scale
- [x] `error`: PravashError with `#[non_exhaustive]`, Cow strings
- [x] P(-1) scaffold hardening: benchmarks, audit, 2-3x SPH speedup

### 0.2 — SPH Acceleration & Surface Tension (complete)

Unlock SPH for real workloads. Required by kiran for water effects.

- [x] **SphSolver** with hisab `SpatialHash` — O(n·k) neighbor queries via `query_cell`
- [x] Neighbor list caching (flat buffer + offset table, single pass, zero per-particle alloc)
- [x] Persistent scratch buffers (densities, snapshot, positions_f32, neighbor cache)
- [x] Surface tension (CSF model with poly6 gradient/laplacian)
- [x] Symmetric pressure formula for momentum conservation
- [x] f64→f32 precision handling, h validation
- [ ] PCISPH or DFSPH pressure solver (density-invariant incompressibility)
- [ ] Adaptive timestep (CFL-driven dt adjustment)

### 0.3 — Grid Navier-Stokes (complete)

Complete the Euler grid pipeline. Required by joshua for fluid simulation.

- [x] Semi-Lagrangian advection (bilinear, velocity + density)
- [x] Pressure projection via GS Poisson solver with warm start
- [x] Divergence-free velocity enforcement
- [x] Boundary conditions: no-slip, free-slip
- [x] Vorticity confinement (2D, applied to grid velocity)
- [x] Buoyancy-driven density advection (smoke, fire)
- [x] `GridConfig` with `smoke()` preset, `Serialize`/`Deserialize`
- [x] Correct dx scaling in advection, diffusion, divergence, projection
- [ ] MacCormack advection (higher-order, less diffusive)
- [ ] Periodic / inflow-outflow boundary conditions
- [ ] FFT Poisson solver (requires DST in hisab for wall boundaries)

### 0.4 — Coupling & Interaction

Bridge particle and grid worlds. Required by impetus for fluid-body interaction.

- [ ] Two-way SPH-rigid body coupling (force exchange via buoyancy module)
- [ ] FLIP/PIC hybrid (particles on grid for low-dissipation advection)
- [ ] Particle-level set surface tracking
- [ ] Added mass computation for submerged bodies
- [ ] Drag integration with velocity fields (not just freestream)

### 0.5 — Shallow Water & Waves

Production-grade surface simulation for kiran.

- [ ] Non-linear shallow water equations (proper convective terms)
- [ ] Bed friction (Manning's equation)
- [ ] Wetting/drying (wet-dry transitions)
- [ ] Wave breaking detection
- [ ] Dispersive correction (Boussinesq-type)
- [ ] Terrain-following coordinate system

### 0.6 — Performance & Parallelism

Scale to production workloads.

- [ ] Rayon parallelism for SPH density/force loops
- [ ] Rayon parallelism for grid operations
- [ ] SIMD-friendly SOA layout option for FluidParticle
- [ ] GPU compute via wgpu (optional feature)
- [ ] Memory pool / arena allocator for particle buffers
- [ ] Profiling integration (tracing spans → flame graphs)

### 0.7 — Multi-phase & Advanced Physics

- [ ] Multi-phase SPH (water-air interface, immiscible fluids)
- [ ] Phase-field method for interface tracking
- [ ] Viscoelastic fluids (Oldroyd-B model for honey/lava)
- [ ] Heat transfer (conduction + convection)
- [ ] Chemical reaction coupling (combustion for fire effects)

### 1.0 — Stable API

- [ ] Public API review and stabilization
- [ ] Complete documentation with examples for each module
- [ ] Migration guide from 0.x
- [ ] Performance regression test suite
- [ ] Published to crates.io

## Consumer Dependency Map

| Consumer | Needs | Milestone |
|----------|-------|-----------|
| **kiran** (water/smoke/fire) | Performant SPH, grid smoke, surface tension | 0.2, 0.3, 0.5 |
| **joshua** (fluid simulation) | Complete Navier-Stokes, pressure projection | 0.3 |
| **impetus** (fluid-body) | Two-way coupling, added mass, drag fields | 0.4 |

## hisab Utilization Plan

| hisab feature | pravash use | milestone |
|---------------|-------------|-----------|
| `SpatialHash` | SPH neighbor queries | 0.2 |
| `FFT` (2D) | Grid pressure Poisson solver | 0.3 |
| `RK4` | Higher-order time integration | 0.3 |
| `KdTree` | Particle surface reconstruction | 0.4 |
| `GJK/EPA` | Fluid-body collision detection | 0.4 |

## Principles

- Every milestone ships with benchmarks proving the wins
- No milestone ships without 80%+ test coverage on new code
- Performance-critical code benchmarked before and after
- hisab first — use AGNOS math crate before reaching for external deps
