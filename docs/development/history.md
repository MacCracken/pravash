# Pravash Development History

## Completed Milestones

### 0.1 — Foundation (2026-03-23)
Core types, basic solvers, project scaffolding. P(-1) scaffold hardening with benchmarks, audits, 2-3x SPH speedup.

### 0.2 — SPH Acceleration & Surface Tension (2026-03-24)
SphSolver with spatial hash (hisab SpatialHash), neighbor caching, persistent buffers, CSF surface tension, symmetric pressure, PCISPH, adaptive timestep.

### 0.3 — Grid Navier-Stokes (2026-03-24)
Semi-Lagrangian + MacCormack advection, GS + DST pressure projection, no-slip/free-slip/periodic boundaries, vorticity confinement, buoyancy-driven flow, correct dx scaling.

### 0.4 — Coupling & Interaction (2026-03-24)
RigidBody with SDF shapes, two-way SPH coupling, FLIP/PIC hybrid, particle-level set, added mass, field-based drag.

## Consumer Dependency Map

| Consumer | What it uses | Milestones |
|----------|-------------|-----------|
| **kiran** (water/smoke/fire) | SPH, grid smoke, surface tension | 0.2, 0.3, 0.5 |
| **joshua** (fluid simulation) | Navier-Stokes, pressure projection | 0.3 |
| **impetus** (fluid-body) | Coupling, added mass, drag | 0.4 |

## hisab Utilization

| hisab feature | pravash use | status |
|---------------|-------------|--------|
| `SpatialHash` | SPH neighbor queries | in use |
| `dst` / `idst` | Grid DST Poisson solver | in use |
| `KdTree` | Particle surface reconstruction | available |
| `GJK/EPA` | Fluid-body collision (2D) | available |
| `RK4` | Higher-order time integration | available |
| `FFT` | Periodic grid pressure solver | available |

## Principles

- Every milestone ships with benchmarks proving the wins
- No milestone ships without 80%+ test coverage on new code
- Performance-critical code benchmarked before and after
- hisab first — use AGNOS math crate before reaching for external deps
