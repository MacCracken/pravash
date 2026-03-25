# Pravash Roadmap

> Last updated: 2026-03-24

### 0.25.3 — Shallow Water & Waves

- [x] Non-linear shallow water equations (proper convective terms)
- [x] Bed friction (Manning's equation)
- [x] Wetting/drying (wet-dry transitions)
- [x] Wave breaking detection
- [x] Dispersive correction (Boussinesq-type)
- [x] Terrain-following coordinate system

### 0.26.3 — Performance & Parallelism

- [x] Rayon parallelism for SPH density/force loops
- [x] Rayon parallelism for grid operations
- [x] SIMD-friendly SOA layout option for FluidParticle
- [x] GPU-agnostic compute interface (`ComputeBackend` trait, packed buffers)
- [x] Memory pool / arena allocator for particle buffers
- [x] Profiling integration (tracing spans → flame graphs)

### 0.27.0 — Multi-phase & Advanced Physics

- [x] Multi-phase SPH (water-air interface, immiscible fluids)
- [x] Phase-field method for interface tracking
- [x] Viscoelastic fluids (Oldroyd-B model for honey/lava)
- [x] Heat transfer (conduction + convection)
- [x] Chemical reaction coupling (combustion for fire effects)

### Post-1.0 — Integration

- [ ] rasayan integration (chemistry crate crosstalk — reaction trait, species transport piping)
- [ ] soorat `ComputeBackend` implementation (GPU-accelerated SPH/grid via wgpu)

### 1.0.0 — Stable API

- [x] Public API review and stabilization
- [x] Complete documentation with examples for each module
- [x] Migration guide from 0.x
- [x] Performance regression test suite
- [ ] Published to crates.io
