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

- [ ] Rayon parallelism for SPH density/force loops
- [ ] Rayon parallelism for grid operations
- [ ] SIMD-friendly SOA layout option for FluidParticle
- [ ] GPU compute via wgpu (optional feature)
- [ ] Memory pool / arena allocator for particle buffers
- [ ] Profiling integration (tracing spans → flame graphs)

### 0.27.0 — Multi-phase & Advanced Physics

- [ ] Multi-phase SPH (water-air interface, immiscible fluids)
- [ ] Phase-field method for interface tracking
- [ ] Viscoelastic fluids (Oldroyd-B model for honey/lava)
- [ ] Heat transfer (conduction + convection)
- [ ] Chemical reaction coupling (combustion for fire effects)

### 1.0.0 — Stable API

- [ ] Public API review and stabilization
- [ ] Complete documentation with examples for each module
- [ ] Migration guide from 0.x
- [ ] Performance regression test suite
- [ ] Published to crates.io
