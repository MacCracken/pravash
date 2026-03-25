# Pravash Roadmap

> Last updated: 2026-03-25

### 0.25.3 — Integration & Research

- [x] kimiya integration (ReactionProvider trait — no-vendor-lock-in crosstalk)
- [x] soorat integration (ComputeBackend trait — GPU-agnostic, no wgpu dependency)
- [x] Deep research: thermodynamic formulas, breakthroughs, new theories to watch

### 0.26.0 — Numerical Quality & Correctness

- [x] Adaptive CFL timestep (auto dt for SPH, grid, shallow water)
- [x] Wendland C2/C4 kernels (drop-in Poly6 replacement, no tensile instability)
- [x] Delta-SPH density diffusion (pressure noise reduction)
- [x] Velocity Verlet integration (symplectic 2nd order, better energy conservation)
- [x] Power-law / Bingham / Herschel-Bulkley viscosity (mud, blood, paint, lava)
- [x] Smagorinsky SGS turbulence model (LES effective viscosity)
- [x] Z-order (Morton code) particle sorting for cache-friendly neighbor access

### 0.27.0 — Algorithmic Leap

- [x] DFSPH solver (divergence-free SPH, 10-50x faster than PCISPH)
- [x] Multigrid V-cycle pressure solver (O(N) grid pressure)
- [x] HLL/HLLC Riemann solver for shallow water (proper shock capturing)
- [x] APIC transfers (affine particle-in-cell, replaces FLIP/PIC blend)
- [x] MLS gradient correction (1st-order consistent SPH near boundaries)
- [x] Gradient correction matrices for free surface accuracy

### 0.28.0 — Architecture & New Physics

- [ ] Staggered MAC grid (eliminate pressure checkerboarding)
- [ ] VOF / CLSVOF free surface tracking
- [ ] Contact angle / wetting (droplets, meniscus, capillary)
- [ ] Phase change (evaporation, condensation, melting, solidification)
- [ ] Ghost fluid method for multi-phase grid interface
- [ ] Sediment transport (bed load, suspended, erosion/deposition)

### 0.29.0 — Advanced Methods

- [ ] MPM extension (material point method — unified multi-material)
- [ ] Compressible Navier-Stokes (Tait EOS, energy equation, shockwaves)
- [ ] IMEX time splitting (implicit stiff terms, explicit advection)
- [ ] Adaptive particle splitting / merging
- [ ] Narrow-band FLIP (surface-only particles, grid interior)
- [ ] SIMD kernel evaluation (f64x4 / f32x8 vectorized inner loops)

### 1.0.0 — Production

- [ ] Neural operator acceleration (learned correctors, FNO for ambient effects)
- [ ] Differentiable simulation API (analytical Jacobians)
- [ ] k-epsilon / k-omega SST turbulence models
- [ ] Green-Naghdi fully nonlinear dispersion
- [ ] Foam / spray / bubble generation heuristics
- [ ] BFECC / reflection-based advection
