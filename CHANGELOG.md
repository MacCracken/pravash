# Changelog

## [0.1.0] - 2026-03-23

### Added
- common: FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig with CFL validation
- sph: Poly6/Spiky/Viscosity kernels, density/pressure computation, pressure + viscosity forces, particle stepping with boundary enforcement, particle block creation
- grid: FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion, kinetic energy
- shallow: ShallowWater heightfield, circular disturbances, wave propagation, volume conservation
- buoyancy: Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime classification
- vortex: 2D vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale
- error: PravashError with #[non_exhaustive], CFL violation detection
