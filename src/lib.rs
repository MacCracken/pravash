//! Pravash — Fluid dynamics simulation for AGNOS
//!
//! Sanskrit: प्रवास (pravash) — journey, flow
//!
//! Particle-based and grid-based fluid simulation. SPH for real-time effects,
//! Euler/Navier-Stokes for accurate simulation, shallow water for surface waves.
//! Built on [hisab](https://crates.io/crates/hisab) for math foundations.
//!
//! # Modules
//!
//! - [`sph`] — Smoothed Particle Hydrodynamics (particle-based fluids)
//! - [`grid`] — Euler/Navier-Stokes grid-based solver
//! - [`shallow`] — Shallow water equations (2D surface waves)
//! - [`buoyancy`] — Buoyancy and drag forces
//! - [`vortex`] — Vortex dynamics and turbulence
//! - [`common`] — Shared types: FluidParticle, FluidConfig, material properties
//! - [`error`] — Error types

pub mod common;
pub mod error;

#[cfg(feature = "sph")]
pub mod sph;

#[cfg(feature = "grid")]
pub mod grid;

#[cfg(feature = "shallow")]
pub mod shallow;

#[cfg(feature = "buoyancy")]
pub mod buoyancy;

#[cfg(feature = "vortex")]
pub mod vortex;

#[cfg(feature = "coupling")]
pub mod coupling;

#[cfg(feature = "logging")]
pub mod logging;

#[cfg(feature = "ai")]
pub mod ai;

pub use common::{FluidConfig, FluidMaterial, FluidParticle};
pub use error::PravashError;
