//! The Rust side of tests/pravash.bcyr's `sph_m5d/*` rows (M5 part d, docs/benchmarks-rust-v-cyrius.md): the
//! multiphase step, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_m5d
//!
//! Each row is the median of 7 runs of 100 steps, in ns per step: one solver, the 25 × 25 block at spacing 0.02 (625
//! particles, as `sph_m5a/dense_625`) copied in before each step, `water_2d`.
//! - `multiphase_one_625`: one phase, water's (`MultiPhaseConfig::single`, water's viscosity);
//! - `multiphase_two_625`: water below y = 0.35 and a phase of rest density 800, gas constant 2000, viscosity 0.002
//!   above it, interface tension 0.05;
//! - `multiphase_two_625_st`: the two phases and the solver's free-surface tension 0.072 as well.

use std::hint::black_box;
use std::time::Instant;

use pravash::sph::{MultiPhaseConfig, PhaseProperties, SphSolver, create_particle_block};
use pravash::{FluidConfig, FluidMaterial};

const RUNS: usize = 7;

fn time(name: &str, iters: usize, mut body: impl FnMut() -> u64) {
    let mut ns = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for _ in 0..iters {
            sink ^= body();
        }
        ns.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    black_box(sink);
    ns.sort_by(f64::total_cmp);
    println!("  {name:<30} {:>12.2} ns", ns[RUNS / 2]);
}

fn main() {
    let mut block = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let mu = FluidMaterial::WATER.viscosity;
    let one = MultiPhaseConfig::single(config.rest_density, config.gas_constant, mu);
    let water = PhaseProperties {
        rest_density: config.rest_density,
        gas_constant: config.gas_constant,
        viscosity: mu,
    };
    let light = PhaseProperties {
        rest_density: 800.0,
        gas_constant: 2000.0,
        viscosity: 0.002,
    };
    let two = MultiPhaseConfig::two_phase(water, light, 0.05);
    let single = block.clone();
    for p in &mut block {
        if p.position[1] > 0.35 {
            p.phase = 1;
        }
    }
    for (name, src, mp, sigma) in [
        ("sph_m5d/multiphase_one_625", &single, &one, 0.0),
        ("sph_m5d/multiphase_two_625", &block, &two, 0.0),
        ("sph_m5d/multiphase_two_625_st", &block, &two, 0.072),
    ] {
        let mut solver = SphSolver::with_surface_tension(sigma);
        let mut ps = src.clone();
        time(name, 100, || {
            ps.copy_from_slice(black_box(src));
            solver
                .step_multiphase(&mut ps, &config, mp)
                .map_or(1, |()| 0)
        });
    }
}
