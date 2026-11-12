//! The Rust side of tests/pravash.bcyr's `sph_m5e/*` rows (M5 part e, docs/benchmarks-rust-v-cyrius.md): the solver's
//! three `parallel` loops at a thread count, on the frozen pravash 1.3.0. Build it with rayon and set the count:
//!
//!   RAYON_NUM_THREADS=4 taskset -c 8-15 cargo run --release --locked --features parallel \
//!       --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_m5e
//!
//! Each row is the median of 7 runs of `iters` steps, in ns per step: one solver with surface tension 0.072, the block
//! copied in before each step, `water_2d`, water's viscosity.
//! - `step_625`: the 25 × 25 block at spacing 0.02 (as `sph_m5a/dense_625_st`), 100 steps a run;
//! - `multiphase_625`: the same block, the particles above y = 0.35 in a phase of rest density 800, gas constant
//!   2000 and viscosity 0.002, interface tension 0.05 (as `sph_m5d/multiphase_two_625_st`), 100 steps a run;
//! - `step_2500`: the 50 × 50 block at spacing 0.01 (about 78 neighbours a particle at h 0.05), 20 steps a run.

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
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    println!("  RAYON_NUM_THREADS={threads}");
    let config = FluidConfig::water_2d();
    let mu = FluidMaterial::WATER.viscosity;
    let dense = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let fine = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.01, 0.001);
    let mut layered = dense.clone();
    for p in &mut layered {
        if p.position[1] > 0.35 {
            p.phase = 1;
        }
    }
    let two = MultiPhaseConfig::two_phase(
        PhaseProperties {
            rest_density: config.rest_density,
            gas_constant: config.gas_constant,
            viscosity: mu,
        },
        PhaseProperties {
            rest_density: 800.0,
            gas_constant: 2000.0,
            viscosity: 0.002,
        },
        0.05,
    );
    for (name, src, iters) in [
        ("sph_m5e/step_625", &dense, 100),
        ("sph_m5e/step_2500", &fine, 20),
    ] {
        let mut solver = SphSolver::with_surface_tension(0.072);
        let mut ps = src.clone();
        time(name, iters, || {
            ps.copy_from_slice(black_box(src));
            solver.step(&mut ps, &config, mu).map_or(1, |()| 0)
        });
    }
    let mut solver = SphSolver::with_surface_tension(0.072);
    let mut ps = layered.clone();
    time("sph_m5e/multiphase_625", 100, || {
        ps.copy_from_slice(black_box(&layered));
        solver
            .step_multiphase(&mut ps, &config, &two)
            .map_or(1, |()| 0)
    });
}
