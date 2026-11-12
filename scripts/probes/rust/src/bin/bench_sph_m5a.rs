//! The Rust side of tests/pravash.bcyr's `sph_m5a/*` rows (M5 part a, docs/benchmarks-rust-v-cyrius.md): the solver's
//! WCSPH step on a dense block, which rust-old/benches doesn't time (its `solver_step` and `solver_step_st` blocks sit at
//! spacing 1/√n against h 0.05, so few pairs interact), on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_m5a
//!
//! Each row is the median of 7 runs of `iters` steps, in ns per step: one solver, the 25 × 25 block at spacing 0.02
//! (625 particles, about 21 neighbours each at h 0.05) copied in before each step (the copy is in both sides' rows),
//! `water_2d`, water's viscosity. `dense_625` is symplectic Euler; `dense_625_st` adds surface tension 0.072;
//! `dense_625_verlet` integrates by velocity Verlet.

use std::hint::black_box;
use std::time::Instant;

use pravash::sph::{SphSolver, create_particle_block};
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
    let block = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let mu = FluidMaterial::WATER.viscosity;
    for (name, st, verlet) in [
        ("sph_m5a/dense_625", 0.0, false),
        ("sph_m5a/dense_625_st", 0.072, false),
        ("sph_m5a/dense_625_verlet", 0.0, true),
    ] {
        let mut solver = SphSolver::with_surface_tension(st);
        solver.use_verlet = verlet;
        let mut ps = block.clone();
        time(name, 300, || {
            ps.copy_from_slice(black_box(&block));
            solver.step(&mut ps, &config, mu).map_or(1, |()| 0)
        });
    }
}
