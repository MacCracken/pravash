//! The Rust side of tests/pravash.bcyr's `sph_m5c/*` rows (M5 part c, docs/benchmarks-rust-v-cyrius.md): the DFSPH
//! step, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_m5c
//!
//! Each row is the median of 7 runs of `iters` steps, in ns per step: one solver, the block copied in before each
//! step, `water_2d`, water's viscosity, at most 3 rounds of each solver.
//! - `dfsph_sparse_64`: the `solver_step` bench's 64-particle block (origin (0.1, 0.3), size 0.5, spacing 1/15),
//!   to 1%;
//! - `dfsph_dense_625`: the 25 × 25 block at spacing 0.02, to 1%; `dfsph_dense_625_3it`: a threshold of 0, so every
//!   round runs.

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
    let sparse = create_particle_block([0.1, 0.3], [0.5, 0.5], 1.0 / 15.0, 0.001);
    let dense = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let mu = FluidMaterial::WATER.viscosity;
    for (name, block, max_err, iters) in [
        ("sph_m5c/dfsph_sparse_64", &sparse, 0.01, 2000),
        ("sph_m5c/dfsph_dense_625", &dense, 0.01, 50),
        ("sph_m5c/dfsph_dense_625_3it", &dense, 0.0, 50),
    ] {
        let mut solver = SphSolver::new();
        let mut ps = block.clone();
        time(name, iters, || {
            ps.copy_from_slice(black_box(block));
            solver
                .step_dfsph(&mut ps, &config, mu, 3, max_err)
                .map_or(1, |()| 0)
        });
    }
}
