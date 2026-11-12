//! The Rust side of tests/pravash.bcyr's `sph_c/*` rows (M4 part c, docs/benchmarks-rust-v-cyrius.md): the part c
//! functions rust-old/benches doesn't time (its `sph_density`, `sph_step` and `sph_pressure_force` groups time the
//! rest), on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_c
//!
//! Each row is the median of 7 runs of `iters` calls, in ns per call. Arguments go through `black_box`:
//! - `viscosity_force_625`: particle 0 of the 25 × 25 block at spacing 0.02 (density 1000, pressure 100, as Rust's
//!   `sph_pressure_force` bench), h 0.05, μ 0.001;
//! - `create_particle_block_1000`: a 40 × 25 block (origin (0.1, 0.3), size (1.0, 0.625), spacing 0.025), freed
//!   per call;
//! - `split_particle_8`: one particle into 8 at offset 0.01, freed per call;
//! - `merge_particles`: two particles.

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use pravash::FluidParticle;
use pravash::sph::{create_particle_block, merge_particles, split_particle, viscosity_force};

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
    let mut block = create_particle_block([0.0, 0.0], [0.5, 0.5], 0.02, 0.001);
    for p in &mut block {
        p.density = 1000.0;
        p.pressure = 100.0;
    }
    time("sph_c/viscosity_force_625", 20_000, || {
        viscosity_force(
            black_box(0),
            black_box(&block),
            black_box(0.05),
            black_box(0.001),
        )[0]
        .to_bits()
    });
    time("sph_c/create_particle_block_1000", 2_000, || {
        let ps = create_particle_block(
            black_box([0.1, 0.3]),
            black_box([1.0, 0.625]),
            black_box(0.025),
            black_box(0.001),
        );
        ps.len() as u64 ^ ps[999].position.x.to_bits()
    });
    let mut parent = FluidParticle::new_2d(0.3, 0.7, 0.04);
    parent.velocity = DVec3::new(1.5, -0.5, 0.25);
    time("sph_c/split_particle_8", 200_000, || {
        let ch = split_particle(black_box(&parent), black_box(8), black_box(0.01));
        ch[7].position.y.to_bits()
    });
    let mut a = FluidParticle::new_2d(0.1, 0.2, 0.01);
    a.velocity = DVec3::new(1.0, 2.0, 0.0);
    let mut b = FluidParticle::new_2d(0.3, 0.1, 0.03);
    b.velocity = DVec3::new(-1.0, 0.5, 0.0);
    time("sph_c/merge_particles", 1_000_000, || {
        merge_particles(black_box(&a), black_box(&b))
            .position
            .x
            .to_bits()
    });
}
