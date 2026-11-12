//! The Rust side of tests/pravash.bcyr's `sph_a/*` rows (M4 part a, docs/benchmarks-rust-v-cyrius.md): the part a
//! functions rust-old/benches doesn't time (its `sph_kernels` group times poly6, spiky_grad and viscosity_laplacian),
//! on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_a
//!
//! Each row is the median of 7 runs of `iters` calls, in ns per call. Arguments go through `black_box`:
//! - the Wendland kernels at (0.3, 1.0); the Tait EOS at (1010, 1000, 1480, 7) and its γ → 0 branch (γ = 0, ln);
//!   the power law (K 0.5, n 0.7) at a strain rate of 2;
//! - `batch_poly6` over 1,000 squared distances in [0, 1.2) at h 1; `total_kinetic_energy` and `max_speed` over
//!   1,000 particles (ns per call of the whole batch).

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use pravash::FluidParticle;
use pravash::sph::{
    NonNewtonianViscosity, batch_poly6, equation_of_state_tait, kernel_wendland_c2,
    kernel_wendland_c4_grad, max_speed, total_kinetic_energy,
};

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
    println!("  {name:<28} {:>10.2} ns", ns[RUNS / 2]);
}

fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

fn main() {
    time("sph_a/wendland_c2", 10_000_000, || {
        kernel_wendland_c2(black_box(0.3), black_box(1.0)).to_bits()
    });
    time("sph_a/wendland_c4_grad", 10_000_000, || {
        kernel_wendland_c4_grad(black_box(0.3), black_box(1.0)).to_bits()
    });
    time("sph_a/tait", 1_000_000, || {
        equation_of_state_tait(
            black_box(1010.0),
            black_box(1000.0),
            black_box(1480.0),
            black_box(7.0),
        )
        .to_bits()
    });
    time("sph_a/tait_gamma0", 1_000_000, || {
        equation_of_state_tait(
            black_box(1010.0),
            black_box(1000.0),
            black_box(1480.0),
            black_box(0.0),
        )
        .to_bits()
    });
    let nn = NonNewtonianViscosity::PowerLaw {
        consistency: 0.5,
        power_index: 0.7,
    };
    time("sph_a/power_law", 1_000_000, || {
        black_box(&nn).effective_viscosity(black_box(2.0)).to_bits()
    });
    let r2: Vec<f64> = (0..1000).map(|k| fx(k) * 0.0012).collect();
    time("sph_a/batch_poly6_1000", 20_000, || {
        let w = batch_poly6(black_box(&r2), black_box(1.0));
        w[999].to_bits()
    });
    let ps: Vec<FluidParticle> = (0..1000)
        .map(|k| {
            let mut p = FluidParticle::new_2d(fx(k % 40) * 0.025, fx(k / 40) * 0.025, 0.001);
            p.velocity = DVec3::new(fx(k % 7) * 0.1, -fx(k % 5) * 0.2, 0.0);
            p
        })
        .collect();
    time("sph_a/total_kinetic_energy_1000", 100_000, || {
        total_kinetic_energy(black_box(&ps)).to_bits()
    });
    time("sph_a/max_speed_1000", 100_000, || {
        max_speed(black_box(&ps)).to_bits()
    });
}
