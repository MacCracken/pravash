//! The Rust side of tests/pravash.bcyr's `debug_d/*` rows (M8 part d, docs/benchmarks-rust-v-cyrius.md) on the frozen
//! pravash 1.3.0.
//!
//!   taskset -c 15 cargo +1.99.1 run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_debug_d
//!
//! Each row is the median of 7 runs, in ns per iteration, a fresh `String` an iteration as `format!` makes:
//! - `debug_d/f64_mix`: `{:?}` of eight f64s, positional and exponential, short and 17-digit, one subnormal-edge;
//! - `debug_d/particle`, `debug_d/particle_pretty`: `{:?}` and `{:#?}` of `FluidParticle::new_2d(0.5, 0.25, 0.001)`;
//! - `debug_d/grid_8x8`: `{:?}` of an 8 × 8 `FluidGrid` after one multigrid step (its scratch and levels filled).

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use pravash::FluidParticle;
use pravash::grid::{FluidGrid, GridConfig};

const RUNS: usize = 7;

fn time(name: &str, iters: usize, mut body: impl FnMut(usize) -> u64) {
    let mut ns = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for k in 0..iters {
            sink ^= body(k);
        }
        #[allow(clippy::cast_precision_loss)]
        ns.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    black_box(sink);
    ns.sort_by(f64::total_cmp);
    println!("  {name:<30} {:>12.2} ns", ns[RUNS / 2]);
}

const MIX: [f64; 8] = [0.1, 293.15, 1e-7, 2.2250738585072014e-308, 123_456.789, -6.478909558163377, 1e16, 0.001];

fn main() {
    time("debug_d/f64_mix", 200_000, |_| {
        let mut s = String::new();
        for x in black_box(MIX) {
            let _ = write!(s, "{x:?}");
        }
        s.len() as u64
    });
    let p = FluidParticle::new_2d(0.5, 0.25, 0.001);
    time("debug_d/particle", 200_000, |_| format!("{:?}", black_box(&p)).len() as u64);
    time("debug_d/particle_pretty", 200_000, |_| format!("{:#?}", black_box(&p)).len() as u64);
    let mut g = FluidGrid::new(8, 8, 0.1).expect("a valid grid");
    for k in 0..64 {
        #[allow(clippy::cast_precision_loss)]
        let x = k as f64;
        g.vx[k] = (x * 0.37).fract() - 0.5;
        g.vy[k] = (x * 0.61).fract() - 0.5;
        g.density[k] = (x * 0.13).fract();
    }
    let mut c = GridConfig::default();
    c.use_multigrid = true;
    let _ = g.step(&c);
    time("debug_d/grid_8x8", 2_000, |_| format!("{:?}", black_box(&g)).len() as u64);
}
