//! The Rust side of tests/pravash.bcyr's `vof_advect/*` and `phase_field_step/*` rows (M3 part d,
//! docs/benchmarks-rust-v-cyrius.md): rust-old/benches has no bench for `VofField` or `PhaseField`, so these twins
//! time the same bodies on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_vof_pf
//!
//! Each row is the median of 7 runs of `iters` calls on one state, in µs per call (the state evolves the same way
//! on both sides, bit for bit):
//! - `vof_advect/n`: an n × n field at dx 0.1 with the rectangle [0.2·L, 0.6·L]² filled (L = n·dx), in the
//!   rotation u = −0.1·(y − c), v = 0.1·(x − c) about the centre c, dt 0.01;
//! - `phase_field_step/n`: an n × n field at dx 0.1 with a disc of radius 0.25·L at the centre set to phase B, in
//!   the same rotation, dt 0.001.

use std::hint::black_box;
use std::time::Instant;

use pravash::phase_field::PhaseField;
use pravash::vof::VofField;

const RUNS: usize = 7;

fn time(name: &str, iters: usize, mut body: impl FnMut() -> u64) {
    let mut us = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for _ in 0..iters {
            sink ^= body();
        }
        us.push(t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0);
    }
    black_box(sink);
    us.sort_by(f64::total_cmp);
    println!("  {name:<24} {:>10.2} us", us[RUNS / 2]);
}

fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

fn rotation(n: usize) -> (Vec<f64>, Vec<f64>) {
    let c = (fx(n) - 1.0) * 0.5;
    let vx = (0..n * n).map(|i| -(fx(i / n) - c) * 0.1).collect();
    let vy = (0..n * n).map(|i| (fx(i % n) - c) * 0.1).collect();
    (vx, vy)
}

fn main() {
    for (n, iters) in [(32, 2000), (64, 500), (128, 100)] {
        let l = fx(n) * 0.1;
        let mut v = VofField::new(n, n, 0.1).expect("valid");
        v.fill_rect(0.2 * l, 0.2 * l, 0.6 * l, 0.6 * l);
        let (vx, vy) = rotation(n);
        time(&format!("vof_advect/{n}x{n}"), iters, || {
            u64::from(black_box(&mut v).advect(&vx, &vy, 0.01).is_ok())
        });
    }
    for (n, iters) in [(32, 2000), (64, 500), (128, 100)] {
        let l = fx(n) * 0.1;
        let mut p = PhaseField::new(n, n, 0.1).expect("valid");
        p.set_circle(0.5 * l, 0.5 * l, 0.25 * l);
        let (vx, vy) = rotation(n);
        time(&format!("phase_field_step/{n}x{n}"), iters, || {
            u64::from(black_box(&mut p).step(&vx, &vy, 0.001).is_ok())
        });
    }
}
