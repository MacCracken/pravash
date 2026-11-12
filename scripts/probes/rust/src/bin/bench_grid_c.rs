//! The Rust side of tests/pravash.bcyr's `mac_step/*`, `kepsilon_step/*` and `ghost_fluid/*` rows (M3 part c,
//! docs/benchmarks-rust-v-cyrius.md): rust-old/benches has no bench for `MacGrid`, `KEpsilon` or
//! `apply_ghost_fluid`, so these twins time the same bodies on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_grid_c
//!
//! Each row is the median of 7 runs of `iters` calls on one state, in µs per call (the state evolves the same way
//! on both sides, bit for bit, so both time the same work):
//! - `mac_step/n`: an n × n MAC grid at dx 0.1, density 1 on a 4 × 2 blob at the floor's centre, the smoke config;
//! - `kepsilon_step/n`: `KEpsilon::new(n, n)` under the solid-body rotation u = −0.5·(y − c), v = 0.5·(x − c),
//!   dx 0.1, dt 0.01;
//! - `ghost_fluid/n`: a zero pressure and a circle's signed distance (radius n·dx/4, centre (n/2 + 0.3, n/2 − 0.2)),
//!   dx 0.1, σ 0.07.

use std::hint::black_box;
use std::time::Instant;

use pravash::grid::{GridConfig, KEpsilon, MacGrid, apply_ghost_fluid};

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

fn main() {
    let cfg = GridConfig::smoke();
    for (n, iters) in [(16, 2000), (32, 500), (64, 100)] {
        let mut g = MacGrid::new(n, n, 0.1).expect("valid");
        for x in n / 2 - 2..n / 2 + 2 {
            g.density[n + x] = 1.0;
            g.density[2 * n + x] = 1.0;
        }
        time(&format!("mac_step/{n}x{n}"), iters, || {
            u64::from(black_box(&mut g).step(&cfg).is_ok())
        });
    }
    for (n, iters) in [(32, 2000), (64, 500), (128, 100)] {
        let mut ke = KEpsilon::new(n, n);
        let c = (fx(n) - 1.0) * 0.5;
        let vx: Vec<f64> = (0..n * n).map(|i| -(fx(i / n) - c) * 0.5).collect();
        let vy: Vec<f64> = (0..n * n).map(|i| (fx(i % n) - c) * 0.5).collect();
        time(&format!("kepsilon_step/{n}x{n}"), iters, || {
            black_box(&mut ke).step(&vx, &vy, n, n, 0.1, 0.01);
            ke.k[n + 1].to_bits()
        });
    }
    for (n, iters) in [(32, 5000), (64, 1000), (128, 200)] {
        let (cx, cy, r) = (fx(n) * 0.5 + 0.3, fx(n) * 0.5 - 0.2, fx(n) * 0.1 * 0.25);
        let ls: Vec<f64> = (0..n * n)
            .map(|i| {
                let (x, y) = (fx(i % n) - cx, fx(i / n) - cy);
                (x * x + y * y).sqrt() * 0.1 - r
            })
            .collect();
        let mut p = vec![0.0; n * n];
        time(&format!("ghost_fluid/{n}x{n}"), iters, || {
            apply_ghost_fluid(black_box(&mut p), &ls, n, n, 0.1, 0.07);
            p[n * n / 2].to_bits()
        });
    }
}
