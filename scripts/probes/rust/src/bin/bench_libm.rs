//! The Rust side of tests/pravash.bcyr's `math/*` rows (M1a part 2, docs/benchmarks-rust-v-cyrius.md):
//! Rust's std transcendentals, which call glibc (cbrt: compiler_builtins'), on the same 1,024 seeded
//! arguments per function, cycled.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_libm
//!
//! The arguments: an xorshift64 draw u in [0, 1) (the top 53 bits times 2^-53) mapped to lo + (hi − lo)·u,
//! the Cyrius bench's operations, so both sides time the same doubles. Each row is 1,000,000 calls, the
//! median of 7 runs, in ns per call; the loop's own cost (an identity body) is printed as the floor.

use std::hint::black_box;
use std::time::Instant;

const NS: usize = 1_000_000;
const RUNS: usize = 7;

struct Args {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn args(xlo: f64, xhi: f64, ylo: f64, yhi: f64) -> Args {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    };
    let mut x = Vec::with_capacity(1024);
    let mut y = Vec::with_capacity(1024);
    for _ in 0..1024 {
        let a = u();
        x.push(xlo + (xhi - xlo) * a);
        let b = u();
        y.push(ylo + (yhi - ylo) * b);
    }
    Args { x, y }
}

fn time(name: &str, a: &Args, f: impl Fn(f64, f64) -> f64) {
    let mut ns = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for i in 0..NS {
            let x = black_box(a.x[i & 1023]);
            let y = black_box(a.y[i & 1023]);
            sink ^= f(x, y).to_bits();
        }
        ns.push(t.elapsed().as_nanos() as f64 / NS as f64);
    }
    black_box(sink);
    ns.sort_by(f64::total_cmp);
    println!("  {name:<24} {:>8.2} ns", ns[RUNS / 2]);
}

fn main() {
    let a = args(-20.0, 20.0, 0.0, 1.0);
    time("floor (identity)", &a, |x, _| x);
    time("math/exp", &a, |x, _| x.exp());
    let a = args(0.001, 1000.0, 0.0, 1.0);
    time("math/ln", &a, |x, _| x.ln());
    time("math/log10", &a, |x, _| x.log10());
    let a = args(-10.0, 10.0, 0.0, 1.0);
    time("math/sin", &a, |x, _| x.sin());
    time("math/cos", &a, |x, _| x.cos());
    time("math/sin_cos", &a, |x, _| {
        let (s, c) = x.sin_cos();
        f64::from_bits(s.to_bits() ^ c.to_bits())
    });
    let a = args(-1000.0, 1000.0, 0.0, 1.0);
    time("math/cbrt", &a, |x, _| x.cbrt());
    let a = args(-100.0, 100.0, -100.0, 100.0);
    time("math/hypot", &a, |x, y| x.hypot(y));
    let a = args(0.1, 10.0, -10.0, 10.0);
    time("math/powf", &a, |x, y| x.powf(y));
    let a = args(-1.0, 1.0, -1.0, 1.0);
    // The Cyrius row is _pv_atan2(x, y) = atan2(y = x, x = y): Rust's x.atan2(y).
    time("math/atan2", &a, |x, y| x.atan2(y));
}
