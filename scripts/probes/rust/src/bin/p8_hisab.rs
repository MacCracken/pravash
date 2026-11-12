//! Probe 8, the Rust side (port plan §4 step 9.8): reference values from hisab 1.4.0, exactly as the
//! oracle calls it, for the three routines the port would otherwise take from hisab 3.4.1 or ganita:
//! - `num::dst` / `num::idst`, through which the grid's DST pressure solve runs;
//! - `num::matrix_inverse` on 3×3 matrices, as the MLS gradient correction calls it (the oracle's
//!   `invert_3x3` hook);
//! - `autodiff::reverse_gradient`, as `KernelDerivatives::autodiff_gradient` calls it.
//!
//!   cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin p8_hisab -- build/p8.bin
//!
//! The file is a stream of little-endian 64-bit words, one record after another:
//! - DST:  1, n, x[n], dst(x)[n];
//! - IDST: 2, n, x[n], idst(x)[n];
//! - inverse: 3, 9, m[9], ok (1 or 0), inv[9] (zeros when ok is 0);
//! - gradient: 4, function id, x[3], grad[3].
//!
//! `scripts/probes/p8_hisab.tcyr` replays every record through the Cyrius candidates.

use std::io::Write as _;

use hisab::autodiff::{Tape, Var, reverse_gradient};
use hisab::num::{dst, idst};

struct Out(Vec<u8>);

impl Out {
    fn w(&mut self, x: u64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn f(&mut self, x: f64) {
        self.w(x.to_bits());
    }
}

/// xorshift64*.
fn next(s: &mut u64) -> f64 {
    let mut x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

fn f0(t: &mut Tape, x: &[Var]) -> Var {
    // x0·x1 + x2·x2·x0 − x1 / x2: arithmetic only.
    let a = t.mul(x[0], x[1]);
    let b = t.mul(x[2], x[2]);
    let c = t.mul(b, x[0]);
    let d = t.div(x[1], x[2]);
    let e = t.add(a, c);
    t.sub(e, d)
}

fn f1(t: &mut Tape, x: &[Var]) -> Var {
    // sin(x0)·exp(x1) + x2: through the tape's transcendentals.
    let s = t.sin(x[0]);
    let e = t.exp(x[1]);
    let p = t.mul(s, e);
    t.add(p, x[2])
}

fn main() {
    let path = std::env::args().nth(1).expect("output path");
    let mut o = Out(Vec::new());
    let mut s = 0x8888_1234_5678_9abc;
    let (mut n_dst, mut n_inv, mut n_ad) = (0, 0, 0);

    for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 14, 30, 62, 126] {
        for _ in 0..4 {
            let x: Vec<f64> = (0..n).map(|_| next(&mut s) * 2.0 - 1.0).collect();
            for (kind, y) in [(1u64, dst(&x).expect("dst")), (2, idst(&x).expect("idst"))] {
                o.w(kind);
                o.w(n as u64);
                x.iter().for_each(|v| o.f(*v));
                y.iter().for_each(|v| o.f(*v));
                n_dst += 1;
            }
        }
    }

    for k in 0..300 {
        // Symmetric positive definite A·Aᵀ + εI, the shape of an MLS moment matrix; every 10th
        // nearly singular; every 25th with a zero z row and column, as 2D SPH gives (singular).
        let mut a = [0.0f64; 9];
        for v in &mut a {
            *v = next(&mut s) * 2.0 - 1.0;
        }
        let mut m = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 3 + c] = (0..3).map(|q| a[r * 3 + q] * a[c * 3 + q]).sum::<f64>();
            }
            m[r * 4] += if k % 10 == 0 { 1e-12 } else { 0.1 };
        }
        if k % 25 == 0 {
            for q in 0..3 {
                m[2 * 3 + q] = 0.0;
                m[q * 3 + 2] = 0.0;
            }
        }
        let inv = pravash::oracle::invert_3x3(&m);
        o.w(3);
        o.w(9);
        m.iter().for_each(|v| o.f(*v));
        o.w(u64::from(inv.is_some()));
        inv.unwrap_or([0.0; 9]).iter().for_each(|v| o.f(*v));
        n_inv += 1;
    }

    for fid in 0..2u64 {
        for _ in 0..100 {
            let x = [next(&mut s) * 4.0 - 2.0, next(&mut s) * 2.0 - 1.0, next(&mut s) * 3.0 + 0.5];
            let g = if fid == 0 { reverse_gradient(f0, &x) } else { reverse_gradient(f1, &x) };
            o.w(4);
            o.w(fid);
            x.iter().for_each(|v| o.f(*v));
            g.iter().for_each(|v| o.f(*v));
            n_ad += 1;
        }
    }

    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&o.0))
        .expect("write");
    println!("wrote {n_dst} DST/IDST, {n_inv} inverse and {n_ad} gradient records to {path}");
}
