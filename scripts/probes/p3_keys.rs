//! Probe 3 (port plan §4 step 9.3), the Rust side: SPH cell keys for 10⁶ seeded positions, as
//! `SphSolver::build_neighbors` computes them (`floor(p as f32 · (1 / h as f32)) as i32`, the same
//! expression as the oracle's `sph_cell_key` hook, which `validate.py` checks).
//!
//!   rustc -O --edition 2024 -o build/p3_keys scripts/probes/p3_keys.rs && ./build/p3_keys build/p3_keys.bin
//!
//! Writes one 24-byte record per case (little-endian): the position's f64 bits, h's f64 bits, and
//! the key as an i64. `p3_keys.cyr` recomputes every key and compares.

use std::hint::black_box;
use std::io::Write as _;

/// xorshift64*, as the golden generator's grids use.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn key(p: f64, h: f64) -> i32 {
    let h_f32 = black_box(h) as f32;
    let inv_cs = 1.0 / h_f32;
    (black_box(p) as f32 * inv_cs).floor() as i32
}

fn main() {
    let path = std::env::args().nth(1).expect("output path");
    let hs = [0.01, 0.025, 0.037_3, 0.05, 0.1, 1.0];
    let mut rng = Rng(0x5ce1_5ce1_5ce1);
    let mut out = Vec::with_capacity(1_000_000 * 24);
    let mut n = 0u64;
    let push = |p: f64, h: f64, out: &mut Vec<u8>| {
        out.extend_from_slice(&p.to_bits().to_le_bytes());
        out.extend_from_slice(&h.to_bits().to_le_bytes());
        out.extend_from_slice(&i64::from(key(p, h)).to_le_bytes());
    };
    // 1. Uniform positions in a box, for every h.
    for k in 0..600_000 {
        let h = hs[k % hs.len()];
        let p = (rng.unit() - 0.5) * 4.0;
        push(p, h, &mut out);
        n += 1;
    }
    // 2. On and next to every cell boundary in [-100h, 100h].
    for &h in &hs {
        for k in -100..=100 {
            let x = f64::from(k) * h;
            for d in [-2i64, -1, 0, 1, 2] {
                push(f64::from_bits((x.to_bits() as i64 + d) as u64), h, &mut out);
                n += 1;
            }
        }
    }
    // 3. Large coordinates, where f32 loses precision (the roadmap's defect, D3): |x| up to 10⁷.
    while n < 1_000_000 {
        let h = hs[(n as usize) % hs.len()];
        let mag = 10f64.powf(rng.unit() * 7.0);
        let p = if rng.next_u64() & 1 == 1 { -mag } else { mag };
        push(p, h, &mut out);
        n += 1;
    }
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(&out))
        .expect("write");
    println!("wrote {n} cases to {path}");
}
