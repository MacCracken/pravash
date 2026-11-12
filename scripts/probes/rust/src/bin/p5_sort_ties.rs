//! Probe 5 (port plan §4 step 9.5): `sort_by_zorder`'s order among particles with equal keys.
//!
//! `sph::sort_by_zorder` calls `sort_unstable_by_key` on a 2D Morton code, so particles in one cell
//! share a key and their relative order is whatever Rust's unstable sort (ipnsort, on 1.99) leaves.
//! This measures how often that differs from the stable order on (key, input index), which is what a
//! Cyrius stable sort would give (ADR 0002 #3).
//!
//! Each particle's input index rides in `temperature`; the key is recomputed by a replica of the
//! private `morton_code_2d`, and checked against the sorted output (it must be non-decreasing).
//!
//!   cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin p5_sort_ties

use pravash::FluidParticle;
use pravash::sph::{create_particle_block, sort_by_zorder};

fn spread_bits(x: u32) -> u64 {
    let mut v = u64::from(x);
    v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
    v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    v = (v | (v << 1)) & 0x5555_5555_5555_5555;
    v
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn morton(x: f64, y: f64, inv_cell: f64) -> u64 {
    let ix = (x * inv_cell).max(0.0) as u32;
    let iy = (y * inv_cell).max(0.0) as u32;
    spread_bits(ix) | (spread_bits(iy) << 1)
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

fn run(name: &str, mut ps: Vec<FluidParticle>, cell: f64) {
    for (i, p) in ps.iter_mut().enumerate() {
        p.temperature = i as f64;
    }
    let min_x = ps.iter().fold(f64::MAX, |m, p| m.min(p.position.x));
    let min_y = ps.iter().fold(f64::MAX, |m, p| m.min(p.position.y));
    let inv = 1.0 / cell;
    let key = |p: &FluidParticle| morton(p.position.x - min_x, p.position.y - min_y, inv);
    // The stable reference: (key, input index).
    let mut stable: Vec<(u64, usize)> = ps.iter().enumerate().map(|(i, p)| (key(p), i)).collect();
    stable.sort();
    sort_by_zorder(&mut ps, cell);
    let got: Vec<usize> = ps.iter().map(|p| p.temperature as usize).collect();
    let monotone = ps.windows(2).all(|w| key(&w[0]) <= key(&w[1]));
    let differ = got.iter().zip(&stable).filter(|(g, s)| **g != s.1).count();
    let mut groups = 0;
    let mut tied = 0;
    let mut k = 0;
    while k < stable.len() {
        let mut e = k;
        while e < stable.len() && stable[e].0 == stable[k].0 {
            e += 1;
        }
        if e - k > 1 {
            groups += 1;
            tied += e - k;
        }
        k = e;
    }
    println!(
        "{name} | {} | {groups} tie groups, {tied} particles | keys non-decreasing: {monotone} | \
         positions differing from the stable order: {differ}",
        ps.len()
    );
}

fn main() {
    println!("scenario | particles | ties | check | result");
    // A lattice block at h/2 spacing: four particles per h-cell, so every cell is a tie group.
    run("block 40 × 40 at h/2, cell h", create_particle_block([0.0, 0.0], [1.0, 1.0], 0.025, 0.001), 0.05);
    run("block 100 × 100 at h/4, cell h", create_particle_block([0.0, 0.0], [1.0, 1.0], 0.01, 0.001), 0.04);
    // A random cloud: ties wherever two particles share a cell.
    let mut s = 0x9e37_79b9_7f4a_7c15;
    let cloud: Vec<FluidParticle> = (0..5000)
        .map(|_| FluidParticle::new_2d(next(&mut s), next(&mut s), 0.001))
        .collect();
    run("random cloud of 5,000 in a unit box, cell 0.05", cloud, 0.05);
    // An already-sorted input: the case a solver sees on its second sort.
    let mut once = create_particle_block([0.0, 0.0], [1.0, 1.0], 0.025, 0.001);
    sort_by_zorder(&mut once, 0.05);
    run("the 40 × 40 block, sorted once already", once, 0.05);
}
