//! Probe 4 (port plan §4 step 9.4): is the SPH neighbour candidate order reproducible without a
//! hash map?
//!
//! `SphSolver::build_neighbors` keys each particle's f32 position, then concatenates hisab's
//! `query_cell` of a probe point in each of the 27 cells around it (dz, dy, dx from -1 to 1). The
//! order every SPH sum follows is that list. The expectation here uses no hash map at all: it
//! groups particle indices by their cell key, in index order, then concatenates the groups in the
//! same offset order. If the oracle's dumped lists equal it, the order is (cell offset, insertion)
//! and the port can build it from sorted arrays.
//!
//! It also counts the probe points whose own key is not the intended cell, `(cx + dx, …)`. At large
//! coordinates the f32 sum `(cx + dx) · h + h/2` rounds into another cell: the roadmap's precision
//! defect (D3), measured here.
//!
//!   cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin p4_neighbor_order

use std::collections::BTreeMap;

use pravash::oracle::sph_cell_key;
use pravash::sph::SphSolver;
use pravash::{FluidConfig, FluidParticle};

fn cloud(offset: f64, three_d: bool, n_side: usize, spacing: f64) -> Vec<FluidParticle> {
    let mut v = Vec::new();
    let nz = if three_d { n_side } else { 1 };
    for k in 0..nz {
        for j in 0..n_side {
            for i in 0..n_side {
                // A deterministic jitter so particles sit off the lattice and across cell faces.
                let jit = ((i * 37 + j * 11 + k * 5) % 13) as f64 * spacing * 0.03;
                let x = offset + i as f64 * spacing + jit;
                let y = offset + j as f64 * spacing - jit;
                let z = if three_d { offset + k as f64 * spacing + jit * 0.5 } else { 0.0 };
                let mut p = FluidParticle::new_2d(x, y, 0.001);
                p.position.z = z;
                v.push(p);
            }
        }
    }
    v
}

fn main() {
    println!("scenario | particles | lists equal the no-hash-map expectation | probe points off their cell");
    for (name, offset, three_d, n_side) in [
        ("2D block at the origin", 0.0, false, 40),
        ("3D cloud at the origin", 0.0, true, 14),
        ("2D block at 1e3", 1e3, false, 40),
        ("2D block at 1e5", 1e5, false, 40),
        ("2D block at 1e6", 1e6, false, 40),
    ] {
        let h = 0.05;
        let mut cfg = FluidConfig::water_2d();
        cfg.smoothing_radius = h;
        cfg.dt = 0.0001;
        cfg.bounds_min = hisab::DVec3::splat(-1e7);
        cfg.bounds_max = hisab::DVec3::splat(1e7);
        let mut ps = cloud(offset, three_d, n_side, h * 0.5);
        let mut s = SphSolver::new();
        s.step(&mut ps, &cfg, 0.01).expect("step");
        let d = s.oracle_dump();
        let h32 = h as f32;

        // The expectation: groups by key, in index order.
        let keys: Vec<[i32; 3]> = d.positions_f32.iter().map(|p| sph_cell_key(*p, h32)).collect();
        let mut groups: BTreeMap<[i32; 3], Vec<usize>> = BTreeMap::new();
        for (i, k) in keys.iter().enumerate() {
            groups.entry(*k).or_default().push(i);
        }
        let mut equal = true;
        let mut probes_off = 0usize;
        let half = h32 * 0.5;
        for (i, k) in keys.iter().enumerate() {
            let mut want = Vec::new();
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        let cell = [k[0] + dx, k[1] + dy, k[2] + dz];
                        // The probe point build_neighbors constructs, and its own key.
                        let probe = [
                            cell[0] as f32 * h32 + half,
                            cell[1] as f32 * h32 + half,
                            cell[2] as f32 * h32 + half,
                        ];
                        let pk = sph_cell_key(probe, h32);
                        if pk != cell {
                            probes_off += 1;
                        }
                        // hisab looks up the probe's own cell.
                        if let Some(g) = groups.get(&pk) {
                            want.extend_from_slice(g);
                        }
                    }
                }
            }
            let got = &d.neighbor_indices
                [d.neighbor_offsets[i] as usize..d.neighbor_offsets[i + 1] as usize];
            if got != want.as_slice() {
                equal = false;
            }
        }
        println!(
            "{name} | {} | {} | {probes_off} of {}",
            ps.len(),
            if equal { "yes, every list" } else { "NO" },
            ps.len() * 27
        );
    }
}
