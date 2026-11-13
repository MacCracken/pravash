//! The Rust side of tests/pravash.bcyr's `coupling_a/*` rows (M6 part a, docs/benchmarks-rust-v-cyrius.md): the
//! indexed drag and its hash, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_coupling_a
//!
//! Each row is the median of 7 runs, in ns per call:
//! - `hash_build_<n>`: `SpatialHash::new(0.05)` and every particle inserted at its f32 position (the port: a cells-only
//!   `PvNeighborGrid_build`), for the `drag_from_particles` bench's block (0.5 at (0.1, 0.1), spacing 0.02) and a
//!   unit block at spacing 0.01;
//! - `drag_indexed_<n>`: `drag_from_particles_indexed` on that hash (the sphere of 0.1 at (0.3, 0.3) moving at
//!   (1, 0, 0), ρ 1000, C_d 0.47, sample radius 0.15; at (0.5, 0.5) on the unit block);
//! - `drag_brute_<n>`: `drag_from_particles` on the unit block, for the indexed row's comparison.

use std::hint::black_box;
use std::time::Instant;

use hisab::{DVec3, SpatialHash, Vec3};
use pravash::FluidParticle;
use pravash::coupling::{BodyShape, RigidBody, drag_from_particles, drag_from_particles_indexed};
use pravash::sph::create_particle_block;

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

fn hash_of(ps: &[FluidParticle], cell: f32) -> SpatialHash {
    let mut hash = SpatialHash::new(cell).expect("a positive cell size");
    for (i, p) in ps.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        hash.insert(
            Vec3::new(
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
            ),
            i,
        );
    }
    hash
}

fn main() {
    let small = create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let unit = create_particle_block([0.0, 0.0], [1.0, 1.0], 0.01, 0.001);
    let mut b_small = RigidBody::new(
        DVec3::new(0.3, 0.3, 0.0),
        1.0,
        BodyShape::Sphere { radius: 0.1 },
    );
    b_small.velocity = DVec3::new(1.0, 0.0, 0.0);
    let mut b_unit = b_small.clone();
    b_unit.position = DVec3::new(0.5, 0.5, 0.0);
    for (ps, body, iters) in [(&small, &b_small, 2000), (&unit, &b_unit, 200)] {
        let n = ps.len();
        time(&format!("coupling_a/hash_build_{n}"), iters / 4, || {
            hash_of(black_box(ps), 0.05).len() as u64
        });
        let hash = hash_of(ps, 0.05);
        time(&format!("coupling_a/drag_indexed_{n}"), iters * 4, || {
            let d = drag_from_particles_indexed(
                black_box(body),
                black_box(ps),
                &hash,
                1000.0,
                0.47,
                0.15,
            );
            d.x.to_bits()
        });
        if n > 1000 {
            time(&format!("coupling_a/drag_brute_{n}"), iters, || {
                let d = drag_from_particles(black_box(body), black_box(ps), 1000.0, 0.47, 0.15);
                d.x.to_bits()
            });
        }
    }
}
