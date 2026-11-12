//! The Rust side of tests/pravash.bcyr's `sph_b/*` rows (M4 part b, docs/benchmarks-rust-v-cyrius.md): the
//! neighbour build and `sort_by_zorder`, on the frozen pravash 1.3.0 and hisab 1.4.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_b
//!
//! Each row is the median of 7 runs of `iters` calls, in ns per call (one call is one whole build or sort).
//! - `SphSolver::build_neighbors` is private, so `build` re-states it line for line on hisab 1.4.0's public
//!   `SpatialHash` (`new`, `clear`, `insert`, `query_cell`), the calls the solver makes: a grid kept across builds
//!   (the solver's `clear` path), the f32 positions cached, the 27 probes per particle into CSR lists. The rows time
//!   the steady state: the grid's cell size is already h.
//! - `neighbor_build_2d_1000`: a 40-wide 2D lattice of 1,000 at spacing 0.025, h 0.05 (36 neighbours a particle);
//!   `neighbor_build_3d_1000`: a 10³ lattice at spacing 0.025, h 0.05 (216 a particle, at the interior).
//! - `zorder_1000`: the 2D lattice in a fixed shuffled order, copied in and sorted at cell 0.025 per call (the copy
//!   is in both sides' rows).

use std::hint::black_box;
use std::time::Instant;

use hisab::{DVec3, SpatialHash, Vec3};
use pravash::FluidParticle;
use pravash::sph::sort_by_zorder;

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
    println!("  {name:<28} {:>12.2} ns", ns[RUNS / 2]);
}

fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

/// `SphSolver`'s neighbour state: the grid, its cell size, the f32 positions and the CSR lists.
struct Nb {
    grid: SpatialHash,
    cell_size: f32,
    pos: Vec<Vec3>,
    indices: Vec<usize>,
    offsets: Vec<u32>,
}

impl Nb {
    fn new() -> Self {
        Self {
            grid: SpatialHash::new(1.0).expect("1 is positive"),
            cell_size: 1.0,
            pos: Vec::new(),
            indices: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// rust-old/src/sph.rs `build_neighbors`, on the same hisab calls.
    #[allow(clippy::cast_possible_truncation)]
    fn build(&mut self, particles: &[FluidParticle], h_f32: f32) {
        let n = particles.len();
        let inv_cs = 1.0 / h_f32;
        let half_cs = h_f32 * 0.5;
        if (self.cell_size - h_f32).abs() > f32::EPSILON {
            self.grid = SpatialHash::new(h_f32).expect("h is positive");
            self.cell_size = h_f32;
        } else {
            self.grid.clear();
        }
        self.pos.clear();
        self.pos.reserve(n);
        for (i, p) in particles.iter().enumerate() {
            let q = Vec3::new(
                p.position[0] as f32,
                p.position[1] as f32,
                p.position[2] as f32,
            );
            self.pos.push(q);
            self.grid.insert(q, i);
        }
        self.indices.clear();
        self.offsets.clear();
        self.offsets.reserve(n + 1);
        self.offsets.push(0);
        for q in &self.pos {
            let cx = (q.x * inv_cs).floor() as i32;
            let cy = (q.y * inv_cs).floor() as i32;
            let cz = (q.z * inv_cs).floor() as i32;
            for dz in -1..=1i32 {
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        #[allow(clippy::cast_precision_loss)]
                        let probe = Vec3::new(
                            (cx + dx) as f32 * h_f32 + half_cs,
                            (cy + dy) as f32 * h_f32 + half_cs,
                            (cz + dz) as f32 * h_f32 + half_cs,
                        );
                        self.indices.extend_from_slice(self.grid.query_cell(probe));
                    }
                }
            }
            self.offsets
                .push(u32::try_from(self.indices.len()).unwrap_or(u32::MAX));
        }
    }
}

fn main() {
    let lat2: Vec<FluidParticle> = (0..1000)
        .map(|k| FluidParticle::new_2d(fx(k % 40) * 0.025, fx(k / 40) * 0.025, 0.001))
        .collect();
    let lat3: Vec<FluidParticle> = (0..1000)
        .map(|k| {
            let mut p = FluidParticle::new_2d(fx(k % 10) * 0.025, fx((k / 10) % 10) * 0.025, 0.001);
            p.position = DVec3::new(p.position.x, p.position.y, fx(k / 100) * 0.025);
            p
        })
        .collect();
    let h = 0.05f32;
    for (name, ps) in [
        ("sph_b/neighbor_build_2d_1000", &lat2),
        ("sph_b/neighbor_build_3d_1000", &lat3),
    ] {
        let mut nb = Nb::new();
        nb.build(ps, h);
        time(name, 2_000, || {
            nb.build(black_box(ps), black_box(h));
            u64::try_from(nb.indices.len()).unwrap_or(0)
        });
    }
    // The same fixed shuffle as the Cyrius side: index k goes to (k · 389) mod 1000 (389 is prime to 1000).
    let mut shuffled = lat2.clone();
    for k in 0..1000 {
        shuffled[(k * 389) % 1000] = lat2[k];
    }
    let mut work = shuffled.clone();
    time("sph_b/zorder_1000", 5_000, || {
        work.copy_from_slice(black_box(&shuffled));
        sort_by_zorder(black_box(&mut work), black_box(0.025));
        work[0].position.x.to_bits()
    });
}
