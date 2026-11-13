//! The Rust side of tests/pravash.bcyr's `mpm_c/*` rows (M6 part c, docs/benchmarks-rust-v-cyrius.md): one MPM step
//! of the plan's three scenes, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_mpm_c
//!
//! Each row is the median of 7 runs of 100 steps, in ns per step: one 16 × 16 solver (dx 0.05), the scene's particles
//! and MPM states copied in before each step, gravity (0, −9.81, 0), dt 1e-3, density 1000 at spacing 0.025:
//! - `elastic_81`: a Neo-Hookean block (E 1e4, ν 0.3), [0.3, 0.5] × [0.45, 0.65];
//! - `sand_80`: a Drucker–Prager column (E 1e4, ν 0.3, φ 30°), [0.35, 0.45] × [0.025, 0.4];
//! - `fluid_144`: a fluid column (K 1e3), [0.05, 0.25] × [0.025, 0.4].

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use pravash::FluidParticle;
use pravash::mpm::{ConstitutiveModel, MpmParticle, MpmSolver};

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

fn block(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    md: ConstitutiveModel,
) -> (Vec<FluidParticle>, Vec<MpmParticle>) {
    let sp = 0.025;
    let v0 = sp * sp;
    let (mut ps, mut ms) = (Vec::new(), Vec::new());
    let mut j = 0.0;
    while y0 + j * sp <= y1 + 1e-12 {
        let mut i = 0.0;
        while x0 + i * sp <= x1 + 1e-12 {
            ps.push(FluidParticle::new_2d(x0 + i * sp, y0 + j * sp, 1000.0 * v0));
            ms.push(MpmParticle::new(md, v0));
            i += 1.0;
        }
        j += 1.0;
    }
    (ps, ms)
}

fn main() {
    let g = DVec3::new(0.0, -9.81, 0.0);
    let scenes = [
        (
            "mpm_c/elastic",
            block(
                0.3,
                0.5,
                0.45,
                0.65,
                ConstitutiveModel::NeoHookean {
                    youngs_modulus: 1e4,
                    poisson_ratio: 0.3,
                },
            ),
        ),
        (
            "mpm_c/sand",
            block(
                0.35,
                0.45,
                0.025,
                0.4,
                ConstitutiveModel::DruckerPrager {
                    youngs_modulus: 1e4,
                    poisson_ratio: 0.3,
                    friction_angle: 30.0,
                },
            ),
        ),
        (
            "mpm_c/fluid",
            block(
                0.05,
                0.25,
                0.025,
                0.4,
                ConstitutiveModel::Fluid {
                    bulk_modulus: 1e3,
                    viscosity: 0.0,
                },
            ),
        ),
    ];
    for (name, (ps, ms)) in &scenes {
        let mut solver = MpmSolver::new(16, 16, 0.05).expect("a valid grid");
        let mut wp = ps.clone();
        let mut wm = ms.clone();
        time(&format!("{name}_{}", ps.len()), 100, || {
            wp.copy_from_slice(ps);
            wm.copy_from_slice(ms);
            solver
                .step(&mut wp, &mut wm, g, 1e-3)
                .expect("a valid step");
            wp[0].position.y.to_bits()
        });
    }
}
