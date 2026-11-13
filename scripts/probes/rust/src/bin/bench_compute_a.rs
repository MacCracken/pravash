//! The Rust side of tests/pravash.bcyr's `compute_a/*` rows (M7 part a, docs/benchmarks-rust-v-cyrius.md): the
//! compute surface, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_compute_a
//!
//! Each row is the median of 7 runs, in ns per iteration:
//! - `from_soa_1000`: `PackedParticles::from_soa` of a 1,000-particle SoA (particle i at
//!   ((i mod 10)·0.01, (⌊i/10⌋ mod 10)·0.01, ⌊i/100⌋·0.01), mass 0.001, velocity (0.1, −0.2, 0.3), density 1000 + i,
//!   pressure i/2), the buffers dropped after;
//! - `unpack_1000`: the three unpacks of that packing into the same SoA;
//! - `dpoly6_dr2`, `dtait_drho`, `dwendland_c2_dr`: one call, the argument k of 1,024 cycled: r² = k·1e-6 at
//!   h 0.04; ρ = 990 + k/50 at ρ₀ 1000, c 1480, γ 7; r = k·1e-4 at h 0.05;
//! - `autodiff_poly6`: `autodiff_gradient` of m_compute.rs's catalogue function 2 (the poly6 kernel, 13 nodes) at
//!   r 0.02, h 0.06;
//! - `sph_params_read`: `serde_json::from_str::<SphKernelParams>` of serde_json's own document for (0.05, 1000, 2000,
//!   0.001, [0, −9.81, 0], 0.001, 1000); `sph_params_read_tie`: the same with dt `1.0000000596046448`, which lands on
//!   an f32 midpoint as an f64 (the port's ledger L15 then reads the text).

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use hisab::autodiff::{Tape, Var};
use pravash::compute::{KernelDerivatives, PackedParticles, SphKernelParams};
use pravash::{FluidParticle, ParticleSoa};

const RUNS: usize = 7;

fn time(name: &str, iters: usize, mut body: impl FnMut(usize) -> u64) {
    let mut ns = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for k in 0..iters {
            sink ^= body(k);
        }
        ns.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    black_box(sink);
    ns.sort_by(f64::total_cmp);
    println!("  {name:<30} {:>12.2} ns", ns[RUNS / 2]);
}

#[allow(clippy::cast_precision_loss)]
fn soa() -> ParticleSoa {
    let ps: Vec<FluidParticle> = (0..1000)
        .map(|i| {
            let mut p = FluidParticle::new(
                DVec3::new(
                    f64::from(i % 10) * 0.01,
                    f64::from((i / 10) % 10) * 0.01,
                    f64::from(i / 100) * 0.01,
                ),
                0.001,
            );
            p.velocity = DVec3::new(0.1, -0.2, 0.3);
            p.density = 1000.0 + f64::from(i);
            p.pressure = 0.5 * f64::from(i);
            p
        })
        .collect();
    ParticleSoa::from_aos(&ps)
}

fn ad2(t: &mut Tape, x: &[Var]) -> Var {
    let h2 = t.mul(x[1], x[1]);
    let r2 = t.mul(x[0], x[0]);
    let d = t.sub(h2, r2);
    let d2 = t.mul(d, d);
    let d3 = t.mul(d2, d);
    let h3 = t.mul(h2, x[1]);
    let h6 = t.mul(h3, h3);
    let h9 = t.mul(h6, h3);
    let k = t.constant(315.0 / (64.0 * std::f64::consts::PI));
    let c = t.div(k, h9);
    t.mul(c, d3)
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let s = soa();
    time("compute_a/from_soa_1000", 2000, |_| {
        let p = PackedParticles::from_soa(black_box(&s));
        u64::from(p.count)
    });
    let p = PackedParticles::from_soa(&s);
    let mut dst = s.clone();
    time("compute_a/unpack_1000", 2000, |_| {
        p.unpack_density_pressure(&mut dst);
        p.unpack_velocity(&mut dst);
        p.unpack_positions(&mut dst);
        dst.density[7].to_bits()
    });
    time("compute_a/dpoly6_dr2", 1_000_000, |k| {
        let r2 = (k % 1024) as f64 * 1e-6;
        KernelDerivatives::dpoly6_dr2(black_box(r2), black_box(0.04)).to_bits()
    });
    time("compute_a/dtait_drho", 1_000_000, |k| {
        let rho = 990.0 + (k % 1024) as f64 / 50.0;
        KernelDerivatives::dtait_drho(black_box(rho), 1000.0, 1480.0, 7.0).to_bits()
    });
    time("compute_a/dwendland_c2_dr", 1_000_000, |k| {
        let r = (k % 1024) as f64 * 1e-4;
        KernelDerivatives::dwendland_c2_dr(black_box(r), black_box(0.05)).to_bits()
    });
    time("compute_a/autodiff_poly6", 200_000, |_| {
        let g = KernelDerivatives::autodiff_gradient(ad2, black_box(&[0.02, 0.06]));
        g[0].to_bits()
    });
    let doc = r#"{"smoothing_radius":0.05,"rest_density":1000.0,"gas_constant":2000.0,"viscosity":0.001,"gravity":[0.0,-9.81,0.0],"dt":0.001,"particle_count":1000}"#;
    let tie = doc.replacen(r#""dt":0.001"#, r#""dt":1.0000000596046448"#, 1);
    for (name, d) in [
        ("compute_a/sph_params_read", doc),
        ("compute_a/sph_params_read_tie", tie.as_str()),
    ] {
        time(name, 200_000, |_| {
            serde_json::from_str::<SphKernelParams>(black_box(d))
                .map(|p| u64::from(p.dt.to_bits()))
                .unwrap_or(0)
        });
    }
}
