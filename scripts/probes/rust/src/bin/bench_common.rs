//! The Rust side of tests/pravash.bcyr's `common/*`, `soa/*`, `arena/*` (M1 part 3) and `m1/*` (M1 part 4) rows (
//! docs/benchmarks-rust-v-cyrius.md): the same bodies on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_common
//!
//! Each row is the median of 7 runs of `iters` bodies, in ns per body. Inputs go through `black_box`, and each body's
//! result is folded into a sink.

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use pravash::bridge::{temperature_to_water_viscosity, tke_to_broadband_noise_db};
use pravash::buoyancy::{reynolds_number, terminal_velocity};
use pravash::vortex::{enstrophy, kolmogorov_scale, lamb_oseen_velocity};
use pravash::{FluidConfig, FluidMaterial, FluidParticle, ParticleArena, ParticleSoa};

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
    println!("  {name:<28} {:>10.1} ns", ns[RUNS / 2]);
}

/// 1,000 particles on a 40-wide lattice, each with a velocity.
fn particles() -> Vec<FluidParticle> {
    (0..1000)
        .map(|i| {
            let mut p =
                FluidParticle::new_2d(f64::from(i % 40) * 0.025, f64::from(i / 40) * 0.025, 0.001);
            p.velocity = DVec3::new(f64::from(i % 7) * 0.1, -f64::from(i % 5) * 0.2, 0.0);
            p
        })
        .collect()
}

fn main() {
    let ps = particles();
    time("common/particle_new_2d", 1_000_000, |k| {
        let x = black_box(f64::from(u32::try_from(k & 1023).unwrap_or(0)));
        FluidParticle::new_2d(x, 0.5, 0.001).position.x.to_bits()
    });
    time("common/kinetic_energy", 1_000_000, |k| {
        black_box(&ps)[k % 1000].kinetic_energy().to_bits()
    });
    let cfg = FluidConfig::water_2d();
    time("common/config_validate", 1_000_000, |_| {
        u64::from(black_box(&cfg).validate().is_ok())
    });
    time("common/check_cfl", 1_000_000, |k| {
        let v = black_box(f64::from(u32::try_from(k & 63).unwrap_or(0)));
        u64::from(black_box(&cfg).check_cfl(v).is_ok())
    });
    time("common/material_custom", 1_000_000, |_| {
        FluidMaterial::custom(black_box(800.0), 0.01, 0.05, 1400.0)
            .map_or(0, |m| m.density.to_bits())
    });
    time("soa/from_aos_1000", 10_000, |_| {
        let soa = ParticleSoa::from_aos(black_box(&ps));
        soa.pos_x[999].to_bits()
    });
    let soa = ParticleSoa::from_aos(&ps);
    time("soa/to_aos_1000", 10_000, |_| {
        let back = black_box(&soa).to_aos();
        back[999].position.x.to_bits()
    });
    let mut out = ps.clone();
    time("soa/write_to_aos_1000", 10_000, |_| {
        black_box(&soa).write_to_aos(&mut out);
        out[999].position.x.to_bits()
    });
    // new(1000), 100 blocks of 10, every other one freed, then compact (500 records moved).
    time("arena/cycle_1000", 10_000, |_| {
        let mut a = ParticleArena::new(black_box(1000));
        let mut hs = Vec::with_capacity(100);
        for _ in 0..100 {
            hs.push(a.alloc(10).expect("room"));
        }
        for h in hs.iter().step_by(2) {
            a.free(*h);
        }
        a.compact() as u64
    });

    // M1 part 4: buoyancy, vortex, bridge, JSON.
    let arg = |k: usize| f64::from(u32::try_from(k & 1023).unwrap_or(0));
    time("m1/terminal_velocity", 1_000_000, |k| {
        terminal_velocity(1.0 + black_box(arg(k)) * 0.001, 9.81, 1.225, 0.47, 0.01)
            .map_or(0, f64::to_bits)
    });
    time("m1/reynolds_number", 1_000_000, |k| {
        reynolds_number(1000.0, black_box(arg(k)) * 0.01, 0.1, 0.001).map_or(0, f64::to_bits)
    });
    time("m1/lamb_oseen_velocity", 1_000_000, |k| {
        lamb_oseen_velocity(1.0, 0.01 + black_box(arg(k)) * 0.0001, 0.001, 1.0).to_bits()
    });
    time("m1/kolmogorov_scale", 1_000_000, |k| {
        kolmogorov_scale(0.000_001, 0.001 + black_box(arg(k)) * 0.001).to_bits()
    });
    let field: Vec<f64> = (0..1000).map(|i| f64::from(i % 17) * 0.1 - 0.8).collect();
    time("m1/enstrophy_1000", 100_000, |_| {
        enstrophy(black_box(&field), 0.01).to_bits()
    });
    time("m1/water_viscosity", 1_000_000, |k| {
        temperature_to_water_viscosity(273.0 + black_box(arg(k)) * 0.0977).to_bits()
    });
    time("m1/broadband_noise_db", 1_000_000, |k| {
        tke_to_broadband_noise_db(0.001 + black_box(arg(k)) * 0.01).to_bits()
    });
    time("json/config_write", 100_000, |_| {
        serde_json::to_string(black_box(&cfg)).map_or(0, |s| s.len() as u64)
    });
    let doc = serde_json::to_string(&cfg).unwrap_or_default();
    time("json/config_read", 100_000, |_| {
        serde_json::from_str::<FluidConfig>(black_box(&doc)).map_or(0, |c| c.dt.to_bits())
    });
    let pdoc = serde_json::to_string(&ps[17]).unwrap_or_default();
    time("json/particle_read", 100_000, |_| {
        serde_json::from_str::<FluidParticle>(black_box(&pdoc)).map_or(0, |p| p.mass.to_bits())
    });
    let small = ParticleSoa::from_aos(&ps[..100]);
    let sdoc = serde_json::to_string(&small).unwrap_or_default();
    time("json/soa_read_100", 10_000, |_| {
        serde_json::from_str::<ParticleSoa>(black_box(&sdoc)).map_or(0, |s| s.len() as u64)
    });
}
