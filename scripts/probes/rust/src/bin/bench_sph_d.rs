//! The Rust side of tests/pravash.bcyr's `sph_d/*` rows (M4 part d, docs/benchmarks-rust-v-cyrius.md): the physics
//! extensions, which rust-old/benches doesn't time, on the frozen pravash 1.3.0.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_sph_d
//!
//! Each row is the median of 7 runs of `iters` calls, in ns per call. The particles are the 25 × 25 block at spacing
//! 0.02 (625; density 1000, velocity (sin 0.1i, cos 0.1i, 0), as Rust's `sph_viscosity_force` bench), with brute-force
//! lists at h 0.05 (Rust's tests' `brute_neighbors`: every pair within h, itself included, in index order). The
//! functions change the particles in place, so each row starts from a fresh copy and both sides make the same sequence
//! of calls (the same states, so the same work: the implicit solve's early exit included).
//! - `viscoelastic_625` (G 100, λ 0.05, dt 0.001), `heat_625` (κ 0.01, dt 0.001), `delta_sph_625` (c₀ 10, δ 0.1,
//!   dt 0.001), `implicit_viscosity_625` (ν 0.5, dt 0.01, 5 sweeps), `gradient_corrections_625` (the matrices, then
//!   dropped);
//! - `invert_3x3`: one moment-like matrix (the oracle hook, hisab's `matrix_inverse`);
//! - `phase_change_625`, `combustion_625` and `reaction_625` (through `&dyn ReactionProvider`; both on a copy at
//!   600–610 K, above ignition), `contact_angle_625`
//!   (θ π/4, σ 0.07, water_2d's walls), `generate_secondary_625` (threshold 0.5, lifetime 2: every particle sheds one)
//!   and `update_secondary_625` (their update, gravity −9.81, dt 0.001; lifetimes of 1e9, so none dies).

use std::hint::black_box;
use std::time::Instant;

use hisab::DVec3;
use pravash::sph::{
    CombustionConfig, HeatConfig, PhaseChangeConfig, ReactionProvider, ViscoelasticConfig,
    apply_contact_angle, apply_delta_sph, apply_implicit_viscosity, compute_gradient_corrections,
    create_particle_block, generate_secondary_particles, update_combustion, update_heat,
    update_phase_change, update_reaction, update_secondary_particles, update_viscoelastic,
};
use pravash::{FluidConfig, FluidParticle};

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
    println!("  {name:<34} {:>12.2} ns", ns[RUNS / 2]);
}

fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

fn brute(ps: &[FluidParticle], h: f64) -> (Vec<u32>, Vec<usize>) {
    let mut o = vec![0u32];
    let mut idx = Vec::new();
    for pi in ps {
        for (j, pj) in ps.iter().enumerate() {
            if pi.distance_to(pj) <= h {
                idx.push(j);
            }
        }
        o.push(u32::try_from(idx.len()).unwrap_or(u32::MAX));
    }
    (o, idx)
}

fn main() {
    let mut base = create_particle_block([0.0, 0.0], [0.5, 0.5], 0.02, 0.001);
    for (i, p) in base.iter_mut().enumerate() {
        p.density = 1000.0;
        p.velocity = DVec3::new((fx(i) * 0.1).sin(), (fx(i) * 0.1).cos(), 0.0);
        p.temperature = 293.15 + fx(i % 11);
        p.fuel = 0.5;
        p.acceleration = DVec3::new(0.0, -9.81, 0.0);
    }
    let (o, idx) = brute(&base, 0.05);
    let h = 0.05;
    let mut ps = base.clone();
    let ve = ViscoelasticConfig::new(100.0, 0.05);
    time("sph_d/viscoelastic_625", 200, || {
        update_viscoelastic(&mut ps, black_box(&o), black_box(&idx), h, &ve, 0.001);
        ps[0].conformation[0].to_bits()
    });
    let mut ps = base.clone();
    let heat = HeatConfig::new(0.01);
    time("sph_d/heat_625", 500, || {
        update_heat(&mut ps, black_box(&o), black_box(&idx), h, &heat, 0.001);
        ps[0].temperature.to_bits()
    });
    let mut ps = base.clone();
    time("sph_d/delta_sph_625", 500, || {
        apply_delta_sph(&mut ps, black_box(&o), black_box(&idx), h, 10.0, 0.1, 0.001);
        ps[0].density.to_bits()
    });
    let mut ps = base.clone();
    time("sph_d/implicit_viscosity_625", 100, || {
        apply_implicit_viscosity(&mut ps, black_box(&o), black_box(&idx), h, 0.5, 0.01, 5);
        ps[0].velocity.x.to_bits()
    });
    time("sph_d/gradient_corrections_625", 200, || {
        let c = compute_gradient_corrections(black_box(&base), black_box(&o), black_box(&idx), h);
        c[312][4].to_bits()
    });
    let m = [2.0, 0.5, 0.1, 0.3, 3.0, 0.7, 0.2, 0.4, 5.0];
    time("sph_d/invert_3x3", 1_000_000, || {
        pravash::oracle::invert_3x3(black_box(&m)).map_or(0, |v| v[4].to_bits())
    });
    let mut ps = base.clone();
    let pc = PhaseChangeConfig::water_ice(0, 1);
    time("sph_d/phase_change_625", 20_000, || {
        update_phase_change(&mut ps, &pc, 0.001);
        ps[0].fuel.to_bits()
    });
    // Above ignition, so the reaction runs (fuel 0.5 depletes toward 0, never reaching it).
    let mut hot = base.clone();
    for (i, p) in hot.iter_mut().enumerate() {
        p.temperature = 600.0 + fx(i % 11);
    }
    let mut ps = hot.clone();
    let cc = CombustionConfig::default();
    time("sph_d/combustion_625", 20_000, || {
        update_combustion(&mut ps, black_box(&cc), 0.001);
        ps[0].fuel.to_bits()
    });
    let mut ps = hot.clone();
    let provider: &dyn ReactionProvider = &cc;
    time("sph_d/reaction_625", 20_000, || {
        update_reaction(&mut ps, black_box(provider), 0.001);
        ps[0].fuel.to_bits()
    });
    let mut ps = base.clone();
    let water = FluidConfig::water_2d();
    time("sph_d/contact_angle_625", 20_000, || {
        apply_contact_angle(&mut ps, &water, std::f64::consts::FRAC_PI_4, 0.07);
        ps[0].acceleration.y.to_bits()
    });
    time("sph_d/generate_secondary_625", 20_000, || {
        let s = generate_secondary_particles(black_box(&base), 0.5, 2.0);
        s.len() as u64
    });
    let mut ss = generate_secondary_particles(&base, 0.5, 2.0);
    for s in &mut ss {
        s.lifetime = 1e9;
    }
    time("sph_d/update_secondary_625", 20_000, || {
        update_secondary_particles(&mut ss, DVec3::new(0.0, -9.81, 0.0), 0.001);
        ss[0].position.y.to_bits()
    });
}
