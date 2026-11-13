//! `debug.tsv`: Rust 1.3.0's `Debug` text, `{:?}` and `{:#?}`, for every public type that derives it (M8 part d).
//!
//! Rows:
//! - `f64::debug` takes `f:` and gives `s:<{:?}>`: the specials, the powers of ten and of two with their neighbours,
//!   the two thresholds where the form turns exponential (1e-4 and 1e16) with theirs, and seeded sweeps of bits,
//!   short decimals and integers. Each is class A: one text.
//! - `f32::debug` takes `f:` (the f32 widened, exactly) and gives `s:<{:?}>` of the f32, over the same kinds.
//! - `<Type>::debug` takes the value and gives `s:<{:?}> s:<{:#?}>`;
//! - `<Type>::debug_after` (the six types with a step) takes a value, the step's arguments and a step count (`i:`),
//!   and gives both texts after the steps: `ShallowWater` dt; `FluidGrid` and `MacGrid` a `GridConfig`; `KEpsilon`
//!   `v:` vx, `v:` vy, nx, ny, dx and dt; `VofField` (`advect`) and `PhaseField` `v:` vx, `v:` vy and dt.
//!
//! A value is its fields in declaration order, private ones included, since the derive prints them all:
//! - an f64 or f32 is `f:` (an f32 widened); an integer or a bool is `i:`; a `DVec3` or a `[f64; N]` / `[f32; N]` is
//!   its N floats; a `Vec` is `v:<n>` and its elements (a `Vec<u8>`'s as `i:`);
//! - a unit variant is `e:<Enum>.<Variant>`, and a variant with fields is that token and its fields;
//! - a nested struct is its own fields, inline; `FluidGrid`'s `mg_levels` is `v:<levels>` and three vectors a level.
//!
//! Values come from the types' constructors and public fields, seeded, with the float specials; the private scratch
//! from the oracle's dumps, after steps that fill it. `PravashError` is not here: 2.0's errors are codes with no
//! payload (ADR 0002 #1), so its `Debug` text has no counterpart.

use hisab::DVec3;
use pravash::buoyancy::FlowRegime;
use pravash::compute::{
    ComputeOp, GridKernelParams, NeuralCorrection, PackedParticles, ShallowKernelParams,
    SphKernelParams,
};
use pravash::coupling::{BodyShape, RigidBody};
use pravash::grid::{BoundaryCondition, FluidGrid, GridConfig, KEpsilon, MacGrid};
use pravash::mpm::{ConstitutiveModel, MpmParticle};
use pravash::phase_field::PhaseField;
use pravash::shallow::{SedimentConfig, ShallowWater};
use pravash::sph::{
    CombustionConfig, HeatConfig, MultiPhaseConfig, NonNewtonianViscosity, PhaseChangeConfig,
    PhaseProperties, SecondaryParticle, SecondaryType, ViscoelasticConfig,
};
use pravash::vof::VofField;
use pravash::{ArenaHandle, FluidConfig, FluidMaterial, FluidParticle, ParticleArena, ParticleSoa};

use crate::grids::{Rng, pow2, specials};
use crate::m_grid::{col, usz};
use crate::tsv::{Class, Module, Tok, b, e, f, i, s};

// ── values ──────────────────────────────────────────────────────────────────────────

/// 64 seeded bits.
fn bits64(rng: &mut Rng) -> u64 {
    #[allow(clippy::cast_sign_loss)]
    let hi = rng.int(0, 0xFFFF_FFFF) as u64;
    #[allow(clippy::cast_sign_loss)]
    let lo = rng.int(0, 0xFFFF_FFFF) as u64;
    (hi << 32) | lo
}

/// A field value whose text exercises the printer: a special, a value on either side of a threshold, an integer, a
/// short decimal, or an ordinary magnitude.
fn val(rng: &mut Rng) -> f64 {
    match rng.int(0, 9) {
        0 => {
            let sp = specials();
            sp[usize::try_from(rng.int(0, 9)).unwrap_or(0)]
        }
        1 => rng.signed_log_uniform(-8.0, -3.0),
        2 => rng.signed_log_uniform(15.0, 20.0),
        #[allow(clippy::cast_precision_loss)]
        3 => rng.int(-2000, 2000) as f64,
        #[allow(clippy::cast_precision_loss)]
        4 => rng.int(-99_999, 99_999) as f64 / 100.0,
        _ => rng.uniform(-10.0, 10.0),
    }
}

/// As `val`, for an f32 field.
#[allow(clippy::cast_possible_truncation)]
fn val32(rng: &mut Rng) -> f32 {
    match rng.int(0, 11) {
        0 => [
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            -0.0,
        ][usize::try_from(rng.int(0, 5)).unwrap_or(0)],
        _ => val(rng) as f32,
    }
}

fn v3(out: &mut Vec<Tok>, v: DVec3) {
    out.extend([f(v.x), f(v.y), f(v.z)]);
}

fn rv3(rng: &mut Rng) -> DVec3 {
    DVec3::new(val(rng), val(rng), val(rng))
}

fn col32(out: &mut Vec<Tok>, v: &[f32]) {
    out.push(Tok::V(v.len()));
    out.extend(v.iter().map(|&x| f(f64::from(x))));
}

fn rcol(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| val(rng)).collect()
}

fn rcol32(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| val32(rng)).collect()
}

fn row<T: std::fmt::Debug>(m: &mut Module, name: &str, input: &[Tok], x: &T) {
    m.row(
        name,
        Class::A,
        input,
        &[s(format!("{x:?}")), s(format!("{x:#?}"))],
    );
}

// ── the float printer ───────────────────────────────────────────────────────────────

fn f64_row(m: &mut Module, x: f64) {
    m.row("f64::debug", Class::A, &[f(x)], &[s(format!("{x:?}"))]);
}

fn f32_row(m: &mut Module, x: f32) {
    m.row(
        "f32::debug",
        Class::A,
        &[f(f64::from(x))],
        &[s(format!("{x:?}"))],
    );
}

fn f64_rows(m: &mut Module, rng: &mut Rng) {
    for x in specials() {
        f64_row(m, x);
        f64_row(m, -x);
    }
    f64_row(m, f64::from_bits(0x000F_FFFF_FFFF_FFFF)); // the largest subnormal
    f64_row(m, f64::EPSILON);
    // The powers of ten that parse, from 1e-323 to 1e308, and both neighbours of each.
    for k in -323..=308 {
        let x: f64 = format!("1e{k}").parse().unwrap_or(1.0);
        f64_row(m, x);
        f64_row(m, f64::from_bits(x.to_bits() + 1));
        f64_row(m, f64::from_bits(x.to_bits() - 1));
    }
    // The thresholds of the exponential form, 1e-4 and 1e16, eight steps of the last bit either side.
    for t in [1e-4_f64, 1e16] {
        for d in 0..=8_u64 {
            f64_row(m, f64::from_bits(t.to_bits() + d));
            f64_row(m, -f64::from_bits(t.to_bits() - d));
        }
    }
    // Every power of two (an exact tie between two shortest candidates happens there), and the binade's largest.
    for k in -1074..=1023 {
        let p = pow2(k);
        f64_row(m, p);
        if k % 5 == 0 && k < 1023 {
            f64_row(m, f64::from_bits(pow2(k + 1).to_bits() - 1));
        }
    }
    for _ in 0..1500 {
        f64_row(m, f64::from_bits(bits64(rng)));
    }
    for _ in 0..400 {
        let digits = rng.int(1, 999_999);
        let scale = rng.int(-12, 22);
        let x: f64 = format!("{digits}e{scale}").parse().unwrap_or(0.0);
        f64_row(m, x);
    }
    for _ in 0..200 {
        #[allow(clippy::cast_precision_loss)]
        f64_row(m, rng.int(-(1 << 53), 1 << 53) as f64);
    }
    for _ in 0..300 {
        f64_row(m, val(rng));
    }
}

fn f32_rows(m: &mut Module, rng: &mut Rng) {
    for x in [
        0.0_f32,
        1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007F_FFFF),
        f32::MAX,
        f32::EPSILON,
        f32::INFINITY,
        f32::NAN,
        0.1,
        0.2,
        1.0 / 3.0,
    ] {
        f32_row(m, x);
        f32_row(m, -x);
    }
    for k in -45..=38 {
        let x: f32 = format!("1e{k}").parse().unwrap_or(1.0);
        if x == 0.0 {
            continue;
        }
        f32_row(m, x);
        f32_row(m, f32::from_bits(x.to_bits() + 1));
        f32_row(m, f32::from_bits(x.to_bits() - 1));
    }
    for t in [1e-4_f32, 1e16] {
        for d in 0..=8_u32 {
            f32_row(m, f32::from_bits(t.to_bits() + d));
            f32_row(m, -f32::from_bits(t.to_bits() - d));
        }
    }
    for k in -149..=127 {
        #[allow(clippy::cast_possible_truncation)]
        let p = pow2(k) as f32;
        f32_row(m, p);
        if k < 127 {
            #[allow(clippy::cast_possible_truncation)]
            let next = pow2(k + 1) as f32;
            f32_row(m, f32::from_bits(next.to_bits() - 1));
        }
    }
    for _ in 0..1200 {
        #[allow(clippy::cast_possible_truncation)]
        let w = (bits64(rng) >> 32) as u32;
        f32_row(m, f32::from_bits(w));
    }
    for _ in 0..300 {
        let digits = rng.int(1, 99_999);
        let scale = rng.int(-10, 20);
        let x: f32 = format!("{digits}e{scale}").parse().unwrap_or(0.0);
        f32_row(m, x);
    }
    for _ in 0..200 {
        f32_row(m, val32(rng));
    }
}

// ── common ──────────────────────────────────────────────────────────────────────────

fn material_toks(x: &FluidMaterial) -> Vec<Tok> {
    vec![
        f(x.density),
        f(x.viscosity),
        f(x.surface_tension),
        f(x.speed_of_sound),
    ]
}

fn particle_toks(out: &mut Vec<Tok>, p: &FluidParticle) {
    v3(out, p.position);
    v3(out, p.velocity);
    v3(out, p.acceleration);
    out.extend([f(p.density), f(p.pressure), f(p.mass), i(p.phase)]);
    out.extend(p.conformation.iter().map(|&c| f(c)));
    out.extend([f(p.temperature), f(p.fuel)]);
}

fn random_particle(rng: &mut Rng) -> FluidParticle {
    let mut p = FluidParticle::new(rv3(rng), val(rng));
    p.velocity = rv3(rng);
    p.acceleration = rv3(rng);
    p.density = val(rng);
    p.pressure = val(rng);
    p.phase = u8::try_from(rng.int(0, 255)).unwrap_or(0);
    p.conformation = [val(rng), val(rng), val(rng)];
    p.temperature = val(rng);
    p.fuel = val(rng);
    p
}

fn soa_toks(x: &ParticleSoa) -> Vec<Tok> {
    let mut out = Vec::new();
    for c in [
        &x.pos_x, &x.pos_y, &x.pos_z, &x.vel_x, &x.vel_y, &x.vel_z, &x.accel_x, &x.accel_y,
        &x.accel_z,
    ] {
        col(&mut out, c);
    }
    for c in [&x.density, &x.pressure, &x.mass] {
        col(&mut out, c);
    }
    out.push(Tok::V(x.phase.len()));
    out.extend(x.phase.iter().map(|&p| i(p)));
    for c in [&x.conf_xx, &x.conf_xy, &x.conf_yy, &x.temperature, &x.fuel] {
        col(&mut out, c);
    }
    out
}

fn config_toks(c: &FluidConfig) -> Vec<Tok> {
    let mut out = vec![f(c.dt)];
    v3(&mut out, c.gravity);
    out.extend([f(c.smoothing_radius), f(c.gas_constant), f(c.rest_density)]);
    v3(&mut out, c.bounds_min);
    v3(&mut out, c.bounds_max);
    out.push(f(c.boundary_damping));
    out
}

fn common_rows(m: &mut Module, rng: &mut Rng) {
    for x in [
        FluidMaterial::WATER,
        FluidMaterial::OIL,
        FluidMaterial::HONEY,
        FluidMaterial::AIR,
        FluidMaterial::LAVA,
    ] {
        row(m, "FluidMaterial::debug", &material_toks(&x), &x);
    }
    for _ in 0..6 {
        let mut x = FluidMaterial::WATER;
        x.density = val(rng);
        x.viscosity = val(rng);
        x.surface_tension = val(rng);
        x.speed_of_sound = val(rng);
        row(m, "FluidMaterial::debug", &material_toks(&x), &x);
    }
    let fresh = FluidParticle::new_2d(0.5, 0.25, 0.001);
    let mut toks = Vec::new();
    particle_toks(&mut toks, &fresh);
    row(m, "FluidParticle::debug", &toks, &fresh);
    for _ in 0..8 {
        let p = random_particle(rng);
        let mut toks = Vec::new();
        particle_toks(&mut toks, &p);
        row(m, "FluidParticle::debug", &toks, &p);
    }
    let empty = ParticleSoa::new();
    row(m, "ParticleSoa::debug", &soa_toks(&empty), &empty);
    for n in [1_usize, 2, 3] {
        let ps: Vec<FluidParticle> = (0..n).map(|_| random_particle(rng)).collect();
        let mut x = ParticleSoa::from_aos(&ps);
        if n == 3 {
            x.mass.truncate(1);
            x.phase.push(7);
        }
        row(m, "ParticleSoa::debug", &soa_toks(&x), &x);
    }
    let mut arena = ParticleArena::new(64);
    let mut handles: Vec<ArenaHandle> = Vec::new();
    for len in [0_usize, 1, 5, 12] {
        if let Some(h) = arena.alloc(len) {
            handles.push(h);
        }
    }
    if let Some(h) = handles.get(1).copied() {
        arena.free(h);
    }
    arena.compact();
    if let Some(h) = arena.alloc(3) {
        handles.push(h);
    }
    for h in handles {
        let d = h.oracle_dump();
        #[allow(clippy::cast_possible_wrap)]
        let toks = [usz(d.start), usz(d.len), i(d.generation as i64)];
        row(m, "ArenaHandle::debug", &toks, &h);
    }
    let water = FluidConfig::water_2d();
    row(m, "FluidConfig::debug", &config_toks(&water), &water);
    for _ in 0..6 {
        let mut c = FluidConfig::water_2d();
        c.dt = val(rng);
        c.gravity = rv3(rng);
        c.smoothing_radius = val(rng);
        c.gas_constant = val(rng);
        c.rest_density = val(rng);
        c.bounds_min = rv3(rng);
        c.bounds_max = rv3(rng);
        c.boundary_damping = val(rng);
        row(m, "FluidConfig::debug", &config_toks(&c), &c);
    }
}

// ── buoyancy, shallow ───────────────────────────────────────────────────────────────

fn shallow_toks(w: &ShallowWater) -> Vec<Tok> {
    let mut out = vec![usz(w.nx), usz(w.ny), f(w.dx)];
    col(&mut out, &w.height);
    col(&mut out, &w.vx);
    col(&mut out, &w.vy);
    col(&mut out, &w.ground);
    out.extend([f(w.gravity), f(w.damping)]);
    col(&mut out, &w.manning_n);
    out.extend([
        f(w.dry_threshold),
        f(w.breaking_threshold),
        f(w.breaking_dissipation),
        f(w.dispersion_coeff),
    ]);
    out.extend([b(w.use_green_naghdi), b(w.use_riemann)]);
    let d = w.oracle_dump();
    for c in [
        &d.scratch_vx,
        &d.scratch_vy,
        &d.scratch_h,
        &d.scratch_lap,
        &d.scratch_line,
    ] {
        col(&mut out, c);
    }
    out
}

fn sediment_toks(c: &SedimentConfig) -> Vec<Tok> {
    vec![
        f(c.shields_critical),
        f(c.grain_diameter),
        f(c.sediment_density),
        f(c.water_density),
        f(c.erosion_rate),
        f(c.deposition_rate),
    ]
}

fn buoyancy_shallow_rows(m: &mut Module, rng: &mut Rng) {
    m.row(
        "DragCoefficient::debug",
        Class::A,
        &[],
        &[
            s(format!("{:?}", pravash::buoyancy::DragCoefficient)),
            s(format!("{:#?}", pravash::buoyancy::DragCoefficient)),
        ],
    );
    for x in [
        FlowRegime::Laminar,
        FlowRegime::Transitional,
        FlowRegime::Turbulent,
    ] {
        row(m, "FlowRegime::debug", &[e("FlowRegime", x)], &x);
    }
    for (nx, ny, steps, gn) in [(3, 3, 0, false), (4, 3, 1, false), (3, 4, 2, true)] {
        let mut w = ShallowWater::new(nx, ny, 0.25, 1.0).expect("a valid basin");
        w.add_disturbance(0.25, 0.25, 0.5, 0.2);
        w.use_green_naghdi = gn;
        if gn {
            w.dispersion_coeff = 0.3;
        }
        for _ in 0..steps {
            let _ = w.step(0.001);
        }
        row(m, "ShallowWater::debug", &shallow_toks(&w), &w);
    }
    let mut w = ShallowWater::new(3, 3, 0.5, 1.0).expect("a valid basin");
    for k in 0..9 {
        w.height[k] = val(rng);
        w.vx[k] = val(rng);
        w.ground[k] = val(rng);
        w.manning_n[k] = val(rng);
    }
    w.gravity = val(rng);
    w.dry_threshold = val(rng);
    w.use_riemann = false;
    row(m, "ShallowWater::debug", &shallow_toks(&w), &w);
    let d = SedimentConfig::default();
    row(m, "SedimentConfig::debug", &sediment_toks(&d), &d);
    for _ in 0..5 {
        let mut c = SedimentConfig::default();
        c.shields_critical = val(rng);
        c.grain_diameter = val(rng);
        c.sediment_density = val(rng);
        c.water_density = val(rng);
        c.erosion_rate = val(rng);
        c.deposition_rate = val(rng);
        row(m, "SedimentConfig::debug", &sediment_toks(&c), &c);
    }
}

// ── grid ────────────────────────────────────────────────────────────────────────────

fn grid_config_toks(c: &GridConfig) -> Vec<Tok> {
    vec![
        f(c.dt),
        f(c.viscosity),
        usz(c.diffusion_iterations),
        usz(c.pressure_iterations),
        e("BoundaryCondition", c.boundary),
        f(c.vorticity_confinement),
        f(c.buoyancy_alpha),
        f(c.ambient_density),
        b(c.use_maccormack),
        b(c.use_bfecc),
        f(c.smagorinsky_cs),
        b(c.use_multigrid),
        b(c.use_cg),
    ]
}

fn fluid_grid_toks(g: &FluidGrid) -> Vec<Tok> {
    let mut out = vec![usz(g.nx), usz(g.ny), f(g.dx)];
    for c in [&g.vx, &g.vy, &g.pressure, &g.density] {
        col(&mut out, c);
    }
    let d = g.oracle_dump();
    for c in [&d.scratch_a, &d.scratch_b, &d.scratch_c, &d.diffuse_rhs] {
        col(&mut out, c);
    }
    out.push(Tok::V(d.mg_levels.len()));
    for lv in &d.mg_levels {
        col(&mut out, &lv.residual);
        col(&mut out, &lv.coarse_rhs);
        col(&mut out, &lv.coarse_p);
    }
    out
}

fn kep_toks(k: &KEpsilon) -> Vec<Tok> {
    let mut out = Vec::new();
    col(&mut out, &k.k);
    col(&mut out, &k.epsilon);
    out.extend([f(k.c_mu), f(k.c_eps1), f(k.c_eps2)]);
    let d = k.oracle_dump();
    col(&mut out, &d.scratch_k);
    col(&mut out, &d.scratch_eps);
    out
}

fn mac_toks(g: &MacGrid) -> Vec<Tok> {
    let mut out = vec![usz(g.nx), usz(g.ny), f(g.dx)];
    for c in [&g.vx, &g.vy, &g.pressure, &g.density] {
        col(&mut out, c);
    }
    let d = g.oracle_dump();
    col(&mut out, &d.scratch_div);
    col(&mut out, &d.scratch_density);
    out
}

fn grid_rows(m: &mut Module, rng: &mut Rng) {
    for x in [
        BoundaryCondition::NoSlip,
        BoundaryCondition::FreeSlip,
        BoundaryCondition::Periodic,
    ] {
        row(
            m,
            "BoundaryCondition::debug",
            &[e("BoundaryCondition", x)],
            &x,
        );
    }
    let d = GridConfig::default();
    row(m, "GridConfig::debug", &grid_config_toks(&d), &d);
    for k in 0..5 {
        let mut c = GridConfig::default();
        c.dt = val(rng);
        c.viscosity = val(rng);
        c.diffusion_iterations = usize::try_from(rng.int(0, 100)).unwrap_or(0);
        c.pressure_iterations = usize::try_from(rng.int(0, 100)).unwrap_or(0);
        c.boundary = [
            BoundaryCondition::NoSlip,
            BoundaryCondition::FreeSlip,
            BoundaryCondition::Periodic,
        ][k % 3];
        c.vorticity_confinement = val(rng);
        c.buoyancy_alpha = val(rng);
        c.ambient_density = val(rng);
        c.use_maccormack = k % 2 == 0;
        c.use_bfecc = k % 3 == 1;
        c.smagorinsky_cs = val(rng);
        c.use_multigrid = k == 2;
        c.use_cg = k == 4;
        row(m, "GridConfig::debug", &grid_config_toks(&c), &c);
    }
    // Fresh grids, and grids after steps: the scratch filled, and the multigrid levels at 8 × 8.
    for (nx, ny, steps, mg) in [(4, 4, 0, false), (4, 5, 1, false), (8, 8, 1, true)] {
        let mut g = FluidGrid::new(nx, ny, 0.1).expect("a valid grid");
        for k in 0..nx * ny {
            g.vx[k] = rng.uniform(-1.0, 1.0);
            g.vy[k] = rng.uniform(-1.0, 1.0);
            g.density[k] = rng.uniform(0.0, 2.0);
        }
        let mut c = GridConfig::default();
        c.use_multigrid = mg;
        c.buoyancy_alpha = 0.5;
        for _ in 0..steps {
            let _ = g.step(&c);
        }
        row(m, "FluidGrid::debug", &fluid_grid_toks(&g), &g);
    }
    let mut ke = KEpsilon::new(2, 2);
    row(m, "KEpsilon::debug", &kep_toks(&ke), &ke);
    let vx = [0.5, -0.25, 1.0, 0.0];
    let vy = [0.1, 0.2, -0.3, 0.4];
    ke.step(&vx, &vy, 2, 2, 0.1, 0.01);
    row(m, "KEpsilon::debug", &kep_toks(&ke), &ke);
    for x in &mut ke.k {
        *x = val(rng);
    }
    ke.c_mu = val(rng);
    row(m, "KEpsilon::debug", &kep_toks(&ke), &ke);
    for steps in [0, 1] {
        let mut g = MacGrid::new(4, 4, 0.25).expect("a valid grid");
        for x in &mut g.vx {
            *x = rng.uniform(-1.0, 1.0);
        }
        for x in &mut g.density {
            *x = rng.uniform(0.0, 2.0);
        }
        for _ in 0..steps {
            let _ = g.step(&GridConfig::default());
        }
        row(m, "MacGrid::debug", &mac_toks(&g), &g);
    }
}

// ── vof, phase field ────────────────────────────────────────────────────────────────

fn vof_toks(v: &VofField) -> Vec<Tok> {
    let mut out = vec![usz(v.nx), usz(v.ny), f(v.dx)];
    col(&mut out, &v.alpha);
    col(&mut out, &v.oracle_dump().scratch);
    out
}

fn pf_toks(p: &PhaseField) -> Vec<Tok> {
    let mut out = vec![usz(p.nx), usz(p.ny), f(p.dx)];
    col(&mut out, &p.phi);
    out.extend([f(p.mobility), f(p.epsilon)]);
    col(&mut out, &p.oracle_dump().scratch);
    out
}

fn interface_rows(m: &mut Module, rng: &mut Rng) {
    for steps in [0, 1] {
        let mut v = VofField::new(4, 4, 0.1).expect("a valid field");
        v.fill_circle(0.2, 0.2, 0.12);
        let u: Vec<f64> = (0..16).map(|_| rng.uniform(-0.5, 0.5)).collect();
        for _ in 0..steps {
            let _ = v.advect(&u, &u, 0.01);
        }
        row(m, "VofField::debug", &vof_toks(&v), &v);
    }
    let mut v = VofField::new(4, 4, 0.5).expect("a valid field");
    v.alpha = rcol(rng, 16);
    row(m, "VofField::debug", &vof_toks(&v), &v);
    for steps in [0, 1] {
        let mut p = PhaseField::new(4, 4, 0.1).expect("a valid field");
        for x in &mut p.phi {
            *x = rng.uniform(-1.0, 1.0);
        }
        let u: Vec<f64> = (0..16).map(|_| rng.uniform(-0.5, 0.5)).collect();
        for _ in 0..steps {
            let _ = p.step(&u, &u, 0.001);
        }
        row(m, "PhaseField::debug", &pf_toks(&p), &p);
    }
    let mut p = PhaseField::new(4, 4, 0.5).expect("a valid field");
    p.phi = rcol(rng, 16);
    p.mobility = val(rng);
    p.epsilon = val(rng);
    row(m, "PhaseField::debug", &pf_toks(&p), &p);
}

// ── sph ─────────────────────────────────────────────────────────────────────────────

fn phase_toks(out: &mut Vec<Tok>, p: &PhaseProperties) {
    out.extend([f(p.rest_density), f(p.gas_constant), f(p.viscosity)]);
}

fn random_phase(rng: &mut Rng) -> PhaseProperties {
    PhaseProperties {
        rest_density: val(rng),
        gas_constant: val(rng),
        viscosity: val(rng),
    }
}

fn nn_toks(out: &mut Vec<Tok>, x: &NonNewtonianViscosity) {
    match *x {
        NonNewtonianViscosity::PowerLaw {
            consistency,
            power_index,
        } => {
            out.push(Tok::E("NonNewtonianViscosity", "PowerLaw".to_string()));
            out.extend([f(consistency), f(power_index)]);
        }
        NonNewtonianViscosity::Bingham {
            yield_stress,
            plastic_viscosity,
        } => {
            out.push(Tok::E("NonNewtonianViscosity", "Bingham".to_string()));
            out.extend([f(yield_stress), f(plastic_viscosity)]);
        }
        NonNewtonianViscosity::HerschelBulkley {
            yield_stress,
            consistency,
            power_index,
        } => {
            out.push(Tok::E(
                "NonNewtonianViscosity",
                "HerschelBulkley".to_string(),
            ));
            out.extend([f(yield_stress), f(consistency), f(power_index)]);
        }
        _ => out.push(Tok::E("NonNewtonianViscosity", "Unknown".to_string())),
    }
}

fn secondary_toks(p: &SecondaryParticle) -> Vec<Tok> {
    let mut out = Vec::new();
    v3(&mut out, p.position);
    v3(&mut out, p.velocity);
    out.push(f(p.lifetime));
    out.push(e("SecondaryType", p.kind));
    out
}

fn sph_rows(m: &mut Module, rng: &mut Rng) {
    for _ in 0..5 {
        let p = random_phase(rng);
        let mut toks = Vec::new();
        phase_toks(&mut toks, &p);
        row(m, "PhaseProperties::debug", &toks, &p);
    }
    let mut configs = vec![
        MultiPhaseConfig::single(1000.0, 2000.0, 0.001),
        MultiPhaseConfig::two_phase(random_phase(rng), random_phase(rng), val(rng)),
    ];
    let mut none = MultiPhaseConfig::single(1.0, 1.0, 1.0);
    none.phases.clear();
    configs.push(none);
    let mut three = MultiPhaseConfig::two_phase(random_phase(rng), random_phase(rng), 0.05);
    three.phases.push(random_phase(rng));
    configs.push(three);
    for c in configs {
        let mut toks = vec![Tok::V(c.phases.len())];
        for p in &c.phases {
            phase_toks(&mut toks, p);
        }
        toks.push(f(c.interface_tension));
        row(m, "MultiPhaseConfig::debug", &toks, &c);
    }
    for _ in 0..4 {
        let v = ViscoelasticConfig::new(val(rng), val(rng));
        row(
            m,
            "ViscoelasticConfig::debug",
            &[f(v.elastic_modulus), f(v.relaxation_time)],
            &v,
        );
        let h = HeatConfig::new(val(rng));
        row(m, "HeatConfig::debug", &[f(h.diffusivity)], &h);
        let c = PhaseChangeConfig::new(
            val(rng),
            val(rng),
            val(rng),
            u8::try_from(rng.int(0, 255)).unwrap_or(0),
            u8::try_from(rng.int(0, 255)).unwrap_or(0),
        );
        let toks = [
            f(c.transition_temperature),
            f(c.latent_heat),
            f(c.heat_capacity),
            i(c.phase_below),
            i(c.phase_above),
        ];
        row(m, "PhaseChangeConfig::debug", &toks, &c);
        let mut b = CombustionConfig::default();
        b.ignition_temperature = val(rng);
        b.reaction_rate = val(rng);
        b.heat_release = val(rng);
        row(
            m,
            "CombustionConfig::debug",
            &[
                f(b.ignition_temperature),
                f(b.reaction_rate),
                f(b.heat_release),
            ],
            &b,
        );
    }
    let w = PhaseChangeConfig::water_ice(1, 0);
    let toks = [
        f(w.transition_temperature),
        f(w.latent_heat),
        f(w.heat_capacity),
        i(w.phase_below),
        i(w.phase_above),
    ];
    row(m, "PhaseChangeConfig::debug", &toks, &w);
    let d = CombustionConfig::default();
    row(
        m,
        "CombustionConfig::debug",
        &[
            f(d.ignition_temperature),
            f(d.reaction_rate),
            f(d.heat_release),
        ],
        &d,
    );
    for _ in 0..2 {
        for x in [
            NonNewtonianViscosity::PowerLaw {
                consistency: val(rng),
                power_index: val(rng),
            },
            NonNewtonianViscosity::Bingham {
                yield_stress: val(rng),
                plastic_viscosity: val(rng),
            },
            NonNewtonianViscosity::HerschelBulkley {
                yield_stress: val(rng),
                consistency: val(rng),
                power_index: val(rng),
            },
        ] {
            let mut toks = Vec::new();
            nn_toks(&mut toks, &x);
            row(m, "NonNewtonianViscosity::debug", &toks, &x);
        }
    }
    for x in [
        SecondaryType::Foam,
        SecondaryType::Spray,
        SecondaryType::Bubble,
    ] {
        row(m, "SecondaryType::debug", &[e("SecondaryType", x)], &x);
        let p = SecondaryParticle {
            position: rv3(rng),
            velocity: rv3(rng),
            lifetime: val(rng),
            kind: x,
        };
        row(m, "SecondaryParticle::debug", &secondary_toks(&p), &p);
    }
}

// ── coupling, mpm ───────────────────────────────────────────────────────────────────

fn shape_toks(out: &mut Vec<Tok>, x: &BodyShape) {
    match *x {
        BodyShape::Sphere { radius } => {
            out.push(Tok::E("BodyShape", "Sphere".to_string()));
            out.push(f(radius));
        }
        BodyShape::Box { half_extents } => {
            out.push(Tok::E("BodyShape", "Box".to_string()));
            out.extend(half_extents.iter().map(|&h| f(h)));
        }
        _ => out.push(Tok::E("BodyShape", "Unknown".to_string())),
    }
}

fn model_toks(out: &mut Vec<Tok>, x: &ConstitutiveModel) {
    match *x {
        ConstitutiveModel::Fluid {
            bulk_modulus,
            viscosity,
        } => {
            out.push(Tok::E("ConstitutiveModel", "Fluid".to_string()));
            out.extend([f(bulk_modulus), f(viscosity)]);
        }
        ConstitutiveModel::NeoHookean {
            youngs_modulus,
            poisson_ratio,
        } => {
            out.push(Tok::E("ConstitutiveModel", "NeoHookean".to_string()));
            out.extend([f(youngs_modulus), f(poisson_ratio)]);
        }
        ConstitutiveModel::DruckerPrager {
            youngs_modulus,
            poisson_ratio,
            friction_angle,
        } => {
            out.push(Tok::E("ConstitutiveModel", "DruckerPrager".to_string()));
            out.extend([f(youngs_modulus), f(poisson_ratio), f(friction_angle)]);
        }
        _ => out.push(Tok::E("ConstitutiveModel", "Unknown".to_string())),
    }
}

fn coupling_mpm_rows(m: &mut Module, rng: &mut Rng) {
    let shapes = [
        BodyShape::Sphere { radius: 0.5 },
        BodyShape::Box {
            half_extents: [0.25, 0.5, 1.0],
        },
        BodyShape::Sphere { radius: val(rng) },
        BodyShape::Box {
            half_extents: [val(rng), val(rng), val(rng)],
        },
    ];
    for x in shapes {
        let mut toks = Vec::new();
        shape_toks(&mut toks, &x);
        row(m, "BodyShape::debug", &toks, &x);
        let mut body = RigidBody::new(rv3(rng), val(rng), x);
        body.velocity = rv3(rng);
        body.force = rv3(rng);
        let mut toks = Vec::new();
        v3(&mut toks, body.position);
        v3(&mut toks, body.velocity);
        toks.push(f(body.mass));
        shape_toks(&mut toks, &body.shape);
        v3(&mut toks, body.force);
        row(m, "RigidBody::debug", &toks, &body);
    }
    let models = [
        ConstitutiveModel::Fluid {
            bulk_modulus: val(rng),
            viscosity: val(rng),
        },
        ConstitutiveModel::NeoHookean {
            youngs_modulus: val(rng),
            poisson_ratio: val(rng),
        },
        ConstitutiveModel::DruckerPrager {
            youngs_modulus: val(rng),
            poisson_ratio: val(rng),
            friction_angle: val(rng),
        },
        ConstitutiveModel::Fluid {
            bulk_modulus: 1000.0,
            viscosity: 0.1,
        },
    ];
    for x in models {
        let mut toks = Vec::new();
        model_toks(&mut toks, &x);
        row(m, "ConstitutiveModel::debug", &toks, &x);
        let mut p = MpmParticle::new(x, val(rng));
        if rng.int(0, 1) == 1 {
            p.deformation_grad = [val(rng), val(rng), val(rng), val(rng)];
        }
        let mut toks: Vec<Tok> = p.deformation_grad.iter().map(|&g| f(g)).collect();
        model_toks(&mut toks, &p.model);
        toks.push(f(p.volume_0));
        row(m, "MpmParticle::debug", &toks, &p);
    }
}

// ── compute ─────────────────────────────────────────────────────────────────────────

fn sph_params_toks(p: &SphKernelParams) -> Vec<Tok> {
    let mut out: Vec<Tok> = [
        p.smoothing_radius,
        p.rest_density,
        p.gas_constant,
        p.viscosity,
    ]
    .iter()
    .map(|&x| f(f64::from(x)))
    .collect();
    out.extend(p.gravity.iter().map(|&g| f(f64::from(g))));
    out.push(f(f64::from(p.dt)));
    out.push(i(p.particle_count));
    out
}

fn grid_params_toks(p: &GridKernelParams) -> Vec<Tok> {
    let mut out = vec![i(p.nx), i(p.ny)];
    out.extend(
        [p.dx, p.dt, p.viscosity, p.buoyancy_alpha, p.ambient_density]
            .iter()
            .map(|&x| f(f64::from(x))),
    );
    out
}

fn shallow_params_toks(p: &ShallowKernelParams) -> Vec<Tok> {
    let mut out = vec![i(p.nx), i(p.ny)];
    out.extend(
        [p.dx, p.dt, p.gravity, p.damping, p.dry_threshold]
            .iter()
            .map(|&x| f(f64::from(x))),
    );
    out
}

fn compute_rows(m: &mut Module, rng: &mut Rng) {
    for n in [0_usize, 1, 2] {
        let ps: Vec<FluidParticle> = (0..n).map(|_| random_particle(rng)).collect();
        let p = PackedParticles::from_soa(&ParticleSoa::from_aos(&ps));
        let mut toks = vec![i(p.count)];
        col32(&mut toks, &p.position_mass);
        col32(&mut toks, &p.velocity);
        col32(&mut toks, &p.density_pressure);
        row(m, "PackedParticles::debug", &toks, &p);
    }
    let d = SphKernelParams::default();
    row(m, "SphKernelParams::debug", &sph_params_toks(&d), &d);
    let g = GridKernelParams::default();
    row(m, "GridKernelParams::debug", &grid_params_toks(&g), &g);
    let h = ShallowKernelParams::default();
    row(
        m,
        "ShallowKernelParams::debug",
        &shallow_params_toks(&h),
        &h,
    );
    for _ in 0..5 {
        let mut p = SphKernelParams::default();
        p.smoothing_radius = val32(rng);
        p.rest_density = val32(rng);
        p.gas_constant = val32(rng);
        p.viscosity = val32(rng);
        p.gravity = [val32(rng), val32(rng), val32(rng)];
        p.dt = val32(rng);
        p.particle_count = u32::try_from(rng.int(0, 0xFFFF_FFFF)).unwrap_or(0);
        row(m, "SphKernelParams::debug", &sph_params_toks(&p), &p);
        let mut g = GridKernelParams::default();
        g.nx = u32::try_from(rng.int(0, 4096)).unwrap_or(0);
        g.ny = u32::try_from(rng.int(0, 0xFFFF_FFFF)).unwrap_or(0);
        g.dx = val32(rng);
        g.dt = val32(rng);
        g.viscosity = val32(rng);
        g.buoyancy_alpha = val32(rng);
        g.ambient_density = val32(rng);
        row(m, "GridKernelParams::debug", &grid_params_toks(&g), &g);
        let mut h = ShallowKernelParams::default();
        h.nx = u32::try_from(rng.int(0, 4096)).unwrap_or(0);
        h.ny = u32::try_from(rng.int(0, 4096)).unwrap_or(0);
        h.dx = val32(rng);
        h.dt = val32(rng);
        h.gravity = val32(rng);
        h.damping = val32(rng);
        h.dry_threshold = val32(rng);
        row(
            m,
            "ShallowKernelParams::debug",
            &shallow_params_toks(&h),
            &h,
        );
    }
    for x in [
        ComputeOp::SphDensity,
        ComputeOp::SphStep,
        ComputeOp::GridAdvect,
        ComputeOp::ShallowStep,
    ] {
        row(m, "ComputeOp::debug", &[e("ComputeOp", x)], &x);
    }
    for (nv, np) in [(0_usize, 0_usize), (3, 1), (2, 4)] {
        let c = NeuralCorrection {
            velocity_delta: rcol32(rng, nv),
            pressure_delta: rcol32(rng, np),
        };
        let mut toks = Vec::new();
        col32(&mut toks, &c.velocity_delta);
        col32(&mut toks, &c.pressure_delta);
        row(m, "NeuralCorrection::debug", &toks, &c);
    }
}

// ── after steps ─────────────────────────────────────────────────────────────────────

/// The rows above load each state's scratch from the oracle's dumps. These take a full state, a step's arguments
/// and a step count, and give the Debug text after the steps, so the port's own steps must leave Rust's scratch.
fn after_rows(m: &mut Module, rng: &mut Rng) {
    for (nx, ny, gn, riemann, steps) in [
        (4, 3, false, true, 1_i64),
        (3, 4, true, true, 2),
        (4, 4, false, false, 3),
    ] {
        let mut w = ShallowWater::new(nx, ny, 0.25, 1.0).expect("a valid basin");
        w.add_disturbance(0.3, 0.3, 0.4, 0.2);
        w.use_green_naghdi = gn;
        w.use_riemann = riemann;
        if gn {
            w.dispersion_coeff = 0.3;
        }
        let mut toks = shallow_toks(&w);
        toks.extend([f(0.001), i(steps)]);
        for _ in 0..steps {
            let _ = w.step(0.001);
        }
        m.row(
            "ShallowWater::debug_after",
            Class::A,
            &toks,
            &[s(format!("{w:?}")), s(format!("{w:#?}"))],
        );
    }
    for (nx, ny, mg, cg, bfecc, steps) in [
        (4, 4, false, false, true, 1_i64),
        (8, 8, true, false, false, 2),
        (5, 4, false, true, false, 1),
    ] {
        let mut g = FluidGrid::new(nx, ny, 0.1).expect("a valid grid");
        for k in 0..nx * ny {
            g.vx[k] = rng.uniform(-1.0, 1.0);
            g.vy[k] = rng.uniform(-1.0, 1.0);
            g.density[k] = rng.uniform(0.0, 2.0);
        }
        let mut c = GridConfig::default();
        c.use_multigrid = mg;
        c.use_cg = cg;
        c.use_bfecc = bfecc;
        c.buoyancy_alpha = 0.5;
        c.vorticity_confinement = 0.1;
        let mut toks = fluid_grid_toks(&g);
        toks.extend(grid_config_toks(&c));
        toks.push(i(steps));
        for _ in 0..steps {
            let _ = g.step(&c);
        }
        m.row(
            "FluidGrid::debug_after",
            Class::A,
            &toks,
            &[s(format!("{g:?}")), s(format!("{g:#?}"))],
        );
    }
    let mut ke = KEpsilon::new(3, 3);
    let u: Vec<f64> = (0..9).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let v: Vec<f64> = (0..9).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let mut toks = kep_toks(&ke);
    col(&mut toks, &u);
    col(&mut toks, &v);
    toks.extend([usz(3), usz(3), f(0.1), f(0.01), i(2)]);
    for _ in 0..2 {
        ke.step(&u, &v, 3, 3, 0.1, 0.01);
    }
    m.row(
        "KEpsilon::debug_after",
        Class::A,
        &toks,
        &[s(format!("{ke:?}")), s(format!("{ke:#?}"))],
    );
    let mut g = MacGrid::new(4, 4, 0.25).expect("a valid grid");
    for x in &mut g.vx {
        *x = rng.uniform(-1.0, 1.0);
    }
    for x in &mut g.density {
        *x = rng.uniform(0.0, 2.0);
    }
    let mut c = GridConfig::default();
    c.buoyancy_alpha = 0.5;
    let mut toks = mac_toks(&g);
    toks.extend(grid_config_toks(&c));
    toks.push(i(2));
    for _ in 0..2 {
        let _ = g.step(&c);
    }
    m.row(
        "MacGrid::debug_after",
        Class::A,
        &toks,
        &[s(format!("{g:?}")), s(format!("{g:#?}"))],
    );
    let mut vf = VofField::new(4, 4, 0.1).expect("a valid field");
    vf.fill_circle(0.2, 0.2, 0.12);
    let u: Vec<f64> = (0..16).map(|_| rng.uniform(-0.5, 0.5)).collect();
    let mut toks = vof_toks(&vf);
    col(&mut toks, &u);
    col(&mut toks, &u);
    toks.extend([f(0.05), i(2)]);
    for _ in 0..2 {
        let _ = vf.advect(&u, &u, 0.05);
    }
    m.row(
        "VofField::debug_after",
        Class::A,
        &toks,
        &[s(format!("{vf:?}")), s(format!("{vf:#?}"))],
    );
    let mut pf = PhaseField::new(4, 4, 0.1).expect("a valid field");
    for x in &mut pf.phi {
        *x = rng.uniform(-1.0, 1.0);
    }
    let mut toks = pf_toks(&pf);
    col(&mut toks, &u);
    col(&mut toks, &u);
    toks.extend([f(0.001), i(2)]);
    for _ in 0..2 {
        let _ = pf.step(&u, &u, 0.001);
    }
    m.row(
        "PhaseField::debug_after",
        Class::A,
        &toks,
        &[s(format!("{pf:?}")), s(format!("{pf:#?}"))],
    );
}

pub fn emit() -> Module {
    let mut m = Module::new("debug");
    let mut rng = Rng::new(0x70_c0_0d);
    f64_rows(&mut m, &mut rng);
    f32_rows(&mut m, &mut rng);
    common_rows(&mut m, &mut rng);
    buoyancy_shallow_rows(&mut m, &mut rng);
    grid_rows(&mut m, &mut rng);
    interface_rows(&mut m, &mut rng);
    sph_rows(&mut m, &mut rng);
    coupling_mpm_rows(&mut m, &mut rng);
    compute_rows(&mut m, &mut rng);
    after_rows(&mut m, &mut rng);
    m
}
