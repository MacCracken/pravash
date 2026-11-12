//! `common.tsv` — `rust-old/src/common.rs` (M1 part 3): materials, particles, the configuration, the SoA
//! layout and the particle arena.
//!
//! Token layouts (the harness parses the same):
//! - a material is its 4 fields in declaration order;
//! - a particle is its 18 values: position, velocity and acceleration (x, y, z each), density, pressure,
//!   mass, `i:` phase, the conformation's 3 values, temperature, fuel;
//! - a config is its 14 values: dt, gravity (3), smoothing radius, gas constant, rest density, bounds_min (3),
//!   bounds_max (3), boundary damping.
//!
//! `validate`, `check_cfl` and `custom` give `-` for `Ok` or `E:<Variant>`. The arena rows replay one
//! scenario each: the capacity, then ops of 4 integers (`code a b c`), and give each op's result, then the
//! arena's private state from the oracle dump (watermark, generation, free list, every slot's mass below the
//! watermark). Op codes:
//! - 0 `alloc(a)`: `i:` Some, then the handle's start, len, generation (0s for None). Every particle of a new
//!   block then gets mass = the op's index + 1 through `get_mut`, so a `compact` shows where each block went;
//! - 1 `free` of issued handle a (mod the count; every Some is issued, stale ones included): `active_count`;
//! - 2 `compact`: its result;
//! - 3 `get` of issued handle a: the slice length;
//! - 4 `free` of a foreign handle (start a, len b, generation c, from another arena): `active_count`;
//! - 5 `get` of a foreign handle (a, b, c): the slice length.

use hisab::DVec3;
use pravash::{ArenaHandle, FluidConfig, FluidMaterial, FluidParticle, ParticleArena, ParticleSoa};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, Tok, err, f, i};

fn usz(x: usize) -> Tok {
    i(i64::try_from(x).unwrap_or(i64::MAX))
}

fn gen_tok(x: u64) -> Tok {
    // The u64 generation as its 64-bit word (two's complement), as the Cyrius word holds it.
    i(i64::from_ne_bytes(x.to_ne_bytes()))
}

fn material(m: &FluidMaterial) -> Vec<Tok> {
    vec![
        f(m.density),
        f(m.viscosity),
        f(m.surface_tension),
        f(m.speed_of_sound),
    ]
}

fn particle(p: &FluidParticle) -> Vec<Tok> {
    vec![
        f(p.position.x),
        f(p.position.y),
        f(p.position.z),
        f(p.velocity.x),
        f(p.velocity.y),
        f(p.velocity.z),
        f(p.acceleration.x),
        f(p.acceleration.y),
        f(p.acceleration.z),
        f(p.density),
        f(p.pressure),
        f(p.mass),
        i(p.phase),
        f(p.conformation[0]),
        f(p.conformation[1]),
        f(p.conformation[2]),
        f(p.temperature),
        f(p.fuel),
    ]
}

fn config(c: &FluidConfig) -> Vec<Tok> {
    vec![
        f(c.dt),
        f(c.gravity.x),
        f(c.gravity.y),
        f(c.gravity.z),
        f(c.smoothing_radius),
        f(c.gas_constant),
        f(c.rest_density),
        f(c.bounds_min.x),
        f(c.bounds_min.y),
        f(c.bounds_min.z),
        f(c.bounds_max.x),
        f(c.bounds_max.y),
        f(c.bounds_max.z),
        f(c.boundary_damping),
    ]
}

fn unit(r: pravash::error::Result<()>) -> Vec<Tok> {
    match r {
        Ok(()) => vec![],
        Err(x) => vec![err(&x)],
    }
}

/// Values that sit at a validation boundary, beside the specials.
fn edges() -> Vec<f64> {
    let mut v = specials();
    v.extend([
        -f64::MIN_POSITIVE,
        f64::EPSILON,
        0.5,
        1.0 - f64::EPSILON / 2.0,
        1.0 + f64::EPSILON,
        1e-300,
        1e300,
        -1e300,
    ]);
    v
}

fn custom_rows(m: &mut Module, rng: &mut Rng) {
    let base = [1000.0, 0.001, 0.072, 1480.0];
    let mut cases: Vec<[f64; 4]> = vec![base];
    for k in 0..4 {
        for &x in &edges() {
            let mut c = base;
            c[k] = x;
            cases.push(c);
        }
    }
    // Two fields bad at once: the first one in Rust's order decides the variant.
    for k in 0..4 {
        for l in (k + 1)..4 {
            for &(x, y) in &[
                (-1.0, -1.0),
                (f64::NAN, -1.0),
                (-1.0, f64::INFINITY),
                (0.0, 0.0),
            ] {
                let mut c = base;
                c[k] = x;
                c[l] = y;
                cases.push(c);
            }
        }
    }
    for _ in 0..200 {
        cases.push([
            rng.uniform(-10.0, 3000.0),
            rng.uniform(-0.01, 100.0),
            rng.uniform(-0.01, 0.5),
            rng.uniform(-10.0, 2000.0),
        ]);
    }
    for c in &cases {
        let out = match FluidMaterial::custom(c[0], c[1], c[2], c[3]) {
            Ok(mat) => material(&mat),
            Err(x) => vec![err(&x)],
        };
        m.row(
            "FluidMaterial::custom",
            Class::A,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &out,
        );
    }
}

fn particle_rows(m: &mut Module, rng: &mut Rng) {
    let sp = specials();
    let mut positions: Vec<[f64; 3]> = Vec::new();
    for &x in &sp {
        positions.push([x, 0.5, -0.25]);
        positions.push([0.25, x, 0.0]);
        positions.push([0.0, 0.0, x]);
    }
    for _ in 0..60 {
        positions.push([
            rng.signed_log_uniform(-20.0, 20.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(-1.0, 1.0),
        ]);
    }
    let masses = [0.0, 0.001, 1.0, -1.0, f64::NAN, f64::INFINITY];
    for (n, p) in positions.iter().enumerate() {
        let mass = masses[n % masses.len()];
        let q = FluidParticle::new(DVec3::new(p[0], p[1], p[2]), mass);
        m.row(
            "FluidParticle::new",
            Class::A,
            &[f(p[0]), f(p[1]), f(p[2]), f(mass)],
            &particle(&q),
        );
        let r = FluidParticle::new_2d(p[0], p[1], mass);
        m.row(
            "FluidParticle::new_2d",
            Class::A,
            &[f(p[0]), f(p[1]), f(mass)],
            &particle(&r),
        );
    }

    // Speed and energy: one-sided specials and ranges, and wide magnitudes (overflow to ∞ in the squares).
    let mut vels: Vec<[f64; 4]> = Vec::new();
    for &x in &sp {
        vels.push([x, 4.0, 0.0, 2.0]);
        vels.push([3.0, x, -1.0, 0.5]);
        vels.push([1.0, 2.0, x, x]);
    }
    for _ in 0..200 {
        vels.push([
            rng.uniform(-50.0, 50.0),
            rng.uniform(-50.0, 50.0),
            rng.uniform(-50.0, 50.0),
            rng.uniform(0.0, 10.0),
        ]);
    }
    for _ in 0..100 {
        vels.push([
            rng.signed_log_uniform(-200.0, 200.0),
            rng.signed_log_uniform(-200.0, 200.0),
            rng.signed_log_uniform(-200.0, 200.0),
            rng.log_uniform(-10.0, 10.0),
        ]);
    }
    for v in &vels {
        let mut p = FluidParticle::new_2d(0.0, 0.0, v[3]);
        p.velocity = DVec3::new(v[0], v[1], v[2]);
        let ins = [f(v[0]), f(v[1]), f(v[2]), f(v[3])];
        m.row(
            "FluidParticle::speed_squared",
            Class::B,
            &ins,
            &[f(p.speed_squared())],
        );
        m.row("FluidParticle::speed", Class::B, &ins, &[f(p.speed())]);
        m.row(
            "FluidParticle::kinetic_energy",
            Class::B,
            &ins,
            &[f(p.kinetic_energy())],
        );
    }

    // Distances, and the SoA's distance_squared on the same pairs.
    let mut pairs: Vec<[f64; 6]> = Vec::new();
    for &x in &sp {
        pairs.push([x, 0.0, 0.0, 3.0, 4.0, 0.0]);
        pairs.push([0.0, 0.0, 0.0, 3.0, x, 12.0]);
    }
    for _ in 0..200 {
        let mut q = [0.0; 6];
        for c in &mut q {
            *c = rng.uniform(-5.0, 5.0);
        }
        pairs.push(q);
    }
    for _ in 0..60 {
        let mut q = [0.0; 6];
        for c in &mut q {
            *c = rng.signed_log_uniform(-160.0, 160.0);
        }
        pairs.push(q);
    }
    for q in &pairs {
        let a = FluidParticle::new(DVec3::new(q[0], q[1], q[2]), 1.0);
        let b = FluidParticle::new(DVec3::new(q[3], q[4], q[5]), 1.0);
        let ins: Vec<Tok> = q.iter().map(|&x| f(x)).collect();
        m.row(
            "FluidParticle::distance_squared_to",
            Class::B,
            &ins,
            &[f(a.distance_squared_to(&b))],
        );
        m.row(
            "FluidParticle::distance_to",
            Class::B,
            &ins,
            &[f(a.distance_to(&b))],
        );
        let soa = ParticleSoa::from_aos(&[a, b]);
        m.row(
            "ParticleSoa::distance_squared",
            Class::B,
            &ins,
            &[f(soa.distance_squared(0, 1))],
        );
    }
}

fn config_rows(m: &mut Module, rng: &mut Rng) {
    m.row(
        "FluidConfig::water_2d",
        Class::A,
        &[],
        &config(&FluidConfig::water_2d()),
    );

    // One field at a time over the edge values, in validation order, then pairs and random configs.
    let set = |c: &mut FluidConfig, k: usize, x: f64| match k {
        0 => c.dt = x,
        1 => c.gravity.x = x,
        2 => c.gravity.y = x,
        3 => c.gravity.z = x,
        4 => c.smoothing_radius = x,
        5 => c.gas_constant = x,
        6 => c.rest_density = x,
        7 => c.bounds_min.x = x,
        8 => c.bounds_min.y = x,
        9 => c.bounds_min.z = x,
        10 => c.bounds_max.x = x,
        11 => c.bounds_max.y = x,
        12 => c.bounds_max.z = x,
        _ => c.boundary_damping = x,
    };
    let mut cases: Vec<FluidConfig> = vec![FluidConfig::water_2d()];
    for k in 0..14 {
        for &x in &edges() {
            let mut c = FluidConfig::water_2d();
            set(&mut c, k, x);
            cases.push(c);
        }
    }
    for k in 0..14 {
        for l in (k + 1)..14 {
            let mut c = FluidConfig::water_2d();
            set(&mut c, k, f64::NAN);
            set(&mut c, l, -1.0);
            cases.push(c);
        }
    }
    for _ in 0..300 {
        let mut c = FluidConfig::water_2d();
        for k in 0..14 {
            let x = match k {
                0 => rng.uniform(-0.001, 0.01),
                4..=6 => rng.uniform(-0.1, 2000.0),
                13 => rng.uniform(-0.2, 1.2),
                _ => rng.uniform(-2.0, 2.0),
            };
            if rng.unit() < 0.3 {
                set(&mut c, k, x);
            }
        }
        cases.push(c);
    }
    for c in &cases {
        m.row(
            "FluidConfig::validate",
            Class::A,
            &config(c),
            &unit(c.validate()),
        );
    }

    // check_cfl: |v|·dt / h against 1, for the default and a few dt, h, over speeds at and around the limit.
    let mut speeds = edges();
    speeds.extend([
        50.0,
        -50.0,
        49.999_999_999_999_99,
        50.000_000_000_000_01,
        1000.0,
    ]);
    for _ in 0..100 {
        speeds.push(rng.signed_log_uniform(-3.0, 4.0));
    }
    for &(dt, h) in &[(0.001, 0.05), (0.0001, 0.01), (1.0, 1.0), (0.1, 0.3)] {
        let mut c = FluidConfig::water_2d();
        c.dt = dt;
        c.smoothing_radius = h;
        for &v in &speeds {
            let mut ins = config(&c);
            ins.push(f(v));
            m.row(
                "FluidConfig::check_cfl",
                Class::A,
                &ins,
                &unit(c.check_cfl(v)),
            );
        }
    }
}

fn soa_rows(m: &mut Module, rng: &mut Rng) {
    // The round trip, on particles with every field set (NaN payloads and signed zeros included).
    let sp = specials();
    for n in 0..60 {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        let mut vals = [0.0; 17];
        for (k, v) in vals.iter_mut().enumerate() {
            *v = if (n + k) % 4 == 0 {
                sp[(n + k) % sp.len()]
            } else {
                rng.uniform(-100.0, 100.0)
            };
        }
        p.position = DVec3::new(vals[0], vals[1], vals[2]);
        p.velocity = DVec3::new(vals[3], vals[4], vals[5]);
        p.acceleration = DVec3::new(vals[6], vals[7], vals[8]);
        p.density = vals[9];
        p.pressure = vals[10];
        p.mass = vals[11];
        p.phase = u8::try_from(n % 256).unwrap_or(0);
        p.conformation = [vals[12], vals[13], vals[14]];
        p.temperature = vals[15];
        p.fuel = vals[16];
        let soa = ParticleSoa::from_aos(&[p]);
        let back = soa.to_aos();
        m.row(
            "ParticleSoa::roundtrip",
            Class::A,
            &particle(&p),
            &particle(&back[0]),
        );
    }

    // consistent_len: from_aos of `n` particles, then every column truncated to its own length.
    for k in 0..120 {
        let n = 6;
        let mut lens = [n; 18];
        if k > 0 {
            for l in &mut lens {
                *l = usize::try_from(rng.int(0, 6)).unwrap_or(0);
            }
            if k % 3 == 0 {
                lens = [lens[0]; 18];
            }
        }
        let mut soa = ParticleSoa::from_aos(&vec![FluidParticle::new_2d(1.0, 2.0, 1.0); n]);
        soa.pos_x.truncate(lens[0]);
        soa.pos_y.truncate(lens[1]);
        soa.pos_z.truncate(lens[2]);
        soa.vel_x.truncate(lens[3]);
        soa.vel_y.truncate(lens[4]);
        soa.vel_z.truncate(lens[5]);
        soa.accel_x.truncate(lens[6]);
        soa.accel_y.truncate(lens[7]);
        soa.accel_z.truncate(lens[8]);
        soa.density.truncate(lens[9]);
        soa.pressure.truncate(lens[10]);
        soa.mass.truncate(lens[11]);
        soa.phase.truncate(lens[12]);
        soa.conf_xx.truncate(lens[13]);
        soa.conf_xy.truncate(lens[14]);
        soa.conf_yy.truncate(lens[15]);
        soa.temperature.truncate(lens[16]);
        soa.fuel.truncate(lens[17]);
        let ins: Vec<Tok> = lens.iter().map(|&l| usz(l)).collect();
        m.row(
            "ParticleSoa::consistent_len",
            Class::A,
            &ins,
            &[
                usz(soa.len()),
                usz(soa.consistent_len()),
                i(i64::from(soa.is_consistent())),
                usz(soa.to_aos().len()),
            ],
        );
    }
}

/// One arena scenario: the op list is generated as it runs, so a `free` or `get` can pick an issued handle.
fn arena_row(m: &mut Module, capacity: usize, ops: &[(i64, i64)], foreign: &[ArenaHandle]) {
    let mut a = ParticleArena::new(capacity);
    let mut issued: Vec<ArenaHandle> = Vec::new();
    let mut ins = vec![usz(capacity)];
    let mut outs: Vec<Tok> = Vec::new();
    for (k, &(code, arg)) in ops.iter().enumerate() {
        match code {
            0 => {
                let n = usize::try_from(arg).unwrap_or(usize::MAX);
                ins.extend([i(0), i(arg), i(0), i(0)]);
                match a.alloc(n) {
                    Some(h) => {
                        let d = h.oracle_dump();
                        outs.extend([i(1), usz(d.start), usz(d.len), gen_tok(d.generation)]);
                        let mass = f64::from(u32::try_from(k + 1).unwrap_or(0));
                        for p in a.get_mut(h) {
                            p.mass = mass;
                        }
                        issued.push(h);
                    }
                    None => outs.extend([i(0), i(0), i(0), i(0)]),
                }
            }
            1 | 3 => {
                ins.extend([i(code), i(arg), i(0), i(0)]);
                if issued.is_empty() {
                    outs.push(i(-1));
                    continue;
                }
                let h = issued[usize::try_from(arg).unwrap_or(0) % issued.len()];
                if code == 1 {
                    a.free(h);
                    outs.push(usz(a.active_count()));
                } else {
                    outs.push(usz(a.get(h).len()));
                }
            }
            2 => {
                ins.extend([i(2), i(0), i(0), i(0)]);
                outs.push(usz(a.compact()));
            }
            _ => {
                let h = foreign[usize::try_from(arg).unwrap_or(0) % foreign.len()];
                let d = h.oracle_dump();
                ins.extend([i(code), usz(d.start), usz(d.len), gen_tok(d.generation)]);
                if code == 4 {
                    a.free(h);
                    outs.push(usz(a.active_count()));
                } else {
                    outs.push(usz(a.get(h).len()));
                }
            }
        }
    }
    let d = a.oracle_dump();
    outs.extend([usz(d.watermark), gen_tok(d.generation)]);
    outs.push(Tok::V(d.free_list.len() * 2));
    for &(s, l) in &d.free_list {
        outs.extend([usz(s), usz(l)]);
    }
    outs.push(Tok::V(d.watermark));
    for p in &d.particles[..d.watermark] {
        outs.push(f(p.mass));
    }
    m.row("ParticleArena::scenario", Class::A, &ins, &outs);
}

fn arena_rows(m: &mut Module, rng: &mut Rng) {
    // Foreign handles: blocks of a larger arena, some past every scenario's range, and one from a later
    // generation.
    let mut other = ParticleArena::new(4096);
    let mut foreign: Vec<ArenaHandle> = Vec::new();
    for n in [1, 5, 40, 300] {
        if let Some(h) = other.alloc(n) {
            foreign.push(h);
        }
    }
    other.free(foreign[0]);
    other.compact();
    if let Some(h) = other.alloc(3) {
        foreign.push(h);
    }

    // The Rust tests' scenarios, as rows.
    arena_row(m, 100, &[(0, 10), (3, 0)], &foreign);
    arena_row(m, 10, &[(0, 0), (0, 11), (0, 10), (0, 1)], &foreign);
    arena_row(m, 20, &[(0, 10), (0, 5), (1, 0), (0, 8)], &foreign);
    arena_row(
        m,
        30,
        &[(0, 10), (0, 10), (0, 10), (1, 0), (1, 1), (0, 20)],
        &foreign,
    );
    arena_row(m, 30, &[(0, 10), (0, 10), (1, 0), (2, 0)], &foreign);
    arena_row(m, 20, &[(0, 5), (0, 5), (1, 1), (0, 15)], &foreign);
    arena_row(
        m,
        20,
        &[(0, 5), (0, 5), (1, 0), (1, 0), (0, 5), (0, 5)],
        &foreign,
    );
    arena_row(
        m,
        20,
        &[(0, 5), (0, 5), (1, 0), (2, 0), (3, 1), (1, 1)],
        &foreign,
    );
    arena_row(m, 10, &[(4, 2), (5, 2), (4, 0), (5, 4)], &foreign);
    arena_row(m, 10, &[(0, 3), (0, -1), (0, 7), (0, 1)], &foreign);
    arena_row(m, 0, &[(0, 0), (0, 1), (2, 0), (1, 0)], &foreign);

    // Seeded sequences: alloc-heavy, free-heavy and compact-heavy mixes, at several capacities.
    for &cap in &[1usize, 7, 16, 64, 200, 1000] {
        for mix in 0..8 {
            let mut ops: Vec<(i64, i64)> = Vec::new();
            let cap_i = i64::try_from(cap).unwrap_or(0);
            for _ in 0..(20 + 10 * mix) {
                let r = rng.unit();
                let (pa, pf, pc) = match mix % 4 {
                    0 => (0.6, 0.9, 0.95),
                    1 => (0.4, 0.85, 0.9),
                    2 => (0.45, 0.75, 0.9),
                    _ => (0.5, 0.95, 0.97),
                };
                let op = if r < pa {
                    (0, rng.int(0, (cap_i / 3).max(1)))
                } else if r < pf {
                    (1, rng.int(0, 50))
                } else if r < pc {
                    (2, 0)
                } else if r < 0.985 {
                    (3, rng.int(0, 50))
                } else {
                    (4 + rng.int(0, 1), rng.int(0, 4))
                };
                ops.push(op);
            }
            arena_row(m, cap, &ops, &foreign);
        }
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("common");
    let mut rng = Rng::new(0xc011_0f1d);
    custom_rows(&mut m, &mut rng);
    particle_rows(&mut m, &mut rng);
    config_rows(&mut m, &mut rng);
    soa_rows(&mut m, &mut rng);
    arena_rows(&mut m, &mut rng);
    m
}
