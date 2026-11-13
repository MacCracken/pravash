//! `coupling.tsv` — `rust-old/src/coupling.rs` (M6). Part a (seed 0x60_c0_0a): the shapes, the rigid bodies, the
//! two-way coupling, added mass, drag and the particle level set; class B (+, −, ×, ÷, √ and Rust's min and max in
//! order), JSON class A.
//!
//! Tokens:
//! - `<shape>` is `e:BodyShape.<Variant>` and four values, the radius and the half-extents (0 for a field the variant
//!   lacks); `<body>` is a `RigidBody`'s 15 tokens: position, velocity, mass, `<shape>`, force;
//! - `<ps>` is `i:<n>` and n particles (18 tokens each, `m_common::particle`); `<bodies>` is `i:<k>` and k bodies.
//!
//! Rows:
//! - `BodyShape::signed_distance` takes `<shape>`, a point and the body position; `::volume` and `::bounding_radius`
//!   take `<shape>`; `::projected_area` takes `<shape>` and a direction;
//! - `RigidBody::new` takes a position, the mass and `<shape>` and gives `<body>` (class A);
//!   `RigidBody::signed_distance` and `::surface_normal` take `<body>` and a point;
//! - `couple_sph_bodies` takes h, the stiffness, the damping, `<ps>` and `<bodies>`, and gives the particles and the
//!   bodies after the call; `integrate_bodies` takes gravity, dt and `<bodies>`; `integrate_bodies_with_added_mass`
//!   takes gravity, dt, ρ, C_a and `<bodies>`; `effective_mass` takes (m, ρ, V, C_a);
//! - `drag_from_particles` takes `<body>`, ρ, C_d, the sample radius and `<ps>`, and gives the force;
//!   `drag_from_particles_indexed` takes the same, then the hash's cell size and `i:<n>` the slice's length, and
//!   `<ps>`: a hisab `SpatialHash` of that cell size (as f32) holds every listed particle's f32 position with its
//!   index, and the call gets the first n. No body is so far out that the query's `center + d` overflows i32 (the
//!   oracle's overflow-checked build panics there);
//! - `particle_level_set` takes `i:` nx, ny and the level set's length, dx, the particle radius and `<ps>`, and gives
//!   the level set after the call, which starts as zeros;
//! - `BodyShape::from_json` and `RigidBody::from_json` decode a serde_json document to `<shape>` or `<body>`, or
//!   `E:Deserialize` (class A). No document needs one of bayan's stricter policies (json.tsv's `_bayan_` rows cover
//!   them).
//!
//! Part b (from its own seed, 0x60_c0_0b, after part a's rows): the `FlipSolver`. `<dump>` is its private state
//! (`oracle_dump`): `v:<nx·ny>` vx, vy, vx_old, vy_old, weight, pressure and div, then `v:<4m>` the APIC matrices.
//! - `FlipSolver::new` takes `i:` nx, ny, dx and the blend, and gives `i:` nx and ny, dx, the blend, `i:` use_apic and
//!   narrow_band_cells, and `i:` the eight vectors' lengths (class A), or `E:`;
//! - `FlipSolver::step` takes `i:` nx and ny, dx, the blend (set after `new(.., 0.95)`), `i:` use_apic, gravity, dt,
//!   `i:<steps>` and `<ps>`, and gives the particles and `<dump>` after that many steps of one fresh solver, or the
//!   first error (class S trajectories);
//! - `FlipSolver::scenario` takes `i:` nx and ny, dx, gravity, dt and `i:<k>` segments, each the blend, `i:` use_apic,
//!   `i:<keep>` (the particles truncated to that many), `<ps>` (appended) and `i:<steps>`, and gives the particles and
//!   `<dump>` after the last, or the first error (class S): the APIC matrices resized as the particle count changes.

use hisab::{DVec3, SpatialHash, Vec3};
use pravash::FluidParticle;
use pravash::coupling::{
    BodyShape, RigidBody, couple_sph_bodies, drag_from_particles, drag_from_particles_indexed,
    effective_mass, integrate_bodies, integrate_bodies_with_added_mass, particle_level_set,
};

use pravash::coupling::FlipSolver;

use crate::grids::{Rng, specials};
use crate::m_common::particle;
use crate::m_grid::{col, usz};
use crate::tsv::{Class, Module, Tok, err, f, s};

fn shape(sh: &BodyShape) -> Vec<Tok> {
    match *sh {
        BodyShape::Sphere { radius } => vec![
            Tok::E("BodyShape", "Sphere".to_string()),
            f(radius),
            f(0.0),
            f(0.0),
            f(0.0),
        ],
        BodyShape::Box { half_extents } => vec![
            Tok::E("BodyShape", "Box".to_string()),
            f(0.0),
            f(half_extents[0]),
            f(half_extents[1]),
            f(half_extents[2]),
        ],
        _ => vec![Tok::E("BodyShape", "Unknown".to_string())],
    }
}

fn v3(out: &mut Vec<Tok>, v: DVec3) {
    out.extend([f(v.x), f(v.y), f(v.z)]);
}

fn body(b: &RigidBody) -> Vec<Tok> {
    let mut out = Vec::new();
    v3(&mut out, b.position);
    v3(&mut out, b.velocity);
    out.push(f(b.mass));
    out.extend(shape(&b.shape));
    v3(&mut out, b.force);
    out
}

fn ps_toks(out: &mut Vec<Tok>, ps: &[FluidParticle]) {
    out.push(usz(ps.len()));
    for p in ps {
        out.extend(particle(p));
    }
}

fn bodies_toks(out: &mut Vec<Tok>, bs: &[RigidBody]) {
    out.push(usz(bs.len()));
    for b in bs {
        out.extend(body(b));
    }
}

fn mk_body(pos: DVec3, vel: DVec3, mass: f64, sh: BodyShape, force: DVec3) -> RigidBody {
    let mut b = RigidBody::new(pos, mass, sh);
    b.velocity = vel;
    b.force = force;
    b
}

/// The shapes every shape function is probed with: the Rust tests' and the benchmarks', degenerate and negative
/// sizes, and the specials as a radius and as each half-extent.
fn shapes() -> Vec<BodyShape> {
    let mut out = vec![
        BodyShape::Sphere { radius: 1.0 },
        BodyShape::Sphere { radius: 0.5 },
        BodyShape::Sphere { radius: 0.1 },
        BodyShape::Sphere { radius: 0.0 },
        BodyShape::Sphere { radius: -0.3 },
        BodyShape::Box {
            half_extents: [1.0, 1.0, 1.0],
        },
        BodyShape::Box {
            half_extents: [1.0, 2.0, 3.0],
        },
        BodyShape::Box {
            half_extents: [0.05, 0.05, 0.05],
        },
        BodyShape::Box {
            half_extents: [0.5, 0.2, 0.1],
        },
        BodyShape::Box {
            half_extents: [0.0, 0.0, 0.0],
        },
        BodyShape::Box {
            half_extents: [-0.1, 0.2, 0.3],
        },
    ];
    for &x in &specials() {
        out.push(BodyShape::Sphere { radius: x });
        out.push(BodyShape::Box {
            half_extents: [x, 0.5, 0.25],
        });
        out.push(BodyShape::Box {
            half_extents: [0.5, x, 0.25],
        });
        out.push(BodyShape::Box {
            half_extents: [0.5, 0.25, x],
        });
    }
    out
}

fn shape_rows(m: &mut Module, rng: &mut Rng) {
    let shs = shapes();
    // Points about the shape's centre: the centre, on and off each axis at the sizes that matter, the corners and
    // edges of a unit box, each special on each axis, and random ones.
    let mut pts: Vec<DVec3> = vec![
        DVec3::ZERO,
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(0.0, 0.0, 0.25),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(0.5, 0.2, 0.1),
        DVec3::new(-0.25, 0.1, 0.05),
        DVec3::new(0.6, 0.0, 0.0),
        DVec3::new(-3.0, 4.0, 12.0),
    ];
    for &x in &specials() {
        pts.push(DVec3::new(x, 0.1, 0.2));
        pts.push(DVec3::new(0.3, x, -0.2));
        pts.push(DVec3::new(0.3, 0.1, x));
    }
    let bases = [
        DVec3::ZERO,
        DVec3::new(0.3, 0.3, 0.0),
        DVec3::new(1e6, -2.0, 0.5),
    ];
    for (k, sh) in shs.iter().enumerate() {
        // Every shape at the origin with the fixed points; the first eleven also about other centres.
        for p in &pts {
            let mut ins = shape(sh);
            v3(&mut ins, *p);
            v3(&mut ins, DVec3::ZERO);
            m.row(
                "BodyShape::signed_distance",
                Class::B,
                &ins,
                &[f(sh.signed_distance(*p, DVec3::ZERO))],
            );
        }
        if k < 11 {
            for b in &bases[1..] {
                for _ in 0..6 {
                    let p = *b
                        + DVec3::new(
                            rng.uniform(-1.5, 1.5),
                            rng.uniform(-1.5, 1.5),
                            rng.uniform(-1.5, 1.5),
                        );
                    let mut ins = shape(sh);
                    v3(&mut ins, p);
                    v3(&mut ins, *b);
                    m.row(
                        "BodyShape::signed_distance",
                        Class::B,
                        &ins,
                        &[f(sh.signed_distance(p, *b))],
                    );
                }
            }
        }
        m.row("BodyShape::volume", Class::B, &shape(sh), &[f(sh.volume())]);
        m.row(
            "BodyShape::bounding_radius",
            Class::B,
            &shape(sh),
            &[f(sh.bounding_radius())],
        );
    }
    // Volumes and bounding radii over sizes k/7 and (k/3, k/7, k/11), whose products round differently in another
    // order (no random numbers drawn: the rows after these keep theirs).
    for k in 1..=40 {
        let kf = f64::from(k);
        let sh = BodyShape::Sphere { radius: kf / 7.0 };
        let bx = BodyShape::Box {
            half_extents: [kf / 3.0, kf / 7.0, kf / 11.0],
        };
        for s in [sh, bx] {
            m.row("BodyShape::volume", Class::B, &shape(&s), &[f(s.volume())]);
            m.row(
                "BodyShape::bounding_radius",
                Class::B,
                &shape(&s),
                &[f(s.bounding_radius())],
            );
        }
    }
    // Directions: the axes and their negatives, diagonals, random unit vectors, zero, and the specials.
    let mut dirs: Vec<DVec3> = vec![
        DVec3::X,
        DVec3::Y,
        DVec3::Z,
        -DVec3::X,
        DVec3::new(1.0, 1.0, 0.0) / 2.0_f64.sqrt(),
        DVec3::new(0.6, -0.8, 0.0),
        DVec3::new(0.48, 0.6, 0.64),
        DVec3::ZERO,
    ];
    for _ in 0..4 {
        let d = DVec3::new(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        );
        dirs.push(d / d.length());
    }
    for &x in &specials() {
        dirs.push(DVec3::new(x, 0.6, 0.0));
    }
    for (k, sh) in shs.iter().enumerate() {
        let take = if k < 11 { dirs.len() } else { 3 };
        for d in &dirs[..take] {
            let mut ins = shape(sh);
            v3(&mut ins, *d);
            m.row(
                "BodyShape::projected_area",
                Class::B,
                &ins,
                &[f(sh.projected_area(*d))],
            );
        }
    }
}

fn body_rows(m: &mut Module, rng: &mut Rng) {
    let shs = shapes();
    for (k, sh) in shs.iter().take(11).enumerate() {
        let pos = DVec3::new(0.1 * k as f64, -0.2, 0.3);
        let mass = [1.0, 10.0, 0.0, -1.0, f64::NAN][k % 5];
        let b = RigidBody::new(pos, mass, *sh);
        let mut ins = Vec::new();
        v3(&mut ins, pos);
        ins.push(f(mass));
        ins.extend(shape(sh));
        m.row("RigidBody::new", Class::A, &ins, &body(&b));
    }
    // Bodies about three centres; points: the Rust test's, the centre (the direction fallback), a box's centre
    // (zero central differences), its faces, edges and corners, a sphere's surface, random ones near the surface,
    // and the specials.
    let centres = [
        DVec3::ZERO,
        DVec3::new(0.3, 0.3, 0.0),
        DVec3::new(-2.0, 5.0, 1e3),
    ];
    for (k, sh) in shs.iter().enumerate() {
        let c = centres[k % 3];
        let b = mk_body(
            c,
            DVec3::new(0.5, 0.0, -0.25),
            2.0,
            *sh,
            DVec3::new(1.0, 2.0, 3.0),
        );
        let mut pts: Vec<DVec3> = vec![
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::ZERO,
            DVec3::new(1e-7, 0.0, 0.0),
            DVec3::new(1e-30, 0.0, 0.0),
            DVec3::new(0.0, 0.5, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.5, 0.2, 0.1),
            DVec3::new(0.05, 0.05, 0.05),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        if k < 11 {
            for _ in 0..8 {
                pts.push(DVec3::new(
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(-1.2, 1.2),
                ));
            }
        }
        if k < 2 {
            for &x in &specials() {
                pts.push(DVec3::new(x, 0.25, 0.0));
            }
        }
        for p in &pts {
            let q = c + *p;
            let mut ins = body(&b);
            v3(&mut ins, q);
            m.row(
                "RigidBody::signed_distance",
                Class::B,
                &ins,
                &[f(b.signed_distance(q))],
            );
            let n = b.surface_normal(q);
            m.row(
                "RigidBody::surface_normal",
                Class::B,
                &ins,
                &[f(n.x), f(n.y), f(n.z)],
            );
        }
    }
}

fn couple_row(m: &mut Module, h: f64, k: f64, c: f64, ps: &[FluidParticle], bs: &[RigidBody]) {
    let mut ins = vec![f(h), f(k), f(c)];
    ps_toks(&mut ins, ps);
    bodies_toks(&mut ins, bs);
    let mut p2 = ps.to_vec();
    let mut b2 = bs.to_vec();
    couple_sph_bodies(&mut p2, &mut b2, h, k, c);
    let mut out = Vec::new();
    ps_toks(&mut out, &p2);
    bodies_toks(&mut out, &b2);
    m.row("couple_sph_bodies", Class::B, &ins, &out);
}

fn integrate_rows(m: &mut Module, g: DVec3, dt: f64, rho: f64, ca: f64, bs: &[RigidBody]) {
    let mut ins = Vec::new();
    v3(&mut ins, g);
    ins.push(f(dt));
    bodies_toks(&mut ins, bs);
    let mut b1 = bs.to_vec();
    integrate_bodies(&mut b1, g, dt);
    let mut out = Vec::new();
    bodies_toks(&mut out, &b1);
    m.row("integrate_bodies", Class::B, &ins, &out);

    let mut ins2 = Vec::new();
    v3(&mut ins2, g);
    ins2.extend([f(dt), f(rho), f(ca)]);
    bodies_toks(&mut ins2, bs);
    let mut b2 = bs.to_vec();
    integrate_bodies_with_added_mass(&mut b2, g, dt, rho, ca);
    let mut out2 = Vec::new();
    bodies_toks(&mut out2, &b2);
    m.row("integrate_bodies_with_added_mass", Class::B, &ins2, &out2);
}

fn drag_row(m: &mut Module, b: &RigidBody, rho: f64, cd: f64, sr: f64, ps: &[FluidParticle]) {
    let mut ins = body(b);
    ins.extend([f(rho), f(cd), f(sr)]);
    ps_toks(&mut ins, ps);
    let d = drag_from_particles(b, ps, rho, cd, sr);
    m.row(
        "drag_from_particles",
        Class::B,
        &ins,
        &[f(d.x), f(d.y), f(d.z)],
    );
}

/// An indexed drag row: the hash holds every particle of `ps` at its f32 position (as Rust's test builds one), and
/// the call gets the first `n`.
#[allow(clippy::too_many_arguments)]
fn drag_indexed_row(
    m: &mut Module,
    b: &RigidBody,
    rho: f64,
    cd: f64,
    sr: f64,
    cell: f64,
    n: usize,
    ps: &[FluidParticle],
) {
    #[allow(clippy::cast_possible_truncation)]
    let mut hash = SpatialHash::new(cell as f32).expect("a positive cell size");
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
    let mut ins = body(b);
    ins.extend([f(rho), f(cd), f(sr), f(cell), usz(n)]);
    ps_toks(&mut ins, ps);
    let d = drag_from_particles_indexed(b, &ps[..n], &hash, rho, cd, sr);
    m.row(
        "drag_from_particles_indexed",
        Class::B,
        &ins,
        &[f(d.x), f(d.y), f(d.z)],
    );
}

/// A seeded cloud of n particles in the box [lo, hi], each with a velocity in [−v, v] per axis (z kept for 3D).
fn cloud(rng: &mut Rng, n: usize, lo: DVec3, hi: DVec3, v: f64, flat: bool) -> Vec<FluidParticle> {
    (0..n)
        .map(|_| {
            let z = if flat { 0.0 } else { rng.uniform(lo.z, hi.z) };
            let mut p = FluidParticle::new(
                DVec3::new(rng.uniform(lo.x, hi.x), rng.uniform(lo.y, hi.y), z),
                rng.log_uniform(-4.0, -1.0),
            );
            p.velocity = DVec3::new(
                rng.uniform(-v, v),
                rng.uniform(-v, v),
                if flat { 0.0 } else { rng.uniform(-v, v) },
            );
            p.acceleration = DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0);
            p.density = 1000.0;
            p
        })
        .collect()
}

/// A block of particles streaming at +x (u, with a small seeded jitter) through [x0, x1] × [y0, y1].
fn stream(
    rng: &mut Rng,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    sp: f64,
    u: f64,
) -> Vec<FluidParticle> {
    let mut out = Vec::new();
    let mut y = y0;
    while y <= y1 {
        let mut x = x0;
        while x <= x1 {
            let mut p = FluidParticle::new_2d(x, y, 0.001);
            p.velocity = DVec3::new(u + rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 0.0);
            p.density = 1000.0;
            out.push(p);
            x += sp;
        }
        y += sp;
    }
    out
}

fn coupling_rows(m: &mut Module, rng: &mut Rng) {
    let o = DVec3::ZERO;
    let sph = |r: f64| BodyShape::Sphere { radius: r };
    let bx = |a: f64, b: f64, c: f64| BodyShape::Box {
        half_extents: [a, b, c],
    };
    // The Rust tests' scenes.
    let mut p0 = FluidParticle::new(o, 1.0);
    p0.density = 1000.0;
    let b0 = RigidBody::new(o, 10.0, sph(0.5));
    couple_row(m, 0.1, 1000.0, 10.0, &[p0], std::slice::from_ref(&b0));
    let mut p1 = FluidParticle::new(DVec3::new(0.6, 0.0, 0.0), 1.0);
    p1.density = 1000.0;
    couple_row(m, 0.2, 1000.0, 10.0, &[p1], std::slice::from_ref(&b0));
    // The benchmark's block about a sphere and a box, and the degenerate shapes.
    let block = pravash::sph::create_particle_block([0.1, 0.1], [0.5, 0.5], 0.05, 0.001);
    let two = vec![
        RigidBody::new(DVec3::new(0.3, 0.3, 0.0), 1.0, sph(0.1)),
        RigidBody::new(DVec3::new(0.5, 0.3, 0.0), 1.0, bx(0.05, 0.05, 0.05)),
    ];
    couple_row(m, 0.05, 1000.0, 10.0, &block, &two);
    couple_row(m, 0.05, 1000.0, 10.0, &block, &[]);
    couple_row(m, 0.05, 1000.0, 10.0, &[], &two);
    // Approaching and receding particles, massless, NaN and ∞ masses, a moving body, negative damping, the specials
    // as h, k and c.
    let mut ps: Vec<FluidParticle> = Vec::new();
    for (k, &(x, vx, mass)) in [
        (0.55, -1.0, 0.01),
        (0.55, 1.0, 0.01),
        (0.45, -1.0, 0.0),
        (0.52, 0.0, f64::NAN),
        (0.58, -0.5, f64::INFINITY),
        (0.62, -2.0, 1e-30),
        (0.5, 0.0, 0.01),
    ]
    .iter()
    .enumerate()
    {
        let mut p = FluidParticle::new(DVec3::new(x, 0.02 * k as f64, 0.0), mass);
        p.velocity = DVec3::new(vx, 0.1, 0.0);
        ps.push(p);
    }
    let mut moving = RigidBody::new(o, 2.0, sph(0.5));
    moving.velocity = DVec3::new(0.5, -0.5, 0.0);
    moving.force = DVec3::new(9.0, 9.0, 9.0);
    let boxb = mk_body(
        DVec3::new(0.1, 0.0, 0.0),
        DVec3::new(-0.2, 0.0, 0.0),
        3.0,
        bx(0.4, 0.3, 0.2),
        o,
    );
    couple_row(m, 0.1, 1000.0, 10.0, &ps, &[moving.clone(), boxb.clone()]);
    couple_row(m, 0.1, 500.0, -10.0, &ps, std::slice::from_ref(&boxb));
    for &x in &specials() {
        couple_row(m, x, 1000.0, 10.0, &ps, &[moving.clone()]);
        couple_row(m, 0.1, x, 10.0, &ps, std::slice::from_ref(&boxb));
        couple_row(m, 0.1, 1000.0, x, &ps, &[moving.clone()]);
    }
    // Seeded clouds about one to four bodies of every kind, 2D and 3D.
    for k in 0..24 {
        let flat = k % 2 == 0;
        let n = 4 + (k * 7) % 40;
        let cl = cloud(
            rng,
            n,
            DVec3::new(-0.6, -0.6, -0.3),
            DVec3::new(0.6, 0.6, 0.3),
            2.0,
            flat,
        );
        let nb = 1 + k % 4;
        let bs: Vec<RigidBody> = (0..nb)
            .map(|j| {
                let sh = if (j + k) % 2 == 0 {
                    sph(rng.uniform(0.05, 0.4))
                } else {
                    bx(
                        rng.uniform(0.05, 0.3),
                        rng.uniform(0.05, 0.3),
                        rng.uniform(0.05, 0.3),
                    )
                };
                mk_body(
                    DVec3::new(
                        rng.uniform(-0.3, 0.3),
                        rng.uniform(-0.3, 0.3),
                        if flat { 0.0 } else { rng.uniform(-0.1, 0.1) },
                    ),
                    DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0),
                    rng.log_uniform(-1.0, 1.0),
                    sh,
                    DVec3::new(rng.uniform(-5.0, 5.0), 0.0, 0.0),
                )
            })
            .collect();
        let h = [0.05, 0.1, 0.2][k % 3];
        couple_row(
            m,
            h,
            rng.log_uniform(2.0, 4.0),
            rng.uniform(0.0, 20.0),
            &cl,
            &bs,
        );
        integrate_rows(
            m,
            DVec3::new(0.0, -9.81, 0.0),
            0.001 * (1 + k % 5) as f64,
            1000.0,
            0.5,
            &bs,
        );
    }
    // Integration: the Rust tests' bodies, massless, NaN, ∞ and negative masses, zero and huge forces, the specials
    // as dt, ρ and C_a, and each coefficient of `AddedMassCoefficient`.
    let g = DVec3::new(0.0, -9.81, 0.0);
    let falls = vec![RigidBody::new(DVec3::new(0.0, 1.0, 0.0), 1.0, sph(0.1))];
    integrate_rows(m, g, 0.01, 1000.0, 0.5, &falls);
    let mut pushed = falls.clone();
    pushed[0].force = DVec3::new(10.0, 0.0, 0.0);
    integrate_rows(m, o, 0.01, 1000.0, 0.5, &pushed);
    let neutral = vec![RigidBody::new(o, 1000.0 * sph(0.1).volume(), sph(0.1))];
    integrate_rows(m, g, 0.01, 1000.0, 0.5, &neutral);
    let odd: Vec<RigidBody> = [0.0, f64::NAN, f64::INFINITY, -2.0, 1e-30, 1e300]
        .iter()
        .enumerate()
        .map(|(k, &mass)| {
            mk_body(
                DVec3::new(k as f64, 0.0, 0.0),
                DVec3::new(1.0, 0.0, -1.0),
                mass,
                if k % 2 == 0 {
                    sph(0.2)
                } else {
                    bx(0.1, 0.2, 0.3)
                },
                DVec3::new(3.0, -1.0, 1e10),
            )
        })
        .collect();
    integrate_rows(m, g, 0.01, 1000.0, 0.5, &odd);
    for &x in &specials() {
        integrate_rows(m, g, x, 1000.0, 0.5, &falls);
        integrate_rows(m, g, 0.01, x, 0.5, &odd[..2]);
        integrate_rows(m, g, 0.01, 1000.0, x, &pushed);
        integrate_rows(m, DVec3::new(x, -9.81, 0.0), 0.01, 1000.0, 0.5, &falls);
    }
    for ca in [0.5, 0.67, 1.0] {
        integrate_rows(m, g, 0.005, 998.2, ca, &odd[3..]);
    }
    // effective_mass over physical ranges and the specials in each argument.
    let mut em: Vec<[f64; 4]> = vec![[1.0, 1000.0, 0.004_188_790_204_786_391, 0.5]];
    for a in 0..4 {
        for &x in &specials() {
            let mut c = [2.0, 1000.0, 0.001, 0.5];
            c[a] = x;
            em.push(c);
        }
    }
    for _ in 0..60 {
        em.push([
            rng.log_uniform(-3.0, 3.0),
            rng.uniform(1.0, 2000.0),
            rng.log_uniform(-6.0, 0.0),
            rng.uniform(0.0, 1.5),
        ]);
    }
    for c in &em {
        m.row(
            "effective_mass",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &[f(effective_mass(c[0], c[1], c[2], c[3]))],
        );
    }
}

fn drag_rows(m: &mut Module, rng: &mut Rng) {
    let o = DVec3::ZERO;
    let sph = |r: f64| BodyShape::Sphere { radius: r };
    let bx = |a: f64, b: f64, c: f64| BodyShape::Box {
        half_extents: [a, b, c],
    };
    // The Rust tests' scenes.
    let still = RigidBody::new(o, 1.0, sph(0.1));
    drag_row(m, &still, 1000.0, 0.47, 0.2, &[]);
    let mut mover = RigidBody::new(DVec3::new(0.5, 0.5, 0.0), 1.0, sph(0.1));
    mover.velocity = DVec3::new(1.0, 0.0, 0.0);
    let line: Vec<FluidParticle> = (0..5)
        .map(|i| FluidParticle::new(DVec3::new(0.45 + i as f64 * 0.025, 0.5, 0.0), 0.01))
        .collect();
    drag_row(m, &mover, 1000.0, 0.47, 0.2, &line);
    for v in [2.0, 0.5] {
        let mut q = FluidParticle::new(DVec3::new(0.12, 0.0, 0.0), 0.001);
        q.velocity = DVec3::new(v, 0.0, 0.0);
        drag_row(m, &still, 1000.0, 0.47, 0.05, &[q]);
    }
    let big = RigidBody::new(o, 1.0, sph(1.0));
    let mut ring = Vec::new();
    for k in 0..16 {
        let a = k as f64 * std::f64::consts::TAU / 16.0;
        // The angle's cosine and sine through the oracle's own std (glibc): the inputs are recorded, not recomputed.
        let mut p = FluidParticle::new(DVec3::new(1.05 * a.cos(), 1.05 * a.sin(), 0.0), 0.001);
        p.velocity = DVec3::new(1.0, 0.0, 0.0);
        ring.push(p);
    }
    drag_row(m, &big, 1000.0, 0.47, 0.2, &ring);
    drag_indexed_row(m, &big, 1000.0, 0.47, 0.2, 0.25, 16, &ring);
    drag_indexed_row(m, &big, 1000.0, 0.47, 0.2, 0.25, 4, &ring);
    drag_indexed_row(m, &big, 1000.0, 0.47, 0.2, 0.25, 0, &ring);
    // Seeded clouds about spheres and boxes, 2D and 3D, at cell sizes on both sides of the sample radius; each body
    // moving or still, some particles moving with it (a relative speed under 1e-20).
    for k in 0..40 {
        let flat = k % 2 == 0;
        let n = 1 + (k * 11) % 120;
        let mut cl = cloud(
            rng,
            n,
            DVec3::new(-0.5, -0.5, -0.25),
            DVec3::new(0.5, 0.5, 0.25),
            1.5,
            flat,
        );
        let sh = if k % 3 == 0 {
            bx(
                rng.uniform(0.05, 0.3),
                rng.uniform(0.05, 0.3),
                rng.uniform(0.05, 0.3),
            )
        } else {
            sph(rng.uniform(0.05, 0.35))
        };
        let mut b = RigidBody::new(
            DVec3::new(rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0),
            1.0,
            sh,
        );
        b.velocity = if k % 5 == 4 {
            o
        } else {
            DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0)
        };
        if k % 7 == 6 {
            for p in &mut cl {
                p.velocity = b.velocity;
            }
        }
        let sr = rng.uniform(0.02, 0.2);
        let rho = rng.uniform(1.0, 1500.0);
        let cd = rng.uniform(0.1, 1.5);
        drag_row(m, &b, rho, cd, sr, &cl);
        let cell = [0.025, 0.05, 0.1, 0.25, 1.0][k % 5];
        drag_indexed_row(m, &b, rho, cd, sr, cell, n, &cl);
        if n > 3 {
            drag_indexed_row(m, &b, rho, cd, sr, cell, n / 2, &cl);
        }
    }
    // The specials as ρ, C_d and the sample radius, a NaN particle, a NaN body position (the query's cell 0), and a
    // body about 10⁶ out, where the f32 hash cells are 0.0625 wide and keys move.
    let mut cl = cloud(
        rng,
        30,
        DVec3::new(-0.3, -0.3, 0.0),
        DVec3::new(0.3, 0.3, 0.0),
        1.0,
        true,
    );
    let mut b = RigidBody::new(o, 1.0, sph(0.15));
    b.velocity = DVec3::new(0.4, 0.1, 0.0);
    for &x in &specials() {
        drag_row(m, &b, x, 0.47, 0.1, &cl);
        drag_row(m, &b, 1000.0, x, 0.1, &cl);
        drag_row(m, &b, 1000.0, 0.47, x, &cl);
        #[allow(clippy::cast_possible_truncation)]
        if (x as f32).is_finite() {
            // A radius that is ±∞ as f32 makes the query walk ±i32::MAX cells, or negate i32::MIN (the oracle never
            // returns, or panics).
            drag_indexed_row(m, &b, 1000.0, 0.47, x, 0.05, 30, &cl);
        }
        drag_indexed_row(m, &b, x, 0.47, 0.1, 0.05, 30, &cl);
    }
    cl[3].position.x = f64::NAN;
    cl[7].velocity.y = f64::INFINITY;
    drag_row(m, &b, 1000.0, 0.47, 0.1, &cl);
    drag_indexed_row(m, &b, 1000.0, 0.47, 0.1, 0.05, 30, &cl);
    let mut nanb = b.clone();
    nanb.position = DVec3::new(f64::NAN, 0.0, 0.0);
    drag_row(m, &nanb, 1000.0, 0.47, 0.1, &cl);
    drag_indexed_row(m, &nanb, 1000.0, 0.47, 0.1, 0.05, 30, &cl);
    for (k, off) in [1e4, 1e6, 3e6].iter().enumerate() {
        let far = cloud(
            rng,
            60,
            DVec3::new(off - 0.3, -0.3, 0.0),
            DVec3::new(off + 0.3, 0.3, 0.0),
            1.0,
            true,
        );
        let mut fb = RigidBody::new(DVec3::new(*off, 0.0, 0.0), 1.0, sph(0.2));
        fb.velocity = DVec3::new(-0.5, 0.25, 0.0);
        drag_row(m, &fb, 1000.0, 0.47, 0.1, &far);
        drag_indexed_row(m, &fb, 1000.0, 0.47, 0.1, [0.05, 0.1, 0.25][k], 60, &far);
    }
    // A 9 × 9 × 9 lattice taller and wider than the query's block, so the walk meets cells past the block in y and z
    // (no random numbers drawn).
    let mut lat = Vec::new();
    for k in 0..9 {
        for j in 0..9 {
            for i in 0..9 {
                let c = |a: i32| f64::from(a) * 0.075 - 0.3;
                let mut p = FluidParticle::new(DVec3::new(c(i), c(j), c(k)), 0.001);
                p.velocity = DVec3::new(
                    f64::from(i) * 0.01,
                    -f64::from(j) * 0.01,
                    f64::from(k) * 0.02,
                );
                lat.push(p);
            }
        }
    }
    let mut lb = RigidBody::new(DVec3::new(0.01, 0.02, 0.03), 1.0, sph(0.05));
    lb.velocity = DVec3::new(0.2, 0.0, 0.0);
    drag_row(m, &lb, 1000.0, 0.47, 0.08, &lat);
    for cell in [0.05, 0.03] {
        drag_indexed_row(m, &lb, 1000.0, 0.47, 0.08, cell, lat.len(), &lat);
    }
    let lbox = mk_body(
        DVec3::new(-0.05, 0.1, -0.02),
        DVec3::new(0.0, 0.3, 0.1),
        1.0,
        bx(0.06, 0.03, 0.04),
        o,
    );
    drag_row(m, &lbox, 1000.0, 0.47, 0.05, &lat);
    drag_indexed_row(m, &lbox, 1000.0, 0.47, 0.05, 0.04, lat.len(), &lat);
}

fn level_set_row(
    m: &mut Module,
    nx: usize,
    ny: usize,
    len: usize,
    dx: f64,
    r: f64,
    ps: &[FluidParticle],
) {
    let mut ins = vec![usz(nx), usz(ny), usz(len), f(dx), f(r)];
    ps_toks(&mut ins, ps);
    let mut ls = vec![0.0; len];
    particle_level_set(&mut ls, nx, ny, dx, ps, r);
    let mut out = vec![Tok::V(len)];
    out.extend(ls.iter().map(|&x| f(x)));
    m.row("particle_level_set", Class::B, &ins, &out);
}

fn level_set_rows(m: &mut Module, rng: &mut Rng) {
    // The Rust tests' grids, the size refusals (a mismatch, an overflowing product), empty grids and particle sets,
    // the specials as dx and as the radius, NaN and ∞ particles, and seeded clouds.
    level_set_row(m, 4, 4, 16, 0.1, 0.05, &[]);
    let one = [FluidParticle::new(DVec3::new(0.2, 0.2, 0.0), 0.01)];
    level_set_row(m, 5, 5, 25, 0.1, 0.05, &one);
    let mid = [FluidParticle::new(DVec3::new(0.5, 0.5, 0.0), 0.01)];
    level_set_row(m, 10, 10, 100, 0.1, 0.1, &mid);
    level_set_row(m, 8, 8, 10, 0.1, 0.05, &[]);
    level_set_row(m, 4, 4, 20, 0.1, 0.05, &one);
    level_set_row(m, usize::MAX, 2, 6, 0.1, 0.05, &one);
    level_set_row(m, 1 << 63, 2, 0, 0.1, 0.05, &one);
    level_set_row(m, 0, 5, 0, 0.1, 0.05, &one);
    level_set_row(m, 3, 0, 0, 0.1, 0.05, &one);
    level_set_row(m, 1, 1, 1, 0.1, 0.05, &one);
    let mut odd = one.to_vec();
    odd.push(FluidParticle::new(DVec3::new(f64::NAN, 0.1, 0.0), 0.01));
    odd.push(FluidParticle::new(
        DVec3::new(f64::INFINITY, 0.1, 0.0),
        0.01,
    ));
    odd.push(FluidParticle::new(DVec3::new(0.25, 0.25, 7.0), 0.01));
    for &x in &specials() {
        level_set_row(m, 4, 3, 12, x, 0.05, &odd);
        level_set_row(m, 3, 4, 12, 0.1, x, &odd);
    }
    for k in 0..16 {
        let n = (k * 5) % 30;
        let cl = cloud(
            rng,
            n,
            DVec3::new(-0.2, -0.2, 0.0),
            DVec3::new(1.4, 1.2, 0.0),
            0.0,
            true,
        );
        let nx = 3 + k % 9;
        let ny = 2 + (k * 3) % 11;
        level_set_row(
            m,
            nx,
            ny,
            nx * ny,
            [0.05, 0.1, 0.2][k % 3],
            rng.uniform(0.0, 0.1),
            &cl,
        );
    }
}

/// The flow scene: a stream at +x past a sphere and a box, advanced by the coupling's accelerations (kinematics in
/// the generator; the rows are the calls on its frames).
fn flow_rows(m: &mut Module, rng: &mut Rng) {
    let mut ps = stream(rng, 0.0, 0.5, 0.15, 0.45, 0.025, 1.0);
    let mut bs = vec![
        RigidBody::new(
            DVec3::new(0.45, 0.3, 0.0),
            2.0,
            BodyShape::Sphere { radius: 0.06 },
        ),
        RigidBody::new(
            DVec3::new(0.6, 0.25, 0.0),
            4.0,
            BodyShape::Box {
                half_extents: [0.04, 0.08, 0.05],
            },
        ),
    ];
    let dt = 0.002;
    let g = DVec3::ZERO;
    for frame in 0..=40 {
        if [0, 1, 5, 20, 40].contains(&frame) {
            couple_row(m, 0.03, 2000.0, 20.0, &ps, &bs);
            integrate_rows(m, g, dt, 1000.0, 0.5, &bs);
            for b in &bs {
                drag_row(m, b, 1000.0, 0.47, 0.05, &ps);
                drag_indexed_row(m, b, 1000.0, 0.47, 0.05, 0.03, ps.len(), &ps);
            }
        }
        for p in &mut ps {
            p.acceleration = DVec3::ZERO;
        }
        couple_sph_bodies(&mut ps, &mut bs, 0.03, 2000.0, 20.0);
        for b in &mut bs {
            let d = drag_from_particles(b, &ps, 1000.0, 0.47, 0.05);
            b.force += d;
        }
        integrate_bodies_with_added_mass(&mut bs, g, dt, 1000.0, 0.5);
        for p in &mut ps {
            p.velocity += p.acceleration * dt;
            p.position += p.velocity * dt;
        }
    }
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let sh_tok = |doc: &str| match serde_json::from_str::<BodyShape>(doc) {
        Ok(v) => shape(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for sh in shapes() {
        docs.push(serde_json::to_string(&sh).unwrap_or_default());
    }
    for _ in 0..10 {
        docs.push(
            serde_json::to_string(&BodyShape::Sphere {
                radius: rng.signed_log_uniform(-5.0, 5.0),
            })
            .unwrap_or_default(),
        );
        docs.push(
            serde_json::to_string(&BodyShape::Box {
                half_extents: [
                    rng.uniform(0.0, 3.0),
                    rng.log_uniform(-3.0, 1.0),
                    -rng.unit(),
                ],
            })
            .unwrap_or_default(),
        );
    }
    for doc in [
        r#"{"Sphere":{"radius":1}}"#,
        r#"{"Sphere":[0.5]}"#,
        r#"{"Box":{"half_extents":[1,2,3]}}"#,
        r#"{"Box":[[1,2,3]]}"#,
        r#"{"Sphere":{"radius":1,"extra":[1,2]}}"#,
        r#"{"Sphere":[0.5,1]}"#,
        r#"{"Sphere":[]}"#,
        r#""Sphere""#,
        r#""Box""#,
        r#"{"Sphere":null}"#,
        r#"{"Sphere":{}}"#,
        r#"{"Sphere":{"radius":null}}"#,
        r#"{"Sphere":{"radius":"1"}}"#,
        r#"{"Sphere":{"radius":1e400}}"#,
        r#"{"Sphere":{"radius":1,"radius":2}}"#,
        r#"{"Sphere":{"half_extents":[1,2,3]}}"#,
        r#"{"Box":{"half_extents":[1,2]}}"#,
        r#"{"Box":{"half_extents":[1,2,3,4]}}"#,
        r#"{"Box":{"half_extents":{"x":1,"y":2,"z":3}}}"#,
        r#"{"Box":{"radius":1}}"#,
        r#"{"Box":{"half_extents":[1,2,3]},"Sphere":{"radius":1}}"#,
        r#"{"Cube":{"radius":1}}"#,
        r#"{"sphere":{"radius":1}}"#,
        r#"{}"#,
        r#"[]"#,
        r#"["Sphere",{"radius":1}]"#,
        "null",
        "1",
        r#"{"Sphere":{"radius":-0.0}}"#,
        r#"{"Sphere":{"radius":1.7976931348623157e308}}"#,
        r#"{"Sphere":{"radius":4.9e-324}}"#,
        r#" { "Sphere" : { "radius" : 2.5 } } "#,
    ] {
        docs.push(doc.to_string());
    }
    for doc in &docs {
        m.row(
            "BodyShape::from_json",
            Class::A,
            &[s(doc.as_str())],
            &sh_tok(doc),
        );
    }

    let body_tok = |doc: &str| match serde_json::from_str::<RigidBody>(doc) {
        Ok(v) => body(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut bdocs: Vec<String> = Vec::new();
    let shs = shapes();
    for k in 0..14 {
        let mut b = mk_body(
            DVec3::new(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ),
            DVec3::new(
                rng.uniform(-1.0, 1.0),
                0.0,
                rng.signed_log_uniform(-3.0, 3.0),
            ),
            rng.log_uniform(-2.0, 2.0),
            shs[k % 11],
            DVec3::new(0.0, -9.81, rng.uniform(-1.0, 1.0)),
        );
        if k == 12 {
            b.mass = f64::NAN;
        }
        if k == 13 {
            b.force.y = f64::NEG_INFINITY;
        }
        bdocs.push(serde_json::to_string(&b).unwrap_or_default());
    }
    let base = serde_json::to_string(&RigidBody::new(
        DVec3::new(1.0, 2.0, 3.0),
        5.0,
        BodyShape::Sphere { radius: 0.5 },
    ))
    .unwrap_or_default();
    bdocs.push(base.clone());
    for (from, to) in [
        (r#""mass":5.0"#, r#""mass":5"#),
        (r#""mass":5.0"#, r#""mass":"5""#),
        (r#""mass":5.0,"#, ""),
        (r#""position":[1.0,2.0,3.0]"#, r#""position":[1.0,2.0]"#),
        (
            r#""position":[1.0,2.0,3.0]"#,
            r#""position":{"x":1,"y":2,"z":3}"#,
        ),
        (
            r#""position":[1.0,2.0,3.0]"#,
            r#""position":[1,2,3],"spin":0"#,
        ),
        (r#"{"Sphere":{"radius":0.5}}"#, r#""Sphere""#),
        (
            r#"{"Sphere":{"radius":0.5}}"#,
            r#"{"Box":{"half_extents":[1,1,1]}}"#,
        ),
        (r#"{"Sphere":{"radius":0.5}}"#, r#"{"Sphere":[0.5]}"#),
        (r#"{"Sphere":{"radius":0.5}}"#, r#"{"Disc":{"radius":0.5}}"#),
        (r#""force":[0.0,0.0,0.0]"#, r#""force":null"#),
    ] {
        bdocs.push(base.replacen(from, to, 1));
    }
    bdocs.push(
        r#"[[0,0,0],[1,0,0],2.5,{"Box":{"half_extents":[0.1,0.2,0.3]}},[0,-1,0]]"#.to_string(),
    );
    bdocs.push(r#"[[0,0,0],[1,0,0],2.5,{"Box":{"half_extents":[0.1,0.2,0.3]}}]"#.to_string());
    bdocs.push("null".to_string());
    bdocs.push("{}".to_string());
    for doc in &bdocs {
        m.row(
            "RigidBody::from_json",
            Class::A,
            &[s(doc.as_str())],
            &body_tok(doc),
        );
    }
}

// ── part b: the FLIP solver ─────────────────────────────────────────────────────────

fn dump(out: &mut Vec<Tok>, s: &FlipSolver) {
    let d = s.oracle_dump();
    for v in [
        &d.vx,
        &d.vy,
        &d.vx_old,
        &d.vy_old,
        &d.weight,
        &d.pressure,
        &d.div,
    ] {
        col(out, v);
    }
    let flat: Vec<f64> = d.apic_c.iter().flat_map(|c| c.iter().copied()).collect();
    col(out, &flat);
}

fn flip_new_row(m: &mut Module, nx: usize, ny: usize, dx: f64, ratio: f64) {
    let ins = [usz(nx), usz(ny), f(dx), f(ratio)];
    let out = match FlipSolver::new(nx, ny, dx, ratio) {
        Ok(s) => {
            let d = s.oracle_dump();
            let mut o = vec![
                usz(s.grid_nx()),
                usz(s.grid_ny()),
                f(s.grid_dx()),
                f(s.flip_ratio),
                Tok::I(i64::from(s.use_apic)),
                usz(s.narrow_band_cells),
            ];
            for len in [
                d.vx.len(),
                d.vy.len(),
                d.vx_old.len(),
                d.vy_old.len(),
                d.weight.len(),
                d.pressure.len(),
                d.div.len(),
                d.apic_c.len(),
            ] {
                o.push(usz(len));
            }
            o
        }
        Err(x) => vec![err(&x)],
    };
    m.row("FlipSolver::new", Class::A, &ins, &out);
}

/// One FLIP step row: a fresh `new(nx, ny, dx, 0.95)` with the blend and APIC set, `steps` steps.
#[allow(clippy::too_many_arguments)]
fn flip_step_row(
    m: &mut Module,
    nx: usize,
    ny: usize,
    dx: f64,
    ratio: f64,
    apic: bool,
    g: DVec3,
    dt: f64,
    steps: usize,
    ps: &[FluidParticle],
) {
    let mut ins = vec![usz(nx), usz(ny), f(dx), f(ratio), Tok::I(i64::from(apic))];
    v3(&mut ins, g);
    ins.extend([f(dt), usz(steps)]);
    ps_toks(&mut ins, ps);
    let mut s = FlipSolver::new(nx, ny, dx, 0.95).expect("a valid grid");
    s.flip_ratio = ratio;
    s.use_apic = apic;
    let mut p = ps.to_vec();
    let mut out = Vec::new();
    for _ in 0..steps {
        if let Err(x) = s.step(&mut p, g, dt) {
            out = vec![err(&x)];
            break;
        }
    }
    if out.is_empty() {
        ps_toks(&mut out, &p);
        dump(&mut out, &s);
    }
    m.row("FlipSolver::step", Class::S, &ins, &out);
}

/// One segment of a scenario: the blend, APIC, the particles kept, those appended, the steps.
struct Seg {
    ratio: f64,
    apic: bool,
    keep: usize,
    add: Vec<FluidParticle>,
    steps: usize,
}

fn flip_scenario_row(
    m: &mut Module,
    nx: usize,
    ny: usize,
    dx: f64,
    g: DVec3,
    dt: f64,
    segs: &[Seg],
) {
    let mut ins = vec![usz(nx), usz(ny), f(dx)];
    v3(&mut ins, g);
    ins.extend([f(dt), usz(segs.len())]);
    let mut s = FlipSolver::new(nx, ny, dx, 0.95).expect("a valid grid");
    let mut p: Vec<FluidParticle> = Vec::new();
    let mut out = Vec::new();
    for sg in segs {
        ins.extend([f(sg.ratio), Tok::I(i64::from(sg.apic)), usz(sg.keep)]);
        ps_toks(&mut ins, &sg.add);
        ins.push(usz(sg.steps));
        if !out.is_empty() {
            continue;
        }
        s.flip_ratio = sg.ratio;
        s.use_apic = sg.apic;
        p.truncate(sg.keep);
        p.extend(sg.add.iter().copied());
        for _ in 0..sg.steps {
            if let Err(x) = s.step(&mut p, g, dt) {
                out = vec![err(&x)];
                break;
            }
        }
    }
    if out.is_empty() {
        ps_toks(&mut out, &p);
        dump(&mut out, &s);
    }
    m.row("FlipSolver::scenario", Class::S, &ins, &out);
}

/// A block of 2D particles over [x0, x1] × [y0, y1] at spacing sp, mass m, moving at (vx, vy).
#[allow(clippy::too_many_arguments)]
fn flip_block(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    sp: f64,
    m: f64,
    vx: f64,
    vy: f64,
) -> Vec<FluidParticle> {
    let mut out = Vec::new();
    let mut j = 0.0;
    while y0 + j * sp <= y1 + 1e-12 {
        let mut i = 0.0;
        while x0 + i * sp <= x1 + 1e-12 {
            let mut p = FluidParticle::new_2d(x0 + i * sp, y0 + j * sp, m);
            p.velocity = DVec3::new(vx, vy, 0.0);
            out.push(p);
            i += 1.0;
        }
        j += 1.0;
    }
    out
}

/// n seeded particles in [lo, hi]² (3D when `deep`), moving up to v per axis, masses log-uniform.
fn flip_cloud(rng: &mut Rng, n: usize, lo: f64, hi: f64, v: f64, deep: bool) -> Vec<FluidParticle> {
    (0..n)
        .map(|_| {
            let mut p = FluidParticle::new(
                DVec3::new(
                    rng.uniform(lo, hi),
                    rng.uniform(lo, hi),
                    if deep { rng.uniform(-0.2, 0.2) } else { 0.0 },
                ),
                rng.log_uniform(-4.0, -1.0),
            );
            p.velocity = DVec3::new(
                rng.uniform(-v, v),
                rng.uniform(-v, v),
                if deep { rng.uniform(-v, v) } else { 0.0 },
            );
            p
        })
        .collect()
}

fn flip_rows(m: &mut Module, rng: &mut Rng) {
    let o = DVec3::ZERO;
    let g = DVec3::new(0.0, -9.81, 0.0);
    // new: the Rust tests', the refusals in Rust's order, the blend's clamp and default, sizes.
    flip_new_row(m, 16, 16, 0.1, 0.95);
    flip_new_row(m, 2, 16, 0.1, 0.95);
    flip_new_row(m, 16, 16, 0.0, 0.95);
    flip_new_row(m, 3, 3, f64::NAN, 0.5);
    flip_new_row(m, 16, 3, 0.1, 0.5);
    flip_new_row(m, 4, 4, 1.0, 0.0);
    flip_new_row(m, 4, 7, 0.2, 1.0);
    flip_new_row(m, 50, 7, 0.02, 0.3);
    flip_new_row(m, usize::MAX, 4, 0.1, 0.5);
    flip_new_row(m, 4, usize::MAX, 0.1, 0.5);
    for &x in &specials() {
        flip_new_row(m, 8, 8, x, 0.5);
        flip_new_row(m, 8, 8, 0.1, x);
    }
    for x in [1.5, -0.5, 2.0, -0.0, 0.5] {
        flip_new_row(m, 6, 5, 0.25, x);
    }
    // The Rust tests' scenes.
    let one = |x: f64, y: f64, vx: f64, vy: f64| {
        let mut p = FluidParticle::new_2d(x, y, 0.01);
        p.velocity = DVec3::new(vx, vy, 0.0);
        vec![p]
    };
    flip_step_row(m, 8, 8, 0.1, 0.95, false, g, 0.01, 1, &[]);
    flip_step_row(
        m,
        16,
        16,
        0.1,
        0.95,
        false,
        o,
        0.1,
        1,
        &one(0.8, 0.8, 1.0, 0.0),
    );
    flip_step_row(
        m,
        16,
        16,
        0.1,
        0.0,
        false,
        g,
        0.01,
        1,
        &one(0.8, 0.8, 0.0, 0.0),
    );
    flip_step_row(
        m,
        8,
        8,
        0.1,
        1.0,
        false,
        o,
        0.01,
        1,
        &one(0.4, 0.4, 1.0, 0.0),
    );
    flip_step_row(
        m,
        8,
        8,
        0.1,
        0.0,
        false,
        o,
        0.01,
        1,
        &one(0.4, 0.4, 1.0, 0.0),
    );
    flip_step_row(
        m,
        8,
        8,
        0.1,
        0.95,
        false,
        o,
        0.1,
        1,
        &one(0.05, 0.05, -10.0, -10.0),
    );
    flip_step_row(
        m,
        8,
        8,
        0.1,
        f64::NAN,
        false,
        o,
        0.01,
        1,
        &one(0.3, 0.3, 0.0, 0.0),
    );
    let mut radial = Vec::new();
    for j in 2..14 {
        for i in 2..14 {
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let x = (f64::from(i) + ox) * 0.1;
                let y = (f64::from(j) + oy) * 0.1;
                let mut p = FluidParticle::new_2d(x, y, 0.001);
                p.velocity = DVec3::new(x - 0.8, y - 0.8, 0.0);
                radial.push(p);
            }
        }
    }
    flip_step_row(m, 16, 16, 0.1, 0.0, false, o, 1e-3, 1, &radial);
    flip_step_row(m, 16, 16, 0.1, 0.0, true, o, 1e-3, 1, &radial);
    let falling: Vec<FluidParticle> = [(0.5, 0.8), (0.6, 0.8), (0.7, 0.8)]
        .iter()
        .map(|&(x, y)| FluidParticle::new_2d(x, y, 0.01))
        .collect();
    flip_step_row(m, 16, 16, 0.1, 0.95, false, g, 0.01, 10, &falling);
    let twenty: Vec<FluidParticle> = (0..20)
        .map(|i| {
            FluidParticle::new_2d(
                0.3 + f64::from(i % 5) * 0.05,
                0.3 + f64::from(i / 5) * 0.05,
                0.01,
            )
        })
        .collect();
    flip_step_row(
        m,
        16,
        16,
        0.1,
        0.9,
        false,
        DVec3::new(0.0, -1.0, 0.0),
        0.01,
        100,
        &twenty,
    );
    // The sloshing tank (a block moving right in the left of the tank) and the falling block, by PIC, FLIP 0.95,
    // FLIP 1.0 and APIC, at frames 1, 2, 10, 60 and 300.
    let tank = flip_block(0.15, 0.55, 0.15, 0.85, 0.05, 0.001, 0.5, 0.0);
    let block = flip_block(0.3, 0.5, 0.8, 1.0, 0.025, 0.001, 0.0, 0.0);
    for (ratio, apic) in [(0.0, false), (0.95, false), (1.0, false), (0.0, true)] {
        for steps in [1, 2, 10, 60, 300] {
            flip_step_row(m, 16, 16, 0.1, ratio, apic, g, 0.005, steps, &tank);
        }
    }
    for (ratio, apic) in [(0.95, false), (0.0, true)] {
        for steps in [1, 2, 10, 60, 300] {
            flip_step_row(m, 16, 24, 0.05, ratio, apic, g, 0.01, steps, &block);
        }
    }
    // A pool at rest without gravity (one sweep converges), and a deep pool under gravity (every sweep runs).
    let pool = flip_block(0.1, 0.9, 0.1, 0.5, 0.05, 0.001, 0.0, 0.0);
    flip_step_row(m, 12, 12, 0.1, 0.95, false, o, 0.01, 3, &pool);
    flip_step_row(m, 12, 12, 0.1, 0.95, false, g, 0.01, 3, &pool);
    // The specials as dt, the blend and each gravity component; as a particle's position, velocity and mass, among
    // others; and a particle outside the grid on each side, and in 3D.
    let five = flip_block(0.2, 0.4, 0.2, 0.3, 0.1, 0.01, 0.3, -0.2);
    for &x in &specials() {
        flip_step_row(m, 8, 8, 0.1, 0.95, false, g, x, 1, &five);
        flip_step_row(m, 8, 8, 0.1, x, false, g, 0.01, 2, &five);
        flip_step_row(
            m,
            8,
            8,
            0.1,
            0.95,
            false,
            DVec3::new(x, -9.81, 0.0),
            0.01,
            1,
            &five,
        );
        flip_step_row(
            m,
            8,
            8,
            0.1,
            0.95,
            true,
            DVec3::new(0.0, x, 0.0),
            0.01,
            1,
            &five,
        );
        for slot in 0..7 {
            let mut ps = five.clone();
            let p = &mut ps[2];
            match slot {
                0 => p.position.x = x,
                1 => p.position.y = x,
                2 => p.position.z = x,
                3 => p.velocity.x = x,
                4 => p.velocity.y = x,
                5 => p.velocity.z = x,
                _ => p.mass = x,
            }
            flip_step_row(m, 8, 8, 0.1, 0.95, slot % 2 == 1, g, 0.01, 2, &ps);
        }
    }
    let mut outside = five.clone();
    outside[0].position = DVec3::new(-0.05, 0.3, 0.0);
    outside[1].position = DVec3::new(0.3, 0.9, 0.0);
    outside[2].position = DVec3::new(1.5, -0.4, 0.0);
    outside[3].position = DVec3::new(0.7, 0.7, 0.0);
    outside[4].position = DVec3::new(0.3, 0.3, 0.25);
    outside[4].velocity.z = 1.0;
    flip_step_row(m, 8, 8, 0.1, 0.95, false, g, 0.01, 3, &outside);
    flip_step_row(m, 8, 8, 0.1, 0.0, true, g, 0.01, 3, &outside);
    flip_step_row(m, 8, 8, 0.1, 0.95, false, g, 1.0, 2, &five);
    // A particle on the corner node moving at f64::MAX in y, under gravity f64::MAX: only that node's vy overflows, and
    // no interior divergence reads it, so vx stays finite and the check must see vy (no random numbers drawn).
    let mut corner = FluidParticle::new_2d(0.0, 0.0, 1.0);
    corner.velocity = DVec3::new(0.0, f64::MAX, 0.0);
    flip_step_row(
        m,
        8,
        8,
        0.1,
        0.95,
        false,
        DVec3::new(0.0, f64::MAX, 0.0),
        1.0,
        1,
        &[corner],
    );
    // Seeded scenes: grid shapes, cell sizes, blends, APIC, gravities, time steps, clouds in 2D and 3D, some past
    // the walls.
    for k in 0..40 {
        let nx = 4 + usize::try_from(rng.int(0, 16)).unwrap_or(0);
        let ny = 4 + usize::try_from(rng.int(0, 16)).unwrap_or(0);
        let dx = rng.log_uniform(-2.0, 0.0);
        let ratio = if k % 5 == 0 {
            rng.uniform(-0.5, 1.5)
        } else {
            rng.unit()
        };
        let apic = k % 3 == 0;
        let gv = DVec3::new(rng.uniform(-3.0, 3.0), rng.uniform(-12.0, 0.0), 0.0);
        let dt = rng.log_uniform(-4.0, -1.3);
        let steps = 1 + usize::try_from(rng.int(0, 4)).unwrap_or(0);
        let ext = dx * (nx.min(ny) as f64 - 1.0);
        let n = usize::try_from(rng.int(0, 60)).unwrap_or(0);
        let lo = if k % 7 == 6 { -0.2 * ext } else { 0.05 * ext };
        let cl = flip_cloud(rng, n, lo, 0.95 * ext, 2.0 * ext, k % 4 == 3);
        flip_step_row(m, nx, ny, dx, ratio, apic, gv, dt, steps, &cl);
    }
    // Scenarios: the APIC matrices truncated, grown, kept across a FLIP segment; all particles dropped and new ones
    // added; a NaN blend in a later segment.
    let c40 = flip_cloud(rng, 40, 0.1, 1.3, 0.5, false);
    let c15 = flip_cloud(rng, 15, 0.1, 1.3, 0.5, false);
    let seg = |ratio: f64, apic: bool, keep: usize, add: &[FluidParticle], steps: usize| Seg {
        ratio,
        apic,
        keep,
        add: add.to_vec(),
        steps,
    };
    flip_scenario_row(
        m,
        16,
        16,
        0.1,
        g,
        0.005,
        &[
            seg(0.0, true, 0, &c40, 3),
            seg(0.0, true, 25, &[], 3),
            seg(0.0, true, 25, &c15, 3),
        ],
    );
    flip_scenario_row(
        m,
        16,
        16,
        0.1,
        g,
        0.005,
        &[
            seg(0.0, true, 0, &c40, 2),
            seg(0.9, false, 40, &[], 2),
            seg(0.0, true, 40, &[], 2),
        ],
    );
    flip_scenario_row(
        m,
        16,
        16,
        0.1,
        g,
        0.005,
        &[
            seg(0.0, true, 0, &c40, 2),
            seg(0.9, false, 30, &[], 2),
            seg(0.0, true, 30, &c15, 2),
        ],
    );
    flip_scenario_row(
        m,
        12,
        10,
        0.1,
        g,
        0.01,
        &[
            seg(0.95, false, 0, &c15, 2),
            seg(f64::NAN, false, 15, &[], 2),
            seg(0.95, false, 15, &[], 2),
        ],
    );
    flip_scenario_row(
        m,
        16,
        16,
        0.1,
        g,
        0.005,
        &[
            seg(0.0, true, 0, &c40, 2),
            seg(0.0, true, 0, &[], 2),
            seg(0.0, true, 0, &c15, 2),
        ],
    );
    for _ in 0..8 {
        let nsegs = 2 + usize::try_from(rng.int(0, 2)).unwrap_or(0);
        let mut segs = Vec::new();
        let mut have = 0usize;
        for _ in 0..nsegs {
            let keep = usize::try_from(rng.int(0, i64::try_from(have).unwrap_or(0))).unwrap_or(0);
            let na = usize::try_from(rng.int(0, 20)).unwrap_or(0);
            let add = flip_cloud(rng, na, 0.1, 1.3, 0.8, false);
            have = keep + na;
            segs.push(Seg {
                ratio: rng.unit(),
                apic: rng.unit() < 0.5,
                keep,
                add,
                steps: 1 + usize::try_from(rng.int(0, 3)).unwrap_or(0),
            });
        }
        flip_scenario_row(m, 16, 16, 0.1, g, 0.005, &segs);
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("coupling");
    let mut rng = Rng::new(0x60_c0_0a);
    shape_rows(&mut m, &mut rng);
    body_rows(&mut m, &mut rng);
    coupling_rows(&mut m, &mut rng);
    drag_rows(&mut m, &mut rng);
    level_set_rows(&mut m, &mut rng);
    flow_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    let mut rng_b = Rng::new(0x60_c0_0b);
    flip_rows(&mut m, &mut rng_b);
    m
}
