//! `mpm.tsv` — `rust-old/src/mpm.rs` (M6 part c, seed 0x60_c0_0c).
//!
//! Tokens: `<model>` is `e:ConstitutiveModel.<Variant>` and five values, the bulk modulus, the viscosity, Young's
//! modulus, Poisson's ratio and the friction angle (0 for a field the variant lacks); `<mp>` is an `MpmParticle`'s 11
//! tokens: the deformation gradient (row-major), `<model>`, the reference volume; `<ps>` is `i:<n>` and n particles
//! (`m_common::particle`); `<mps>` is `i:<m>` and m `<mp>`; `<dump>` is the solver's private state (`oracle_dump`),
//! `v:<nx·ny>` mass, vx, vy, fx and fy.
//!
//! Rows:
//! - `ConstitutiveModel::validate` takes `<model>` and gives nothing (Ok) or `E:` (class A);
//! - `MpmParticle::new` takes `<model>` and the volume and gives `<mp>` (class A); `MpmParticle::det_f` takes `<mp>`
//!   (class B); `MpmParticle::stress` takes `<mp>` and gives four values (class C: ln, and the SVD's hypot, atan2,
//!   sin and cos; the corrected oracle where glibc misrounds);
//! - `MpmSolver::new` takes `i:` nx and ny and dx, and gives `i:` nx and ny, dx and `i:` the five fields' lengths, or
//!   `E:` (class A);
//! - `MpmSolver::step` takes `i:` nx and ny, dx, gravity, dt, `i:<steps>`, `<ps>` and `<mps>`, and gives the particles,
//!   the MPM states and `<dump>` after that many steps of one fresh solver, or the first error (class S, on the
//!   transcendentals above and Drucker–Prager's ln, exp, sin and hypot: the corrected oracle where glibc misrounds);
//! - `ConstitutiveModel::from_json` decodes a serde_json document to `<model>` or `E:Deserialize` (class A).

use hisab::DVec3;
use pravash::FluidParticle;
use pravash::mpm::{ConstitutiveModel, MpmParticle, MpmSolver};

use crate::grids::{Rng, specials};
use crate::m_common::particle;
use crate::m_grid::{col, usz};
use crate::tsv::{Class, Module, Tok, err, f, s};

fn model(m: &ConstitutiveModel) -> Vec<Tok> {
    let (name, v) = match *m {
        ConstitutiveModel::Fluid {
            bulk_modulus,
            viscosity,
        } => ("Fluid", [bulk_modulus, viscosity, 0.0, 0.0, 0.0]),
        ConstitutiveModel::NeoHookean {
            youngs_modulus,
            poisson_ratio,
        } => ("NeoHookean", [0.0, 0.0, youngs_modulus, poisson_ratio, 0.0]),
        ConstitutiveModel::DruckerPrager {
            youngs_modulus,
            poisson_ratio,
            friction_angle,
        } => (
            "DruckerPrager",
            [0.0, 0.0, youngs_modulus, poisson_ratio, friction_angle],
        ),
        _ => ("Unknown", [0.0; 5]),
    };
    let mut out = vec![Tok::E("ConstitutiveModel", name.to_string())];
    out.extend(v.iter().map(|&x| f(x)));
    out
}

fn mp_toks(m: &MpmParticle) -> Vec<Tok> {
    let mut out: Vec<Tok> = m.deformation_grad.iter().map(|&x| f(x)).collect();
    out.extend(model(&m.model));
    out.push(f(m.volume_0));
    out
}

fn ps_toks(out: &mut Vec<Tok>, ps: &[FluidParticle]) {
    out.push(usz(ps.len()));
    for p in ps {
        out.extend(particle(p));
    }
}

fn mps_toks(out: &mut Vec<Tok>, ms: &[MpmParticle]) {
    out.push(usz(ms.len()));
    for m in ms {
        out.extend(mp_toks(m));
    }
}

fn fluid(k: f64, mu: f64) -> ConstitutiveModel {
    ConstitutiveModel::Fluid {
        bulk_modulus: k,
        viscosity: mu,
    }
}

fn neo(e: f64, nu: f64) -> ConstitutiveModel {
    ConstitutiveModel::NeoHookean {
        youngs_modulus: e,
        poisson_ratio: nu,
    }
}

fn dp(e: f64, nu: f64, phi: f64) -> ConstitutiveModel {
    ConstitutiveModel::DruckerPrager {
        youngs_modulus: e,
        poisson_ratio: nu,
        friction_angle: phi,
    }
}

fn with_f(m: ConstitutiveModel, v0: f64, fg: [f64; 4]) -> MpmParticle {
    let mut p = MpmParticle::new(m, v0);
    p.deformation_grad = fg;
    p
}

fn validate_rows(m: &mut Module) {
    let mut ms = vec![
        fluid(1000.0, 0.01),
        fluid(1000.0, 0.0),
        fluid(50.0, 0.0),
        neo(1000.0, 0.3),
        neo(1e4, 0.5),
        neo(1e4, 0.499_999),
        neo(1e4, -1.0),
        neo(1e4, -0.999_999),
        neo(1e4, 0.0),
        dp(1e4, 0.3, 30.0),
        dp(1e4, 0.3, 0.0),
        dp(1e4, 0.3, -0.0),
        dp(1e4, 0.3, 90.0),
        dp(1e4, 0.3, 89.999),
        dp(1e4, 0.3, -1.0),
        dp(0.0, 0.3, 30.0),
        dp(1e4, 0.6, 30.0),
    ];
    for &x in &specials() {
        ms.push(fluid(x, 0.01));
        ms.push(fluid(1000.0, x));
        ms.push(neo(x, 0.3));
        ms.push(neo(1e4, x));
        ms.push(dp(x, 0.3, 30.0));
        ms.push(dp(1e4, x, 30.0));
        ms.push(dp(1e4, 0.3, x));
    }
    for md in &ms {
        let out = match md.validate() {
            Ok(()) => vec![],
            Err(x) => vec![err(&x)],
        };
        m.row("ConstitutiveModel::validate", Class::A, &model(md), &out);
    }
    for (md, v0) in [
        (fluid(1000.0, 0.01), 0.001),
        (neo(1e4, 0.3), 6.25e-4),
        (dp(1e4, 0.3, 30.0), -1.0),
        (fluid(f64::NAN, 0.0), f64::INFINITY),
    ] {
        let mut ins = model(&md);
        ins.push(f(v0));
        m.row(
            "MpmParticle::new",
            Class::A,
            &ins,
            &mp_toks(&MpmParticle::new(md, v0)),
        );
    }
}

/// The deformation gradients the stress is probed at: the identity, stretches and compressions, shears, rational
/// rotations (3-4-5, 5-12-13) alone and composed with a stretch, the Rust tests', an inversion and a singular one, a
/// det outside the step's clamp, and each special in each entry.
fn gradients(rng: &mut Rng) -> Vec<[f64; 4]> {
    let mut out = vec![
        [1.0, 0.0, 0.0, 1.0],
        [1.5, 0.0, 0.0, 1.0],
        [0.9, 0.0, 0.0, 1.0],
        [0.8, 0.0, 0.0, 0.8],
        [0.95, 0.0, 0.0, 0.95],
        [1.2, 0.0, 0.0, 1.1],
        [1.3, 0.0, 0.0, 0.7],
        [1.5, 0.2, -0.3, 0.8],
        [0.7, -0.4, 0.9, 1.1],
        [2.0, 0.0, 0.0, 0.5],
        [1.0, 0.3, 0.0, 1.0],
        [1.0, 0.0, -0.25, 1.0],
        [0.6, -0.8, 0.8, 0.6],
        [5.0 / 13.0, 12.0 / 13.0, -12.0 / 13.0, 5.0 / 13.0],
        [0.6 * 1.2, -0.8 * 0.9, 0.8 * 1.2, 0.6 * 0.9],
        [-1.0, 0.0, 0.0, 1.0],
        [1.0, 2.0, 0.5, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [200.0, 0.0, 0.0, 100.0],
        [1e-3, 0.0, 0.0, 0.05],
        [1.0, 0.0, 0.0, -1.0],
    ];
    for &x in &specials() {
        for k in 0..4 {
            let mut g = [1.1, 0.05, -0.02, 0.9];
            g[k] = x;
            out.push(g);
        }
    }
    for _ in 0..40 {
        out.push([
            1.0 + rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            1.0 + rng.uniform(-0.3, 0.3),
        ]);
    }
    out
}

fn particle_rows(m: &mut Module, rng: &mut Rng) {
    let gs = gradients(rng);
    let models = [
        fluid(1000.0, 0.01),
        fluid(50.0, 0.0),
        neo(1000.0, 0.3),
        neo(1e4, 0.45),
        neo(1e4, -0.5),
        dp(1e4, 0.3, 30.0),
        dp(1e4, 0.2, 45.0),
        dp(5e3, 0.35, 0.0),
    ];
    for (k, md) in models.iter().enumerate() {
        for g in &gs {
            let p = with_f(*md, 0.001, *g);
            if k == 0 {
                m.row(
                    "MpmParticle::det_f",
                    Class::B,
                    &mp_toks(&p),
                    &[f(p.det_f())],
                );
            }
            let st = p.stress();
            m.row(
                "MpmParticle::stress",
                Class::C,
                &mp_toks(&p),
                &[f(st[0]), f(st[1]), f(st[2]), f(st[3])],
            );
        }
    }
    // The specials as each model parameter (stress doesn't validate), at a stretched F.
    for &x in &specials() {
        for md in [
            fluid(x, 0.0),
            neo(x, 0.3),
            neo(1e4, x),
            dp(x, 0.3, 30.0),
            dp(1e4, x, 30.0),
            dp(1e4, 0.3, x),
        ] {
            let p = with_f(md, 0.001, [1.2, 0.1, 0.0, 0.85]);
            let st = p.stress();
            m.row(
                "MpmParticle::stress",
                Class::C,
                &mp_toks(&p),
                &[f(st[0]), f(st[1]), f(st[2]), f(st[3])],
            );
        }
    }
}

fn solver_new_row(m: &mut Module, nx: usize, ny: usize, dx: f64) {
    let ins = [usz(nx), usz(ny), f(dx)];
    let out = match MpmSolver::new(nx, ny, dx) {
        Ok(sv) => {
            let d = sv.oracle_dump();
            let mut o = vec![usz(d.nx), usz(d.ny), f(d.dx)];
            for len in [d.mass.len(), d.vx.len(), d.vy.len(), d.fx.len(), d.fy.len()] {
                o.push(usz(len));
            }
            o
        }
        Err(x) => vec![err(&x)],
    };
    m.row("MpmSolver::new", Class::A, &ins, &out);
}

/// One step row: a fresh solver, `steps` steps of the particles and their MPM states.
#[allow(clippy::too_many_arguments)]
fn step_row(
    m: &mut Module,
    nx: usize,
    ny: usize,
    dx: f64,
    g: DVec3,
    dt: f64,
    steps: usize,
    ps: &[FluidParticle],
    ms: &[MpmParticle],
) {
    let mut ins = vec![
        usz(nx),
        usz(ny),
        f(dx),
        f(g.x),
        f(g.y),
        f(g.z),
        f(dt),
        usz(steps),
    ];
    ps_toks(&mut ins, ps);
    mps_toks(&mut ins, ms);
    let mut sv = MpmSolver::new(nx, ny, dx).expect("a valid grid");
    let mut p = ps.to_vec();
    let mut q = ms.to_vec();
    let mut out = Vec::new();
    for _ in 0..steps {
        if let Err(x) = sv.step(&mut p, &mut q, g, dt) {
            out = vec![err(&x)];
            break;
        }
    }
    if out.is_empty() {
        ps_toks(&mut out, &p);
        mps_toks(&mut out, &q);
        let d = sv.oracle_dump();
        for v in [&d.mass, &d.vx, &d.vy, &d.fx, &d.fy] {
            col(&mut out, v);
        }
    }
    m.row("MpmSolver::step", Class::S, &ins, &out);
}

/// A block of particles over [x0, x1] × [y0, y1] at spacing sp, each of mass ρ·sp² and reference volume sp², all of
/// `md` with F = fg.
#[allow(clippy::too_many_arguments)]
fn block(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    sp: f64,
    rho: f64,
    md: ConstitutiveModel,
    fg: [f64; 4],
) -> (Vec<FluidParticle>, Vec<MpmParticle>) {
    let v0 = sp * sp;
    let (mut ps, mut ms) = (Vec::new(), Vec::new());
    let mut j = 0.0;
    while y0 + j * sp <= y1 + 1e-12 {
        let mut i = 0.0;
        while x0 + i * sp <= x1 + 1e-12 {
            ps.push(FluidParticle::new_2d(x0 + i * sp, y0 + j * sp, rho * v0));
            ms.push(with_f(md, v0, fg));
            i += 1.0;
        }
        j += 1.0;
    }
    (ps, ms)
}

fn step_rows(m: &mut Module, rng: &mut Rng) {
    let g = DVec3::new(0.0, -9.81, 0.0);
    let o = DVec3::ZERO;
    let id = [1.0, 0.0, 0.0, 1.0];
    // new: the Rust tests', every refusal in order, sizes.
    solver_new_row(m, 16, 16, 0.1);
    solver_new_row(m, 3, 16, 0.1);
    solver_new_row(m, 16, 3, f64::NAN);
    solver_new_row(m, 8, 8, 0.0);
    solver_new_row(m, 4, 9, 0.25);
    solver_new_row(m, usize::MAX, 4, 0.1);
    for &x in &specials() {
        solver_new_row(m, 6, 5, x);
    }
    // The Rust tests' scenes.
    let one = |x: f64, y: f64, md: ConstitutiveModel, v0: f64| {
        (
            vec![FluidParticle::new_2d(x, y, 0.01)],
            vec![MpmParticle::new(md, v0)],
        )
    };
    step_row(m, 8, 8, 0.1, g, 0.01, 1, &[], &[]);
    let (p1, m1) = one(0.8, 0.8, fluid(1000.0, 0.01), 0.001);
    step_row(m, 16, 16, 0.1, g, 0.01, 1, &p1, &m1);
    let (p2, m2) = one(0.4, 0.4, neo(1e4, 0.5), 0.001);
    step_row(m, 8, 8, 0.1, o, 0.01, 1, &p2, &m2);
    let (p3, m3) = one(0.4, 0.4, fluid(1e3, 0.0), -1.0);
    step_row(m, 8, 8, 0.1, o, 0.01, 1, &p3, &m3);
    let (p4, m4) = one(0.35, 0.05, fluid(1e3, 0.0), 0.001);
    for steps in [1, 10, 200] {
        step_row(m, 8, 8, 0.1, g, 0.005, steps, &p4, &m4);
    }
    let (p5, m5) = block(
        0.25,
        0.425,
        0.25,
        0.425,
        0.025,
        1.6,
        fluid(50.0, 0.0),
        [0.8, 0.0, 0.0, 0.8],
    );
    step_row(m, 16, 16, 0.05, o, 1e-3, 20, &p5, &m5);
    // The plan's scenes, at frames 1, 2, 10, 60 and 300: an elastic block falling to the floor, a sand column and a
    // fluid column collapsing (density 1000, so the wave speed allows dt 1e-3).
    let elastic = block(0.3, 0.5, 0.45, 0.65, 0.025, 1000.0, neo(1e4, 0.3), id);
    let sand = block(
        0.35,
        0.45,
        0.025,
        0.4,
        0.025,
        1000.0,
        dp(1e4, 0.3, 30.0),
        id,
    );
    let water = block(0.05, 0.25, 0.025, 0.4, 0.025, 1000.0, fluid(1e3, 0.0), id);
    for (ps, ms) in [&elastic, &sand, &water] {
        for steps in [1, 2, 10, 60, 300] {
            step_row(m, 16, 16, 0.05, g, 1e-3, steps, ps, ms);
        }
    }
    // A mixed column (sand on fluid on an elastic base), and the sand at another friction angle.
    let (mut pm, mut mm) = block(0.3, 0.45, 0.025, 0.1, 0.025, 1000.0, neo(1e4, 0.3), id);
    let (pa, ma) = block(0.3, 0.45, 0.125, 0.2, 0.025, 1000.0, fluid(1e3, 0.0), id);
    let (pb, mb) = block(0.3, 0.45, 0.225, 0.3, 0.025, 1000.0, dp(1e4, 0.3, 40.0), id);
    pm.extend(pa);
    pm.extend(pb);
    mm.extend(ma);
    mm.extend(mb);
    for steps in [1, 10, 60] {
        step_row(m, 16, 16, 0.05, g, 1e-3, steps, &pm, &mm);
    }
    let sand0 = block(0.35, 0.45, 0.025, 0.3, 0.025, 1000.0, dp(1e4, 0.3, 0.0), id);
    step_row(m, 16, 16, 0.05, g, 1e-3, 30, &sand0.0, &sand0.1);
    // The refusals in Rust's order: dt, the counts, each model, each volume; the specials as dt, as gravity, in a
    // particle's slots, in F's entries (the det clamp), and as the reference volume.
    let (pf, mf) = block(
        0.2,
        0.3,
        0.2,
        0.25,
        0.05,
        1000.0,
        neo(1e4, 0.3),
        [1.05, 0.02, 0.0, 0.97],
    );
    for &x in &specials() {
        step_row(m, 8, 8, 0.1, g, x, 1, &pf, &mf);
        step_row(m, 8, 8, 0.1, DVec3::new(x, -9.81, 0.0), 1e-3, 2, &pf, &mf);
        step_row(m, 8, 8, 0.1, DVec3::new(0.0, x, 0.0), 1e-3, 2, &pf, &mf);
        for slot in 0..5 {
            let mut ps = pf.clone();
            let p = &mut ps[1];
            match slot {
                0 => p.position.x = x,
                1 => p.position.y = x,
                2 => p.velocity.x = x,
                3 => p.velocity.y = x,
                _ => p.mass = x,
            }
            step_row(m, 8, 8, 0.1, g, 1e-3, 2, &ps, &mf);
        }
        for k in 0..4 {
            let mut ms = mf.clone();
            ms[2].deformation_grad[k] = x;
            step_row(m, 8, 8, 0.1, g, 1e-3, 2, &pf, &ms);
        }
        let mut mv = mf.clone();
        mv[0].volume_0 = x;
        step_row(m, 8, 8, 0.1, g, 1e-3, 1, &pf, &mv);
    }
    step_row(m, 8, 8, 0.1, g, 1e-3, 1, &pf, &mf[1..]);
    let mut mbad = mf.clone();
    mbad[3].model = dp(1e4, 0.3, 90.0);
    mbad[1].volume_0 = 0.0;
    step_row(m, 8, 8, 0.1, g, 1e-3, 1, &pf, &mbad);
    let mut mclamp = mf.clone();
    mclamp[0].model = fluid(1e-9, 0.0);
    mclamp[0].deformation_grad = [200.0, 0.0, 0.0, 100.0];
    mclamp[1].model = fluid(1e-9, 0.0);
    mclamp[1].deformation_grad = [1e-3, 0.0, 0.0, 0.05];
    step_row(m, 8, 8, 0.1, g, 1e-3, 2, &pf, &mclamp);
    // Determinants on both sides of each end of the clamp [1e-4, 1e4] (no random numbers drawn): 5e-4 and 9e3 are
    // kept, 5e-5 and 1.1e4 reset.
    let mut mrange = mf.clone();
    for (k, fg) in [
        [0.01, 0.0, 0.0, 0.05],
        [100.0, 0.0, 0.0, 90.0],
        [0.001, 0.0, 0.0, 0.05],
        [110.0, 0.0, 0.0, 100.0],
    ]
    .iter()
    .enumerate()
    {
        mrange[k].model = fluid(1e-9, 0.0);
        mrange[k].deformation_grad = *fg;
    }
    step_row(m, 8, 8, 0.1, g, 1e-3, 1, &pf, &mrange);
    // Particles outside the walls, moving outward, in 3D.
    let mut pw = pf.clone();
    pw[0].position = DVec3::new(-0.05, 0.3, 0.0);
    pw[0].velocity = DVec3::new(-1.0, 0.0, 0.0);
    pw[1].position = DVec3::new(0.69, 0.3, 0.0);
    pw[1].velocity = DVec3::new(2.0, 0.0, 0.0);
    pw[2].position = DVec3::new(0.3, 0.71, 0.1);
    pw[2].velocity = DVec3::new(0.0, 3.0, -0.5);
    pw[3].position = DVec3::new(0.3, -0.2, 0.0);
    step_row(m, 8, 8, 0.1, g, 1e-2, 3, &pw, &mf);
    // Seeded scenes: grid shapes, cell sizes, gravities, time steps, mixed materials with perturbed F, masses and
    // volumes.
    for k in 0..30 {
        let nx = 4 + usize::try_from(rng.int(0, 14)).unwrap_or(0);
        let ny = 4 + usize::try_from(rng.int(0, 14)).unwrap_or(0);
        let dx = rng.log_uniform(-2.0, -0.5);
        let gv = DVec3::new(rng.uniform(-2.0, 2.0), rng.uniform(-12.0, 0.0), 0.0);
        let dt = rng.log_uniform(-4.5, -2.5);
        let steps = 1 + usize::try_from(rng.int(0, 4)).unwrap_or(0);
        let ext = dx * (nx.min(ny) as f64 - 1.0);
        let n = usize::try_from(rng.int(0, 40)).unwrap_or(0);
        let mut ps = Vec::new();
        let mut ms = Vec::new();
        for _ in 0..n {
            let v0 = rng.log_uniform(-5.0, -2.0);
            let mut p = FluidParticle::new_2d(
                rng.uniform(0.05, 0.95) * ext,
                rng.uniform(0.05, 0.95) * ext,
                1000.0 * v0 * rng.uniform(0.5, 1.5),
            );
            p.velocity = DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0);
            ps.push(p);
            let md = match rng.int(0, 3) {
                0 => fluid(rng.log_uniform(1.0, 3.5), rng.uniform(0.0, 0.01)),
                1 => neo(rng.log_uniform(2.0, 4.0), rng.uniform(-0.5, 0.45)),
                _ => dp(
                    rng.log_uniform(2.0, 4.0),
                    rng.uniform(0.0, 0.45),
                    rng.uniform(0.0, 60.0),
                ),
            };
            let e = 0.05;
            ms.push(with_f(
                md,
                v0,
                [
                    1.0 + rng.uniform(-e, e),
                    rng.uniform(-e, e),
                    rng.uniform(-e, e),
                    1.0 + rng.uniform(-e, e),
                ],
            ));
        }
        let _ = k;
        step_row(m, nx, ny, dx, gv, dt, steps, &ps, &ms);
    }
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let tk = |doc: &str| match serde_json::from_str::<ConstitutiveModel>(doc) {
        Ok(v) => model(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for md in [
        fluid(1000.0, 0.01),
        neo(1e4, 0.3),
        dp(1e4, 0.3, 30.0),
        fluid(f64::NAN, 0.0),
        dp(1e4, 0.3, f64::INFINITY),
    ] {
        docs.push(serde_json::to_string(&md).unwrap_or_default());
    }
    for _ in 0..10 {
        docs.push(
            serde_json::to_string(&fluid(rng.log_uniform(0.0, 6.0), rng.unit()))
                .unwrap_or_default(),
        );
        docs.push(
            serde_json::to_string(&neo(rng.log_uniform(0.0, 6.0), rng.uniform(-1.0, 0.5)))
                .unwrap_or_default(),
        );
        docs.push(
            serde_json::to_string(&dp(
                rng.log_uniform(0.0, 6.0),
                rng.uniform(-1.0, 0.5),
                rng.uniform(0.0, 90.0),
            ))
            .unwrap_or_default(),
        );
    }
    for doc in [
        r#"{"Fluid":{"bulk_modulus":1000,"viscosity":0}}"#,
        r#"{"Fluid":[1000.0,0.01]}"#,
        r#"{"NeoHookean":[10000.0,0.3]}"#,
        r#"{"DruckerPrager":[10000.0,0.3,30.0]}"#,
        r#"{"DruckerPrager":[10000.0,0.3]}"#,
        r#"{"NeoHookean":[10000.0,0.3,30.0]}"#,
        r#"{"Fluid":{"bulk_modulus":1000}}"#,
        r#"{"Fluid":{"bulk_modulus":1000,"viscosity":0,"extra":"x"}}"#,
        r#"{"Fluid":{"viscosity":0,"bulk_modulus":1000}}"#,
        r#"{"NeoHookean":{"youngs_modulus":1e4,"poisson_ratio":0.3,"friction_angle":30}}"#,
        r#"{"DruckerPrager":{"youngs_modulus":1e4,"poisson_ratio":0.3}}"#,
        r#"{"Fluid":{"bulk_modulus":1000,"bulk_modulus":2000,"viscosity":0}}"#,
        r#"{"Fluid":{"bulk_modulus":"1000","viscosity":0}}"#,
        r#"{"Fluid":{"bulk_modulus":null,"viscosity":0}}"#,
        r#"{"Fluid":{"bulk_modulus":1e999,"viscosity":0}}"#,
        r#""Fluid""#,
        r#"{"Fluid":null}"#,
        r#"{"Solid":{"youngs_modulus":1e4,"poisson_ratio":0.3}}"#,
        r#"{"fluid":{"bulk_modulus":1000,"viscosity":0}}"#,
        r#"{"Fluid":{"bulk_modulus":1000,"viscosity":0},"NeoHookean":{"youngs_modulus":1,"poisson_ratio":0}}"#,
        r#"{}"#,
        r#"[]"#,
        "null",
        r#" { "DruckerPrager" : { "friction_angle" : 30 , "youngs_modulus" : 1E4 , "poisson_ratio" : 0.3 } } "#,
    ] {
        docs.push(doc.to_string());
    }
    for doc in &docs {
        m.row(
            "ConstitutiveModel::from_json",
            Class::A,
            &[s(doc.as_str())],
            &tk(doc),
        );
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("mpm");
    let mut rng = Rng::new(0x60_c0_0c);
    validate_rows(&mut m);
    particle_rows(&mut m, &mut rng);
    step_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    m
}
