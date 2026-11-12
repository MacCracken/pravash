//! `phase_field.tsv` — `rust-old/src/phase_field.rs`, `PhaseField` (M3 part d).
//!
//! A state is `i:` nx, `i:` ny, dx, `v:` phi, mobility, epsilon; the scratch dump is `i:` the scratch length. A
//! row's input field is made as `m_vof.rs`'s are: `PhaseField::new(nx, ny, dx)`, or `new(4, 4, 0.1)` when that is
//! refused, then the public fields.
//!
//! Rows:
//! - `PhaseField::new` takes `i:` nx, ny, dx and gives the state and the dump, or `E:<Variant>`;
//! - `PhaseField::set_circle` (cx, cy, radius) and `PhaseField::set_rect` (x_min, y_min, x_max, y_max) take a state
//!   and the shape and give the state;
//! - `PhaseField::is_phase_b` over every cell (`v:<n>` flags), `PhaseField::is_interface` with `v:<k>` thresholds
//!   (`v:<k·n>`, threshold-major), and `PhaseField::phase_b_area`;
//! - `PhaseField::step` takes a state, `v:` vx, `v:` vy (both empty for no advection), dt and `i:` nsteps, and gives
//!   the state and the dump after nsteps steps, or `E:<Variant>` from the first that fails;
//! - `PhaseField::from_json` decodes a serde_json document.
//!
//! Every operation is +, −, ×, ÷ and clamp, so every row is expected bit-exact.

use pravash::phase_field::PhaseField;

use crate::grids::{Rng, specials};
use crate::m_grid::{col, fx, usz};
use crate::tsv::{Class, Module, Tok, b, err, f, s};

fn state(p: &PhaseField) -> Vec<Tok> {
    let mut out = vec![usz(p.nx), usz(p.ny), f(p.dx)];
    col(&mut out, &p.phi);
    out.extend([f(p.mobility), f(p.epsilon)]);
    out
}

fn dump(p: &PhaseField) -> Vec<Tok> {
    vec![usz(p.oracle_dump().scratch.len())]
}

fn fresh(src: &PhaseField) -> PhaseField {
    let small = src.nx < 1 << 16 && src.ny < 1 << 16;
    let mut p = (if small {
        PhaseField::new(src.nx, src.ny, src.dx)
    } else {
        PhaseField::new(4, 4, 0.1)
    })
    .or_else(|_| PhaseField::new(4, 4, 0.1))
    .expect("a 4 x 4 field");
    p.nx = src.nx;
    p.ny = src.ny;
    p.dx = src.dx;
    p.phi.clone_from(&src.phi);
    p.mobility = src.mobility;
    p.epsilon = src.epsilon;
    p
}

/// A random field: φ uniform in [−1, 1) with exact ±1 and 0s, and the specials when `wild`. `new` refuses a side
/// below 4, which `step` accepts down to 3, so the field is built at least 4 × 4 and then cut to nx × ny.
fn random_pf(rng: &mut Rng, nx: usize, ny: usize, wild: bool) -> PhaseField {
    let mut p = PhaseField::new(nx.max(4), ny.max(4), rng.log_uniform(-2.0, 0.0)).expect("valid");
    p.nx = nx;
    p.ny = ny;
    p.phi.truncate(nx * ny);
    let sp = specials();
    for v in &mut p.phi {
        let r = rng.unit();
        *v = if r < 0.1 {
            -1.0
        } else if r < 0.2 {
            1.0
        } else if r < 0.25 {
            0.0
        } else if wild && r < 0.35 {
            sp[usize::try_from(rng.int(0, 9)).unwrap_or(0)]
        } else {
            rng.uniform(-1.0, 1.0)
        };
    }
    p
}

fn new_rows(m: &mut Module) {
    let mut cases: Vec<(usize, usize, f64)> = vec![(4, 4, 0.1), (16, 9, 0.05), (5, 32, 1.0)];
    for (nx, ny) in [(3, 4), (4, 3), (0, 4), (4, 0), (2, 10)] {
        cases.push((nx, ny, 0.1));
    }
    for &x in &specials() {
        cases.push((5, 4, x));
        cases.push((3, 4, x));
    }
    for (nx, ny, dx) in cases {
        let out = match PhaseField::new(nx, ny, dx) {
            Ok(p) => {
                let mut o = state(&p);
                o.extend(dump(&p));
                o
            }
            Err(x) => vec![err(&x)],
        };
        m.row(
            "PhaseField::new",
            Class::A,
            &[usz(nx), usz(ny), f(dx)],
            &out,
        );
    }
}

fn shape_rows(m: &mut Module, rng: &mut Rng) {
    let circles: Vec<[f64; 3]> = vec![
        [1.0, 1.0, 0.3],
        [0.45, 0.45, 0.2],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, -0.3],
        [f64::NAN, 0.5, 0.3],
        [0.3, 0.3, f64::INFINITY],
    ];
    for c in &circles {
        let mut p = PhaseField::new(20, 20, 0.1).expect("valid");
        let mut ins = state(&p);
        ins.extend(c.iter().map(|&x| f(x)));
        p.set_circle(c[0], c[1], c[2]);
        m.row("PhaseField::set_circle", Class::A, &ins, &state(&p));
    }
    let rects: Vec<[f64; 4]> = vec![
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.0, 1.0, 0.6],
        [0.7, 0.7, 0.7, 0.7],
        [1.0, 1.0, 0.5, 0.5],
        [-1.0, 0.25, 0.45, 2.0],
        [0.0, f64::NAN, 1.0, 1.0],
        [
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ],
    ];
    for r in &rects {
        let mut p = random_pf(rng, 12, 9, false);
        let mut ins = state(&p);
        ins.extend(r.iter().map(|&x| f(x)));
        p.set_rect(r[0], r[1], r[2], r[3]);
        m.row("PhaseField::set_rect", Class::A, &ins, &state(&p));
    }
}

const THRESHOLDS: [f64; 7] = [0.5, 0.0, 1.0, 0.1, -0.1, f64::NAN, f64::INFINITY];

fn read_rows(m: &mut Module, rng: &mut Rng) {
    let mut fields = vec![PhaseField::new(10, 10, 0.1).expect("valid")];
    for k in 0..8 {
        fields.push(random_pf(rng, 4 + k % 5, 4 + (k * 3) % 6, k % 2 == 1));
    }
    let mut p = PhaseField::new(20, 20, 0.1).expect("valid");
    p.set_circle(1.0, 1.0, 0.3);
    fields.push(p);
    for p in &fields {
        let ins = state(p);
        let n = p.nx * p.ny;
        let mut phase = vec![Tok::V(n)];
        let mut inter = vec![Tok::V(THRESHOLDS.len() * n)];
        for y in 0..p.ny {
            for x in 0..p.nx {
                phase.push(b(p.is_phase_b(x, y)));
            }
        }
        for &t in &THRESHOLDS {
            for y in 0..p.ny {
                for x in 0..p.nx {
                    inter.push(b(p.is_interface(x, y, t)));
                }
            }
        }
        let mut with_t = ins.clone();
        with_t.push(Tok::V(THRESHOLDS.len()));
        with_t.extend(THRESHOLDS.iter().map(|&t| f(t)));
        m.row("PhaseField::is_phase_b", Class::A, &ins, &phase);
        m.row("PhaseField::is_interface", Class::A, &with_t, &inter);
        m.row(
            "PhaseField::phase_b_area",
            Class::B,
            &ins,
            &[f(p.phase_b_area())],
        );
    }
}

fn step_row(m: &mut Module, p: &PhaseField, vx: &[f64], vy: &[f64], dt: f64, nsteps: usize) {
    let mut ins = state(p);
    col(&mut ins, vx);
    col(&mut ins, vy);
    ins.extend([f(dt), usz(nsteps)]);
    let mut w = fresh(p);
    let mut out = None;
    for _ in 0..nsteps {
        if let Err(x) = w.step(vx, vy, dt) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| {
        let mut o = state(&w);
        o.extend(dump(&w));
        o
    });
    m.row("PhaseField::step", Class::S, &ins, &out);
}

const FRAMES: [usize; 6] = [0, 1, 2, 10, 60, 300];

fn single_steps(m: &mut Module, p: &PhaseField, vx: &[f64], vy: &[f64], dt: f64) {
    let mut w = fresh(p);
    let mut at = 0;
    for &k in &FRAMES {
        while at < k {
            if w.step(vx, vy, dt).is_err() {
                return;
            }
            at += 1;
        }
        step_row(m, &w, vx, vy, dt, 1);
    }
}

#[allow(clippy::too_many_lines)] // one block per scene
fn step_rows(m: &mut Module, rng: &mut Rng) {
    let none: Vec<f64> = Vec::new();
    // A uniform field (Rust's test_uniform_stays_uniform), and a linear ramp that sharpens
    // (test_interface_sharpens): bit-level trajectories.
    let uniform = PhaseField::new(10, 10, 0.1).expect("valid");
    let mut ramp = PhaseField::new(40, 6, 0.1).expect("valid");
    for y in 0..6 {
        for x in 0..40 {
            ramp.phi[y * 40 + x] = (fx(x) / 20.0 - 1.0).clamp(-1.0, 1.0);
        }
    }
    for n in [1, 2, 10, 60, 300] {
        step_row(m, &uniform, &none, &none, 0.01, n);
        step_row(m, &ramp, &none, &none, 0.001, n);
    }
    // A disc relaxing; a strip carried right (test_advection_moves_interface); a step front at high Péclet with
    // no mobility (test_phase_field_upwind_advection_bounded); a disc in a rotation; a random flow.
    let mut disc = PhaseField::new(16, 16, 0.1).expect("valid");
    disc.set_circle(0.8, 0.7, 0.35);
    single_steps(m, &disc, &none, &none, 0.002);
    let mut strip = PhaseField::new(30, 6, 0.1).expect("valid");
    strip.set_rect(0.5, 0.0, 1.0, 0.6);
    let n = 30 * 6;
    single_steps(m, &strip, &vec![1.0; n], &vec![0.0; n], 0.001);
    let mut front = PhaseField::new(32, 8, 0.1).expect("valid");
    front.mobility = 0.0;
    for y in 0..8 {
        for x in 0..32 {
            front.phi[y * 32 + x] = if x < 10 { 1.0 } else { -1.0 };
        }
    }
    let n = 32 * 8;
    single_steps(m, &front, &vec![1.0; n], &vec![0.0; n], 0.05);
    let (c, n) = (7.5, 16 * 16);
    let vx: Vec<f64> = (0..n).map(|i| -(fx(i / 16) - c) * 0.2).collect();
    let vy: Vec<f64> = (0..n).map(|i| (fx(i % 16) - c) * 0.2).collect();
    single_steps(m, &disc, &vx, &vy, 0.005);
    let vx: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let vy: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    single_steps(m, &disc, &vx, &vy, 0.01);
    // Random states (the specials among them), random mobility and ε, with or without a flow; a large step that
    // clamps; a negative mobility.
    for k in 0..16 {
        let (nx, ny) = (3 + k % 6, 3 + (k * 5) % 7);
        let mut p = random_pf(rng, nx, ny, k % 2 == 0);
        p.mobility = rng.uniform(0.0, 3.0);
        p.epsilon = rng.log_uniform(-2.0, 0.0);
        let n = nx * ny;
        if k % 3 == 0 {
            step_row(m, &p, &none, &none, rng.log_uniform(-4.0, -2.0), 1);
        } else {
            let vx: Vec<f64> = (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let vy: Vec<f64> = (0..n).map(|_| rng.uniform(-2.0, 2.0)).collect();
            step_row(m, &p, &vx, &vy, rng.log_uniform(-4.0, -2.0), 1 + k % 2);
        }
    }
    let p = random_pf(rng, 8, 8, false);
    step_row(m, &p, &none, &none, 10.0, 1);
    let mut p = random_pf(rng, 8, 8, false);
    p.mobility = -5.0;
    step_row(m, &p, &none, &none, 0.01, 2);
    // The cell size, which step doesn't check.
    for dx in [0.0, f64::NAN, -0.1] {
        let mut p = disc.clone();
        p.dx = dx;
        step_row(m, &p, &none, &none, 0.001, 1);
    }
    // The refusals, in Rust's order (test_phase_field_rejects_wrong_velocity_length among them).
    let ok = PhaseField::new(8, 8, 0.1).expect("valid");
    let z = vec![0.0; 64];
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        step_row(m, &ok, &none, &none, dt, 1);
    }
    step_row(m, &ok, &[0.0; 10], &[0.0; 10], 0.001, 1);
    step_row(m, &ok, &none, &z, 0.001, 1);
    step_row(m, &ok, &z, &none, 0.001, 1);
    step_row(m, &ok, &z, &[0.0; 63], 0.001, 1);
    let mut bad = ok.clone();
    bad.phi.pop();
    step_row(m, &bad, &none, &none, 0.001, 1);
    step_row(m, &bad, &[0.0; 10], &none, 0.001, 1);
    let mut bad = ok.clone();
    bad.nx = 2;
    bad.ny = 32;
    step_row(m, &bad, &none, &none, 0.001, 1);
    let mut bad = ok.clone();
    bad.nx = 32;
    bad.ny = 2;
    step_row(m, &bad, &none, &none, 0.001, 1);
    // u = 0 everywhere with v > 0, and ±∞ in φ: the upwind choice for u = 0 shows through 0·∞.
    let mut edge = random_pf(rng, 8, 8, false);
    edge.phi[3 * 8 + 3] = f64::INFINITY;
    edge.phi[4 * 8 + 5] = f64::NEG_INFINITY;
    step_row(m, &edge, &z, &vec![0.5; 64], 0.001, 1);
    // The smallest grid step accepts: 3 × 3, one interior cell.
    let p = random_pf(rng, 4, 4, false);
    let mut p3 = p.clone();
    p3.nx = 3;
    p3.ny = 3;
    p3.phi.truncate(9);
    step_row(m, &p3, &none, &none, 0.01, 1);
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let tok = |doc: &str| match serde_json::from_str::<PhaseField>(doc) {
        Ok(p) => state(&p),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..5 {
        docs.push(serde_json::to_string(&random_pf(rng, 4 + k % 2, 4, false)).unwrap_or_default());
    }
    let base =
        serde_json::to_string(&PhaseField::new(4, 4, 0.5).expect("valid")).unwrap_or_default();
    for (from, to) in [
        (r#""nx":4"#, r#""nx":4.0"#),
        (r#""mobility":1.0"#, r#""mobility":"1""#),
        (r#""epsilon":1.0"#, r#""epsilon":null"#),
        (r#""epsilon":1.0"#, r#""epsilon":1.0,"scratch":[]"#),
        (r#","mobility":1.0"#, ""),
        (r#""phi":["#, r#""phi":[[1.0],"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push("[4,4,0.5,[-1.0,1.0],1.0,1.0]".to_string());
    docs.push("[4,4,0.5,[-1.0,1.0],1.0]".to_string());
    for doc in &docs {
        m.row(
            "PhaseField::from_json",
            Class::A,
            &[s(doc.as_str())],
            &tok(doc),
        );
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("phase_field");
    let mut rng = Rng::new(0x0_f1d);
    new_rows(&mut m);
    shape_rows(&mut m, &mut rng);
    read_rows(&mut m, &mut rng);
    step_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    m
}
