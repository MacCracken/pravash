//! `vof.tsv` — `rust-old/src/vof.rs`, `VofField` (M3 part d).
//!
//! A state is `i:` nx, `i:` ny, dx, `v:` alpha; the scratch dump is `i:` the scratch length. A row's input field is
//! made by `fresh`: `VofField::new(nx, ny, dx)`, or `new(4, 4, 0.1)` when that is refused (or the size is too large
//! to build), then given the public fields. Its scratch is therefore that constructor's, which `advect` keeps when
//! it runs no sub-step (a NaN Courant number, from a cell size of 0 or NaN); the Cyrius loader builds it the same
//! way.
//!
//! Rows:
//! - `VofField::new` takes `i:` nx, ny, dx and gives nx, ny, dx, α's length and first value and the dump, or
//!   `E:<Variant>`;
//! - the cell queries over every cell (`is_surface`, `is_full`, `is_empty`: `v:<n>` flags; the `_with_threshold`
//!   forms take `v:<k>` thresholds after the state and give `v:<k·n>`, threshold-major), `interface_normal` over
//!   every cell (`v:<2n>`), and `total_volume`;
//! - `VofField::fill_rect` (x_min, y_min, x_max, y_max) and `VofField::fill_circle` (cx, cy, radius) take a state
//!   and the shape and give the state;
//! - `VofField::advect` takes a state, `v:` vx, `v:` vy, dt and `i:` nsteps, and gives the state and the dump after
//!   nsteps calls, or `E:<Variant>` from the first that fails;
//! - `VofField::from_json` decodes a serde_json document.
//!
//! Every operation is +, −, ×, ÷, √, ceil, clamp and the saturating `as usize`, so every row is expected bit-exact.
//! The velocity fields are rational (no transcendental), as every input is.

use pravash::vof::VofField;

use crate::grids::{Rng, specials};
use crate::m_grid::{col, fx, usz};
use crate::tsv::{Class, Module, Tok, b, err, f, s};

fn state(v: &VofField) -> Vec<Tok> {
    let mut out = vec![usz(v.nx), usz(v.ny), f(v.dx)];
    col(&mut out, &v.alpha);
    out
}

fn dump(v: &VofField) -> Vec<Tok> {
    vec![usz(v.oracle_dump().scratch.len())]
}

/// The row's input field: `new(nx, ny, dx)`, or a 4 × 4 field when that is refused or too large, then the public
/// fields (module comment).
fn fresh(src: &VofField) -> VofField {
    let small = src.nx < 1 << 16 && src.ny < 1 << 16;
    let mut v = (if small {
        VofField::new(src.nx, src.ny, src.dx)
    } else {
        VofField::new(4, 4, 0.1)
    })
    .or_else(|_| VofField::new(4, 4, 0.1))
    .expect("a 4 x 4 field");
    v.nx = src.nx;
    v.ny = src.ny;
    v.dx = src.dx;
    v.alpha.clone_from(&src.alpha);
    v
}

/// A random field: α uniform in [0, 1), with a few exact 0s and 1s, and the edge values when `wild`.
fn random_vof(rng: &mut Rng, nx: usize, ny: usize, wild: bool) -> VofField {
    let mut v = VofField::new(nx, ny, rng.log_uniform(-2.0, 0.0)).expect("valid");
    let edges = [
        0.0,
        1.0,
        1e-6,
        1.0 - 1e-6,
        2e-6,
        -0.0,
        -0.1,
        1.2,
        f64::NAN,
        f64::INFINITY,
    ];
    for a in &mut v.alpha {
        let r = rng.unit();
        *a = if r < 0.15 {
            0.0
        } else if r < 0.3 {
            1.0
        } else if wild && r < 0.45 {
            edges[usize::try_from(rng.int(0, 9)).unwrap_or(0)]
        } else {
            rng.unit()
        };
    }
    v
}

/// A velocity field: 0 uniform (0.5, 0.25), 1 a solid-body rotation about the centre, 2 random in [−1, 1),
/// 3 a rotation ω = 2/(1 + r²) (rational), scaled by `scale`.
fn flow(rng: &mut Rng, nx: usize, ny: usize, kind: usize, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let (cx, cy) = ((fx(nx) - 1.0) * 0.5, (fx(ny) - 1.0) * 0.5);
    let mut vx = vec![0.0; nx * ny];
    let mut vy = vec![0.0; nx * ny];
    for i in 0..nx * ny {
        let (x, y) = (fx(i % nx) - cx, fx(i / nx) - cy);
        let (u, v) = match kind {
            0 => (0.5, 0.25),
            1 => (-y * 0.1, x * 0.1),
            2 => (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            _ => {
                let w = 2.0 / (1.0 + x * x + y * y);
                (-y * w * 0.1, x * w * 0.1)
            }
        };
        vx[i] = u * scale;
        vy[i] = v * scale;
    }
    (vx, vy)
}

fn new_rows(m: &mut Module) {
    let mut cases: Vec<(usize, usize, f64)> = vec![(4, 4, 0.1), (16, 9, 0.05), (5, 32, 1.0)];
    for (nx, ny) in [(3, 4), (4, 3), (0, 4), (4, 0), (3, 3)] {
        cases.push((nx, ny, 0.1));
    }
    for &x in &specials() {
        cases.push((5, 4, x));
        cases.push((3, 4, x));
    }
    for (nx, ny, dx) in cases {
        let out = match VofField::new(nx, ny, dx) {
            Ok(v) => {
                let mut o = vec![
                    usz(v.nx),
                    usz(v.ny),
                    f(v.dx),
                    usz(v.alpha.len()),
                    f(v.alpha[0]),
                ];
                o.extend(dump(&v));
                o
            }
            Err(x) => vec![err(&x)],
        };
        m.row("VofField::new", Class::A, &[usz(nx), usz(ny), f(dx)], &out);
    }
}

const THRESHOLDS: [f64; 8] = [1e-6, 0.0, 0.5, 0.25, -0.1, f64::NAN, 1.0, 1e-300];

fn flags(v: &VofField, q: impl Fn(usize, usize) -> bool) -> Vec<Tok> {
    let mut out = vec![Tok::V(v.nx * v.ny)];
    for y in 0..v.ny {
        for x in 0..v.nx {
            out.push(b(q(x, y)));
        }
    }
    out
}

fn flags_t(v: &VofField, q: impl Fn(usize, usize, f64) -> bool) -> Vec<Tok> {
    let mut out = vec![Tok::V(THRESHOLDS.len() * v.nx * v.ny)];
    for &t in &THRESHOLDS {
        for y in 0..v.ny {
            for x in 0..v.nx {
                out.push(b(q(x, y, t)));
            }
        }
    }
    out
}

fn read_rows(m: &mut Module, rng: &mut Rng) {
    let mut fields = vec![VofField::new(8, 8, 0.1).expect("valid")];
    for k in 0..8 {
        fields.push(random_vof(rng, 4 + k % 5, 4 + (k * 3) % 6, k % 2 == 1));
    }
    let mut v = VofField::new(20, 20, 0.1).expect("valid");
    v.fill_circle(1.0, 1.0, 0.3);
    fields.push(v);
    let mut v = VofField::new(20, 20, 0.1).expect("valid");
    for y in 0..20 {
        for x in 0..10 {
            v.alpha[y * 20 + x] = 1.0;
        }
    }
    fields.push(v);
    // Gradients near the normal's 1e-20 floor: about 1e-12 (above it) and 1e-21 (below it).
    for step in [1e-13, 1e-22] {
        let mut v = VofField::new(8, 8, 0.1).expect("valid");
        for (i, a) in v.alpha.iter_mut().enumerate() {
            *a = step * fx(i % 8);
        }
        fields.push(v);
    }
    for v in &fields {
        let ins = state(v);
        let mut with_t = ins.clone();
        with_t.push(Tok::V(THRESHOLDS.len()));
        with_t.extend(THRESHOLDS.iter().map(|&t| f(t)));
        m.row(
            "VofField::is_surface",
            Class::A,
            &ins,
            &flags(v, |x, y| v.is_surface(x, y)),
        );
        m.row(
            "VofField::is_full",
            Class::A,
            &ins,
            &flags(v, |x, y| v.is_full(x, y)),
        );
        m.row(
            "VofField::is_empty",
            Class::A,
            &ins,
            &flags(v, |x, y| v.is_empty(x, y)),
        );
        m.row(
            "VofField::is_surface_with_threshold",
            Class::A,
            &with_t,
            &flags_t(v, |x, y, t| v.is_surface_with_threshold(x, y, t)),
        );
        m.row(
            "VofField::is_full_with_threshold",
            Class::A,
            &with_t,
            &flags_t(v, |x, y, t| v.is_full_with_threshold(x, y, t)),
        );
        m.row(
            "VofField::is_empty_with_threshold",
            Class::A,
            &with_t,
            &flags_t(v, |x, y, t| v.is_empty_with_threshold(x, y, t)),
        );
        let mut nrm = vec![Tok::V(2 * v.nx * v.ny)];
        for y in 0..v.ny {
            for x in 0..v.nx {
                let (a, c) = v.interface_normal(x, y);
                nrm.extend([f(a), f(c)]);
            }
        }
        m.row("VofField::interface_normal", Class::B, &ins, &nrm);
        m.row(
            "VofField::total_volume",
            Class::B,
            &ins,
            &[f(v.total_volume())],
        );
    }
}

fn fill_rows(m: &mut Module, rng: &mut Rng) {
    let base = VofField::new(10, 10, 0.1).expect("valid");
    let rects: Vec<[f64; 4]> = vec![
        [0.2, 0.2, 0.5, 0.5],
        [0.0, 0.0, 0.9, 0.9],
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.5, 0.2, 0.2],
        [-1.0, 0.25, 0.45, 2.0],
        [f64::NAN, 0.0, 1.0, 1.0],
        [
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ],
        [0.1, 0.1, 0.1 + 3.0 * 0.1, 0.7],
    ];
    for r in &rects {
        let mut v = base.clone();
        let mut ins = state(&v);
        ins.extend(r.iter().map(|&x| f(x)));
        v.fill_rect(r[0], r[1], r[2], r[3]);
        m.row("VofField::fill_rect", Class::A, &ins, &state(&v));
    }
    let circles: Vec<[f64; 3]> = vec![
        [1.0, 1.0, 0.3],
        [0.45, 0.45, 0.2],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, -0.3],
        [0.5, f64::NAN, 0.3],
        [0.3, 0.3, f64::INFINITY],
        [0.4, 0.5, 0.1 * 2.0],
    ];
    for c in &circles {
        let mut v = random_vof(rng, 10, 10, false);
        let mut ins = state(&v);
        ins.extend(c.iter().map(|&x| f(x)));
        v.fill_circle(c[0], c[1], c[2]);
        m.row("VofField::fill_circle", Class::A, &ins, &state(&v));
    }
}

fn advect_row(m: &mut Module, v: &VofField, vx: &[f64], vy: &[f64], dt: f64, nsteps: usize) {
    let mut ins = state(v);
    col(&mut ins, vx);
    col(&mut ins, vy);
    ins.extend([f(dt), usz(nsteps)]);
    let mut w = fresh(v);
    let mut out = None;
    for _ in 0..nsteps {
        if let Err(x) = w.advect(vx, vy, dt) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| {
        let mut o = state(&w);
        o.extend(dump(&w));
        o
    });
    m.row("VofField::advect", Class::S, &ins, &out);
}

const FRAMES: [usize; 6] = [0, 1, 2, 10, 60, 300];

fn single_steps(m: &mut Module, v: &VofField, vx: &[f64], vy: &[f64], dt: f64) {
    let mut w = fresh(v);
    let mut at = 0;
    for &k in &FRAMES {
        while at < k {
            if w.advect(vx, vy, dt).is_err() {
                return;
            }
            at += 1;
        }
        advect_row(m, &w, vx, vy, dt, 1);
    }
}

fn advect_rows(m: &mut Module, rng: &mut Rng) {
    // A block in a rotation: bit-level trajectories.
    let mut block = VofField::new(12, 12, 0.1).expect("valid");
    block.fill_rect(0.3, 0.3, 0.6, 0.7);
    let (vx, vy) = flow(rng, 12, 12, 1, 1.0);
    for n in [1, 2, 10, 60, 300] {
        advect_row(m, &block, &vx, &vy, 0.05, n);
    }
    // Still fluid (Rust's test_vof_advect_stationary), a circle carried by a uniform flow, a rational vortex, a
    // random flow, and a fast one that needs the most sub-steps.
    let mut still = VofField::new(10, 10, 0.1).expect("valid");
    still.fill_rect(0.3, 0.3, 0.6, 0.6);
    let z = vec![0.0; 100];
    advect_row(m, &still, &z, &z, 0.01, 3);
    let mut disc = VofField::new(14, 14, 0.1).expect("valid");
    disc.fill_circle(0.5, 0.6, 0.3);
    let (vx, vy) = flow(rng, 14, 14, 0, 1.0);
    single_steps(m, &disc, &vx, &vy, 0.02);
    let (vx, vy) = flow(rng, 14, 14, 3, 4.0);
    single_steps(m, &disc, &vx, &vy, 0.05);
    let (vx, vy) = flow(rng, 14, 14, 2, 1.0);
    single_steps(m, &disc, &vx, &vy, 0.1);
    let (vx, vy) = flow(rng, 14, 14, 0, 1e5);
    advect_row(m, &disc, &vx, &vy, 0.01, 1);
    let (vx, vy) = flow(rng, 14, 14, 0, 30.0);
    advect_row(m, &disc, &vx, &vy, 0.01, 2);
    // Random states (the edge values among them) under random flows, one step each.
    for k in 0..16 {
        let (nx, ny) = (4 + k % 6, 4 + (k * 5) % 7);
        let v = random_vof(rng, nx, ny, k % 2 == 0);
        let scale = rng.log_uniform(-1.0, 1.0);
        let (vx, vy) = flow(rng, nx, ny, 2, scale);
        advect_row(m, &v, &vx, &vy, rng.log_uniform(-3.0, -1.0), 1);
    }
    // The cell size, which advect doesn't check: 0 (no flow: a NaN count, no sub-step; flow: 1024 of them), NaN,
    // negative.
    let (vx, vy) = flow(rng, 10, 10, 0, 1.0);
    for dx in [0.0, f64::NAN, -0.1] {
        let mut v = still.clone();
        v.dx = dx;
        advect_row(m, &v, &z, &z, 0.01, 1);
        advect_row(m, &v, &vx, &vy, 0.01, 1);
    }
    // The refusals, in Rust's order (Rust's test_vof_rejects_short_velocity among them).
    let ok = VofField::new(8, 8, 0.1).expect("valid");
    let z64 = vec![0.0; 64];
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        advect_row(m, &ok, &z64, &z64, dt, 1);
    }
    advect_row(m, &ok, &[0.0; 10], &z64, 0.01, 1);
    advect_row(m, &ok, &z64, &[0.0; 65], 0.01, 1);
    let mut nan = z64.clone();
    nan[20] = f64::NAN;
    advect_row(m, &ok, &nan, &z64, 0.01, 1);
    let mut inf = z64.clone();
    inf[63] = f64::NEG_INFINITY;
    advect_row(m, &ok, &z64, &inf, 0.01, 1);
    let mut bad = ok.clone();
    bad.nx = 3;
    advect_row(m, &bad, &z64, &z64, 0.01, 1);
    let mut bad = ok.clone();
    bad.alpha.pop();
    advect_row(m, &bad, &z64, &z64, 0.01, 1);
    advect_row(m, &bad, &nan, &z64, 0.0, 1);
    // Faces whose velocity is exactly 0 (a still flow, and neighbours of opposite velocity): the donor choice shows
    // only through 0·∞ and 0·NaN, so these fields carry the edge values.
    let wild = random_vof(rng, 9, 9, true);
    let z81 = vec![0.0; 81];
    advect_row(m, &wild, &z81, &z81, 0.01, 1);
    let alt: Vec<f64> = (0..81)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    advect_row(m, &wild, &alt, &alt, 0.01, 1);
    let mut edge = VofField::new(8, 8, 0.1).expect("valid");
    edge.alpha.fill(0.5);
    edge.alpha[3 * 8 + 3] = f64::NAN;
    edge.alpha[4 * 8 + 4] = f64::INFINITY;
    edge.alpha[2 * 8 + 5] = -0.1;
    advect_row(m, &edge, &z64, &z64, 0.01, 1);
    // A Courant number of 300: 600 sub-steps, under the 1024 cap, with α still between 0 and 1.
    let (vx, vy) = flow(rng, 14, 14, 0, 3000.0);
    advect_row(m, &disc, &vx, &vy, 0.01, 1);
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let tok = |doc: &str| match serde_json::from_str::<VofField>(doc) {
        Ok(v) => state(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..5 {
        docs.push(serde_json::to_string(&random_vof(rng, 4 + k % 2, 4, false)).unwrap_or_default());
    }
    let base = serde_json::to_string(&VofField::new(4, 4, 0.5).expect("valid")).unwrap_or_default();
    for (from, to) in [
        (r#""nx":4"#, r#""nx":4.5"#),
        (r#""ny":4"#, r#""ny":-4"#),
        (r#""dx":0.5"#, r#""dx":null"#),
        (r#""dx":0.5"#, r#""dx":0.5,"scratch":[1]"#),
        (r#","dx":0.5"#, ""),
        (r#""alpha":["#, r#""alpha":[true,"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push("[4,4,0.5,[0.0,1.0]]".to_string());
    docs.push("[4,4,0.5]".to_string());
    for doc in &docs {
        m.row(
            "VofField::from_json",
            Class::A,
            &[s(doc.as_str())],
            &tok(doc),
        );
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("vof");
    let mut rng = Rng::new(0x0_f0f);
    new_rows(&mut m);
    read_rows(&mut m, &mut rng);
    fill_rows(&mut m, &mut rng);
    advect_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    m
}
