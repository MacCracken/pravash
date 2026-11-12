//! `shallow.tsv` — `rust-old/src/shallow.rs` (M2).
//!
//! A state is 11 scalar tokens and the 5 public fields: `i:` nx, `i:` ny, dx, gravity, damping, dry_threshold,
//! breaking_threshold, breaking_dissipation, dispersion_coeff, `i:` use_green_naghdi, `i:` use_riemann, then
//! `v:<n>` height, vx, vy, ground and manning_n (each with its own length, as Rust's public Vecs may have). The
//! scratch buffers are not part of it: `step` resizes and overwrites them before reading.
//!
//! Part 1 (construction and the read-only functions, sediment, JSON):
//! - `ShallowWater::new` gives nx, ny, dx, the 8 parameters, each public field's length and first value, and the
//!   5 scratch lengths (the oracle dump), or `E:<Variant>`;
//! - `depth_at`, `surface_at`, `is_wet`, `is_breaking` give every cell's value, in index order;
//! - `add_disturbance` gives the height field; the four reductions their f64;
//! - `update_sediment` takes the state, `v:<n>` concentrations, the 6 config values and dt, and gives the ground
//!   and concentration fields;
//! - `ShallowWater::from_json` decodes a serde_json document to a state or `E:Deserialize`.
//!
//! Part 2 (the step): `ShallowWater::step` takes a state, dt and `i:` nsteps, and gives the state after nsteps
//! steps, or `E:<Variant>` from the first step that fails. Two kinds of row (port plan §5):
//! Scenes are 8 × 8 (1D: 24 × 3), so a state is a few hundred tokens.
//! - bit-level trajectories for the scenes that aren't chaotic (lakes at rest, a linear wave): from the initial
//!   state, nsteps ∈ {1, 2, 10, 60, 300};
//! - class S for the rest (dam breaks, a shoreline, a drop, the Riemann, friction, breaking and dispersion
//!   variants): one step from the oracle's own state at frames {0, 1, 2, 10, 60, 300}.

use pravash::shallow::{SedimentConfig, ShallowWater, update_sediment};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, Tok, b, err, f, i, s};

/// A usize as its 64-bit word (two's complement): usize::MAX is `i:-1`, as a Cyrius word holds it.
fn usz(x: usize) -> Tok {
    i(i64::from_ne_bytes(x.to_ne_bytes()))
}

fn col(out: &mut Vec<Tok>, v: &[f64]) {
    out.push(Tok::V(v.len()));
    out.extend(v.iter().map(|&x| f(x)));
}

pub fn state(sw: &ShallowWater) -> Vec<Tok> {
    let mut out = vec![
        usz(sw.nx),
        usz(sw.ny),
        f(sw.dx),
        f(sw.gravity),
        f(sw.damping),
        f(sw.dry_threshold),
        f(sw.breaking_threshold),
        f(sw.breaking_dissipation),
        f(sw.dispersion_coeff),
        b(sw.use_green_naghdi),
        b(sw.use_riemann),
    ];
    for v in [&sw.height, &sw.vx, &sw.vy, &sw.ground, &sw.manning_n] {
        col(&mut out, v);
    }
    out
}

/// A random wet-and-dry state on an nx × ny grid: a sloping, bumpy bed under a water level, so some cells are dry;
/// velocities, roughness, and parameters near the defaults (or exactly them).
pub fn random_state(rng: &mut Rng, nx: usize, ny: usize) -> ShallowWater {
    let dx = rng.log_uniform(-2.0, 0.0);
    let level = rng.uniform(0.2, 2.0);
    let mut sw = ShallowWater::new(nx, ny, dx, level).expect("valid grid");
    let n = nx * ny;
    let slope = rng.uniform(-0.3, 0.3);
    for k in 0..n {
        let x = f64::from(u32::try_from(k % nx).unwrap_or(0));
        sw.ground[k] = slope * x * dx + rng.uniform(-0.2, 0.2) * level;
        if rng.unit() < 0.15 {
            sw.ground[k] = level + rng.uniform(0.0, 0.1);
        }
        sw.height[k] = level + rng.uniform(-0.05, 0.05) * level;
        sw.vx[k] = rng.uniform(-1.0, 1.0);
        sw.vy[k] = rng.uniform(-1.0, 1.0);
        sw.manning_n[k] = if rng.unit() < 0.5 {
            0.0
        } else {
            rng.uniform(0.0, 0.1)
        };
    }
    if rng.unit() < 0.5 {
        sw.gravity = rng.uniform(1.0, 20.0);
        sw.damping = rng.uniform(0.99, 1.0);
        sw.dry_threshold = rng.log_uniform(-6.0, -2.0);
        sw.breaking_threshold = rng.uniform(0.0, 1.0);
        sw.breaking_dissipation = rng.uniform(0.0, 10.0);
        sw.dispersion_coeff = if rng.unit() < 0.5 {
            0.0
        } else {
            rng.uniform(0.0, 0.5)
        };
        sw.use_green_naghdi = rng.unit() < 0.5;
        sw.use_riemann = rng.unit() < 0.5;
    }
    sw
}

fn new_rows(m: &mut Module) {
    let mut cases: Vec<(usize, usize, f64, f64)> =
        vec![(3, 3, 0.1, 1.0), (32, 32, 0.1, 1.0), (4, 9, 0.05, 0.0)];
    for (nx, ny) in [
        (0, 3),
        (2, 3),
        (3, 2),
        (2, 2),
        (3, 0),
        (usize::MAX, 3),
        (3, usize::MAX),
    ] {
        cases.push((nx, ny, 0.1, 1.0));
    }
    cases.push((1 << 32, 1 << 32, 0.1, 1.0));
    cases.push((usize::MAX, usize::MAX, 0.1, 1.0));
    for &x in &specials() {
        cases.push((5, 4, x, 1.0));
        cases.push((5, 4, 0.1, x));
        cases.push((2, 4, x, x));
    }
    for (nx, ny, dx, h) in cases {
        let out = match ShallowWater::new(nx, ny, dx, h) {
            Ok(sw) => {
                let mut o = vec![
                    usz(sw.nx),
                    usz(sw.ny),
                    f(sw.dx),
                    f(sw.gravity),
                    f(sw.damping),
                    f(sw.dry_threshold),
                    f(sw.breaking_threshold),
                    f(sw.breaking_dissipation),
                    f(sw.dispersion_coeff),
                    b(sw.use_green_naghdi),
                    b(sw.use_riemann),
                ];
                for v in [&sw.height, &sw.vx, &sw.vy, &sw.ground, &sw.manning_n] {
                    o.push(usz(v.len()));
                    o.push(f(v[0]));
                }
                let d = sw.oracle_dump();
                for v in [
                    &d.scratch_vx,
                    &d.scratch_vy,
                    &d.scratch_h,
                    &d.scratch_lap,
                    &d.scratch_line,
                ] {
                    o.push(usz(v.len()));
                }
                o
            }
            Err(x) => vec![err(&x)],
        };
        m.row(
            "ShallowWater::new",
            Class::A,
            &[usz(nx), usz(ny), f(dx), f(h)],
            &out,
        );
    }
}

fn read_rows(m: &mut Module, rng: &mut Rng) {
    for k in 0..40 {
        let nx = 3 + (k % 7);
        let ny = 3 + ((k * 3) % 6);
        let mut sw = random_state(rng, nx, ny);
        if k % 10 == 9 {
            // A NaN and an infinity in the fields: the reads must not panic.
            sw.height[nx + 1] = f64::NAN;
            sw.ground[2 * nx - 2] = f64::INFINITY;
        }
        let st = state(&sw);
        let n = nx * ny;
        let cells = || (0..n).map(|c| (c % nx, c / nx));
        let mut depth = vec![Tok::V(n)];
        let mut surf = vec![Tok::V(n)];
        let mut wet = vec![Tok::V(n)];
        let mut brk = vec![Tok::V(n)];
        for (x, y) in cells() {
            depth.push(f(sw.depth_at(x, y)));
            surf.push(f(sw.surface_at(x, y)));
            wet.push(b(sw.is_wet(x, y)));
            brk.push(b(sw.is_breaking(x, y)));
        }
        m.row("ShallowWater::depth_at", Class::B, &st, &depth);
        m.row("ShallowWater::surface_at", Class::A, &st, &surf);
        m.row("ShallowWater::is_wet", Class::A, &st, &wet);
        m.row("ShallowWater::is_breaking", Class::A, &st, &brk);
        m.row(
            "ShallowWater::total_volume",
            Class::B,
            &st,
            &[f(sw.total_volume())],
        );
        m.row(
            "ShallowWater::interior_volume",
            Class::B,
            &st,
            &[f(sw.interior_volume())],
        );
        for rest in [1.0, sw.height[n / 2]] {
            let mut ins = st.clone();
            ins.push(f(rest));
            m.row(
                "ShallowWater::max_wave_height",
                Class::B,
                &ins,
                &[f(sw.max_wave_height(rest))],
            );
        }
        for cfl in [0.5, 1.0] {
            let mut ins = st.clone();
            ins.push(f(cfl));
            m.row("ShallowWater::cfl_dt", Class::B, &ins, &[f(sw.cfl_dt(cfl))]);
        }
        // add_disturbance: a drop inside the grid, one at a corner, one off it, and the radius edges.
        let w = sw.dx * f64::from(u32::try_from(nx).unwrap_or(0));
        let h = sw.dx * f64::from(u32::try_from(ny).unwrap_or(0));
        for (cx, cy, r, a) in [
            (w * 0.5, h * 0.5, w * 0.3, 0.2),
            (0.0, 0.0, sw.dx * 1.5, -0.1),
            (w * 2.0, h * 2.0, w, 0.5),
            (w * 0.5, h * 0.5, 0.0, 1.0),
            (w * 0.5, h * 0.5, -1.0, 1.0),
            (w * 0.4, h * 0.6, sw.dx, f64::NAN),
        ] {
            let mut d = sw.clone();
            d.add_disturbance(cx, cy, r, a);
            let mut ins = st.clone();
            ins.extend([f(cx), f(cy), f(r), f(a)]);
            let mut out = Vec::new();
            col(&mut out, &d.height);
            m.row("ShallowWater::add_disturbance", Class::B, &ins, &out);
        }
    }
    // A dry, still domain: cfl_dt is f64::MAX.
    let mut dry = ShallowWater::new(4, 4, 0.1, 0.0).expect("valid");
    dry.ground.iter_mut().for_each(|g| *g = 1.0);
    let mut ins = state(&dry);
    ins.push(f(0.5));
    m.row(
        "ShallowWater::cfl_dt",
        Class::B,
        &ins,
        &[f(dry.cfl_dt(0.5))],
    );
    // Short fields: the volumes' guards.
    let mut short = ShallowWater::new(4, 4, 0.1, 1.0).expect("valid");
    short.ground.truncate(10);
    let st = state(&short);
    m.row(
        "ShallowWater::total_volume",
        Class::B,
        &st,
        &[f(short.total_volume())],
    );
    m.row(
        "ShallowWater::interior_volume",
        Class::B,
        &st,
        &[f(short.interior_volume())],
    );
}

fn sediment_rows(m: &mut Module, rng: &mut Rng) {
    let d = SedimentConfig::default();
    m.row(
        "SedimentConfig::default",
        Class::A,
        &[],
        &[
            f(d.shields_critical),
            f(d.grain_diameter),
            f(d.sediment_density),
            f(d.water_density),
            f(d.erosion_rate),
            f(d.deposition_rate),
        ],
    );
    for k in 0..40 {
        let nx = 3 + (k % 6);
        let ny = 3 + ((k * 5) % 5);
        let mut sw = random_state(rng, nx, ny);
        // Fast flow over a rough bed erodes; slow flow deposits.
        let fast = k % 2 == 0;
        for v in sw.vx.iter_mut() {
            *v *= if fast { 5.0 } else { 0.1 };
        }
        let n = nx * ny;
        let mut conc: Vec<f64> = (0..n).map(|_| rng.uniform(-0.01, 0.2)).collect();
        let mut cfg = SedimentConfig::default();
        if k % 3 == 1 {
            cfg.shields_critical = rng.uniform(0.0, 0.1);
            cfg.erosion_rate = rng.uniform(0.0, 0.1);
            cfg.deposition_rate = rng.uniform(0.0, 1.0);
        }
        let dt = match k % 8 {
            5 => 0.0,
            6 => f64::NAN,
            7 => -0.01,
            _ => rng.uniform(0.001, 0.5),
        };
        if k == 4 {
            conc.truncate(n - 1);
        }
        let mut ins = state(&sw);
        col(&mut ins, &conc);
        ins.extend([
            f(cfg.shields_critical),
            f(cfg.grain_diameter),
            f(cfg.sediment_density),
            f(cfg.water_density),
            f(cfg.erosion_rate),
            f(cfg.deposition_rate),
            f(dt),
        ]);
        update_sediment(&mut sw, &mut conc, &cfg, dt);
        let mut out = Vec::new();
        col(&mut out, &sw.ground);
        col(&mut out, &conc);
        m.row("update_sediment", Class::C, &ins, &out);
    }
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let tk = |doc: &str| match serde_json::from_str::<ShallowWater>(doc) {
        Ok(v) => state(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..12 {
        let mut sw = random_state(rng, 3 + k % 4, 3 + k % 3);
        if k == 11 {
            sw.vx[0] = f64::NAN;
        }
        docs.push(serde_json::to_string(&sw).unwrap_or_default());
    }
    let base = serde_json::to_string(&ShallowWater::new(3, 3, 0.5, 1.0).expect("valid"))
        .unwrap_or_default();
    for (from, to) in [
        (r#""nx":3"#, r#""nx":3.0"#),
        (r#""nx":3"#, r#""nx":-3"#),
        (r#""use_riemann":false"#, r#""use_riemann":0"#),
        (r#""use_riemann":false"#, r#""use_riemann":true"#),
        (
            r#""damping":0.999"#,
            r#""damping":0.999,"scratch_vx":[1,2]"#,
        ),
        (r#","use_green_naghdi":false"#, ""),
        (
            r#""height":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]"#,
            r#""height":[1.0]"#,
        ),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push(
        "[3,3,0.5,[1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],9.81,0.999,\
         [0,0,0,0,0,0,0,0,0],0.0001,0.5,5,0,false,false]"
            .to_string(),
    );
    for doc in &docs {
        m.row(
            "ShallowWater::from_json",
            Class::A,
            &[s(doc.as_str())],
            &tk(doc),
        );
    }
}

fn step_row(m: &mut Module, sw: &ShallowWater, dt: f64, nsteps: usize) {
    let mut ins = state(sw);
    ins.extend([f(dt), usz(nsteps)]);
    let mut s = sw.clone();
    let mut out = None;
    for _ in 0..nsteps {
        if let Err(x) = s.step(dt) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| state(&s));
    m.row("ShallowWater::step", Class::S, &ins, &out);
}

const FRAMES: [usize; 6] = [0, 1, 2, 10, 60, 300];

/// Bit-level trajectory rows: the initial state stepped 1, 2, 10, 60 and 300 times.
fn trajectory(m: &mut Module, sw: &ShallowWater, dt: f64) {
    for n in [1, 2, 10, 60, 300] {
        step_row(m, sw, dt, n);
    }
}

/// Class S rows: one step from the oracle's state at each frame (the scene stepped that far first; a scene that
/// fails earlier stops there).
fn single_steps(m: &mut Module, sw: &ShallowWater, dt: f64) {
    let mut s = sw.clone();
    let mut at = 0;
    for &k in &FRAMES {
        while at < k {
            if s.step(dt).is_err() {
                return;
            }
            at += 1;
        }
        step_row(m, &s, dt, 1);
    }
}

fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

#[allow(clippy::too_many_lines)] // one block per scene
fn step_rows(m: &mut Module, rng: &mut Rng) {
    let n = 8;
    // A lake at rest, flat, and over a slope with a bump (well-balanced: it must stay at rest).
    let flat = ShallowWater::new(n, n, 0.1, 1.0).expect("valid");
    trajectory(m, &flat, 0.001);
    let mut slope = ShallowWater::new(n, n, 0.1, 1.0).expect("valid");
    for k in 0..n * n {
        let (x, y) = (fx(k % n), fx(k / n));
        slope.ground[k] = 0.02 * x
            + 0.1 * (1.0 - ((x - 4.0) * (x - 4.0) + (y - 3.0) * (y - 3.0)) / 16.0).max(0.0);
        slope.height[k] = 1.0;
    }
    trajectory(m, &slope, 0.001);
    let mut rslope = slope.clone();
    rslope.use_riemann = true;
    trajectory(m, &rslope, 0.002);
    // A small linear wave along x (amplitude 1e-3 of the depth).
    let mut wave = ShallowWater::new(n, n, 0.1, 1.0).expect("valid");
    for k in 0..n * n {
        let x = fx(k % n);
        wave.height[k] = 1.0 + 0.001 * (1.0 - ((x - 3.5) / 3.5) * ((x - 3.5) / 3.5));
    }
    trajectory(m, &wave, 0.001);

    // 1D dam break (24 × 3), onto a wet and onto a dry bed, both schemes.
    for (lo, riemann) in [(0.5, false), (0.5, true), (0.0, true), (0.0, false)] {
        let mut d = ShallowWater::new(24, 3, 0.05, lo).expect("valid");
        for k in 0..72 {
            if k % 24 < 12 {
                d.height[k] = 1.0;
            }
        }
        d.use_riemann = riemann;
        single_steps(m, &d, 0.002);
    }
    // 2D dam break: a raised quadrant.
    let mut q = ShallowWater::new(n, n, 0.1, 0.5).expect("valid");
    for k in 0..n * n {
        if k % n < 4 && k / n < 4 {
            q.height[k] = 1.0;
        }
    }
    single_steps(m, &q, 0.001);
    let mut qr = q.clone();
    qr.use_riemann = true;
    single_steps(m, &qr, 0.001);
    // A wet/dry shoreline: a beach rising out of the water.
    let mut beach = ShallowWater::new(n, n, 0.1, 0.4).expect("valid");
    for k in 0..n * n {
        beach.ground[k] = 0.06 * fx(k % n);
        beach.height[k] = beach.ground[k].max(0.4);
    }
    beach.add_disturbance(0.2, 0.4, 0.25, 0.05);
    single_steps(m, &beach, 0.001);
    let mut beachr = beach.clone();
    beachr.use_riemann = true;
    single_steps(m, &beachr, 0.001);
    // A drop, with friction, with breaking and with each dispersion operator.
    let mut drop = ShallowWater::new(n, n, 0.1, 1.0).expect("valid");
    drop.add_disturbance(0.4, 0.35, 0.3, 0.3);
    single_steps(m, &drop, 0.0015);
    let mut rough = drop.clone();
    for (k, v) in rough.manning_n.iter_mut().enumerate() {
        *v = 0.01 + 0.0005 * fx(k % 50);
    }
    single_steps(m, &rough, 0.002);
    let mut steep = ShallowWater::new(n, n, 0.1, 1.0).expect("valid");
    steep.add_disturbance(0.4, 0.4, 0.25, 0.8);
    steep.breaking_threshold = 0.3;
    single_steps(m, &steep, 0.001);
    for (c, gn, riemann) in [
        (1.0 / 3.0, false, false),
        (1.0 / 3.0, true, false),
        (0.2, false, true),
    ] {
        let mut d = drop.clone();
        d.dispersion_coeff = c;
        d.use_green_naghdi = gn;
        d.use_riemann = riemann;
        single_steps(m, &d, 0.001);
    }
    // Random wet-and-dry states, one step each, on both schemes and a range of dt (the damping powf varies).
    for k in 0..24 {
        let mut s = random_state(rng, 4 + k % 6, 3 + k % 5);
        s.use_riemann = k % 2 == 1;
        let dt = [0.001, 0.0005, 0.0023, 0.004][k % 4];
        step_row(m, &s, dt, 1);
    }

    // The refusals: dt, the state's lengths and parameters, and a step that diverges.
    let ok = ShallowWater::new(4, 4, 0.1, 1.0).expect("valid");
    for dt in [0.0, -0.001, f64::NAN, f64::INFINITY] {
        step_row(m, &ok, dt, 1);
    }
    let mut bad: Vec<ShallowWater> = Vec::new();
    let mut s = ok.clone();
    s.vx.truncate(15);
    bad.push(s);
    let mut s = ok.clone();
    s.manning_n.push(0.0);
    bad.push(s);
    let mut s = ok.clone();
    s.nx = 2;
    bad.push(s);
    for (field, x) in [
        (0, 0.0),
        (0, f64::NAN),
        (1, -9.81),
        (1, f64::INFINITY),
        (2, -0.1),
        (2, f64::NAN),
    ] {
        let mut s = ok.clone();
        match field {
            0 => s.dx = x,
            1 => s.gravity = x,
            _ => s.dispersion_coeff = x,
        }
        bad.push(s);
    }
    let mut s = ok.clone();
    s.height[5] = 1e300;
    s.vx[6] = 1e300;
    bad.push(s);
    for s in &bad {
        step_row(m, s, 0.001, 1);
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("shallow");
    let mut rng = Rng::new(0x5_4a11);
    new_rows(&mut m);
    read_rows(&mut m, &mut rng);
    sediment_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    step_rows(&mut m, &mut rng);
    m
}
