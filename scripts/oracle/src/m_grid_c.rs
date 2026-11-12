//! `grid.tsv`, part c: `rust-old/src/grid.rs`'s `KEpsilon`, `apply_ghost_fluid` and `MacGrid` (M3 part c). The
//! rows follow part a's and b's in the same module, from their own seed, so theirs keep their values.
//!
//! A k-ε state is `v:` k, `v:` epsilon, c_mu, c_eps1, c_eps2; its scratch dump is `i:` scratch_k's and
//! scratch_eps's lengths. A MAC state is `i:` nx, `i:` ny, dx, then `v:` vx, vy, pressure and density; its scratch
//! dump is `i:` scratch_div's and scratch_density's lengths. A config is part a's 13 tokens.
//!
//! Rows:
//! - `KEpsilon::new` takes `i:` nx, ny and gives the state and the dump;
//! - `KEpsilon::turbulent_viscosity` takes a state and gives `v:<n>`, ν_t at every cell of k;
//! - `KEpsilon::step` takes a state, `v:` vx, `v:` vy, nx, ny, dx, dt and `i:` nsteps, and gives the state and the
//!   dump after nsteps steps (a refused step changes nothing: Rust warns and returns). The input model is made
//!   by `KEpsilon::new(k.len(), 1)`, then given the public fields, so its scratch is k's length;
//! - `apply_ghost_fluid` takes `v:` pressure, `v:` level set, nx, ny, dx, σ, and gives the pressure;
//! - `MacGrid::new` gives nx, ny, dx, each field's length and first value, and the dump, or `E:<Variant>`;
//! - the reads (`velocity_at` and `divergence_at` over every cell, `max_speed`, `cfl_dt` with `v:<k>` CFL factors
//!   after the state) take a state;
//! - `MacGrid::step` takes a state, a config and `i:` nsteps, and gives the state and the dump after nsteps steps,
//!   or `E:<Variant>` from the first that fails. The input grid is fresh (`MacGrid::new`, then the public fields),
//!   as part a's are;
//! - `MacGrid::from_json` and `KEpsilon::from_json` decode a serde_json document.
//!
//! Every operation is +, −, ×, ÷ or √ (and the bilinear sampler's floor), so every row is expected bit-exact.

use pravash::grid::{BoundaryCondition, GridConfig, KEpsilon, MacGrid, apply_ghost_fluid};

use crate::grids::{Rng, specials};
use crate::m_grid::{col, config, fx, usz};
use crate::tsv::{Class, Module, Tok, err, f, s};

fn ke_state(m: &KEpsilon) -> Vec<Tok> {
    let mut out = Vec::new();
    col(&mut out, &m.k);
    col(&mut out, &m.epsilon);
    out.extend([f(m.c_mu), f(m.c_eps1), f(m.c_eps2)]);
    out
}

fn ke_dump(m: &KEpsilon) -> Vec<Tok> {
    let d = m.oracle_dump();
    vec![usz(d.scratch_k.len()), usz(d.scratch_eps.len())]
}

/// A model with `src`'s public fields, made as the Cyrius loader makes it: `new(k.len(), 1)`, so the scratch is
/// k's length.
fn ke_fresh(src: &KEpsilon) -> KEpsilon {
    let mut m = KEpsilon::new(src.k.len(), 1);
    m.k.clone_from(&src.k);
    m.epsilon.clone_from(&src.epsilon);
    m.c_mu = src.c_mu;
    m.c_eps1 = src.c_eps1;
    m.c_eps2 = src.c_eps2;
    m
}

/// A random model: k and ε log-uniform over many decades (ε down past the 1e-20 floor), with `specials` sprinkled
/// in when `wild`.
fn random_ke(rng: &mut Rng, n: usize, wild: bool) -> KEpsilon {
    let mut m = KEpsilon::new(n, 1);
    let sp = specials();
    let pick = |rng: &mut Rng| sp[usize::try_from(rng.int(0, 9)).unwrap_or(0)];
    for i in 0..n {
        m.k[i] = rng.log_uniform(-12.0, 0.0);
        m.epsilon[i] = rng.log_uniform(-24.0, 0.0);
        if wild && rng.unit() < 0.15 {
            m.k[i] = pick(rng);
        }
        if wild && rng.unit() < 0.15 {
            m.epsilon[i] = pick(rng);
        }
    }
    m
}

/// A velocity field on an nx × ny grid: 0 a shear u = 0.3·y, 1 a solid-body rotation, 2 random in [−2, 2).
fn flow(rng: &mut Rng, nx: usize, ny: usize, kind: usize) -> (Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    let mut vx = vec![0.0; n];
    let mut vy = vec![0.0; n];
    let (cx, cy) = ((fx(nx) - 1.0) * 0.5, (fx(ny) - 1.0) * 0.5);
    for i in 0..n {
        let (x, y) = (fx(i % nx), fx(i / nx));
        match kind {
            0 => vx[i] = 0.3 * y,
            1 => {
                vx[i] = -(y - cy) * 0.5;
                vy[i] = (x - cx) * 0.5;
            }
            _ => {
                vx[i] = rng.uniform(-2.0, 2.0);
                vy[i] = rng.uniform(-2.0, 2.0);
            }
        }
    }
    (vx, vy)
}

#[allow(clippy::too_many_arguments)] // the step's own arguments, then the step count
fn ke_step_row(
    m: &mut Module,
    ke: &KEpsilon,
    vx: &[f64],
    vy: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dt: f64,
    nsteps: usize,
) {
    let mut ins = ke_state(ke);
    col(&mut ins, vx);
    col(&mut ins, vy);
    ins.extend([usz(nx), usz(ny), f(dx), f(dt), usz(nsteps)]);
    let mut s = ke_fresh(ke);
    for _ in 0..nsteps {
        s.step(vx, vy, nx, ny, dx, dt);
    }
    let mut out = ke_state(&s);
    out.extend(ke_dump(&s));
    m.row("KEpsilon::step", Class::S, &ins, &out);
}

fn ke_rows(m: &mut Module, rng: &mut Rng) {
    for (nx, ny) in [(0, 0), (0, 7), (1, 1), (3, 3), (8, 8), (16, 4), (5, 9)] {
        let ke = KEpsilon::new(nx, ny);
        let mut out = ke_state(&ke);
        out.extend(ke_dump(&ke));
        m.row("KEpsilon::new", Class::A, &[usz(nx), usz(ny)], &out);
    }

    let mut states = vec![KEpsilon::new(8, 8)];
    for k in 0..10 {
        states.push(random_ke(rng, 9 + 7 * k, k % 2 == 1));
    }
    let mut c = random_ke(rng, 30, false);
    c.c_mu = 0.2;
    c.c_eps1 = 1.0;
    c.c_eps2 = 2.5;
    states.push(c);
    for ke in &states {
        let out: Vec<Tok> = std::iter::once(Tok::V(ke.k.len()))
            .chain((0..ke.k.len()).map(|i| f(ke.turbulent_viscosity(i))))
            .collect();
        m.row(
            "KEpsilon::turbulent_viscosity",
            Class::B,
            &ke_state(ke),
            &out,
        );
    }

    // Trajectories from the default state under each flow.
    for (n, kind, dx, dt) in [
        (8, 0, 0.1, 0.01),
        (10, 1, 0.05, 0.005),
        (12, 2, 0.1, 0.02),
        (6, 2, 1.0, 0.1),
    ] {
        let (vx, vy) = flow(rng, n, n, kind);
        let ke = KEpsilon::new(n, n);
        for steps in [1, 2, 10, 50] {
            ke_step_row(m, &ke, &vx, &vy, n, n, dx, dt, steps);
        }
    }
    // Random models (specials among them), a non-square grid, the constants changed, one or three steps.
    for k in 0..12 {
        let (nx, ny) = (3 + k % 6, 3 + (k * 5) % 7);
        let (vx, vy) = flow(rng, nx, ny, 2);
        let ke = random_ke(rng, nx * ny, k % 3 != 0);
        let dt = rng.log_uniform(-4.0, -1.0);
        ke_step_row(
            m,
            &ke,
            &vx,
            &vy,
            nx,
            ny,
            rng.log_uniform(-2.0, 0.0),
            dt,
            1 + 2 * (k % 2),
        );
    }
    let (vx, vy) = flow(rng, 9, 9, 2);
    ke_step_row(
        m,
        &states[11].clone(),
        &vx[..30],
        &vy[..30],
        10,
        3,
        0.1,
        0.01,
        2,
    );
    // A large step: k and ε hit their 1e-10 floors.
    let (vx, vy) = flow(rng, 7, 7, 2);
    ke_step_row(m, &random_ke(rng, 49, false), &vx, &vy, 7, 7, 0.1, 50.0, 1);

    // The refusals: nothing changes. The shapes of Rust's test_kepsilon_size_mismatch_no_panic (the second runs:
    // 16 × 4 is 64 cells, k's length).
    let ke = random_ke(rng, 64, false);
    let z10 = [0.0; 10];
    let (vx, vy) = flow(rng, 8, 8, 2);
    ke_step_row(m, &ke, &z10, &z10, 8, 8, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy, 16, 4, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy[..63], 8, 8, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy, 2, 32, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy, 32, 2, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy, usize::MAX, 3, 0.1, 0.01, 1);
    ke_step_row(m, &ke, &vx, &vy, 1 << 33, 1 << 31, 0.1, 0.01, 1);
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        ke_step_row(m, &ke, &vx, &vy, 8, 8, bad, 0.01, 1);
        ke_step_row(m, &ke, &vx, &vy, 8, 8, 0.1, bad, 1);
    }
    let mut short = ke.clone();
    short.epsilon.truncate(60);
    ke_step_row(m, &short, &vx, &vy, 8, 8, 0.1, 0.01, 1);
}

fn ghost_row(m: &mut Module, p0: &[f64], ls: &[f64], nx: usize, ny: usize, dx: f64, sigma: f64) {
    let mut ins = Vec::new();
    col(&mut ins, p0);
    col(&mut ins, ls);
    ins.extend([usz(nx), usz(ny), f(dx), f(sigma)]);
    let mut p = p0.to_vec();
    apply_ghost_fluid(&mut p, ls, nx, ny, dx, sigma);
    let mut out = Vec::new();
    col(&mut out, &p);
    m.row("apply_ghost_fluid", Class::B, &ins, &out);
}

/// A circle's signed distance (negative inside): √((x − cx)² + (y − cy)²)·dx − r.
fn circle(nx: usize, ny: usize, dx: f64, cx: f64, cy: f64, r: f64) -> Vec<f64> {
    (0..nx * ny)
        .map(|i| {
            let (x, y) = (fx(i % nx) - cx, fx(i / nx) - cy);
            (x * x + y * y).sqrt() * dx - r
        })
        .collect()
}

fn ghost_rows(m: &mut Module, rng: &mut Rng) {
    for (n, dx, r, sigma) in [
        (16, 0.1, 0.5, 0.07),
        (16, 0.05, 0.3, 0.0728),
        (20, 1.0, 6.0, 1.0),
        (12, 0.1, 0.35, -0.5),
        (9, 0.2, 0.0, 0.07),
    ] {
        let c = (fx(n) - 1.0) * 0.5;
        let ls = circle(n, n, dx, c + 0.3, c - 0.2, r);
        let p: Vec<f64> = (0..n * n).map(|_| rng.uniform(-1.0, 1.0)).collect();
        ghost_row(m, &p, &ls, n, n, dx, sigma);
        ghost_row(m, &vec![0.0; n * n], &ls, n, n, dx, sigma);
    }
    // A plane (no curvature), a non-square circle, random fields, zeros (a product of 0 is no crossing), the
    // specials (NaN never crosses; ±∞ crosses and spreads), σ zero, NaN and ∞.
    let plane: Vec<f64> = (0..10 * 7).map(|i| (fx(i % 10) - 4.5) * 0.1).collect();
    ghost_row(m, &vec![1.0; 70], &plane, 10, 7, 0.1, 0.07);
    let ls = circle(11, 17, 0.1, 5.0, 8.0, 0.4);
    ghost_row(m, &vec![0.25; 187], &ls, 11, 17, 0.1, 0.07);
    for k in 0..6 {
        let (nx, ny) = (3 + k * 2, 4 + k);
        let ls: Vec<f64> = (0..nx * ny).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let p: Vec<f64> = (0..nx * ny).map(|_| rng.uniform(-5.0, 5.0)).collect();
        ghost_row(
            m,
            &p,
            &ls,
            nx,
            ny,
            rng.log_uniform(-2.0, 0.0),
            rng.uniform(-1.0, 1.0),
        );
    }
    let mut ls = circle(10, 10, 0.1, 4.5, 4.5, 0.25);
    for (&i, &x) in [3usize, 33, 44, 45, 55, 66]
        .iter()
        .zip([0.0, -0.0, f64::NAN, 0.0, f64::INFINITY, -1e300].iter())
    {
        ls[i] = x;
    }
    ghost_row(m, &vec![0.5; 100], &ls, 10, 10, 0.1, 0.07);
    let ls = circle(8, 8, 0.1, 3.5, 3.5, 0.2);
    for sigma in [0.0, f64::NAN, f64::INFINITY] {
        ghost_row(m, &vec![0.5; 64], &ls, 8, 8, 0.1, sigma);
    }
    // The refusals (the pressure unchanged): Rust's test_ghost_fluid_size_mismatch_no_panic, the sizes, dx.
    ghost_row(m, &[0.0; 10], &[0.0; 64], 8, 8, 0.1, 0.07);
    ghost_row(m, &vec![0.5; 64], &ls[..63], 8, 8, 0.1, 0.07);
    ghost_row(m, &vec![0.5; 64], &ls, 2, 32, 0.1, 0.07);
    ghost_row(m, &vec![0.5; 64], &ls, 32, 2, 0.1, 0.07);
    ghost_row(m, &vec![0.5; 64], &ls, usize::MAX, 3, 0.1, 0.07);
    for dx in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        ghost_row(m, &vec![0.5; 64], &ls, 8, 8, dx, 0.07);
    }
}

fn mac_state(g: &MacGrid) -> Vec<Tok> {
    let mut out = vec![usz(g.nx), usz(g.ny), f(g.dx)];
    for v in [&g.vx, &g.vy, &g.pressure, &g.density] {
        col(&mut out, v);
    }
    out
}

fn mac_dump(g: &MacGrid) -> Vec<Tok> {
    let d = g.oracle_dump();
    vec![usz(d.scratch_div.len()), usz(d.scratch_density.len())]
}

/// A fresh grid with `src`'s public fields (a size `new` refuses, or one whose products would overflow, which
/// `new` doesn't check, gets a 4 × 4 grid with the fields set over it).
fn mac_fresh(src: &MacGrid) -> MacGrid {
    let small = src.nx < 1 << 16 && src.ny < 1 << 16;
    let mut g = (if small {
        MacGrid::new(src.nx, src.ny, src.dx)
    } else {
        MacGrid::new(4, 4, 0.1)
    })
    .or_else(|_| MacGrid::new(4, 4, 0.1))
    .expect("a 4 x 4 grid");
    g.nx = src.nx;
    g.ny = src.ny;
    g.dx = src.dx;
    g.vx.clone_from(&src.vx);
    g.vy.clone_from(&src.vy);
    g.pressure.clone_from(&src.pressure);
    g.density.clone_from(&src.density);
    g
}

/// A random MAC state: face velocities in [−2, 2), pressure around 0, density in [0, 2).
fn random_mac(rng: &mut Rng, nx: usize, ny: usize) -> MacGrid {
    let mut g = MacGrid::new(nx, ny, rng.log_uniform(-2.0, 0.0)).expect("valid grid");
    for v in g.vx.iter_mut().chain(g.vy.iter_mut()) {
        *v = rng.uniform(-2.0, 2.0);
    }
    for v in &mut g.pressure {
        *v = rng.uniform(-0.5, 0.5);
    }
    for v in &mut g.density {
        *v = rng.uniform(0.0, 2.0);
    }
    g
}

fn mac_new_rows(m: &mut Module) {
    let mut cases: Vec<(usize, usize, f64)> = vec![(4, 4, 0.1), (16, 9, 0.05), (5, 32, 1.0)];
    for (nx, ny) in [(3, 4), (4, 3), (0, 4), (4, 0), (3, 3)] {
        cases.push((nx, ny, 0.1));
    }
    for &x in &specials() {
        cases.push((5, 4, x));
        cases.push((3, 4, x));
    }
    for (nx, ny, dx) in cases {
        let out = match MacGrid::new(nx, ny, dx) {
            Ok(g) => {
                let mut o = vec![usz(g.nx), usz(g.ny), f(g.dx)];
                for v in [&g.vx, &g.vy, &g.pressure, &g.density] {
                    o.push(usz(v.len()));
                    o.push(f(v[0]));
                }
                o.extend(mac_dump(&g));
                o
            }
            Err(x) => vec![err(&x)],
        };
        m.row("MacGrid::new", Class::A, &[usz(nx), usz(ny), f(dx)], &out);
    }
}

fn mac_read_rows(m: &mut Module, rng: &mut Rng) {
    let mut grids = vec![MacGrid::new(8, 8, 0.1).expect("valid")];
    for k in 0..10 {
        grids.push(random_mac(rng, 4 + k % 5, 4 + (k * 3) % 7));
    }
    let mut g = random_mac(rng, 6, 5);
    g.vx[3] = f64::NAN;
    g.vy[8] = f64::INFINITY;
    g.vx[20] = -0.0;
    grids.push(g);
    let mut g = MacGrid::new(5, 5, 0.1).expect("valid");
    g.vx[7] = 1e-25;
    grids.push(g);
    for g in &grids {
        let ins = mac_state(g);
        let (nx, ny) = (g.nx, g.ny);
        let mut vel = vec![Tok::V(2 * nx * ny)];
        let mut div = vec![Tok::V(nx * ny)];
        for y in 0..ny {
            for x in 0..nx {
                let (u, v) = g.velocity_at(x, y);
                vel.extend([f(u), f(v)]);
                div.push(f(g.divergence_at(x, y)));
            }
        }
        m.row("MacGrid::velocity_at", Class::B, &ins, &vel);
        m.row("MacGrid::divergence_at", Class::B, &ins, &div);
        m.row("MacGrid::max_speed", Class::B, &ins, &[f(g.max_speed())]);
        let cfls = [0.5, 1.0, 0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300, 2.0];
        let mut row = ins.clone();
        row.push(Tok::V(cfls.len()));
        row.extend(cfls.iter().map(|&c| f(c)));
        let out: Vec<Tok> = std::iter::once(Tok::V(cfls.len()))
            .chain(cfls.iter().map(|&c| f(g.cfl_dt(c))))
            .collect();
        m.row("MacGrid::cfl_dt", Class::B, &row, &out);
    }
}

fn mac_step_row(m: &mut Module, g: &MacGrid, c: &GridConfig, nsteps: usize) {
    let mut ins = mac_state(g);
    ins.extend(config(c));
    ins.push(usz(nsteps));
    let mut s = mac_fresh(g);
    let mut out = None;
    for _ in 0..nsteps {
        if let Err(x) = s.step(c) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| {
        let mut o = mac_state(&s);
        o.extend(mac_dump(&s));
        o
    });
    m.row("MacGrid::step", Class::S, &ins, &out);
}

const FRAMES: [usize; 6] = [0, 1, 2, 10, 60, 300];

/// Class S rows: one step from the oracle's state at each frame (a scene that fails earlier stops there).
fn mac_single_steps(m: &mut Module, g: &MacGrid, c: &GridConfig) {
    let mut s = mac_fresh(g);
    let mut at = 0;
    for &k in &FRAMES {
        while at < k {
            if s.step(c).is_err() {
                return;
            }
            at += 1;
        }
        mac_step_row(m, &s, c, 1);
    }
}

#[allow(clippy::too_many_lines)] // one block per scene
fn mac_step_rows(m: &mut Module, rng: &mut Rng) {
    let smoke = GridConfig::smoke();
    let mut calm = smoke;
    calm.buoyancy_alpha = 0.0;

    // Still tanks: empty, and uniform density (Rust's test_mac_buoyancy_does_not_accumulate_on_walls).
    let tank = MacGrid::new(10, 10, 0.1).expect("valid");
    let mut full = MacGrid::new(8, 8, 0.1).expect("valid");
    full.density.fill(1.0);
    for n in [1, 2, 10, 60, 300] {
        mac_step_row(m, &tank, &smoke, n);
        mac_step_row(m, &full, &smoke, n);
    }

    // A buoyant blob near the floor, under a few configs.
    let mut blob = MacGrid::new(10, 10, 0.1).expect("valid");
    for x in 3..7 {
        blob.density[10 + x] = 1.0;
        blob.density[20 + x] = 1.0;
    }
    let mut variants = vec![smoke];
    let mut c = smoke;
    c.dt = 0.02;
    c.pressure_iterations = 80;
    variants.push(c);
    let mut c = smoke;
    c.buoyancy_alpha = -0.3;
    c.ambient_density = 0.4;
    c.pressure_iterations = 7;
    variants.push(c);
    let mut c = smoke;
    // MacGrid reads only dt, pressure_iterations and the buoyancy pair: the rest is ignored.
    c.viscosity = -1.0;
    c.boundary = BoundaryCondition::Periodic;
    c.use_cg = true;
    c.diffusion_iterations = 0;
    c.vorticity_confinement = f64::NAN;
    variants.push(c);
    for c in &variants {
        mac_single_steps(m, &blob, c);
    }

    // The projection on arbitrary face velocities (Rust's test_mac_projection_divergence_free, fewer sweeps), and
    // a density carried by a uniform flow (test_mac_advects_density).
    let mut proj = MacGrid::new(16, 16, 0.1).expect("valid");
    for (k, v) in proj.vx.iter_mut().enumerate() {
        *v = fx((k * 13) % 7) - 3.0;
    }
    for (k, v) in proj.vy.iter_mut().enumerate() {
        *v = fx((k * 5) % 11) - 5.0;
    }
    let mut c = calm;
    c.pressure_iterations = 400;
    mac_step_row(m, &proj, &c, 1);
    let mut carry = MacGrid::new(16, 16, 0.1).expect("valid");
    carry.density[8 * 16 + 4] = 1.0;
    for y in 0..16 {
        for x in 1..16 {
            carry.vx[y * 17 + x] = 0.5;
        }
    }
    mac_single_steps(m, &carry, &calm);

    // Random states, one step each, on random configs.
    for k in 0..24 {
        let g = random_mac(rng, 4 + k % 7, 4 + (k * 3) % 6);
        let mut c = smoke;
        c.dt = [0.1, 0.05, 0.02, 0.01][k % 4];
        c.pressure_iterations = [1, 5, 20, 40][(k / 2) % 4];
        c.buoyancy_alpha = if rng.unit() < 0.3 {
            0.0
        } else {
            rng.uniform(-0.5, 0.5)
        };
        c.ambient_density = if rng.unit() < 0.5 {
            0.0
        } else {
            rng.uniform(0.0, 1.0)
        };
        mac_step_row(m, &g, &c, 1);
    }

    // The refusals, in Rust's order: the config, then the state; and steps that diverge.
    let ok = MacGrid::new(6, 6, 0.1).expect("valid");
    let mut bad_cfg: Vec<GridConfig> = Vec::new();
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut c = smoke;
        c.dt = dt;
        bad_cfg.push(c);
    }
    let mut c = smoke;
    c.pressure_iterations = 0;
    bad_cfg.push(c);
    for (alpha, amb) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (0.0, f64::NAN),
        (0.1, f64::NEG_INFINITY),
    ] {
        let mut c = smoke;
        c.buoyancy_alpha = alpha;
        c.ambient_density = amb;
        bad_cfg.push(c);
    }
    let mut c = smoke;
    c.pressure_iterations = 0;
    c.dt = 0.0;
    bad_cfg.push(c);
    for c in &bad_cfg {
        mac_step_row(m, &ok, c, 1);
    }
    let mut bad: Vec<MacGrid> = Vec::new();
    let mut g = ok.clone();
    g.nx = 3;
    bad.push(g);
    let mut g = ok.clone();
    g.ny = usize::MAX;
    bad.push(g);
    let mut g = ok.clone();
    g.vx.truncate(41);
    bad.push(g);
    let mut g = ok.clone();
    g.vy.push(0.0);
    bad.push(g);
    let mut g = ok.clone();
    g.density.clear();
    bad.push(g);
    let mut g = ok.clone();
    g.pressure.push(0.0);
    bad.push(g);
    for x in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut g = ok.clone();
        g.dx = x;
        bad.push(g);
    }
    let mut g = ok.clone();
    g.vx[10] = 1.7e308;
    g.vx[11] = -1.7e308;
    bad.push(g);
    let mut g = ok.clone();
    g.density[14] = f64::NAN;
    bad.push(g);
    let mut g = ok.clone();
    g.vy[20] = f64::INFINITY;
    bad.push(g);
    let mut g = ok.clone();
    g.density[20] = 1e308;
    bad.push(g);
    for g in &bad {
        mac_step_row(m, g, &smoke, 1);
    }
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let mac_tok = |doc: &str| match serde_json::from_str::<MacGrid>(doc) {
        Ok(v) => mac_state(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..6 {
        let mut g = random_mac(rng, 4 + k % 2, 4);
        if k == 5 {
            g.vy[0] = f64::NAN;
        }
        docs.push(serde_json::to_string(&g).unwrap_or_default());
    }
    let base = serde_json::to_string(&MacGrid::new(4, 4, 0.5).expect("valid")).unwrap_or_default();
    let z16 = "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]";
    let z20 = "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]";
    let vx_full = format!(r#""vx":{z20}"#);
    for (from, to) in [
        (r#""nx":4"#, r#""nx":4.0"#),
        (r#""ny":4"#, r#""ny":-1"#),
        (r#""dx":0.5"#, r#""dx":true"#),
        (r#""dx":0.5"#, r#""dx":0.5,"scratch_div":[1,2]"#),
        (r#","dx":0.5"#, ""),
        (vx_full.as_str(), r#""vx":[1.0]"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push(format!("[4,4,0.5,{z20},{z20},{z16},{z16}]"));
    docs.push(format!("[4,4,0.5,{z20},{z20},{z16}]"));
    for doc in &docs {
        m.row(
            "MacGrid::from_json",
            Class::A,
            &[s(doc.as_str())],
            &mac_tok(doc),
        );
    }

    let ke_tok = |doc: &str| match serde_json::from_str::<KEpsilon>(doc) {
        Ok(v) => ke_state(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..5 {
        docs.push(serde_json::to_string(&random_ke(rng, 4 + k, false)).unwrap_or_default());
    }
    let base = serde_json::to_string(&KEpsilon::new(2, 2)).unwrap_or_default();
    for (from, to) in [
        (r#""c_mu":0.09"#, r#""c_mu":9e-2"#),
        (r#""c_mu":0.09"#, r#""c_mu":"0.09""#),
        (r#""c_eps2":1.92"#, r#""c_eps2":null"#),
        (r#","c_eps1":1.44"#, ""),
        (r#""c_eps2":1.92"#, r#""c_eps2":1.92,"scratch_k":[]"#),
        (r#""epsilon":[0.00001,0.00001,0.00001,0.00001]"#, r#""epsilon":[]"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push("[[1.0,2.0],[3.0],0.09,1.44,1.92]".to_string());
    docs.push("[[1.0,2.0],[3.0],0.09,1.44]".to_string());
    docs.push("{}".to_string());
    for doc in &docs {
        m.row(
            "KEpsilon::from_json",
            Class::A,
            &[s(doc.as_str())],
            &ke_tok(doc),
        );
    }
}

/// Part c's rows, appended to `m` (the grid module), from their own seed.
pub fn rows(m: &mut Module) {
    let mut rng = Rng::new(0x6_73c);
    ke_rows(m, &mut rng);
    ghost_rows(m, &mut rng);
    mac_new_rows(m);
    mac_read_rows(m, &mut rng);
    mac_step_rows(m, &mut rng);
    json_rows(m, &mut rng);
}
