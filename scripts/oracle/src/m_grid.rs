//! `grid.tsv` — `rust-old/src/grid.rs`, `FluidGrid` (M3 parts a and b). Part c's rows (`KEpsilon`,
//! `apply_ghost_fluid`, `MacGrid`) are in `m_grid_c.rs`, appended to the same module.
//!
//! A state is `i:` nx, `i:` ny, dx, then `v:<n>` vx, vy, pressure and density (each with its own length, as Rust's
//! public Vecs may have). The scratch buffers are not part of it: `step` resizes them and writes each before reading
//! it, so a row's input grid is always made fresh (`FluidGrid::new`, then the public fields), and the Cyrius loader
//! starts with empty scratch. A config is dt, viscosity, `i:` diffusion_iterations, `i:` pressure_iterations,
//! `e:BoundaryCondition.<Variant>`, vorticity_confinement, buoyancy_alpha, ambient_density, `i:` use_maccormack,
//! `i:` use_bfecc, smagorinsky_cs, `i:` use_multigrid, `i:` use_cg. The scratch dump is the four scratch lengths,
//! `i:` the level count, then each level's three lengths (finest first).
//!
//! Rows:
//! - `FluidGrid::new` gives nx, ny, dx, each public field's length and first value, and the scratch dump, or
//!   `E:<Variant>`;
//! - `GridConfig::smoke` and `GridConfig::default` give the config;
//! - the reads (`cell_count`, `idx`, `velocity_at`, `speed_at` over every cell, `max_speed`, `cfl_dt`,
//!   `total_kinetic_energy`) take a state; `cfl_dt` takes `v:<k>` viscosities and `v:<k>` CFL factors after it, and
//!   gives `v:<k>` time steps;
//! - `FluidGrid::diffuse` takes `v:<n>` field, nx, ny, diff_rate, dt, dx, iterations, and gives the field;
//! - `oracle::pressure_solve_gauss_seidel` takes `v:` pressure, `v:` div, nx, ny, iterations and gives the pressure;
//!   `oracle::pressure_solve_multigrid` takes `v:` pressure, `v:` div, nx, ny, nu_smooth, cycles and gives the
//!   pressure, `i:` the level count and each level's residual, coarse_rhs and coarse_p;
//!   `oracle::pressure_solve_dst` takes `v:` pressure, `v:` div, nx, ny and gives the pressure (class C: its sines
//!   and cosines are glibc's, and the corrected oracle rounds them correctly); `oracle::pressure_solve_cg` takes
//!   `v:` pressure, `v:` div, nx, ny, iterations and gives the pressure, or `E:Diverged`. The systems are shared by
//!   every pressure solver; part b adds DST systems at the transform lengths where glibc misrounds a sine;
//! - `FluidGrid::from_json`, `GridConfig::from_json`, `BoundaryCondition::from_json` decode a serde_json document;
//! - `FluidGrid::step` takes a state, a config and `i:` nsteps, and gives the state and the scratch dump after
//!   nsteps steps, or `E:<Variant>` from the first that fails. Part a's rows take the periodic (Gauss–Seidel) and
//!   multigrid projections, part b's (after part a's, from their own seed) the DST and CG ones. Bit-level
//!   trajectories (nsteps ∈ {1, 2, 10, 60, 300}) for the still tanks, class S single steps at frames {0, 1, 2, 10,
//!   60, 300} for the rest.
//!
//! Scenes are 10 × 10 (16 × 16 for the multigrid levels, 14 × 14 where the DST's transform length is 12, the first
//! at which glibc misrounds one of its sines), so a state is a few hundred tokens. Inputs use only exact or
//! correctly rounded operations (grids.rs).

use pravash::grid::{BoundaryCondition, FluidGrid, GridConfig};
use pravash::oracle::{
    pressure_solve_cg, pressure_solve_dst, pressure_solve_gauss_seidel, pressure_solve_multigrid,
};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, Tok, b, e, err, f, i, s};

/// A usize as its 64-bit word (two's complement): usize::MAX is `i:-1`, as a Cyrius word holds it.
pub fn usz(x: usize) -> Tok {
    i(i64::from_ne_bytes(x.to_ne_bytes()))
}

pub fn col(out: &mut Vec<Tok>, v: &[f64]) {
    out.push(Tok::V(v.len()));
    out.extend(v.iter().map(|&x| f(x)));
}

pub fn fx(x: usize) -> f64 {
    f64::from(u32::try_from(x).unwrap_or(0))
}

pub fn state(g: &FluidGrid) -> Vec<Tok> {
    let mut out = vec![usz(g.nx), usz(g.ny), f(g.dx)];
    for v in [&g.vx, &g.vy, &g.pressure, &g.density] {
        col(&mut out, v);
    }
    out
}

pub fn config(c: &GridConfig) -> Vec<Tok> {
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

/// The scratch lengths and the multigrid levels' lengths.
fn dump(g: &FluidGrid) -> Vec<Tok> {
    let d = g.oracle_dump();
    let mut out = vec![
        usz(d.scratch_a.len()),
        usz(d.scratch_b.len()),
        usz(d.scratch_c.len()),
        usz(d.diffuse_rhs.len()),
        usz(d.mg_levels.len()),
    ];
    for lv in &d.mg_levels {
        out.extend([
            usz(lv.residual.len()),
            usz(lv.coarse_rhs.len()),
            usz(lv.coarse_p.len()),
        ]);
    }
    out
}

/// A fresh grid with `src`'s public fields: what the Cyrius loader builds from the tokens (scratch as `new` makes it,
/// no multigrid levels). A size `new` refuses gets a 4 × 4 grid with the fields set over it.
fn fresh(src: &FluidGrid) -> FluidGrid {
    let mut g = FluidGrid::new(src.nx, src.ny, src.dx)
        .or_else(|_| FluidGrid::new(4, 4, 0.1))
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

/// A random state: velocities and pressure around 0, density in [0, 2), a cell size near 0.1.
fn random_grid(rng: &mut Rng, nx: usize, ny: usize) -> FluidGrid {
    let dx = rng.log_uniform(-2.0, 0.0);
    let mut g = FluidGrid::new(nx, ny, dx).expect("valid grid");
    for k in 0..nx * ny {
        g.vx[k] = rng.uniform(-2.0, 2.0);
        g.vy[k] = rng.uniform(-2.0, 2.0);
        g.pressure[k] = rng.uniform(-0.5, 0.5);
        g.density[k] = rng.uniform(0.0, 2.0);
    }
    g
}

fn new_rows(m: &mut Module) {
    let mut cases: Vec<(usize, usize, f64)> = vec![(4, 4, 0.1), (16, 9, 0.05), (32, 32, 1.0)];
    for (nx, ny) in [
        (3, 4),
        (4, 3),
        (0, 4),
        (4, 0),
        (3, 3),
        (usize::MAX, 4),
        (4, usize::MAX),
        (usize::MAX, usize::MAX),
        (1 << 32, 1 << 32),
        (1 << 33, 1 << 31),
    ] {
        cases.push((nx, ny, 0.1));
    }
    cases.push((usize::MAX, 4, f64::NAN));
    cases.push((1 << 32, 1 << 32, -1.0));
    for &x in &specials() {
        cases.push((5, 4, x));
        cases.push((3, 4, x));
    }
    for (nx, ny, dx) in cases {
        let out = match FluidGrid::new(nx, ny, dx) {
            Ok(g) => {
                let mut o = vec![usz(g.nx), usz(g.ny), f(g.dx)];
                for v in [&g.vx, &g.vy, &g.pressure, &g.density] {
                    o.push(usz(v.len()));
                    o.push(f(v[0]));
                }
                o.extend(dump(&g));
                o
            }
            Err(x) => vec![err(&x)],
        };
        m.row("FluidGrid::new", Class::A, &[usz(nx), usz(ny), f(dx)], &out);
    }
    for (name, c) in [
        ("GridConfig::smoke", GridConfig::smoke()),
        ("GridConfig::default", GridConfig::default()),
    ] {
        m.row(name, Class::A, &[], &config(&c));
    }
}

fn read_rows(m: &mut Module, rng: &mut Rng) {
    for k in 0..30 {
        let nx = 4 + (k % 6);
        let ny = 4 + ((k * 5) % 5);
        let mut g = random_grid(rng, nx, ny);
        match k % 10 {
            // A still grid: max_speed 0, cfl_dt f64::MAX·cfl.
            3 => {
                g.vx.iter_mut().for_each(|v| *v = 0.0);
                g.vy.iter_mut().for_each(|v| *v = -0.0);
            }
            // A NaN and an infinity in the fields: the reads must not panic.
            9 => {
                g.vx[nx + 1] = f64::NAN;
                g.vy[2 * nx - 2] = f64::INFINITY;
                g.density[3] = f64::NAN;
            }
            // Fields of different lengths: the zips stop at the shortest.
            7 => {
                g.vy.truncate(nx * ny - 3);
                g.density.truncate(nx);
            }
            _ => {}
        }
        let st = state(&g);
        let n = nx * ny;
        m.row(
            "FluidGrid::cell_count",
            Class::A,
            &st,
            &[usz(g.cell_count())],
        );
        let mut idx = vec![Tok::V(n)];
        let mut vel = vec![Tok::V(2 * n)];
        let mut spd = vec![Tok::V(n)];
        let cells = g.vx.len().min(g.vy.len());
        for c in 0..n {
            let (x, y) = (c % nx, c / nx);
            idx.push(usz(g.idx(x, y)));
            if c < cells {
                let (u, v) = g.velocity_at(x, y);
                vel.extend([f(u), f(v)]);
                spd.push(f(g.speed_at(x, y)));
            }
        }
        m.row("FluidGrid::idx", Class::A, &st, &idx);
        if cells == n {
            m.row("FluidGrid::velocity_at", Class::A, &st, &vel);
            m.row("FluidGrid::speed_at", Class::B, &st, &spd);
        }
        m.row("FluidGrid::max_speed", Class::B, &st, &[f(g.max_speed())]);
        m.row(
            "FluidGrid::total_kinetic_energy",
            Class::B,
            &st,
            &[f(g.total_kinetic_energy())],
        );
        // cfl_dt over eight (viscosity, cfl) pairs at once: the 1e-20 edges, NaN in each, a factor above 1.
        let pairs = [
            (0.0, 0.5),
            (1e-4, 0.5),
            (1e-20, 1.0),
            (2e-20, 1.0),
            (10.0, 0.9),
            (f64::NAN, 0.5),
            (1e-4, f64::NAN),
            (1e-4, 2.0),
        ];
        let mut ins = st.clone();
        ins.push(Tok::V(pairs.len()));
        ins.extend(pairs.iter().map(|&(v, _)| f(v)));
        ins.push(Tok::V(pairs.len()));
        ins.extend(pairs.iter().map(|&(_, c)| f(c)));
        let mut out = vec![Tok::V(pairs.len())];
        out.extend(pairs.iter().map(|&(v, c)| f(g.cfl_dt(v, c))));
        m.row("FluidGrid::cfl_dt", Class::B, &ins, &out);
    }
}

/// A `FluidGrid::diffuse` call: field, nx, ny, diff_rate, dt, dx, iterations.
type DiffuseCase = (Vec<f64>, usize, usize, f64, f64, f64, usize);

fn diffuse_rows(m: &mut Module, rng: &mut Rng) {
    let mut cases: Vec<DiffuseCase> = Vec::new();
    for k in 0..24 {
        let nx = 3 + (k % 7);
        let ny = 3 + ((k * 3) % 6);
        let field: Vec<f64> = (0..nx * ny).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let rate = rng.log_uniform(-5.0, 0.0);
        let dt = rng.log_uniform(-3.0, -1.0);
        let dx = rng.log_uniform(-2.0, 0.0);
        let iters = [0, 1, 2, 20][k % 4];
        cases.push((field, nx, ny, rate, dt, dx, iters));
    }
    // The benchmark's spike (scaled down), and the refusals: a short field, a small grid, an overflowing size.
    let mut spike = vec![0.0; 64];
    spike[32] = 100.0;
    cases.push((spike.clone(), 8, 8, 0.1, 0.01, 0.1, 20));
    cases.push((spike.clone(), 8, 7, 0.1, 0.01, 0.1, 20));
    cases.push((vec![1.0; 6], 2, 3, 0.1, 0.01, 0.1, 5));
    cases.push((vec![1.0; 6], 3, 2, 0.1, 0.01, 0.1, 5));
    cases.push((spike, usize::MAX, 8, 0.1, 0.01, 0.1, 5));
    cases.push((vec![2.0; 9], 3, 3, 0.0, 0.01, 0.1, 5));
    for (mut field, nx, ny, rate, dt, dx, iters) in cases {
        let mut ins = Vec::new();
        col(&mut ins, &field);
        ins.extend([usz(nx), usz(ny), f(rate), f(dt), f(dx), usz(iters)]);
        FluidGrid::diffuse(&mut field, nx, ny, rate, dt, dx, iters);
        let mut out = Vec::new();
        col(&mut out, &field);
        m.row("FluidGrid::diffuse", Class::B, &ins, &out);
    }
}

/// The pressure systems every solver is run on: (pressure guess, right-hand side, nx, ny). The right-hand side has a
/// zero ring (as the step's divergence leaves it); the guess is zero or a warm start, one with a non-zero ring
/// (Gauss–Seidel holds it).
pub fn systems(rng: &mut Rng) -> Vec<(Vec<f64>, Vec<f64>, usize, usize)> {
    let mut out = Vec::new();
    for (k, &(nx, ny)) in [
        (4, 4),
        (5, 7),
        (8, 8),
        (10, 10),
        (13, 13),
        (16, 12),
        (9, 17),
        (24, 24),
        (32, 32),
    ]
    .iter()
    .enumerate()
    {
        let n = nx * ny;
        let mut rhs = vec![0.0; n];
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                rhs[y * nx + x] = rng.uniform(-1.0, 1.0) * 0.05;
            }
        }
        let mut p = vec![0.0; n];
        if k % 3 == 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    p[y * nx + x] = rng.uniform(-0.1, 0.1);
                }
            }
        }
        if k == 2 {
            for v in p.iter_mut().take(nx) {
                *v = 0.25;
            }
        }
        out.push((p, rhs, nx, ny));
    }
    // A point source, as a single divergent cell gives.
    let mut rhs = vec![0.0; 144];
    rhs[6 * 12 + 5] = 1.0;
    out.push((vec![0.0; 144], rhs, 12, 12));
    out
}

fn solver_rows(m: &mut Module, rng: &mut Rng) {
    for (p0, rhs, nx, ny) in systems(rng) {
        let mut ins = Vec::new();
        col(&mut ins, &p0);
        col(&mut ins, &rhs);
        ins.extend([usz(nx), usz(ny)]);
        for iters in [1, 10, 40] {
            let mut p = p0.clone();
            let mut row = ins.clone();
            row.push(usz(iters));
            let out = match pressure_solve_gauss_seidel(&mut p, &rhs, nx, ny, iters) {
                Ok(()) => {
                    let mut o = Vec::new();
                    col(&mut o, &p);
                    o
                }
                Err(x) => vec![err(&x)],
            };
            m.row("oracle::pressure_solve_gauss_seidel", Class::B, &row, &out);
        }
        for (nu, cycles) in [(4, 1), (4, 4), (2, 0), (1, 2)] {
            let mut p = p0.clone();
            let mut row = ins.clone();
            row.extend([usz(nu), usz(cycles)]);
            let out = match pressure_solve_multigrid(&mut p, &rhs, nx, ny, nu, cycles) {
                Ok(levels) => {
                    let mut o = Vec::new();
                    col(&mut o, &p);
                    o.push(usz(levels.len()));
                    for lv in &levels {
                        col(&mut o, &lv.residual);
                        col(&mut o, &lv.coarse_rhs);
                        col(&mut o, &lv.coarse_p);
                    }
                    o
                }
                Err(x) => vec![err(&x)],
            };
            m.row("oracle::pressure_solve_multigrid", Class::B, &row, &out);
        }
        dst_row(m, &p0, &rhs, nx, ny);
        for iters in [1, 10, 40, 200] {
            cg_row(m, &p0, &rhs, nx, ny, iters);
        }
    }
}

/// One `oracle::pressure_solve_dst` row.
fn dst_row(m: &mut Module, p0: &[f64], rhs: &[f64], nx: usize, ny: usize) {
    let mut ins = Vec::new();
    col(&mut ins, p0);
    col(&mut ins, rhs);
    ins.extend([usz(nx), usz(ny)]);
    let mut p = p0.to_vec();
    let out = match pressure_solve_dst(&mut p, rhs, nx, ny) {
        Ok(()) => {
            let mut o = Vec::new();
            col(&mut o, &p);
            o
        }
        Err(x) => vec![err(&x)],
    };
    m.row("oracle::pressure_solve_dst", Class::C, &ins, &out);
}

/// One `oracle::pressure_solve_cg` row.
fn cg_row(m: &mut Module, p0: &[f64], rhs: &[f64], nx: usize, ny: usize, iters: usize) {
    let mut ins = Vec::new();
    col(&mut ins, p0);
    col(&mut ins, rhs);
    ins.extend([usz(nx), usz(ny), usz(iters)]);
    let mut p = p0.to_vec();
    let out = match pressure_solve_cg(&mut p, rhs, nx, ny, iters) {
        Ok(()) => {
            let mut o = Vec::new();
            col(&mut o, &p);
            o
        }
        Err(x) => vec![err(&x)],
    };
    m.row("oracle::pressure_solve_cg", Class::B, &ins, &out);
}

/// A random right-hand side with a zero ring, and a zero guess or (`warm`) a random interior one.
fn random_system(rng: &mut Rng, nx: usize, ny: usize, warm: bool) -> (Vec<f64>, Vec<f64>) {
    let n = nx * ny;
    let mut rhs = vec![0.0; n];
    let mut p = vec![0.0; n];
    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            rhs[y * nx + x] = rng.uniform(-1.0, 1.0) * 0.05;
            if warm {
                p[y * nx + x] = rng.uniform(-0.1, 0.1);
            }
        }
    }
    (p, rhs)
}

/// Part b's own systems: the DST at the transform lengths where glibc misrounds a sine or a cosine (12: four sines;
/// 18, 25: two and seven; 38 and 36: eleven and four; 60: two cosines of the eigenvalues), and the edges (an empty
/// interior, which leaves the guess; a one-cell side, whose sine is sin(π/2) and whose cosine is cos(π/2)). CG
/// on the smaller ones, on a zero right-hand side, and on an empty system (hisab refuses it: Diverged).
fn transform_rows(m: &mut Module, rng: &mut Rng) {
    for (nx, ny, warm) in [
        (14, 14, false),
        (20, 27, true),
        (40, 38, false),
        (62, 62, false),
        (14, 6, true),
    ] {
        let (p, rhs) = random_system(rng, nx, ny, warm);
        dst_row(m, &p, &rhs, nx, ny);
        // CG on the larger two would only repeat the 32 x 32 rows, at 300 KB.
        if nx * ny < 1000 {
            cg_row(m, &p, &rhs, nx, ny, 40);
        }
    }
    for (nx, ny) in [(3, 3), (3, 5), (5, 3), (3, 4)] {
        let (_, rhs) = random_system(rng, nx, ny, false);
        dst_row(m, &vec![0.5; nx * ny], &rhs, nx, ny);
        cg_row(m, &vec![0.5; nx * ny], &rhs, nx, ny, 10);
    }
    // An empty interior: the DST returns at once, the guess untouched (ring included).
    dst_row(m, &[0.5; 8], &[0.0; 8], 2, 4);
    dst_row(m, &[0.5; 8], &[0.0; 8], 4, 2);
    // A zero right-hand side: CG's first residual is already under its tolerance (1e-14 when ‖b‖ = 0).
    cg_row(m, &[0.0; 16], &[0.0; 16], 4, 4, 10);
    cg_row(m, &[], &[], 0, 0, 10);
}

fn json_rows(m: &mut Module, rng: &mut Rng) {
    let grid_tok = |doc: &str| match serde_json::from_str::<FluidGrid>(doc) {
        Ok(v) => state(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for k in 0..8 {
        let mut g = random_grid(rng, 4 + k % 3, 4 + k % 2);
        if k == 7 {
            g.vx[0] = f64::NAN;
        }
        docs.push(serde_json::to_string(&g).unwrap_or_default());
    }
    let base =
        serde_json::to_string(&FluidGrid::new(4, 4, 0.5).expect("valid")).unwrap_or_default();
    let zeros = "[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]";
    let vx_full = format!(r#""vx":{zeros}"#);
    let rho_full = format!(r#""density":{zeros}"#);
    for (from, to) in [
        (r#""nx":4"#, r#""nx":4.0"#),
        (r#""nx":4"#, r#""nx":-4"#),
        (r#""dx":0.5"#, r#""dx":"0.5""#),
        (r#""dx":0.5"#, r#""dx":0.5,"scratch_a":[1,2]"#),
        (r#","ny":4"#, ""),
        (vx_full.as_str(), r#""vx":[1.0]"#),
        (rho_full.as_str(), r#""density":[1,2,true]"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push(format!("[4,4,0.5,{zeros},{zeros},{zeros},{zeros}]"));
    docs.push(format!("[4,4,0.5,{zeros},{zeros},{zeros}]"));
    for doc in &docs {
        m.row(
            "FluidGrid::from_json",
            Class::A,
            &[s(doc.as_str())],
            &grid_tok(doc),
        );
    }

    let cfg_tok = |doc: &str| match serde_json::from_str::<GridConfig>(doc) {
        Ok(c) => config(&c),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    let mut docs: Vec<String> = Vec::new();
    for c in random_configs(rng, 6) {
        docs.push(serde_json::to_string(&c).unwrap_or_default());
    }
    let base = serde_json::to_string(&GridConfig::smoke()).unwrap_or_default();
    for (from, to) in [
        (r#""boundary":"NoSlip""#, r#""boundary":"FreeSlip""#),
        (r#""boundary":"NoSlip""#, r#""boundary":{"Periodic":null}"#),
        (r#""boundary":"NoSlip""#, r#""boundary":"Wall""#),
        (r#""boundary":"NoSlip""#, r#""boundary":0"#),
        (r#""use_cg":false"#, r#""use_cg":1"#),
        (r#""pressure_iterations":40"#, r#""pressure_iterations":-1"#),
        (r#","use_bfecc":false"#, ""),
        (r#""dt":0.1"#, r#""dt":null"#),
    ] {
        docs.push(base.replacen(from, to, 1));
    }
    docs.push(r#"[0.05,0.0,3,7,"Periodic",0.0,-0.5,1.0,true,false,0.1,true,false]"#.to_string());
    for doc in &docs {
        m.row(
            "GridConfig::from_json",
            Class::A,
            &[s(doc.as_str())],
            &cfg_tok(doc),
        );
    }

    for doc in [
        r#""NoSlip""#,
        r#""FreeSlip""#,
        r#""Periodic""#,
        r#"{"Periodic":null}"#,
        r#""Open""#,
        r#""noslip""#,
        "1",
        "null",
    ] {
        let out = match serde_json::from_str::<BoundaryCondition>(doc) {
            Ok(v) => vec![e("BoundaryCondition", v)],
            Err(_) => vec![Tok::Err("Deserialize".to_string())],
        };
        m.row("BoundaryCondition::from_json", Class::A, &[s(doc)], &out);
    }
}

/// Configs for the part a paths (periodic walls, or multigrid on walls), each scheme and force on or off.
fn random_configs(rng: &mut Rng, n: usize) -> Vec<GridConfig> {
    (0..n)
        .map(|k| {
            let mut c = GridConfig::smoke();
            c.dt = [0.1, 0.05, 0.02, 0.01][k % 4];
            c.viscosity = if rng.unit() < 0.3 {
                0.0
            } else {
                rng.log_uniform(-5.0, -2.0)
            };
            c.diffusion_iterations = 1 + (k % 3) * 10;
            c.pressure_iterations = [1, 10, 25, 40][(k / 2) % 4];
            c.boundary = [
                BoundaryCondition::NoSlip,
                BoundaryCondition::FreeSlip,
                BoundaryCondition::Periodic,
            ][k % 3];
            c.use_multigrid = c.boundary != BoundaryCondition::Periodic || rng.unit() < 0.3;
            c.vorticity_confinement = if rng.unit() < 0.4 {
                0.0
            } else {
                rng.uniform(0.0, 1.0)
            };
            c.buoyancy_alpha = rng.uniform(-0.5, 0.5);
            c.ambient_density = if rng.unit() < 0.5 {
                0.0
            } else {
                rng.uniform(0.0, 1.0)
            };
            c.use_maccormack = rng.unit() < 0.4;
            c.use_bfecc = rng.unit() < 0.3;
            c.smagorinsky_cs = if rng.unit() < 0.5 {
                0.0
            } else {
                rng.uniform(0.05, 0.2)
            };
            c
        })
        .collect()
}

fn step_row(m: &mut Module, g: &FluidGrid, c: &GridConfig, nsteps: usize) {
    let mut ins = state(g);
    ins.extend(config(c));
    ins.push(usz(nsteps));
    let mut s = fresh(g);
    let mut out = None;
    for _ in 0..nsteps {
        if let Err(x) = s.step(c) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| {
        let mut o = state(&s);
        o.extend(dump(&s));
        o
    });
    m.row("FluidGrid::step", Class::S, &ins, &out);
}

const FRAMES: [usize; 6] = [0, 1, 2, 10, 60, 300];

/// Bit-level trajectory rows: the initial state stepped 1, 2, 10, 60 and 300 times.
fn trajectory(m: &mut Module, g: &FluidGrid, c: &GridConfig) {
    for n in [1, 2, 10, 60, 300] {
        step_row(m, g, c, n);
    }
}

/// Class S rows: one step from the oracle's state at each frame (the scene stepped that far first, on a grid made
/// fresh for every step as the rows are; a scene that fails earlier stops there).
fn single_steps(m: &mut Module, g: &FluidGrid, c: &GridConfig) {
    let mut s = fresh(g);
    let mut at = 0;
    for &k in &FRAMES {
        while at < k {
            if s.step(c).is_err() {
                return;
            }
            at += 1;
        }
        step_row(m, &s, c, 1);
    }
}

/// A buoyant plume: density 1 and an upward kick on a strip near the floor.
fn plume(n: usize) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, 0.1).expect("valid");
    let mid = n / 2;
    for x in mid - 2..mid + 2 {
        let k = 2 * n + x;
        g.density[k] = 1.0;
        g.vy[k] = 1.0;
    }
    g
}

/// A vortex: u = ω × r with ω = s/(1 + r²) about the centre (rational, no transcendental).
fn vortex(n: usize) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, 0.1).expect("valid");
    let c = (fx(n) - 1.0) * 0.5;
    for k in 0..n * n {
        let (x, y) = (fx(k % n) - c, fx(k / n) - c);
        let w = 2.0 / (1.0 + x * x + y * y);
        g.vx[k] = -y * w;
        g.vy[k] = x * w;
        g.density[k] = if x * x + y * y < 4.0 { 1.0 } else { 0.0 };
    }
    g
}

/// An inflow: a jet along x through the middle rows, dye in it.
fn inflow(n: usize) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, 0.1).expect("valid");
    for y in n / 2 - 1..=n / 2 {
        for x in 1..n / 3 {
            g.vx[y * n + x] = 2.0;
            g.density[y * n + x] = 1.0;
        }
    }
    g
}

#[allow(clippy::too_many_lines)] // one block per scene
fn step_rows(m: &mut Module, rng: &mut Rng) {
    let n = 10;
    let smoke = GridConfig::smoke();
    let mut mg = smoke;
    mg.use_multigrid = true;
    let mut periodic = smoke;
    periodic.boundary = BoundaryCondition::Periodic;

    // Still tanks: nothing moves, on each wall rule.
    let tank = FluidGrid::new(n, n, 0.1).expect("valid");
    trajectory(m, &tank, &mg);
    let mut fs = mg;
    fs.boundary = BoundaryCondition::FreeSlip;
    trajectory(m, &tank, &fs);
    trajectory(m, &tank, &periodic);
    // A tank of uniform ambient-density fluid: buoyancy cancels.
    let mut level = tank.clone();
    level.density.iter_mut().for_each(|d| *d = 0.5);
    let mut amb = mg;
    amb.ambient_density = 0.5;
    trajectory(m, &level, &amb);

    // The buoyant plume on every scheme and force variant.
    let p = plume(n);
    let mut variants: Vec<GridConfig> = vec![mg, fs, periodic];
    let mut c = mg;
    c.use_maccormack = true;
    c.dt = 0.05;
    variants.push(c);
    let mut c = mg;
    c.use_bfecc = true;
    c.dt = 0.05;
    variants.push(c);
    let mut c = mg;
    c.use_bfecc = true;
    c.use_maccormack = true;
    c.boundary = BoundaryCondition::FreeSlip;
    variants.push(c);
    let mut c = periodic;
    c.use_maccormack = true; // ignored on periodic walls
    variants.push(c);
    let mut c = mg;
    c.smagorinsky_cs = 0.15;
    variants.push(c);
    let mut c = mg;
    c.vorticity_confinement = 0.0;
    c.viscosity = 0.0;
    variants.push(c);
    let mut c = mg;
    c.buoyancy_alpha = -0.2;
    c.ambient_density = 0.3;
    c.pressure_iterations = 7;
    variants.push(c);
    for c in &variants {
        single_steps(m, &p, c);
    }
    // The multigrid levels: a 16 × 16 plume (two levels).
    single_steps(m, &plume(16), &mg);
    let mut c = mg;
    c.pressure_iterations = 1;
    c.use_bfecc = true;
    single_steps(m, &plume(16), &c);

    // A vortex and an inflow.
    let v = vortex(n);
    single_steps(m, &v, &periodic);
    let mut c = mg;
    c.buoyancy_alpha = 0.0;
    single_steps(m, &v, &c);
    let mut c = periodic;
    c.vorticity_confinement = 2.0;
    c.smagorinsky_cs = 0.2;
    single_steps(m, &v, &c);
    let jet = inflow(n);
    single_steps(m, &jet, &mg);
    let mut c = periodic;
    c.use_bfecc = true;
    c.dt = 0.02;
    single_steps(m, &jet, &c);

    // Random states, one step each, on random configs.
    let configs = random_configs(rng, 24);
    for (k, c) in configs.iter().enumerate() {
        let g = random_grid(rng, 4 + k % 7, 4 + (k * 3) % 6);
        step_row(m, &g, c, 1);
    }

    // The refusals: the config's checks in Rust's order, the state's, and steps that diverge.
    let ok = FluidGrid::new(6, 6, 0.1).expect("valid");
    let mut bad_cfg: Vec<GridConfig> = Vec::new();
    for dt in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut c = mg;
        c.dt = dt;
        bad_cfg.push(c);
    }
    for (field, x) in [
        (0, -1e-3),
        (0, f64::NAN),
        (0, f64::INFINITY),
        (1, f64::NAN),
        (1, -0.1),
        (2, f64::NAN),
        (2, -1.0),
        (3, f64::NAN),
        (4, f64::NEG_INFINITY),
    ] {
        let mut c = mg;
        match field {
            0 => c.viscosity = x,
            1 => c.smagorinsky_cs = x,
            2 => c.vorticity_confinement = x,
            3 => c.buoyancy_alpha = x,
            _ => c.ambient_density = x,
        }
        bad_cfg.push(c);
    }
    let mut c = mg;
    c.pressure_iterations = 0;
    bad_cfg.push(c);
    let mut c = mg;
    c.diffusion_iterations = 0;
    bad_cfg.push(c);
    // Two failures at once: the first in Rust's order wins.
    let mut c = mg;
    c.viscosity = -1.0;
    c.dt = 0.0;
    bad_cfg.push(c);
    for c in &bad_cfg {
        step_row(m, &ok, c, 1);
    }
    let mut bad: Vec<FluidGrid> = Vec::new();
    let mut g = ok.clone();
    g.nx = 3;
    bad.push(g);
    let mut g = ok.clone();
    g.ny = usize::MAX;
    bad.push(g);
    let mut g = ok.clone();
    g.vx.truncate(35);
    bad.push(g);
    let mut g = ok.clone();
    g.density.push(0.0);
    bad.push(g);
    let mut g = ok.clone();
    g.pressure.clear();
    bad.push(g);
    for x in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut g = ok.clone();
        g.dx = x;
        bad.push(g);
    }
    // A NaN velocity (diverged after advection) and overflowing ones (non-finite after projection).
    let mut g = ok.clone();
    g.vx[14] = f64::NAN;
    bad.push(g);
    let mut g = ok.clone();
    g.vy[20] = f64::NAN;
    g.density[20] = 1.0;
    bad.push(g);
    let mut g = ok.clone();
    g.vx[14] = 1.7e308;
    g.vx[16] = -1.7e308;
    bad.push(g);
    let mut g = ok.clone();
    g.density[21] = 1e308;
    bad.push(g);
    for g in &bad {
        step_row(m, g, &mg, 1);
        step_row(m, g, &periodic, 1);
    }
}

/// Configs for the part b paths: the DST (no-slip or free-slip walls, no multigrid) or CG (any walls, multigrid
/// set or not: CG comes first), each scheme and force on or off.
fn random_configs_b(rng: &mut Rng, n: usize) -> Vec<GridConfig> {
    (0..n)
        .map(|k| {
            let mut c = GridConfig::smoke();
            c.dt = [0.1, 0.05, 0.02, 0.01][k % 4];
            c.viscosity = if rng.unit() < 0.3 {
                0.0
            } else {
                rng.log_uniform(-5.0, -2.0)
            };
            c.diffusion_iterations = 1 + (k % 3) * 10;
            c.pressure_iterations = [1, 10, 25, 40][(k / 2) % 4];
            c.use_cg = k % 2 == 1;
            c.boundary = if c.use_cg {
                [
                    BoundaryCondition::NoSlip,
                    BoundaryCondition::FreeSlip,
                    BoundaryCondition::Periodic,
                ][(k / 2) % 3]
            } else {
                [BoundaryCondition::NoSlip, BoundaryCondition::FreeSlip][(k / 2) % 2]
            };
            c.use_multigrid = c.use_cg && rng.unit() < 0.3;
            c.vorticity_confinement = if rng.unit() < 0.4 {
                0.0
            } else {
                rng.uniform(0.0, 1.0)
            };
            c.buoyancy_alpha = rng.uniform(-0.5, 0.5);
            c.ambient_density = if rng.unit() < 0.5 {
                0.0
            } else {
                rng.uniform(0.0, 1.0)
            };
            c.use_maccormack = rng.unit() < 0.4;
            c.use_bfecc = rng.unit() < 0.3;
            c.smagorinsky_cs = if rng.unit() < 0.5 {
                0.0
            } else {
                rng.uniform(0.05, 0.2)
            };
            c
        })
        .collect()
}

/// Part b's step rows: the DST projection (the smoke default: no-slip walls, no multigrid, no CG) and the CG one.
fn step_rows_b(m: &mut Module, rng: &mut Rng) {
    let n = 10;
    let smoke = GridConfig::smoke();
    let mut fs = smoke;
    fs.boundary = BoundaryCondition::FreeSlip;
    let mut cg = smoke;
    cg.use_cg = true;
    let mut cg_fs = cg;
    cg_fs.boundary = BoundaryCondition::FreeSlip;
    let mut cg_per = cg;
    cg_per.boundary = BoundaryCondition::Periodic;

    // Still tanks on each projection.
    let tank = FluidGrid::new(n, n, 0.1).expect("valid");
    for c in [&smoke, &fs, &cg, &cg_per] {
        trajectory(m, &tank, c);
    }

    // The buoyant plume: the DST on every scheme and force variant, CG at several budgets.
    let p = plume(n);
    let mut variants: Vec<GridConfig> = vec![smoke, fs, cg, cg_fs, cg_per];
    let mut c = smoke;
    c.use_maccormack = true;
    c.dt = 0.05;
    variants.push(c);
    let mut c = smoke;
    c.use_bfecc = true;
    c.dt = 0.05;
    variants.push(c);
    let mut c = fs;
    c.use_bfecc = true;
    c.use_maccormack = true;
    variants.push(c);
    let mut c = smoke;
    c.smagorinsky_cs = 0.15;
    variants.push(c);
    let mut c = smoke;
    c.vorticity_confinement = 0.0;
    c.viscosity = 0.0;
    variants.push(c);
    let mut c = smoke;
    c.buoyancy_alpha = -0.2;
    c.ambient_density = 0.3;
    variants.push(c);
    for iters in [1, 10, 200] {
        let mut c = cg;
        c.pressure_iterations = iters;
        variants.push(c);
    }
    let mut c = cg;
    c.use_multigrid = true; // CG comes first
    c.use_maccormack = true;
    variants.push(c);
    for c in &variants {
        single_steps(m, &p, c);
    }

    // A plume whose DST length (12) has misrounded sines: the corrected oracle's step rows.
    let p14 = plume(14);
    single_steps(m, &p14, &smoke);
    let mut c = fs;
    c.use_bfecc = true;
    single_steps(m, &p14, &c);
    single_steps(m, &p14, &cg);

    // A vortex and an inflow.
    let v = vortex(n);
    let mut c = smoke;
    c.buoyancy_alpha = 0.0;
    single_steps(m, &v, &c);
    let mut c = cg_per;
    c.vorticity_confinement = 2.0;
    c.smagorinsky_cs = 0.2;
    single_steps(m, &v, &c);
    let jet = inflow(n);
    single_steps(m, &jet, &smoke);
    single_steps(m, &jet, &cg_fs);

    // Random states, one step each, on random configs; two at sizes whose transform lengths misround.
    let configs = random_configs_b(rng, 24);
    for (k, c) in configs.iter().enumerate() {
        let g = random_grid(rng, 4 + k % 7, 4 + (k * 3) % 6);
        step_row(m, &g, c, 1);
    }
    for (nx, ny) in [(20, 27), (14, 40)] {
        let g = random_grid(rng, nx, ny);
        step_row(m, &g, &smoke, 1);
        step_row(m, &g, &cg, 1);
    }

    // The checks after the solve: velocities that overflow in the projection (non-finite after it, Diverged), and
    // a NaN velocity (Diverged after advection), on each projection.
    let ok = FluidGrid::new(6, 6, 0.1).expect("valid");
    let mut bad: Vec<FluidGrid> = Vec::new();
    let mut g = ok.clone();
    g.vx[14] = f64::NAN;
    bad.push(g);
    let mut g = ok.clone();
    g.vx[14] = 1.7e308;
    g.vx[16] = -1.7e308;
    bad.push(g);
    let mut g = ok.clone();
    g.density[21] = 1e308;
    bad.push(g);
    let mut g = ok.clone();
    g.vy[20] = 1e300;
    bad.push(g);
    for g in &bad {
        step_row(m, g, &smoke, 1);
        step_row(m, g, &cg, 1);
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("grid");
    let mut rng = Rng::new(0x6_71d);
    new_rows(&mut m);
    read_rows(&mut m, &mut rng);
    diffuse_rows(&mut m, &mut rng);
    solver_rows(&mut m, &mut rng);
    json_rows(&mut m, &mut rng);
    step_rows(&mut m, &mut rng);
    // Part b's rows, from their own seed, so part a's keep their values.
    let mut rng = Rng::new(0x6_72b);
    transform_rows(&mut m, &mut rng);
    step_rows_b(&mut m, &mut rng);
    // Part c's: KEpsilon, apply_ghost_fluid, MacGrid (m_grid_c.rs).
    crate::m_grid_c::rows(&mut m);
    m
}
