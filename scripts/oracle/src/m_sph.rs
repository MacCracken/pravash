//! `sph.tsv` — `rust-old/src/sph.rs` (M4). Part a: the kernels, the equations of state, the non-Newtonian viscosity
//! models and the batch and whole-system helpers.
//!
//! Rows:
//! - each kernel (`kernel_poly6`, `kernel_spiky_grad`, `kernel_viscosity_laplacian`, `kernel_wendland_c2`, `_c2_grad`,
//!   `_c4`, `_c4_grad`) takes h and `v:<k>` distances, and gives `v:<k>` values (class B: +, −, ×, ÷ and π);
//! - `equation_of_state` takes (ρ, ρ₀, k) (class B); `equation_of_state_tait` takes (ρ, ρ₀, c, γ) (class C: ln and
//!   powf, the corrected oracle where glibc misrounds);
//! - `NonNewtonianViscosity::effective_viscosity` takes `e:NonNewtonianViscosity.<Variant>`, the four fields
//!   (consistency, power_index, yield_stress, plastic_viscosity; 0 for one the variant lacks) and `v:<k>` strain
//!   rates, and gives `v:<k>` viscosities (class C: powf);
//! - `batch_poly6` takes `v:` squared distances and h; `batch_distance_squared` takes a position (x, y, z) and `v:<3k>`
//!   positions, and gives `v:<k>` (class B);
//! - `total_kinetic_energy` and `max_speed` take `i:<n>` and n particles (18 tokens each, `m_common::particle`).
//!
//! Part b (from its own seed, after part a's rows): the neighbour grid and the z-order sort.
//! - `SphSolver::neighbor_lists` takes h, `i:<n>` and n positions (x, y, z), and gives what a fresh solver's first
//!   step builds from them (`neighbor_lists` and `oracle_dump`): the grid's f32 cell size (widened), `i:` its point
//!   and cell counts, `v:<3n>` the f32 positions (widened), `v:<3n>` each particle's own cell key (`sph_cell_key`),
//!   `v:<n+1>` the CSR offsets and `v:<m>` the neighbour indices, in the order the solver sums them (class A). The
//!   lists depend only on the positions and h; no position's key is saturated, where the solver's `cx + 1` would
//!   overflow (the oracle's build panics on it);
//! - `sort_by_zorder` takes the cell size, `i:<n>` and n particles, and gives `v:<n>` the Morton codes in Rust's
//!   output order (as words; `v:0` when it returns early) and the particles in the stable order on (code, input
//!   index), which Rust's unstable sort need not keep for ties (ADR 0002 #3): class A on both.

use hisab::DVec3;
use pravash::sph::SphSolver;
use pravash::sph::{
    NonNewtonianViscosity, batch_distance_squared, batch_poly6, equation_of_state,
    equation_of_state_tait, kernel_poly6, kernel_spiky_grad, kernel_viscosity_laplacian,
    kernel_wendland_c2, kernel_wendland_c2_grad, kernel_wendland_c4, kernel_wendland_c4_grad,
    max_speed, sort_by_zorder, total_kinetic_energy,
};
use pravash::{FluidConfig, FluidParticle};

use crate::grids::{Rng, specials};
use crate::m_common::particle;
use crate::tsv::{Class, Module, Tok, f, i};

/// The smoothing radii every kernel is evaluated at: the specials (0, ±∞, NaN, the smallest values), negatives,
/// the usual SPH range, and sizes whose h⁶ or h⁹ overflows or underflows.
fn radii() -> Vec<f64> {
    let mut h = specials();
    h.extend([
        -0.1,
        1e-3,
        0.0125,
        0.05,
        0.1,
        0.7,
        1.0,
        3.0,
        1e3,
        1e40,
        1e-40,
        0.05 + 1e-17,
    ]);
    h
}

/// The distances for radius h: multiples of h across [−0.1h, 2.4h] (0, 1 and 2 exactly among them), random ones,
/// and the specials.
fn distances(rng: &mut Rng, h: f64) -> Vec<f64> {
    let mut r: Vec<f64> = [
        -0.1, 0.0, 1e-3, 0.25, 0.5, 0.999, 1.0, 1.0001, 1.2, 1.5, 1.999, 2.0, 2.0001, 2.4,
    ]
    .iter()
    .map(|t| t * h)
    .collect();
    for _ in 0..8 {
        r.push(rng.uniform(0.0, 2.4) * if h.is_finite() { h } else { 1.0 });
    }
    r.extend([
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        h,
        2.0 * h,
    ]);
    r
}

type Kernel = fn(f64, f64) -> f64;

fn kernel_rows(m: &mut Module, rng: &mut Rng) {
    let kernels: [(&str, Kernel); 7] = [
        ("kernel_poly6", kernel_poly6),
        ("kernel_spiky_grad", kernel_spiky_grad),
        ("kernel_viscosity_laplacian", kernel_viscosity_laplacian),
        ("kernel_wendland_c2", kernel_wendland_c2),
        ("kernel_wendland_c2_grad", kernel_wendland_c2_grad),
        ("kernel_wendland_c4", kernel_wendland_c4),
        ("kernel_wendland_c4_grad", kernel_wendland_c4_grad),
    ];
    for h in radii() {
        let r = distances(rng, h);
        for (name, k) in kernels {
            let mut ins = vec![f(h), Tok::V(r.len())];
            ins.extend(r.iter().map(|&x| f(x)));
            let mut out = vec![Tok::V(r.len())];
            out.extend(r.iter().map(|&x| f(k(x, h))));
            m.row(name, Class::B, &ins, &out);
        }
    }
}

fn eos_rows(m: &mut Module, rng: &mut Rng) {
    let sp = specials();
    for k in 0..120 {
        let (d, r0, g) = if k < 30 {
            (sp[k % sp.len()], 1000.0, sp[(k / 10) % sp.len()])
        } else {
            (
                rng.uniform(900.0, 1100.0),
                rng.uniform(500.0, 1500.0),
                rng.log_uniform(-1.0, 5.0),
            )
        };
        m.row(
            "equation_of_state",
            Class::B,
            &[f(d), f(r0), f(g)],
            &[f(equation_of_state(d, r0, g))],
        );
    }
    // Tait: γ across the |γ| < 1e-12 limit and the usual values, ρ/ρ₀ near 1, the floor on ρ₀, the specials.
    let gammas = [
        0.0,
        -0.0,
        1e-13,
        -1e-13,
        9.9e-13,
        1e-12,
        -1e-12,
        1.1e-12,
        1e-6,
        0.5,
        1.0,
        1.4,
        7.0,
        7.15,
        -1.0,
        -7.0,
        30.0,
        f64::NAN,
        f64::INFINITY,
    ];
    let rests = [1000.0, 1.0, 1e-12, 0.0, -1.0, 1e-10, f64::NAN];
    for &g in &gammas {
        for &r0 in &rests {
            for _ in 0..3 {
                let d = r0.max(1e-10) * rng.uniform(0.9, 1.1);
                let c = rng.uniform(10.0, 1500.0);
                m.row(
                    "equation_of_state_tait",
                    Class::C,
                    &[f(d), f(r0), f(c), f(g)],
                    &[f(equation_of_state_tait(d, r0, c, g))],
                );
            }
        }
    }
    // Arguments where glibc 2.44 misrounds: pow(x, 7), pow(x, 1.4) and ln(x) at ρ/ρ₀ = x (ρ₀ = 1), found by an MPFR
    // search (the corrected oracle's rows, tests/golden/sph.cr.tsv).
    for (g, xs) in [
        (
            7.0,
            [
                0.932_913_024_166_896_1,
                0.922_254_830_494_574_6,
                0.904_590_947_283_188_5,
                0.970_352_069_149_676,
            ],
        ),
        (
            1.4,
            [
                0.900_539_010_473_246_2,
                1.082_521_949_799_253,
                0.926_759_028_331_358_6,
                1.024_197_643_128_195_4,
            ],
        ),
        (
            0.0,
            [
                0.921_890_293_015_856_8,
                0.955_961_285_651_867_1,
                1.056_449_724_131_400_8,
                1.067_640_851_921_145_3,
            ],
        ),
    ] {
        for x in xs {
            m.row(
                "equation_of_state_tait",
                Class::C,
                &[f(x), f(1.0), f(1480.0), f(g)],
                &[f(equation_of_state_tait(x, 1.0, 1480.0, g))],
            );
        }
    }
    for &d in &sp {
        for &c in &[0.0, 50.0, f64::NAN] {
            m.row(
                "equation_of_state_tait",
                Class::C,
                &[f(d), f(1000.0), f(c), f(7.0)],
                &[f(equation_of_state_tait(d, 1000.0, c, 7.0))],
            );
            m.row(
                "equation_of_state_tait",
                Class::C,
                &[f(d), f(1000.0), f(c), f(0.0)],
                &[f(equation_of_state_tait(d, 1000.0, c, 0.0))],
            );
        }
    }
}

/// A model's tokens: its variant, then consistency, power_index, yield_stress and plastic_viscosity.
fn model_toks(v: &NonNewtonianViscosity) -> Vec<Tok> {
    match *v {
        NonNewtonianViscosity::PowerLaw {
            consistency,
            power_index,
        } => vec![
            Tok::E("NonNewtonianViscosity", "PowerLaw".to_string()),
            f(consistency),
            f(power_index),
            f(0.0),
            f(0.0),
        ],
        NonNewtonianViscosity::Bingham {
            yield_stress,
            plastic_viscosity,
        } => vec![
            Tok::E("NonNewtonianViscosity", "Bingham".to_string()),
            f(0.0),
            f(0.0),
            f(yield_stress),
            f(plastic_viscosity),
        ],
        NonNewtonianViscosity::HerschelBulkley {
            yield_stress,
            consistency,
            power_index,
        } => vec![
            Tok::E("NonNewtonianViscosity", "HerschelBulkley".to_string()),
            f(consistency),
            f(power_index),
            f(yield_stress),
            f(0.0),
        ],
        _ => vec![],
    }
}

fn viscosity_rows(m: &mut Module, rng: &mut Rng) {
    let mut models = vec![];
    for &(k, n) in &[
        (0.5, 0.7),
        (2.0, 1.3),
        (1.0, 1.0),
        (0.0, 0.5),
        (1e-3, 0.0),
        (3.0, 2.0),
        (1.0, f64::NAN),
    ] {
        models.push(NonNewtonianViscosity::PowerLaw {
            consistency: k,
            power_index: n,
        });
    }
    for &(ty, mu) in &[
        (10.0, 0.1),
        (0.0, 1.0),
        (5.0, 0.0),
        (1e-3, 1e-3),
        (f64::NAN, 1.0),
    ] {
        models.push(NonNewtonianViscosity::Bingham {
            yield_stress: ty,
            plastic_viscosity: mu,
        });
    }
    for &(ty, k, n) in &[
        (10.0, 0.5, 0.7),
        (0.0, 2.0, 1.3),
        (5.0, 0.0, 0.5),
        (1.0, 1.0, 1.0),
    ] {
        models.push(NonNewtonianViscosity::HerschelBulkley {
            yield_stress: ty,
            consistency: k,
            power_index: n,
        });
    }
    for _ in 0..6 {
        models.push(NonNewtonianViscosity::HerschelBulkley {
            yield_stress: rng.log_uniform(-2.0, 2.0),
            consistency: rng.log_uniform(-3.0, 1.0),
            power_index: rng.uniform(0.2, 2.0),
        });
    }
    let mut rates: Vec<f64> = vec![
        0.0,
        -0.0,
        1e-12,
        1e-10,
        1e-3,
        0.5,
        1.0,
        7.0,
        1e3,
        1e10,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        // Strain rates where glibc misrounds pow(s, 0.7 − 1) and pow(s, 1.3 − 1), found by an MPFR search.
        16.920_573_787_902_896,
        74.322_287_648_657_83,
        63.915_817_101_158_98,
        42.522_401_217_009_31,
        4.932_696_350_626_567,
        66.616_743_959_181_05,
    ];
    for _ in 0..6 {
        rates.push(rng.log_uniform(-6.0, 4.0));
    }
    for v in &models {
        let mut ins = model_toks(v);
        ins.push(Tok::V(rates.len()));
        ins.extend(rates.iter().map(|&x| f(x)));
        let mut out = vec![Tok::V(rates.len())];
        out.extend(rates.iter().map(|&x| f(v.effective_viscosity(x))));
        m.row(
            "NonNewtonianViscosity::effective_viscosity",
            Class::C,
            &ins,
            &out,
        );
    }
}

fn batch_rows(m: &mut Module, rng: &mut Rng) {
    for &h in &[0.05, 1.0, 0.0, -1.0, f64::NAN, 0.1] {
        let r2: Vec<f64> = (0..20)
            .map(|k| {
                if k < 4 {
                    [0.0, h * h, h * h * 1.0001, -1.0][k]
                } else {
                    rng.uniform(0.0, 1.5) * h * h
                }
            })
            .collect();
        let mut ins = vec![Tok::V(r2.len())];
        ins.extend(r2.iter().map(|&x| f(x)));
        ins.push(f(h));
        let out_v = batch_poly6(&r2, h);
        let mut out = vec![Tok::V(out_v.len())];
        out.extend(out_v.iter().map(|&x| f(x)));
        m.row("batch_poly6", Class::B, &ins, &out);
    }
    m.row("batch_poly6", Class::B, &[Tok::V(0), f(0.05)], &[Tok::V(0)]);
    for k in 0..6 {
        let pos = DVec3::new(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        );
        let n = [0, 1, 5, 17, 3, 9][k];
        let mut ps: Vec<DVec3> = (0..n)
            .map(|_| {
                DVec3::new(
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(-2.0, 2.0),
                )
            })
            .collect();
        if k == 4 {
            ps[0] = pos;
            ps[1] = DVec3::new(f64::NAN, 0.0, 0.0);
            ps[2] = DVec3::new(f64::INFINITY, 1e300, -0.0);
        }
        let mut ins = vec![f(pos.x), f(pos.y), f(pos.z), Tok::V(3 * n)];
        for p in &ps {
            ins.extend([f(p.x), f(p.y), f(p.z)]);
        }
        let d = batch_distance_squared(pos, &ps);
        let mut out = vec![Tok::V(d.len())];
        out.extend(d.iter().map(|&x| f(x)));
        m.row("batch_distance_squared", Class::B, &ins, &out);
    }
}

fn total_rows(m: &mut Module, rng: &mut Rng) {
    let sp = specials();
    for k in 0..12 {
        let n = [0, 1, 2, 7, 30, 100][k % 6];
        let ps: Vec<FluidParticle> = (0..n)
            .map(|j| {
                let mut p =
                    FluidParticle::new_2d(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), 0.0);
                p.mass = rng.log_uniform(-4.0, 0.0);
                p.velocity = DVec3::new(
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(-3.0, 3.0),
                );
                if k >= 6 && j % 5 == 1 {
                    p.velocity.x = sp[(j + k) % sp.len()];
                    p.mass = sp[(j * 3 + k) % sp.len()];
                }
                p
            })
            .collect();
        let mut ins = vec![i(i64::try_from(n).unwrap_or(0))];
        for p in &ps {
            ins.extend(particle(p));
        }
        m.row(
            "total_kinetic_energy",
            Class::B,
            &ins,
            &[f(total_kinetic_energy(&ps))],
        );
        m.row("max_speed", Class::B, &ins, &[f(max_speed(&ps))]);
    }
}

/// One neighbour-list row: a fresh solver's first step on particles at `pos`, smoothing radius h.
fn neighbor_row(m: &mut Module, pos: &[DVec3], h: f64) {
    let mut ps: Vec<FluidParticle> = pos.iter().map(|&p| FluidParticle::new(p, 0.001)).collect();
    let mut cfg = FluidConfig::water_2d();
    cfg.smoothing_radius = h;
    let mut solver = SphSolver::new();
    let _ = solver.step(&mut ps, &cfg, 0.0);
    let d = solver.oracle_dump();
    let (offsets, indices) = solver.neighbor_lists();
    assert_eq!(offsets.len(), pos.len() + 1, "the step built the lists");
    let h32 = h as f32;
    let mut ins = vec![f(h), i(i64::try_from(pos.len()).unwrap_or(0))];
    for p in pos {
        ins.extend([f(p.x), f(p.y), f(p.z)]);
    }
    let mut out = vec![
        f(f64::from(d.cell_size)),
        i(i64::try_from(d.grid_len).unwrap_or(0)),
        i(i64::try_from(d.grid_cells).unwrap_or(0)),
        Tok::V(3 * pos.len()),
    ];
    for q in &d.positions_f32 {
        out.extend(q.iter().map(|&c| f(f64::from(c))));
    }
    out.push(Tok::V(3 * pos.len()));
    for q in &d.positions_f32 {
        let k = pravash::oracle::sph_cell_key(*q, h32);
        out.extend(k.iter().map(|&c| i(i64::from(c))));
    }
    out.push(Tok::V(offsets.len()));
    out.extend(offsets.iter().map(|&o| i(i64::from(o))));
    out.push(Tok::V(indices.len()));
    out.extend(indices.iter().map(|&j| i(i64::try_from(j).unwrap_or(0))));
    m.row("SphSolver::neighbor_lists", Class::A, &ins, &out);
}

/// A jittered lattice of `side`^d points at spacing `sp` (d = 2: z = 0), shifted by `off`.
fn lattice(rng: &mut Rng, side: usize, three_d: bool, sp: f64, jit: f64, off: f64) -> Vec<DVec3> {
    let mut v = Vec::new();
    let nz = if three_d { side } else { 1 };
    for z in 0..nz {
        for y in 0..side {
            for x in 0..side {
                let j = |r: &mut Rng| if jit > 0.0 { r.uniform(-jit, jit) } else { 0.0 };
                let (a, b, c) = (j(rng), j(rng), j(rng));
                v.push(DVec3::new(
                    off + crate::m_grid::fx(x) * sp + a,
                    off + crate::m_grid::fx(y) * sp + b,
                    if three_d {
                        off + crate::m_grid::fx(z) * sp + c
                    } else {
                        0.0
                    },
                ));
            }
        }
    }
    v
}

fn neighbor_rows(m: &mut Module, rng: &mut Rng) {
    // Lattices on cell boundaries (h a power of two, so k·h is exact; and h = 0.05, which isn't), jittered ones,
    // 2D and 3D.
    for &(side, three_d, sp, h) in &[
        (8, false, 0.125, 0.125),
        (8, false, 0.05, 0.05),
        (6, true, 0.125, 0.25),
        (5, true, 0.05, 0.1),
        (10, false, 0.025, 0.05),
    ] {
        neighbor_row(m, &lattice(rng, side, three_d, sp, 0.0, 0.0), h);
        neighbor_row(m, &lattice(rng, side, three_d, sp, sp * 0.1, 0.0), h);
    }
    // Random clouds, negative coordinates, and large offsets where the f32 positions and probes lose precision (the
    // probes land in other cells: D3).
    for &(n, l, off, h) in &[
        (60, 0.5, 0.0, 0.1),
        (200, 1.0, -0.5, 0.1),
        (120, 0.3, -7.25, 0.07),
        (80, 0.5, 1e3, 0.05),
        (80, 0.5, 1e5, 0.05),
        (80, 0.5, 1e6, 0.05),
        (80, 0.5, 3e6, 0.05),
        (40, 2.0, 0.0, 1.0),
        (500, 1.0, 0.0, 0.08),
    ] {
        let pts: Vec<DVec3> = (0..n)
            .map(|k| {
                DVec3::new(
                    off + rng.uniform(0.0, l),
                    off + rng.uniform(0.0, l),
                    if k % 2 == 0 {
                        0.0
                    } else {
                        off + rng.uniform(0.0, l)
                    },
                )
            })
            .collect();
        neighbor_row(m, &pts, h);
    }
    // A cell size within f32::EPSILON of the fresh grid's 1.0, which keeps that grid (and its 1/cell) while the
    // probes use 1/h; one 2·EPSILON away, which renews it; and one that rounds to 1.0 in f32.
    let near_one: Vec<DVec3> = (0..30)
        .map(|_| {
            DVec3::new(
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
            )
        })
        .collect();
    for h in [
        1.0,
        1.0 + f64::from(f32::EPSILON),
        1.0 - f64::from(f32::EPSILON) / 2.0,
        1.0 + 2.0 * f64::from(f32::EPSILON),
        1.0 + 1e-9,
    ] {
        neighbor_row(m, &near_one, h);
    }
    // One particle, duplicates, a NaN coordinate (its key is 0), points a hair either side of a cell boundary.
    neighbor_row(m, &[DVec3::new(0.3, 0.4, 0.0)], 0.1);
    let mut dup = vec![DVec3::new(0.21, 0.37, 0.05); 5];
    dup.push(DVec3::new(0.22, 0.37, 0.05));
    neighbor_row(m, &dup, 0.1);
    let mut nan = lattice(rng, 4, false, 0.1, 0.0, 0.0);
    nan[5].x = f64::NAN;
    nan[9].y = -0.0;
    neighbor_row(m, &nan, 0.1);
    let edge: Vec<DVec3> = [
        0.1,
        0.1 - 1e-12,
        0.1 + 1e-12,
        0.2,
        0.2 - 1e-9,
        -1e-12,
        0.0,
        -0.0,
    ]
    .iter()
    .map(|&x| DVec3::new(x, 0.05, 0.0))
    .collect();
    neighbor_row(m, &edge, 0.1);
    // The kept grid again, on integer points (where its 1/cell and 1/h key apart) and on a cluster at 4.5e6 (where
    // its probes drift a cell, the D3 defect): the only rows that tell the grid's key from 1/h. No RNG, so the rows
    // after keep their values.
    let mut kept = Vec::new();
    for z in -1..3 {
        for y in -1..3 {
            for x in -1..3 {
                kept.push(DVec3::new(f64::from(x), f64::from(y), f64::from(z)));
            }
        }
    }
    for k in 0..4 {
        let o = 0.5 * f64::from(k);
        kept.push(DVec3::new(4.5e6 + o, 4.5e6 - o, 4.5e6 + o));
    }
    for h in [
        1.0 + f64::from(f32::EPSILON),
        1.0 - f64::from(f32::EPSILON) / 2.0,
    ] {
        neighbor_row(m, &kept, h);
    }
}

fn zorder_row(m: &mut Module, input: &[FluidParticle], cell: f64) {
    let mut ins = vec![f(cell), i(i64::try_from(input.len()).unwrap_or(0))];
    for p in input {
        ins.extend(particle(p));
    }
    let mut ps = input.to_vec();
    sort_by_zorder(&mut ps, cell);
    let sorted = ps.len() >= 2 && cell.is_finite() && cell > 0.0;
    let mut out = Vec::new();
    if sorted {
        let inv = 1.0 / cell;
        let (mx, my) = input.iter().fold((f64::MAX, f64::MAX), |(a, b), p| {
            (a.min(p.position.x), b.min(p.position.y))
        });
        let code = |p: &FluidParticle| {
            let spread = |x: u32| {
                let mut v = u64::from(x);
                v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
                v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
                v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
                v = (v | (v << 2)) & 0x3333_3333_3333_3333;
                (v | (v << 1)) & 0x5555_5555_5555_5555
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ix = ((p.position.x - mx) * inv).max(0.0) as u32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let iy = ((p.position.y - my) * inv).max(0.0) as u32;
            spread(ix) | (spread(iy) << 1)
        };
        out.push(Tok::V(ps.len()));
        out.extend(
            ps.iter()
                .map(|p| i(i64::from_ne_bytes(code(p).to_ne_bytes()))),
        );
        // The stable order: each output particle's input index (the first unused input with the same bits), then
        // the particles sorted on (code, input index).
        let bits = |p: &FluidParticle| {
            particle(p)
                .iter()
                .map(|t| format!("{t:?}"))
                .collect::<Vec<_>>()
        };
        let mut used = vec![false; input.len()];
        let mut tagged: Vec<(u64, usize, FluidParticle)> = ps
            .iter()
            .map(|p| {
                let b = bits(p);
                let j = (0..input.len())
                    .find(|&j| !used[j] && bits(&input[j]) == b)
                    .expect("a permutation");
                used[j] = true;
                (code(p), j, *p)
            })
            .collect();
        tagged.sort_by_key(|&(c, j, _)| (c, j));
        for (_, _, p) in &tagged {
            out.extend(particle(p));
        }
    } else {
        out.push(Tok::V(0));
        for p in &ps {
            out.extend(particle(p));
        }
    }
    m.row("sort_by_zorder", Class::A, &ins, &out);
}

fn zorder_rows(m: &mut Module, rng: &mut Rng) {
    let block = |nx: usize, ny: usize, sp: f64, ox: f64| -> Vec<FluidParticle> {
        let mut v = Vec::new();
        for y in 0..ny {
            for x in 0..nx {
                v.push(FluidParticle::new_2d(
                    ox + crate::m_grid::fx(x) * sp,
                    crate::m_grid::fx(y) * sp,
                    0.001,
                ));
            }
        }
        v
    };
    // Rust's test shapes: a block in a shuffled order, at its own spacing (distinct codes) and at a coarse cell
    // (many ties); negative coordinates.
    for &(nx, ny, sp, ox, cell) in &[
        (10, 10, 0.1, 0.0, 0.1),
        (10, 10, 0.1, 0.0, 0.35),
        (5, 5, 0.1, -3.0, 0.1),
        (7, 3, 0.25, -1.0, 1.0),
    ] {
        let mut ps = block(nx, ny, sp, ox);
        for k in (1..ps.len()).rev() {
            let j = usize::try_from(rng.int(0, i64::try_from(k).unwrap_or(0))).unwrap_or(0);
            ps.swap(k, j);
        }
        for (k, p) in ps.iter_mut().enumerate() {
            p.velocity = DVec3::new(crate::m_grid::fx(k), 0.0, 0.0);
        }
        zorder_row(m, &ps, cell);
    }
    // Random clouds, special coordinates (NaN, ±∞, huge: saturated codes), ties, and the early returns.
    let sp = specials();
    for k in 0..8 {
        let n = [2, 3, 17, 40, 64, 9, 12, 30][k];
        let ps: Vec<FluidParticle> = (0..n)
            .map(|j| {
                let mut p =
                    FluidParticle::new_2d(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), 0.001);
                p.mass = crate::m_grid::fx(j);
                if k >= 5 && j % 4 == 1 {
                    p.position.x = sp[(j + k) % sp.len()];
                }
                if k == 7 && j % 5 == 0 {
                    p.position.y = 1e12;
                }
                p
            })
            .collect();
        zorder_row(m, &ps, [0.1, 0.5, 0.03, 1.0, 0.2, 0.1, 0.25, 0.05][k]);
    }
    // Every coordinate positive, so both minima are the particles' own (each row above has a minimum ≤ 0 or returns
    // early). No RNG.
    for &(ox, oy, cell) in &[(0.37, 1.25, 0.1), (5.0, 7.5, 0.3)] {
        let mut ps = block(6, 4, 0.1, ox);
        ps.reverse();
        for p in &mut ps {
            p.position.y += oy;
        }
        zorder_row(m, &ps, cell);
    }
    let pair = vec![
        FluidParticle::new_2d(0.5, 0.5, 1.0),
        FluidParticle::new_2d(0.1, 0.1, 2.0),
    ];
    for cell in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        zorder_row(m, &pair, cell);
    }
    zorder_row(m, &pair[..1], 0.1);
    zorder_row(m, &[], 0.1);
}

pub fn emit() -> Module {
    let mut m = Module::new("sph");
    let mut rng = Rng::new(0x0_5f4a);
    kernel_rows(&mut m, &mut rng);
    eos_rows(&mut m, &mut rng);
    viscosity_rows(&mut m, &mut rng);
    batch_rows(&mut m, &mut rng);
    total_rows(&mut m, &mut rng);
    // Part b's rows, from their own seed.
    let mut rng = Rng::new(0x0_5f4b);
    neighbor_rows(&mut m, &mut rng);
    zorder_rows(&mut m, &mut rng);
    m
}
