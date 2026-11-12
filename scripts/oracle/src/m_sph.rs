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
//!
//! Part c (from its own seed, after part b's rows): the brute-force density and forces, the free step, the
//! particle block, splitting and merging. `<ps>` is `i:<n>` and n particles.
//! - `compute_density` and `pressure_force` take h and `<ps>`, and give each particle's value in index order:
//!   `v:<n>` densities, `v:<3n>` forces (class B); `viscosity_force` takes h, μ and `<ps>` (class B);
//! - `step` takes a config (14 tokens, `m_common::config`), μ, `i:<steps>` and `<ps>`, and gives the n particles
//!   after that many steps, or the first step's `E:` (class S, a trajectory as the shallow-water rows);
//! - `create_particle_block` takes the origin, the size, the spacing and the mass, and gives `v:<count>` and the
//!   particles (class A);
//! - `split_particle` takes the parent (18 tokens), `i:<children>` and the offset, and gives `v:<k>` and the
//!   children (class C: cos and sin, the corrected oracle where glibc misrounds);
//! - `merge_particles` takes two particles and gives the merged one (class B).
//!
//! Part d (from its own seed, after part c's rows): the physics extensions. `<lists>` is CSR neighbour lists:
//! `v:<n+1>` offsets and `v:<m>` indices, brute-force (every pair within h, self included, as Rust's tests build
//! them), shuffled per particle on some rows (the sums follow the list order), and invalid on others (the functions
//! then change nothing). Each gives the n particles after the call (class B) unless stated:
//! - `update_viscoelastic` takes h, G, λ, dt, `<ps>`, `<lists>`; `update_heat` h, κ, dt, `<ps>`, `<lists>`;
//!   `apply_delta_sph` h, c₀, δ, dt, `<ps>`, `<lists>`; `apply_implicit_viscosity` h, ν, dt, `i:<iterations>`,
//!   `<ps>`, `<lists>`;
//! - `compute_gradient_corrections` takes h, `<ps>`, `<lists>` and gives `v:<9n>` row-major matrices (class B);
//!   `oracle::invert_3x3` takes 9 values and gives 9 or `n:` (class B); `apply_gradient_correction` takes 9 values
//!   and a vector and gives a vector (class B);
//! - `update_phase_change` takes T*, L, c, `i:` below, `i:` above, dt, `<ps>`;
//! - `update_reaction` (through `CombustionConfig`'s `ReactionProvider`) and `update_combustion` take the ignition
//!   temperature, the rate, the heat release, dt, `<ps>`; `CombustionConfig::react` takes those three, T, fuel, dt
//!   and gives (consumed, heat) (class B);
//! - `apply_contact_angle` takes a config (14), θ, σ, `<ps>` (class C: cos, the corrected oracle where glibc
//!   misrounds);
//! - `generate_secondary_particles` takes the threshold, the lifetime, `<ps>` and gives `v:<k>` secondaries (8
//!   tokens: position, velocity, lifetime, `e:SecondaryType.<Kind>`); `update_secondary_particles` takes gravity,
//!   dt, `v:<k>` secondaries and gives the survivors (class B).
//!
//! M5 part a (from its own seed, after part d's rows): the solver's WCSPH step.
//! - `SphSolver::step` takes a config (14), μ, σ (its `surface_tension`), `i:` use_verlet, `i:<steps>` and `<ps>`,
//!   and gives the particles after that many steps of one fresh solver, or the first error (class S trajectories:
//!   the frames 1, 2, 10, 60 and 300 of each scene, each a row from the start);
//! - `SphSolver::adaptive_dt` takes h, μ, the CFL factor, dt_min, dt_max and `<ps>` (class B).

use hisab::DVec3;
use pravash::sph::SphSolver;
use pravash::sph::{
    CombustionConfig, HeatConfig, NonNewtonianViscosity, PhaseChangeConfig, ReactionProvider,
    SecondaryParticle, SecondaryType, ViscoelasticConfig, apply_contact_angle, apply_delta_sph,
    apply_gradient_correction, apply_implicit_viscosity, batch_distance_squared, batch_poly6,
    compute_density, compute_gradient_corrections, create_particle_block, equation_of_state,
    equation_of_state_tait, generate_secondary_particles, kernel_poly6, kernel_spiky_grad,
    kernel_viscosity_laplacian, kernel_wendland_c2, kernel_wendland_c2_grad, kernel_wendland_c4,
    kernel_wendland_c4_grad, max_speed, merge_particles, pressure_force, sort_by_zorder,
    split_particle, step, total_kinetic_energy, update_combustion, update_heat,
    update_phase_change, update_reaction, update_secondary_particles, update_viscoelastic,
    viscosity_force,
};
use pravash::{FluidConfig, FluidParticle};

use crate::grids::{Rng, specials};
use crate::m_common::{config, particle};
use crate::m_grid::usz;
use crate::tsv::{Class, Module, Tok, err, f, i};

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

// ── Part c ──────────────────────────────────────────────────────────────────────────

/// `i:<n>` and the particles.
fn ptoks(ps: &[FluidParticle]) -> Vec<Tok> {
    let mut v = vec![i(i64::try_from(ps.len()).unwrap_or(0))];
    for p in ps {
        v.extend(particle(p));
    }
    v
}

/// A random cloud of n particles in [off, off + l)², z 0 or (3D) in the same range, with velocities in (−1, 1),
/// densities around ρ₀, pressures of both signs and masses near m.
fn cloud(rng: &mut Rng, n: usize, l: f64, off: f64, three_d: bool, m: f64) -> Vec<FluidParticle> {
    (0..n)
        .map(|_| {
            let mut p =
                FluidParticle::new_2d(off + rng.uniform(0.0, l), off + rng.uniform(0.0, l), m);
            if three_d {
                p.position.z = off + rng.uniform(0.0, l);
            }
            p.velocity = DVec3::new(
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                if three_d { rng.uniform(-1.0, 1.0) } else { 0.0 },
            );
            p.density = rng.uniform(500.0, 1500.0);
            p.pressure = rng.uniform(-2e3, 2e4);
            p.mass = m * rng.uniform(0.5, 1.5);
            p
        })
        .collect()
}

/// The three brute-force rows on one particle set: density, pressure force, viscosity force, for every index.
fn force_row(m: &mut Module, ps: &[FluidParticle], h: f64, mu: f64) {
    let n = ps.len();
    let mut ins = vec![f(h)];
    ins.extend(ptoks(ps));
    let mut out = vec![Tok::V(n)];
    out.extend((0..n).map(|k| f(compute_density(k, ps, h))));
    m.row("compute_density", Class::B, &ins, &out);
    let mut out = vec![Tok::V(3 * n)];
    for k in 0..n {
        out.extend(pressure_force(k, ps, h).map(f));
    }
    m.row("pressure_force", Class::B, &ins, &out);
    let mut vins = vec![f(h), f(mu)];
    vins.extend(ptoks(ps));
    let mut out = vec![Tok::V(3 * n)];
    for k in 0..n {
        out.extend(viscosity_force(k, ps, h, mu).map(f));
    }
    m.row("viscosity_force", Class::B, &vins, &out);
}

fn force_rows(m: &mut Module, rng: &mut Rng) {
    // Random clouds, 2D and 3D, at the usual radii and at ones every pair or no pair is within.
    for &(n, l, off, three_d, h, mu) in &[
        (40, 0.3, 0.0, false, 0.05, 0.001),
        (60, 0.2, -0.1, false, 0.1, 0.01),
        (50, 0.2, 0.0, true, 0.08, 1.0),
        (30, 0.1, 5.0, true, 1.0, 0.0),
        (25, 1.0, 0.0, false, 0.01, -0.5),
        (1, 1.0, 0.0, false, 0.05, 0.001),
        (0, 1.0, 0.0, false, 0.05, 0.001),
    ] {
        let ps = cloud(rng, n, l, off, three_d, 0.01);
        force_row(m, &ps, h, mu);
    }
    // A lattice on the radius exactly (spacing h: r² = h² is inside, (h·√2)² outside), coincident particles
    // (r² = 0 < 1e-20, skipped by the forces, counted by the density), and pairs a hair either side of 1e-20.
    let mut lat = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            let mut p = FluidParticle::new_2d(
                crate::m_grid::fx(x) * 0.125,
                crate::m_grid::fx(y) * 0.125,
                0.02,
            );
            p.density = 1000.0 + crate::m_grid::fx(x + 3 * y);
            p.pressure = 50.0 * crate::m_grid::fx(x * y) - 300.0;
            p.velocity = DVec3::new(crate::m_grid::fx(x) * 0.1, -crate::m_grid::fx(y) * 0.2, 0.0);
            lat.push(p);
        }
    }
    force_row(m, &lat, 0.125, 0.01);
    let mut dup = cloud(rng, 6, 0.05, 0.0, false, 0.01);
    dup.push(dup[0]);
    dup.push(dup[2]);
    let mut near = dup[1];
    near.position.x += 9e-11; // r² ≈ 8.1e-21: under 1e-20
    dup.push(near);
    let mut far = dup[3];
    far.position.y += 1.1e-10; // r² ≈ 1.21e-20: over it
    dup.push(far);
    force_row(m, &dup, 0.05, 0.1);
    // Special values: densities at and below the 1e-10 floor (0, negative, NaN, ∞), masses and pressures 0,
    // negative, NaN and ±∞, a NaN position.
    let sp = specials();
    for k in 0..4 {
        let mut ps = cloud(rng, 16, 0.1, 0.0, k % 2 == 1, 0.01);
        for (j, p) in ps.iter_mut().enumerate() {
            match (j + k) % 5 {
                0 => p.density = sp[(j * 3 + k) % sp.len()],
                1 => p.mass = sp[(j + 2 * k) % sp.len()],
                2 => p.pressure = sp[(j * 5 + k) % sp.len()],
                3 if j == 3 && k == 3 => p.position.x = f64::NAN,
                _ => {}
            }
        }
        force_row(m, &ps, [0.05, 0.1, 0.03, 0.05][k], 0.001);
    }
    // The radius itself: 0, negative, NaN, ∞, huge and tiny (the coefficients over- and underflow).
    let small = cloud(rng, 8, 0.1, 0.0, false, 0.01);
    for h in [0.0, -0.05, f64::NAN, f64::INFINITY, 1e40, 1e-40, 1e-3] {
        force_row(m, &small, h, 0.001);
    }
}

/// One step row: `steps` steps of `step` from ps (stopping at the first error, which is the row's output).
fn step_row(m: &mut Module, ps: &[FluidParticle], cfg: &FluidConfig, mu: f64, steps: usize) {
    let mut ins = config(cfg);
    ins.extend([f(mu), usz(steps)]);
    ins.extend(ptoks(ps));
    let mut s = ps.to_vec();
    let mut out = None;
    for _ in 0..steps {
        if let Err(x) = step(&mut s, cfg, mu) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| s.iter().flat_map(particle).collect());
    m.row("step", Class::S, &ins, &out);
}

/// A 2D block of water whose interior density is ρ₀: its mass is ρ₀ over the interior particle's density at mass 1.
fn water_block(nx: usize, ny: usize, sp: f64, ox: f64, oy: f64, h: f64) -> Vec<FluidParticle> {
    let size = [crate::m_grid::fx(nx) * sp, crate::m_grid::fx(ny) * sp];
    let mut ps = create_particle_block([ox, oy], size, sp, 1.0);
    let mid = (ny / 2) * nx + nx / 2;
    let m = 1000.0 / compute_density(mid.min(ps.len().saturating_sub(1)), &ps, h);
    for p in &mut ps {
        p.mass = m;
    }
    ps
}

fn step_rows(m: &mut Module, rng: &mut Rng) {
    let water = FluidConfig::water_2d();
    // A dam break: an 8 × 8 block at the floor, its trajectory at 1, 2, 10, 60 and 300 steps.
    let dam = water_block(8, 8, 0.025, 0.0125, 0.0125, 0.05);
    for steps in [1, 2, 10, 60, 300] {
        step_row(m, &dam, &water, 0.001, steps);
    }
    // One step from the trajectory's states at frames 1, 10 and 60 (each a row of its own).
    let mut s = dam.clone();
    let mut at = 0;
    for frame in [1, 10, 60] {
        while at < frame {
            if step(&mut s, &water, 0.001).is_err() {
                break;
            }
            at += 1;
        }
        step_row(m, &s, &water, 0.001, 1);
    }
    // A 3D cube in a closed box (every axis has hi > lo), viscous.
    let mut cube_cfg = FluidConfig::water_2d();
    cube_cfg.bounds_max = DVec3::new(0.5, 0.5, 0.5);
    cube_cfg.smoothing_radius = 0.06;
    let mut cube = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let mut p = FluidParticle::new_2d(
                    0.1 + crate::m_grid::fx(x) * 0.03,
                    0.1 + crate::m_grid::fx(y) * 0.03,
                    0.02,
                );
                p.position.z = 0.1 + crate::m_grid::fx(z) * 0.03;
                cube.push(p);
            }
        }
    }
    for steps in [1, 10, 60] {
        step_row(m, &cube, &cube_cfg, 0.5, steps);
    }
    // A random splash: velocities toward and through every wall, z off the 2D plane (water_2d's z has hi = lo, so
    // only its low wall acts), and the Rust tests' single particles (one falling, one below the floor).
    let mut splash = cloud(rng, 24, 0.9, 0.05, false, 0.01);
    for (j, p) in splash.iter_mut().enumerate() {
        p.velocity = DVec3::new(rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0), 0.0);
        if j % 6 == 0 {
            p.position.z = [0.2, -0.2, 0.0, -0.0][(j / 6) % 4];
        }
    }
    for steps in [1, 5, 30] {
        step_row(m, &splash, &water, 0.01, steps);
    }
    let mut one = FluidParticle::new_2d(0.5, 0.5, 0.01);
    one.density = 1000.0;
    step_row(m, &[one], &water, 0.001, 1);
    step_row(m, &[one], &water, 0.001, 50);
    let mut below = FluidParticle::new_2d(0.5, -0.1, 0.01);
    below.density = 1000.0;
    step_row(m, &[below], &water, 0.001, 1);
    step_row(m, &[], &water, 0.001, 1);
    // Viscosities 0 and negative, damping 0 and 1, a box whose hi is below lo on no axis but equal on two.
    let mut cfg = FluidConfig::water_2d();
    cfg.boundary_damping = 0.0;
    step_row(m, &splash, &cfg, 0.0, 5);
    cfg.boundary_damping = 1.0;
    step_row(m, &splash, &cfg, -0.01, 5);
    cfg.bounds_max = DVec3::new(1.0, 0.0, 0.0);
    step_row(m, &splash, &cfg, 0.001, 3);
    // The errors: each of the config's checks, in its order, and a diverging step (a NaN position, an ∞ mass).
    let small = cloud(rng, 5, 0.2, 0.3, false, 0.01);
    for k in 0..9 {
        let mut c = FluidConfig::water_2d();
        match k {
            0 => c.dt = -1.0,
            1 => c.smoothing_radius = f64::NAN,
            2 => c.rest_density = 0.0,
            3 => c.gas_constant = f64::INFINITY,
            4 => c.boundary_damping = 1.5,
            5 => c.gravity.y = f64::NAN,
            6 => c.bounds_min.x = f64::NEG_INFINITY,
            7 => c.bounds_min.y = 2.0,
            _ => c.dt = 0.0,
        }
        step_row(m, &small, &c, 0.001, 1);
        step_row(m, &[], &c, 0.001, 1);
    }
    let mut nan = small.clone();
    nan[2].position.y = f64::NAN;
    step_row(m, &nan, &water, 0.001, 1);
    // An ∞ mass beside a neighbour: its density, then its pressure, are ∞, and the neighbour's acceleration NaN.
    let mut inf = small.clone();
    inf[4].mass = f64::INFINITY;
    inf[4].position = inf[1].position;
    inf[4].position.x += 0.01;
    step_row(m, &inf, &water, 0.001, 1);
    let mut late = small.clone();
    late[4].position = late[3].position;
    late[4].position.x += 1e-3;
    late[4].pressure = 1e308;
    late[4].density = f64::NAN;
    step_row(m, &late, &water, 0.001, 3);
    // One component diverging alone: an ∞ x velocity reaches a neighbour's x acceleration through viscosity, while y
    // and z stay finite (Rust checks each).
    let mut one_axis = small.clone();
    one_axis[0].position = one_axis[1].position;
    one_axis[0].position.x += 0.01;
    one_axis[0].velocity = DVec3::new(f64::INFINITY, 0.0, 0.0);
    step_row(m, &one_axis, &water, 0.001, 1);
}

fn block_row(m: &mut Module, origin: [f64; 2], size: [f64; 2], spacing: f64, mass: f64) {
    let ps = create_particle_block(origin, size, spacing, mass);
    let ins = [
        f(origin[0]),
        f(origin[1]),
        f(size[0]),
        f(size[1]),
        f(spacing),
        f(mass),
    ];
    let mut out = vec![Tok::V(ps.len())];
    for p in &ps {
        out.extend(particle(p));
    }
    m.row("create_particle_block", Class::A, &ins, &out);
}

fn block_rows(m: &mut Module, rng: &mut Rng) {
    // Rust's test shapes, quotients a hair under an integer (0.3/0.1 = 2.9999999999999996, ceil 3), negative and
    // zero sizes, and the early returns.
    block_row(m, [0.0, 0.0], [0.1, 0.1], 0.05, 0.01);
    block_row(m, [1.0, 2.0], [0.1, 0.1], 0.05, 0.01);
    block_row(m, [0.0, 0.0], [1.0, 1.0], 0.1, 0.01);
    block_row(m, [0.0, 0.0], [0.3, 0.7], 0.1, 0.001);
    block_row(m, [-0.25, 0.5], [0.6, 0.2], 0.2, 1.0);
    block_row(m, [0.0, 0.0], [0.1 + 1e-17, 0.05], 0.05, 0.01);
    block_row(m, [0.0, 0.0], [-0.5, 0.3], 0.1, 0.01);
    block_row(m, [0.0, 0.0], [-0.05, 0.3], 0.1, 0.01);
    block_row(m, [0.0, 0.0], [0.0, 0.3], 0.1, 0.01);
    block_row(m, [0.0, 0.0], [0.3, -0.0], 0.1, 0.01);
    for sp in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        block_row(m, [0.0, 0.0], [0.3, 0.3], sp, 0.01);
    }
    for sz in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        block_row(m, [0.0, 0.0], [sz, 0.3], 0.1, 0.01);
        block_row(m, [0.0, 0.0], [0.3, sz], 0.1, 0.01);
    }
    // Saturated counts: ∞ × ∞ overflows (`checked_mul`: empty) and ∞ × 0 is 0 (empty). A count of 0 along x with ∞
    // rows would loop 2⁶⁴ times in Rust, so it has no row.
    block_row(m, [0.0, 0.0], [1e300, 1e300], 1e-300, 0.01);
    block_row(m, [0.0, 0.0], [1e300, 0.0], 1e-300, 0.01);
    // NaN and huge origins, NaN and negative masses, a tiny spacing over a small size.
    block_row(m, [f64::NAN, 0.0], [0.2, 0.2], 0.1, 0.01);
    block_row(m, [1e300, -1e300], [0.2, 0.2], 0.1, f64::NAN);
    block_row(m, [0.0, 0.0], [1e-9, 2e-9], 1e-10, -1.0);
    for _ in 0..10 {
        let sp = rng.log_uniform(-2.5, -0.5);
        let size = [rng.uniform(0.0, 12.0) * sp, rng.uniform(0.0, 12.0) * sp];
        let origin = [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)];
        block_row(m, origin, size, sp, rng.log_uniform(-4.0, 0.0));
    }
}

fn split_row(m: &mut Module, parent: &FluidParticle, n: usize, offset: f64) {
    let mut ins = particle(parent);
    ins.extend([usz(n), f(offset)]);
    let ch = split_particle(parent, n, offset);
    let mut out = vec![Tok::V(ch.len())];
    for p in &ch {
        out.extend(particle(p));
    }
    m.row("split_particle", Class::C, &ins, &out);
}

fn split_rows(m: &mut Module, rng: &mut Rng) {
    let mut parent = FluidParticle::new_2d(0.3, 0.7, 0.04);
    parent.velocity = DVec3::new(1.5, -0.5, 0.25);
    parent.density = 1003.0;
    parent.pressure = 120.0;
    parent.temperature = 350.0;
    parent.fuel = 0.4;
    parent.phase = 2;
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 33, 64] {
        split_row(m, &parent, n, 0.01);
    }
    for offset in [0.0, -0.02, 1e10, f64::NAN, f64::INFINITY] {
        split_row(m, &parent, 6, offset);
    }
    // Child counts with an angle glibc misrounds (an MPFR search over n ≤ 300: n = 19's k = 11 sine, 27's k = 13
    // sine, 39's k = 1 sine, 52's k = 13 cosine, 61's k = 19 cosine), from the origin so the ulp shows.
    let origin = FluidParticle::new_2d(0.0, 0.0, 1.0);
    for n in [19, 27, 39, 52, 61] {
        split_row(m, &origin, n, 1.0);
    }
    // Parents with special masses and positions.
    let sp = specials();
    for k in 0..6 {
        let mut p = FluidParticle::new_2d(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.01);
        match k {
            0 => p.mass = 0.0,
            1 => p.mass = f64::NAN,
            2 => p.mass = -0.5,
            3 => p.position.x = f64::INFINITY,
            4 => p.mass = sp[3],
            _ => p.position.z = 0.75,
        }
        split_row(m, &p, 3 + k, rng.uniform(0.0, 0.05));
    }
}

fn merge_rows(m: &mut Module, rng: &mut Rng) {
    let sp = specials();
    let rnd = |rng: &mut Rng, phase: u8| {
        let mut p = FluidParticle::new_2d(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0);
        p.position.z = rng.uniform(-1.0, 1.0);
        p.velocity = DVec3::new(
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        );
        p.acceleration = DVec3::new(rng.uniform(-9.0, 9.0), 1.0, 2.0);
        p.density = rng.uniform(900.0, 1100.0);
        p.pressure = rng.uniform(-500.0, 5000.0);
        p.mass = rng.log_uniform(-4.0, 0.0);
        p.phase = phase;
        p.conformation = [
            rng.uniform(0.5, 1.5),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.5, 1.5),
        ];
        p.temperature = rng.uniform(250.0, 400.0);
        p.fuel = rng.uniform(0.0, 1.0);
        p
    };
    for k in 0..30 {
        let a = rnd(rng, (k % 3) as u8);
        let mut b = rnd(rng, 1);
        match k {
            20 => b.mass = -a.mass,
            21 => {
                b.mass = 0.0;
            }
            22 => b.mass = f64::NAN,
            23 => b.mass = f64::INFINITY,
            24 => b.position.x = f64::NAN,
            25 => b.density = f64::NEG_INFINITY,
            26 => b.mass = sp[(k * 7) % sp.len()],
            27 => b.mass = -2.0,
            _ => {}
        }
        let mut a2 = a;
        if k == 21 {
            a2.mass = 0.0;
        }
        let mut ins = particle(&a2);
        ins.extend(particle(&b));
        m.row(
            "merge_particles",
            Class::B,
            &ins,
            &particle(&merge_particles(&a2, &b)),
        );
    }
}

// ── Part d ──────────────────────────────────────────────────────────────────────────

/// A part d call on particles and CSR lists.
type ListedFn<'a> = &'a dyn Fn(&mut [FluidParticle], &[u32], &[usize]);
/// A part d call taking its scalars (h first) as well.
type ParamFn<'a> = &'a dyn Fn(&[f64], &mut [FluidParticle], &[u32], &[usize]);

/// Brute-force CSR lists: for each particle, every particle within h (`distance_to <= h`, itself included), in index
/// order, or shuffled per particle when `rng` is given.
fn brute_lists(ps: &[FluidParticle], h: f64, mut rng: Option<&mut Rng>) -> (Vec<u32>, Vec<usize>) {
    let mut offsets = vec![0u32];
    let mut indices = Vec::new();
    for pi in ps {
        let mut l: Vec<usize> = (0..ps.len())
            .filter(|&j| pi.distance_to(&ps[j]) <= h)
            .collect();
        if let Some(r) = rng.as_deref_mut() {
            for k in (1..l.len()).rev() {
                let j = usize::try_from(r.int(0, i64::try_from(k).unwrap_or(0))).unwrap_or(0);
                l.swap(k, j);
            }
        }
        indices.extend(l);
        offsets.push(u32::try_from(indices.len()).unwrap_or(u32::MAX));
    }
    (offsets, indices)
}

fn lists_tok(o: &[u32], idx: &[usize]) -> Vec<Tok> {
    let mut v = vec![Tok::V(o.len())];
    v.extend(o.iter().map(|&x| i(x)));
    v.push(Tok::V(idx.len()));
    v.extend(idx.iter().map(|&x| usz(x)));
    v
}

/// The four invalid list shapes for n particles from valid ones: too few offsets, a decreasing pair, a last offset
/// past the indices, an index at n.
fn invalid_lists(o: &[u32], idx: &[usize], n: usize) -> Vec<(Vec<u32>, Vec<usize>)> {
    let mut out = vec![(o[..o.len() - 1].to_vec(), idx.to_vec())];
    if o.len() >= 3 {
        let mut d = o.to_vec();
        d.swap(1, 2);
        if d[1] > d[2] {
            out.push((d, idx.to_vec()));
        }
    }
    let mut past = o.to_vec();
    if let Some(l) = past.last_mut() {
        *l += 1;
    }
    out.push((past, idx.to_vec()));
    if !idx.is_empty() {
        let mut bad = idx.to_vec();
        bad[0] = n;
        out.push((o.to_vec(), bad));
    }
    out
}

/// A cloud with the fields part d reads varied: conformation, temperature, fuel, acceleration, density.
fn rich_cloud(rng: &mut Rng, n: usize, l: f64, three_d: bool) -> Vec<FluidParticle> {
    let mut ps = cloud(rng, n, l, 0.0, three_d, 0.01);
    for p in &mut ps {
        p.conformation = [
            rng.uniform(0.5, 2.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.5, 2.0),
        ];
        p.temperature = rng.uniform(250.0, 700.0);
        p.fuel = rng.uniform(0.0, 1.0);
        p.acceleration = DVec3::new(
            rng.uniform(-30.0, 30.0),
            rng.uniform(-30.0, 30.0),
            rng.uniform(-3.0, 3.0),
        );
    }
    ps
}

/// A row taking `lead` scalars, `<ps>` and `<lists>`, giving the particles after `f`.
fn listed_row(
    m: &mut Module,
    name: &str,
    lead: &[Tok],
    ps: &[FluidParticle],
    o: &[u32],
    idx: &[usize],
    f: ListedFn,
) {
    let mut ins = lead.to_vec();
    ins.extend(ptoks(ps));
    ins.extend(lists_tok(o, idx));
    let mut out = ps.to_vec();
    f(&mut out, o, idx);
    let toks: Vec<Tok> = out.iter().flat_map(particle).collect();
    m.row(name, Class::B, &ins, &toks);
}

/// The listed rows of one function: random clouds (2D and 3D) with index-order and shuffled lists, a lattice, the
/// invalid shapes, n = 0, and the scalars' specials through `vary`.
fn listed_rows(m: &mut Module, rng: &mut Rng, name: &str, base: &[f64], call: ParamFn) {
    let h = base[0];
    let lead = |v: &[f64]| -> Vec<Tok> {
        v.iter()
            .enumerate()
            .map(|(k, &x)| {
                if name == "apply_implicit_viscosity" && k == 3 {
                    usz(x as usize)
                } else {
                    f(x)
                }
            })
            .collect()
    };
    let run = |m: &mut Module, v: &[f64], ps: &[FluidParticle], o: &[u32], idx: &[usize]| {
        listed_row(m, name, &lead(v), ps, o, idx, &|q, a, b| call(v, q, a, b));
    };
    for &(n, l, three_d) in &[(30, 0.15, false), (40, 0.12, true), (12, 0.06, false)] {
        let ps = rich_cloud(rng, n, l, three_d);
        let (o, idx) = brute_lists(&ps, h, None);
        run(m, base, &ps, &o, &idx);
        let (o2, idx2) = brute_lists(&ps, h, Some(rng));
        run(m, base, &ps, &o2, &idx2);
    }
    let mut lat = create_particle_block([0.0, 0.0], [0.2, 0.2], 0.02, 0.001);
    for (k, p) in lat.iter_mut().enumerate() {
        p.density = 1000.0 + crate::m_grid::fx(k % 7);
        p.velocity.x = p.position.y;
        p.temperature = 293.15 + crate::m_grid::fx(k % 5);
    }
    let (o, idx) = brute_lists(&lat, h, None);
    run(m, base, &lat, &o, &idx);
    // Valid lists that start past index 0 (Rust allows a first offset above 0).
    let small = rich_cloud(rng, 6, 0.05, false);
    let (o, idx) = brute_lists(&small, h, None);
    let mut shifted_idx = vec![0usize, 1];
    shifted_idx.extend(&idx);
    let shifted_o: Vec<u32> = o.iter().map(|&x| x + 2).collect();
    run(m, base, &small, &shifted_o, &shifted_idx);
    for (bo, bi) in invalid_lists(&o, &idx, small.len()) {
        run(m, base, &small, &bo, &bi);
    }
    run(m, base, &[], &[0], &[]);
    // Specials of each scalar after h, one at a time.
    let sp = [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-30, 1e30];
    for k in 1..base.len() {
        if name == "apply_implicit_viscosity" && k == 3 {
            continue;
        }
        for &x in &sp {
            let mut v = base.to_vec();
            v[k] = x;
            run(m, &v, &small, &o, &idx);
        }
    }
    for hv in [0.0, f64::NAN, 1.0] {
        let mut v = base.to_vec();
        v[0] = hv;
        run(m, &v, &small, &o, &idx);
    }
}

fn viscoelastic_rows(m: &mut Module, rng: &mut Rng) {
    listed_rows(
        m,
        rng,
        "update_viscoelastic",
        &[0.05, 100.0, 0.05, 0.001],
        &|v, ps, o, idx| {
            update_viscoelastic(ps, o, idx, v[0], &ViscoelasticConfig::new(v[1], v[2]), v[3]);
        },
    );
    // Rust's simple-shear test shape: u = γ·y, no relaxation, G 0.
    let mut shear = create_particle_block([0.0, 0.0], [0.2, 0.2], 0.02, 0.001);
    for p in &mut shear {
        p.density = 1000.0;
        p.velocity.x = p.position.y;
    }
    let (o, idx) = brute_lists(&shear, 0.05, None);
    listed_row(
        m,
        "update_viscoelastic",
        &[f(0.05), f(0.0), f(1e12), f(0.01)],
        &shear,
        &o,
        &idx,
        &|q, a, b| update_viscoelastic(q, a, b, 0.05, &ViscoelasticConfig::new(0.0, 1e12), 0.01),
    );
}

fn heat_rows(m: &mut Module, rng: &mut Rng) {
    listed_rows(
        m,
        rng,
        "update_heat",
        &[0.05, 0.01, 0.001],
        &|v, ps, o, idx| {
            update_heat(ps, o, idx, v[0], &HeatConfig::new(v[1]), v[2]);
        },
    );
}

fn delta_rows(m: &mut Module, rng: &mut Rng) {
    listed_rows(
        m,
        rng,
        "apply_delta_sph",
        &[0.05, 10.0, 0.1, 0.001],
        &|v, ps, o, idx| {
            apply_delta_sph(ps, o, idx, v[0], v[1], v[2], v[3]);
        },
    );
    // Rust's test shape: two particles 0.02 apart, densities 1000 and 1100.
    let mut two = vec![
        FluidParticle::new_2d(0.0, 0.0, 1.0),
        FluidParticle::new_2d(0.02, 0.0, 1.0),
    ];
    two[0].density = 1000.0;
    two[1].density = 1100.0;
    let (o, idx) = brute_lists(&two, 0.05, None);
    listed_row(
        m,
        "apply_delta_sph",
        &[f(0.05), f(10.0), f(0.1), f(1e-3)],
        &two,
        &o,
        &idx,
        &|q, a, b| apply_delta_sph(q, a, b, 0.05, 10.0, 0.1, 1e-3),
    );
}

fn implicit_rows(m: &mut Module, rng: &mut Rng) {
    listed_rows(
        m,
        rng,
        "apply_implicit_viscosity",
        &[0.05, 0.5, 0.01, 5.0],
        &|v, ps, o, idx| {
            apply_implicit_viscosity(ps, o, idx, v[0], v[1], v[2], v[3] as usize);
        },
    );
    // Rust's test shape (alternating velocities, ν·dt far past the explicit limit, 200 sweeps), and 0, 1 and 50
    // sweeps of a cloud (the early exit included).
    let mut alt = create_particle_block([0.0, 0.0], [0.1, 0.1], 0.02, 0.001);
    for (k, p) in alt.iter_mut().enumerate() {
        p.density = 100.0;
        p.velocity.x = if k % 2 == 0 { 1.0 } else { -1.0 };
    }
    let (o, idx) = brute_lists(&alt, 0.05, None);
    for iters in [200usize, 0, 1, 50] {
        listed_row(
            m,
            "apply_implicit_viscosity",
            &[f(0.05), f(10.0), f(1.0), usz(iters)],
            &alt,
            &o,
            &idx,
            &|q, a, b| apply_implicit_viscosity(q, a, b, 0.05, 10.0, 1.0, iters),
        );
    }
}

fn mls_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, ps: &[FluidParticle], o: &[u32], idx: &[usize], h: f64| {
        let mut ins = vec![f(h)];
        ins.extend(ptoks(ps));
        ins.extend(lists_tok(o, idx));
        let c = compute_gradient_corrections(ps, o, idx, h);
        let mut out = vec![Tok::V(9 * c.len())];
        for mat in &c {
            out.extend(mat.iter().map(|&x| f(x)));
        }
        m.row("compute_gradient_corrections", Class::B, &ins, &out);
    };
    // 3D clouds (invertible moments), a 2D lattice (z has no moment: singular, the identity kept), mixed lists.
    for &(n, l, three_d, h) in &[
        (30, 0.1, true, 0.06),
        (25, 0.08, true, 0.05),
        (20, 0.1, false, 0.05),
    ] {
        let ps = rich_cloud(rng, n, l, three_d);
        let (o, idx) = brute_lists(&ps, h, None);
        row(m, &ps, &o, &idx, h);
        let (o2, idx2) = brute_lists(&ps, h, Some(rng));
        row(m, &ps, &o2, &idx2, h);
    }
    let mut cube = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let mut p = FluidParticle::new_2d(
                    crate::m_grid::fx(x) * 0.02,
                    crate::m_grid::fx(y) * 0.02,
                    0.001,
                );
                p.position.z = crate::m_grid::fx(z) * 0.02;
                p.density = 1000.0;
                cube.push(p);
            }
        }
    }
    let (o, idx) = brute_lists(&cube, 0.05, None);
    row(m, &cube, &o, &idx, 0.05);
    let small = rich_cloud(rng, 6, 0.04, true);
    let (o, idx) = brute_lists(&small, 0.05, None);
    for (bo, bi) in invalid_lists(&o, &idx, small.len()) {
        row(m, &small, &bo, &bi, 0.05);
    }
    for h in [0.0, f64::NAN, -0.05, f64::INFINITY] {
        row(m, &small, &o, &idx, h);
    }
    let mut odd = small.clone();
    odd[2].density = f64::NAN;
    odd[3].mass = f64::INFINITY;
    row(m, &odd, &o, &idx, 0.05);
    row(m, &[], &[0], &[], 0.05);
}

fn invert_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, a: [f64; 9]| {
        let ins: Vec<Tok> = a.iter().map(|&x| f(x)).collect();
        let out = match pravash::oracle::invert_3x3(&a) {
            Some(inv) => inv.iter().map(|&x| f(x)).collect(),
            None => vec![Tok::None],
        };
        m.row("oracle::invert_3x3", Class::B, &ins, &out);
    };
    for _ in 0..40 {
        let mut a = [0.0; 9];
        for x in &mut a {
            *x = rng.uniform(-2.0, 2.0);
        }
        row(m, a);
    }
    // Pivoting (a 0 and a tiny leading entry), the identity, a permutation, rank 2, the zero matrix, a uniformly tiny
    // matrix (singular by the absolute 1e-12 threshold), pivots either side of 1e-12, huge entries.
    row(m, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]);
    row(m, [1e-14, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
    row(m, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    row(m, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    row(m, [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0]);
    row(m, [0.0; 9]);
    row(m, [1e-13, 0.0, 0.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1e-13]);
    for d in [1e-12, 0.999_999e-12, 1.000_001e-12] {
        row(m, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d]);
    }
    row(m, [1e300, 2e300, 0.0, 1.0, 3.0, 0.5, 0.25, 1e-300, 4.0]);
    // Non-finite entries: hisab's dense substitution multiplies ±∞ by the unit vector's zeros (NaN).
    for k in 0..9 {
        let mut a = [2.0, 0.5, 0.1, 0.3, 3.0, 0.7, 0.2, 0.4, 5.0];
        a[k] = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY][k % 3];
        row(m, a);
    }
}

fn gradient_correction_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, c: [f64; 9], g: DVec3| {
        let mut ins: Vec<Tok> = c.iter().map(|&x| f(x)).collect();
        ins.extend([f(g.x), f(g.y), f(g.z)]);
        let r = apply_gradient_correction(&c, g);
        m.row(
            "apply_gradient_correction",
            Class::B,
            &ins,
            &[f(r.x), f(r.y), f(r.z)],
        );
    };
    row(
        m,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        DVec3::new(2.0, 3.0, 4.0),
    );
    let sp = specials();
    for k in 0..20 {
        let mut c = [0.0; 9];
        for x in &mut c {
            *x = rng.uniform(-3.0, 3.0);
        }
        let mut g = DVec3::new(
            rng.uniform(-50.0, 50.0),
            rng.uniform(-50.0, 50.0),
            rng.uniform(-50.0, 50.0),
        );
        if k >= 14 {
            c[k % 9] = sp[k % sp.len()];
            g.y = sp[(k * 3) % sp.len()];
        }
        row(m, c, g);
    }
}

fn phase_change_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, c: &PhaseChangeConfig, ps: &[FluidParticle]| {
        let mut ins = vec![
            f(c.transition_temperature),
            f(c.latent_heat),
            f(c.heat_capacity),
            i(c.phase_below),
            i(c.phase_above),
            f(0.01),
        ];
        ins.extend(ptoks(ps));
        let mut out = ps.to_vec();
        update_phase_change(&mut out, c, 0.01);
        let toks: Vec<Tok> = out.iter().flat_map(particle).collect();
        m.row("update_phase_change", Class::B, &ins, &toks);
    };
    // Every branch: below and above the phases (and a third), temperatures either side of T* by up to a full latent
    // heat's worth, progress from −0.5 to 1.5 and NaN, a NaN temperature.
    let water = PhaseChangeConfig::water_ice(0, 1);
    let sp = specials();
    let mut ps = Vec::new();
    for k in 0..60 {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        p.phase = [0, 1, 2][k % 3];
        p.temperature = 273.15 + rng.uniform(-200.0, 200.0);
        p.fuel = [0.0, 0.3, 0.97, 1.0, -0.5, 1.5, rng.uniform(0.0, 1.0)][k % 7];
        if k % 13 == 5 {
            p.fuel = f64::NAN;
        }
        if k % 17 == 3 {
            p.temperature = sp[k % sp.len()];
        }
        if k % 11 == 4 {
            p.temperature = 273.15;
        }
        ps.push(p);
    }
    row(m, &water, &ps);
    for (lh, cp) in [
        (0.0, 2090.0),
        (334_000.0, 0.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (-5.0, -5.0),
    ] {
        row(m, &PhaseChangeConfig::new(273.15, lh, cp, 0, 1), &ps);
    }
    row(
        m,
        &PhaseChangeConfig::new(373.15, 2.26e6, 4186.0, 1, 1),
        &ps,
    );
    row(
        m,
        &PhaseChangeConfig::new(f64::NAN, 1000.0, 10.0, 0, 2),
        &ps,
    );
    row(m, &PhaseChangeConfig::new(500.0, 1000.0, 10.0, 2, 0), &[]);
    // Exact ties, in binary-exact numbers (T* 0, L 10, c 2): the energy toward the other phase equal to the latent heat
    // still needed (heating from 0 and from half way, cooling), and the energy back equal to the stored heat. No RNG.
    let tie = PhaseChangeConfig::new(0.0, 10.0, 2.0, 0, 1);
    let mut ties = Vec::new();
    for &(phase, t, fuel) in &[
        (0u8, 5.0, 0.0),
        (0, 2.5, 0.5),
        (1, -5.0, 0.0),
        (0, -2.5, 0.5),
        (1, 2.5, 0.5),
    ] {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        p.phase = phase;
        p.temperature = t;
        p.fuel = fuel;
        ties.push(p);
    }
    row(m, &tie, &ties);
}

fn combustion_rows(m: &mut Module, rng: &mut Rng) {
    let sp = specials();
    let mut ps = Vec::new();
    for k in 0..40 {
        let mut p = FluidParticle::new_2d(0.0, 0.0, 1.0);
        p.temperature = [
            300.0,
            499.0,
            500.0,
            501.0,
            600.0,
            1000.0,
            rng.uniform(200.0, 900.0),
        ][k % 7];
        p.fuel = [1.0, 0.0, 0.001, -0.2, 0.5, rng.uniform(0.0, 1.0)][k % 6];
        if k % 9 == 4 {
            p.temperature = sp[k % sp.len()];
        }
        if k % 10 == 7 {
            p.fuel = sp[(k * 3) % sp.len()];
        }
        ps.push(p);
    }
    let mut fast = CombustionConfig::default();
    fast.reaction_rate = 1000.0;
    let mut nan_ign = CombustionConfig::default();
    nan_ign.ignition_temperature = f64::NAN;
    let mut negative = CombustionConfig::default();
    negative.reaction_rate = -150.0;
    negative.heat_release = -10.0;
    let configs = [CombustionConfig::default(), fast, nan_ign, negative];
    for (k, c) in configs.iter().enumerate() {
        let dt = [0.01, 1.0, 0.01, 0.01][k];
        let mut ins = vec![
            f(c.ignition_temperature),
            f(c.reaction_rate),
            f(c.heat_release),
            f(dt),
        ];
        ins.extend(ptoks(&ps));
        let mut a = ps.clone();
        update_reaction(&mut a, c, dt);
        m.row(
            "update_reaction",
            Class::B,
            &ins,
            &a.iter().flat_map(particle).collect::<Vec<_>>(),
        );
        let mut b = ps.clone();
        update_combustion(&mut b, c, dt);
        m.row(
            "update_combustion",
            Class::B,
            &ins,
            &b.iter().flat_map(particle).collect::<Vec<_>>(),
        );
    }
    for k in 0..30 {
        let c = configs[k % 4];
        let t = if k < 20 {
            rng.uniform(300.0, 800.0)
        } else {
            sp[k % sp.len()]
        };
        let fuel = [rng.uniform(0.0, 1.0), 0.0, -1.0, 1.0][k % 4];
        let dt = [0.01, 0.0, 1e6, -0.01][k % 4];
        let (cons, heat) = c.react(t, fuel, dt);
        m.row(
            "CombustionConfig::react",
            Class::B,
            &[
                f(c.ignition_temperature),
                f(c.reaction_rate),
                f(c.heat_release),
                f(t),
                f(fuel),
                f(dt),
            ],
            &[f(cons), f(heat)],
        );
    }
}

fn contact_angle_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, cfg: &FluidConfig, theta: f64, sigma: f64, ps: &[FluidParticle]| {
        let mut ins = config(cfg);
        ins.extend([f(theta), f(sigma)]);
        ins.extend(ptoks(ps));
        let mut out = ps.to_vec();
        apply_contact_angle(&mut out, cfg, theta, sigma);
        m.row(
            "apply_contact_angle",
            Class::C,
            &ins,
            &out.iter().flat_map(particle).collect::<Vec<_>>(),
        );
    };
    // Particles near each wall, in corners (two walls: the normal normalised), mid-domain, outside it.
    let mut near = Vec::new();
    for k in 0..30 {
        let mut p = FluidParticle::new_2d(rng.uniform(-0.1, 1.1), rng.uniform(-0.1, 1.1), 0.001);
        p.position.z = rng.uniform(-0.05, 0.55);
        if k % 5 == 0 {
            p.position.x = rng.uniform(0.0, 0.04);
            p.position.y = rng.uniform(0.96, 1.0);
        }
        p.acceleration = DVec3::new(rng.uniform(-1.0, 1.0), -9.81, 0.0);
        near.push(p);
    }
    let water = FluidConfig::water_2d();
    let mut cube = FluidConfig::water_2d();
    cube.bounds_max = DVec3::new(1.0, 1.0, 0.5);
    let q = std::f64::consts::FRAC_PI_4;
    for theta in [
        q,
        std::f64::consts::FRAC_PI_2,
        3.0 * q,
        0.0,
        std::f64::consts::PI,
        -1.0,
        7.0,
    ] {
        row(m, &water, theta, 0.07, &near);
        row(m, &cube, theta, 0.07, &near);
    }
    // Angles whose cosine glibc misrounds (an MPFR search over k·π/4096: k = 1729, 2171, 2554).
    for k in [1729u32, 2171, 2554] {
        row(
            m,
            &cube,
            std::f64::consts::PI * f64::from(k) / 4096.0,
            0.07,
            &near,
        );
    }
    // The early returns and odd configs: NaN θ or σ, h 0, a NaN bound (Rust's `hi <= lo` keeps that axis's walls), an
    // inverted box, σ negative.
    row(m, &cube, f64::NAN, 0.07, &near);
    row(m, &cube, q, f64::INFINITY, &near);
    row(m, &cube, q, -0.5, &near);
    let mut zero_h = cube;
    zero_h.smoothing_radius = 0.0;
    row(m, &zero_h, q, 0.07, &near);
    let mut nan_hi = cube;
    nan_hi.bounds_max.z = f64::NAN;
    row(m, &nan_hi, q, 0.07, &near);
    let mut inverted = cube;
    inverted.bounds_min.x = 2.0;
    row(m, &inverted, q, 0.07, &near);
    row(m, &cube, q, 0.07, &[]);
    // Particles exactly h from a wall (h 0.0625, so x − lo and hi − x are exact): `< h` leaves them out. No RNG.
    let mut exact = cube;
    exact.smoothing_radius = 0.0625;
    let at: Vec<FluidParticle> = [
        (0.0625, 0.5),
        (0.9375, 0.5),
        (0.5, 0.0625),
        (0.0625, 0.9375),
        (0.06, 0.5),
    ]
    .iter()
    .map(|&(x, y)| {
        let mut p = FluidParticle::new_2d(x, y, 0.001);
        p.position.z = 0.25;
        p
    })
    .collect();
    row(m, &exact, q, 0.07, &at);
}

fn secondary_tok(s: &SecondaryParticle) -> Vec<Tok> {
    vec![
        f(s.position.x),
        f(s.position.y),
        f(s.position.z),
        f(s.velocity.x),
        f(s.velocity.y),
        f(s.velocity.z),
        f(s.lifetime),
        Tok::E("SecondaryType", format!("{:?}", s.kind)),
    ]
}

fn secondary_rows(m: &mut Module, rng: &mut Rng) {
    let gen_row = |m: &mut Module, thr: f64, life: f64, ps: &[FluidParticle]| {
        let mut ins = vec![f(thr), f(life)];
        ins.extend(ptoks(ps));
        let s = generate_secondary_particles(ps, thr, life);
        let mut out = vec![Tok::V(s.len())];
        for x in &s {
            out.extend(secondary_tok(x));
        }
        m.row("generate_secondary_particles", Class::B, &ins, &out);
    };
    let sp = specials();
    let mut ps = rich_cloud(rng, 40, 1.0, true);
    ps[3].acceleration.x = f64::NAN;
    ps[7].acceleration.y = f64::INFINITY;
    ps[11].acceleration = DVec3::ZERO;
    for thr in [10.0, 30.0, 0.0, -1.0, f64::NAN, f64::INFINITY, sp[3]] {
        gen_row(m, thr, 2.0, &ps);
    }
    gen_row(m, 10.0, f64::NAN, &ps);
    gen_row(m, 10.0, 2.0, &[]);
    let upd_row = |m: &mut Module, g: DVec3, dt: f64, ss: &[SecondaryParticle]| {
        let mut ins = vec![f(g.x), f(g.y), f(g.z), f(dt), Tok::V(ss.len())];
        for x in ss {
            ins.extend(secondary_tok(x));
        }
        let mut v = ss.to_vec();
        update_secondary_particles(&mut v, g, dt);
        let mut out = vec![Tok::V(v.len())];
        for x in &v {
            out.extend(secondary_tok(x));
        }
        m.row("update_secondary_particles", Class::B, &ins, &out);
    };
    let mut ss = Vec::new();
    for k in 0..45 {
        ss.push(SecondaryParticle {
            position: DVec3::new(
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0),
            ),
            velocity: DVec3::new(
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ),
            lifetime: [
                0.02,
                0.01,
                0.005,
                1.0,
                0.0,
                -1.0,
                f64::NAN,
                rng.uniform(0.0, 0.05),
            ][k % 8],
            kind: [
                SecondaryType::Foam,
                SecondaryType::Spray,
                SecondaryType::Bubble,
            ][k % 3],
        });
    }
    let g = DVec3::new(0.0, -9.81, 0.0);
    for dt in [0.01, 0.0, 0.03, 1.0, -0.01, f64::NAN] {
        upd_row(m, g, dt, &ss);
    }
    upd_row(m, DVec3::new(f64::NAN, 1.0, -2.0), 0.01, &ss);
    upd_row(m, g, 0.01, &[]);
}

// ── M5 part a ───────────────────────────────────────────────────────────────────────

/// One solver row: `steps` steps of a fresh solver with σ and Verlet as given, stopping at the first error.
#[allow(clippy::too_many_arguments)]
fn solver_row(
    m: &mut Module,
    ps: &[FluidParticle],
    cfg: &FluidConfig,
    mu: f64,
    sigma: f64,
    verlet: bool,
    steps: usize,
) {
    let mut ins = config(cfg);
    ins.extend([f(mu), f(sigma), i(i64::from(verlet)), usz(steps)]);
    ins.extend(ptoks(ps));
    let mut solver = SphSolver::with_surface_tension(sigma);
    solver.use_verlet = verlet;
    let mut s = ps.to_vec();
    let mut out = None;
    for _ in 0..steps {
        if let Err(x) = solver.step(&mut s, cfg, mu) {
            out = Some(vec![err(&x)]);
            break;
        }
    }
    let out = out.unwrap_or_else(|| s.iter().flat_map(particle).collect());
    m.row("SphSolver::step", Class::S, &ins, &out);
}

fn trajectory_rows(
    m: &mut Module,
    ps: &[FluidParticle],
    cfg: &FluidConfig,
    mu: f64,
    sigma: f64,
    verlet: bool,
    frames: &[usize],
) {
    for &k in frames {
        solver_row(m, ps, cfg, mu, sigma, verlet, k);
    }
}

/// A disc of particles at spacing sp around (cx, cy), radius r, mass set for ρ₀ at its centre.
fn droplet(cx: f64, cy: f64, r: f64, sp: f64, h: f64) -> Vec<FluidParticle> {
    let n = (2.0 * r / sp).ceil() as i64;
    let mut ps = Vec::new();
    for y in -n..=n {
        for x in -n..=n {
            let (dx, dy) = (x as f64 * sp, y as f64 * sp);
            if dx * dx + dy * dy <= r * r {
                ps.push(FluidParticle::new_2d(cx + dx, cy + dy, 1.0));
            }
        }
    }
    let mid = ps
        .iter()
        .position(|p| p.position.x == cx && p.position.y == cy)
        .unwrap_or(0);
    let mass = 1000.0 / compute_density(mid, &ps, h);
    for p in &mut ps {
        p.mass = mass;
    }
    ps
}

fn solver_rows(m: &mut Module, rng: &mut Rng) {
    let frames = [1, 2, 10, 60, 300];
    let water = FluidConfig::water_2d();
    // A hydrostatic tank: a 10 × 6 block resting on the floor (density ρ₀ at its middle).
    let tank = water_block(10, 6, 0.025, 0.0125, 0.0125, 0.05);
    trajectory_rows(m, &tank, &water, 0.001, 0.0, false, &frames);
    // A dam break in the left corner, by symplectic Euler and by velocity Verlet.
    let dam = water_block(6, 10, 0.025, 0.0125, 0.0125, 0.05);
    trajectory_rows(m, &dam, &water, 0.001, 0.0, false, &frames);
    trajectory_rows(m, &dam, &water, 0.001, 0.0, true, &frames);
    // A droplet in zero gravity with surface tension (the CSF term), both integrators.
    let mut zero_g = FluidConfig::water_2d();
    zero_g.gravity = DVec3::ZERO;
    let drop = droplet(0.5, 0.5, 0.075, 0.025, 0.05);
    trajectory_rows(m, &drop, &zero_g, 0.01, 0.5, false, &[1, 2, 10, 60]);
    trajectory_rows(m, &drop, &zero_g, 0.01, 0.5, true, &[1, 2, 10, 60]);
    // A 3D cube in a closed box, viscous, with surface tension.
    let mut cube_cfg = FluidConfig::water_2d();
    cube_cfg.bounds_max = DVec3::new(0.5, 0.5, 0.5);
    cube_cfg.smoothing_radius = 0.06;
    let mut cube = Vec::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let mut p = FluidParticle::new_2d(
                    0.1 + crate::m_grid::fx(x) * 0.03,
                    0.1 + crate::m_grid::fx(y) * 0.03,
                    0.02,
                );
                p.position.z = 0.1 + crate::m_grid::fx(z) * 0.03;
                cube.push(p);
            }
        }
    }
    trajectory_rows(m, &cube, &cube_cfg, 0.5, 0.2, false, &[1, 10, 60]);
    // A random splash through every wall (z off the plane), Euler and Verlet.
    let mut splash = cloud(rng, 24, 0.9, 0.05, false, 0.01);
    for (j, p) in splash.iter_mut().enumerate() {
        p.velocity = DVec3::new(rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0), 0.0);
        if j % 6 == 0 {
            p.position.z = [0.2, -0.2, 0.0, -0.0][(j / 6) % 4];
        }
    }
    trajectory_rows(m, &splash, &water, 0.01, 0.0, false, &[1, 5, 30]);
    trajectory_rows(m, &splash, &water, 0.01, 0.0, true, &[1, 5, 30]);
    // Far from the origin, where the f32 keys and probes lose cells (D3): a block at x ≈ 1e6.
    let mut far_cfg = FluidConfig::water_2d();
    far_cfg.bounds_min = DVec3::new(-2e6, -1.0, 0.0);
    far_cfg.bounds_max = DVec3::new(2e6, 2.0, 0.0);
    let mut far = water_block(6, 4, 0.025, 0.0125, 0.0125, 0.05);
    for p in &mut far {
        p.position.x += 1e6;
    }
    trajectory_rows(m, &far, &far_cfg, 0.001, 0.0, false, &[1, 10]);
    // The Rust tests' single particles, n = 0, σ negative and NaN (no CSF), viscosity 0 and negative.
    let mut one = FluidParticle::new_2d(0.5, 0.5, 0.01);
    one.density = 1000.0;
    solver_row(m, &[one], &water, 0.001, 0.0, false, 1);
    solver_row(m, &[one], &water, 0.0, 0.0, true, 20);
    let mut below = FluidParticle::new_2d(0.5, -0.1, 0.01);
    below.density = 1000.0;
    solver_row(m, &[below], &water, 0.001, 0.0, false, 1);
    solver_row(m, &[], &water, 0.001, 0.0, false, 1);
    for sigma in [-0.5, f64::NAN, f64::INFINITY] {
        solver_row(m, &drop, &zero_g, 0.01, sigma, false, 2);
    }
    solver_row(m, &dam, &water, 0.0, 0.0, false, 5);
    solver_row(m, &dam, &water, -0.01, 0.0, false, 5);
    // The errors: the config's checks, h beyond f32 either way (its `as f32` is ∞ or 0), and divergence.
    let small = cloud(rng, 5, 0.2, 0.3, false, 0.01);
    let mut bad = FluidConfig::water_2d();
    bad.dt = 0.0;
    solver_row(m, &small, &bad, 0.001, 0.0, false, 1);
    for h in [1e39, 1e-46] {
        let mut c = FluidConfig::water_2d();
        c.smoothing_radius = h;
        solver_row(m, &small, &c, 0.001, 0.0, false, 1);
    }
    let mut nan = small.clone();
    nan[2].position.y = f64::NAN;
    solver_row(m, &nan, &water, 0.001, 0.0, false, 1);
    let mut inf = small.clone();
    inf[4].mass = f64::INFINITY;
    inf[4].position = inf[1].position;
    inf[4].position.x += 0.01;
    solver_row(m, &inf, &water, 0.001, 0.0, true, 1);
    // One component diverging alone (an ∞ x velocity, through viscosity, into a neighbour's x acceleration). No RNG.
    let mut one_axis = small.clone();
    one_axis[0].position = one_axis[1].position;
    one_axis[0].position.x += 0.01;
    one_axis[0].velocity = DVec3::new(f64::INFINITY, 0.0, 0.0);
    solver_row(m, &one_axis, &water, 0.001, 0.0, false, 1);
}

fn adaptive_dt_rows(m: &mut Module, rng: &mut Rng) {
    let row = |m: &mut Module, ps: &[FluidParticle], v: [f64; 5]| {
        let mut ins: Vec<Tok> = v.iter().map(|&x| f(x)).collect();
        ins.extend(ptoks(ps));
        let dt = SphSolver::adaptive_dt(ps, v[0], v[1], v[2], v[3], v[4]);
        m.row("SphSolver::adaptive_dt", Class::B, &ins, &[f(dt)]);
    };
    let mut ps = rich_cloud(rng, 20, 0.5, true);
    let base = [0.05, 0.001, 0.4, 1e-5, 0.01];
    row(m, &ps, base);
    // Each term binding: fast particles, a large acceleration, a large viscosity; at rest (none binds: f64::MAX).
    let mut fast = ps.clone();
    fast[3].velocity.x = 300.0;
    row(m, &fast, base);
    let mut hard = ps.clone();
    hard[5].acceleration.y = -1e6;
    row(m, &hard, base);
    row(m, &ps, [0.05, 50.0, 0.4, 1e-9, 0.01]);
    let mut rest = ps.clone();
    for p in &mut rest {
        p.velocity = DVec3::ZERO;
        p.acceleration = DVec3::ZERO;
    }
    row(m, &rest, [0.05, 0.0, 0.4, 1e-5, 0.01]);
    row(m, &rest, [0.05, 0.0, 0.4, 1e-5, 1e300]);
    // Inverted bounds (swapped), each early return, NaN speeds and accelerations, n = 0.
    row(m, &ps, [0.05, 0.001, 0.4, 0.1, 0.01]);
    for v in [
        [0.0, 0.001, 0.4, 1e-5, 0.01],
        [f64::NAN, 0.001, 0.4, 1e-5, 0.01],
        [0.05, 0.001, 0.0, 1e-5, 0.01],
        [0.05, 0.001, f64::INFINITY, 1e-5, 0.01],
        [0.05, 0.001, 0.4, f64::NAN, 0.01],
        [0.05, 0.001, 0.4, 1e-5, f64::NEG_INFINITY],
        [0.05, f64::NAN, 0.4, 1e-5, 0.01],
        [0.05, -1.0, 0.4, 1e-5, 0.01],
    ] {
        row(m, &ps, v);
    }
    ps[2].velocity.y = f64::NAN;
    ps[7].acceleration.x = f64::INFINITY;
    row(m, &ps, base);
    row(m, &[], base);
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
    // Part c's rows, from their own seed.
    let mut rng = Rng::new(0x0_5f4c);
    force_rows(&mut m, &mut rng);
    step_rows(&mut m, &mut rng);
    block_rows(&mut m, &mut rng);
    split_rows(&mut m, &mut rng);
    merge_rows(&mut m, &mut rng);
    // Part d's rows, from their own seed.
    let mut rng = Rng::new(0x0_5f4d);
    viscoelastic_rows(&mut m, &mut rng);
    heat_rows(&mut m, &mut rng);
    delta_rows(&mut m, &mut rng);
    implicit_rows(&mut m, &mut rng);
    mls_rows(&mut m, &mut rng);
    invert_rows(&mut m, &mut rng);
    gradient_correction_rows(&mut m, &mut rng);
    phase_change_rows(&mut m, &mut rng);
    combustion_rows(&mut m, &mut rng);
    contact_angle_rows(&mut m, &mut rng);
    secondary_rows(&mut m, &mut rng);
    // M5 part a's rows, from their own seed.
    let mut rng = Rng::new(0x0_5f5a);
    solver_rows(&mut m, &mut rng);
    adaptive_dt_rows(&mut m, &mut rng);
    m
}
