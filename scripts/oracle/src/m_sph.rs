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

use hisab::DVec3;
use pravash::FluidParticle;
use pravash::sph::{
    NonNewtonianViscosity, batch_distance_squared, batch_poly6, equation_of_state,
    equation_of_state_tait, kernel_poly6, kernel_spiky_grad, kernel_viscosity_laplacian,
    kernel_wendland_c2, kernel_wendland_c2_grad, kernel_wendland_c4, kernel_wendland_c4_grad,
    max_speed, total_kinetic_energy,
};

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

pub fn emit() -> Module {
    let mut m = Module::new("sph");
    let mut rng = Rng::new(0x0_5f4a);
    kernel_rows(&mut m, &mut rng);
    eos_rows(&mut m, &mut rng);
    viscosity_rows(&mut m, &mut rng);
    batch_rows(&mut m, &mut rng);
    total_rows(&mut m, &mut rng);
    m
}
