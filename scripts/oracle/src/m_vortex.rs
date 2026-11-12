//! `vortex.tsv` — `rust-old/src/vortex.rs` (M1 part 4). `lamb_oseen_velocity` (exp) and `kolmogorov_scale` (ln,
//! ln, exp) are class C, gated at 0 ulp against the corrected oracle (ADR 0004); the rest is class B. `enstrophy`
//! takes `v:<n>` then the n values, then dx.

use pravash::vortex::{
    enstrophy, kolmogorov_scale, lamb_oseen_velocity, rankine_velocity, vorticity_2d,
};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, Tok, f};

fn grid(
    base: &[f64],
    rng: &mut Rng,
    k: usize,
    draw: &dyn Fn(&mut Rng, usize) -> f64,
) -> Vec<Vec<f64>> {
    let mut out = vec![base.to_vec()];
    for a in 0..base.len() {
        for &x in &specials() {
            let mut c = base.to_vec();
            c[a] = x;
            out.push(c);
        }
    }
    for _ in 0..k {
        out.push((0..base.len()).map(|a| draw(rng, a)).collect());
    }
    out
}

pub fn emit() -> Module {
    let mut m = Module::new("vortex");
    let mut rng = Rng::new(0x5_0e7);
    let wide = |r: &mut Rng, _: usize| r.signed_log_uniform(-100.0, 100.0);
    for c in grid(&[1.5, -0.25], &mut rng, 200, &wide) {
        m.row(
            "vorticity_2d",
            Class::B,
            &[f(c[0]), f(c[1])],
            &[f(vorticity_2d(c[0], c[1]))],
        );
    }
    // Lamb–Oseen: Γ, r, ν, t over physical ranges (r around the core √(4νt)), the specials, r ≤ 0 and νt ≤ 0.
    let lo = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-10.0, 10.0),
        1 => r.log_uniform(-4.0, 1.0),
        2 => r.log_uniform(-7.0, -1.0),
        _ => r.uniform(0.0, 100.0),
    };
    let mut lcases = grid(&[1.0, 0.1, 0.001, 1.0], &mut rng, 700, &lo);
    lcases.extend([
        vec![1.0, 0.1, 0.0, 1.0],
        vec![1.0, 0.1, 0.001, 0.0],
        vec![1.0, 0.1, -0.001, 1.0],
    ]);
    for c in &lcases {
        m.row(
            "lamb_oseen_velocity",
            Class::C,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &[f(lamb_oseen_velocity(c[0], c[1], c[2], c[3]))],
        );
    }
    let rk = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-10.0, 10.0),
        _ => r.uniform(-0.1, 2.0),
    };
    let mut rcases = grid(&[1.0, 0.05, 0.1], &mut rng, 300, &rk);
    rcases.extend([
        vec![1.0, 0.1, 0.1],
        vec![2.0, 0.1, 0.0],
        vec![1.0, 0.0, 0.1],
    ]);
    for c in &rcases {
        m.row(
            "rankine_velocity",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2])],
            &[f(rankine_velocity(c[0], c[1], c[2]))],
        );
    }
    // Enstrophy: empty, single specials, and random fields of up to 64 values (sum order matters).
    let mut fields: Vec<(Vec<f64>, f64)> =
        vec![(vec![], 0.1), (vec![-0.0], 1.0), (vec![0.0, -0.0], 1.0)];
    for &x in &specials() {
        fields.push((vec![1.0, x, 2.0], 0.5));
        fields.push((vec![1.0, 2.0], x));
    }
    for _ in 0..150 {
        let n = usize::try_from(rng.int(0, 64)).unwrap_or(0);
        let v: Vec<f64> = (0..n).map(|_| rng.uniform(-20.0, 20.0)).collect();
        fields.push((v, rng.log_uniform(-3.0, 0.0)));
    }
    for (v, dx) in &fields {
        let mut ins = vec![Tok::V(v.len())];
        ins.extend(v.iter().map(|&x| f(x)));
        ins.push(f(*dx));
        m.row("enstrophy", Class::B, &ins, &[f(enstrophy(v, *dx))]);
    }
    let kz = |r: &mut Rng, a: usize| match a {
        0 => r.log_uniform(-7.0, -2.0),
        _ => r.log_uniform(-6.0, 3.0),
    };
    for c in grid(&[0.000001, 0.1], &mut rng, 800, &kz) {
        m.row(
            "kolmogorov_scale",
            Class::C,
            &[f(c[0]), f(c[1])],
            &[f(kolmogorov_scale(c[0], c[1]))],
        );
    }
    m
}
