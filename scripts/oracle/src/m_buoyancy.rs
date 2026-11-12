//! `buoyancy.tsv` — `rust-old/src/buoyancy.rs` (M1 part 4): the two forces, the terminal velocity, the Reynolds
//! number and the flow regime. Arithmetic rows are class B (Rust's order, expected 0 ulp); `terminal_velocity` and
//! `reynolds_number` give `f:` or `E:<Variant>`; `classify_flow` gives `e:FlowRegime.<Variant>`.

use pravash::buoyancy::{
    buoyancy_force, classify_flow, drag_force, reynolds_number, terminal_velocity,
};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, Tok, e, err, f};

fn res(r: pravash::error::Result<f64>) -> Vec<Tok> {
    match r {
        Ok(x) => vec![f(x)],
        Err(x) => vec![err(&x)],
    }
}

/// `n`-argument cases: each argument over the specials with the others at `base`, then `k` random draws.
fn cases(
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
        for &x in &[-1e300, 1e300, 1e-300, -2.5, 0.5] {
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
    let mut m = Module::new("buoyancy");
    let mut rng = Rng::new(0xb0_7a_9c);
    let wide = |r: &mut Rng, _: usize| {
        if r.unit() < 0.7 {
            r.uniform(-50.0, 2000.0)
        } else {
            r.signed_log_uniform(-150.0, 150.0)
        }
    };
    for c in cases(&[1000.0, 9.81, 0.001], &mut rng, 300, &wide) {
        m.row(
            "buoyancy_force",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2])],
            &[f(buoyancy_force(c[0], c[1], c[2]))],
        );
    }
    for c in cases(&[1.225, 30.0, 0.47, 0.5], &mut rng, 300, &wide) {
        m.row(
            "drag_force",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &[f(drag_force(c[0], c[1], c[2], c[3]))],
        );
    }
    for c in cases(&[1.0, 9.81, 1.225, 0.47, 0.01], &mut rng, 300, &wide) {
        m.row(
            "terminal_velocity",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3]), f(c[4])],
            &res(terminal_velocity(c[0], c[1], c[2], c[3], c[4])),
        );
    }
    for c in cases(&[1000.0, 1.0, 0.1, 0.001], &mut rng, 300, &wide) {
        m.row(
            "reynolds_number",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &res(reynolds_number(c[0], c[1], c[2], c[3])),
        );
    }
    // Both thresholds and their neighbours, both signs, the specials, a sweep.
    let mut res_ = specials();
    for t in [2300.0f64, 4000.0] {
        for x in [
            t,
            f64::from_bits(t.to_bits() - 1),
            f64::from_bits(t.to_bits() + 1),
        ] {
            res_.extend([x, -x]);
        }
    }
    for _ in 0..200 {
        res_.push(rng.uniform(-6000.0, 6000.0));
    }
    for _ in 0..50 {
        res_.push(rng.signed_log_uniform(-10.0, 10.0));
    }
    for &x in &res_ {
        m.row(
            "classify_flow",
            Class::A,
            &[f(x)],
            &[e("FlowRegime", classify_flow(x))],
        );
    }
    m
}
