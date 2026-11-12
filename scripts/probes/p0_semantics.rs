//! p0_semantics.rs — M0 step 4: Rust's value for each case p0_semantics.cyr probes, as the oracle
//! computes it (rustc 1.99.1, std → host glibc). Inputs pass through `black_box`, so LLVM cannot fold
//! a call at compile time.
//!
//!   rustc -O --edition 2024 -o build/p0_rust scripts/probes/p0_semantics.rs && ./build/p0_rust > build/p0_rust.tsv
//!
//! Prints: case <TAB> type <TAB> bits.

use std::hint::black_box as bb;

fn f(case: &str, x: f64) {
    println!("{case}\tf64\t{:#x}", x.to_bits());
}

/// The SPH cell key as `SphSolver::build_neighbors` computes it.
fn key(p: f64, h: f64) -> i32 {
    let h_f32 = bb(h) as f32;
    let inv_cs = 1.0 / h_f32;
    let pos = bb(p) as f32;
    (pos * inv_cs).floor() as i32
}

fn main() {
    let nan = bb(f64::NAN);
    f("min(nan,1)", nan.min(bb(1.0)));
    f("min(1,nan)", bb(1.0f64).min(nan));
    f("max(nan,1)", nan.max(bb(1.0)));
    f("max(1,nan)", bb(1.0f64).max(nan));
    f("min(+0,-0)", bb(0.0f64).min(bb(-0.0)));
    f("max(+0,-0)", bb(0.0f64).max(bb(-0.0)));
    f("clamp(nan,0,1)", nan.clamp(bb(0.0), bb(1.0)));
    f("clamp(2,0,1)", bb(2.0f64).clamp(bb(0.0), bb(1.0)));

    f("floor(-0.5)", bb(-0.5f64).floor());
    f("floor(-0)", bb(-0.0f64).floor());
    f("ceil(-0.5)", bb(-0.5f64).ceil());
    f("sqrt(2)", bb(2.0f64).sqrt());
    f("abs(-0)", bb(-0.0f64).abs());

    for (name, x) in [("exp(-0.7)", -0.7f64), ("exp(1)", 1.0), ("exp(5.3)", 5.3)] {
        f(name, bb(x).exp());
    }
    for (name, x) in [("ln(0.3)", 0.3f64), ("ln(2)", 2.0), ("ln(10)", 10.0)] {
        f(name, bb(x).ln());
    }
    for (name, x) in [("log10(0.25)", 0.25f64), ("log10(3)", 3.0)] {
        f(name, bb(x).log10());
    }

    f("pow(2,0.5)", bb(2.0f64).powf(bb(0.5)));
    f("pow(1.5,2.3)", bb(1.5f64).powf(bb(2.3)));
    f("pow(0.9,0.333)", bb(0.9f64).powf(bb(0.333)));
    f("atan2(1,-1)", bb(1.0f64).atan2(bb(-1.0)));
    f("atan2(0.3,0.7)", bb(0.3f64).atan2(bb(0.7)));
    f("atan2(-2,-0.5)", bb(-2.0f64).atan2(bb(-0.5)));
    f("hypot(3,4)", bb(3.0f64).hypot(bb(4.0)));
    f("hypot(0.001,2.5)", bb(0.001f64).hypot(bb(2.5)));
    f("hypot(1.1,2.2)", bb(1.1f64).hypot(bb(2.2)));
    f("cbrt(2)", bb(2.0f64).cbrt());
    f("cbrt(0.001)", bb(0.001f64).cbrt());
    f("cbrt(27)", bb(27.0f64).cbrt());

    for (s, c, x) in [("sin(0.5)", "cos(0.5)", 0.5f64), ("sin(1)", "cos(1)", 1.0), ("sin(3)", "cos(3)", 3.0)] {
        f(s, bb(x).sin());
        f(c, bb(x).cos());
    }

    f("to_radians(30)", bb(30.0f64).to_radians());
    f("PI", std::f64::consts::PI);
    f("powi(1.1,2)", bb(1.1f64).powi(2));
    f("powi(1.1,8)", bb(1.1f64).powi(8));

    println!("f32(0.1)\tf32\t{:#x}", (bb(0.1f64) as f32).to_bits());
    for (name, p) in [
        ("key(1.37)", 1.37f64),
        ("key(0.475)", 0.475),
        ("key(12345.678)", 12345.678),
        ("key(-0.0001)", -0.0001),
        ("key(1e10)", 1e10),
        ("key(0.15)", 0.15),
        ("key(nan)", f64::NAN),
    ] {
        println!("{name}\ti32\t{:#x}", key(p, 0.05));
    }
    println!("i32(1e10)\ti32\t{:#x}", bb(1e10f64) as i32);

    for (name, x) in [
        ("usize(floor(-3.7))", -3.7f64),
        ("usize(floor(7.9))", 7.9),
        ("usize(1e29)", 1e29),
        ("usize(0)", 0.0),
    ] {
        println!("{name}\tu64\t{:#x}", bb(x).floor() as usize);
    }
    println!("usize(nan)\tu64\t{:#x}", nan as usize);
    println!("usize(inf)\tu64\t{:#x}", bb(f64::INFINITY) as usize);
    println!("u32(5e9)\tu32\t{:#x}", bb(5e9f64) as u32);
}
