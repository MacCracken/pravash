//! `compute.tsv` — `rust-old/src/compute.rs` (M7 part a, seed 0x70_c0_0a).
//!
//! Tokens: `<soa>` is a `ParticleSoa`'s 18 `v:` columns (`m_json::soa_toks`, the phase as `i:`); `<packed>` is a
//! `PackedParticles`: `i:` count, then `v:` position_mass, velocity and density_pressure, each f32 widened (`f:`).
//!
//! Rows:
//! - `PackedParticles::from_soa` takes `<soa>` and gives `<packed>` (class A: `as f32` rounds to nearest even);
//! - `PackedParticles::unpack_density_pressure`, `::unpack_velocity` and `::unpack_positions` take `<packed>` and
//!   `<soa>` and give the SoA after (class A);
//! - `SphKernelParams::default`, `GridKernelParams::default` and `ShallowKernelParams::default` give the fields (f32
//!   widened, u32 as `i:`); each `::from_json` decodes a serde_json document to them or `E:Deserialize` (class A);
//! - `KernelDerivatives::dpoly6_dr2` takes (r², h) and `::dwendland_c2_dr` (r, h) (class B); `::deos_drho` takes k
//!   (class A); `::dtait_drho` takes (ρ, ρ₀, c, γ) (class C: powf, the corrected oracle where glibc misrounds);
//! - `KernelDerivatives::autodiff_gradient` takes `i:<f>`, one of the catalogue's functions below, and `v:<n>` inputs,
//!   and gives `v:<n>` the gradient (class D: the port's tape is hisab 3.4.1's, whose sweep sums in another order,
//!   port-parity §4.8). The catalogue, which tests/parity.tcyr and validate.py implement on their own tapes:
//!   0. x0·x1 + (x2·x2)·x0 − x1/x2;
//!   1. sin(x0)·exp(x1) + x2;
//!   2. the poly6 kernel of r = x0 at h = x1: (315/(64π))/h⁹ · ((h² − r²)·(h² − r²))·(h² − r²);
//!   3. (ln(x0)·x1 − cos(x2)) + x0^2.5;
//!   4. −(x0 − 3)·x1 + x0/x1;
//!   5. x0^1.5 · x1 (a pow whose upstream gradient isn't 1, so the sweep's (g·n)·x^(n−1) order shows).

use hisab::autodiff::{Tape, Var};
use pravash::compute::{
    GridKernelParams, KernelDerivatives, PackedParticles, ShallowKernelParams, SphKernelParams,
};
use pravash::{FluidParticle, ParticleSoa};
use serde::de::DeserializeOwned;

use crate::grids::{Rng, specials};
use crate::m_grid::usz;
use crate::m_json::soa_toks;
use crate::tsv::{Class, Module, Tok, f, s};

fn f32s(out: &mut Vec<Tok>, v: &[f32]) {
    out.push(Tok::V(v.len()));
    out.extend(v.iter().map(|&x| f(f64::from(x))));
}

fn packed(p: &PackedParticles) -> Vec<Tok> {
    let mut out = vec![Tok::I(i64::from(p.count))];
    f32s(&mut out, &p.position_mass);
    f32s(&mut out, &p.velocity);
    f32s(&mut out, &p.density_pressure);
    out
}

fn sph_toks(p: &SphKernelParams) -> Vec<Tok> {
    let mut out: Vec<Tok> = [
        p.smoothing_radius,
        p.rest_density,
        p.gas_constant,
        p.viscosity,
    ]
    .iter()
    .map(|&x| f(f64::from(x)))
    .collect();
    out.extend(p.gravity.iter().map(|&x| f(f64::from(x))));
    out.push(f(f64::from(p.dt)));
    out.push(Tok::I(i64::from(p.particle_count)));
    out
}

fn grid_toks(p: &GridKernelParams) -> Vec<Tok> {
    let mut out = vec![Tok::I(i64::from(p.nx)), Tok::I(i64::from(p.ny))];
    out.extend(
        [p.dx, p.dt, p.viscosity, p.buoyancy_alpha, p.ambient_density]
            .iter()
            .map(|&x| f(f64::from(x))),
    );
    out
}

fn shallow_toks(p: &ShallowKernelParams) -> Vec<Tok> {
    let mut out = vec![Tok::I(i64::from(p.nx)), Tok::I(i64::from(p.ny))];
    out.extend(
        [p.dx, p.dt, p.gravity, p.damping, p.dry_threshold]
            .iter()
            .map(|&x| f(f64::from(x))),
    );
    out
}

/// A seeded SoA of n particles, the specials sprinkled in, and values whose f32 rounding is a tie or a subnormal.
fn soa_of(rng: &mut Rng, n: usize) -> ParticleSoa {
    let mut ps = Vec::new();
    let sp = specials();
    for i in 0..n {
        let mut p = FluidParticle::new(
            hisab::DVec3::new(
                rng.uniform(-10.0, 10.0),
                rng.signed_log_uniform(-40.0, 40.0),
                rng.unit(),
            ),
            rng.log_uniform(-5.0, 1.0),
        );
        p.velocity = hisab::DVec3::new(rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), -0.0);
        p.density = rng.uniform(900.0, 1100.0);
        p.pressure = rng.signed_log_uniform(-3.0, 6.0);
        if i % 3 == 1 {
            p.position.x = sp[i % sp.len()];
            p.velocity.y = 16_777_217.0;
            p.density = 1e-46;
        }
        ps.push(p);
    }
    ParticleSoa::from_aos(&ps)
}

fn packing_rows(m: &mut Module, rng: &mut Rng) {
    let mut cases: Vec<ParticleSoa> = vec![ParticleSoa::new()];
    for n in [1, 2, 3, 5, 8, 13] {
        cases.push(soa_of(rng, n));
    }
    // Mismatched columns (the Rust test's short pressure, a short mass, a long position).
    let mut short = soa_of(rng, 4);
    short.pressure.truncate(1);
    cases.push(short);
    let mut short_m = soa_of(rng, 5);
    short_m.mass.truncate(3);
    cases.push(short_m);
    let mut long = soa_of(rng, 3);
    long.pos_x.push(7.0);
    cases.push(long);
    for soa in &cases {
        let p = PackedParticles::from_soa(soa);
        m.row(
            "PackedParticles::from_soa",
            Class::A,
            &soa_toks(soa),
            &packed(&p),
        );
    }
    // Unpacks: written buffers, short buffers (the Rust test's), a count above the buffers and the SoA, SoA columns
    // shorter than the count, and an empty buffer.
    for k in 0..12 {
        let n = 1 + k % 6;
        let src = soa_of(rng, n);
        let mut p = PackedParticles::from_soa(&src);
        for x in p.density_pressure.iter_mut() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *x = rng.uniform(-2000.0, 2000.0) as f32;
            }
        }
        for x in p.velocity.iter_mut() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *x = rng.uniform(-5.0, 5.0) as f32;
            }
        }
        for x in p.position_mass.iter_mut() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *x = rng.uniform(-1.0, 1.0) as f32;
            }
        }
        match k % 4 {
            1 => {
                p.density_pressure.truncate(3);
                p.velocity.truncate(5);
                p.position_mass.clear();
            }
            2 => p.count += 3,
            3 => p.density_pressure.push(9.0),
            _ => {}
        }
        let mut dst = soa_of(rng, if k % 5 == 4 { n.saturating_sub(1) } else { n });
        if k % 6 == 5 {
            dst.vel_z.truncate(1);
            dst.pos_y.clear();
        }
        for (name, which) in [
            ("PackedParticles::unpack_density_pressure", 0),
            ("PackedParticles::unpack_velocity", 1),
            ("PackedParticles::unpack_positions", 2),
        ] {
            let mut ins = packed(&p);
            ins.extend(soa_toks(&dst));
            let mut out = dst.clone();
            match which {
                0 => p.unpack_density_pressure(&mut out),
                1 => p.unpack_velocity(&mut out),
                _ => p.unpack_positions(&mut out),
            }
            m.row(name, Class::A, &ins, &soa_toks(&out));
        }
    }
}

/// Decimals for an f32 field at the places a double rounding (decimal → f64 → f32) can differ from serde_json's single
/// one: f32 midpoints written exactly, their shortest f64 digits (which round to the midpoint in f64 but lie off it),
/// the subnormal midpoints, the overflow threshold 2^128 − 2^103, and integers past 2^53 beside a midpoint.
const F32_DECIMALS: &[&str] = &[
    // 1 + 2^-24, the midpoint above 1 (ties to even: 1), its shortest f64 digits (above it) and the next below
    "1.000000059604644775390625",
    "1.0000000596046448",
    "1.0000000596046447",
    "1.0000000596046447753906250000000000001",
    "1.0000000596046447753906249999999999999",
    "-1.0000000596046448",
    // 1 + 3·2^-24, whose tie goes up
    "1.000000178813934326171875",
    "1.0000001788139343",
    "1.0000001788139342",
    // an integer mantissa with an exponent
    "1000000059604644775390625e-24",
    "1000000059604644775390626e-24",
    "100000005960464477539062.4e-23",
    // 2^-150 (half the smallest subnormal) and 3·2^-150
    "7.00649232162408535461864791644958065640130970938257885878534141944895541342930300743319094181060791015625e-46",
    "7.006492321624086e-46",
    "7.006492321624085e-46",
    "-7.006492321624086e-46",
    "2.101947696487225606385594374934874196920392912814773657635602425834686624028790902229957282543182373046875e-45",
    "2.1019476964872256e-45",
    "2.1019476964872255e-45",
    // the overflow threshold: exact (to ∞, refused), just under, f32::MAX, and negative
    "340282356779733661637539395458142568448.0",
    "340282356779733661637539395458142568447.9",
    "3.4028235677973362e38",
    "3.4028234663852886e38",
    "-3.4028235677973366e38",
    // midpoints in [2^25, 2^26) whose decimals end in 0 (even neighbour below, then above), and beside one
    "33554450.0",
    "33554470.0",
    "33554450.000000001",
    "33554449.999999999",
    // integers (`u64 as f32` / `i64 as f32`)
    "1152921573326323713",
    "1152921573326323712",
    "-1152921573326323713",
    "9223372036854775807",
    "-9223372036854775808",
    "16777217",
    "-0.0",
];

fn json_row<T: DeserializeOwned>(
    m: &mut Module,
    name: &str,
    doc: &str,
    toks: &dyn Fn(&T) -> Vec<Tok>,
) {
    let out = match serde_json::from_str::<T>(doc) {
        Ok(v) => toks(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    m.row(name, Class::A, &[s(doc)], &out);
}

#[allow(clippy::cast_possible_truncation)]
fn params_rows(m: &mut Module, rng: &mut Rng) {
    m.row(
        "SphKernelParams::default",
        Class::A,
        &[],
        &sph_toks(&SphKernelParams::default()),
    );
    m.row(
        "GridKernelParams::default",
        Class::A,
        &[],
        &grid_toks(&GridKernelParams::default()),
    );
    m.row(
        "ShallowKernelParams::default",
        Class::A,
        &[],
        &shallow_toks(&ShallowKernelParams::default()),
    );
    let mut sdocs = Vec::new();
    for k in 0..12 {
        let mut p = SphKernelParams::default();
        p.smoothing_radius = rng.log_uniform(-3.0, 0.0) as f32;
        p.rest_density = rng.uniform(500.0, 1500.0) as f32;
        p.gas_constant = rng.uniform(100.0, 3000.0) as f32;
        p.viscosity = rng.log_uniform(-4.0, -1.0) as f32;
        p.gravity = [0.0, -9.81, rng.uniform(-1.0, 1.0) as f32];
        p.dt = rng.log_uniform(-4.0, -2.0) as f32;
        p.particle_count = u32::try_from(rng.int(0, 100_000)).unwrap_or(0);
        if k == 10 {
            p.dt = f32::NAN;
        }
        if k == 11 {
            p.particle_count = u32::MAX;
        }
        sdocs.push(serde_json::to_string(&p).unwrap_or_default());
    }
    let mut b0 = SphKernelParams::default();
    b0.smoothing_radius = 0.05;
    let base = serde_json::to_string(&b0).unwrap_or_default();
    for (from, to) in [
        (r#""particle_count":0"#, r#""particle_count":-1"#),
        (r#""particle_count":0"#, r#""particle_count":4294967296"#),
        (r#""particle_count":0"#, r#""particle_count":1.0"#),
        (r#""dt":0.0"#, r#""dt":1e300"#),
        (r#""dt":0.0"#, r#""dt":1e400"#),
        (r#""dt":0.0"#, r#""dt":0.1"#),
        (r#""dt":0.0"#, r#""dt":3.4028235677973366e38"#),
        (r#""dt":0.0"#, r#""dt":1e-46"#),
        (r#""dt":0.0"#, r#""dt":7"#),
        (r#""gravity":[0.0,0.0,0.0]"#, r#""gravity":[0.0,0.0]"#),
        (
            r#""gravity":[0.0,0.0,0.0]"#,
            r#""gravity":[0.0,0.0,0.0,0.0]"#,
        ),
        (r#""gravity":[0.0,0.0,0.0]"#, r#""gravity":null"#),
        (r#","dt":0.0"#, ""),
        (r#""dt":0.0"#, r#""dt":0.0,"extra":1"#),
    ] {
        sdocs.push(base.replacen(from, to, 1));
    }
    // serde_json's f32 read (`float_roundtrip`) rounds the decimal to f32 once and refuses one that rounds to ±∞; the
    // port's stand-in (ledger L15) rounds the f64 and settles an f64 that lands on an f32 midpoint from the text.
    let mut f32_cases: Vec<String> = F32_DECIMALS.iter().map(|c| (*c).to_string()).collect();
    let tie = "1.000000059604644775390625";
    f32_cases.push(format!("{tie}{}1", "0".repeat(900)));
    f32_cases.push(format!("{tie}{}", "0".repeat(900)));
    f32_cases.push(format!("0.{}1000000059604644775390625e61", "0".repeat(60)));
    for c in &f32_cases {
        sdocs.push(base.replacen(r#""dt":0.0"#, &format!(r#""dt":{c}"#), 1));
    }
    // The text is found by the number's document-order position: an unknown field's numbers and a string holding
    // digits and an escaped quote come first, and the gravity array holds another midpoint.
    sdocs.push(base.replacen(
        r#""dt":0.0"#,
        r#""extra":["1.0\"5",1.0000000596046447,{"k":-2e0}],"dt":1.0000000596046448"#,
        1,
    ));
    sdocs.push(base.replacen(
        r#""gravity":[0.0,0.0,0.0]"#,
        r#""gravity":[1.0000000596046447,1.0000000596046448,-1.0000000596046448]"#,
        1,
    ));
    sdocs.push("[1.0000000596046448,1.0000000596046447,2000,0.001,[0,1.0000000596046448,0],1.0000000596046447,100]".to_string());
    sdocs.push("[0.05,1000,2000,0.001,[0,-9.81,0],0.001,100]".to_string());
    sdocs.push("[0.05,1000,2000,0.001,[0,-9.81,0],0.001]".to_string());
    sdocs.push("null".to_string());
    sdocs.push("{}".to_string());
    for doc in &sdocs {
        json_row(m, "SphKernelParams::from_json", doc, &sph_toks);
    }
    let mut gdocs = Vec::new();
    for _ in 0..8 {
        let mut p = GridKernelParams::default();
        p.nx = u32::try_from(rng.int(0, 512)).unwrap_or(0);
        p.ny = u32::try_from(rng.int(0, 512)).unwrap_or(0);
        p.dx = rng.log_uniform(-3.0, 0.0) as f32;
        p.dt = rng.log_uniform(-3.0, -1.0) as f32;
        p.viscosity = rng.log_uniform(-6.0, -2.0) as f32;
        p.buoyancy_alpha = rng.uniform(0.0, 2.0) as f32;
        p.ambient_density = rng.uniform(0.5, 1.5) as f32;
        gdocs.push(serde_json::to_string(&p).unwrap_or_default());
    }
    let gbase = serde_json::to_string(&GridKernelParams::default()).unwrap_or_default();
    for (from, to) in [
        (r#""nx":0"#, r#""nx":-3"#),
        (r#""nx":0"#, r#""nx":64.0"#),
        (r#""dx":0.0"#, r#""dx":"0.1""#),
        (r#""dx":0.0"#, r#""dx":0.30000001192092896"#),
        (r#""dx":0.0"#, r#""dx":-1.0000001788139343"#),
        (r#","ambient_density":0.0"#, ""),
    ] {
        gdocs.push(gbase.replacen(from, to, 1));
    }
    gdocs.push("[16,16,0.1,0.01,0.001,1,1.2]".to_string());
    for doc in &gdocs {
        json_row(m, "GridKernelParams::from_json", doc, &grid_toks);
    }
    let mut hdocs = Vec::new();
    for _ in 0..8 {
        let mut p = ShallowKernelParams::default();
        p.nx = u32::try_from(rng.int(0, 512)).unwrap_or(0);
        p.ny = u32::try_from(rng.int(0, 512)).unwrap_or(0);
        p.dx = rng.log_uniform(-2.0, 1.0) as f32;
        p.dt = rng.log_uniform(-3.0, -1.0) as f32;
        p.gravity = 9.81;
        p.damping = rng.unit() as f32;
        p.dry_threshold = rng.log_uniform(-6.0, -2.0) as f32;
        hdocs.push(serde_json::to_string(&p).unwrap_or_default());
    }
    let hbase = serde_json::to_string(&ShallowKernelParams::default()).unwrap_or_default();
    for (from, to) in [
        (r#""ny":0"#, r#""ny":4294967295"#),
        (r#""ny":0"#, r#""ny":null"#),
        (r#""damping":0.0"#, r#""damping":-1e39"#),
        (r#""damping":0.0"#, r#""damping":1.0000000596046448"#),
        (r#""dx":0.0"#, r#""dx":1152921573326323713"#),
    ] {
        hdocs.push(hbase.replacen(from, to, 1));
    }
    for doc in &hdocs {
        json_row(m, "ShallowKernelParams::from_json", doc, &shallow_toks);
    }
}

fn derivative_rows(m: &mut Module, rng: &mut Rng) {
    // dpoly6/dr² and dWendland/dr over h (the specials, negatives, the SPH range) and r across [0, 1.2h].
    let mut hs = specials();
    hs.extend([-0.1, 1e-3, 0.0125, 0.05, 0.1, 1.0, 3.0, 1e3, 1e40, 1e-40]);
    for &h in &hs {
        let hh = if h.is_finite() && h > 0.0 { h } else { 1.0 };
        let mut rs: Vec<f64> = [0.0, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0, 1.0001, 1.2]
            .iter()
            .map(|t| t * hh)
            .collect();
        for _ in 0..4 {
            rs.push(rng.uniform(0.0, 1.2) * hh);
        }
        rs.extend([-0.0, -0.1 * hh, f64::NAN, f64::INFINITY]);
        for &r in &rs {
            m.row(
                "KernelDerivatives::dpoly6_dr2",
                Class::B,
                &[f(r * r), f(h)],
                &[f(KernelDerivatives::dpoly6_dr2(r * r, h))],
            );
            m.row(
                "KernelDerivatives::dwendland_c2_dr",
                Class::B,
                &[f(r), f(h)],
                &[f(KernelDerivatives::dwendland_c2_dr(r, h))],
            );
        }
    }
    for &k in &specials() {
        m.row(
            "KernelDerivatives::deos_drho",
            Class::A,
            &[f(k)],
            &[f(KernelDerivatives::deos_drho(k))],
        );
    }
    m.row(
        "KernelDerivatives::deos_drho",
        Class::A,
        &[f(2000.0)],
        &[f(KernelDerivatives::deos_drho(2000.0))],
    );
    // dTait/dρ over physical ranges, γ through 0 and 1, ρ₀ under the 1e-10 floor, and the specials in each place.
    let mut cases: Vec<[f64; 4]> = vec![
        [1000.0, 1000.0, 10.0, 0.0],
        [1000.0, 1000.0, 10.0, 7.0],
        [1000.0, 1000.0, 10.0, 1.0],
        [1100.0, 1e-12, 10.0, 7.0],
        [1100.0, -5.0, 10.0, 7.0],
    ];
    for a in 0..4 {
        for &x in &specials() {
            let mut c = [1050.0, 1000.0, 20.0, 7.0];
            c[a] = x;
            cases.push(c);
        }
    }
    for _ in 0..200 {
        cases.push([
            rng.uniform(800.0, 1200.0),
            rng.uniform(900.0, 1100.0),
            rng.log_uniform(0.0, 3.0),
            rng.uniform(0.0, 8.0),
        ]);
    }
    for c in &cases {
        m.row(
            "KernelDerivatives::dtait_drho",
            Class::C,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &[f(KernelDerivatives::dtait_drho(c[0], c[1], c[2], c[3]))],
        );
    }
}

fn ad0(t: &mut Tape, x: &[Var]) -> Var {
    let a = t.mul(x[0], x[1]);
    let b = t.mul(x[2], x[2]);
    let c = t.mul(b, x[0]);
    let d = t.div(x[1], x[2]);
    let e = t.add(a, c);
    t.sub(e, d)
}

fn ad1(t: &mut Tape, x: &[Var]) -> Var {
    let s = t.sin(x[0]);
    let e = t.exp(x[1]);
    let p = t.mul(s, e);
    t.add(p, x[2])
}

fn ad2(t: &mut Tape, x: &[Var]) -> Var {
    let h2 = t.mul(x[1], x[1]);
    let r2 = t.mul(x[0], x[0]);
    let d = t.sub(h2, r2);
    let d2 = t.mul(d, d);
    let d3 = t.mul(d2, d);
    let h3 = t.mul(h2, x[1]);
    let h6 = t.mul(h3, h3);
    let h9 = t.mul(h6, h3);
    let k = t.constant(315.0 / (64.0 * std::f64::consts::PI));
    let c = t.div(k, h9);
    t.mul(c, d3)
}

fn ad3(t: &mut Tape, x: &[Var]) -> Var {
    let l = t.ln(x[0]);
    let a = t.mul(l, x[1]);
    let c = t.cos(x[2]);
    let b = t.sub(a, c);
    let p = t.powf(x[0], 2.5);
    t.add(b, p)
}

fn ad4(t: &mut Tape, x: &[Var]) -> Var {
    let three = t.constant(3.0);
    let a = t.sub(x[0], three);
    let n = t.neg(a);
    let b = t.mul(n, x[1]);
    let c = t.div(x[0], x[1]);
    t.add(b, c)
}

fn ad5(t: &mut Tape, x: &[Var]) -> Var {
    let p = t.powf(x[0], 1.5);
    t.mul(p, x[1])
}

fn autodiff_rows(m: &mut Module, rng: &mut Rng) {
    type F = fn(&mut Tape, &[Var]) -> Var;
    let fs: [(F, usize); 6] = [(ad0, 3), (ad1, 3), (ad2, 2), (ad3, 3), (ad4, 2), (ad5, 2)];
    for (id, (fun, n)) in fs.iter().enumerate() {
        for k in 0..24 {
            let x: Vec<f64> = (0..*n)
                .map(|j| match (id, j) {
                    (2, 0) => rng.uniform(0.0, 0.05),
                    (2, _) => rng.uniform(0.05, 0.1),
                    (3 | 5, 0) => rng.uniform(0.1, 4.0),
                    (_, 1) if id == 4 => rng.uniform(0.5, 3.0),
                    (_, 2) if id == 0 => {
                        rng.uniform(0.5, 2.0) * if k % 2 == 0 { 1.0 } else { -1.0 }
                    }
                    _ => rng.uniform(-2.0, 2.0),
                })
                .collect();
            let g = KernelDerivatives::autodiff_gradient(fun, &x);
            let mut ins = vec![usz(id), Tok::V(x.len())];
            ins.extend(x.iter().map(|&v| f(v)));
            let mut out = vec![Tok::V(g.len())];
            out.extend(g.iter().map(|&v| f(v)));
            m.row("KernelDerivatives::autodiff_gradient", Class::D, &ins, &out);
        }
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("compute");
    let mut rng = Rng::new(0x70_c0_0a);
    packing_rows(&mut m, &mut rng);
    params_rows(&mut m, &mut rng);
    derivative_rows(&mut m, &mut rng);
    autodiff_rows(&mut m, &mut rng);
    m
}
