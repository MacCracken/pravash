//! `math.tsv` — what Rust **std** returns for every f64 / f32 operation pravash 1.3.0's library
//! code uses: the reference for the M0 step 8 baselines (port plan §4 step 5, §7.1).
//!
//! The calls, counted in `rust-old/src` outside test code (2026-11-11): max 128, is_finite 101,
//! sqrt 46, min 37, abs 34, clamp 22, floor 21, ln 8, powi 6 (exponent 2 or 8, always a literal),
//! powf 6, exp 5, cos 4, ceil 3, is_nan 3, hypot 3, sin 2, sin_cos 2, cbrt 2, atan2 2,
//! to_radians 1, log10 1. Then the conversions the probe (port-parity §1) found need helpers:
//! `as usize` (19 sites), `as u32` (2), `as f32` and the SPH cell key's f32 floor `as i32` (3).
//!
//! Where each result comes from (rustc 1.99.1, x86_64-unknown-linux-gnu):
//! - **glibc's libm** (dynamic): sin, cos (and `sin_cos`, which LLVM lowers to glibc's `sincos`),
//!   atan2, hypot, exp, ln (`log`), log10, powf (`pow`). These rows are glibc's bits (`glibc.rs`):
//!   the header records the glibc, and `--check` refuses another.
//! - **compiler_builtins, linked into the binary**: cbrt. rustc resolves `f64::cbrt` to its own
//!   `compiler_builtins::math::libm_math::cbrt` (`nm` shows `cbrt` defined in the binary and not
//!   imported), so these rows are platform-free. `validate.py` proves every one correctly rounded
//!   (port-parity §3).
//! - **inline code**: sqrt (`sqrtsd`), abs, floor, ceil (`roundsd`), min, max, clamp, is_nan,
//!   is_finite, to_radians (one multiply by the folded π/180), powi with a literal exponent (LLVM
//!   expands `x.powi(2)` to `x·x` and `x.powi(8)` to three squarings), and the saturating casts.
//!
//! Every input passes through `black_box`, so each row is a run-time call, never a fold. The
//! exponent of `powi` is *not* black-boxed: pravash always writes it as a literal, so the rows
//! record the expansion its code actually runs (`f64::powi2`, `f64::powi8`).
//!
//! Classes:
//! - **A**: exact on both sides: sqrt, abs, floor, ceil, min, max, clamp, is_nan, is_finite,
//!   the casts and the cell key;
//! - **B**: arithmetic: to_radians, powi2, powi8;
//! - **C**: transcendental: sin_cos, sin, cos, atan2, hypot, exp, ln, log10, powf (glibc), cbrt
//!   (compiler_builtins).
//!
//! Inputs are the special values, the ranges pravash feeds each function, and seeded sweeps.
//! Adapted from impetus's `m_math.rs`.

use std::hint::black_box;

use crate::grids::{Rng, pow2, specials};
use crate::tsv::{Class, Module, f, i};

const PI: f64 = core::f64::consts::PI;

fn unary(m: &mut Module, name: &str, class: Class, fun: fn(f64) -> f64, xs: &[f64]) {
    for &x in xs {
        let y = fun(black_box(x));
        m.row(name, class, &[f(x)], &[f(y)]);
    }
}

fn binary(m: &mut Module, name: &str, class: Class, fun: fn(f64, f64) -> f64, xy: &[(f64, f64)]) {
    for &(x, y) in xy {
        let r = fun(black_box(x), black_box(y));
        m.row(name, class, &[f(x), f(y)], &[f(r)]);
    }
}

/// Every ordered pair of `xs`.
fn grid(xs: &[f64]) -> Vec<(f64, f64)> {
    xs.iter()
        .flat_map(|&x| xs.iter().map(move |&y| (x, y)))
        .collect()
}

/// Specials, ±2^k at every 16th binade, and a wide signed log-uniform sweep.
fn wide(seed: u64, n: usize) -> Vec<f64> {
    let mut v = specials();
    for k in (-1074..=1023).step_by(16) {
        let x = pow2(k);
        v.push(x);
        v.push(-x);
    }
    let mut rng = Rng::new(seed);
    v.extend((0..n).map(|_| rng.signed_log_uniform(-300.0, 300.0)));
    v
}

fn sweep(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

fn log_sweep(seed: u64, n: usize, lo_exp10: f64, hi_exp10: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| rng.log_uniform(lo_exp10, hi_exp10))
        .collect()
}

fn pairs(seed: u64, n: usize, lo_exp10: f64, hi_exp10: f64) -> Vec<(f64, f64)> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let x = rng.signed_log_uniform(lo_exp10, hi_exp10);
            (x, rng.signed_log_uniform(lo_exp10, hi_exp10))
        })
        .collect()
}

fn uniform_pairs(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let x = rng.uniform(lo, hi);
            (x, rng.uniform(lo, hi))
        })
        .collect()
}

fn powi2(x: f64) -> f64 {
    x.powi(2)
}

fn powi8(x: f64) -> f64 {
    x.powi(8)
}

/// `x as usize` as the 64-bit word (a value ≥ 2^63 reads negative as an `i:` token).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn as_usize(x: f64) -> i64 {
    (black_box(x) as usize) as i64
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn as_u32(x: f64) -> i64 {
    i64::from(black_box(x) as u32)
}

#[allow(clippy::cast_possible_truncation)]
fn as_f32(x: f64) -> f64 {
    f64::from(black_box(x) as f32)
}

/// The doubles either side of every integer in `lo..=hi`, the halves, and the integers.
fn around_integers(lo: i32, hi: i32) -> Vec<f64> {
    let mut v = Vec::new();
    for k in lo..=hi {
        let x = f64::from(k);
        v.push(x);
        v.push(x + 0.5);
        v.push(f64::from_bits(x.to_bits().wrapping_add(1)));
        v.push(f64::from_bits(x.to_bits().wrapping_sub(1)));
    }
    v
}

#[allow(clippy::too_many_lines)] // one grid per function, in the module comment's order
pub fn emit() -> Module {
    let mut m = Module::new("math");

    // ── exact (class A) ──
    let mut xs = wide(0x5157, 300);
    xs.extend(log_sweep(0x5158, 300, -40.0, 20.0));
    xs.extend((1..=100).map(|k| f64::from(k * k)));
    xs.extend(sweep(0x5159, 200, 0.0, 1e4));
    unary(&mut m, "f64::sqrt", Class::A, f64::sqrt, &xs);
    unary(&mut m, "f64::abs", Class::A, f64::abs, &wide(0xab5, 100));

    // floor / ceil: grid cell indices (position / dx, position / h), the neighbour hash and the
    // particle-block sizes. Ties and the doubles either side of an integer, then sweeps.
    let mut halves = around_integers(-20, 20);
    halves.extend([
        0.499_999_999_999_999_94,
        -0.499_999_999_999_999_94,
        4_503_599_627_370_495.5,
        -4_503_599_627_370_495.5,
        4_503_599_627_370_497.0,
    ]);
    halves.extend(sweep(0x7a11, 300, -1e4, 1e4));
    halves.extend(sweep(0x7a13, 200, -2.0, 1100.0));
    halves.extend(wide(0x7a12, 100));
    unary(&mut m, "f64::floor", Class::A, f64::floor, &halves);
    unary(&mut m, "f64::ceil", Class::A, f64::ceil, &halves);

    // min / max: every special pair (NaN on either side, ±0 both ways, ±∞), equal pairs, a value
    // against 0.0 both ways (pravash writes `x.max(0.0)` and `x.max(1e-6)` all over), and random
    // pairs.
    let mut mm = grid(&specials());
    let mut rng = Rng::new(0x3a3a);
    for _ in 0..60 {
        let x = rng.signed_log_uniform(-30.0, 30.0);
        mm.push((x, x));
        mm.push((x, 0.0));
        mm.push((0.0, x));
        mm.push((x, f64::NAN));
        mm.push((x, 1e-6));
        mm.push((x, 1e-12));
    }
    mm.extend(pairs(0x3a3b, 300, -300.0, 300.0));
    binary(&mut m, "f64::max", Class::A, f64::max, &mm);
    binary(&mut m, "f64::min", Class::A, f64::min, &mm);

    // clamp: valid bounds only (min ≤ max, neither NaN; Rust panics otherwise), with the probes at
    // and around each bound, ±0, NaN and a random value. pravash's own bounds first.
    let mut rng = Rng::new(0xc1a2);
    let mut bounds = vec![
        (0.0, 0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1024.0),
        (-1.5, 1.5),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.0),
    ];
    for _ in 0..80 {
        let a = rng.signed_log_uniform(-10.0, 10.0);
        let c = rng.signed_log_uniform(-10.0, 10.0);
        bounds.push(if a <= c { (a, c) } else { (c, a) });
    }
    for &(lo, hi) in &bounds {
        let probes = [
            lo,
            hi,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            rng.signed_log_uniform(-10.0, 10.0),
        ];
        for &x in &probes {
            let r = black_box(x).clamp(black_box(lo), black_box(hi));
            m.row("f64::clamp", Class::A, &[f(x), f(lo), f(hi)], &[f(r)]);
        }
    }

    let mut preds = specials();
    preds.extend([
        f64::from_bits(0xfff8_0000_0000_0000), // -NaN
        f64::from_bits(0x7ff0_0000_0000_0001), // signalling-pattern NaN
        -f64::MAX,
        -f64::from_bits(1),
    ]);
    preds.extend(log_sweep(0xf1f1, 20, -300.0, 300.0));
    for &x in &preds {
        m.row(
            "f64::is_finite",
            Class::A,
            &[f(x)],
            &[i(i64::from(black_box(x).is_finite()))],
        );
        m.row(
            "f64::is_nan",
            Class::A,
            &[f(x)],
            &[i(i64::from(black_box(x).is_nan()))],
        );
    }

    // `as usize` (grid indices after floor and clamping, the VOF sub-step count, the particle
    // block size) and `as u32` (the Morton code): truncation, saturation at both ends, NaN → 0.
    let mut casts = specials();
    casts.extend(around_integers(-5, 5));
    casts.extend(sweep(0xca51, 200, -1e3, 1e5));
    casts.extend([
        4_294_967_295.0,
        4_294_967_295.5,
        4_294_967_296.0,
        9_223_372_036_854_775_807.0,  // rounds to 2^63
        9_223_372_036_854_777_856.0,  // 2^63 + 2048
        18_446_744_073_709_549_568.0, // the largest double below 2^64
        18_446_744_073_709_551_616.0, // 2^64
    ]);
    casts.extend(wide(0xca52, 100));
    for &x in &casts {
        m.row("f64::as_usize", Class::A, &[f(x)], &[i(as_usize(x))]);
        m.row("f64::as_u32", Class::A, &[f(x)], &[i(as_u32(x))]);
    }

    // `as f32`: the SPH positions and the compute packing. Round to nearest even, overflow to ±∞,
    // subnormals; the exact halfway doubles between neighbouring f32 values.
    let mut narrow = wide(0xf32a, 300);
    narrow.extend(sweep(0xf32b, 300, -2.0, 2.0));
    narrow.extend(sweep(0xf32c, 200, -1e4, 1e4));
    let mut rng = Rng::new(0xf32d);
    for _ in 0..200 {
        // A random f32, then the double halfway to its successor: a tie the cast must round to even.
        #[allow(clippy::cast_possible_truncation)]
        let a = rng.signed_log_uniform(-30.0, 30.0) as f32;
        let b = f32::from_bits(a.to_bits() + 1);
        let mid = (f64::from(a) + f64::from(b)) * 0.5;
        narrow.extend([mid, f64::from(a)]);
    }
    narrow.extend([
        f64::from(f32::MAX),
        f64::from(f32::MAX) * (1.0 + f64::EPSILON),
        3.402_823_576e38, // past f32::MAX by more than half an ulp: +∞
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::from_bits(1)),
        f64::from(f32::from_bits(1)) * 0.5, // a tie with zero
    ]);
    unary(&mut m, "f64::as_f32", Class::A, as_f32, &narrow);

    // The SPH cell key, through the oracle's own hook: floor(p_f32 · (1 / h_f32)) as i32 per axis.
    // Smoothing radii pravash uses, positions across a box, on and next to cell boundaries, and
    // the large coordinates where f32 loses precision (the roadmap's defect, D3).
    let hs = [0.01, 0.025, 0.037_3, 0.05, 0.1, 1.0];
    let mut rng = Rng::new(0x5ce1);
    for &h in &hs {
        let mut ps = sweep(rng.int(1, 1 << 40).unsigned_abs(), 120, -2.0, 2.0);
        for k in -40..=40 {
            let x = f64::from(k) * h;
            ps.extend([
                x,
                f64::from_bits(x.to_bits().wrapping_add(1)),
                f64::from_bits(x.to_bits().wrapping_sub(1)),
            ]);
        }
        ps.extend(log_sweep(rng.int(1, 1 << 40).unsigned_abs(), 40, 2.0, 8.0));
        ps.extend([
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e10,
            -1e10,
        ]);
        #[allow(clippy::cast_possible_truncation)]
        let h_f32 = black_box(h) as f32;
        for &p in &ps {
            #[allow(clippy::cast_possible_truncation)]
            let p32 = black_box(p) as f32;
            let key = pravash::oracle::sph_cell_key([p32, p32, p32], h_f32);
            m.row(
                "oracle::sph_cell_key",
                Class::A,
                &[f(p), f(h)],
                &[i(i64::from(key[0]))],
            );
        }
    }

    // ── arithmetic (class B) ──
    // to_radians: Drucker–Prager's friction angle (degrees, [0, 90)) and the contact angle.
    let mut deg = specials();
    deg.extend((-720..=720).step_by(15).map(f64::from));
    deg.extend(sweep(0xde61, 300, -720.0, 720.0));
    deg.extend(sweep(0xde63, 200, 0.0, 90.0));
    deg.extend(wide(0xde62, 100));
    unary(&mut m, "f64::to_radians", Class::B, f64::to_radians, &deg);

    // powi with pravash's literal exponents: squared distances and speeds (2), the Wendland /
    // poly6 kernel term (8).
    let mut pw = specials();
    pw.extend(sweep(0x9101, 300, -100.0, 100.0));
    pw.extend(sweep(0x9104, 300, -1.5, 1.5));
    pw.extend(log_sweep(0x9102, 200, -150.0, 150.0));
    pw.extend(
        log_sweep(0x9103, 100, -150.0, 150.0)
            .into_iter()
            .map(|x| -x),
    );
    unary(&mut m, "f64::powi2", Class::B, powi2, &pw);
    unary(&mut m, "f64::powi8", Class::B, powi8, &pw);

    // ── transcendental (class C) ──
    // sin / cos / sin_cos: the DST eigenvalues cos(π (k + 1) / (n + 1)), computed in grid.rs's
    // order for the interior sizes the solver runs; contact and split angles in one turn; MPM's
    // SVD angles (atan2 results in [−π, π]); tiny, accumulated and huge angles.
    let mut ang = specials();
    for n in [6_i32, 14, 30, 62, 126, 254] {
        for k in 0..n {
            ang.push(PI * f64::from(k + 1) / f64::from(n + 1));
        }
    }
    ang.extend(sweep(0x5101, 1000, -2.0 * PI, 2.0 * PI));
    ang.extend(sweep(0x5106, 300, 0.0, PI));
    ang.extend(sweep(0x5102, 300, -1e3, 1e3));
    ang.extend(log_sweep(0x5103, 200, -12.0, 0.0));
    ang.extend(log_sweep(0x5104, 200, 0.0, 15.0));
    ang.extend(wide(0x5105, 100));
    for &x in &ang {
        let (s, c) = black_box(x).sin_cos();
        m.row("f64::sin_cos", Class::C, &[f(x)], &[f(s), f(c)]);
    }
    unary(&mut m, "f64::sin", Class::C, f64::sin, &ang);
    unary(&mut m, "f64::cos", Class::C, f64::cos, &ang);

    // atan2(y, x): MPM's 2×2 polar decomposition takes `g.atan2(f)` and `h.atan2(e)` of
    // deformation-gradient sums; then every quadrant, the special grid (±0 and ±∞ in each slot)
    // and wide pairs.
    let mut at = grid(&specials());
    at.extend(uniform_pairs(0xa7a1, 600, -1.0, 1.0));
    at.extend(uniform_pairs(0xa7a4, 400, -10.0, 10.0));
    at.extend(pairs(0xa7a2, 300, -10.0, 10.0));
    at.extend(pairs(0xa7a3, 200, -300.0, 300.0));
    binary(&mut m, "f64::atan2", Class::C, f64::atan2, &at);

    // hypot: MPM's SVD (`e.hypot(h)`, `f.hypot(g)`, the deviatoric norm). Pythagorean triples
    // (exact answers), unit- and gradient-scale sweeps, the special grid, and pairs spanning the
    // range (overflow and underflow avoidance).
    let mut hy = grid(&specials());
    for (a, b) in [
        (3.0, 4.0),
        (5.0, 12.0),
        (8.0, 15.0),
        (7.0, 24.0),
        (20.0, 21.0),
    ] {
        hy.push((a, b));
        hy.push((-a, b));
        hy.push((b, -a));
    }
    hy.extend(uniform_pairs(0x4791, 600, -1.0, 1.0));
    hy.extend(uniform_pairs(0x4792, 400, -10.0, 10.0));
    hy.extend(pairs(0x4793, 400, -300.0, 300.0));
    hy.extend(pairs(0x4794, 200, -160.0, -150.0));
    binary(&mut m, "f64::hypot", Class::C, f64::hypot, &hy);

    // exp: Lamb–Oseen's exp(−r²/core) (≤ 0), the Kolmogorov scale exp(¾ ln ν − ¼ ln ε), MPM's
    // Hencky exp(h) (around 0), phase change's exp(−rate·dt); the full finite range and its edges.
    let mut ex = specials();
    ex.extend(sweep(0xe001, 600, -50.0, 0.0));
    ex.extend(sweep(0xe002, 300, -745.2, 0.0));
    ex.extend(sweep(0xe003, 200, -1e-3, 0.0));
    ex.extend(sweep(0xe005, 400, -5.0, 5.0));
    ex.extend(sweep(0xe006, 200, -30.0, 10.0));
    ex.extend(sweep(0xe004, 200, -745.2, 709.8));
    ex.extend((-745..=709).step_by(7).map(f64::from));
    ex.extend([
        709.782_712_893_384,
        709.782_712_893_384_1,
        -708.396_418_532_264_1,
        -745.133_219_101_941_1,
        -745.133_219_101_941_2,
        1e-300,
        -1e-300,
    ]);
    unary(&mut m, "f64::exp", Class::C, f64::exp, &ex);

    // ln: Tait's ln(ρ/ρ₀) near 1, MPM's ln of singular values and of J = det F, the Kolmogorov
    // scale's ln ν and ln ε; then the whole positive range, subnormals and the negatives (NaN).
    let mut lg = specials();
    lg.extend(sweep(0x1001, 600, 0.5, 2.0));
    lg.extend(sweep(0x1002, 200, 0.999, 1.001));
    lg.extend(log_sweep(0x1003, 400, -12.0, 2.0));
    lg.extend(log_sweep(0x1004, 300, -300.0, 300.0));
    lg.extend(log_sweep(0x1005, 50, -323.0, -308.0));
    lg.extend(sweep(0x1006, 50, -10.0, 0.0));
    unary(&mut m, "f64::ln", Class::C, f64::ln, &lg);

    // log10: the bridge's 20 log10(u_rms / v_ref); powers of ten (parsed, so exact decimals),
    // then the whole range.
    let mut l10 = specials();
    for k in -30..=30 {
        if let Ok(x) = format!("1e{k}").parse::<f64>() {
            l10.push(x);
        }
    }
    l10.extend(log_sweep(0x1a01, 400, -6.0, 6.0));
    l10.extend(log_sweep(0x1a02, 300, -300.0, 300.0));
    l10.extend(sweep(0x1a03, 30, -10.0, 0.0));
    unary(&mut m, "f64::log10", Class::C, f64::log10, &l10);

    // powf: Tait's (ρ/ρ₀)^γ (base near 1, γ = 7 and others), the non-Newtonian sr^(n − 1)
    // (shear rate over decades, exponent in [−0.9, 2]), shallow water's damping^(dt / 0.001),
    // the bridge's 10^(b / (t − c)); then random positive pairs, the special grid, and negative
    // bases with integer exponents.
    let mut pf: Vec<(f64, f64)> = Vec::new();
    let mut rng = Rng::new(0x9001);
    for g in [1.0, 1.4, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0] {
        for _ in 0..40 {
            pf.push((rng.uniform(0.5, 2.0), g));
            pf.push((rng.uniform(0.98, 1.02), g - 1.0));
        }
    }
    for _ in 0..300 {
        pf.push((rng.log_uniform(-6.0, 6.0), rng.uniform(-0.9, 2.0)));
    }
    for _ in 0..200 {
        pf.push((rng.uniform(0.9, 1.0), rng.log_uniform(-1.0, 2.0)));
    }
    for _ in 0..200 {
        pf.push((10.0, rng.uniform(-8.0, 8.0)));
    }
    for _ in 0..300 {
        let x = rng.log_uniform(-30.0, 30.0);
        pf.push((x, rng.uniform(-10.0, 10.0)));
    }
    pf.extend(grid(&specials()));
    for _ in 0..50 {
        let x = -rng.uniform(0.1, 10.0);
        #[allow(clippy::cast_precision_loss)]
        let y = rng.int(-5, 5) as f64;
        pf.push((x, y));
    }
    binary(&mut m, "f64::powf", Class::C, f64::powf, &pf);

    // cbrt: Manning friction's cbrt(depth) (depth ≥ the dry threshold, up to ~100 m), perfect
    // cubes, then the whole range.
    let mut cb = specials();
    cb.extend((-20..=20).map(|k| f64::from(k * k * k)));
    cb.extend(log_sweep(0xcb01, 400, -6.0, 2.0));
    cb.extend(wide(0xcb02, 300));
    unary(&mut m, "f64::cbrt", Class::C, f64::cbrt, &cb);

    m
}
