//! Shared input grids.
//!
//! Every input here must be reproducible bit for bit on any machine, so the inputs of every
//! row are the same wherever the generator runs. Only correctly rounded or exact operations
//! are used: + − × ÷, `floor`, `str::parse::<f64>` (correctly rounded, in core) and bit
//! construction. **Never a transcendental std f64 method** — those call the platform's
//! glibc libm, which is what the `math` module *records*, not what a grid may depend on.
//! Log-uniform values are built from a binary exponent and a mantissa, without `10^x`.
//! Copied from impetus's oracle (which took it from kimiya's) unchanged in behaviour.

/// Special values every float function is probed with.
pub fn specials() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1), // smallest subnormal
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

/// 2^k exactly, for k in −1074 ..= 1023 (subnormal powers included).
pub fn pow2(k: i32) -> f64 {
    assert!((-1074..=1023).contains(&k), "pow2({k}) out of range");
    if k >= -1022 {
        let biased = u64::try_from(k + 1023).unwrap_or(0);
        f64::from_bits(biased << 52)
    } else {
        let shift = u32::try_from(k + 1074).unwrap_or(0);
        f64::from_bits(1_u64 << shift)
    }
}

/// Deterministic xorshift64* — fixed seeds keep the sweeps identical across runs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, 1): the top 53 bits as a fraction.
    pub fn unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let top = (self.next_u64() >> 11) as f64;
        top * (1.0 / 9_007_199_254_740_992.0) // 2^-53
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform integer in [lo, hi].
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        #[allow(clippy::cast_sign_loss)]
        let span = (hi - lo + 1) as u64;
        #[allow(clippy::cast_possible_wrap)]
        let off = (self.next_u64() % span) as i64;
        lo + off
    }

    /// Roughly 10^u, u uniform in [lo_exp10, hi_exp10): a binary exponent drawn uniformly
    /// over the matching binades, times a uniform mantissa in [1, 2). Log-uniform across
    /// binades, linear within one; built without a transcendental (module comment).
    pub fn log_uniform(&mut self, lo_exp10: f64, hi_exp10: f64) -> f64 {
        const LOG2_10: f64 = core::f64::consts::LOG2_10;
        #[allow(clippy::cast_possible_truncation)]
        let lo = (lo_exp10 * LOG2_10).floor() as i64;
        #[allow(clippy::cast_possible_truncation)]
        let hi = (hi_exp10 * LOG2_10).floor() as i64 - 1;
        let k = self.int(lo.max(-1074), hi.min(1023));
        let mant = 1.0 + self.unit();
        let k = i32::try_from(k).unwrap_or(0);
        // A subnormal 2^k times the mantissa rounds once, by IEEE rules: still deterministic.
        mant * pow2(k)
    }

    /// As `log_uniform`, with a random sign.
    pub fn signed_log_uniform(&mut self, lo_exp10: f64, hi_exp10: f64) -> f64 {
        let x = self.log_uniform(lo_exp10, hi_exp10);
        if self.next_u64() & 1 == 1 { -x } else { x }
    }
}
