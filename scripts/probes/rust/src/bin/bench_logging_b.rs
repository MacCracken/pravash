//! The Rust side of tests/pravash.bcyr's `logging_b/*` and `ai_b/*` rows (M7 part b, docs/benchmarks-rust-v-cyrius.md)
//! on the frozen pravash 1.3.0 and its tracing 0.1.44.
//!
//!   taskset -c 15 cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin bench_logging_b
//!
//! Each row is the median of 7 runs, in ns per iteration:
//! - `logging_b/span_off`: `trace_span!("sph::step", n = k).entered()` and its drop, with no subscriber (tracing off);
//! - `logging_b/span_on`: the same into a subscriber that enables everything and does nothing;
//! - `logging_b/warn_off`: `tracing::warn!(n = k, "invalid neighbor lists ignored")` with no subscriber;
//! - `logging_b/filter_parse`: `EnvFilter::builder().parse("pravash::sph=trace,pravash=warn,info")`;
//! - `ai_b/daimon_read`, `ai_b/daimon_write`, `ai_b/daimon_debug`: serde_json's read and write of
//!   `{"endpoint":"https://daimon.example:8090/v1","api_key":"sk-0123456789abcdef"}`, and `format!("{:?}")` of it.

use std::hint::black_box;
use std::time::Instant;

use pravash::ai::DaimonConfig;
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::EnvFilter;

const RUNS: usize = 7;

fn time(name: &str, iters: usize, mut body: impl FnMut(usize) -> u64) {
    let mut ns = Vec::with_capacity(RUNS);
    let mut sink = 0u64;
    for _ in 0..RUNS {
        let t = Instant::now();
        for k in 0..iters {
            sink ^= body(k);
        }
        #[allow(clippy::cast_precision_loss)]
        ns.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    black_box(sink);
    ns.sort_by(f64::total_cmp);
    println!("  {name:<30} {:>12.2} ns", ns[RUNS / 2]);
}

/// Enables everything and keeps nothing.
struct Null;

impl Subscriber for Null {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, _: &Event<'_>) {}
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

fn main() {
    time("logging_b/span_off", 10_000_000, |k| {
        let _s = tracing::trace_span!("sph::step", n = black_box(k)).entered();
        1
    });
    tracing::subscriber::with_default(Null, || {
        time("logging_b/span_on", 2_000_000, |k| {
            let _s = tracing::trace_span!("sph::step", n = black_box(k)).entered();
            1
        });
    });
    time("logging_b/warn_off", 10_000_000, |k| {
        tracing::warn!(n = black_box(k), "invalid neighbor lists ignored");
        1
    });
    time("logging_b/filter_parse", 200_000, |_| {
        u64::from(
            EnvFilter::builder()
                .parse(black_box("pravash::sph=trace,pravash=warn,info"))
                .is_ok(),
        )
    });
    let doc = r#"{"endpoint":"https://daimon.example:8090/v1","api_key":"sk-0123456789abcdef"}"#;
    time("ai_b/daimon_read", 1_000_000, |_| {
        serde_json::from_str::<DaimonConfig>(black_box(doc)).map_or(0, |c| c.endpoint.len() as u64)
    });
    let c: DaimonConfig = serde_json::from_str(doc).unwrap_or_default();
    time("ai_b/daimon_write", 1_000_000, |_| {
        serde_json::to_string(black_box(&c)).map_or(0, |s| s.len() as u64)
    });
    time("ai_b/daimon_debug", 1_000_000, |_| {
        format!("{:?}", black_box(&c)).len() as u64
    });
}
