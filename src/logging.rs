//! Structured logging and profiling via tracing.
//!
//! - [`init`] / [`init_with_level`] — standard log output controlled by `PRAVASH_LOG`
//! - [`init_profiling`] — chrome://tracing JSON output for flame graph analysis
//!
//! All pravash operations emit `trace_span!` events. With profiling enabled,
//! these produce a `trace-{timestamp}.json` file that can be loaded into
//! `chrome://tracing` or processed by tools like `inferno`.

/// Initialize logging with default "info" level.
pub fn init() {
    init_with_level("info");
}

/// Initialize logging with a custom default level.
/// Respects the `PRAVASH_LOG` env var (e.g., `PRAVASH_LOG=trace`).
pub fn init_with_level(default_level: &str) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter =
        EnvFilter::try_from_env("PRAVASH_LOG").unwrap_or_else(|_| EnvFilter::new(default_level));
    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .try_init();
}

/// Initialize profiling with chrome://tracing JSON output.
///
/// Returns a [`FlushGuard`](tracing_chrome::FlushGuard) that must be held
/// alive for the duration of profiling. When dropped, it flushes and closes
/// the trace file.
///
/// The trace file is written to the current directory as `trace-{pid}.json`.
///
/// # Usage
///
/// ```ignore
/// let _guard = pravash::logging::init_profiling();
/// // ... run simulation ...
/// // guard dropped here, trace file flushed
/// ```
pub fn init_profiling() -> tracing_chrome::FlushGuard {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let (chrome_layer, guard) = ChromeLayerBuilder::new().include_args(true).build();
    let _ = tracing_subscriber::registry().with(chrome_layer).try_init();
    guard
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_does_not_panic() {
        super::init();
    }

    // init_profiling is tested via the example and integration tests.
    // Cannot unit-test here since it registers a global subscriber.
}
