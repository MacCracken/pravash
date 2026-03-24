//! Structured logging via `PRAVASH_LOG` env var.

pub fn init() { init_with_level("info"); }

pub fn init_with_level(default_level: &str) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter = EnvFilter::try_from_env("PRAVASH_LOG")
        .unwrap_or_else(|_| EnvFilter::new(default_level));
    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .try_init();
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_does_not_panic() { super::init(); }
}
