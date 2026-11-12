.PHONY: check fmt clippy test test-oracle audit deny bench coverage build doc clean

check: fmt clippy test test-oracle audit

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-features --all-targets -- -D warnings

test:
	cargo test --all-features

# The oracle hooks without `parallel` (--all-features above runs them with it).
test-oracle:
	cargo test --features oracle,coupling --test oracle

audit:
	cargo audit

deny:
	cargo deny check

bench:
	./scripts/bench-history.sh

coverage:
	cargo llvm-cov --all-features --html --output-dir coverage/

build:
	cargo build --release --all-features

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

clean:
	cargo clean
	rm -rf coverage/
