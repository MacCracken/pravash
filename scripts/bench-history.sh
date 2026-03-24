#!/usr/bin/env bash
# bench-history.sh — Run benchmarks and append results to CSV history.
#
# Usage: ./scripts/bench-history.sh [label]
#   label: optional tag for this run (e.g., "baseline", "post-audit")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HISTORY_FILE="$PROJECT_DIR/bench-history.csv"
LABEL="${1:-$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Ensure header exists
if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,label,benchmark,time_ns" > "$HISTORY_FILE"
fi

echo "Running benchmarks..."
BENCH_OUTPUT=$(cargo bench --manifest-path "$PROJECT_DIR/Cargo.toml" \
    --features sph,grid,shallow 2>&1) || {
    echo "Benchmark run failed:"
    echo "$BENCH_OUTPUT"
    exit 1
}

# Convert value+unit to nanoseconds using awk (no bc dependency)
to_ns() {
    local value="$1" unit="$2"
    case "$unit" in
        ps)      awk "BEGIN { printf \"%.4f\", $value * 0.001 }" ;;
        ns)      echo "$value" ;;
        µs|μs|us) awk "BEGIN { printf \"%.4f\", $value * 1000 }" ;;
        ms)      awk "BEGIN { printf \"%.4f\", $value * 1000000 }" ;;
        s)       awk "BEGIN { printf \"%.4f\", $value * 1000000000 }" ;;
        *)       echo "$value" ;;
    esac
}

# Parse criterion output. Track the current benchmark name from "Benchmarking X" lines,
# then extract the estimate from "time: [lo est hi]" lines.
CURRENT_NAME=""
ENTRIES=0

while IFS= read -r line; do
    # Match "Benchmarking sph_kernels/poly6" (but not "Benchmarking ...: Warming up" etc.)
    if [[ "$line" =~ ^Benchmarking\ ([a-zA-Z0-9_/]+)$ ]]; then
        CURRENT_NAME="${BASH_REMATCH[1]}"
        continue
    fi

    # Match "time:   [1.2345 ns 1.2456 ns 1.2567 ns]"
    if [[ "$line" =~ time:.*\[([0-9.]+)\ ([a-zµμ]+)\ ([0-9.]+)\ ([a-zµμ]+)\ ([0-9.]+)\ ([a-zµμ]+)\] ]]; then
        estimate="${BASH_REMATCH[3]}"
        unit="${BASH_REMATCH[4]}"
        ns=$(to_ns "$estimate" "$unit")

        if [ -n "$CURRENT_NAME" ] && [ -n "$ns" ]; then
            echo "$TIMESTAMP,$LABEL,$CURRENT_NAME,$ns" >> "$HISTORY_FILE"
            ENTRIES=$((ENTRIES + 1))
        fi
    fi
done <<< "$BENCH_OUTPUT"

echo ""
echo "Results appended to $HISTORY_FILE"
echo "Label: $LABEL"
echo "Timestamp: $TIMESTAMP"
echo "Entries added: $ENTRIES"
