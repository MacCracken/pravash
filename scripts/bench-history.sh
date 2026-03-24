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

if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,label,benchmark,time_ns" > "$HISTORY_FILE"
fi

echo "Running benchmarks..."
BENCH_OUTPUT=$(cargo bench --manifest-path "$PROJECT_DIR/Cargo.toml" \
    --all-features 2>&1) || {
    echo "Benchmark run failed:"
    echo "$BENCH_OUTPUT"
    exit 1
}

# Use awk to parse criterion output: track "Benchmarking X" lines (no colon),
# then extract the middle estimate from "time: [lo est hi]" lines.
ENTRIES=$(echo "$BENCH_OUTPUT" | awk -v ts="$TIMESTAMP" -v label="$LABEL" -v outfile="$HISTORY_FILE" '
/^Benchmarking [a-zA-Z]/ && !/:/ {
    name = $2
    next
}
/time:/ {
    # Some lines have the name before "time:": "sph_step/9_particles   time: ..."
    if ($0 ~ /^[a-zA-Z].*time:/) {
        name = $1
    }
    match($0, /\[([0-9.]+) ([a-z]+) ([0-9.]+) ([a-z]+)/, arr)
    if (arr[3] != "" && arr[4] != "" && name != "") {
        val = arr[3]
        unit = arr[4]
        if (unit == "ps") val = val * 0.001
        else if (unit == "ns") val = val * 1
        else if (unit == "us" || unit == "µs") val = val * 1000
        else if (unit == "ms") val = val * 1000000
        else if (unit == "s") val = val * 1000000000
        printf "%s,%s,%s,%.4f\n", ts, label, name, val >> outfile
        count++
    }
}
END { print count+0 }
')

echo ""
echo "Results appended to $HISTORY_FILE"
echo "Label: $LABEL"
echo "Timestamp: $TIMESTAMP"
echo "Entries added: $ENTRIES"
