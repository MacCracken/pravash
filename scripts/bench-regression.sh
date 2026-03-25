#!/usr/bin/env bash
# bench-regression.sh — Compare current benchmarks against a baseline.
#
# Usage: ./scripts/bench-regression.sh [baseline_label] [threshold_pct]
#   baseline_label: label in bench-history.csv to compare against (default: last non-current)
#   threshold_pct:  max allowed regression percentage (default: 20)
#
# Exit code 0 = no regression, 1 = regression detected.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HISTORY_FILE="$PROJECT_DIR/bench-history.csv"
THRESHOLD="${2:-20}"

if [ ! -f "$HISTORY_FILE" ]; then
    echo "No bench-history.csv found. Run bench-history.sh first."
    exit 1
fi

# Run current benchmarks
echo "Running current benchmarks..."
CURRENT_LABEL="regression-check"
bash "$SCRIPT_DIR/bench-history.sh" "$CURRENT_LABEL" > /dev/null 2>&1

# Find baseline label (use provided or second-to-last unique label)
if [ -n "${1:-}" ]; then
    BASELINE_LABEL="$1"
else
    BASELINE_LABEL=$(awk -F, '{print $2}' "$HISTORY_FILE" | grep -v "^label$" | grep -v "$CURRENT_LABEL" | tail -1)
fi

echo "Baseline: $BASELINE_LABEL"
echo "Threshold: ${THRESHOLD}%"
echo ""

# Compare
FAILED=0
while IFS=, read -r _ts _label bench current_ns; do
    # Find matching baseline
    baseline_ns=$(grep ",$BASELINE_LABEL,$bench," "$HISTORY_FILE" | tail -1 | cut -d, -f4)
    if [ -z "$baseline_ns" ]; then
        continue
    fi

    # Calculate regression percentage
    pct=$(awk "BEGIN { diff = ($current_ns - $baseline_ns) / $baseline_ns * 100; printf \"%.1f\", diff }")

    # Use relaxed threshold for sub-microsecond benchmarks (high noise)
    effective_thr=$(awk "BEGIN { if ($baseline_ns < 1000) printf \"%.0f\", $THRESHOLD * 2; else printf \"%.0f\", $THRESHOLD }")

    if awk "BEGIN { exit !($pct > $effective_thr) }"; then
        echo "REGRESSION: $bench  ${pct}% slower ($baseline_ns -> $current_ns ns)"
        FAILED=1
    else
        echo "  ok: $bench  ${pct}%"
    fi
done < <(grep ",$CURRENT_LABEL," "$HISTORY_FILE")

echo ""
if [ "$FAILED" -eq 1 ]; then
    echo "FAIL: Performance regressions detected (threshold: ${THRESHOLD}%)"
    exit 1
else
    echo "PASS: No regressions beyond ${THRESHOLD}%"
    exit 0
fi
