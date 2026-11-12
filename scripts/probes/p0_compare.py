"""p0_compare.py — join p0_semantics' Cyrius and Rust outputs into a Markdown table.

    python3 -I scripts/probes/p0_compare.py build/p0_cyrius.tsv build/p0_rust.tsv

For float types the last column is the ulp distance from Rust's value (0 = same bits; NaN payloads
compare by bits). For integer types it is "same" or "DIFFERS" at the type's width.
"""

import sys

WIDTH = {"f64": 64, "f32": 32, "i32": 32, "u32": 32, "u64": 64}


def ordered(bits: int, width: int) -> int:
    """Map float bits to a monotone integer, so ulp distance is a difference."""
    sign = 1 << (width - 1)
    return -(bits & (sign - 1)) if bits & sign else bits


def main() -> int:
    cy_path, rs_path = sys.argv[1], sys.argv[2]
    rust = {}
    with open(rs_path, encoding="utf-8") as fh:
        for line in fh:
            case, ty, bits = line.rstrip("\n").split("\t")
            rust[case] = (ty, int(bits, 16))
    rows = []
    worst = 0
    with open(cy_path, encoding="utf-8") as fh:
        for line in fh:
            case, variant, ty, bits = line.rstrip("\n").split("\t")
            width = WIDTH[ty]
            mask = (1 << width) - 1
            cy = int(bits, 16) & mask
            rty, rbits = rust[case]
            rb = rbits & mask
            if ty in ("f64", "f32"):
                d = abs(ordered(cy, width) - ordered(rb, width))
                worst = max(worst, d)
                verdict = f"{d} ulp" if d else "same bits"
            else:
                verdict = "same" if cy == rb else "**DIFFERS**"
            rows.append(f"| `{case}` | `{variant}` | `{rb:#x}` | `{cy:#x}` | {verdict} |")
    missing = sorted(set(rust) - {r.split("`")[1] for r in rows})
    print("| Case | Cyrius | Rust bits | Cyrius bits | Result |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    if missing:
        print(f"\nRust cases with no Cyrius probe: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
