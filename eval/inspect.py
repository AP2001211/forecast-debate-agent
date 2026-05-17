"""
Inspect predictions on a dataset (resolved or not).

Useful when the dataset doesn't have ground truth and we just want to
eyeball whether the agent is producing sensible-looking predictions.

Usage:
    python -m eval.inspect --predictions eval/data/predictions_sports.jsonl
    python -m eval.inspect --predictions eval/data/predictions_sports.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path


def load_predictions(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def format_one(rec: dict, idx: int, total: int) -> str:
    """Format a single prediction record for human reading."""
    title = rec.get("title", "(no title)")
    category = rec.get("category", "?")
    outcomes = rec.get("outcomes", []) or []
    probs = rec.get("probabilities", []) or []
    resolved = rec.get("resolved_outcome")
    ok = rec.get("ok", True)
    elapsed = rec.get("elapsed_sec")

    # Sort probabilities high to low for readability
    sorted_probs = sorted(probs, key=lambda p: -p.get("probability", 0))

    lines: list[str] = []
    lines.append(f"[{idx}/{total}] {title}")
    meta = f"  category={category}, n_outcomes={len(outcomes)}"
    if elapsed is not None:
        meta += f", elapsed={elapsed:.1f}s"
    if not ok:
        meta += "  (FAILED — used fallback)"
    lines.append(meta)

    if resolved:
        resolved_value = resolved.get("value") if isinstance(resolved, dict) else None
        if resolved_value:
            lines.append(f"  resolved: {resolved_value}")

    # Show top 5 predicted outcomes (or all if <= 5)
    show_n = min(5, len(sorted_probs))
    if show_n > 0:
        lines.append("  predictions (top {}):".format(show_n))
        for p in sorted_probs[:show_n]:
            market = p.get("market", "?")
            prob = p.get("probability", 0)
            bar_len = int(prob * 30)
            bar = "█" * bar_len + " " * (30 - bar_len)
            lines.append(f"    {prob:>5.1%}  {bar}  {market}")
        if len(sorted_probs) > show_n:
            lines.append(f"    ... +{len(sorted_probs) - show_n} more")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect agent predictions.")
    parser.add_argument(
        "--predictions", required=True, help="Predictions JSONL file path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only show first N predictions"
    )
    parser.add_argument(
        "--filter-category", default=None,
        help="Only show predictions from this category",
    )
    args = parser.parse_args()

    if not Path(args.predictions).exists():
        print(f"No predictions file at {args.predictions}", file=sys.stderr)
        return 1

    records = load_predictions(args.predictions)
    if args.filter_category:
        records = [r for r in records if r.get("category") == args.filter_category]
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        print("No records to show.")
        return 0

    total = len(records)
    print(f"\nShowing {total} prediction(s) from {args.predictions}\n")
    print("=" * 78)
    for i, rec in enumerate(records, start=1):
        print()
        print(format_one(rec, i, total))
    print()
    print("=" * 78)

    # Quick summary stats
    n_ok = sum(1 for r in records if r.get("ok", True))
    n_failed = total - n_ok
    elapsed_total = sum(r.get("elapsed_sec", 0) or 0 for r in records)
    elapsed_avg = elapsed_total / total if total > 0 else 0
    print(f"\nSummary: {n_ok}/{total} ok, {n_failed} failed, "
          f"avg latency {elapsed_avg:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
