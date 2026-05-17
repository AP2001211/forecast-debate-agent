"""
Sweep calibration parameters against an existing predictions JSONL file.
No API calls — pure re-calibration of stored probabilities.

Usage:
    python -m eval.tune_calibration
    python -m eval.tune_calibration --predictions eval/data/predictions_step4.jsonl
    python -m eval.tune_calibration --top 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

# Parameter grid
BASES = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
SLOPES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
MULTI_NS = [5, 8, 10, 15, 20]
MULTI_THRESHES = [0.10, 0.15, 0.20, 0.25, 0.30]

DEFAULT_PREDICTIONS = "eval/data/predictions_step4.jsonl"


def load_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def extract_resolved_set(resolved_outcome: dict | None) -> set[str] | None:
    if not resolved_outcome:
        return None
    value = resolved_outcome.get("value")
    if not isinstance(value, list):
        return None
    return {str(v) for v in value}


def apply_multi_outcome_aware(
    probs: list[float],
    base: float,
    slope: float,
    multi_n: int,
    multi_thresh: float,
) -> list[float]:
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    if n >= multi_n and max(probs) < multi_thresh:
        return [1.0 / n] * n

    uniform_p = 1.0 / n
    confidence = max(probs) - uniform_p
    alpha_eff = min(1.0, max(0.0, base + slope * confidence))
    return [(1.0 - alpha_eff) * p + alpha_eff * uniform_p for p in probs]


def score_combo(
    records: list[dict],
    base: float,
    slope: float,
    multi_n: int,
    multi_thresh: float,
) -> float:
    briers: list[float] = []
    for rec in records:
        resolved_set = extract_resolved_set(rec.get("resolved_outcome"))
        if resolved_set is None:
            continue
        outcomes: list[str] = rec.get("outcomes") or []
        probs_dicts: list[dict] = rec.get("probabilities") or []
        if not outcomes or not probs_dicts:
            continue

        pred_map = {item["market"]: float(item["probability"]) for item in probs_dicts}
        n = len(outcomes)
        probs = [pred_map.get(o, 1.0 / n) for o in outcomes]

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        cal = apply_multi_outcome_aware(probs, base, slope, multi_n, multi_thresh)

        total = sum(cal)
        if total > 0:
            cal = [p / total for p in cal]

        b = sum((p - (1.0 if o in resolved_set else 0.0)) ** 2 for o, p in zip(outcomes, cal))
        briers.append(b)

    return statistics.mean(briers) if briers else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep calibration params against existing predictions.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--top", type=int, default=10, help="How many top combos to print")
    args = parser.parse_args()

    if not Path(args.predictions).exists():
        print(f"[tune] no predictions file at {args.predictions}", file=sys.stderr)
        print("[tune] run `python -m eval.run_eval` first", file=sys.stderr)
        return 1

    records = load_records(args.predictions)
    n_scoreable = sum(1 for r in records if extract_resolved_set(r.get("resolved_outcome")))
    print(f"[tune] loaded {len(records)} records, {n_scoreable} scoreable")

    combos = list(itertools.product(BASES, SLOPES, MULTI_NS, MULTI_THRESHES))
    print(f"[tune] sweeping {len(combos)} parameter combinations...")

    results: list[tuple[float, float, float, int, float]] = []
    for base, slope, multi_n, multi_thresh in combos:
        s = score_combo(records, base, slope, multi_n, multi_thresh)
        results.append((s, base, slope, multi_n, multi_thresh))
    results.sort()

    print(f"\nTop {args.top} calibration settings (lower Brier = better):")
    print(f"{'Brier':>8}  {'base':>5}  {'slope':>5}  {'multi_n':>7}  {'thresh':>6}")
    print("-" * 45)
    for score, base, slope, multi_n, multi_thresh in results[: args.top]:
        print(f"{score:8.4f}  {base:5.2f}  {slope:5.2f}  {multi_n:7d}  {multi_thresh:6.2f}")

    # Current defaults for comparison
    current = score_combo(records, 0.05, 0.30, 10, 0.20)
    print(f"\nCurrent defaults (base=0.05 slope=0.30 multi_n=10 thresh=0.20): Brier {current:.4f}")

    best_score, best_base, best_slope, best_multi_n, best_thresh = results[0]
    if best_score < current:
        improvement = current - best_score
        print(f"Best combo improves by {improvement:.4f} ({improvement / current * 100:.1f}%)")
    else:
        print("Current defaults are already optimal on this dataset.")

    print(f"\nTo apply best combo, run in PowerShell:")
    print(f"  $env:PROPHET_CAL_BASE='{best_base}'")
    print(f"  $env:PROPHET_CAL_SLOPE='{best_slope}'")
    print(f"  $env:PROPHET_CAL_MULTI_N='{best_multi_n}'")
    print(f"  $env:PROPHET_CAL_MULTI_THRESH='{best_thresh}'")
    print(f"  $env:PROPHET_CALIBRATION_MODE='multi_outcome_aware'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
