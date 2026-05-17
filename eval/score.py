"""
Score predictions against ground-truth outcomes using Brier score.

Brier score for one event:
    BS = sum over outcomes o of (p_o - y_o)^2
where y_o is 1 if outcome o resolved positive, else 0.

Note: when only one outcome resolves positive (the usual case), this
matches the standard multi-class Brier. When multiple outcomes resolve
positive (e.g. "top 4 finishers"), we treat each as y=1.

Overall score = mean of per-event Brier across all *scored* events.
An event is scored only if it has a non-null `resolved_outcome` AND our
prediction returned a non-empty probabilities list.

Usage:
    python -m eval.score
    python -m eval.score --predictions eval/data/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_PREDICTIONS = "eval/data/predictions.jsonl"


def load_predictions(path: str) -> list[dict]:
    """Load all prediction records from a JSONL file."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[score] skipping malformed line: {e}", file=sys.stderr)
    return records


def extract_resolved_set(resolved_outcome: dict | None) -> set[str] | None:
    """
    Pull the set of outcomes-that-resolved-positive from the dataset's
    resolved_outcome structure. Per the docs:
        "value" is always a list of strings, drawn from outcomes.
    Returns None if the event isn't resolved.
    """
    if not resolved_outcome:
        return None
    value = resolved_outcome.get("value")
    if not isinstance(value, list):
        return None
    return {str(v) for v in value}


def brier_for_event(probs: list[dict], resolved_set: set[str], outcomes: list[str]) -> float:
    """
    Compute Brier score for one event.

    probs: our prediction list of {"market": str, "probability": float}
    resolved_set: set of outcomes that resolved positive
    outcomes: the full outcomes list from the event (defines the universe)

    Returns the sum of squared differences across all outcomes.
    """
    # Build a lookup from our predictions
    pred_map = {item["market"]: float(item["probability"]) for item in probs}

    total = 0.0
    for o in outcomes:
        p = pred_map.get(o, 0.0)  # if we somehow didn't predict an outcome, treat as 0
        y = 1.0 if o in resolved_set else 0.0
        total += (p - y) ** 2
    return total


def score_all(records: list[dict]) -> dict:
    """
    Compute aggregate and per-event Brier scores.

    Returns a dict containing:
      - overall_brier: mean across all scored events
      - n_scored: count of events scored
      - n_skipped_unresolved: events without ground truth
      - n_skipped_no_prediction: events where our prediction was empty
      - per_event: list of {ticker, title, category, brier, ...}
      - by_category: {category: mean_brier}
      - uniform_baseline: what we'd score by predicting uniform on every event
    """
    per_event: list[dict] = []
    n_skipped_unresolved = 0
    n_skipped_no_prediction = 0

    category_briers: dict[str, list[float]] = defaultdict(list)
    uniform_briers: list[float] = []
    our_briers: list[float] = []

    for rec in records:
        resolved_set = extract_resolved_set(rec.get("resolved_outcome"))
        if resolved_set is None:
            n_skipped_unresolved += 1
            continue

        outcomes = rec.get("outcomes") or []
        probs = rec.get("probabilities") or []

        if not outcomes or not probs:
            n_skipped_no_prediction += 1
            continue

        brier = brier_for_event(probs, resolved_set, outcomes)
        our_briers.append(brier)

        # Compute the uniform baseline for the same event, for comparison
        n = len(outcomes)
        uniform_p = 1.0 / n if n > 0 else 0.0
        uniform_brier = sum(
            (uniform_p - (1.0 if o in resolved_set else 0.0)) ** 2 for o in outcomes
        )
        uniform_briers.append(uniform_brier)

        category = rec.get("category") or "Unknown"
        category_briers[category].append(brier)

        per_event.append({
            "market_ticker": rec.get("market_ticker"),
            "title": rec.get("title"),
            "category": category,
            "n_outcomes": n,
            "resolved": sorted(resolved_set),
            "brier": brier,
            "uniform_brier": uniform_brier,
            "delta_vs_uniform": brier - uniform_brier,  # negative = we did better
            "elapsed_sec": rec.get("elapsed_sec"),
            "ok": rec.get("ok", True),
            "error": rec.get("error"),
        })

    overall = statistics.mean(our_briers) if our_briers else float("nan")
    uniform_baseline = statistics.mean(uniform_briers) if uniform_briers else float("nan")

    by_category = {
        cat: {
            "n": len(briers),
            "mean_brier": statistics.mean(briers),
        }
        for cat, briers in category_briers.items()
    }

    return {
        "overall_brier": overall,
        "uniform_baseline": uniform_baseline,
        "improvement_vs_uniform": uniform_baseline - overall,  # positive = we did better
        "n_scored": len(our_briers),
        "n_skipped_unresolved": n_skipped_unresolved,
        "n_skipped_no_prediction": n_skipped_no_prediction,
        "by_category": by_category,
        "per_event": per_event,
    }


def print_report(report: dict, verbose: bool = False) -> None:
    """Pretty-print a scoring report to stdout."""
    print("=" * 70)
    print("BRIER SCORE REPORT")
    print("=" * 70)

    print(f"\nEvents scored:           {report['n_scored']}")
    print(f"Events skipped (no GT):  {report['n_skipped_unresolved']}")
    print(f"Events skipped (no pred): {report['n_skipped_no_prediction']}")

    print(f"\nOur Brier score:         {report['overall_brier']:.4f}  (lower is better)")
    print(f"Uniform-prior baseline:  {report['uniform_baseline']:.4f}")
    delta = report["improvement_vs_uniform"]
    sign = "+" if delta >= 0 else ""
    print(f"Improvement over uniform: {sign}{delta:.4f}  "
          f"({'better than' if delta > 0 else 'worse than' if delta < 0 else 'tied with'} uniform)")

    print("\nPer-category breakdown:")
    if report["by_category"]:
        # Sort by mean_brier ascending (best categories first)
        for cat, stats in sorted(
            report["by_category"].items(), key=lambda kv: kv[1]["mean_brier"]
        ):
            print(f"  {cat:<20s}  n={stats['n']:>2d}  mean_brier={stats['mean_brier']:.4f}")
    else:
        print("  (no categories)")

    # Best and worst events — useful for spotting strengths and failure modes
    per_event = report["per_event"]
    if per_event:
        sorted_best = sorted(per_event, key=lambda e: e["brier"])

        print("\nBest predictions (lowest Brier):")
        for e in sorted_best[:5]:
            title = (e["title"] or "")[:60]
            print(f"  {e['brier']:.4f}  [{e['category']:<14s}]  {title}")

        print("\nWorst predictions (highest Brier):")
        for e in sorted_best[-5:][::-1]:
            title = (e["title"] or "")[:60]
            print(f"  {e['brier']:.4f}  [{e['category']:<14s}]  {title}")

    if verbose:
        print("\nFull per-event scores:")
        for e in sorted(per_event, key=lambda e: e["brier"]):
            title = (e["title"] or "")[:50]
            print(
                f"  brier={e['brier']:.4f}  resolved={e['resolved']}  "
                f"({e['category']})  {title}"
            )

    print("\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score predictions against ground truth.")
    parser.add_argument(
        "--predictions", default=DEFAULT_PREDICTIONS, help="Predictions JSONL file"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print every event's score"
    )
    parser.add_argument(
        "--save-report", default=None,
        help="Also save full report as JSON to this path",
    )
    args = parser.parse_args()

    if not Path(args.predictions).exists():
        print(f"[score] no predictions file at {args.predictions}", file=sys.stderr)
        print("[score] run `python -m eval.run_eval` first", file=sys.stderr)
        return 1

    records = load_predictions(args.predictions)
    if not records:
        print("[score] predictions file is empty", file=sys.stderr)
        return 1

    report = score_all(records)
    print_report(report, verbose=args.verbose)

    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[score] full report saved to {args.save_report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
