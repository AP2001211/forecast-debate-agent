"""
Run the agent against a dataset and save predictions to JSONL.

Resumable: if the predictions file already contains entries for some events,
they are skipped. Useful when the API hiccups partway through and we don't
want to redo (or re-pay-for) the successful calls.

Usage:
    # Run on the default sample-resolved dataset
    python -m eval.run_eval

    # Run on a custom events file
    python -m eval.run_eval --events eval/data/events_resolved.json

    # Force a fresh run (delete previous predictions)
    python -m eval.run_eval --fresh

    # Limit to first N events (useful for smoke-testing)
    python -m eval.run_eval --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the project root importable so we can `from predict import predict`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict  # noqa: E402


DEFAULT_EVENTS = "eval/data/events_resolved.json"
DEFAULT_PREDICTIONS = "eval/data/predictions.jsonl"


def load_events(path: str) -> list[dict]:
    """Load the events JSON file (a list of event dicts)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level of {path}, got {type(data).__name__}")
    return data


def load_existing_predictions(path: str) -> dict[str, dict]:
    """
    Load any predictions that already exist in the JSONL file.

    Returns a dict mapping market_ticker -> stored prediction record.
    Skips malformed lines silently (defensive).
    """
    existing: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return existing

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ticker = rec.get("market_ticker")
            if ticker:
                existing[ticker] = rec
    return existing


def append_prediction(path: str, record: dict) -> None:
    """Append a single prediction record to the JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(
    events_path: str = DEFAULT_EVENTS,
    predictions_path: str = DEFAULT_PREDICTIONS,
    fresh: bool = False,
    limit: int | None = None,
) -> None:
    # Optionally wipe the predictions file for a clean run
    if fresh and Path(predictions_path).exists():
        Path(predictions_path).unlink()
        print(f"[run] deleted {predictions_path} for fresh run")

    events = load_events(events_path)
    if limit is not None:
        events = events[:limit]
    print(f"[run] loaded {len(events)} events from {events_path}")

    existing = load_existing_predictions(predictions_path)
    if existing:
        print(f"[run] resuming: {len(existing)} predictions already exist")

    n_skipped = 0
    n_done = 0
    n_failed = 0
    t_start = time.time()

    for i, event in enumerate(events, start=1):
        ticker = event.get("market_ticker") or event.get("event_ticker")
        if not ticker:
            print(f"[run]   event {i} has no ticker, skipping")
            continue

        if ticker in existing:
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            result = predict(event)
            ok = True
            err: str | None = None
        except Exception as e:  # noqa: BLE001
            # Shouldn't happen — predict() is supposed to never raise — but
            # we belt-and-brace anyway so a bug in predict() doesn't kill
            # the whole run.
            result = {"probabilities": []}
            ok = False
            err = repr(e)
            n_failed += 1

        elapsed = time.time() - t0
        n_done += 1

        record = {
            "market_ticker": ticker,
            "event_ticker": event.get("event_ticker"),
            "title": event.get("title"),
            "category": event.get("category"),
            "outcomes": event.get("outcomes"),
            "resolved_outcome": event.get("resolved_outcome"),  # carry through for scoring
            "probabilities": result.get("probabilities", []),
            "ok": ok,
            "error": err,
            "elapsed_sec": round(elapsed, 2),
        }
        append_prediction(predictions_path, record)

        # Progress line
        if ok:
            # Show the top-2 outcomes by probability for a quick eyeball check
            sorted_probs = sorted(
                record["probabilities"], key=lambda p: -p["probability"]
            )
            top = ", ".join(
                f"{p['market']}={p['probability']:.2f}" for p in sorted_probs[:2]
            )
            print(f"[run] {i}/{len(events)}  {ticker}  ({elapsed:.1f}s)  {top}")
        else:
            print(f"[run] {i}/{len(events)}  {ticker}  FAILED: {err}")

    elapsed_total = time.time() - t_start
    print(
        f"\n[run] done. {n_done} new, {n_skipped} skipped (already done), "
        f"{n_failed} failed, in {elapsed_total:.1f}s total"
    )
    print(f"[run] predictions saved to {predictions_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the forecasting agent against a dataset.")
    parser.add_argument("--events", default=DEFAULT_EVENTS, help="Events JSON file")
    parser.add_argument("--out", default=DEFAULT_PREDICTIONS, help="Predictions JSONL output")
    parser.add_argument("--fresh", action="store_true", help="Delete prior predictions and start over")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N events (smoke test)")
    args = parser.parse_args()

    run(
        events_path=args.events,
        predictions_path=args.out,
        fresh=args.fresh,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
