"""
Main entry point for the forecasting agent.

The grader calls `predict(event)` with an event dict and expects back a
probabilities distribution. The contract:
  - exactly one entry per outcome in event['outcomes']
  - labels match the input outcomes exactly
  - probabilities in [0, 1], summing to 1

This file is the public surface. All complexity (LLM call, calibration)
lives in modules called from here. Any exception inside results in a
uniform fallback rather than a crash — completion rate matters.

Pipeline (Step 1):
    event -> agent.forecast() -> calibration.shrink_to_uniform()
          -> utils.normalize_probabilities() -> validate -> return

If anything fails: utils.safe_fallback() -> uniform distribution.
"""

from __future__ import annotations

import sys
import traceback

# Load .env automatically when this module is imported, so the OPENROUTER_API_KEY
# is available without the caller having to source the file manually.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional — if not installed, expect env vars
    # to be set externally.
    pass

from utils import (
    get_outcomes,
    normalize_probabilities,
    safe_fallback,
    validate_output,
)


def predict(event: dict) -> dict:
    """
    Predict a probability distribution over the event's outcomes.

    Returns a dict matching the grader's required output schema.
    Guaranteed to never raise — any error inside results in a uniform
    fallback distribution.
    """
    try:
        outcomes = get_outcomes(event)
        if not outcomes:
            return {"probabilities": []}

        # Single-outcome edge case: no LLM call needed.
        if len(outcomes) == 1:
            return {"probabilities": [{"market": outcomes[0], "probability": 1.0}]}

        # Run the LLM forecaster. Imported lazily so that if openai isn't
        # installed (e.g. during the Step 0 test pass), the rest of the
        # module still works for the trivial paths.
        from agent import forecast
        from calibration import shrink_to_uniform

        try:
            raw_probs = forecast(event)
        except Exception as e:  # noqa: BLE001
            print(f"[predict] forecast() failed: {e}", file=sys.stderr)
            return safe_fallback(event)

        # Calibrate (light shrinkage toward uniform)
        calibrated = shrink_to_uniform(raw_probs)

        # Normalize, validate, return
        output = {"probabilities": normalize_probabilities(outcomes, calibrated)}

        if not validate_output(output, event):
            print("[predict] validation failed, falling back to uniform", file=sys.stderr)
            return safe_fallback(event)

        return output

    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return safe_fallback(event)


if __name__ == "__main__":
    sample_event = {
        "event_ticker": "task-001",
        "market_ticker": "task-001",
        "title": "Who will win: Pittsburgh or Atlanta?",
        "description": "Predict the winner of the scheduled matchup.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-03-21T23:59:59+00:00",
        "outcomes": ["Pittsburgh", "Atlanta"],
    }
    import json
    print(json.dumps(predict(sample_event), indent=2))
