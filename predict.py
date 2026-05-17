"""
Main entry point for the forecasting agent.

The grader calls `predict(event)` with an event dict and expects back a
probabilities distribution. The contract:
  - exactly one entry per outcome in event['outcomes']
  - labels match the input outcomes exactly
  - probabilities in [0, 1], summing to 1

Pipeline:
    event -> agent.forecast() -> calibration.calibrate()
          -> utils.normalize_probabilities() -> validate -> return

If anything fails: utils.safe_fallback() -> uniform distribution.

Calibration mode is set via PROPHET_CALIBRATION_MODE env var.
Default: "multi_outcome_aware".
Options: "confidence_aware", "fixed", "multi_outcome_aware", "none".
"""

from __future__ import annotations

import os
import sys
import traceback

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from utils import (
    get_outcomes,
    normalize_probabilities,
    safe_fallback,
    validate_output,
)


CALIBRATION_MODE = os.environ.get("PROPHET_CALIBRATION_MODE", "multi_outcome_aware")


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

        if len(outcomes) == 1:
            return {
                "probabilities": [
                    {"market": outcomes[0], "probability": 1.0}
                ]
            }

        from agent import forecast
        from calibration import calibrate

        try:
            raw_probs = forecast(event)

            if len(raw_probs) != len(outcomes):
                print(
                    f"[predict] forecast returned {len(raw_probs)} probabilities "
                    f"for {len(outcomes)} outcomes",
                    file=sys.stderr,
                )
                return safe_fallback(event)

        except Exception as e:  # noqa: BLE001
            print(f"[predict] forecast() failed: {e}", file=sys.stderr)
            return safe_fallback(event)

        try:
            calibrated = calibrate(raw_probs, mode=CALIBRATION_MODE)

            if len(calibrated) != len(outcomes):
                print(
                    f"[predict] calibration returned {len(calibrated)} probabilities "
                    f"for {len(outcomes)} outcomes",
                    file=sys.stderr,
                )
                return safe_fallback(event)

        except Exception as e:  # noqa: BLE001
            print(f"[predict] calibration failed: {e}", file=sys.stderr)
            calibrated = raw_probs

        output = {
            "probabilities": normalize_probabilities(outcomes, calibrated)
        }

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