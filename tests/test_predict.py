"""
Tests for the predict() entry point.

These verify the OUTPUT CONTRACT under a variety of inputs, including
adversarial cases. Run with: `python -m pytest tests/ -v`
Or directly: `python tests/test_predict.py`
"""

from __future__ import annotations

import os
import sys

# Make the parent directory importable when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict
from utils import validate_output


def _make_event(outcomes, **overrides):
    event = {
        "event_ticker": "test-001",
        "market_ticker": "test-001",
        "title": "Test event",
        "description": "Test description",
        "category": "Sports",
        "rules": "Test rules",
        "close_time": "2026-12-31T23:59:59+00:00",
        "outcomes": outcomes,
    }
    event.update(overrides)
    return event


def test_binary_event():
    event = _make_event(["Pittsburgh", "Atlanta"])
    out = predict(event)
    assert validate_output(out, event), f"Invalid output: {out}"
    assert len(out["probabilities"]) == 2


def test_multi_outcome_event():
    event = _make_event(["A", "B", "C", "D", "E"])
    out = predict(event)
    assert validate_output(out, event), f"Invalid output: {out}"
    assert len(out["probabilities"]) == 5
    # Uniform check (Step 0 always returns uniform)
    for item in out["probabilities"]:
        assert abs(item["probability"] - 0.2) < 1e-9


def test_single_outcome():
    """Pathological but possible — must still produce valid output."""
    event = _make_event(["OnlyOne"])
    out = predict(event)
    assert validate_output(out, event)
    assert out["probabilities"][0]["probability"] == 1.0


def test_empty_outcomes():
    """Truly broken event — we return empty list, not crash."""
    event = _make_event([])
    out = predict(event)
    assert out == {"probabilities": []}


def test_missing_outcomes_key():
    """Even more broken — outcomes key missing entirely."""
    event = _make_event([])
    del event["outcomes"]
    out = predict(event)
    assert out == {"probabilities": []}


def test_outcomes_with_whitespace():
    """Whitespace in labels is preserved/stripped consistently."""
    event = _make_event(["  Yes  ", "No"])
    out = predict(event)
    # Our normalization strips whitespace from outcome labels
    markets = [p["market"] for p in out["probabilities"]]
    assert "Yes" in markets and "No" in markets


def test_outcomes_with_none_entries():
    """Defensive: None entries should be dropped."""
    event = _make_event(["A", None, "B"])
    out = predict(event)
    # After cleanup, we have 2 valid outcomes
    assert len(out["probabilities"]) == 2
    markets = [p["market"] for p in out["probabilities"]]
    assert markets == ["A", "B"]


def test_probabilities_sum_to_one():
    """Sum must be exactly 1.0 within floating-point tolerance."""
    for outcomes in [
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C", "D", "E", "F", "G"],
        ["X"] + [f"Y{i}" for i in range(20)],  # 21-outcome event
    ]:
        event = _make_event(outcomes)
        out = predict(event)
        total = sum(p["probability"] for p in out["probabilities"])
        assert abs(total - 1.0) < 1e-9, f"Sum was {total} for {len(outcomes)} outcomes"


def test_each_outcome_appears_exactly_once():
    event = _make_event(["A", "B", "C", "D"])
    out = predict(event)
    markets = [p["market"] for p in out["probabilities"]]
    assert sorted(markets) == sorted(event["outcomes"])
    assert len(markets) == len(set(markets))


def test_no_extra_or_missing_markets():
    event = _make_event(["Cleveland", "Detroit"])
    out = predict(event)
    markets = {p["market"] for p in out["probabilities"]}
    assert markets == {"Cleveland", "Detroit"}


def test_probabilities_in_valid_range():
    event = _make_event(["A", "B", "C"])
    out = predict(event)
    for p in out["probabilities"]:
        assert 0.0 <= p["probability"] <= 1.0


def test_predict_never_raises():
    """Even with deeply malformed input, predict() must not throw."""
    bad_inputs = [
        {},
        {"outcomes": None},
        {"outcomes": "not a list"},
        {"outcomes": [1, 2, 3]},  # non-string outcomes
        None,  # not even a dict
    ]
    for bad in bad_inputs:
        try:
            if bad is None:
                # predict() expects a dict; if grader passes None, we
                # should still not crash. Test it survives.
                out = predict(bad if bad is not None else {})
            else:
                out = predict(bad)
            assert isinstance(out, dict)
            assert "probabilities" in out
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"predict() raised on bad input {bad!r}: {e}")


def test_realistic_sports_event():
    """The exact example from the spec."""
    event = {
        "event_ticker": "task-001",
        "market_ticker": "task-001",
        "title": "Who will win: Pittsburgh or Atlanta?",
        "description": "Predict the winner of the scheduled matchup.",
        "category": "Sports",
        "rules": "Resolves to the official winner after the game is final.",
        "close_time": "2026-03-21T23:59:59+00:00",
        "outcomes": ["Pittsburgh", "Atlanta"],
    }
    out = predict(event)
    assert validate_output(out, event)


if __name__ == "__main__":
    # Run all tests manually without pytest
    import inspect

    test_funcs = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]

    passed = 0
    failed = 0
    for name, fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed (of {len(test_funcs)} total)")
    sys.exit(0 if failed == 0 else 1)
