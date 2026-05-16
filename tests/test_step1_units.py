"""
Unit tests for the Step 1 components — parsing, label matching, calibration.

These don't make real API calls. They test the pure-logic pieces in
isolation. End-to-end LLM tests are in test_live.py (separate, since they
cost money and need internet).

Run with: `python tests/test_step1_units.py`
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import _match_outcome, _normalize_label, _parse_response, _build_prompt
from calibration import shrink_to_uniform


# ---------- label matching ----------


def test_exact_match():
    outcomes = ["Pittsburgh", "Atlanta"]
    assert _match_outcome("Pittsburgh", outcomes) == "Pittsburgh"
    assert _match_outcome("Atlanta", outcomes) == "Atlanta"


def test_case_insensitive_match():
    outcomes = ["Yes", "No"]
    assert _match_outcome("YES", outcomes) == "Yes"
    assert _match_outcome("no", outcomes) == "No"


def test_whitespace_stripped_match():
    outcomes = ["Cleveland", "Detroit"]
    assert _match_outcome("  Cleveland  ", outcomes) == "Cleveland"


def test_substring_containment_match():
    """Model returned a longer label than canonical."""
    outcomes = ["Pittsburgh", "Atlanta"]
    assert _match_outcome("Pittsburgh Steelers", outcomes) == "Pittsburgh"


def test_substring_ambiguous_no_match():
    """If returned label could match multiple canonicals, refuse to guess."""
    outcomes = ["New York Yankees", "New York Mets"]
    # "New York" is contained in both — must not match either
    assert _match_outcome("New York", outcomes) is None


def test_no_match_returns_none():
    outcomes = ["Yes", "No"]
    assert _match_outcome("Maybe", outcomes) is None


# ---------- response parsing ----------


def test_parse_valid_json_binary():
    content = '{"reasoning": "test", "probabilities": {"Yes": 0.7, "No": 0.3}}'
    probs = _parse_response(content, ["Yes", "No"])
    assert probs == [0.7, 0.3]


def test_parse_valid_json_multi_outcome():
    content = (
        '{"reasoning": "test", "probabilities": '
        '{"A": 0.5, "B": 0.3, "C": 0.2}}'
    )
    probs = _parse_response(content, ["A", "B", "C"])
    assert probs == [0.5, 0.3, 0.2]


def test_parse_strips_markdown_fences():
    content = (
        '```json\n{"reasoning": "x", "probabilities": {"Yes": 0.6, "No": 0.4}}\n```'
    )
    probs = _parse_response(content, ["Yes", "No"])
    assert probs == [0.6, 0.4]


def test_parse_preserves_outcome_order():
    """Output must match input outcome order regardless of JSON key order."""
    content = '{"probabilities": {"B": 0.2, "A": 0.5, "C": 0.3}}'
    probs = _parse_response(content, ["A", "B", "C"])
    assert probs == [0.5, 0.2, 0.3]


def test_parse_handles_fuzzy_labels():
    """Model returns mutated labels — parser still recovers."""
    content = (
        '{"probabilities": {"pittsburgh": 0.65, "ATLANTA": 0.35}}'
    )
    probs = _parse_response(content, ["Pittsburgh", "Atlanta"])
    assert probs == [0.65, 0.35]


def test_parse_raises_on_invalid_json():
    try:
        _parse_response("this is not json", ["Yes", "No"])
        assert False, "should have raised"
    except ValueError:
        pass


def test_parse_raises_on_missing_outcome():
    """Model didn't return a probability for one of the outcomes."""
    content = '{"probabilities": {"Yes": 0.7}}'
    try:
        _parse_response(content, ["Yes", "No"])
        assert False, "should have raised"
    except ValueError:
        pass


def test_parse_raises_on_non_numeric_probability():
    content = '{"probabilities": {"Yes": "high", "No": "low"}}'
    try:
        _parse_response(content, ["Yes", "No"])
        assert False, "should have raised"
    except ValueError:
        pass


def test_parse_raises_on_missing_probabilities_field():
    content = '{"reasoning": "test"}'
    try:
        _parse_response(content, ["Yes", "No"])
        assert False, "should have raised"
    except ValueError:
        pass


# ---------- calibration ----------


def test_shrinkage_pulls_extreme_toward_uniform():
    """A 0.9 / 0.1 prediction should move closer to 0.5 / 0.5."""
    calibrated = shrink_to_uniform([0.9, 0.1], alpha=0.1)
    # (1-0.1)*0.9 + 0.1*0.5 = 0.81 + 0.05 = 0.86
    # (1-0.1)*0.1 + 0.1*0.5 = 0.09 + 0.05 = 0.14
    assert abs(calibrated[0] - 0.86) < 1e-9
    assert abs(calibrated[1] - 0.14) < 1e-9


def test_shrinkage_keeps_uniform_unchanged():
    calibrated = shrink_to_uniform([0.5, 0.5], alpha=0.1)
    assert abs(calibrated[0] - 0.5) < 1e-9
    assert abs(calibrated[1] - 0.5) < 1e-9


def test_shrinkage_preserves_sum():
    """If input sums to 1, output should too (since shrinking toward 1/N preserves sum)."""
    inputs = [
        [0.7, 0.3],
        [0.5, 0.3, 0.2],
        [0.25, 0.25, 0.25, 0.25],
        [0.9, 0.05, 0.03, 0.02],
    ]
    for probs in inputs:
        out = shrink_to_uniform(probs, alpha=0.15)
        assert abs(sum(out) - 1.0) < 1e-9, f"sum drift for {probs}: {sum(out)}"


def test_shrinkage_alpha_zero_is_identity():
    probs = [0.95, 0.03, 0.02]
    out = shrink_to_uniform(probs, alpha=0.0)
    for a, b in zip(out, probs):
        assert abs(a - b) < 1e-9


def test_shrinkage_alpha_one_is_uniform():
    out = shrink_to_uniform([0.95, 0.03, 0.02], alpha=1.0)
    assert all(abs(p - 1.0 / 3) < 1e-9 for p in out)


def test_shrinkage_single_outcome_returns_one():
    assert shrink_to_uniform([1.0]) == [1.0]
    assert shrink_to_uniform([0.5]) == [1.0]  # forced to certainty


def test_shrinkage_empty_input():
    assert shrink_to_uniform([]) == []


def test_shrinkage_clamps_alpha():
    """Out-of-range alphas should be clamped, not crash."""
    out_high = shrink_to_uniform([0.7, 0.3], alpha=10.0)  # treated as 1.0
    assert all(abs(p - 0.5) < 1e-9 for p in out_high)
    out_low = shrink_to_uniform([0.7, 0.3], alpha=-1.0)  # treated as 0.0
    assert abs(out_low[0] - 0.7) < 1e-9


# ---------- prompt construction (smoke check) ----------


def test_prompt_includes_outcomes():
    event = {
        "title": "Will X happen?",
        "description": "Test",
        "category": "Test",
        "rules": "Test",
        "close_time": "2026-01-01T00:00:00Z",
        "outcomes": ["Yes", "No"],
    }
    sys_p, user_p = _build_prompt(event, ["Yes", "No"])
    assert "Yes" in user_p and "No" in user_p
    assert "Will X happen?" in user_p
    assert "EXACT" in sys_p  # we emphasize exact labels


def test_prompt_handles_missing_optional_fields():
    """description, rules can be empty/None without crashing."""
    event = {
        "title": "Test",
        "outcomes": ["A", "B"],
        # description, rules, category, close_time all missing
    }
    sys_p, user_p = _build_prompt(event, ["A", "B"])
    assert "A" in user_p and "B" in user_p


if __name__ == "__main__":
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
