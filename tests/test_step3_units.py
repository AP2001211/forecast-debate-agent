"""
Unit tests for the Step 3 changes:
  - Smart-quote label normalization
  - Loose JSON parser (handles prose-around-JSON)
  - Confidence-aware calibration

Run with: `python tests/test_step3_units.py`

No API calls. No money spent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import _match_outcome, _normalize_label, _parse_json_loose, _parse_response
from calibration import (
    calibrate,
    confidence_aware_shrink,
    shrink_to_uniform,
)


# ---------- smart-quote label matching (the Survivor bug) ----------


def test_smart_double_quotes_match():
    """The exact failure from Step 2 — curly quotes vs straight quotes."""
    canonical = "Benjamin \u201cCoach\u201d Wade"  # canonical with curly quotes
    returned = 'Benjamin "Coach" Wade'             # model returned straight quotes
    assert _match_outcome(returned, [canonical]) == canonical


def test_smart_single_quotes_match():
    canonical = "It\u2019s a name"  # right single quote
    returned = "It's a name"
    assert _match_outcome(returned, [canonical]) == canonical


def test_em_dash_match():
    canonical = "Game 5 \u2014 winner"  # em dash
    returned = "Game 5 - winner"
    assert _match_outcome(returned, [canonical]) == canonical


def test_non_breaking_space_match():
    canonical = "New\u00a0York"  # NBSP
    returned = "New York"
    assert _match_outcome(returned, [canonical]) == canonical


def test_accented_normalization():
    canonical = "Viktor Orb\u00e1n"  # á
    returned = "Viktor Orba\u0301n"   # a + combining acute (decomposed form)
    # NFKC should normalize the decomposed form
    assert _match_outcome(returned, [canonical]) == canonical


def test_normalize_collapses_whitespace():
    assert _normalize_label("  hello   world  ") == "hello world"


def test_existing_matches_still_work():
    """Make sure the new normalization didn't break old behavior."""
    outcomes = ["Pittsburgh", "Atlanta"]
    assert _match_outcome("Pittsburgh", outcomes) == "Pittsburgh"
    assert _match_outcome("ATLANTA", outcomes) == "Atlanta"
    assert _match_outcome("Pittsburgh Steelers", outcomes) == "Pittsburgh"


# ---------- loose JSON parser ----------


def test_loose_json_pure():
    data = _parse_json_loose('{"a": 1, "b": 2}')
    assert data == {"a": 1, "b": 2}


def test_loose_json_with_prose_before():
    """The SCOTUS failure mode: model wrote prose then JSON."""
    content = (
        "Let me think about this. I'll consider the voting patterns.\n\n"
        '{"reasoning": "x", "probabilities": {"A": 0.7, "B": 0.3}}'
    )
    data = _parse_json_loose(content)
    assert data["probabilities"] == {"A": 0.7, "B": 0.3}


def test_loose_json_with_prose_after():
    content = (
        '{"probabilities": {"A": 0.6, "B": 0.4}}\n\n'
        "Hope this helps!"
    )
    data = _parse_json_loose(content)
    assert data["probabilities"] == {"A": 0.6, "B": 0.4}


def test_loose_json_with_code_fences():
    content = '```json\n{"a": 1}\n```'
    assert _parse_json_loose(content) == {"a": 1}


def test_loose_json_raises_on_no_json():
    try:
        _parse_json_loose("just prose, no json")
        assert False, "should have raised"
    except ValueError:
        pass


def test_parse_response_with_prose():
    """End-to-end: prose + JSON survives."""
    content = (
        "Analyzing the matchup.\n"
        '{"probabilities": {"Yes": 0.7, "No": 0.3}}'
    )
    probs = _parse_response(content, ["Yes", "No"])
    assert probs == [0.7, 0.3]


# ---------- confidence-aware calibration ----------


def test_confidence_aware_barely_touches_uniform():
    """0.5/0.5 prediction should stay essentially uniform."""
    out = confidence_aware_shrink([0.5, 0.5])
    # confidence = 0, alpha = base = 0.05
    # (1-0.05)*0.5 + 0.05*0.5 = 0.5 exactly
    assert abs(out[0] - 0.5) < 1e-9
    assert abs(out[1] - 0.5) < 1e-9


def test_confidence_aware_moderate_shrinks_lightly():
    """0.7/0.3 → alpha = 0.05 + 0.30*(0.7 - 0.5) = 0.11"""
    out = confidence_aware_shrink([0.7, 0.3])
    expected_alpha = 0.05 + 0.30 * 0.2
    expected_0 = (1 - expected_alpha) * 0.7 + expected_alpha * 0.5
    expected_1 = (1 - expected_alpha) * 0.3 + expected_alpha * 0.5
    assert abs(out[0] - expected_0) < 1e-9
    assert abs(out[1] - expected_1) < 1e-9


def test_confidence_aware_extreme_shrinks_more():
    """0.95/0.05 should pull noticeably toward 0.5."""
    out = confidence_aware_shrink([0.95, 0.05])
    # confidence = 0.45, alpha = 0.05 + 0.30*0.45 = 0.185
    # 0.95 becomes (1-0.185)*0.95 + 0.185*0.5 ≈ 0.867
    # Sanity: pulled at least 0.05 toward the center
    assert out[0] < 0.95 - 0.05  # shrunk by at least 0.05
    assert out[1] > 0.05 + 0.05


def test_confidence_aware_preserves_sum():
    """Shrinkage toward 1/N preserves sum=1."""
    for probs in [[0.5, 0.5], [0.7, 0.3], [0.95, 0.05], [0.4, 0.35, 0.25], [0.6, 0.2, 0.1, 0.1]]:
        out = confidence_aware_shrink(probs)
        assert abs(sum(out) - 1.0) < 1e-9, f"sum drift on {probs}: {sum(out)}"


def test_confidence_aware_more_aggressive_than_fixed_at_extremes():
    """At high confidence, the new method should shrink more than fixed alpha=0.10."""
    extreme = [0.95, 0.05]
    fixed = shrink_to_uniform(extreme, alpha=0.10)
    aware = confidence_aware_shrink(extreme)
    # aware[0] should be smaller than fixed[0] (more shrinkage = closer to 0.5)
    assert aware[0] < fixed[0]
    assert aware[1] > fixed[1]


def test_confidence_aware_less_aggressive_than_fixed_at_uniform():
    """Near uniform, the new method should touch less than fixed alpha=0.10."""
    near_uniform = [0.52, 0.48]
    fixed = shrink_to_uniform(near_uniform, alpha=0.10)
    aware = confidence_aware_shrink(near_uniform)
    # fixed pulls a flat 10%; aware uses base=0.05 + tiny slope. Aware should
    # be closer to the original prediction.
    assert abs(aware[0] - 0.52) < abs(fixed[0] - 0.52)


def test_calibrate_dispatcher_modes():
    """All three modes should work."""
    p = [0.7, 0.3]
    cw = calibrate(p, mode="confidence_aware")
    fx = calibrate(p, mode="fixed")
    no = calibrate(p, mode="none")
    assert no == p
    assert cw != p
    assert fx != p


def test_calibrate_dispatcher_unknown_mode():
    try:
        calibrate([0.5, 0.5], mode="bogus")
        assert False, "should have raised"
    except ValueError:
        pass


def test_calibrate_single_outcome_forced_certainty():
    assert calibrate([0.5], mode="confidence_aware") == [1.0]
    assert calibrate([0.5], mode="fixed") == [1.0]


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
