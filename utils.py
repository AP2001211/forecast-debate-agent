"""
Utility helpers for the forecasting agent.

These are dependency-free helpers used by predict.py and the rest of the
pipeline. The most important function here is `safe_fallback`, which
guarantees we always return a valid distribution no matter what blows up
upstream.
"""

from __future__ import annotations

from typing import Any


def get_outcomes(event: dict) -> list[str]:
    """
    Extract the outcomes list from an event dict, with light cleanup.

    The grader's event dict always has 'outcomes' as a list of strings, but
    we defensively strip whitespace and drop empties so we never produce
    mismatched labels.
    """
    raw = event.get("outcomes") or []
    cleaned: list[str] = []
    for o in raw:
        if o is None:
            continue
        s = str(o).strip()
        if s:
            cleaned.append(s)
    return cleaned


def uniform_distribution(outcomes: list[str]) -> list[dict]:
    """
    Return a uniform probability distribution over the given outcomes.

    Used as the ultimate fallback when everything else fails. If outcomes
    is empty (shouldn't happen but we're paranoid), returns an empty list
    — the caller is responsible for producing a valid output even in that
    edge case.
    """
    n = len(outcomes)
    if n == 0:
        return []
    p = 1.0 / n
    return [{"market": o, "probability": p} for o in outcomes]


def normalize_probabilities(
    outcomes: list[str], probs: list[float]
) -> list[dict]:
    """
    Build the final output structure, ensuring:
      - exactly one entry per outcome (in the original outcome order)
      - every probability is between 0 and 1
      - probabilities sum to exactly 1.0 (renormalized if needed)
      - no NaN / inf / negative values sneak through

    If anything is invalid, we fall back to uniform over `outcomes`.
    """
    if len(outcomes) == 0:
        return []

    if len(probs) != len(outcomes):
        return uniform_distribution(outcomes)

    # Sanitize: clamp to [0, 1], replace bad values with a tiny epsilon
    EPS = 1e-9
    clean: list[float] = []
    for p in probs:
        try:
            pv = float(p)
        except (TypeError, ValueError):
            pv = EPS
        # Reject NaN / inf
        if pv != pv or pv in (float("inf"), float("-inf")):
            pv = EPS
        if pv < 0:
            pv = EPS
        if pv > 1:
            pv = 1.0
        clean.append(pv)

    total = sum(clean)
    if total <= 0:
        # All zeros (or worse) — fall back to uniform
        return uniform_distribution(outcomes)

    # Renormalize so they sum to exactly 1
    normalized = [p / total for p in clean]

    # Tiny correction for floating-point drift: push any rounding error
    # into the largest entry so the sum is exactly 1.0
    drift = 1.0 - sum(normalized)
    if abs(drift) > 0:
        idx = normalized.index(max(normalized))
        normalized[idx] += drift

    return [
        {"market": o, "probability": p}
        for o, p in zip(outcomes, normalized)
    ]


def safe_fallback(event: dict) -> dict:
    """
    The ultimate safety net. Returns a valid uniform distribution over the
    event's outcomes, in the exact output shape the grader expects.

    This is what we return when the LLM call fails, JSON parsing fails,
    labels don't match, or anything else goes wrong. Better a uniform
    prediction (Brier ~0.25 on binary) than a crash (completion penalty).
    """
    outcomes = get_outcomes(event)

    if not outcomes:
        # Pathological event with no outcomes — return an empty list.
        # The grader will likely reject this, but there's nothing useful
        # we can produce. Log it for debugging.
        return {"probabilities": []}

    return {"probabilities": uniform_distribution(outcomes)}


def validate_output(output: Any, event: dict) -> bool:
    """
    Sanity-check that an output dict satisfies the grader's contract.
    Returns True if valid, False otherwise. Used in tests.

    Contract:
      - top-level dict with key 'probabilities'
      - value is a list of {'market': str, 'probability': float}
      - every outcome in event['outcomes'] appears exactly once
      - no extra markets
      - every probability in [0, 1]
      - probabilities sum to 1.0 (within tolerance)
    """
    if not isinstance(output, dict):
        return False
    if "probabilities" not in output:
        return False

    probs = output["probabilities"]
    if not isinstance(probs, list):
        return False

    expected = get_outcomes(event)
    if len(probs) != len(expected):
        return False

    seen_markets: set[str] = set()
    total = 0.0
    for item in probs:
        if not isinstance(item, dict):
            return False
        if "market" not in item or "probability" not in item:
            return False
        market = item["market"]
        prob = item["probability"]
        if not isinstance(market, str):
            return False
        if market not in expected:
            return False
        if market in seen_markets:
            return False
        seen_markets.add(market)
        try:
            pv = float(prob)
        except (TypeError, ValueError):
            return False
        if pv < 0 or pv > 1:
            return False
        if pv != pv:  # NaN
            return False
        total += pv

    if seen_markets != set(expected):
        return False

    if abs(total - 1.0) > 1e-6:
        return False

    return True
