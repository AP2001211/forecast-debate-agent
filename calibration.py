"""
Probability calibration layer.

Purpose:
  - Keep agent.py focused on raw forecasting.
  - Apply all probability calibration here.
  - Avoid overconfident wrong predictions.
  - Handle unknown future categories by using probability shape and outcome count,
    not hardcoded category labels.

Modes:
  - "multi_outcome_aware" default
  - "confidence_aware"
  - "fixed"
  - "none"
"""

from __future__ import annotations

import os


DEFAULT_ALPHA = 0.10

DEFAULT_BASE = 0.05
DEFAULT_SLOPE = 0.30

DEFAULT_MULTI_N = 10
DEFAULT_MULTI_THRESH = 0.20

DEFAULT_LARGE_N = 15
DEFAULT_LARGE_BASE_BOOST = 0.03
DEFAULT_LARGE_SLOPE_BOOST = 0.08

DEFAULT_EXTREME_TOP = 0.75
DEFAULT_EXTREME_EXTRA_SHRINK = 0.08


def _normalize(probs: list[float]) -> list[float]:
    if not probs:
        return []

    cleaned = []
    for p in probs:
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        cleaned.append(max(0.0, min(1.0, p)))

    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)

    return [p / total for p in cleaned]


def shrink_to_uniform(probs: list[float], alpha: float = DEFAULT_ALPHA) -> list[float]:
    probs = _normalize(probs)
    n = len(probs)

    if n == 0:
        return []
    if n == 1:
        return [1.0]

    alpha = max(0.0, min(1.0, alpha))
    uniform_p = 1.0 / n

    calibrated = [
        (1.0 - alpha) * p + alpha * uniform_p
        for p in probs
    ]

    return _normalize(calibrated)


def confidence_aware_shrink(
    probs: list[float],
    base: float = DEFAULT_BASE,
    slope: float = DEFAULT_SLOPE,
) -> list[float]:
    probs = _normalize(probs)
    n = len(probs)

    if n == 0:
        return []
    if n == 1:
        return [1.0]

    uniform_p = 1.0 / n
    top_p = max(probs)
    confidence = max(0.0, top_p - uniform_p)

    alpha_eff = base + slope * confidence
    alpha_eff = max(0.0, min(1.0, alpha_eff))

    calibrated = [
        (1.0 - alpha_eff) * p + alpha_eff * uniform_p
        for p in probs
    ]

    return _normalize(calibrated)


def multi_outcome_aware(
    probs: list[float],
    base: float = DEFAULT_BASE,
    slope: float = DEFAULT_SLOPE,
    multi_n: int = DEFAULT_MULTI_N,
    multi_thresh: float = DEFAULT_MULTI_THRESH,
    large_n: int = DEFAULT_LARGE_N,
    large_base_boost: float = DEFAULT_LARGE_BASE_BOOST,
    large_slope_boost: float = DEFAULT_LARGE_SLOPE_BOOST,
    extreme_top: float = DEFAULT_EXTREME_TOP,
    extreme_extra_shrink: float = DEFAULT_EXTREME_EXTRA_SHRINK,
) -> list[float]:
    """
    Shape-aware calibration.

    Logic:
      1. Normalize first.
      2. If many outcomes and model has weak signal, return uniform.
      3. If many outcomes, apply slightly stronger shrink.
      4. If very high top probability, add a small humility tax.
      5. Never hard-cap strong market signals.
    """
    probs = _normalize(probs)
    n = len(probs)

    if n == 0:
        return []
    if n == 1:
        return [1.0]

    top_p = max(probs)

    # Large-field weak-signal case:
    # If top probability is very low, the model is basically guessing.
    if n >= multi_n and top_p < multi_thresh:
        return [1.0 / n] * n

    effective_base = base
    effective_slope = slope

    # Unknown final categories are handled by event shape:
    # large fields get more conservative calibration.
    if n >= large_n:
        effective_base += large_base_boost
        effective_slope += large_slope_boost

    # Very confident forecasts get a little extra shrink,
    # but not a destructive hard cap.
    if top_p >= extreme_top:
        effective_base += extreme_extra_shrink

    return confidence_aware_shrink(
        probs,
        base=effective_base,
        slope=effective_slope,
    )


def calibrate(probs: list[float], mode: str = "multi_outcome_aware") -> list[float]:
    """
    Dispatcher used by predict.py.

    Env vars:
      PROPHET_CAL_BASE
      PROPHET_CAL_SLOPE
      PROPHET_CAL_MULTI_N
      PROPHET_CAL_MULTI_THRESH
      PROPHET_CAL_LARGE_N
      PROPHET_CAL_LARGE_BASE_BOOST
      PROPHET_CAL_LARGE_SLOPE_BOOST
      PROPHET_CAL_EXTREME_TOP
      PROPHET_CAL_EXTREME_EXTRA_SHRINK
    """
    base = float(os.environ.get("PROPHET_CAL_BASE", DEFAULT_BASE))
    slope = float(os.environ.get("PROPHET_CAL_SLOPE", DEFAULT_SLOPE))

    multi_n = int(os.environ.get("PROPHET_CAL_MULTI_N", DEFAULT_MULTI_N))
    multi_thresh = float(os.environ.get("PROPHET_CAL_MULTI_THRESH", DEFAULT_MULTI_THRESH))

    large_n = int(os.environ.get("PROPHET_CAL_LARGE_N", DEFAULT_LARGE_N))
    large_base_boost = float(
        os.environ.get("PROPHET_CAL_LARGE_BASE_BOOST", DEFAULT_LARGE_BASE_BOOST)
    )
    large_slope_boost = float(
        os.environ.get("PROPHET_CAL_LARGE_SLOPE_BOOST", DEFAULT_LARGE_SLOPE_BOOST)
    )

    extreme_top = float(os.environ.get("PROPHET_CAL_EXTREME_TOP", DEFAULT_EXTREME_TOP))
    extreme_extra_shrink = float(
        os.environ.get("PROPHET_CAL_EXTREME_EXTRA_SHRINK", DEFAULT_EXTREME_EXTRA_SHRINK)
    )

    if mode == "multi_outcome_aware":
        return multi_outcome_aware(
            probs,
            base=base,
            slope=slope,
            multi_n=multi_n,
            multi_thresh=multi_thresh,
            large_n=large_n,
            large_base_boost=large_base_boost,
            large_slope_boost=large_slope_boost,
            extreme_top=extreme_top,
            extreme_extra_shrink=extreme_extra_shrink,
        )

    if mode == "confidence_aware":
        return confidence_aware_shrink(probs, base=base, slope=slope)

    if mode == "fixed":
        return shrink_to_uniform(probs)

    if mode == "none":
        return _normalize(probs)

    raise ValueError(f"Unknown calibration mode: {mode!r}")