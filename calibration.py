"""
Probability calibration layer — Step 3.

Three modes available:

1. `shrink_to_uniform(probs, alpha)` — fixed-strength shrinkage toward
   1/N. The original Step 1 baseline. Kept for ablation.

2. `confidence_aware_shrink(probs, base, slope)` — Step 3 default.
   Shrinks more aggressively when the top probability is far from
   uniform. This protects against confident-wrong predictions, which
   pay the largest Brier penalty.

3. `calibrate(probs, mode)` — dispatcher used by predict.py.

Math for confidence-aware shrink:
    confidence = max(probs) - 1/N
    alpha_eff = base + slope * confidence
    p_calibrated = (1 - alpha_eff) * p_raw + alpha_eff * (1/N)

With base=0.05 and slope=0.30:
  - 0.50/0.50  → alpha=0.05 → barely touched
  - 0.70/0.30  → alpha=0.11 → light shrink
  - 0.90/0.10  → alpha=0.17 → harder shrink
  - 0.99/0.01  → alpha=0.20 → harder still

This matches the observed failure mode in Step 2 (Pakistan 0.68 →
Bangladesh won, Brier 0.92): we want confident calls to pay a humility
tax before they get to be wrong.
"""

from __future__ import annotations

import os


# Step 1 default (kept for comparison)
DEFAULT_ALPHA = 0.10

# Step 3 defaults
# Lowered so we trust market-anchored outputs instead of pulling them toward uniform.
DEFAULT_BASE = float(os.environ.get("PROPHET_CAL_BASE", "0.02"))
DEFAULT_SLOPE = float(os.environ.get("PROPHET_CAL_SLOPE", "0.15"))

# Step 5 defaults for multi-outcome rule
DEFAULT_MULTI_N = 10       # minimum outcome count to trigger the rule
DEFAULT_MULTI_THRESH = 0.20  # top-prob threshold below which we force uniform


def shrink_to_uniform(probs: list[float], alpha: float = DEFAULT_ALPHA) -> list[float]:
    """
    Fixed-strength shrinkage: p_cal = (1-alpha)*p + alpha*(1/N).
    Used as ablation baseline.
    """
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    a = max(0.0, min(1.0, alpha))
    uniform_p = 1.0 / n
    return [(1.0 - a) * p + a * uniform_p for p in probs]


def confidence_aware_shrink(
    probs: list[float],
    base: float = DEFAULT_BASE,
    slope: float = DEFAULT_SLOPE,
) -> list[float]:
    """
    Confidence-aware shrinkage: shrink harder when more confident.

    Args:
        probs: raw probabilities
        base: minimum shrinkage even at zero confidence
        slope: additional shrinkage per unit of (max(p) - 1/N)

    Returns calibrated probabilities. Preserves sum if input sums to 1
    (shrinkage toward uniform preserves total mass).
    """
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    uniform_p = 1.0 / n
    confidence = max(probs) - uniform_p  # in [0, 1 - 1/N], roughly

    alpha_eff = base + slope * confidence
    alpha_eff = max(0.0, min(1.0, alpha_eff))

    return [(1.0 - alpha_eff) * p + alpha_eff * uniform_p for p in probs]


def multi_outcome_aware(
    probs: list[float],
    base: float = DEFAULT_BASE,
    slope: float = DEFAULT_SLOPE,
    multi_n: int = DEFAULT_MULTI_N,
    multi_thresh: float = DEFAULT_MULTI_THRESH,
) -> list[float]:
    """
    Multi-outcome aware calibration (Step 5 default).

    If the event has >= multi_n outcomes AND the model's top prob is below
    multi_thresh (i.e. the model has no real signal), return exactly uniform.
    This avoids paying the Brier penalty for noisy near-uniform distributions
    on large multi-outcome events like reality TV.

    Otherwise delegates to confidence_aware_shrink.
    """
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    if n >= multi_n and max(probs) < multi_thresh:
        return [1.0 / n] * n

    return confidence_aware_shrink(probs, base=base, slope=slope)


def calibrate(probs: list[float], mode: str = "multi_outcome_aware") -> list[float]:
    """
    Top-level dispatcher. predict.py calls this.

    Reads calibration params from env vars (PROPHET_CAL_BASE, PROPHET_CAL_SLOPE,
    PROPHET_CAL_MULTI_N, PROPHET_CAL_MULTI_THRESH) so tuning doesn't require
    code changes.

    mode:
        "multi_outcome_aware" - Step 5 default (confidence-aware + uniform cutoff)
        "confidence_aware"    - Step 3 (no multi-outcome cutoff)
        "fixed"               - Step 1 fixed-alpha shrinkage
        "none"                - identity (no calibration)
    """
    base = float(os.environ.get("PROPHET_CAL_BASE", DEFAULT_BASE))
    slope = float(os.environ.get("PROPHET_CAL_SLOPE", DEFAULT_SLOPE))
    multi_n = int(os.environ.get("PROPHET_CAL_MULTI_N", DEFAULT_MULTI_N))
    multi_thresh = float(os.environ.get("PROPHET_CAL_MULTI_THRESH", DEFAULT_MULTI_THRESH))

    if mode == "multi_outcome_aware":
        return multi_outcome_aware(probs, base=base, slope=slope,
                                   multi_n=multi_n, multi_thresh=multi_thresh)
    if mode == "confidence_aware":
        return confidence_aware_shrink(probs, base=base, slope=slope)
    if mode == "fixed":
        return shrink_to_uniform(probs)
    if mode == "none":
        return list(probs)
    raise ValueError(f"Unknown calibration mode: {mode!r}")