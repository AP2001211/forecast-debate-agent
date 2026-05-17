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


# Step 1 default (kept for comparison)
DEFAULT_ALPHA = 0.10

# Step 3 defaults
DEFAULT_BASE = 0.05   # always-applied shrinkage
DEFAULT_SLOPE = 0.30  # additional shrinkage per unit of confidence-above-uniform


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


def calibrate(probs: list[float], mode: str = "confidence_aware") -> list[float]:
    """
    Top-level dispatcher. predict.py calls this.

    mode:
        "confidence_aware" - Step 3 default
        "fixed"            - Step 1 fixed-alpha shrinkage
        "none"             - identity (no calibration)
    """
    if mode == "confidence_aware":
        return confidence_aware_shrink(probs)
    if mode == "fixed":
        return shrink_to_uniform(probs)
    if mode == "none":
        return list(probs)
    raise ValueError(f"Unknown calibration mode: {mode!r}")