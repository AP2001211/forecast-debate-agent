"""
Probability calibration layer.

The LLM produces raw probabilities. Brier score penalizes overconfident
wrong answers heavily, so we apply a calibration step that pulls
probabilities toward the uniform prior (1/N over N outcomes).

The shrinkage formula:
    p_calibrated = (1 - alpha) * p_raw + alpha * (1 / N)

With alpha=0.10, a raw probability of 0.90 on a binary becomes 0.86.
A raw probability of 0.50 stays at 0.50. The pull is symmetric around
the uniform prior.

This is a deliberately simple baseline. In Step 4+ we'll consider:
  - alpha tuned per-category (Sports vs Economics vs Entertainment)
  - alpha that scales with the LLM's self-reported confidence
  - non-linear methods like temperature scaling, fitted on sample-resolved
"""

from __future__ import annotations


# Strength of shrinkage toward uniform prior.
# 0.0 = no calibration (trust the model fully)
# 1.0 = full shrinkage (always return uniform)
# 0.10 = pull 10% of the way toward uniform. Light touch for a strong
#        model that's already reasonably well-calibrated.
DEFAULT_ALPHA = 0.10


def shrink_to_uniform(probs: list[float], alpha: float = DEFAULT_ALPHA) -> list[float]:
    """
    Pull each probability `alpha` of the way toward the uniform prior 1/N.

    Args:
        probs: raw probabilities. Assumed (but not required) to be non-negative.
        alpha: shrinkage strength in [0, 1]. Clamped if outside.

    Returns:
        Calibrated probabilities, same length as input. Does NOT renormalize
        to sum to 1 — the caller (predict.py's normalize_probabilities) handles
        that. We don't need to here because shrinkage toward 1/N preserves
        any sum=1 input as sum=1 output.

    Special cases:
        - Empty list: returns empty list.
        - Single outcome: returns [1.0] regardless of input.
    """
    n = len(probs)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Clamp alpha to [0, 1]
    a = max(0.0, min(1.0, alpha))
    uniform_p = 1.0 / n

    return [(1.0 - a) * p + a * uniform_p for p in probs]
