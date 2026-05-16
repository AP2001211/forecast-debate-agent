"""
Single-call LLM forecaster.

Sends one event to Claude Sonnet 4.5 (via OpenRouter) and parses a JSON
response into raw outcome probabilities. Does NOT do calibration — that's
calibration.py's job. Does NOT do fallback — that's predict.py's job.

The contract: forecast(event) -> list[float] of length len(outcomes),
in the same order as event["outcomes"]. Raises on any failure
(parse error, API error, label mismatch, etc.) — the caller is responsible
for catching and falling back.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from utils import get_outcomes


# OpenRouter exposes an OpenAI-compatible API. We point the SDK at it.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model selection. Easy to swap by changing this string.
MODEL = "anthropic/claude-sonnet-4.5"

# Generation parameters. Low temperature for consistency; modest token cap
# since we only need a JSON object with ~N probabilities.
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Hard timeout per call. The grader allows 30s/event; we stay well under.
REQUEST_TIMEOUT_SECONDS = 25


def _build_client() -> OpenAI:
    """Build the OpenAI client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to your .env file "
            "or export it before running."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _build_prompt(event: dict, outcomes: list[str]) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the forecasting call.

    The system prompt sets the role and output contract.
    The user prompt contains the event details.
    """
    system_prompt = (
        "You are a careful probabilistic forecaster. Your job is to assign "
        "calibrated probabilities to the possible outcomes of real-world "
        "events. You will be scored by Brier score, which rewards being "
        "well-calibrated and punishes overconfident wrong predictions.\n\n"
        "Important principles:\n"
        "- Think about both sides before committing to a probability.\n"
        "- Avoid extreme probabilities (0.95+ or 0.05-) unless the evidence "
        "is overwhelming. For uncertain events, stay closer to balanced.\n"
        "- Your probabilities must sum to 1 across all outcomes.\n"
        "- Use the EXACT outcome labels provided. Do not paraphrase, "
        "translate, or modify them in any way.\n\n"
        "Respond ONLY with a single JSON object of the form:\n"
        '{"reasoning": "<2-4 sentence analysis>", '
        '"probabilities": {"<outcome_label>": <float>, ...}}\n'
        "No other text. No markdown fences. Pure JSON."
    )

    # Format outcomes as a quoted list to make exact-match clearer to the model
    outcomes_display = "\n".join(f'  - "{o}"' for o in outcomes)

    # Pull optional fields safely
    description = event.get("description") or ""
    category = event.get("category") or "Unknown"
    rules = event.get("rules") or ""
    close_time = event.get("close_time") or "unknown"

    user_prompt = (
        f"Event: {event.get('title', '(no title)')}\n"
        f"Category: {category}\n"
        f"Description: {description}\n"
        f"Resolution rules: {rules}\n"
        f"Closes at: {close_time}\n\n"
        f"Possible outcomes (use these EXACT labels in your JSON keys):\n"
        f"{outcomes_display}\n\n"
        f"Assign a probability to each outcome. The probabilities must "
        f"sum to 1.0. Return only the JSON object as specified."
    )

    return system_prompt, user_prompt


def _normalize_label(s: str) -> str:
    """Aggressively normalize a label for fuzzy matching."""
    return s.strip().lower()


def _match_outcome(returned_label: str, outcomes: list[str]) -> str | None:
    """
    Try to match a model-returned label to one of the canonical outcomes.

    Strategy:
      1. Exact match.
      2. Case-insensitive + whitespace-stripped match.
      3. Substring containment (returned in canonical, or canonical in
         returned). Helpful when the model says "Pittsburgh Steelers" but
         the canonical is "Pittsburgh".

    Returns the canonical outcome label, or None if no match.
    """
    if returned_label in outcomes:
        return returned_label

    norm_returned = _normalize_label(returned_label)
    for o in outcomes:
        if _normalize_label(o) == norm_returned:
            return o

    # Substring containment — be careful not to match too aggressively.
    # Only match if exactly one outcome is contained / contains.
    candidates = []
    for o in outcomes:
        no = _normalize_label(o)
        if no in norm_returned or norm_returned in no:
            candidates.append(o)
    if len(candidates) == 1:
        return candidates[0]

    return None


def _parse_response(content: str, outcomes: list[str]) -> list[float]:
    """
    Parse the model's JSON response into a probability list aligned to
    `outcomes`. Raises ValueError if the response can't be mapped.
    """
    # Strip markdown fences if the model added them despite instructions
    text = content.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse model response as JSON: {e}\nRaw: {text[:300]}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    probs_dict = data.get("probabilities")
    if not isinstance(probs_dict, dict):
        raise ValueError("Missing or non-dict 'probabilities' field")

    # Build a list aligned to the canonical outcomes order
    result: list[float] = []
    matched_returned: set[str] = set()
    for canonical in outcomes:
        found_prob: float | None = None
        for returned_label, returned_prob in probs_dict.items():
            if returned_label in matched_returned:
                continue
            matched = _match_outcome(returned_label, outcomes)
            if matched == canonical:
                try:
                    found_prob = float(returned_prob)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Non-numeric probability for {returned_label!r}: {returned_prob!r}"
                    )
                matched_returned.add(returned_label)
                break
        if found_prob is None:
            raise ValueError(
                f"No probability returned for outcome {canonical!r}. "
                f"Model returned: {list(probs_dict.keys())}"
            )
        result.append(found_prob)

    return result


def forecast(event: dict) -> list[float]:
    """
    Run the single-call LLM forecaster on an event.

    Returns a list of raw (un-calibrated) probabilities aligned to
    event['outcomes']. Length matches len(outcomes). May not sum to
    exactly 1.0 — the caller normalizes.

    Raises on any failure (network, parse, label mismatch). The caller
    in predict.py catches all exceptions and falls back to uniform.
    """
    outcomes = get_outcomes(event)
    if not outcomes:
        raise ValueError("Event has no outcomes; cannot forecast.")

    client = _build_client()
    system_prompt, user_prompt = _build_prompt(event, outcomes)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content")

    return _parse_response(content, outcomes)
