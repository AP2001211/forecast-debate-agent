"""
LLM forecaster — Step 3.

Changes from Step 1:
  - Aggressive Unicode punctuation normalization in label matching
    (fixes the Survivor curly-quote failure)
  - One retry on JSON parse failure, with a sterner "JSON only" reminder
    (fixes the SCOTUS plain-text failure)
  - Optional web search via OpenRouter ":online" model suffix
    (helps sports / entertainment events the model can't know from training)

Toggle web search via the USE_WEB_SEARCH constant or the
PROPHET_USE_WEB_SEARCH environment variable ("1" / "0").
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata

from openai import OpenAI

from utils import get_outcomes


# ---- configuration ---------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BASE_MODEL = "anthropic/claude-sonnet-4.5"

# Step 3c: toggle web search. The ":online" suffix tells OpenRouter to
# run a web search and inject results into the model's context before
# generation. Costs ~$0.005-0.01 extra per call.
USE_WEB_SEARCH = os.environ.get("PROPHET_USE_WEB_SEARCH", "1") == "1"

TEMPERATURE = 0.3
MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 45  # bumped from 25 — web search adds latency


def _build_client() -> OpenAI:
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


def _model_name() -> str:
    """Resolve the model string, with optional :online suffix."""
    return f"{BASE_MODEL}:online" if USE_WEB_SEARCH else BASE_MODEL


# ---- label normalization (Step 3a) -----------------------------------------

# Map "smart" Unicode punctuation to plain ASCII equivalents. This is the
# fix for the Survivor failure (model returned straight quotes, canonical
# label had curly quotes).
_UNICODE_PUNCT_FIXES = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201A": "'",  # single low-9 quotation mark
    "\u201B": "'",  # single high-reversed-9 quotation mark
    "\u201C": '"',  # left double quotation mark
    "\u201D": '"',  # right double quotation mark
    "\u201E": '"',  # double low-9 quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
    "\u00A0": " ",  # non-breaking space
    "\u2026": "...",  # ellipsis
}


def _normalize_label(s: str) -> str:
    """
    Aggressively normalize a label for fuzzy matching.
    - lowercase
    - strip leading/trailing whitespace
    - collapse internal whitespace
    - replace smart quotes / dashes / non-breaking spaces with ASCII
    - apply NFKC Unicode normalization (handles accent variants etc.)
    """
    s = unicodedata.normalize("NFKC", s)
    for orig, repl in _UNICODE_PUNCT_FIXES.items():
        s = s.replace(orig, repl)
    s = s.strip().lower()
    s = " ".join(s.split())  # collapse runs of whitespace
    return s


def _match_outcome(returned_label: str, outcomes: list[str]) -> str | None:
    """
    Match a model-returned label to one of the canonical outcomes.
    Strategy: exact -> normalized -> substring containment (only if unique).
    """
    if returned_label in outcomes:
        return returned_label

    norm_returned = _normalize_label(returned_label)
    for o in outcomes:
        if _normalize_label(o) == norm_returned:
            return o

    candidates = []
    for o in outcomes:
        no = _normalize_label(o)
        if no in norm_returned or norm_returned in no:
            candidates.append(o)
    if len(candidates) == 1:
        return candidates[0]

    return None


# ---- prompt construction ---------------------------------------------------

def _build_prompt(event: dict, outcomes: list[str]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the forecasting call."""
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
        "translate, or modify them in any way. Copy them character-for-character.\n\n"
        "OUTPUT FORMAT (strict):\n"
        "Respond with ONE JSON object and NOTHING ELSE. No prose. No markdown. "
        "No code fences. No explanation outside the JSON.\n\n"
        "Schema:\n"
        '{"reasoning": "<2-4 sentence analysis>", '
        '"probabilities": {"<outcome_label>": <float>, ...}}'
    )

    outcomes_display = "\n".join(f'  - "{o}"' for o in outcomes)
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
        f"Possible outcomes (use these EXACT labels in your JSON keys, "
        f"copied character-for-character including any punctuation):\n"
        f"{outcomes_display}\n\n"
        f"Assign a probability to each outcome. The probabilities must "
        f"sum to 1.0. Return ONLY the JSON object as specified."
    )

    return system_prompt, user_prompt


# ---- response parsing ------------------------------------------------------

def _parse_json_loose(content: str) -> dict:
    """
    Parse JSON from model output, tolerating common deviations:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - Trailing text after the JSON object
    """
    text = content.strip()

    # Strip code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the first balanced { ... } block from the text
    # (handles the case where the model adds prose before/after)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response. Raw: {text[:300]}")

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Could not parse extracted JSON: {e}\nExtracted: {text[start:i+1][:300]}"
                    )
    raise ValueError(f"Unbalanced braces in response. Raw: {text[:300]}")


def _extract_probs(
    response_data: dict, outcomes: list[str]
) -> list[float]:
    """
    Pull aligned probability list from a parsed JSON response.
    Raises ValueError on any mismatch.
    """
    if not isinstance(response_data, dict):
        raise ValueError(f"Expected JSON object, got {type(response_data).__name__}")

    probs_dict = response_data.get("probabilities")
    if not isinstance(probs_dict, dict):
        raise ValueError("Missing or non-dict 'probabilities' field")

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


def _parse_response(content: str, outcomes: list[str]) -> list[float]:
    """
    Parse the model's JSON response into a probability list aligned to
    `outcomes`. Raises ValueError if the response can't be mapped.
    """
    data = _parse_json_loose(content)
    return _extract_probs(data, outcomes)


# ---- main forecast call ----------------------------------------------------

def _call_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    extra_user_messages: list[dict] | None = None,
) -> str:
    """Make a single chat completion call and return raw content."""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if extra_user_messages:
        messages.extend(extra_user_messages)

    # NOTE: we do NOT use response_format={"type": "json_object"} when web
    # search is enabled, because some providers reject that combination.
    # Our parser is robust enough to handle prose-around-JSON anyway.
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if not USE_WEB_SEARCH:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content")
    return content


def forecast(event: dict) -> list[float]:
    """
    Run the LLM forecaster on an event.

    Returns a list of raw (un-calibrated) probabilities aligned to
    event['outcomes']. Length matches len(outcomes).

    Step 3 changes:
      - Tries the call once. If JSON parsing fails, retries ONCE with a
        sterner "JSON only" reminder. This rescues the SCOTUS-style
        prose-instead-of-JSON failure mode without doubling our cost on
        normal events.
    """
    outcomes = get_outcomes(event)
    if not outcomes:
        raise ValueError("Event has no outcomes; cannot forecast.")

    client = _build_client()
    system_prompt, user_prompt = _build_prompt(event, outcomes)
    model = _model_name()

    # First attempt
    try:
        content = _call_once(client, model, system_prompt, user_prompt)
        return _parse_response(content, outcomes)
    except ValueError as parse_err:
        # Probably a JSON or label-matching failure. Retry once with a
        # firmer reminder. Network/API errors won't get caught here.
        print(f"[agent] first attempt failed: {parse_err}", file=sys.stderr)
        print("[agent] retrying with stricter JSON reminder...", file=sys.stderr)

    # Retry: append the failed output context and a stern reminder.
    retry_message = {
        "role": "user",
        "content": (
            "Your previous response was not valid JSON or did not match the "
            "required schema. Return ONLY a single JSON object with the exact "
            "schema specified. No prose, no explanation outside the JSON, "
            "no markdown. Use the exact outcome labels provided, copied "
            "character-for-character."
        ),
    }
    content = _call_once(
        client, model, system_prompt, user_prompt,
        extra_user_messages=[retry_message],
    )
    return _parse_response(content, outcomes)