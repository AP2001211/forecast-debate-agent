"""
LLM forecaster — Step 4.

Changes from Step 3:
  - Search-reframing in the system prompt: explicitly instructs the model
    to use search for pre-event context (odds, form, news) and to ignore
    any post-event results that appear in search results.

This is important because the resolved-dataset eval can leak (search
returns the actual outcome), and at real submission time the events
will be open — so the model needs the habit of reasoning from pre-event
information regardless.
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

USE_WEB_SEARCH = os.environ.get("PROPHET_USE_WEB_SEARCH", "1") == "1"

TEMPERATURE = 0.3
MAX_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 45


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
    return f"{BASE_MODEL}:online" if USE_WEB_SEARCH else BASE_MODEL


# ---- label normalization ---------------------------------------------------

_UNICODE_PUNCT_FIXES = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00A0": " ", "\u2026": "...",
}


def _normalize_label(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for orig, repl in _UNICODE_PUNCT_FIXES.items():
        s = s.replace(orig, repl)
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def _match_outcome(returned_label: str, outcomes: list[str]) -> str | None:
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


# ---- prompt construction (Step 4: search reframing) ------------------------

# This system prompt is intentionally explicit about treating the task as
# forecasting, not lookup. Even when search results contain the outcome,
# we want the model to reason from pre-event context. At real submission
# time the outcome won't be available anyway, so building this habit now
# means our resolved-dataset evals more closely reflect submission behavior.
_SYSTEM_PROMPT = """You are a careful probabilistic forecaster. Your job is to assign calibrated probabilities to possible outcomes of real-world events. You will be scored by Brier score, which rewards calibration and punishes overconfident wrong predictions.

CRITICAL RULES FOR USING SEARCH:
You may have access to web search. If you do, use it for PRE-EVENT context only:
  - Recent form, head-to-head records, injury reports
  - Betting odds, market prices, expert predictions
  - News about the event leading up to it
  - Historical base rates for similar events

If search results show what already happened (a final score, an announced winner, an election result), you MUST IGNORE that information. Pretend you don't see it. Reason about what would have been predictable BEFORE the event using only pre-event context.

Why this matters: at submission time you will see events whose outcomes haven't happened yet. The skill is forecasting from incomplete information, not retrieval. A model that just looks up resolved results is useless. A model that reasons from pre-event signals is what we need.

CALIBRATION PRINCIPLES:
- Think about both sides before committing to a probability.
- Avoid extreme probabilities (above 0.95 or below 0.05) unless evidence is overwhelming.
- For events where you have little pre-event signal, stay near the uniform prior (1/N over N outcomes).
- Your probabilities must sum to 1.0 across all outcomes.

LABEL HANDLING:
Use the EXACT outcome labels provided. Copy them character-for-character, including any punctuation marks (smart quotes, accents, hyphens). Do not paraphrase or translate.

OUTPUT FORMAT (strict):
Respond with ONE JSON object and NOTHING ELSE. No prose. No markdown. No code fences. No explanation outside the JSON.

Schema:
{"reasoning": "<2-4 sentence pre-event analysis>", "probabilities": {"<outcome_label>": <float>, ...}}"""


def _build_prompt(event: dict, outcomes: list[str]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the forecasting call."""
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
        f"Reason about what would have been predictable BEFORE this event "
        f"using pre-event signals only. If search results reveal the actual "
        f"outcome, ignore that and forecast as if you didn't know.\n\n"
        f"Return ONLY the JSON object."
    )

    return _SYSTEM_PROMPT, user_prompt


# ---- response parsing ------------------------------------------------------

def _parse_json_loose(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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


def _extract_probs(response_data: dict, outcomes: list[str]) -> list[float]:
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
    data = _parse_json_loose(content)
    return _extract_probs(data, outcomes)


# ---- main forecast call ----------------------------------------------------

def _call_once(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str,
    extra_user_messages: list[dict] | None = None,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if extra_user_messages:
        messages.extend(extra_user_messages)

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
    Retries once on parse/match failure with a sterner reminder.
    """
    outcomes = get_outcomes(event)
    if not outcomes:
        raise ValueError("Event has no outcomes; cannot forecast.")

    client = _build_client()
    system_prompt, user_prompt = _build_prompt(event, outcomes)
    model = _model_name()

    try:
        content = _call_once(client, model, system_prompt, user_prompt)
        return _parse_response(content, outcomes)
    except ValueError as parse_err:
        print(f"[agent] first attempt failed: {parse_err}", file=sys.stderr)
        print("[agent] retrying with stricter JSON reminder...", file=sys.stderr)

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