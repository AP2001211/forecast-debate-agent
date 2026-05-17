"""
LLM forecaster — market-aware, leakage-filtered evidence, raw probabilities.

Important design:
  - Evidence pass can use web search.
  - Evidence pass filters out resolved/post-event information.
  - Final forecast uses the filtered evidence brief.
  - Final forecast is non-online by default to reduce leakage.
  - Calibration is NOT done here. predict.py -> calibration.py handles it.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from utils import get_outcomes


# ---- configuration ---------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
BASE_MODEL = "anthropic/claude-sonnet-4.5"

USE_WEB_SEARCH = os.environ.get("PROPHET_USE_WEB_SEARCH", "1") == "1"
USE_EVIDENCE_BRIEF = os.environ.get("PROPHET_USE_EVIDENCE_BRIEF", "1") == "1"
FINAL_FORECAST_USE_WEB_SEARCH = os.environ.get("PROPHET_FINAL_USE_WEB_SEARCH", "0") == "1"

USE_KALSHI_MARKET_DATA = os.environ.get("PROPHET_USE_KALSHI_MARKET_DATA", "1") == "1"
ALLOW_CLOSED_MARKET_DATA = os.environ.get("PROPHET_ALLOW_CLOSED_MARKET_DATA", "0") == "1"

TEMPERATURE = 0.3
MAX_TOKENS = 1024
EVIDENCE_MAX_TOKENS = 1200
REQUEST_TIMEOUT_SECONDS = 45
KALSHI_TIMEOUT_SECONDS = 4
MAX_KALSHI_MARKETS = 60
MIN_PROB = 1e-6

_KALSHI_CACHE: dict[str, dict | None] = {}


def _build_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The openai package is not installed. Install dependencies with "
            "`pip install -r requirements.txt` before running live forecasts."
        ) from e

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _online_model_name() -> str:
    return f"{BASE_MODEL}:online" if USE_WEB_SEARCH else BASE_MODEL


def _forecast_model_name() -> str:
    return f"{BASE_MODEL}:online" if FINAL_FORECAST_USE_WEB_SEARCH else BASE_MODEL


# ---- helpers ---------------------------------------------------------------

def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_probs(probs: list[float]) -> list[float]:
    cleaned = []
    for p in probs:
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        cleaned.append(max(MIN_PROB, min(1.0, p)))

    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)

    return [p / total for p in cleaned]


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

    return candidates[0] if len(candidates) == 1 else None


# ---- prompts ---------------------------------------------------------------

_SYSTEM_PROMPT = """You are a careful probabilistic forecaster. Your job is to assign calibrated probabilities to possible outcomes of real-world events. You will be scored by Brier score, which rewards calibration and punishes overconfident wrong predictions.

CRITICAL RULES:
- Forecast from information that would have been available before the event close time.
- If you see final results, winners, scores, resolved outcomes, vote totals, eliminations, or post-event summaries, ignore them.
- Use market-implied probabilities as a strong prior when available, but do not blindly copy the market.
- Look for small justified edges from injuries, breaking news, matchup dynamics, polling shifts, procedural developments, or public overreaction.
- Avoid double-counting correlated evidence.
- If sources disagree strongly, increase uncertainty.
- Do not assume famous, popular, incumbent, or highly visible outcomes are more likely unless supported by evidence or market pricing.
- For large multi-outcome markets, rank plausible favorites and give longshots small but nonzero probabilities.
- Your probabilities must sum to 1.0 across all outcomes.

LABEL HANDLING:
Use the EXACT outcome labels provided. Copy them character-for-character.

OUTPUT FORMAT:
Respond with ONE JSON object and NOTHING ELSE.

Schema:
{"reasoning": "<2-4 sentence analysis>", "probabilities": {"<outcome_label>": <float>, ...}}"""


_EVIDENCE_SYSTEM_PROMPT = """You are a leakage-filtering research analyst for a probabilistic forecasting agent.

Your job is NOT to predict the final answer. Your job is to gather only information that would have been knowable before the event close time, and filter out anything that reveals or strongly implies the resolved outcome.

Rules:
- Use search when available.
- Treat the event close time as the cutoff.
- Any source published after the close time is post-event and must be discarded.
- Any source that states the final score, winner, eliminated contestant, election result, award winner, vote count, or resolved outcome must be discarded.
- If a source mixes pre-event context with resolved outcome information, only keep the clearly pre-event parts.
- Do not include final results in the usable evidence.
- Do not include phrasing like "eventually won", "went on to win", "defeated", "was eliminated", "was announced as winner", or equivalent resolved-outcome statements.
- If unsure whether a fact leaks the outcome, put it in discarded_post_event_or_resolution_info, not usable_pre_event_evidence.

Prioritize usable pre-event:
1. Market-implied probabilities, prediction-market prices, sportsbook odds, polling/model averages, or futures odds available before close.
2. Current standings, injuries, form, head-to-head, polling, endorsements, fundamentals, or expert forecasts available before close.
3. Uncertainty, stale-data warnings, thin-market warnings, conflicting signals, and market disagreement.

Return ONE JSON object and nothing else.

Schema:
{
  "market_prior": "<pre-event market/odds/polling prior only, or 'not found'>",
  "usable_pre_event_evidence": ["<only facts knowable before close>", "..."],
  "uncertainties": ["<uncertainty or caveat>", "..."],
  "discarded_post_event_or_resolution_info": ["<brief reason info was discarded, without revealing the answer if possible>", "..."],
  "leakage_risk": "low|medium|high",
  "suggested_baseline": "<how the forecaster should anchor before calibration>"
}"""


def _market_search_hint(event: dict, outcomes: list[str]) -> str:
    title = event.get("title", "(no title)")
    category = (event.get("category") or "").lower()
    n = len(outcomes)

    generic = (
        f'Search for market prices and odds for "{title}". Prefer recent '
        "pre-event sources and map them onto the exact outcome labels."
    )

    if "sport" in category:
        if n == 2:
            return (
                generic
                + " For this head-to-head sports market, prioritize moneyline odds, "
                  "injuries, home/away, rest, recent form, and matchup news."
            )
        return (
            generic
            + " For this season/award/tournament sports market, prioritize futures odds, "
              "standings/brackets, remaining schedule, injuries, and expert consensus."
        )

    if "election" in category or "politic" in category:
        return (
            generic
            + " For this political market, prioritize prediction-market prices, "
              "polling averages, election models, endorsements, fundraising, and fundamentals."
        )

    if "economic" in category:
        return (
            generic
            + " For this economics market, prioritize consensus forecasts, recent data, "
              "central-bank guidance, inflation/GDP/unemployment trends, and adjacent buckets."
        )

    if "entertainment" in category:
        return (
            generic
            + " For this entertainment market, prioritize odds, credible pre-event leaks, "
              "episode timing, fandom consensus, award predictors, and avoid fame-only reasoning."
        )

    if n >= 8:
        return (
            generic
            + " This is a large multi-outcome market, so search for ranked odds, "
              "forecast lists, or expert consensus."
        )

    return generic


# ---- Kalshi retrieval ------------------------------------------------------

def _http_get_json(url: str, timeout: int = KALSHI_TIMEOUT_SECONDS) -> dict | None:
    if url in _KALSHI_CACHE:
        return _KALSHI_CACHE[url]

    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "prophet-agent/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"[agent] market data fetch failed for {url}: {e}", file=sys.stderr)
        _KALSHI_CACHE[url] = None
        return None

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"[agent] market data JSON parse failed for {url}: {e}", file=sys.stderr)
        _KALSHI_CACHE[url] = None
        return None

    _KALSHI_CACHE[url] = data
    return data


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_to_prob(value: object, *, is_cents: bool = False) -> float | None:
    price = _as_float(value)
    if price is None:
        return None
    if is_cents or price > 1.0:
        price = price / 100.0
    if 0.0 <= price <= 1.0:
        return price
    return None


def _best_orderbook_price(orderbook: dict, side: str) -> float | None:
    candidates = []

    for key in (side, f"{side}_dollars", f"{side}_cents"):
        levels = orderbook.get(key)
        if not isinstance(levels, list):
            continue

        for level in levels:
            if not isinstance(level, list) or not level:
                continue
            price = _price_to_prob(level[0], is_cents=key.endswith("_cents"))
            if price is not None:
                candidates.append(price)

    return max(candidates) if candidates else None


def _market_probability_from_orderbook(market_ticker: str) -> dict | None:
    url = f"{KALSHI_BASE_URL}/markets/{urllib.parse.quote(market_ticker)}/orderbook"
    data = _http_get_json(url)
    if not data:
        return None

    orderbook = data.get("orderbook_fp") or data.get("orderbook")
    if not isinstance(orderbook, dict):
        return None

    yes_bid = _best_orderbook_price(orderbook, "yes")
    no_bid = _best_orderbook_price(orderbook, "no")
    yes_ask = 1.0 - no_bid if no_bid is not None else None

    if yes_bid is not None and yes_ask is not None:
        yes_prob = (yes_bid + yes_ask) / 2.0
    elif yes_bid is not None:
        yes_prob = yes_bid
    elif yes_ask is not None:
        yes_prob = yes_ask
    else:
        return None

    yes_prob = max(0.0, min(1.0, yes_prob))

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_mid": yes_prob,
        "no_mid": 1.0 - yes_prob,
    }


def _fetch_kalshi_market(market_ticker: str) -> dict | None:
    url = f"{KALSHI_BASE_URL}/markets/{urllib.parse.quote(market_ticker)}"
    data = _http_get_json(url)
    market = data.get("market") if isinstance(data, dict) else None
    return market if isinstance(market, dict) else None


def _fetch_kalshi_event_markets(event_ticker: str) -> list[dict]:
    url = (
        f"{KALSHI_BASE_URL}/events/{urllib.parse.quote(event_ticker)}"
        "?with_nested_markets=true"
    )
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return []

    event = data.get("event")
    if isinstance(event, dict) and isinstance(event.get("markets"), list):
        return [m for m in event["markets"] if isinstance(m, dict)]

    if isinstance(data.get("markets"), list):
        return [m for m in data["markets"] if isinstance(m, dict)]

    return []


def _market_label(market: dict) -> str:
    return str(
        market.get("yes_sub_title")
        or market.get("subtitle")
        or market.get("title")
        or market.get("ticker")
        or market.get("market_ticker")
        or ""
    )


def _market_yes_probability(market: dict) -> float | None:
    yes_bid = _price_to_prob(
        _first_not_none(market.get("yes_bid"), market.get("yes_bid_dollars")),
        is_cents=market.get("yes_bid") is not None,
    )
    yes_ask = _price_to_prob(
        _first_not_none(market.get("yes_ask"), market.get("yes_ask_dollars")),
        is_cents=market.get("yes_ask") is not None,
    )

    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0

    for key in ("last_price", "last_price_dollars"):
        prob = _price_to_prob(market.get(key), is_cents=not key.endswith("_dollars"))
        if prob is not None:
            return prob

    if yes_bid is not None:
        return yes_bid
    if yes_ask is not None:
        return yes_ask

    ticker = str(market.get("ticker") or market.get("market_ticker") or "")
    if ticker:
        book = _market_probability_from_orderbook(ticker)
        if book:
            return _as_float(book.get("yes_mid"))

    return None


def _match_market_to_outcome(market: dict, outcomes: list[str]) -> str | None:
    label = _normalize_label(_market_label(market))
    ticker = _normalize_label(str(market.get("ticker") or market.get("market_ticker") or ""))

    matches = []
    for outcome in outcomes:
        no = _normalize_label(outcome)
        if no and (no in label or no in ticker):
            matches.append(outcome)

    return matches[0] if len(matches) == 1 else None


def _format_prob(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1%}"


def _event_is_closed(event: dict) -> bool:
    close_time = event.get("close_time")
    if not close_time:
        return False

    try:
        close_dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
    except ValueError:
        return False

    if close_dt.tzinfo is None:
        close_dt = close_dt.replace(tzinfo=timezone.utc)

    return close_dt <= datetime.now(timezone.utc)


def _kalshi_market_data_hint(event: dict, outcomes: list[str]) -> str | None:
    if not USE_KALSHI_MARKET_DATA:
        return None

    if not ALLOW_CLOSED_MARKET_DATA and _event_is_closed(event):
        return None

    market_ticker = str(event.get("market_ticker") or "")
    event_ticker = str(event.get("event_ticker") or market_ticker or "")

    if not market_ticker and not event_ticker:
        return None

    lines = []

    if market_ticker and len(outcomes) == 2:
        market = _fetch_kalshi_market(market_ticker) or {}
        book = _market_probability_from_orderbook(market_ticker)

        if book:
            yes_prob = _as_float(book.get("yes_mid"))
            yes_label = _market_label(market) or "YES"

            lines.append(
                "Kalshi explicit retrieval: "
                f"{market_ticker} orderbook implies YES '{yes_label}' around "
                f"{_format_prob(yes_prob)} "
                f"(best bid {_format_prob(book.get('yes_bid'))}, "
                f"best ask {_format_prob(book.get('yes_ask'))}). "
                "Map YES to the matching outcome label if clear."
            )

    if event_ticker:
        markets = _fetch_kalshi_event_markets(event_ticker)
        matched = []

        for market in markets[:MAX_KALSHI_MARKETS]:
            outcome = _match_market_to_outcome(market, outcomes)
            prob = _market_yes_probability(market)
            ticker = str(market.get("ticker") or market.get("market_ticker") or "")

            if outcome and prob is not None:
                matched.append((outcome, prob, ticker))

        if matched:
            matched.sort(key=lambda item: item[1], reverse=True)

            pieces = [
                f"{outcome}={prob:.1%} ({ticker})"
                for outcome, prob, ticker in matched[: min(12, len(matched))]
            ]

            lines.append(
                "Kalshi explicit retrieval: matched market-implied probabilities: "
                + "; ".join(pieces)
                + "."
            )

    return "\n".join(lines) if lines else None


# ---- evidence pass ---------------------------------------------------------

def _build_evidence_prompt(event: dict, outcomes: list[str]) -> str:
    outcomes_display = ", ".join(f'"{o}"' for o in outcomes)
    explicit_market_data = _kalshi_market_data_hint(event, outcomes)

    explicit_section = ""
    if explicit_market_data:
        explicit_section = (
            "Explicit market-data retrieval from Kalshi public API:\n"
            f"{explicit_market_data}\n\n"
            "Use this retrieved market data before relying on general web snippets.\n\n"
        )

    return (
        f"Event: {event.get('title', '(no title)')}\n"
        f"Category: {event.get('category') or 'Unknown'}\n"
        f"Description: {event.get('description') or ''}\n"
        f"Resolution rules: {event.get('rules') or ''}\n"
        f"Closes at: {event.get('close_time') or 'unknown'}\n"
        f"Possible outcomes: {outcomes_display}\n\n"
        f"{explicit_section}"
        f"Research plan: {_market_search_hint(event, outcomes)}\n\n"
        "Build a leakage-filtered evidence brief using only pre-event information."
    )


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
                return json.loads(text[start : i + 1])

    raise ValueError(f"Unbalanced braces in response. Raw: {text[:300]}")


def _format_evidence_brief(content: str) -> str:
    """
    Keep only leakage-safe fields from the research call.
    Never pass discarded/resolution information to the final forecaster.
    """
    try:
        data = _parse_json_loose(content)
    except ValueError:
        return ""

    safe_data = {
        "market_prior": data.get("market_prior", "not found"),
        "usable_pre_event_evidence": data.get("usable_pre_event_evidence", []),
        "uncertainties": data.get("uncertainties", []),
        "leakage_risk": data.get("leakage_risk", "unknown"),
        "suggested_baseline": data.get("suggested_baseline", ""),
    }

    return json.dumps(safe_data, ensure_ascii=False, separators=(",", ":"))[:3000]


def _call_once(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    extra_user_messages: list[dict] | None = None,
    max_tokens: int = MAX_TOKENS,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if extra_user_messages:
        messages.extend(extra_user_messages)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }

    if ":online" not in model:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content

    if not content:
        raise ValueError("Model returned empty content")

    return content


def _research_evidence(client, event: dict, outcomes: list[str]) -> str | None:
    if not USE_EVIDENCE_BRIEF:
        return None

    model = _online_model_name()

    try:
        content = _call_once(
            client,
            model,
            _EVIDENCE_SYSTEM_PROMPT,
            _build_evidence_prompt(event, outcomes),
            max_tokens=EVIDENCE_MAX_TOKENS,
        )
    except Exception as e:
        print(f"[agent] evidence brief failed, continuing without it: {e}", file=sys.stderr)
        return None

    brief = _format_evidence_brief(content)
    return brief or None


# ---- final forecast prompt -------------------------------------------------

def _build_prompt(
    event: dict,
    outcomes: list[str],
    evidence_brief: str | None = None,
) -> tuple[str, str]:
    outcomes_display = "\n".join(f'  - "{o}"' for o in outcomes)

    evidence_section = ""
    if evidence_brief:
        evidence_section = (
            "Leakage-filtered pre-event evidence brief from a separate research pass:\n"
            f"{evidence_brief}\n\n"
            "Use this filtered brief as your primary evidence base. "
            "Do not infer or reconstruct any discarded resolution information. "
            "Start from the market/odds prior in the brief when one exists.\n\n"
        )

    explicit_market_data = _kalshi_market_data_hint(event, outcomes)
    explicit_market_section = ""
    if explicit_market_data:
        explicit_market_section = (
            "Explicit pre-event market-data retrieval:\n"
            f"{explicit_market_data}\n\n"
            "Use this as a market-prior signal unless it appears stale, thin, or mismapped.\n\n"
        )

    user_prompt = (
        f"Event: {event.get('title', '(no title)')}\n"
        f"Category: {event.get('category') or 'Unknown'}\n"
        f"Description: {event.get('description') or ''}\n"
        f"Resolution rules: {event.get('rules') or ''}\n"
        f"Closes at: {event.get('close_time') or 'unknown'}\n\n"
        f"{evidence_section}"
        f"{explicit_market_section}"
        f"Market-awareness plan:\n"
        f"{_market_search_hint(event, outcomes)}\n\n"
        f"Possible outcomes:\n"
        f"{outcomes_display}\n\n"
        "Return ONLY the JSON object."
    )

    return _SYSTEM_PROMPT, user_prompt


# ---- response parsing ------------------------------------------------------

def _extract_probs(response_data: dict, outcomes: list[str]) -> list[float]:
    if not isinstance(response_data, dict):
        raise ValueError(f"Expected JSON object, got {type(response_data).__name__}")

    probs_dict = response_data.get("probabilities")
    if not isinstance(probs_dict, dict):
        raise ValueError("Missing or non-dict 'probabilities' field")

    result = []
    matched_returned = set()

    for canonical in outcomes:
        found_prob = None

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


# ---- main forecast ---------------------------------------------------------

def forecast(event: dict) -> list[float]:
    """
    Run the LLM forecaster.
    Returns raw normalized LLM probabilities.
    Calibration happens in predict.py via calibration.py.
    """
    outcomes = get_outcomes(event)

    if not outcomes:
        raise ValueError("Event has no outcomes.")

    client = _build_client()

    evidence_brief = _research_evidence(client, event, outcomes)
    system_prompt, user_prompt = _build_prompt(event, outcomes, evidence_brief)

    model = _forecast_model_name()

    try:
        content = _call_once(client, model, system_prompt, user_prompt)
        raw_probs = _parse_response(content, outcomes)
        return _normalize_probs(raw_probs)

    except ValueError as parse_err:
        print(f"[agent] first attempt failed: {parse_err}", file=sys.stderr)
        print("[agent] retrying with stricter JSON reminder...", file=sys.stderr)

    retry_message = {
        "role": "user",
        "content": (
            "Your previous response was not valid JSON or did not match the schema. "
            "Return ONLY one JSON object with exact outcome labels and numeric probabilities."
        ),
    }

    content = _call_once(
        client,
        model,
        system_prompt,
        user_prompt,
        extra_user_messages=[retry_message],
    )

    raw_probs = _parse_response(content, outcomes)
    return _normalize_probs(raw_probs)
