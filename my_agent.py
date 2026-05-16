import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "openai/gpt-4o-mini"


def calibrate(p: float, confidence: float) -> float:
    # Pull probabilities toward 50% when confidence is low
    shrink = 0.35 + 0.65 * confidence
    p = 0.5 + (p - 0.5) * shrink

    # Avoid extreme overconfidence
    p = max(0.03, min(0.97, p))
    return round(p, 4)


def predict(event: dict) -> dict:
    outcomes = event["outcomes"]

    # Start simple: binary events only
    if len(outcomes) != 2:
        n = len(outcomes)
        return {
            "probabilities": [
                {"market": outcome, "probability": round(1 / n, 4)}
                for outcome in outcomes
            ]
        }

    prompt = f"""
You are a forecasting agent.

Event title: {event.get("title")}
Description: {event.get("description")}
Category: {event.get("category")}
Rules: {event.get("rules")}
Close time: {event.get("close_time")}
Outcomes: {outcomes}

Estimate the probability of outcome "{outcomes[0]}".

Think using:
- optimistic case for "{outcomes[0]}"
- pessimistic case against "{outcomes[0]}"
- judge decision
- confidence

Return ONLY valid JSON:
{{
  "raw_probability": 0.0,
  "confidence": 0.0
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content.strip()
        data = json.loads(text)

        raw_p = float(data["raw_probability"])
        confidence = float(data["confidence"])

        p0 = calibrate(raw_p, confidence)
        p1 = round(1 - p0, 4)

        return {
            "probabilities": [
                {"market": outcomes[0], "probability": p0},
                {"market": outcomes[1], "probability": p1},
            ]
        }

    except Exception:
        # Safe fallback: never crash
        return {
            "probabilities": [
                {"market": outcomes[0], "probability": 0.5},
                {"market": outcomes[1], "probability": 0.5},
            ]
        }