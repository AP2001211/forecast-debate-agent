"""
FastAPI HTTP wrapper for the forecasting agent.

The grader POSTs an event dict to /predict and expects back our
probabilities response. /health is a liveness check.

CRITICAL: this handler NEVER returns a non-200 to the grader, even on
internal errors. Every error path falls back to a valid uniform
distribution over the event's outcomes. A 500 would be counted as a
missed prediction (reducing n_matched on the leaderboard); a 200 with
uniform is at least Brier ~0.5 on a binary, which is much better than
zero credit.

Usage (local):
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy to Render/Railway:
    - Start command: uvicorn server:app --host 0.0.0.0 --port $PORT
    - Set OPENROUTER_API_KEY in environment
    - Submit your URL to https://www.prophethacks.com/submit-endpoint
"""

from __future__ import annotations

import os
import sys
import time
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from predict import predict


app = FastAPI(title="Prophet Hacks Forecasting Agent", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Liveness check. The PROPHET_MODEL env var is purely informational."""
    return {
        "status": "ok",
        "model": os.environ.get("PROPHET_MODEL", "unknown"),
    }


def _uniform_fallback(outcomes: list) -> dict:
    """Build a valid uniform distribution as a last-resort response."""
    if not outcomes:
        return {"probabilities": []}
    n = len(outcomes)
    return {
        "probabilities": [
            {"market": str(o), "probability": 1.0 / n} for o in outcomes
        ]
    }


@app.post("/predict")
async def predict_endpoint(request: Request) -> JSONResponse:
    """
    The submission endpoint. Always returns 200 with a valid probabilities
    response. Internal errors fall back to a uniform distribution over the
    event's outcomes rather than propagating as 500s.
    """
    t0 = time.time()

    # 1. Parse request body. Malformed JSON shouldn't crash us.
    try:
        event = await request.json()
    except Exception as e:  # noqa: BLE001
        print(f"[server] bad request body: {e}", file=sys.stderr)
        return JSONResponse(content={"probabilities": []}, status_code=200)

    if not isinstance(event, dict):
        print(f"[server] event is not a dict: {type(event).__name__}", file=sys.stderr)
        return JSONResponse(content={"probabilities": []}, status_code=200)

    outcomes = event.get("outcomes") or []

    # 2. Call predict(). It should never raise (it has its own try/except),
    # but we belt-and-brace.
    try:
        result = predict(event)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        result = _uniform_fallback(outcomes)

    # 3. Validate shape. If predict() returned anything weird, fall back.
    if not isinstance(result, dict) or "probabilities" not in result:
        print("[server] predict() returned unexpected format, falling back", file=sys.stderr)
        result = _uniform_fallback(outcomes)

    elapsed = time.time() - t0
    ticker = event.get("market_ticker") or event.get("event_ticker") or "?"
    n_probs = len(result.get("probabilities", []))
    print(f"[server] {ticker} -> {n_probs} probs in {elapsed:.1f}s")

    return JSONResponse(content=result, status_code=200)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)