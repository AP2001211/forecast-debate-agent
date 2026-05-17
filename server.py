"""
FastAPI HTTP wrapper for the forecasting agent.

The grader POSTs an event dict to /predict and expects back our
probabilities response. /health is a liveness check.

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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from predict import predict

app = FastAPI(title="Prophet Hacks Forecasting Agent", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": os.environ.get("PROPHET_MODEL", "unknown")}


@app.post("/predict")
def predict_endpoint(event: dict) -> dict:
    try:
        result = predict(event)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not isinstance(result, dict) or "probabilities" not in result:
        raise HTTPException(status_code=500, detail="predict() returned unexpected format")

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
