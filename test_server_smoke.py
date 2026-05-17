"""One-shot integration test for server.py — no separate terminal needed."""
import json
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

# Test 1: health
r = client.get("/health")
print(f"GET /health -> {r.status_code} {r.json()}")
assert r.status_code == 200
assert r.json()["status"] == "ok"

# Test 2: malformed body (must return 200, not 500)
r = client.post("/predict", content="not valid json", headers={"Content-Type": "application/json"})
print(f"POST /predict [malformed] -> {r.status_code} {r.json()}")
assert r.status_code == 200, "Server returned non-200 on malformed body!"
assert "probabilities" in r.json()

# Test 3: real predict call (this makes a real LLM API call ~$0.005)
event = {
    "event_ticker": "smoke-001",
    "market_ticker": "smoke-001",
    "title": "Who will win: Pittsburgh or Atlanta?",
    "description": "Predict the winner.",
    "category": "Sports",
    "rules": "Resolves to the official winner.",
    "close_time": "2026-12-31T23:59:59+00:00",
    "outcomes": ["Pittsburgh", "Atlanta"]
}
r = client.post("/predict", json=event)
print(f"POST /predict [real] -> {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200
probs = r.json()["probabilities"]
assert len(probs) == 2
markets = {p["market"] for p in probs}
assert markets == {"Pittsburgh", "Atlanta"}
total = sum(p["probability"] for p in probs)
assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1.0"

print("\nAll 3 server smoke tests passed.")
