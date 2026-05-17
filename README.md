# Cost-Aware Calibrated Forecasting Agent

A probabilistic forecasting agent for [Prophet Hacks 2026](https://www.prophethacks.com/) (Forecasting Track). Predicts calibrated probability distributions over real-world event outcomes, optimized for [Brier score](https://en.wikipedia.org/wiki/Brier_score).

**Live endpoint:** `https://web-production-df9b3.up.railway.app/predict`

**Brier score:** 0.6290 (honest baseline on `sample-resolved`, vs uniform 0.6912)

## What it does

The agent receives an event dict (title, description, outcomes, etc.) via HTTP POST and returns a probability distribution over the possible outcomes. Designed to be:

- **Calibrated** — avoids overconfidence (the biggest Brier-score penalty)
- **Robust** — never crashes, never returns non-200, never returns malformed output
- **Cost-aware** — single LLM call per event; ~$0.15 for a 200-event evaluation

## Architecture

```
event dict (POST /predict)
    ↓
predict.py: get_outcomes(event)
    ↓
agent.forecast()              ← LLM call (Claude Sonnet 4.5 via OpenRouter, with web search)
    ↓                         ← retry once on JSON parse failure
calibration.calibrate()       ← confidence-aware shrinkage + multi-outcome rule
    ↓
utils.normalize_probabilities()  ← exact sum-to-1, exact label match
    ↓
utils.validate_output()       ← contract check
    ↓
return {probabilities: [...]}
```

**Every step has a fallback.** If the LLM call fails, JSON parses wrong, labels mismatch, or anything else breaks, `predict()` returns a uniform distribution over the outcomes rather than raising. The FastAPI server wraps this in another layer: any error inside the handler returns 200 with uniform fallback instead of 500.

## Key design decisions

### Anti-leakage search reframing

OpenRouter's `:online` model suffix enables web search. We discovered early that on resolved test datasets, search would return *the actual final outcomes*, and the model would just report those at 99% confidence. This produced a fake-good Brier of 0.06 — which would not generalize to the real eval (where events haven't happened yet).

The fix is a system prompt that explicitly instructs the model to use search for *pre-event context only* (odds, form, news, expert predictions) and to **ignore** any search results that reveal final outcomes. This drops our resolved-set Brier from 0.06 back to an honest 0.63, but reflects the agent's true forecasting behavior on unresolved events.

### Confidence-aware calibration

Brier score punishes confident-wrong predictions quadratically. We shrink probabilities toward the uniform prior, with shrinkage strength scaled by how *confident* the prediction is:

```
confidence = max(probs) - 1/N
alpha_eff  = base + slope × confidence       (default base=0.05, slope=0.30)
p_cal      = (1 - alpha_eff) × p_raw + alpha_eff × (1/N)
```

A 0.50/0.50 prediction is barely touched; a 0.95/0.05 prediction is pulled meaningfully toward 0.5. This protects against the worst Brier outcomes.

### Multi-outcome rule

For events with ≥10 outcomes (like Survivor episode eliminations or NHL Calder Trophy candidates), if the model's top probability is below 20%, we return exactly uniform. This avoids the variance of "noisy near-uniform" distributions that score slightly worse than clean uniform on Brier.

### Unicode label normalization

The grader matches predicted labels against canonical outcome strings exactly. Smart quotes, em dashes, non-breaking spaces, and decomposed Unicode characters were all causing silent label mismatches → fallback firing → lost Brier credit. Fixed via aggressive NFKC normalization and a punctuation substitution map.

### One-shot retry on parse failure

If the model returns prose instead of JSON (rare but it happens), we retry once with a sterner system reminder. Catches the SCOTUS-style "let me think about this..." failure mode.

## Repo structure

```
forecast-debate-agent/
├── predict.py              Entry point — the function the grader calls
├── agent.py                LLM call, prompt, JSON parsing, label matching
├── calibration.py          Probability shrinkage / calibration modes
├── utils.py                Output validation, safe fallback, normalization
├── server.py               FastAPI HTTP wrapper for deployment
│
├── requirements.txt        Python dependencies
├── Procfile                Railway/Render start command
├── .env.example            Template for OPENROUTER_API_KEY
│
├── tests/
│   ├── test_predict.py     13 output-contract tests
│   ├── test_step1_units.py 25 parser/calibration unit tests
│   └── test_step3_units.py 22 smart-quote/loose-JSON tests
├── test_server_smoke.py    3 in-process server integration tests
│
└── eval/
    ├── fetch_dataset.py    Downloads datasets via prophet CLI
    ├── run_eval.py         Runs predict() over a dataset, saves JSONL
    ├── score.py            Computes Brier scores with per-category breakdown
    ├── inspect.py          Pretty-prints predictions for human review
    └── tune_calibration.py Sweeps calibration parameters (no API calls)
```

## Quick start (for organizers / reviewers)

### Requirements

- Python 3.11+
- An OpenRouter API key ([openrouter.ai](https://openrouter.ai))

### Setup

```bash
git clone https://github.com/AP2001211/forecast-debate-agent.git
cd forecast-debate-agent
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Run tests (no API calls, no money spent)

```bash
python tests/test_predict.py        # 13 contract tests
python tests/test_step1_units.py    # 25 unit tests
python tests/test_step3_units.py    # 22 unit tests
```

All 60 should pass.

### Test the agent on a single event (one real API call, ~$0.005)

```bash
python predict.py
```

### Run the HTTP server locally

```bash
python server.py
```

Then in another terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"event_ticker":"test","market_ticker":"test","title":"Pittsburgh vs Atlanta","description":"test","category":"Sports","rules":"test","close_time":"2026-12-31T23:59:59+00:00","outcomes":["Pittsburgh","Atlanta"]}'
```

### Reproduce our Brier evaluation

```bash
# Fetch the resolved dataset (free, no API call)
python -m eval.fetch_dataset

# Run our agent against all 26 events (~$0.15 in OpenRouter credits)
python -m eval.run_eval --fresh

# Score against ground truth
python -m eval.score
```

Expected output: Brier score around 0.63, beating the uniform baseline of 0.69.

## Endpoint contract

The deployed agent accepts POST requests to `/predict` with an event dict:

```json
{
  "event_ticker": "task-001",
  "market_ticker": "task-001",
  "title": "Who will win: Pittsburgh or Atlanta?",
  "description": "...",
  "category": "Sports",
  "rules": "...",
  "close_time": "2026-03-21T23:59:59+00:00",
  "outcomes": ["Pittsburgh", "Atlanta"]
}
```

Returns a probability distribution:

```json
{
  "probabilities": [
    {"market": "Pittsburgh", "probability": 0.55},
    {"market": "Atlanta", "probability": 0.45}
  ]
}
```

The server **always** returns 200 with valid output. Any internal error falls back to a uniform distribution over the event's outcomes rather than returning a non-200.

## Configuration

All behavior is controllable via environment variables — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key |
| `PROPHET_USE_WEB_SEARCH` | `1` | Set to `0` to disable `:online` search |
| `PROPHET_CALIBRATION_MODE` | `multi_outcome_aware` | Also: `confidence_aware`, `fixed`, `none` |
| `PROPHET_CAL_BASE` | `0.05` | Base shrinkage strength |
| `PROPHET_CAL_SLOPE` | `0.30` | Confidence-dependent shrinkage slope |
| `PROPHET_CAL_MULTI_N` | `10` | Outcome-count threshold for uniform-forcing rule |
| `PROPHET_CAL_MULTI_THRESH` | `0.20` | Top-prob threshold for uniform-forcing rule |
| `PROPHET_MODEL` | `unknown` | Informational, shown in `/health` |

## Deployment

The agent is deployed on [Railway](https://railway.app) using the included `Procfile`:

```
web: uvicorn server:app --host 0.0.0.0 --port $PORT
```

To deploy elsewhere (Render, Fly.io, etc.): same start command, set `OPENROUTER_API_KEY` in the platform's environment variables.

## Brier score results

| Configuration | Brier | Notes |
|---|---|---|
| Uniform prior baseline | 0.6912 | Always guess 1/N |
| Step 1 (no search, fixed α=0.10) | 0.6449 | Single LLM call, basic calibration |
| Step 3 (no search, confidence-aware) | 0.6469 | Bugfixes + better calibration |
| Step 4 (search + anti-leakage prompt) | **0.6290** | Current production, honest baseline |
| Step 3 (search, no anti-leakage) | 0.0589 | Inflated — search retrieved final scores |
| Perfect | 0.0 | |

Improvement over uniform: ~9% on `sample-resolved` (26 events across Sports, Politics, Elections, Entertainment).

Per-category breakdown of our 0.6290:

| Category | n | Mean Brier | Notes |
|---|---|---|---|
| Sports | 16 | 0.5424 | Best signal — model knows famous teams |
| Elections | 3 | 0.6028 | Mostly uniform on local primaries |
| Politics | 3 | 0.8497 | Hard — Senate vote counts have no pre-event signal |
| Entertainment | 4 | 0.8291 | Reality TV with 14–20 contestants is near-unpredictable |

## Tech stack

- **Language:** Python 3.11
- **LLM:** [Claude Sonnet 4.5](https://www.anthropic.com/) via [OpenRouter](https://openrouter.ai/)
- **Web search:** OpenRouter `:online` model suffix
- **HTTP server:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Deployment:** [Railway](https://railway.app)
- **Dataset access:** [`ai-prophet`](https://pypi.org/project/ai-prophet/) CLI

## License

MIT — see `LICENSE`.

## Acknowledgments

Built for [Prophet Hacks 2026](https://www.prophethacks.com/) at UChicago. Thanks to the organizers for the well-designed evaluation harness and clear submission spec.
