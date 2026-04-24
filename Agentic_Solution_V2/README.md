# Acquirer Identification Engine

An agentic AI tool that helps investment bankers identify the most likely acquirers for a target company and generates MD-ready rationales in under a minute.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-2.6+-E92063.svg)
![Anthropic](https://img.shields.io/badge/Claude-Sonnet%204.6-D97757.svg)
![Tests](https://img.shields.io/badge/tests-passing-green.svg)

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [How It Works](#how-it-works)
4. [Architecture](#architecture)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Technology Stack](#technology-stack)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Design Decisions](#design-decisions)
13. [Limitations](#limitations)
14. [API Reference](#api-reference)

---

## Overview

The Acquirer Identification Engine ingests a dataset of historical M&A transactions, accepts a target company profile (sector, enterprise value, geography), and returns the ten most likely acquirers with a complete one-page rationale for each. Every claim in every rationale is grounded in the underlying data — no hallucinations, no generic marketing language, no unverifiable assertions.

The output is designed to meet a specific bar: a Vice President should be able to hand the rationales directly to a Managing Director without editing.

**Key capabilities:**
- Deterministic scoring engine with transparent weights
- Agentic LLM layer with tool use and dynamic routing
- Strict output validation via Pydantic with automatic repair loop
- Production-grade observability (structured logs, cost tracking, trace spans)
- Smart retry policy with typed backoff strategies
- Web UI, REST API, CLI, and Docker deployment paths

---

## The Problem

A junior investment banker typically spends two to four hours researching potential acquirers for a new engagement: pulling comparable deals from Pitchbook, scanning historical transactions in CapIQ, synthesizing rationale in PowerPoint. For a mid-market M&A firm running many concurrent mandates, this is a significant time tax.

The goal of this tool is to compress that workflow to under a minute while producing output of higher quality than the manual equivalent. The rationales are structured, internally consistent, grounded in cited evidence, and written in the direct, numbers-heavy voice of an experienced analyst.

---

## How It Works

The system uses a two-stage pipeline. Stage one is deterministic and handles ranking. Stage two is agentic and handles rationale generation.

### Stage 1 — Deterministic Scoring

The CSV is parsed with Pandas. Transactions are grouped by acquirer, and any acquirer with fewer than two deals is filtered out (a single data point is insufficient for pattern recognition). For each qualifying acquirer, the system computes a structured evidence packet containing:

- Sector and sub-sector distribution
- Deal size statistics (minimum, median, maximum, count in the target band)
- Median EV/EBITDA and EV/Revenue multiples on closed deals
- Deal type mix and geographic distribution
- Top strategic rationale tags
- Close rate, most recent deal year, acquirer type

Each acquirer is then scored against the target profile on five weighted factors:

| Factor | Weight | Rewards |
|--------|--------|---------|
| Sector match | 40 | Deals in exact target sector (1.0), adjacent sectors (0.6), other healthcare (0.3) |
| Size-band fit | 25 | Deals within 0.5x–2x of target enterprise value |
| Recency | 20 | Most recent activity in the last three years |
| Close rate | 10 | ≥70% of announced deals reach close |
| Volume | 10 | Log-scaled deal count (prevents single-deal dominance) |

A separate **conviction gate** runs on top of scoring. To earn a "High" conviction rating, an acquirer must have ≥2 deals in the exact target sector (or ≥4 adjacent + ≥1 exact), ≥1 deal in the target size band, and activity within the last three years. Pure adjacency (adjacent sectors only, no exact-sector deals) is capped at "Medium" regardless of other criteria. The gate is deterministic and binding — the LLM cannot override it.

The top ten acquirers by score advance to Stage 2.

### Stage 2 — Agentic Rationale Generation

Each of the top ten acquirers receives an independent call to the language model. The model operates as an agent with access to five tools:

| Tool | Purpose |
|------|---------|
| `shortlist_acquirers` | Run Stage 1 scoring, return top N ranked candidates |
| `get_evidence_packet` | Retrieve the full structured profile for one acquirer |
| `get_relevant_transactions` | Select the 2–3 most relevant prior deals |
| `compute_valuation_envelope` | Project EV range from multiples + target EBITDA |
| `check_conviction_gate` | Return the deterministic conviction level with gate detail |

The per-acquirer agent loop:

1. **Call Claude** with the system prompt, pre-loaded evidence, and available tool schemas
2. **Inspect `stop_reason`** — if `tool_use`, dispatch the tool and loop; if `end_turn`, parse the output
3. **Validate the final JSON** against the `AcquirerRationale` Pydantic schema, which enforces:
   - Transaction ID citation in the precedent section
   - Absence of banned marketing phrases
   - Valuation context containing both a multiple (Nx) and an EV figure ($)
   - Character budgets per field
4. **On validation failure**, invoke the repair loop: send the specific `ValidationError` back to the model with instructions to fix each issue
5. **Force the deterministic conviction value** into the output before returning

All ten per-acquirer agents run concurrently via `asyncio.gather` with a `Semaphore` that caps concurrency at a configurable value (tuned to the rate-limit tier of the Anthropic API key in use).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Client                                                          │
│  ├─ Web UI (frontend/index.html)                                 │
│  ├─ CLI (run_cli.py)                                             │
│  └─ HTTP clients (curl, Postman, other services)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI (api.py)                                                │
│  ├─ POST /api/v1/identify-acquirers                              │
│  ├─ GET  /health                                                 │
│  ├─ Lifespan: AsyncAnthropic client created ONCE at startup      │
│  └─ Serves frontend/index.html at /                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator (agent.identify_acquirers)                         │
│                                                                  │
│  Stage 1: Deterministic                                          │
│  ├─ evidence.build_evidence_packets(df, target)                  │
│  ├─ scoring.rank_acquirers(packets, target) → top 10             │
│  └─ scoring.compute_conviction(packet, target) → H/M/L           │
│                                                                  │
│  Stage 2: Agentic (asyncio.gather + Semaphore)                   │
│  For each of top 10, concurrently:                               │
│    ├─ Pre-load evidence + transactions + gate detail             │
│    ├─ Agent loop (max 4 iterations):                             │
│    │   ├─ call_claude(tools=TOOL_SCHEMAS)                        │
│    │   ├─ if stop_reason == "tool_use" → dispatch → loop         │
│    │   └─ if stop_reason == "end_turn" → break                   │
│    ├─ Extract JSON from response (brace-matched)                 │
│    ├─ Validate via AcquirerRationale Pydantic schema             │
│    └─ On validation failure → repair loop (one shot)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Cross-cutting concerns                                          │
│  ├─ observability.py: structlog + RunMetrics + trace spans       │
│  ├─ retry.py: tenacity with typed backoff (30s for 429s)         │
│  ├─ schemas.py: Pydantic models with @field_validator rules      │
│  └─ config.py: Pydantic Settings, secrets as SecretStr           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- An Anthropic API key ([get one at console.anthropic.com](https://console.anthropic.com))

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Agentic_Solution_V2

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate           # macOS/Linux
venv\Scripts\Activate.ps1          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and paste your Anthropic API key
```

### Run the Web UI

```bash
python run_api.py
```

Open `http://localhost:8000` in a browser. Select a sector, adjust the enterprise value, pick a geography, and click **Identify Acquirers**. The ten ranked rationales appear in roughly 60 to 80 seconds on tier-1 Anthropic rate limits.

---

## Configuration

All runtime configuration is loaded from a `.env` file. A template is provided in `.env.example`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | _required_ | Your Anthropic API key. Never committed to git. |
| `MODEL` | `claude-sonnet-4-6` | Model identifier |
| `MAX_TOKENS` | `2000` | Maximum output tokens per LLM call |
| `TEMPERATURE` | `0.3` | Sampling temperature (low enough for consistency, high enough for natural prose) |
| `MAX_CONCURRENT_REQUESTS` | `2` | Maximum parallel LLM calls. Tuned for tier-1 rate limits (8K output tokens/min). Raise on higher tiers. |
| `LOG_LEVEL` | `INFO` | Standard Python log levels |
| `LOG_JSON` | `false` | Set to `true` for JSON-formatted logs (production ingestion) |
| `CSV_PATH` | `data/ma_transactions_500.csv` | Path to the input dataset |

---

## Usage

### Three ways to run the pipeline

#### 1. Web UI (recommended for interactive use)

```bash
python run_api.py
```

Navigate to `http://localhost:8000` for the form-based UI, or `http://localhost:8000/docs` for the auto-generated Swagger documentation.

#### 2. Command-line interface (useful for CI or scripting)

```bash
# Default target: Healthcare Services, $200M, Regional
python run_cli.py

# Custom target profile
python run_cli.py --sector "Health IT" --size 300 --geography "Northeast"
```

The CLI produces rich-formatted terminal output with progress indicators, acquirer cards, and a final metrics summary showing total cost, duration, and token counts.

#### 3. REST API (for programmatic integration)

```bash
curl -X POST http://localhost:8000/api/v1/identify-acquirers \
  -H "Content-Type: application/json" \
  -d '{
    "sector": "Healthcare Services",
    "size_mm": 200,
    "geography": "Regional"
  }'
```

The response is the complete `RationaleSet`: ten validated rationales, total cost in USD, total duration in seconds, and token counts.

---

## Project Structure

```
Agentic_Solution_V2/
├── src/acquirer_engine/          # Application package (9 modules)
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings — loads .env
│   ├── schemas.py                # All Pydantic models + field validators
│   ├── scoring.py                # Pure scoring functions + conviction gating
│   ├── evidence.py               # CSV ingestion + evidence packet builder
│   ├── tools.py                  # 5 LLM-callable tools with JSON Schema
│   ├── observability.py          # structlog + RunMetrics + trace spans
│   ├── retry.py                  # Tenacity decorators for LLM retries
│   ├── agent.py                  # Orchestrator — tool loop, validation, repair
│   └── api.py                    # FastAPI application + lifespan
│
├── tests/                        # Pytest suite (no API key required)
│   ├── conftest.py               # Shared fixtures
│   ├── test_scoring.py           # Scoring + conviction gating
│   ├── test_schemas.py           # Validator rules + cost math
│   ├── test_tools.py             # Tool dispatch + error paths
│   └── test_observability.py     # Metrics accumulation
│
├── frontend/
│   └── index.html                # Self-contained web UI
│
├── data/
│   └── ma_transactions_500.csv   # Sample dataset (500 historical deals)
│
├── run_api.py                    # FastAPI server entry point
├── run_cli.py                    # CLI entry point (rich output)
├── Dockerfile                    # Production container image
├── docker-compose.yml            # Local stack (API + Redis)
├── requirements.txt              # Pinned Python dependencies
├── pytest.ini                    # Test runner configuration
├── .env.example                  # Environment variable template
├── .gitignore                    # Excludes .env, venv/, __pycache__
└── README.md                     # This file
```

---

## Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| LLM client | `anthropic` | Async Anthropic SDK |
| Data validation | `pydantic` | Type-safe models with field validators |
| Configuration | `pydantic-settings` | Environment-driven config with secret handling |
| Web framework | `fastapi` + `uvicorn` | REST API with auto-generated OpenAPI docs |
| Data processing | `pandas` | CSV parsing + groupby aggregations |
| Structured logging | `structlog` | JSON-ready log output |
| Retry logic | `tenacity` | Declarative retry policies |
| HTTP client | `httpx` | Underlies Anthropic SDK; used in tests |
| CLI output | `rich` | Formatted terminal rendering |
| Testing | `pytest` + `pytest-asyncio` | Test runner with async support |
| Coverage | `pytest-cov` | Coverage reporting |

All versions are pinned in `requirements.txt` with `>=` minimums. Upgrade as appropriate.

---

## Testing

The test suite runs in under two seconds and does not require an API key. It covers every layer that doesn't require an LLM call:

```bash
pytest                              # All tests
pytest tests/test_scoring.py -v     # Just scoring tests
pytest --cov=src/acquirer_engine    # With coverage report
```

### What's covered

- **Scoring math**: component scores (sector, size, recency, close rate, volume), composite weighted score, edge cases (empty data, extreme values)
- **Conviction gating**: all three gate conditions, pure-adjacency cap, edge cases
- **Pydantic validators**: transaction ID citation, banned phrase rejection, valuation format enforcement, type validation
- **Tool dispatch**: unknown tools, invalid input, successful execution, error path
- **Cost tracking**: real token counts producing real USD, zero-token edge case, metrics accumulation

### What's not covered (by design)

The agent loop itself requires a mocked `AsyncAnthropic` client for deterministic testing. The architecture supports this (the client is dependency-injected), but the current suite prioritizes the layers where unit tests have the highest leverage.

---

## Deployment

The system is designed for staged deployment.

### V1 — Pilot Deployment (Azure, 5–20 users)

Target architecture for validating the product with a small group of users before investing in enterprise infrastructure:

- **Compute**: Azure Container Apps (the included Dockerfile runs as-is)
- **Authentication**: Microsoft Entra ID (native SSO)
- **Secrets**: Azure Key Vault with managed identity
- **Persistence**: Azure Cosmos DB for run history and feedback
- **Observability**: Application Insights via OpenTelemetry
- **Frontend hosting**: Azure Static Web Apps

Estimated operating cost: ~$40/month at pilot volume (compute plus LLM tokens).

### V2 — Enterprise Deployment (AWS, 200+ users)

Target architecture for full production once pilot adoption justifies the infrastructure investment:

- **Compute**: AWS ECS Fargate behind ALB with WAF
- **Authentication**: Amazon Cognito federated to Entra ID
- **Caching**: ElastiCache Redis keyed on `hash(acquirer + target + csv_hash)` — biggest cost lever at scale
- **Persistence**: Aurora PostgreSQL with CDC to S3 for audit logging
- **LLM runtime**: AWS Bedrock (keeps data inside the VPC for compliance)
- **Observability**: LangSmith + CloudWatch + X-Ray
- **Orchestration upgrade**: LangGraph with a critic agent validating the rationale writer's output as a separate call
- **Feedback loop**: SQS → Lambda → SageMaker for preference-pair training on accumulated user feedback

Estimated operating cost: $800–$1,500/month at 200 users with ~500 runs per day, after cache warm-up.

### Docker

A `Dockerfile` and `docker-compose.yml` are provided:

```bash
docker compose up
```

This builds the API container and starts it alongside a Redis instance. The Redis container is a placeholder for the caching layer; wiring it in is a straightforward addition.

---

## Design Decisions

### Why raw Anthropic SDK instead of LangChain or LangGraph?

At this scope, the raw SDK produces clearer code. The agent loop in `agent.py` is explicit and debuggable — dynamic routing happens on the `resp.stop_reason` check, not inside framework abstractions. The tool schemas, Pydantic validators, and scoring functions are framework-neutral by design, so migrating to LangGraph for multi-agent orchestration (when needed for V2) is an incremental refactor rather than a rewrite.

### Why Pydantic everywhere?

Pydantic enforces the type contract at runtime where bugs are cheap to find, rather than in production where they're expensive. Every value that crosses a module boundary has a defined type. Field validators turn documented rules ("every precedent must cite a transaction ID") into enforced constraints that block malformed output from reaching the UI.

### Why `temperature=0.3`?

Low enough that conviction levels and transaction citations are stable across repeated runs of the same input. High enough that the prose reads naturally instead of like templated boilerplate. The deterministic layers (scoring, conviction gating) don't depend on temperature at all — only the rationale text varies.

### Why concurrency = 2?

Tuned to the Anthropic tier-1 rate limit of 8,000 output tokens per minute. With `MAX_TOKENS=2000` and two concurrent requests, each wave produces up to 4,000 output tokens, leaving comfortable headroom. On higher rate-limit tiers, this value can be raised by changing `.env` alone — no code changes.

### Why separate retry strategies per exception type?

A one-second retry after a 429 rate-limit error is guaranteed to fail because the rate-limit window is 60 seconds sliding. A one-second retry after a transient connection error is appropriate. The `_smart_wait` function in `retry.py` routes each exception type to the correct backoff pattern.

### Why pre-load evidence in the user prompt?

An earlier version had the agent call `get_evidence_packet`, `get_relevant_transactions`, and `check_conviction_gate` on every iteration — three tool round-trips per acquirer, adding 30+ seconds of latency. Pre-loading that data in the user prompt means the agent typically completes in a single iteration. Tools remain available for optional computations (e.g., `compute_valuation_envelope` on a specific target EBITDA estimate).

---

## Limitations

These are known and documented.

- **Dataset-bounded knowledge**. The system only knows what's in the CSV. There's no external enrichment (SEC filings, news, live deal pipeline). An acquirer not present in the data cannot be surfaced.
- **Thin-evidence acquirers**. Acquirers with the minimum two deals are scored and produce rationales, but the rationale prose is honest about the sample size ("thin evidence") and the conviction gate reflects it.
- **Geography in scoring**. Geography is passed to the LLM prompt but does not factor into the quantitative scoring. Adding it as a sixth weighted factor is a straightforward extension.
- **No caching**. Every request to `/api/v1/identify-acquirers` makes fresh LLM calls. Redis is set up in the Docker compose but not yet wired in. At pilot volume, cache hit rate would be near zero; the seam is in place for when it becomes worthwhile.
- **Non-determinism in prose**. At temperature 0.3, the wording varies slightly between runs of the same input. Conviction levels, transaction citations, and computed metrics are stable.

---

## API Reference

### `POST /api/v1/identify-acquirers`

Run the full pipeline for a target profile.

**Request body:**
```json
{
  "sector": "Healthcare Services",
  "size_mm": 200,
  "geography": "Regional",
  "profile_notes": "Mid-market, private, strong EBITDA margins"
}
```

**Response body:**
```json
{
  "target": { "sector": "...", "size_mm": 200, "geography": "..." },
  "rationales": [
    {
      "acquirer_name": "...",
      "rank": 1,
      "acquirer_overview": "...",
      "strategic_fit_thesis": "...",
      "precedent_activity": "- Acme Corp (2023, $180M, 13.2x EV/EBITDA) (MA-2023-0145)\n- ...",
      "valuation_context": "At 13.2x median EV/EBITDA on 5 closed deals, ...",
      "risk_flags": "- ...\n- ...",
      "conviction": { "level": "High", "rationale": "..." }
    }
    // ... 9 more
  ],
  "total_cost_usd": 0.4123,
  "total_duration_seconds": 68.5,
  "total_input_tokens": 65421,
  "total_output_tokens": 19834,
  "cache_hits": 0
}
```

### `GET /health`

Simple liveness probe.

**Response:**
```json
{ "status": "ok", "model": "claude-sonnet-4-6" }
```

### `GET /docs`

Auto-generated Swagger UI for interactive API exploration.

---

## Contributing

This repository is a portfolio/demonstration project. If you'd like to extend it:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for your change
4. Ensure `pytest` passes
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for the full text.

In short: you can use, copy, modify, and distribute this code freely, including for commercial purposes, as long as the copyright notice and license text are included. The software is provided "as is" with no warranty.
