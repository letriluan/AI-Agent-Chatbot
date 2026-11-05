# AI Agent Chatbot

A full-stack SQL analytics application that turns natural-language questions into **inspectable SQL, answers and charts**.

**Python · FastAPI · LangGraph · LangChain · Groq · SQLite · Matplotlib · JavaScript · Docker · GitHub Actions · Render**

[![CI](https://github.com/letriluan/AI-Agent-Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/letriluan/AI-Agent-Chatbot/actions/workflows/ci.yml) [![Manual availability diagnostic](https://github.com/letriluan/AI-Agent-Chatbot/actions/workflows/availability.yml/badge.svg)](https://github.com/letriluan/AI-Agent-Chatbot/actions/workflows/availability.yml)

## [Try the Live Demo →](https://ai-agent-chatbot-demo.onrender.com/)

*First load may take ~30 s (free tier wakes from sleep).*

## Results at a glance

| Area | Measured result |
| --- | --- |
| Tests and deployment | **314 tests passed**; Render deploys after CI checks pass |
| Five-run reliability | **80/80 SQL matches**, 16/16 cases correct in every run; **20/20 chart checks** |
| Second schema | **24/24 SQL matches; 4/4 chart checks** on synthetic transit data |
| Answer numbers backed by SQL results | **71/75 holdout; 16/17 transit** scored answers supported |
| Availability | Free monitoring every **30 min**; confirmed cold start **32.85 s** |

Results measured on 5 November 2026. Synthetic evaluation results are not a general accuracy guarantee. [Protocols, costs and limitations](docs/PROJECT1_EVALUATION.md).

[Project](#why-i-built-it) · [Design](#architecture-and-design) · [CI/CD](#cicd-and-deployment) · [Results](#evaluation-and-results) · [Run Locally](#quick-start) · [Engineering Notes](docs/ENGINEERING.md)

## Why I Built It

Relational databases contain useful information, but accessing it often requires SQL knowledge. I built this chatbot to let users ask questions in natural language and receive answers or charts directly from the database.

The goal is to make everyday data exploration easier for users who are unfamiliar with SQL, while allowing technical users to inspect the generated queries and verify the results.

The project connects a browser interface, an agent workflow, bounded database execution and chart rendering in one deployed application. The demo uses the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): **9 related tables and 99,441 orders**.

## Demo and Example Output

The interface places the project context and ten sample questions beside a working test panel. The desktop layout keeps both panels within one screen; smaller screens use project/demo tabs. Long answers scroll inside the test panel. Users can inspect and copy SQL, copy answers and download charts. Each question is independent; include the filters and context you need in that question.

![Project introduction, sample questions and the chatbot test panel](docs/images/project-demo.png)

**Example question:** “Plot monthly total order item prices as a line chart. Join order items to orders, group by purchase month in YYYY-MM format, include all statuses and exclude freight.”

<details>
<summary>View an actual monthly chart</summary>

![Monthly item-price totals generated during a live Olist evaluation](docs/images/olist-month-line.png)

The metric is the sum of item prices across all order statuses, excluding freight; it is not realised revenue. Missing months are not filled with artificial zero values.

</details>

## Architecture and Design

The fresh-question path below is simplified; cache shortcuts are described in the table that follows.

```mermaid
flowchart TD
    A[Browser question and FastAPI validation] --> B{Write request?}
    B -->|Yes| C[Refuse before model or database access]
    B -->|No| D[Schema-aware SQL generation]
    D --> E[Validated read-only SQL execution]
    E -->|Failure with attempts remaining| F[Repair using execution feedback]
    F --> E
    E -->|Attempts exhausted| G[Controlled failure response]
    E -->|Success| H[Optional plot query and Matplotlib rendering]
    H --> I[Answer, inspectable SQL and optional chart]
    I --> J[Cache successful response]
```

| Design choice | How it solves the problem |
| --- | --- |
| **FastAPI + browser UI** | Separates request validation and backend execution from the interface. Users can inspect the SQL behind each answer. |
| **LangGraph + Groq-hosted LLM** | Makes schema context, SQL generation and execution-feedback retries explicit. Main queries have at most three execution attempts. The tested model is `openai/gpt-oss-20b`, served through Groq. |
| **Read-only SQLite execution** | Validates SQL, opens read-only connections and uses a SQLite authorizer. Common write requests are refused before cache lookup or model calls. |
| **Three cache levels** | Reuses final answers/charts, question-to-SQL mappings and SQL results. Entries expire, database fingerprints invalidate stale data, and prompt changes affect answer/query cache identity. |
| **Bounded requests** | SQL has a **5-second limit per execution**, including plot queries. The public demo allows **5 questions per rolling minute per signed browser session**, with no daily or total quota. Request size, concurrent work and model output are also bounded. |
| **Build-time indexes** | Generic join/filter indexes reduce scans. The Render image additionally prepares Olist-specific month/order and order/price indexes for monthly aggregation. |
| **Matplotlib charts** | A separate read-only plot query supplies numeric data to bar/line renderers. Titles, axes and monthly dates are formatted in code. |
| **Request metrics** | Request IDs, cache source, attempt counts, stage timings and provider token metadata support diagnosis. Structured metrics logs omit questions, SQL and result rows. |

The 5-second SQL bound is not the total response timeout: generating SQL, repairing a query, summarising results and rendering a chart take additional time. A plotting query is generated separately from the main answer query; chart evaluations check its data independently.

[Engineering Notes](docs/ENGINEERING.md) explain cache behaviour, privacy checks, presentation rules and optional tracing.

## CI/CD and Deployment

GitHub Actions provides automated validation; Render is configured to deploy the connected `main` branch after CI checks pass. The dashboard setting was inspected on **5 November 2026**; `render.yaml` also declares `autoDeployTrigger: checksPass`. Blueprint automatic configuration sync is paused so configuration changes require a reviewed manual sync after CI succeeds.

```mermaid
flowchart LR
    A[Push to main] --> B[GitHub Actions CI]
    B --> C[Lint, tests and offline evaluations]
    C --> D[Compose validation and Docker build]
    D --> E[Render deploys after CI checks pass]
    E --> F[Build image, database and analytics indexes]
    F --> G[Readiness check and live demo]
```

- **CI:** dependency checks, Ruff, pytest, offline development/first-use references, Compose validation and a Docker build. It runs without provider credentials.
- **CD:** Render builds the repository's Docker image and deploys the web service, using `/ready` as its health check. SQLite is packaged into the image; no separate PostgreSQL service is required.
- **Secrets:** the Groq key and session-signing secret are supplied at runtime. Local `.env` files are excluded from Git and the Docker build context.

Render reads the GitHub check results before deploying; the CI workflow itself does not publish an image or call the Render deployment API. See [Render's CI integration rules](https://render.com/docs/deploys#integrating-with-ci). This setting has not been tested here by deliberately failing a deployment-branch commit.

See the [Render deployment guide](docs/DEPLOY_RENDER.md) for configuration and hosting limits.

**Availability:** UptimeRobot Free checks `/health` every **30 minutes**; automatic incident detection and recovery are verified. Render Free can sleep and return wake-related errors. [Monitoring settings, cold-start measurement and limitations](docs/PROJECT1_EVALUATION.md#availability-and-wake-up-observations).

## Evaluation and Results

| Check | Latest evidence | Scope |
| --- | --- | --- |
| **Automated regressions** | **314 tests passed**; GitHub CI passed | Identifier grounding, connection recovery and existing application controls |
| Five-run holdout | **80/80 SQL matches; 20/20 chart checks** | The same 16 synthetic questions repeated five times; all 16 passed every run |
| Second-schema evaluation | **24/24 SQL matches; 4/4 chart checks** | User-approved synthetic transit schema; not real Adelaide Metro data |
| Answer numbers backed by SQL results | **71/75 holdout; 16/17 transit** scored answers supported | Literal number checks; context flags and unscored answers documented separately |

SQL checks compare result values, columns, duplicates and required ordering. These small, now-exposed suites do not establish accuracy on arbitrary schemas or complete narrative factuality. The earlier **161-test** result is a historical baseline, not the current test count.

[Evaluation protocols, November results and historical baselines](docs/PROJECT1_EVALUATION.md#evaluation-results-and-history) · [Completion audit](docs/evaluations/project1-completion-2026-11-05/REPORT.md)

### What Changed After Failures

Earlier checks exposed unnecessary output columns, integer division and unsupported currency labels. Fixes combined clearer query instructions with deterministic presentation rules. On Render, distinct-order joins and monthly aggregations also exceeded the earlier 2-second SQL bound; build-time indexes improved those query plans. The current SQL limit is 5 seconds, and timeout failures now explain the execution limit instead of blaming the question's wording. See the [deployment notes](docs/DEPLOY_RENDER.md#monthly-chart-performance-correction) for constrained-container measurements.

## Quick Start

Use the [hosted demo](https://ai-agent-chatbot-demo.onrender.com/) without setup, or run locally with Docker Compose and your own Groq key:

```bash
git clone https://github.com/letriluan/AI-Agent-Chatbot.git
cd AI-Agent-Chatbot
```

Copy `.env.example` to `.env` (`cp .env.example .env` on macOS/Linux, or `Copy-Item .env.example .env` in PowerShell). Set `GROQ_API_KEY` and a `GROQ_MODEL` available to your account. The tested model is `openai/gpt-oss-20b` **hosted by Groq**.

```bash
docker compose up --build
```

Open **http://127.0.0.1:8000** and ask: “How many orders are there in total?”

Compose first ingests the CSVs into a dataset volume, then starts the API as a non-root user with read-only data and a separate writable cache. Redis is not required. Unlike Render's packaged database, the Compose dataset volume uses the generic loader; to add the demo-specific monthly indexes after ingestion, run:

```bash
docker compose run --rm init-db python deploy/prepare_analytics.py --db db/app.sqlite
```

Stop the services with `docker compose down`. Volumes are retained; rerunning ingestion rebuilds the dataset and its generic indexes. Reapply the optional Olist-specific indexes after rebuilding that volume.

<details>
<summary>Local Python setup (Python 3.12)</summary>

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Copy `.env.example` to `.env` and configure Groq, then:

```bash
python -m pip install -r requirements.txt
python -m pip check
python load_db.py --data data --db db/app.sqlite
python deploy/prepare_analytics.py --db db/app.sqlite
python -m uvicorn src.server.app:app --port 8000
```

`GET /health` checks the process. `GET /ready` checks database tables without calling Groq. `POST /ask` accepts a non-blank question of up to 2,000 characters.

</details>

## Run Tests and Evaluations

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
python -m evals.run --mode offline --output reports/offline.json
python -m evals.holdout --mode offline --output reports/holdout-offline
python -m evals.holdout --suite transit --mode offline --output reports/transit-offline
```

Offline evaluations validate references without calling an LLM. Live evaluations make real Groq API calls and use isolated synthetic databases:

```bash
python -m evals.run --mode live --delay-seconds 12 --output reports/live.json
python -m evals.holdout --mode live --delay-seconds 15 --output reports/holdout-rerun
python -m evals.reliability --mode live --runs 5 --delay-seconds 15 --output reports/reliability-live
python -m evals.holdout --suite transit --mode live --delay-seconds 15 --output reports/transit-first-live
```

Live evaluations measure execution-result correctness, chart behaviour and literal numeric support separately. They do not establish complete narrative factuality. The repeated-run command requires a new output directory and refuses to pool different source/configuration fingerprints. Evaluation cost estimates use dated Groq standard rates only when complete token usage is recorded; they do not measure your bill. Generated reports under `reports/` are ignored by Git; selected evidence is archived under `docs/evaluations/`.

## Repository Guide

| Path | Purpose |
| --- | --- |
| `src/agent/` | Workflow, schema context, SQL controls, privacy, cache, charts and metrics |
| `src/server/`, `src/ui/` | FastAPI backend and browser interface |
| `load_db.py`, `data/` | Generic CSV ingestion and Olist source data |
| `deploy/`, `render.yaml` | Olist demo indexes, startup and Render configuration |
| `tests/`, `evals/` | Regressions, fixtures and evaluation runners |
| `docs/` | Engineering notes, deployment guide and archived results |
| `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml` | Packaging, local services and CI |

## Current Scope and Next Steps

This is a deployed project demo with a fixed example dataset. It does not provide authenticated users, tenant-level permissions, a business-data upload workflow or durable billing controls. Questions are independent, rather than a stateful multi-turn conversation. Provider-side schema and allowed query results can be sent to Groq.

Write-intent recognition and column-name privacy checks are conservative, incomplete controls; the SQLite executor independently enforces read-only access. Public limits use signed browser sessions rather than verified identities. Cache and quota state use ephemeral storage on Render, so they can reset after replacement or redeployment.

Next steps are to accumulate longer availability history and test on real Project 2 data when available. The identifier/recovery fixes are deployed and the new five-run study is complete. The eight identifier cases are now exposed regression cases; future prompt changes need another fresh evaluation set. End-user authentication, full narrative factuality and hosted LangSmith delivery remain outside the verified scope. Monetary estimates are distinct from measured billing. This deployment remains a portfolio demo.
