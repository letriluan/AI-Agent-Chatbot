# Deploy the SQLite Demo on Render

This setup deploys one Docker web service. **No Render PostgreSQL database is needed.** The Olist SQLite database is generated from committed CSV files during image build, then made read-only. Every new container starts with the same dataset; cache and quota state are disposable.

## 1. Use the deployment branch

The deployment files are included on the repository's `main` branch: `render.yaml`, the updated Dockerfile, `deploy/start.py` and the demo controls. Select `main` when creating the Blueprint. Never upload your local `.env` to GitHub.

## 2. Create a separate demo key

Use a dedicated Groq key for the public demo. Keep it on a free-only account/project or configure provider-side restrictions that your account actually supports. Local admission limits are not a billing guarantee. Do not paste the key into GitHub, this document, frontend JavaScript, a Docker build argument or a screenshot.

## 3. Create the service

1. Sign in to [Render Dashboard](https://dashboard.render.com/).
2. Choose **New → Blueprint**, connect GitHub, and select `letriluan/AI-Agent-Chatbot` with the branch containing these changes.
3. Render reads the root `render.yaml`. Review the service: Docker runtime, **Free** plan, Singapore region, `/ready` health check. There should be **no Postgres resource and no paid disk**.
4. When prompted for `GROQ_API_KEY`, enter the demo key directly into Render's secret environment-variable field. Paste only the key value, without surrounding quotes. The Blueprint uses `sync: false`, so no secret value is stored in YAML.
5. Confirm `GROQ_MODEL=openai/gpt-oss-20b` is available to the demo account. Click **Deploy Blueprint** and watch the build logs.
6. Wait for the service to become live. Open the generated HTTPS `onrender.com` URL. Render assigns the actual URL; the repository does not assume one in advance.

If a Blueprint flow is unavailable, create **New → Web Service**, connect the repo, select Docker and Free, keep the Docker start command unchanged, set health check `/ready`, and manually copy the non-secret environment values from `render.yaml`. Add the key separately in **Environment**.

The startup script binds to `0.0.0.0:$PORT` and uses exactly one Uvicorn worker. Do not replace it with a multi-worker command or trust all forwarded headers.

## 4. Verify the live demo

- `/health` returns `{"status":"ok"}`.
- `/ready` returns ready status after inspecting the packaged database. It does not call Groq or establish that credentials work.
- Ask: `How many orders are there in total?` Expected count: **99,441**.
- Ask: `Plot the number of orders by order_status as a bar chart.` Check the image and download link.
- Repeated questions should use the cache, but still consume the admission quota.
- Reaching a limit returns **HTTP 429** with a safe message and `Retry-After`. The UI displays that message. This is not an API-key error.
- Add the actual live URL to the README only after verifying it.

## Demo limits

| Control | Default |
| --- | --- |
| Concurrent agent requests | 1; additional requests rejected rather than queued |
| Requests per signed browser session | 5 in a rolling 60 seconds |
| Total requests across the service | No hourly, daily or lifetime quota |
| JSON payload | At most 8,192 bytes, including streamed/chunked bodies |
| Question | At most 2,000 characters |
| LLM completion cap | 1,024 tokens per model call, including whatever the provider counts toward that limit |
| Demo provider timeout/retries | 20 seconds per call; SDK automatic retries disabled |
| SQL execution | Read-only, existing row/time limits |

A question can require several model calls: generation, up to two SQL repairs, an optional chart query and a summary. A request quota is therefore **not a token, API-call or dollar quota**. Admitted requests are counted even on provider failure; blocked/busy requests never enter the agent. Validation failures may also reserve a slot because admission is an API dependency. A user can exhaust the public quota; these limits reduce resource abuse but are not full denial-of-service protection.

Each visitor receives a random browser-session cookie signed by the server. It is HttpOnly and SameSite=Lax; the Render Blueprint sets Secure and generates a stable signing secret (`DEMO_SESSION_SECRET`). The cookie is issued even on API errors. Visitors behind the same proxy get separate session allowances. Forwarding headers are not used as identity.

This is a per-browser limit, not authenticated per-human identity: clearing cookies, using incognito or disabling cookies can obtain a new session. There is intentionally no shared hourly/daily/total quota. The remaining single-request concurrency guard still rejects overlapping work with a short retry message. Provider quotas still apply.

Usage records store a hash of the session ID and admission timestamps, not question text or raw IP. Entries older than 60 seconds are pruned during successful admissions.

## State, secrets and shutdown

- Dataset: `/app/dataset/app.sqlite`, built into the image with read-only file permissions; application connections also enforce read-only access.
- Cache: `/app/cache/cache.sqlite`.
- Admission state: `/app/cache/demo-usage.sqlite`.
- The last two survive a process/container restart only while that writable layer is retained. **Render can discard them on sleep, replacement, redeploy or restart. Limits then start fresh.** No daily or total spending cap is implemented. Use provider-side free/budget controls if needed.
- Keep the supplied single-worker/single-instance configuration. Admission concurrency is process-local; SQLite counters coordinate access only when they share the same file.
- Runtime secrets are not copied into the image. People administering the hosting account can access runtime configuration; visitors are not given the Groq key.
- Disable tracing for this demo unless you deliberately configure and review it. Access logs are disabled by the startup script; application logs exclude SQL, questions and result rows.
- To stop usage immediately, suspend the Render service or revoke the dedicated Groq key. Set `DEMO_MODE=true`; disabling it disables these admission controls.

Render Free currently sleeps after 15 minutes of inactivity and may take about a minute to resume. Free storage is ephemeral. Those restrictions affect this demo; the **30-day expiry is for free Render Postgres**, which this setup does not create. Free services remain subject to Render's usage policies; free hosting does not imply unlimited Groq usage.

## Local verification and reproduction

The earlier deployment-preparation image was tested locally with **512 MB RAM**, `PORT=10000`, a non-root process and no mounted dataset. Live SQL and a chart succeeded using local Groq credentials supplied at runtime; the next request was rejected at the configured test limit, including when a forged forwarding header was sent. Health, readiness and static assets remained available. The earlier transport-peer quota was retained across a container restart that kept its writable layer. The updated policy uses signed browser sessions instead; that historical smoke check does not validate the new session policy. The image contained no `.env` and no Groq key in its configured environment metadata.

The previous preparation passed 127 offline tests on Windows and Docker/Linux. The revised browser-session policy passes **130 tests on Windows and Docker/Linux**; tests cover the sixth request being rejected, allowance returning at 60 seconds, independent browsers, no aggregate hourly/daily limit, signed cookies and first-request failure accounting, in addition to existing regressions. This is local evidence, not confirmation of an actual Render deployment, an official Blueprint API validation or a load test on Render's CPU allocation. The historical 30-case and 16-case live suites were not rerun for the new output cap; this preparation used a live SQL/chart smoke check plus offline regressions.

To run the existing Compose setup with demo controls locally (with a configured local `.env`):

```bash
docker compose -f compose.yaml -f compose.demo.yaml up --build
```

This uses the local Compose dataset/cache volumes. Render instead uses the image's packaged dataset directly. Keep `.env` values local. Render Environment fields should contain bare values without dotenv quote characters.

For offline tests:

```bash
python -m pytest -q
python -m ruff check .
docker compose -f compose.yaml -f compose.demo.yaml config --quiet
```

## Official references

- [Render free service limits](https://render.com/docs/free)
- [Blueprint fields and secret prompts](https://render.com/docs/blueprint-spec)
- [Runtime environment variables](https://render.com/docs/configure-environment-variables)
- [Web service port binding](https://render.com/docs/web-services#port-binding)
- [Groq rate limits](https://console.groq.com/docs/rate-limits)

## Monthly chart performance correction

The current SQL execution timeout is **5 seconds per query attempt**, including plot queries, with at most three main-query attempts. `SQL_TIMEOUT_SECONDS=5.0` is declared in `render.yaml` and `.env.example` and is also the code default. This is a SQL execution bound, not the total request or Groq API timeout. Measurements below describe the earlier 2-second configuration.

The monthly item-price chart also exhausted three 2-second SQL attempts on Render. `deploy/prepare_analytics.py` now adds an Olist-specific month/order index and a covering order/price index during the Docker build, keeping the generic CSV loader independent of this dataset. The exact failing SQL returned 24 months and a total item price of 13,591,643.70 in three local Docker runs at 0.1 CPU / 512 MB, taking 707–793 ms. A live Groq request under the same limits produced the monthly chart on its first SQL attempt; the distinct delivered-order JOIN still returned 96,478. These are local constrained-container measurements, not Render timing guarantees. Windows regressions: 161 passed. Those measurements used the earlier 2-second SQL limit; the current default is 5 seconds. Remaining timeout failures explain the time limit rather than asking users to rephrase valid questions.

### Earlier JOIN correction

A live Olist distinct-order JOIN exhausted three 2-second SQL attempts on the first Render deployment. Build-time indexes now cover inferred child join keys and low-cardinality text filters paired with a table primary key. On the rebuilt dataset, the exact query returned 96,478 in three local Docker runs capped at 0.1 CPU / 512 MB, taking 406–496 ms per run. These earlier measurements used a 2-second query timeout; the current default is 5 seconds. Rebuild/redeploy the Docker image to obtain these indexes; restarting an old image does not change its packaged database. Windows regressions: 131 passed.
