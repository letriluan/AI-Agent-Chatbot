# Project 1: reliability, second schema and observability

Status as of **5 November 2026**. The offline fixture checks and the separately archived live studies have distinct scopes.

**User-confirmed execution constraint (5 November 2026): free-only Groq usage.** Paid credits, paid fallbacks and billing upgrades are not authorised. A local key was configured on 5 November; authentication and availability of `openai/gpt-oss-20b` were checked without displaying the key. The user was instructed to use the same confirmed free-only organisation as Render. Having a key or a list-price estimate does not establish that an account is free-only. If the free allowance is exhausted, retain partial results and wait rather than incur charges.

| Roadmap item | Completed locally | Still needed |
| --- | --- | --- |
| 1. Deploy after CI | Maintainer confirmed Render setting; Blueprint and README updated | An intentionally failing deployment-branch experiment was not performed |
| 2. Five repeated runs | Original baseline: 77/80, 13/16 cases pass all five. Fixed-version rerun: run 1 passes 16/16; run 2 passes six then stops at HTTP 429 | Resume the incomplete fixed-version study after free allowance is available; no all-five fixed-version claim |
| 3. Second schema | First live run preserved at 23/24 SQL matches; local fix then passes the affected case and eight new identifier/name cases | Full fixed-version 24-case transit rerun has not started because of the free-tier restriction; real Project 2 data remains future work |
| 4. Answer numbers | New live reports include numeric-support counts; all seven automatic context/result-shape flags across both studies were reviewed | Full semantic factuality remains unmeasured; no-number answers do not count as successful grounding checks |
| 5. Availability | Process/database probe, one archived live observation and scheduled workflow | Publish workflow and collect history; confirm actual platform sleep for a cold-start claim |
| 6. Estimated question cost | Dated estimates cover 24/24 transit questions and 77/80 holdout observations | Failed-request usage is unavailable; no full holdout mean or measured bill is claimed |

The subsequent local fix is validated separately: **4/4 affected regressions and 8/8 newly authored identifier/name cases passed**, including three chart checks with visual inspection. The `S4` lookup now returns `R2` on its first generated-SQL attempt. A controlled pre-network error and one naturally observed connection/timeout failure were each recovered with one application retry. **314 offline tests passed**. The original reports are unchanged; this targeted follow-up is not a replacement five-run reliability study and the source changes are not yet deployed to Render. [Follow-up report and frozen evidence](evaluations/lookup-connection-followup-2026-11-05/REPORT.md).

## Local environment

The requested full fixed-version rerun stopped on the free-tier restriction: **22/23 saved observations passed**, with one HTTP 429 retained. The completed first run includes a naturally recovered connection failure and four visually inspected charts. The second run and subsequent runs remain incomplete; the full transit rerun has not started. [Immutable partial evidence and resume instructions](evaluations/fixed-rerun-partial-2026-11-05/REPORT.md). The Groq console displayed Free as the current plan; no billing change was made.

Use Python 3.12 and the existing pinned requirements. A local `.venv` and ignored `.env` copied from `.env.example` have been prepared for this checkout. Set `GROQ_API_KEY` in that local file using your editor. Keep the key out of chat, source control and report files. Use a free-only account/project or provider-side restrictions as described in [deployment guidance](DEPLOY_RENDER.md).

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pytest -q
python -m ruff check .
python -m evals.holdout --mode offline --output reports/holdout-offline
python -m evals.holdout --suite transit --mode offline --output reports/transit-offline
```

If the environment was installed with `uv`, check dependency compatibility with `uv pip check --python .venv\Scripts\python.exe`.

### Free-only live setup

Groq account billing controls determine the tier; a key or local flag cannot turn a paid organisation into a free one. Use the Free organisation, without upgrading to Developer. [Groq billing documentation](https://console.groq.com/docs/billing-faqs).

The local evaluation session uses the same bounded client settings as the demo: a 1,024-token completion cap, 20-second provider timeout and **zero SDK automatic retries**, with tracing off. Set these process settings before the live commands:

```powershell
$env:DEMO_MODE = 'true'
$env:LANGCHAIN_TRACING_V2 = 'false'
$env:LANGSMITH_TRACING = 'false'
```

The later connection-handling fix adds `LLM_CONNECTION_RETRIES=2`: at most two **application-level** connection/timeout retries shared across one question, while SDK retries remain disabled. HTTP errors, including quota and billing/access rejections, still do not retry. This changed configuration is recorded in the new follow-up; it was not applied retroactively to the original five runs. Recovered requests with missing usage keep an incomplete-usage flag. [Recovery policy](ENGINEERING.md#connection-recovery).

Use `--delay-seconds 30` for the current free-only session. The account's actual token/request allowances are shared across all keys and the Render demo; the [official limit summary](https://console.groq.com/docs/rate-limits) is not a guarantee of remaining allowance. A free-tier limit can require continuing in a later allowance window.

The runners stop on provider HTTP 401/402/403/429, save the current observation and return a nonzero incomplete status. They never upgrade or switch providers. A stopped batch can be continued with the **same frozen sources, model and configuration**:

```powershell
python -m evals.holdout --suite transit --mode live --delay-seconds 30 --output reports/transit-first-live-2026-11-05 --resume
python -m evals.reliability --mode live --runs 5 --delay-seconds 30 --output reports/reliability-live-2026-11-05 --resume
```

Resume archives the prior partial run, reuses completed runs and skips already observed questions. A saved provider-limit failure stays in the results and denominator; it is not silently retried or replaced with a successful answer. No all-five reliability result is claimed until all planned observations have been collected.

## Repeated-run protocol

```powershell
python -m evals.reliability --mode live --runs 5 --delay-seconds 15 --output reports/reliability-live-2026-11-05
```

This evaluates the existing 16 exposed holdout questions five times: **80 fresh questions**, plus a cached repeat of each question. Questions can make several provider calls, including SQL generation, repairs, optional chart queries and summarisation. The runner uses a different temporary cache per case per run; cached repeats verify reuse and do not count as additional independent trials. No web deployment changes are required.

Prompts, agent implementation, suite, fixture and evaluation sources are hashed. Model, output cap, demo settings and SQL limits are recorded; different fingerprints/configurations cannot be pooled. Do not change prompts during measurement. This measures local fresh-cache model variation, not end-to-end behaviour of the Render deployment or a blinded external benchmark.

Each `run-XX/results.json` retains SQL, executed rows, answer, chart checks, token usage, numeric-support results and errors. Chart PNGs are kept in that run's directory. `summary.json` and `REPORT.md` are updated after each complete run. An interrupted collection is labelled incomplete and has no all-five success claim. Existing non-empty output directories are refused to preserve evidence.

Per-case pass means correct execution results plus any required chart checks. The report separately counts SQL matches, all-five successes, mixed outcomes and provider failures. Failures remain in the denominator. Numeric support is separate from SQL/chart success.

The five-run study is complete: **77/80 successful observations**, with all 77 returned SQL results matching. Three requests failed with `APIConnectionError` (run 4: h12/h15; run 5: h16). **13/16 cases passed every run (81.25%)**; the three mixed cases each failed once because of a connection error. Chart cases succeeded in 18/20 observations; all 18 produced PNGs are covered by visual inspection of their exact image hashes. Numeric support was 67 supported, six contextual/result-shape flags and seven unscored observations. The flags repeat the date filter or say “three bars” for three result rows; they were manually reviewed. No full semantic factuality rate is claimed. [Full report, raw runs and image review](evaluations/reliability-live-2026-11-05/REPORT.md).

An offline rehearsal validates references only:

```powershell
python -m evals.reliability --mode offline --runs 5 --output reports/reliability-offline-2026-11-05
```

## Test the existing Render deployment

The public deployment already has a provider key, so testing its current Olist dataset does **not** require a local Groq key. The user confirmed the hosted account is free-only before model-backed requests were sent.

```powershell
python load_db.py --data data --db db/render-validation.sqlite
python deploy/prepare_analytics.py --db db/render-validation.sqlite
python -m evals.render --free-only-confirmed --output reports/render-live-new
```

The HTTP runner uses one signed browser session and at least 15 seconds between requests. It checks the ten current demo questions, compares returned SQL with independent Olist reference queries, saves three chart PNGs, repeats three queries to check cache reuse, and tests blank/oversized inputs. It stops on unexpected HTTP errors rather than bypassing quotas, retrying the model automatically or changing account settings. Generated SQL runs locally only through the bounded read-only executor. Source CSV hashes and observed provider-call counts are retained; credentials and session cookies are not recorded.

**5 November result:** ten demo checks passed (nine strict SQL matches plus write refusal); the nine data requests were uncached and made 21 observed model calls. Three charts were visually inspected, three cache repeats preserved SQL/images with zero model calls, and both invalid inputs returned HTTP 422. Three automatic numeric-support flags were manually reviewed as question/result-shape context: “at least one item” and two “Top 5” headings. [Full report and archived evidence](evaluations/render-live-2026-11-05/REPORT.md).

This tests the existing hosted version. It does not deploy this branch's new tools. The public API does not expose the deployed commit/model identity, chart-query rows, cache bypass or database switching. It cannot run the original synthetic holdout or transit suite, and repeated cache hits are not fresh-model reliability evidence. Chart images need separate visual inspection. Numeric-support flags remain separate from SQL correctness, including contextual phrases such as “at least one item” and “Top 5”.

## Synthetic transit schema

`fact_stop_events` has one scheduled stop event per trip and service date. It references `dim_route`, `dim_stop`, `dim_date` and `dim_time`. The fixture contains 14 events, six trips, four routes and six stops. An unused route/stop/date, tied stop counts and after-midnight events exercise edge cases. All route names and data are invented. This is a simplified star-schema adaptation, not a complete GTFS feed or an Adelaide Metro dataset.

Arrival/departure values are integer seconds from service-day midnight; 25:05 remains part of its original service day. This follows the [GTFS service-time convention](https://gtfs.org/documentation/schedule/reference/#field-types), without claiming a full GTFS importer.

The 24 cases cover distinct counts versus stop-event counts, relationships, left joins with zeros, date boundaries, an empty Sunday, duration/dwell aggregation, after-midnight filtering and four real-renderer chart checks.

```powershell
python -m evals.holdout --suite transit --mode live --delay-seconds 15 --output reports/transit-first-live-2026-11-05
```

These developer-written questions have not been used to alter production prompts. After the first actual model run they become exposed regression cases. Record that first result even if it fails. Any later prompt improvement needs another fresh evaluation, and real Adelaide Metro data must be evaluated separately.

The first live run is complete: **23/24 strict SQL matches, 4/4 chart checks, 24/24 cached repeats and zero provider exceptions**. `t15` treated stop code `S4` as a stop name and returned no route rather than `R2`, despite three attempts. The failure is preserved and the prompts remained unchanged. Numeric support was 16 supported, one question-context flag and seven unscored answers. All 24 cases have complete reported token usage; the dated standard-list-price estimate averages **USD 0.000248950 per question**, not a measured bill. [Results, visual review and failure analysis](evaluations/transit-first-live-2026-11-05/REPORT.md).

## Numeric support and its limits

The checker compares numeric claims with **actual executed, redacted SQL cells**, allowing separators, matching date/time strings, single-word number forms and displayed decimal rounding. It excludes Markdown list numbering. It does not use expected/reference rows, question text, SQL constants or separate chart results to excuse an unsupported number. Compound word numbers are left for review. Empty/no-number answers are unscored, not successful factuality checks.

This catches invented numbers. It cannot verify that a number belongs to the right category, that a currency/unit is justified, that all required results were mentioned, or that an empty-result explanation is correct. A semantically wrong answer can pass; a correct restatement of question dates can be flagged. Review flags before drawing conclusions.

The [retrospective report](evaluations/numeric-support-archived-holdout-2026-11-05.json) rechecks the **archived 18 October answers** without model calls. Of 16 answers: 14 have supported numbers, one is flagged and one contains no numeric claims. `h09` repeats April 2026, April 30 and May 1 from the question, while SQL returned only order IDs. Those dates are not in the returned cells; this flag is not evidence of a false answer. This is not a new 5 November live accuracy score.

To review a different archived report:

```powershell
python -m evals.grounding --report reports/transit-first-live-2026-11-05/results.json --output reports/transit-numeric-review.json
```

## Availability and wake-up observations

```powershell
python -m evals.availability --output reports/availability-new.json
```

The probe calls `/health`, then `/ready`, requires the expected JSON rather than accepting a loading page, and records HTTP status and elapsed time. It never calls `/ask`. The [5 November observation](evaluations/render-availability-2026-11-05.json) returned healthy JSON in **32,583.573 ms**, followed by ready JSON in **218.493 ms**. This is one observed request pair, not an uptime percentage or a confirmed cold start.

The separate GitHub workflow samples twice an hour and uploads 30-day-retention artifacts. It becomes active after publication on the default branch and can also be run manually. Each run needs only standard Python and no provider credentials. Workflow failures surface failed samples; they do not modify the service.

Render Free services [sleep after 15 minutes without inbound traffic](https://render.com/docs/free#spinning-down-on-idle). Monitoring requests can wake or keep an instance active, which affects wake-up measurements. For a quiet-window experiment, pause external checks and avoid visiting the demo for at least 20 minutes, then run with `--declared-quiet-seconds 1200`. Other users can still generate traffic, so this declaration does not confirm platform sleep. Correlate with Render events/logs before calling a measurement a confirmed cold start. Report sampled availability over a stated observation window, not continuous uptime or an SLA.

## Cost estimates

Rates checked on **5 November 2026** for `openai/gpt-oss-20b`: **USD 0.075 per million input tokens** and **USD 0.30 per million output tokens**, from the [official model pricing page](https://console.groq.com/docs/model/openai/gpt-oss-20b).

`estimated_usd = (input_tokens × 0.075 + output_tokens × 0.30) / 1,000,000`.

Only complete recorded provider usage is estimated. If any successful model call lacks input/output counts, `token_usage_complete` is false. Unknown models and provider failures without usage remain unestimated. Zero-call cached answers have complete zero usage. The repeated-run summary reports coverage, the known subtotal and a per-question mean only if all observations have estimates. Cached-input/batch discounts and free-tier credits are not applied; this is a standard-list-price estimate, never a measured bill or a billing limit.

The completed reports cover all 24 transit questions (mean standard-list-price equivalent **USD 0.000248950**) and 77/80 holdout observations (known subtotal **USD 0.019320525**). The three connection failures did not return usage, so the full holdout mean is unavailable. [Machine-readable estimates and coverage](evaluations/cost-estimates-2026-11-05.json).

## Evidence boundaries

Initial local verification: **255 pytest tests passed** on Windows/Python 3.12.13; Ruff, dependency compatibility and both Compose configurations passed. The offline rehearsal validated **80/80 reference checks across five runs**, the transit suite **24/24**, and the development suite **30/30**. [Initial offline record](evaluations/project1-offline-verification-2026-11-05.json). After adding the HTTP runner and free-tier stop/resume controls, **274 tests passed** with the real key replaced by a dummy key in the offline test process. [Later setup/test record](evaluations/local-groq-setup-2026-11-05.json).

Keep historical reports unchanged. New reference checks, scripted tests and live observations must retain their separate scopes. Do not turn offline reference passes into model-quality or cross-domain accuracy claims. Inspect charts visually after live evaluation; automatic rendering checks do not constitute visual review. The new workflow has not yet run on GitHub, and a Docker image has not been rebuilt for these changes.

Docker Desktop's engine was unavailable during verification; only Compose configuration validation was possible. The offline fixture work made no model calls. Public Render checks use the hosted key under the user's free-only confirmation. The local key is authenticated and both synthetic live studies are complete; their separate reports preserve the observed SQL failure, connection failures and missing-usage limits.
