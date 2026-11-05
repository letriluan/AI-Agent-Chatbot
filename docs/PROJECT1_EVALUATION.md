# Project 1: reliability, second schema and observability

Status as of **5 November 2026**: **6/6 agreed roadmap deliverables have evidence**, including external automatic monitoring. [Completion audit](evaluations/project1-completion-2026-11-05/REPORT.md). The offline fixture checks and the separately archived live studies have distinct scopes.

**Execution constraint (5 November 2026): free-only Groq usage.** Paid credits, paid fallbacks and billing upgrades are not authorised. The initial local key used the same user-confirmed free-only organisation as Render. After that collection stopped on HTTP 429, the user configured a separate Free-account key locally. A new complete batch was started in separate output directories; the restricted attempt is preserved and is not pooled with the new batch. The console displayed Free as Current Plan before the new batch. The API key's organisation was not independently established through the API. Render's existing credential was retained. Having a key or a list-price estimate alone does not establish a free-only account. Every runner stops on provider quota/access restrictions instead of attempting a paid fallback.

| Roadmap item | Verified evidence | Limits and future work |
| --- | --- | --- |
| 1. Deploy after CI | Dashboard CI gate inspected; main CI passed and the next Auto-Deploy reached Render. Early Blueprint build canceled; Auto Sync now No | Configuration changes require reviewed Manual sync after CI; a deliberately failing main commit was not tested |
| 2. Five repeated runs | New complete fixed-version study: **80/80**, all 16 cases pass all five runs, **20/20 charts**; original baseline and quota-stopped attempt preserved | Broader future regressions remain useful; these are exposed synthetic questions, not a general accuracy guarantee |
| 3. Second schema | Full fixed-version rerun: **24/24 SQL, 4/4 charts**; S4 returns R2 on first SQL attempt. Original 23/24 and eight-case identifier follow-up preserved | Real Project 2 data and additional unseen schemas remain future work |
| 4. Answer numbers | Fixed holdout: **71/75 scored (94.7%)**; transit: **16/17 (94.1%)**. Five question-context flags manually reviewed; 12 unscored observations | Full semantic factuality remains unmeasured; no-number answers are not successful numeric checks |
| 5. Availability | UptimeRobot Free automatic detection and recovery verified; 30-minute interval, 60-second timeout. Log-confirmed cold start: **32.85 s** | Free-service wake 503s remain recorded; longer history needed for sustained uptime claims |
| 6. Estimated question cost | Fixed holdout: **USD 0.019544175**, 76/80 complete usage; transit: **USD 0.005380875**, 23/24. Original estimates preserved | Five recovered requests have missing usage; no full-suite mean or measured bill is claimed |

The subsequent local fix is validated separately: **4/4 affected regressions and 8/8 newly authored identifier/name cases passed**, including three chart checks with visual inspection. The `S4` lookup now returns `R2` on its first generated-SQL attempt. A controlled pre-network error and one naturally observed connection/timeout failure were each recovered with one application retry. **314 offline tests passed**. The original reports are unchanged; this targeted follow-up is not a replacement five-run reliability study and the source changes deployed to Render on 5 November in commit `5df7de9`. [Follow-up report and frozen evidence](evaluations/lookup-connection-followup-2026-11-05/REPORT.md).

## Local environment

The earlier fixed-version attempt stopped at **22/23 saved observations** with HTTP 429, and is [preserved unchanged](evaluations/fixed-rerun-partial-2026-11-05/REPORT.md). After the user supplied a separate Free-account key, a new complete five-run batch achieved **80/80 SQL and 20/20 chart checks**, with **16/16 cases correct in all five runs**. The two collections are not combined. [Complete fixed-version study](evaluations/reliability-fixed-live-2026-11-05/REPORT.md). Render kept its existing credential and its hosted SQL/chart smoke checks passed.

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

The original pre-fix five-run baseline is complete: **77/80 successful observations**, with all 77 returned SQL results matching. Three requests failed with `APIConnectionError` (run 4: h12/h15; run 5: h16). **13/16 cases passed every run (81.25%)**; the three mixed cases each failed once because of a connection error. Chart cases succeeded in 18/20 observations; all 18 produced PNGs are covered by visual inspection of their exact image hashes. Numeric support was 67 supported, six contextual/result-shape flags and seven unscored observations. The flags repeat the date filter or say “three bars” for three result rows; they were manually reviewed. No full semantic factuality rate is claimed. [Full report, raw runs and image review](evaluations/reliability-live-2026-11-05/REPORT.md).

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

This runner tests the currently deployed Olist version; running it does not deploy application changes. The public API does not expose the deployed commit/model identity, chart-query rows, cache bypass or database switching. It cannot run the original synthetic holdout or transit suite, and repeated cache hits are not fresh-model reliability evidence. Chart images need separate visual inspection. Numeric-support flags remain separate from SQL correctness, including contextual phrases such as “at least one item” and “Top 5”.

## Synthetic transit schema

`fact_stop_events` has one scheduled stop event per trip and service date. It references `dim_route`, `dim_stop`, `dim_date` and `dim_time`. The fixture contains 14 events, six trips, four routes and six stops. An unused route/stop/date, tied stop counts and after-midnight events exercise edge cases. All route names and data are invented. This is a simplified star-schema adaptation, not a complete GTFS feed or an Adelaide Metro dataset.

Arrival/departure values are integer seconds from service-day midnight; 25:05 remains part of its original service day. This follows the [GTFS service-time convention](https://gtfs.org/documentation/schedule/reference/#field-types), without claiming a full GTFS importer.

The 24 cases cover distinct counts versus stop-event counts, relationships, left joins with zeros, date boundaries, an empty Sunday, duration/dwell aggregation, after-midnight filtering and four real-renderer chart checks.

```powershell
python -m evals.holdout --suite transit --mode live --delay-seconds 15 --output reports/transit-first-live-2026-11-05
```

At the original first-use collection, these developer-written questions had not been used to alter production prompts. That first 23/24 result is preserved. Its stop-code failure informed a later fix, validated separately on eight new identifier/name cases and then this exposed 24-case regression suite. Future prompt changes need fresh evaluation questions, and real Adelaide Metro data must be evaluated separately.

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

The separate GitHub diagnostic is manually dispatched and retains artifacts for 30 days. UptimeRobot owns the automatic 30-minute schedule. Each GitHub diagnostic needs only standard Python and no provider credentials. Failures surface failed samples; they do not modify the service.

**Deployed measurement, 5 November:** the initial GitHub diagnostic passed when manually dispatched. Scheduled GitHub delivery was never verified. UptimeRobot Free is now the primary automatic monitor: GET `/health`, case-sensitive `"status":"ok"`, missing-keyword incident, every 30 minutes with a 60-second timeout. Automatic detection and first recovery were independently recorded by the provider; subsequent wake failures remain in the history. GitHub now provides an on-demand diagnostic only. [Automatic monitoring evidence and caveats](evaluations/project1-completion-2026-11-05/REPORT.md). For the controlled experiment, monitoring was paused and the service received no agent probes for 20 minutes. Render logged shutdown of instance `rxg2z` at 05:40:32 UTC. The wake request began at 05:45:32.674 UTC, initially received HTTP 503, and obtained healthy JSON after **32,851.516 ms**. Render logged startup of a new instance `jp69m` at 05:45:56 UTC; the same deployment remained Live. Immediate warm health/readiness checks took **234.478 / 198.684 ms**. The first monitoring artifact recorded **313.694 / 258.188 ms**, with no model calls. [Deployment, cold-start and monitoring evidence](evaluations/render-fixed-deployment-2026-11-05/REPORT.md). The generic probe's cold-start flag remains false; a separate review establishes attribution from the platform lifecycle logs.

Render Free services [sleep after 15 minutes without inbound traffic](https://render.com/docs/free#spinning-down-on-idle). Monitoring requests can wake or keep an instance active, which affects wake-up measurements. For a quiet-window experiment, pause external checks and avoid visiting the demo for at least 20 minutes, then run with `--declared-quiet-seconds 1200`. Other users can still generate traffic, so this declaration does not confirm platform sleep. Correlate with Render events/logs before calling a measurement a confirmed cold start. Report sampled availability over a stated observation window, not continuous uptime or an SLA.

## Cost estimates

Rates checked on **5 November 2026** for `openai/gpt-oss-20b`: **USD 0.075 per million input tokens** and **USD 0.30 per million output tokens**, from the [official model pricing page](https://console.groq.com/docs/model/openai/gpt-oss-20b).

`estimated_usd = (input_tokens × 0.075 + output_tokens × 0.30) / 1,000,000`.

Only complete recorded provider usage is estimated. If any successful model call lacks input/output counts, `token_usage_complete` is false. Unknown models and provider failures without usage remain unestimated. Zero-call cached answers have complete zero usage. The repeated-run summary reports coverage, the known subtotal and a per-question mean only if all observations have estimates. Cached-input/batch discounts and free-tier credits are not applied; this is a standard-list-price estimate, never a measured bill or a billing limit.

The original reports cover all 24 transit questions (mean standard-list-price equivalent **USD 0.000248950**) and 77/80 holdout observations (known subtotal **USD 0.019320525**). The three connection failures did not return usage, so the full holdout mean is unavailable. [Machine-readable estimates and coverage](evaluations/cost-estimates-2026-11-05.json).

The fixed-version studies have a known standard-list-price subtotal of **USD 0.024925050**, covering **99/104** observations: holdout USD 0.019544175 (76/80) and transit USD 0.005380875 (23/24). Five recovered requests have incomplete usage, so no full-suite mean is claimed. [Fixed-version price calculation and coverage](evaluations/cost-estimates-fixed-2026-11-05.json).

## Evidence boundaries

Initial local verification: **255 pytest tests passed** on Windows/Python 3.12.13; Ruff, dependency compatibility and both Compose configurations passed. The offline rehearsal validated **80/80 reference checks across five runs**, the transit suite **24/24**, and the development suite **30/30**. [Initial offline record](evaluations/project1-offline-verification-2026-11-05.json). After adding the HTTP runner and free-tier stop/resume controls, **274 tests passed** with the real key replaced by a dummy key in the offline test process. [Later setup/test record](evaluations/local-groq-setup-2026-11-05.json).

Keep historical reports unchanged. New reference checks, scripted tests and live observations must retain their separate scopes. Do not turn offline reference passes into model-quality or cross-domain accuracy claims. Inspect charts visually after live evaluation; automatic rendering checks do not constitute visual review. GitHub CI passed on the merged commit, including the offline evaluations and Linux Docker image build. Historical GitHub diagnostic observations remain archived. UptimeRobot now owns automatic monitoring; GitHub retains manual dispatch only.

Docker Desktop's local engine was unavailable; Compose configuration was checked locally and the image was subsequently built by GitHub CI and Render. The offline fixture work made no model calls. Public Render checks use the hosted key under the user's free-only confirmation. The local key is authenticated and both synthetic live studies are complete; their separate reports preserve the observed SQL failure, connection failures and missing-usage limits.

## Evaluation results and history

**Fixed-version five-run result, 5 November 2026:** **80/80 observations and 20/20 chart checks passed; all 16 cases passed every run (100%)**. Four naturally interrupted requests recovered within the shared two-retry limit. The incomplete earlier 22/23 attempt stopped on Free-tier quota and remains separate. [Complete fixed-version study](evaluations/reliability-fixed-live-2026-11-05/REPORT.md) · [Preserved quota-restricted attempt](evaluations/fixed-rerun-partial-2026-11-05/REPORT.md).

### Public-demo checks

After deploying the fixes on 5 November, two fresh hosted questions passed independent SQL-result checks (99,441 total orders and a chart of order statuses). They made five observed model calls; both cache repeats passed with zero model calls, including identical chart-image reuse. Health, readiness, homepage, write refusal and invalid-input checks also passed. Render's existing user-confirmed Free Groq credential was retained. [Fixed-version deployment smoke and chart review](evaluations/render-fixed-deployment-2026-11-05/REPORT.md).

On **5 November 2026**, all **ten current demo checks passed** through the public Render API: nine SQL results matched independent Olist references and the write request was refused with zero model calls. The nine data questions were uncached and made 21 observed model calls; all three chart PNGs were visually inspected. Three cache repeats preserved SQL/images with zero model calls, and two invalid inputs returned HTTP 422. This is one known-question deployment check, not the five-run synthetic reliability study or transit evaluation. [Earlier full-suite report, raw results and chart review](evaluations/render-live-2026-11-05/REPORT.md).

On **18 October 2026**, the original five sample questions were tested against the Render `/ask` endpoint after deploying the 5-second SQL configuration. The first four requests used no cached answers and succeeded on their first SQL attempt. Results were compared with reference queries over the source dataset, and both chart images were visually inspected.

| Sample | Verified result | Observed total time |
| --- | --- | --- |
| Total orders | **99,441** | 3.05 s |
| Orders by status | Correct counts across eight statuses and a bar chart | 5.15 s |
| Distinct delivered orders with items | **96,478**, counting each order once | 3.26 s |
| Monthly item-price totals | **24 months**, reference totals and a line chart | 7.83 s |
| Delete cancelled orders | Explicit refusal, empty SQL and **zero model calls** | 0.34 s |

These are single-run end-to-end observations, including network/provider work, not performance guarantees or a load test. The browser was also checked for chart rendering and cached-response reuse. [Archived public-demo results](evaluations/render-demo-2026-10-18.json).

The five added prompts (06–10) were also verified against the live API and source-data references before the UI release: customer-state rankings, delivered-order percentage, payment-type counts/chart, product-category rankings and average item price. All succeeded on their first SQL attempt. [Added-sample results](evaluations/added-samples-2026-10-18.json).

### Regression and earlier fixture evaluations

| Check | Evidence | Scope |
| --- | --- | --- |
| Earlier automated regressions | **161 passed** locally on Windows; Linux GitHub CI passed | Historical baseline before the November evaluation additions |
| Earlier development SQL suite | **30/30** strict matches | Questions used during prompt development |
| Earlier new-question first-use suite | **16/16** main SQL matches | Separate synthetic fixture; 12 SQL-only and four chart requests |
| First-use chart checks | **4/4 passed** | Plot-query rows, renderer inputs, chart type, labels, PNG output and cached-image reuse |

Strict SQL matching compares returned values, column count, duplicates and required ordering, rather than SQL wording. The first-use questions were evaluated with frozen prompts before using them as regression cases; they are no longer unseen. These small suites do not establish correctness on arbitrary schemas or factuality of every generated sentence.

[First-use protocol and report](HOLDOUT_EVALUATION.md) · [Development results and validation history](VALIDATION.md)

### November evaluation additions

- The **full fixed-version transit rerun passed 24/24 SQL and 4/4 chart checks**, including the corrected `S4` → `R2` lookup on its first SQL attempt. All 24 cached repeats used zero model calls. Numeric support was **16/17 scored answers (94.1%)**, one question-context flag reviewed manually and seven unscored. Known list-price subtotal **USD 0.005380875** covers 23/24 questions; one recovered request has incomplete usage. This is synthetic regression data, and the original 23/24 first-use failure remains preserved. [Complete transit rerun](evaluations/transit-fixed-live-2026-11-05/REPORT.md).

- The **fixed-version five-run study passed 80/80 observations**, with **16/16 cases correct in every run and 20/20 chart checks**. All 80 cached repeats used zero model calls. Numeric support was **71/75 scored answers (94.7%)**, four question-date context flags reviewed manually, and five unscored answers. Known list-price estimate **USD 0.019544175** covers **76/80** questions; recovered requests with incomplete usage remain unknown, so no full mean or measured bill is claimed. [Full fixed-version report](evaluations/reliability-fixed-live-2026-11-05/REPORT.md).

- **314 automated tests passed** locally on Windows/Python 3.12, including identifier grounding, bounded connection recovery and free-tier stop/resume controls. Lint, dependency compatibility and Compose configuration checks also passed. The merged commit also passed GitHub CI, including the Linux Docker build, and deployed to Render. [Latest fix validation](evaluations/lookup-connection-followup-2026-11-05/REPORT.md) · [Initial offline verification](evaluations/project1-offline-verification-2026-11-05.json).
- The **original pre-fix baseline** of five fresh-cache regression runs collected **80 observations**: **77 returned SQL results matched**, while three requests failed with `APIConnectionError`. **13/16 questions passed in every run (81.25%)**; chart cases succeeded in **18/20** observations, with two connection failures. Every returned SQL result matched; failed requests remain in the denominator. These are 16 exposed questions repeated five times, not 80 unseen questions. [Repeated-run report and raw evidence](evaluations/reliability-live-2026-11-05/REPORT.md).
- A first live run on **24 synthetic transit questions** achieved **23/24 strict SQL matches and 4/4 chart checks**, with all four PNGs visually inspected. One stop-code lookup incorrectly filtered the stop-name column; the failure is preserved. This uses an invented five-table star schema, not Adelaide Metro data or a general cross-domain benchmark. [First-live report and failure analysis](evaluations/transit-first-live-2026-11-05/REPORT.md).
- A conservative numeric-support check compares answer numbers with executed SQL cells. Applied retrospectively to the archived October holdout, **14 answers have supported numbers, one is flagged and one has no numeric claims**. The flagged answer repeats dates supplied in the question but not returned by SQL. This is a review flag, not proof of a false statement; numeric support does not establish semantic factuality. [Retrospective report](evaluations/numeric-support-archived-holdout-2026-11-05.json).
- The controlled idle trial confirmed actual shutdown/restart in Render logs and measured **32.85 s to healthy**, preserving an initial 503. UptimeRobot automatic detection and recovery are now verified at the user-selected 30-minute interval. [Completion audit](evaluations/project1-completion-2026-11-05/REPORT.md) · [Confirmed cold-start study](evaluations/render-fixed-deployment-2026-11-05/REPORT.md).
- Initial dated token-based cost estimates cover **24/24 transit questions and 77/80 holdout observations**. Missing usage on the three failed requests is not treated as zero, so no complete holdout mean is claimed. Estimates use standard list prices and do not represent measured charges. [Estimate record and coverage](evaluations/cost-estimates-2026-11-05.json).
- A separate local follow-up corrected the stop-code lookup: **4/4 affected regressions and 8/8 new identifier/name cases passed**, with three chart PNGs visually reviewed. A controlled pre-network failure and one naturally observed transport failure both recovered after one application retry. Original benchmark failures remain preserved; this is a targeted follow-up, not a new five-run reliability study. These fixes deployed to Render in commit `5df7de9` on 5 November. [Follow-up evidence](evaluations/lookup-connection-followup-2026-11-05/REPORT.md).

[Protocol, commands and remaining work](PROJECT1_EVALUATION.md)

### Fixed-version Render rollout

The identifier and connection-recovery fixes reached Render in commit `5df7de9` on 5 November. Main CI passed at 05:23:08 UTC; Render's subsequent Auto-Deploy started at 05:23:11 UTC and became live at 05:24:39 UTC. An earlier Blueprint-triggered build was canceled and Blueprint Auto Sync disabled after it exposed a separate configuration-deployment path. Health, readiness, homepage, write refusal (zero model calls) and invalid-input checks passed on the new deployment.
