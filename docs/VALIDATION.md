## New-question evaluation update

The frozen-prompt first-use suite passed 16/16 main SQL queries and 4/4 chart checks, with 16/16 cached repeats and zero exceptions. Windows and Docker/Linux each passed 114 offline tests. See [the protocol and archived report](HOLDOUT_EVALUATION.md). The development results below remain historical evidence.

# SQL projection fix and final verification — 18 October 2026

Generation and repair prompts distinguish scalar extrema (one metric), ranked record lists (identifier plus ranking metric), and ordinary filtered lists (identifier unless other fields are requested). Fields used only for filtering or ordering are not automatically selected. These are general prompt rules, not case-specific result rewrites.

## Final live run

- Groq model: `openai/gpt-oss-20b`.
- All 30 original questions, references and expected outputs were unchanged.
- Strict execution match: **30/30**, first attempt **30/30**, provider errors **0**.
- Previously failing q11/q12 returned `id, amount`; q30 returned `id` only.
- Cache repeats: **30/30** hits.
- Median cold latency: **1,231.716 ms**; P95 **2,541.400 ms**; median cached latency **2.404 ms**.
- Prompt SHA-256: `a264ae47da52f728caa78d5e0c741b07875db973b793141b9a960a7348c1b7b9`.
- No repairs triggered; live repair success rate is unavailable.

## Charts and checks

Four live charts passed on synthetic and Olist data: status bar counts and monthly line totals. Actual SQL rows and renderer inputs matched independent references. All four PNGs were visually inspected: descriptive titles and axes, non-repeating monthly ticks, no inferred currency. Cached repeats preserved images without additional model calls.

Windows: **96 tests passed**. Docker/Linux: **96 tests passed**. Ruff passed on both. Docker image rebuilt successfully.

## Limits and history

The first full projection run scored 29/30: a scalar maximum question was treated as a ranked record request. After clarifying that distinction, the final full run scored 30/30. Earlier targeted runs revealed sensitivity in filtered versus implicit ranking requests. Historical reports are retained locally. This small synthetic benchmark does not guarantee accuracy on unseen questions or all narrative claims. Prompt guidance does not enforce a formal runtime projection contract.

No commits or pushes were made.

---

## Historical verification notes

# Verification status

Verified on 18 October 2026. Python 3.12; Groq model `openai/gpt-oss-20b`.

| Check | Result |
| --- | --- |
| Windows offline regressions | 87 passed |
| Earlier Linux container regression check | 86 passed before the final reporting-only test was added |
| Ruff / pip check | Passed |
| Synthetic reference SQL | 30/30 reference queries validated |
| Full live suite, revised prompt | 27/30 strict matches; no provider errors; 30/30 cached repeats |
| Live chart validation | 4/4 numeric, renderer, PNG and cached-image checks passed |
| Live fraction check | Correct after floating-point division instruction |
| Olist ingestion/query | Nine tables, foreign-key check passed; 99,441 orders |
| Docker/Compose | Build/startup, non-root process, read-only database and health/readiness verified; image rebuilt after prompt update |
| Hosted LangSmith delivery | Not verified |
| GitHub-hosted CI | Prepared but not run remotely; changes not pushed |

## Full live run

The 30 questions and references were unchanged, with a fresh isolated database/cache and 12-second pacing. Prompt SHA-256: `e499b42f9c843a51f2ff271c43e14b11c9cd7e753b81bbcacfaecbb52c75a54b`.

Median cold graph latency: 1,396.870 ms; P95: 4,682.343 ms; median cache latency: 2.833 ms. These are synthetic local measurements, not production guarantees. No repairs were triggered, so live repair success rate is unavailable.

Strict mismatches q11/q12/q30 returned the correct records with extra columns. Manual projection matched the reference, but the score remains 90%. The questions' required projections should be clarified before a future benchmark version.

Manual answer review found unsupported currency symbols and q18 expanding WA to Washington. No automatic narrative factuality score is claimed.

## Charts

Both bar and line scenarios were checked on synthetic and Olist data. Reference rows matched the actual plotting query and x/y arrays passed into Matplotlib. PNGs were inspected and cache repeats preserved the image without LLM calls. Labels/titles remain generic, and the month axis is categorical with missing months omitted. Visual presentation can be improved without changing the verified numeric results.

## Running checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
python -m evals.run --mode offline --output reports/offline.json
python -m evals.run --mode live --delay-seconds 12 --output reports/live.json
```

Live mode requires valid local Groq credentials and can incur usage charges. Reports are ignored by Git. The account no longer exposed the old llama model, so the template/default was changed to the tested Groq-hosted model above.

## Boundaries

No public deployment, authentication, user/tenant permissions, rate limiter or billing controls are implemented. Trace delivery is optional and unverified. Numeric SQL/chart checks do not establish that every narrative statement is grounded. No commits or pushes were made during this verification.

## Subsequent grounding/presentation update

Local code now preserves literal filter codes without LLM geographic expansion, filters unsupported common currency markers, derives descriptive chart labels and uses a true monthly calendar axis without repeated month labels. Windows regressions: 96 passed. Eleven targeted live code/currency checks passed with no provider errors; their strict SQL score remained 9/11 due to two existing extra-column mismatches. Four live chart numeric/renderer/cache checks passed. Final images were rerendered from those live rows after the monthly tick correction and visually inspected. This targeted run does not replace the earlier full 30-case score.
