# Fixed-version five-run live evaluation

**5/5 complete runs; 80/80 observations passed; all 16 cases passed every run (100%).** All **20/20 chart checks** passed and all **80/80 cached repeats** used zero model calls. No request-level provider errors or mixed case outcomes remained in this collection.

| Run | SQL matches | Chart checks | Cache repeats |
| --- | --- | --- | --- |
| 1 | 16/16 | 4/4 | 16/16 |
| 2 | 16/16 | 4/4 | 16/16 |
| 3 | 16/16 | 4/4 | 16/16 |
| 4 | 16/16 | 4/4 | 16/16 |
| 5 | 16/16 | 4/4 | 16/16 |

## Method and version

Collection ran from **05:28:05 to 06:10:05 UTC on 5 November 2026** on the identifier/retry implementation deployed as commit `5df7de9`. Every case used a new temporary cache and the frozen synthetic holdout fixture. Source, prompt, fixture and evaluation fingerprints matched throughout; no tuning occurred during collection. Configuration: `openai/gpt-oss-20b`, temperature 0, demo mode, 1,024 output-token cap, five-second SQL bound, SDK retries disabled, at most two extra connection attempts shared across one question, and 30-second pacing between questions.

The user supplied a separate Free-account local key after the initial organisation hit quota. Free was displayed as Current Plan in the console before this batch. The key's organisation was not independently identified through the API. This complete study is separate from the earlier incomplete 22/23 attempt and original 77/80 baseline; no observations were pooled across those studies. No paid fallback or billing upgrade was used. Render's existing credential was retained.

This is an exposed 16-case regression suite, not unseen questions or an external benchmark. The result describes these observations; it is not a general reliability guarantee or a controlled causal comparison with the earlier network conditions.

## Connection recovery

Four user-question requests naturally encountered connection/timeout failures and recovered: run 1 h06 (one retry), run 2 h06 (one), run 4 h01 (two), and run 4 h08 (one). All five failed transport attempts remain counted in metrics. None was an injected fault in this batch. The underlying network cause was not established. These were bounded application retries inside a measured request, not manual replacement trials or retries of HTTP quota errors.

## Numeric support and charts

Automatic numeric checking returned **71 supported, four flagged and five unscored observations**: **71/75 scored answers (94.7%)** were supported without manual interpretation. All four flags are h09's repetitions of the question's April 2026 / April 30 / May 1 date boundaries; result IDs 5, 6 and 7 match SQL. They were manually reviewed as question context, while the raw flags remain unchanged. No-number answers are unscored, and full semantic factuality is not measured.

Chart checks independently compare SQL plot rows, renderer arrays, labels, PNG validity/nonblank content and cached-image preservation. All 20 PNGs are covered by direct visual inspection of their exact hashes; see `visual-review.json`. Month labels are observed categories with equal spacing, not elapsed-time axes.

## Dated list-price estimate

Known standard-list-price subtotal: **USD 0.019544175**, covering **76/80 complete-usage observations**. Four recovered requests have incomplete usage; their full cost is unknown, not zero. No full per-question mean or actual bill is claimed. Each available per-question estimate remains in its raw record. Rates checked 5 November: USD 0.075/M input and 0.30/M output tokens for [Groq gpt-oss-20b](https://console.groq.com/docs/model/openai/gpt-oss-20b), without free credits or caching discounts.

Raw `run-XX/results.json`, `summary.json` and the original generated `RUNNER_REPORT.md` are preserved. `review.json` records the manual flag review, recovery counts and checksums of earlier evidence.
