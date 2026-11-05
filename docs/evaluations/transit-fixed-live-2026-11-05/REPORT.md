# Fixed-version full synthetic transit evaluation

**24/24 strict SQL matches; 4/4 chart checks; 24/24 cached repeats with zero model calls.** All 24 observations completed without a request-level error or quota restriction.

The original first-use baseline remains unchanged at **23/24**. Its stop-code error was fixed before the separately archived targeted follow-up and this complete regression rerun. The previously failing t15 now filters `fact_stop_events.stop_key = 'S4'`, returns **R2** and answers **R2** on the **first SQL attempt**.

## Method and scope

Collection ran **06:10:09–06:22:44 UTC on 5 November 2026** using the same frozen identifier/retry version as the final five-run study and Render commit `5df7de9`. Model: `openai/gpt-oss-20b`; temperature 0; demo mode; 1,024 output-token cap; five-second SQL bound; SDK retries disabled; at most two extra connection attempts shared across one question; 30-second pacing. Every case had a new temporary cache, followed by a cached repeat. Source, prompt, fixture and evaluation fingerprints matched; no tuning occurred during collection.

The same separately supplied user Free-account key as the final holdout study was used. No paid fallback or billing upgrade was attempted. The earlier quota-restricted attempt and original first-use studies are separate evidence, not pooled into this score. The console showed Free as Current Plan before the new study; the key's organisation was not independently identified through the API.

This fixture has an invented five-table star schema, 14 stop events, six trips, four routes and six stops. These 24 questions are now exposed regression cases. This is not Adelaide Metro data, a complete GTFS feed or a general unseen-schema accuracy claim. The eight earlier identifier/name ambiguity cases remain separate evidence.

## Recovery, answer numbers and charts

One naturally observed connection/timeout failure on **t23** recovered with **one application retry**. The failed attempt stays in metrics and its full token usage remains unknown. No fault was injected in this batch; the underlying network cause was not established.

Numeric support: **16 supported, one flagged, seven unscored**; **16/17 scored answers (94.1%)** were automatically supported. The t09 flag is **24:00**, the service-day threshold supplied by the question, while the returned trip **T6** matches SQL. It was manually reviewed as question context, and the raw flag remains unchanged. Full semantic factuality is not measured.

All four PNGs were visually inspected. SQL plot rows, renderer inputs, labels, PNG validity/nonblank content and cached-image preservation passed independently. Dates and service hours use equally spaced categories; gaps such as service hour 10 to 25 are not elapsed-time spacing.

## Dated list-price estimate

Known standard-list-price subtotal: **USD 0.005380875**, covering **23/24** complete-usage questions. The recovered t23 request has incomplete usage; its full cost is unknown, not zero, so no full-suite mean or actual bill is claimed. Per-question estimates are retained in `results.json`. Rates checked 5 November: USD 0.075/M input and 0.30/M output tokens for [Groq gpt-oss-20b](https://console.groq.com/docs/model/openai/gpt-oss-20b), without free credits or caching discounts.

`results.json` preserves all observations. `review.json`, `visual-review.json` and `manifest.json` retain the source check, manual flag review, image hashes and checksums.
