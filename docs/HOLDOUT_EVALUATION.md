# First-use evaluation on new questions

Verified on 18 October 2026 with Groq `openai/gpt-oss-20b`.

## Protocol

The original 30-case suite was used during prompt development. This separate suite contains 16 newly authored questions: 12 SQL-only requests and four chart requests. Its separate synthetic database includes a one-to-many item relationship, tied amounts, zero amounts, a customer without orders, a valid empty answer and a missing calendar month. The schema remains in the same orders/customers domain: this is not cross-domain evaluation.

All reference queries and manually specified expected rows were validated offline before the first model call. The application prompts were frozen at SHA-256 `a264ae47da52f728caa78d5e0c741b07875db973b793141b9a960a7348c1b7b9`, identical to the preceding development run. Neither application code nor prompts were tuned using these new live outputs. Inputs were fingerprinted and checked before every case. Each case used a fresh cache; a second call checked reuse. The live run used 15-second pacing and optional tracing was disabled.

## First-use results

| Measure | Result |
| --- | --- |
| SQL-only requests | 12/12 strict matches |
| Main SQL across all requests, including charts | 16/16 strict matches |
| Chart datasets and renderer inputs | 4/4 matched independent references |
| Chart PNG, labels, type and cached-image checks | 4/4 passed |
| Execution/evaluator exceptions | 0 |
| Answer-cache repeats | 16/16 hits |
| Offline regressions | 114 passed on Windows; 114 passed on Docker/Linux |
| Ruff | Passed on both environments |

Strict matching includes row width, values, duplicate multiplicity and specified order, with numeric comparison rounded to six decimal places. Chart checks observe the actual query result and actual x/y arrays passed into the real Matplotlib renderer; they do not substitute reference data into the application. Four PNGs were visually inspected. The missing March has no artificial zero point; connecting lines interpolate between available points. Narrative factuality and semantic quality of every axis label are not automatically scored.

The frozen [first-use report](evaluations/holdout-first-use-2026-10-18.json) includes generated SQL, synthetic rows, answers, chart checks, metrics and fingerprints. No credentials or real customer data are included.

## Reproduce

From the repository root, after installing development dependencies:

```bash
python -m evals.holdout --mode offline --output reports/holdout-offline
python -m evals.holdout --mode live --delay-seconds 15 --output reports/holdout-rerun
```

Live mode uses local Groq credentials and incurs provider usage. Offline mode validates references only and does not measure model quality. CI runs the offline mode without credentials. Reports and chart PNGs are saved incrementally in the selected output folder.

## Interpretation

This is a small handcrafted first-use evaluation, not a blinded external benchmark or proof of production readiness. Several skills overlap the development suite. The new cases were not used to revise prompts before or after this first run. Once exposed, they are regression cases; future prompt changes should be assessed on another fresh, frozen set. Do not combine these 16 cases with the development 30 and describe all 46 as unseen. A single successful run does not measure reliability across repeated model calls or model versions.

No commits or pushes were made.
