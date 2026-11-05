# Repeated-run evaluation

Mode: live; suite: holdout; completed runs: 5/5.

SQL matches: 77/80 observations.
A case passes when both its SQL and any required chart checks pass. Numeric support is scored separately.

| Case | Passed runs | All planned runs passed | Numeric-supported runs |
| --- | --- | --- | --- |
| h01 | 5/5 | True | 5 |
| h02 | 5/5 | True | 5 |
| h03 | 5/5 | True | 5 |
| h04 | 5/5 | True | 5 |
| h05 | 5/5 | True | 5 |
| h06 | 5/5 | True | 5 |
| h07 | 5/5 | True | 5 |
| h08 | 5/5 | True | 5 |
| h09 | 5/5 | True | 0 |
| h10 | 5/5 | True | 5 |
| h11 | 5/5 | True | 5 |
| h12 | 4/5 | False | 0 |
| h13 | 5/5 | True | 5 |
| h14 | 5/5 | True | 5 |
| h15 | 4/5 | False | 3 |
| h16 | 4/5 | False | 4 |

Mixed outcomes: h12, h15, h16.
Numeric support: 67 supported, 6 unsupported, 7 unscored.
Numeric support does not verify labels, units, completeness, or which row a number describes.
Known standard-list-price estimate (USD): 0.019320525; coverage 77/80.
This is not a measured bill. Failures without token usage prevent a complete per-question estimate.

Reports use synthetic data and fresh caches. The original holdout is an exposed regression suite, not unseen data.

## Scope and evidence

Completed 5 November 2026 using the user-confirmed free-only Groq organisation. Every fresh observation used an isolated local cache; cached repeats were separate verification requests. Prompts, implementation, fixtures, model ID and recorded configuration stayed fixed.

Known reported model calls: 172; known reported tokens: 136,728. Failed requests may omit usage. Cached repeats with zero model calls: 77/80. Provider exceptions: 3.

These 16 questions were first exposed in October and are a regression suite. Five repeats are not 80 unique unseen questions and do not establish accuracy on arbitrary schemas. The separate first-use transit report includes its own observed failure.

All chart PNGs are covered by visual inspection of their exact image hashes; unchanged identical images reuse the existing inspection. Plot-query rows and actual renderer arrays are also checked automatically. The numerical-support check is separate from SQL correctness and complete narrative semantics.

Raw evidence is preserved under run-01 through run-05. Source/configuration fingerprints, per-case outcomes and cost coverage are in [summary.json](summary.json). Image coverage and raw numeric flags are in [review.json](review.json).

## Connection failures and numeric review

Run 4: h12 and h15; run 5: h16. All three failures were APIConnectionError, with no token usage returned to the evaluation report. Their underlying network cause was not established. Every returned SQL result matched the independent reference; the overall success count remains 77/80, and 13/16 cases passed all five runs (81.25%). Chart cases succeeded in 18/20 observations; the other two had connection failures.

The six automatic numeric-support flags were reviewed: five repeat the h09 date filter, and one says three bars for the three h15 result rows. These are context/result-shape references rather than invented values. Seven observations are unscored, including three failed requests and four successful empty answers. This does not establish full narrative factuality.

Cost coverage is 77/80. The known standard-list-price subtotal is USD 0.019320525. A complete mean for all 80 questions is unavailable because the three failed requests lack usage. These are list-price estimates, not measured charges.
