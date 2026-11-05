# Fixed-version rerun: stopped at free-tier quota

The five-run collection started on 5 November 2026 using the fixed identifier/retry implementation, fresh per-case caches, frozen sources, `openai/gpt-oss-20b`, SDK retries disabled and a request-wide limit of two connection retries. The user restricted all usage to the same free-only Groq organisation as Render.

Run 1 completed at **16/16 SQL and 4/4 chart checks**. Run 2 passed six cases, then h07 received **HTTP 429** during answer generation. The runner stopped immediately and did not start transit, retry HTTP errors, change accounts/models or upgrade billing. In total, **22/23 saved observations passed**, including the quota failure in the denominator. This is **not a completed five-run reliability study**.

One naturally observed connection failure on run 1 h09 recovered with one application retry. Its token usage remains incomplete. The one numeric-support flag repeats the question's date boundaries; IDs 5, 6 and 7 match SQL. Four charts were visually inspected. The known list-price subtotal covers 21/23 observations; missing usage is unknown, not zero. See `review.json` for the dated subtotal and checksums.

The Groq console showed Free as the current plan and 200,000 daily / 8,000 minute tokens for this model. The exact quota window responsible for this 429 and its reset time were not verified. No full fixed-version transit score is available; the earlier full 23/24 baseline and targeted corrected-case follow-up remain separate evidence.

The task-local resume directory remains `reports/fixed-reliability-live-2026-11-05`. Resume only after the free allowance is available, with the same frozen sources and runtime configuration. Saved observations, including the quota failure, must be preserved; use a separate output directory for any additional replacement trial. Once that batch is complete, collect the 24-case transit suite in a new output directory.
