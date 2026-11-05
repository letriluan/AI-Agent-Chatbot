# Engineering Notes

## Methods and Architecture

**Design choices:** LangGraph makes the retry path explicit; SQLite keeps the demo and reference evaluations self-contained; FastAPI separates the backend from the browser UI. Groq is the configured LLM provider, with model selection kept in environment settings. These choices favour a small, inspectable application over additional gateway, vector-database or Redis services that this workflow does not require.

```mermaid
flowchart TD
    A[Browser question] --> B[FastAPI validation]
    B --> C{Answer cache}
    C -->|Hit| D[Cached answer and chart]
    C -->|Miss| E[Schema and query generation]
    E --> F[SQL checks and read-only execution]
    F -->|Retryable result| G[Repair using feedback]
    G --> F
    F -->|Success| H[Answer and optional chart]
    H --> I[Cache successful response]
    F -->|Attempts exhausted| J[Controlled failure response]
```

### Schema and repair

SQLite metadata supplies the table, column and relationship context. The main LangGraph path permits up to three query execution attempts, with previous SQL and execution feedback supplied to repair attempts. After exhaustion, a deterministic failure response is returned rather than asking the model to invent an answer from failed execution.

SQL prompts distinguish between scalar questions such as a maximum amount, ranked lists that need an identifier and ranking metric, and filtered lists that only need the requested identifiers. This reduces unnecessary output columns. Fractions use floating-point division to avoid SQLite integer truncation. These are prompt instructions, not a formal runtime guarantee of semantic correctness.

Code-like values following a related entity (for example `stop S4`) receive bounded, parameterised lookups against single-column declared primary keys. Confirmed matches add only the supplied value and its matching table/column to schema context; no names or sample rows are exported. Probes exclude recognised personal columns and are capped at four codes, twelve key lookups and 100 ms of SQLite work (connection/lock waits retain their existing one-second bound). Exact values are verified without overriding the key's native index collation. This is optional context, not an SQL rewrite or a complete entity resolver. Explicit name filters are kept as names.

Empty-result retry triggers use whole words, so `stop` does not match `top`. Repair instructions preserve explicit identifiers and filters; an empty result is not permission to drop a code, widen dates or replace exact equality with a wildcard. Prompt hashes invalidate earlier query/answer cache entries when these instructions change.

### Write-intent refusal

Common English and Vietnamese requests to change database records or structure are refused deterministically before graph construction, cache lookup, model calls or database access. The response has `ok=false`, empty SQL, no image and zero model calls. Requests are not silently translated into SELECT queries. This conservative phrase recogniser is not a complete natural-language policy engine; the database executor independently enforces read-only access. Regression cases include valid analytics and chart-editing requests to check false positives.

### Database controls

The agent opens an existing database with `mode=ro` and `query_only`. An authorizer rejects write operations, internal SQLite table reads and access to recognised personal-data columns, including attempts to alias those columns. A progress handler interrupts expensive queries; an outer `LIMIT` bounds returned rows even if the generated SQL contains its own limit.

Text-based checks and column-name privacy rules are intentionally conservative and incomplete. They do not replace a reviewed data-access policy or tenant-specific permissions.

### Caching and charts

Three cache levels store answers, question-to-SQL mappings and query results with expiry and database fingerprints. Question keys preserve punctuation and Unicode to avoid conflating inputs such as `1.5` and `15`. Prompt changes and the cache version are included in query/answer cache identity.

Successful answers can reuse their chart. When only a query mapping is available, plotting requests rebuild the visualisation. Failed query results are not persisted as successful answers. Matplotlib rendering is serialized within each process to avoid concurrent global plotting state.

### Grounded presentation

Coded filters supplied by the user (such as `WA`) are summarised directly from query results without expanding them into geographic names. Ordinary generated answers receive a conservative filter for common currency markers when no currency is supplied by the question or result data. This does not infer units, convert currencies or replace a complete grounding evaluation.

Charts use descriptive result-column labels and deterministic titles. Monthly series use a calendar axis with distinct month ticks; missing months are not silently zero-filled. Count axes use integer ticks, and large values use separators without inventing currency units.

### Observability

Responses include request IDs, cache source, attempts and metrics for total latency, graph-stage timings, LLM calls and token usage when returned by the provider. `token_usage_complete` is false if any successful model call omits input/output counts. A zero-call cache hit has complete zero usage. Structured local logs omit questions, SQL and result rows.

### Connection recovery

`LLM_CONNECTION_RETRIES` defaults to 2 and accepts 0–2. This is a shared budget of extra attempts across a whole question, rather than a new budget for each graph node. Only Groq connection errors (including SDK timeout errors) are retried, with one- then two-second backoff. SDK retries are disabled in all modes to avoid stacking policies. HTTP errors, including access/billing errors and HTTP 429, are not retried; free-only evaluations still stop at provider limits. [Official SDK error and retry behaviour](https://github.com/groq/groq-python#retries).

Each attempt increments `llm_calls`; failures, connection retries and backoff are recorded separately. A transport failure makes token usage incomplete even if a retry succeeds, since a lost response may hide provider work. On exhaustion, partial safe metrics survive on the exception and in the evaluation record. Exception messages, provider headers and request contents stay out of logs. The HTTP API returns a sanitised 503 with a correlation ID and `Retry-After: 5` for an exhausted connection failure. Per-call timeouts remain 20 seconds in demo mode and 30 seconds locally; at most two extra calls and three seconds of backoff are added to a question.

Optional LangSmith tracing uses LangGraph's existing tracing integration. Trace inputs/outputs are hidden by default. No remote dashboard or cost estimate is fabricated; hosted trace delivery requires valid credentials and explicit enablement.

## Optional Tracing

Set the following locally to enable LangSmith:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=ai-database-agent
LANGSMITH_HIDE_INPUTS=true
LANGSMITH_HIDE_OUTPUTS=true
```

Token counts are reported when the provider supplies them. Evaluation reports can estimate standard-list-price USD cost with the dated model-specific configuration in `evals/cost.py`. Missing usage and unknown models have no estimate. Requests that raise before usage is returned prevent complete cost coverage; cached-input discounts, free-tier credits and actual bills are not inferred. [Pricing source](https://console.groq.com/docs/model/openai/gpt-oss-20b), checked 5 November 2026.

