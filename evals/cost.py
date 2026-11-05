"""Dated list-price estimates from reported token usage; never billing claims."""
from decimal import Decimal


PRICING = {
    "model": "openai/gpt-oss-20b",
    "currency": "USD",
    "input_per_million": "0.075",
    "output_per_million": "0.30",
    "checked_on": "2026-11-05",
    "source": "https://console.groq.com/docs/model/openai/gpt-oss-20b",
}


def estimate(metrics, model):
    if model != PRICING["model"]:
        return dict(available=False, reason="No verified rate for this model")
    calls = metrics.get("llm_calls")
    if calls is None or calls < 0 or metrics.get("token_usage_complete") is not True:
        return dict(available=False, reason="Complete provider token usage was not recorded")
    counts = [metrics.get(key) for key in ("input_tokens", "output_tokens")]
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
        return dict(available=False, reason="Invalid token usage")
    if calls and sum(counts) == 0:
        return dict(available=False, reason="Provider usage is missing")
    amount = (Decimal(counts[0]) * Decimal(PRICING["input_per_million"])
              + Decimal(counts[1]) * Decimal(PRICING["output_per_million"])) / Decimal(1_000_000)
    return dict(available=True, estimated_usd=str(amount), pricing=PRICING,
                basis="Standard list price; no provider cache/batch discounts or free-tier credits",
                actual_bill_measured=False)
