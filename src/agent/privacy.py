from __future__ import annotations

"""
src/agent/privacy.py

Privacy helpers for result + text redaction.

Goal:
- Protect personal information and personal names
- Do NOT redact business/entity names (e.g., franchise_name, supplier_name)

Approach:
- Redact columns that clearly look like personal name fields
- Redact columns that clearly look like Personally Identifiable Information (PII) (email/phone/address/card)
- Apply a small text-level safety net on the final narrative
"""

import re
from typing import Any, Dict, List


# Strong signals for personal-name columns
PERSON_NAME_EXACT = {
    "first_name",
    "last_name",
    "full_name",
    "customer_name",
    "person_name",
    "contact_name",
}

# If a column contains one of these AND "name", treat it as personal
PERSON_NAME_CONTEXT_WORDS = {
    "customer",
    "person",
    "contact",
    "employee",
    "staff",
    "user",
    "recipient",
}

# Personal columns/keywords (non-name)
PII_KEYS = {
    "email_address",
    "email",
    "phone_number",
    "phone",
    "address",
    "cardnumber",
    "card_number",
}

PII_SUBSTRINGS = ["email", "phone", "address", "card"]


def _is_person_name_column(col: str) -> bool:
    """True only when the column strongly indicates an individual's name."""
    cl = (col or "").strip().lower()

    if cl in PERSON_NAME_EXACT:
        return True

    if "name" in cl and any(w in cl for w in PERSON_NAME_CONTEXT_WORDS):
        return True

    # Generic "name" is ambiguous → do NOT redact by default
    return False


def _is_pii_column(col: str) -> bool:
    """Detect PII columns like email/phone/address/card."""
    cl = (col or "").strip().lower()
    if cl in PII_KEYS:
        return True
    return any(s in cl for s in PII_SUBSTRINGS)


def redact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact only personal information:
    - Personal name columns (when clearly personal)
    - PII columns (email/phone/address/card)
    """
    if not result.get("ok"):
        return result

    cols: List[str] = result.get("columns", []) or []
    rows = result.get("rows", []) or []

    name_idx = {i for i, c in enumerate(cols) if _is_person_name_column(c)}
    pii_idx = {i for i, c in enumerate(cols) if _is_pii_column(c)}

    if not name_idx and not pii_idx:
        return result

    redacted_rows = []
    for row in rows:
        row2 = list(row)

        for i in name_idx:
            if i < len(row2) and row2[i] is not None:
                row2[i] = "[REDACTED_NAME]"

        for i in pii_idx:
            if i < len(row2) and row2[i] is not None:
                row2[i] = "[REDACTED_PII]"

        redacted_rows.append(row2)

    return {**result, "rows": redacted_rows}


# =========================
# Text safety net (personal only)
# =========================
_NAME_CUE_PATTERNS = [
    re.compile(r"(?i)\b(customer\s+name\s*:\s*)(.+)"),
    re.compile(r"(?i)\b(contact\s+name\s*:\s*)(.+)"),
    re.compile(r"(?i)\b(employee\s+name\s*:\s*)(.+)"),
]


def redact_text(text: str) -> str:
    """
    Redact personal names only when the text explicitly labels them as a name field.
    Avoid redacting normal proper nouns like franchise/product names.
    """
    if not text:
        return text

    out = text
    for pat in _NAME_CUE_PATTERNS:
        out = pat.sub(r"\1[REDACTED_NAME]", out)
    return out
