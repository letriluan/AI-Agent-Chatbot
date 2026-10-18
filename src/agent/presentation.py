"""Deterministic display labels and conservative answer formatting."""
import re


def human_label(name):
    return " ".join(str(name).replace("_", " ").split()).capitalize()


def chart_labels(columns, rows, question):
    x, y = columns[:2]
    q = question.lower()
    if x.lower() in {"x", "category", "label"}:
        values = [str(row[0]) for row in rows]
        if values and all(re.fullmatch(r"\d{4}-\d{2}", v) for v in values):
            x = "month"
        elif "status" in q:
            x = "order_status" if "order" in q else "status"
        elif "state" in q:
            x = "state"
        else:
            x = "category"
    if y.lower() in {"y", "value", "metric"}:
        if "count" in q or "number of" in q:
            y = "order_count" if "order" in q else "count"
        elif "price" in q:
            y = "total_item_price" if "item" in q else "total_price"
        elif "amount" in q:
            y = "average_amount" if "average" in q else "total_amount"
        else:
            y = "value"
    xlabel, ylabel = human_label(x), human_label(y)
    return f"{ylabel} by {xlabel.lower()}", xlabel, ylabel


def literal_codes(question, sql):
    """Only codes explicitly supplied by the user AND used as SQL literal values."""
    return [code for code in dict.fromkeys(re.findall(r"\b[A-Z]{2,5}\b", question))
            if re.search(r"['\"]" + re.escape(code) + r"['\"]", sql)]


def coded_result_answer(question, sql, result):
    codes = literal_codes(question, sql)
    if not codes or not result.get("rows"):
        return None
    columns = result.get("columns", [])
    lines = ["Results for " + ", ".join(codes) + ":"]
    for row in result["rows"]:
        lines.append("; ".join(f"{human_label(col)}: {value}" for col, value in zip(columns, row)))
    return "\n".join(lines)


def remove_unsupported_currency(text, question, result):
    evidence = question + " " + str(result.get("columns", [])).replace("_", " ") + " " + str(result.get("rows", []))
    # Formatting cannot supply an absent currency or imply a conversion.
    if not re.search(r"[$€£¥₹]|\b(?:USD|AUD|BRL|EUR|GBP|JPY|INR|dollars?|euros?|pounds?|yen|reais)\b", evidence, re.I):
        text = re.sub(r"[$€£¥₹]", "", text)
        text = re.sub(r"\b(?:USD|AUD|BRL|EUR|GBP|JPY|INR|dollars?|euros?|pounds?|yen|reais)\b", "", text, flags=re.I)
    return text
