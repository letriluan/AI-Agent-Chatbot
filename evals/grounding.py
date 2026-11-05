"""Conservative numeric support checks, not a semantic factuality judge."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import argparse
import hashlib
import json
from pathlib import Path
import re


# Keep dates/times whole: matching the year alone must not validate a wrong date.
TOKEN = re.compile(
    r"(?<![\w.])(?:\d{4}-\d{2}(?:-\d{2})?|\d{1,3}:\d{2}(?::\d{2})?|"
    r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?)(?!\w|\.\d)"
)
WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
WORDS.update(dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))))
WORD_NUMBER = re.compile(r"\b(?:" + "|".join(WORDS) + r")\b", re.I)


def claims(text):
    text = str(text or '').translate(str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2014\u2212'}))
    # Markdown list indices are formatting, not claims about the data.
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    found = [m.group() for m in TOKEN.finditer(text)]
    # Single-word numbers are supported; compound word numbers need human review.
    found.extend(m.group().lower() for m in WORD_NUMBER.finditer(text))
    return found


def number(token):
    try:
        if token.lower() in WORDS:
            return Decimal(WORDS[token.lower()])
        return Decimal(token.replace(",", "").replace("−", "-").rstrip("%"))
    except InvalidOperation:
        return None


def rounded_equal(candidate, precision, value):
    try:
        return candidate is not None and candidate.quantize(precision, rounding=ROUND_HALF_UP) == value
    except InvalidOperation:
        return False


def check_numeric_support(answer, rows):
    """Check literal numbers against executed, redacted result cells only.

    No question text, reference answer, SQL constants or chart data can excuse
    an unsupported number. Correct SQL and complete answer semantics are scored
    separately. No-number answers are unscored, never counted as successes.
    """
    tokens = claims(answer)
    evidence = []
    for row in rows:
        for value in row:
            if value is not None and not isinstance(value, bool):
                evidence.extend(claims(str(value)))
    unsupported = []
    for token in tokens:
        value = number(token)
        supported = token in evidence
        if value is not None:
            supported = any(number(item) == value for item in evidence)
            # Permit displayed decimal rounding, but never round to an integer.
            plain = token.replace(",", "").rstrip("%")
            if not supported and "." in plain and "e" not in plain.lower():
                precision = Decimal(1).scaleb(-len(plain.split(".")[1]))
                supported = any(
                    rounded_equal(candidate, precision, value)
                    for item in evidence if (candidate := number(item)) is not None
                )
        if not supported:
            unsupported.append(token)
    # Compound word numbers cannot be verified by matching individual words.
    compound = bool(re.search(
        r"\b(?:hundred|thousand|million|billion)\b|\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[ -]+(?:one|two|three|four|five|six|seven|eight|nine)\b",
        str(answer or ""), re.I,
    ))
    status = "unsupported" if unsupported else "needs_review" if compound else "supported" if tokens else "no_numeric_claims"
    return dict(status=status, claims=tokens, unsupported=unsupported,
                passed=status == "supported", semantic_factuality_measured=False)


def review_report(path):
    path = Path(path)
    original = json.loads(path.read_text(encoding='utf-8'))
    if original.get('mode') != 'live':
        raise ValueError('Numeric support needs recorded live answers and actual SQL rows')
    records = []
    for record in original['records']:
        check = (check_numeric_support(record.get('answer'), record['actual_rows'])
                 if isinstance(record.get('actual_rows'), list) and isinstance(record.get('answer'), str)
                 else dict(status='unavailable', passed=False))
        records.append(dict(id=record['id'], **check))
    return dict(mode='retrospective_numeric_support', source_report=path.name,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                new_model_calls=0, semantic_factuality_measured=False, records=records,
                summary=dict(total=len(records), supported=sum(r['status'] == 'supported' for r in records),
                             unsupported=sum(r['status'] == 'unsupported' for r in records),
                             unscored=sum(r['status'] not in {'supported', 'unsupported'} for r in records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='An existing live results.json file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Choose a new output file to preserve previous evidence')
    report = review_report(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
