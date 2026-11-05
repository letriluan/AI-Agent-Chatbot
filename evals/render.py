"""Validate the public Olist demo through HTTP, without a local provider key.

This is a deployment check. Public cache hits are retained and are never counted
as independent fresh-model trials; this endpoint cannot load synthetic fixtures.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import time

import httpx
from PIL import Image, ImageStat

from .availability import sample
from .grounding import check_numeric_support
from .run import equivalent
from src.agent.privacy import redact_result
from src.agent.sql_tools import run_sql


REFERENCES = [
    ('orders', 'SELECT COUNT(*) FROM olist_orders_dataset', False),
    ('statuses', 'SELECT order_status,COUNT(*) FROM olist_orders_dataset GROUP BY order_status', True),
    ('delivered_join', "SELECT COUNT(DISTINCT o.order_id) FROM olist_orders_dataset o JOIN olist_order_items_dataset i ON i.order_id=o.order_id WHERE o.order_status='delivered'", False),
    ('monthly_items', "SELECT strftime('%Y-%m',o.order_purchase_timestamp),SUM(i.price) FROM olist_orders_dataset o JOIN olist_order_items_dataset i ON i.order_id=o.order_id GROUP BY 1 ORDER BY 1", True),
    ('write_refusal', None, False),
    ('customer_states', 'SELECT customer_state,COUNT(*) n FROM olist_customers_dataset GROUP BY customer_state ORDER BY n DESC,customer_state LIMIT 5', False),
    ('delivered_percentage', "SELECT 100.0*SUM(CASE WHEN order_status='delivered' THEN 1 ELSE 0 END)/COUNT(*) FROM olist_orders_dataset", False),
    ('payment_types', 'SELECT payment_type,COUNT(*) FROM olist_order_payments_dataset GROUP BY payment_type', True),
    ('product_categories', 'SELECT p.product_category_name,COUNT(*) n FROM olist_order_items_dataset i JOIN olist_products_dataset p ON p.product_id=i.product_id WHERE p.product_category_name IS NOT NULL GROUP BY p.product_category_name ORDER BY n DESC,p.product_category_name LIMIT 5', False),
    ('average_price', 'SELECT AVG(price) FROM olist_order_items_dataset', False),
]


class SampleQuestions(HTMLParser):
    def __init__(self):
        super().__init__()
        self.questions = []

    def handle_starttag(self, tag, attrs):
        question = dict(attrs).get('data-question')
        if tag == 'button' and question:
            self.questions.append(question)


def assess_response(item, payload, db, output):
    record = dict(item)
    record.update(ok=payload.get('ok'), sql=payload.get('sql'), answer=payload.get('answer'),
                  cache_hit=payload.get('cache_hit'), cache_source=payload.get('cache_source'),
                  attempts=payload.get('attempts'), metrics=payload.get('metrics', {}),
                  request_id=payload.get('request_id'))
    if item['reference_sql'] is None:
        record['passed'] = (payload.get('ok') is False and payload.get('sql') == ''
                            and payload.get('metrics', {}).get('llm_calls') == 0
                            and 'read-only' in str(payload.get('answer', '')).lower())
        return record
    actual = redact_result(run_sql(str(db), payload.get('sql') or '', max_rows=200))
    record['actual_rows'] = actual.get('rows', [])
    record['sql_correct'] = actual['ok'] and equivalent(record['actual_rows'], item['expected_rows'], item['ordered'])
    record['numeric_support'] = (check_numeric_support(payload.get('answer'), record['actual_rows'])
                                 if actual['ok'] else dict(status='unavailable', passed=False))
    record['passed'] = payload.get('ok') is True and record['sql_correct']
    if item['chart']:
        record['png_valid'] = False
        try:
            png = base64.b64decode(payload.get('image_b64') or '', validate=True)
            with Image.open(io.BytesIO(png)) as image:
                record['png_valid'] = (image.format == 'PNG' and min(image.size) > 100
                                       and max(ImageStat.Stat(image.convert('RGB')).var) > 1)
            if record['png_valid']:
                path = Path(output) / f"{item['id']}.png"
                path.write_bytes(png)
                record.update(chart_file=path.name, chart_sha256=hashlib.sha256(png).hexdigest())
        except (ValueError, OSError):
            pass
        record['passed'] = record['passed'] and record['png_valid']
        record['plot_query_data_verified'] = False
        record['chart_visually_reviewed'] = False
    return record


def run(db, output, *, free_only_confirmed=False, delay_seconds=15):
    if not free_only_confirmed:
        raise ValueError('Confirm the Render deployment uses a free-only Groq account first')
    if delay_seconds < 15:
        raise ValueError('Keep at least 15 seconds between requests; never bypass demo quotas')
    db = Path(db)
    if not db.is_file():
        raise ValueError('Build the local Olist reference database first')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    parser = SampleQuestions()
    parser.feed(Path('src/ui/index.html').read_text(encoding='utf-8'))
    if len(parser.questions) != len(REFERENCES):
        raise ValueError('Review the suite when the UI questions change')
    items = []
    for (case_id, reference, chart), question in zip(REFERENCES, parser.questions):
        expected = run_sql(str(db), reference, max_rows=200) if reference else None
        if expected is not None and not expected['ok']:
            raise ValueError(f'Invalid independent reference: {case_id}')
        items.append(dict(id=case_id, question=question, reference_sql=reference,
                          expected_rows=expected['rows'] if expected else None,
                          chart=chart, ordered=case_id not in {'statuses', 'payment_types'}))
    base_url = 'https://ai-agent-chatbot-demo.onrender.com'
    report = dict(mode='live_render_deployment', started_at=datetime.now(timezone.utc).isoformat(),
                  endpoint=base_url + '/ask', free_only_confirmed=True, source_dataset='Committed Olist CSV files',
                  source_csv_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('data').glob('*.csv'))},
                  deployed_revision=None, deployed_model_identity_verified=False,
                  fresh_model_reliability_measured=False, synthetic_suites_measured=False,
                  semantic_factuality_measured=False, records=[], cache_checks=[], validation_checks=[])

    def save():
        records = report['records']
        report['summary'] = dict(
            expected_cases=len(items), completed_cases=len(records),
            passed=sum(r.get('passed') is True for r in records),
            sql_correct=sum(r.get('sql_correct') is True for r in records),
            sql_cases=sum(r['reference_sql'] is not None for r in records),
            cache_hit_cases=sum(r.get('cache_hit') is True for r in records),
            observed_llm_calls=sum(r.get('metrics', {}).get('llm_calls', 0) for r in records),
            numeric_supported=sum(r.get('numeric_support', {}).get('status') == 'supported' for r in records),
            numeric_unsupported=sum(r.get('numeric_support', {}).get('status') == 'unsupported' for r in records),
            png_checks_passed=sum(r.get('png_valid') is True for r in records),
            cache_checks_passed=sum(c.get('passed') is True for c in report['cache_checks']),
            validation_checks_passed=sum(c.get('passed') is True for c in report['validation_checks']),
        )
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    last_request = None

    def post(client, question):
        nonlocal last_request
        if last_request is not None:
            time.sleep(max(0, delay_seconds - (time.monotonic() - last_request)))
        last_request = time.monotonic()
        start = time.perf_counter()
        response = client.post('/ask', json={'question': question})
        elapsed = round((time.perf_counter() - start) * 1000, 3)
        return response, elapsed

    report['availability'] = sample(base_url)
    save()
    if not report['availability']['available']:
        report['stopped_reason'] = 'Health or readiness check failed'
        save()
        return report
    originals = {}
    with httpx.Client(base_url=base_url, timeout=120, follow_redirects=False) as client:
        page = client.get('/')
        remote = SampleQuestions()
        remote.feed(page.text)
        report['homepage_http_status'] = page.status_code
        report['homepage_sample_questions_match'] = remote.questions == parser.questions
        for item in items:
            response, elapsed = post(client, item['question'])
            if response.status_code != 200:
                report['records'].append(dict(item, passed=False, http_status=response.status_code, elapsed_ms=elapsed))
                report['stopped_reason'] = 'HTTP error; no quota bypass, account upgrade or automatic model retry attempted'
                save()
                return report
            payload = response.json()
            record = assess_response(item, payload, db, output)
            record.update(http_status=response.status_code, elapsed_ms=elapsed)
            report['records'].append(record)
            originals[item['id']] = payload
            save()
            print(json.dumps(dict(id=item['id'], passed=record['passed'], cache_hit=record['cache_hit'],
                                  llm_calls=record['metrics'].get('llm_calls'), elapsed_ms=elapsed)), flush=True)
        for case_id in ['orders', 'statuses', 'monthly_items']:
            item = next(i for i in items if i['id'] == case_id)
            response, elapsed = post(client, item['question'])
            check = dict(id=case_id, http_status=response.status_code, elapsed_ms=elapsed, passed=False)
            if response.status_code == 200:
                payload = response.json()
                original = originals[case_id]
                check.update(cache_hit=payload.get('cache_hit'), metrics=payload.get('metrics', {}))
                check['passed'] = (payload.get('cache_hit') is True and payload.get('metrics', {}).get('llm_calls') == 0
                                   and payload.get('sql') == original.get('sql')
                                   and payload.get('image_b64') == original.get('image_b64'))
            report['cache_checks'].append(check)
            save()
            if response.status_code != 200:
                report['stopped_reason'] = 'Cache verification encountered an HTTP error'
                save()
                return report
        for name, question in [('blank', '   '), ('too_long', 'x' * 2001)]:
            response, elapsed = post(client, question)
            report['validation_checks'].append(dict(id=name, http_status=response.status_code,
                                                     elapsed_ms=elapsed, passed=response.status_code == 422))
            save()
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def successful(report):
    return bool(report.get('completed_at') and not report.get('stopped_reason')
                and report['summary']['completed_cases'] == len(REFERENCES)
                and report['summary']['passed'] == len(REFERENCES)
                and len(report['cache_checks']) == 3 and all(c['passed'] for c in report['cache_checks'])
                and len(report['validation_checks']) == 2 and all(c['passed'] for c in report['validation_checks']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='db/render-validation.sqlite')
    parser.add_argument('--output', required=True)
    parser.add_argument('--free-only-confirmed', action='store_true')
    args = parser.parse_args()
    report = run(args.db, args.output, free_only_confirmed=args.free_only_confirmed)
    print(json.dumps(report['summary'], indent=2))
    if not successful(report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
