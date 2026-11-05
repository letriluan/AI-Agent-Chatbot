"""Frozen first-use evaluation: SQL and actual chart rendering on a separate fixture."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from unittest.mock import patch
from datetime import datetime, timezone

from PIL import Image, ImageStat
from .holdout_fixture import create_database
from .run import configured_graph, equivalent
from .grounding import check_numeric_support
from .cost import estimate
from src.agent.sql_tools import run_sql
from src.agent.privacy import redact_result
from src.agent.observability import connection_retry_limit


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(suite='holdout'):
    if suite not in {'holdout', 'transit', 'identifier'}:
        raise ValueError('Unknown evaluation suite')
    return json.loads(Path(__file__).with_name(f'{suite}_cases.json').read_text(encoding='utf-8'))


def source_fingerprints(suite='holdout'):
    # Include implementation as well as prompts: code changes must not be pooled.
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / 'src' / 'agent').glob('*.py'))
    sources += [Path(__file__), Path(__file__).with_name('run.py'),
                Path(__file__).with_name('grounding.py'), Path(__file__).with_name('cost.py'),
                Path(__file__).with_name(f'{suite}_cases.json'),
                Path(__file__).with_name(f'{suite}_fixture.py')]
    if suite == 'identifier':
        sources.append(Path(__file__).with_name('transit_fixture.py'))
    return {p.name: fingerprint(p) for p in sources}


def runtime_configuration(graph):
    return dict(model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'),
                max_output_tokens=int(os.getenv('LLM_MAX_TOKENS', '1024')),
                demo_mode=os.getenv('DEMO_MODE', 'false').lower() == 'true',
                sql_max_rows=graph.SQL_MAX_ROWS, sql_timeout_seconds=graph.SQL_TIMEOUT_SECONDS,
                temperature=0, connection_retry_limit=connection_retry_limit(), sdk_max_retries=0)


def chart_observers(graph, captured):
    original_build = graph._build_plot_image
    original_bar, original_line = graph.plot_bar_png_base64, graph.plot_line_png_base64

    def build(result, question):
        captured['result'] = result
        return original_build(result, question)

    def renderer(original, kind):
        def render(xs, ys, **kwargs):
            captured.update(kind=kind, rows=[list(row) for row in zip(xs, ys)], labels=kwargs)
            return original(xs, ys, **kwargs)
        return render

    return (patch.object(graph, '_build_plot_image', build),
            patch.object(graph, 'plot_bar_png_base64', renderer(original_bar, 'bar')),
            patch.object(graph, 'plot_line_png_base64', renderer(original_line, 'line')))


def run(mode, output, delay_seconds=15, *, suite='holdout', resume=False, selected_ids=None):
    if mode not in {'offline', 'live'} or delay_seconds < 0:
        raise ValueError('Use offline/live mode and a nonnegative delay')
    items = cases(suite)
    if selected_ids is not None:
        if not selected_ids or len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - {item['id'] for item in items}:
            raise ValueError('Choose unique case IDs that exist in the selected suite')
        items = [item for item in items if item['id'] in selected_ids]
    if suite == 'identifier':
        from .identifier_fixture import create_database as build_database
    elif suite == 'transit':
        from .transit_fixture import create_database as build_database
    else:
        build_database = create_database
    if mode == 'live':
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv('GROQ_API_KEY'):
            raise ValueError('Live evaluation requires local Groq credentials.')
        from src.agent import graph
    output = Path(output)
    results_path = output / 'results.json'
    if mode == 'live' and results_path.exists() and not resume:
        raise ValueError('Choose a new output directory to preserve previous live evidence')
    output.mkdir(parents=True, exist_ok=True)
    frozen = source_fingerprints(suite)
    report = dict(mode=mode, suite=suite, fingerprints=frozen,
                  model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b') if mode == 'live' else None,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  records=[], expected_cases=len(items), selected_case_ids=selected_ids, complete=False, narrative_accuracy_measured=False,
                  numeric_support_measured=mode == 'live', agent_quality_measured=mode == 'live')
    report['configuration'] = runtime_configuration(graph) if mode == 'live' else None
    if resume:
        if not results_path.is_file():
            raise ValueError('No saved run to resume')
        previous = json.loads(results_path.read_text(encoding='utf-8'))
        if any(previous.get(key) != report.get(key) for key in
               ('mode', 'suite', 'fingerprints', 'model', 'configuration', 'expected_cases', 'selected_case_ids')):
            raise ValueError('Cannot resume with changed sources, model, suite or configuration')
        if previous.get('complete'):
            raise ValueError('This run is already complete; preserve its evidence')
        prior_records = previous.get('records', [])
        if len(prior_records) > len(items) or any(
            any(record.get(key) != items[index].get(key) for key in
                ('id', 'kind', 'question', 'reference_sql', 'expected_rows', 'ordered'))
            for index, record in enumerate(prior_records)
        ):
            raise ValueError('Saved records do not match the frozen case prefix')
        history = output / 'history'
        history.mkdir(exist_ok=True)
        (history / f'results-{uuid.uuid4().hex}.json').write_bytes(results_path.read_bytes())
        report = previous
        report.setdefault('resume_events', []).append(dict(
            resumed_at=datetime.now(timezone.utc).isoformat(),
            preserved_observations=len(prior_records), previous_stop_reason=report.pop('stopped_reason', None)))
        report.pop('stopped_at', None)
    already_observed = len(report['records'])
    with tempfile.TemporaryDirectory() as temp:
        db = Path(temp) / 'holdout.sqlite'
        build_database(db)
        for index, item in enumerate(items):
            if index < already_observed:
                continue
            if mode == 'live' and index:
                time.sleep(delay_seconds)
            if frozen != source_fingerprints(suite):
                raise ValueError('Evaluation sources changed during run')
            if mode == 'live' and runtime_configuration(graph) != report['configuration']:
                raise ValueError('Evaluation configuration changed during run')
            expected = run_sql(str(db), item['reference_sql'], max_rows=200)
            valid = expected['ok'] and equivalent(expected['rows'], item['expected_rows'], item['ordered'])
            if not valid:
                raise AssertionError(f"Invalid independent reference: {item['id']}")
            record = dict(item, reference_valid=True)
            if mode == 'live':
                captured = {}
                try:
                    # Fresh cache for every case, including overlapping SQL/chart intents.
                    with configured_graph(db, Path(temp) / f"{item['id']}.sqlite"):
                        observers = chart_observers(graph, captured)
                        with observers[0], observers[1], observers[2]:
                            response = graph.ask(item['question'])
                        actual = run_sql(str(db), response.get('sql', ''), max_rows=50)
                        record.update(generated_sql=response.get('sql'), actual_rows=actual.get('rows'),
                                      sql_correct=bool(actual['ok'] and equivalent(actual['rows'], item['expected_rows'], item['ordered'])),
                                      answer=response.get('answer'), metrics=response.get('metrics'), attempts=response.get('attempts'))
                        record['numeric_support'] = (check_numeric_support(response.get('answer'), redact_result(actual)['rows'])
                                                     if actual['ok'] else dict(status='unavailable', passed=False))
                        record['estimated_cost'] = estimate(response.get('metrics') or {}, report['model'])
                        cached = graph.ask(item['question'])
                        record['cache_hit'] = cached.get('cache_hit', False)
                        record['cached_llm_calls'] = cached.get('metrics', {}).get('llm_calls')
                        if item['kind'] != 'sql':
                            data = captured.get('result', {})
                            png = base64.b64decode(response.get('image_b64') or '')
                            assert png.startswith(b'\x89PNG\r\n\x1a\n'), 'Missing PNG'
                            with Image.open(io.BytesIO(png)) as image:
                                nonblank = max(ImageStat.Stat(image.convert('RGB')).var) > 1
                            (output / f"{item['id']}.png").write_bytes(png)
                            labels = captured.get('labels', {})
                            record.update(plot_rows=data.get('rows'), renderer_rows=captured.get('rows'), labels=labels,
                                chart_query_correct=bool(data.get('ok') and len(data.get('columns', [])) == 2 and equivalent(data['rows'], item['expected_rows'], item['ordered'])),
                                renderer_correct=captured.get('kind') == item['kind'] and equivalent(captured.get('rows', []), [[str(x), float(y)] for x, y in item['expected_rows']], item['ordered']),
                                labels_present=all(labels.get(k, '').strip().lower() not in {'', 'x', 'y', 'comparison', 'trend'} for k in ['title', 'xlabel', 'ylabel']),
                                png_nonblank=nonblank,
                                cache_preserves_image=bool(cached.get('cache_hit') and cached.get('image_b64') == response.get('image_b64') and record['cached_llm_calls'] == 0))
                            record['chart_correct'] = all(record[k] for k in ['chart_query_correct', 'renderer_correct', 'labels_present', 'png_nonblank', 'cache_preserves_image'])
                        record['passed'] = record['sql_correct'] and record.get('chart_correct', True)
                except Exception as exc:
                    record.update(passed=False, error_type=type(exc).__name__)
                    partial = getattr(exc, 'agent_metrics', None)
                    if partial is not None:
                        record['metrics'] = partial
                        record['estimated_cost'] = estimate(partial, report['model'])
                    status = getattr(exc, 'status_code', None)
                    if status in {401, 402, 403, 429}:
                        record['provider_http_status'] = status
                        report['stopped_reason'] = ('provider_rate_limit' if status == 429
                                                    else 'provider_auth_or_billing_restriction')
            report['records'].append(record)
            report['summary'] = dict(reference_passed=sum(r['reference_valid'] for r in report['records']), total=len(report['records']))
            if mode == 'live':
                report['summary'].update(sql_correct=sum(r.get('sql_correct', False) for r in report['records']),
                    charts_correct=sum(r.get('chart_correct', False) for r in report['records']),
                    charts_total=sum(r['kind'] != 'sql' for r in report['records']),
                    errors=sum('error_type' in r for r in report['records']), cache_hits=sum(r.get('cache_hit', False) for r in report['records']),
                    numeric_supported=sum(r.get('numeric_support', {}).get('status') == 'supported' for r in report['records']),
                    numeric_unsupported=sum(r.get('numeric_support', {}).get('status') == 'unsupported' for r in report['records']),
                    numeric_unscored=sum(r.get('numeric_support', {}).get('status') not in {'supported', 'unsupported'} for r in report['records']))
            (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(id=item['id'], passed=record.get('passed'), sql_correct=record.get('sql_correct'), chart_correct=record.get('chart_correct'), error_type=record.get('error_type'))), flush=True)
            if report.get('stopped_reason'):
                break
    if frozen != source_fingerprints(suite):
        raise ValueError('Evaluation sources changed during run')
    if mode == 'live' and runtime_configuration(graph) != report['configuration']:
        raise ValueError('Evaluation configuration changed during run')
    report['complete'] = len(report['records']) == len(items) and not report.get('stopped_reason')
    report['completed_at' if report['complete'] else 'stopped_at'] = datetime.now(timezone.utc).isoformat()
    (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--output', default='reports/holdout')
    parser.add_argument('--delay-seconds', type=float, default=15)
    parser.add_argument('--suite', choices=['holdout', 'transit', 'identifier'], default='holdout')
    parser.add_argument('--case', action='append', dest='selected_ids', help='Select one case ID; repeat to select more')
    parser.add_argument('--resume', action='store_true', help='Continue a stopped run without repeating saved observations')
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error('Delay must be nonnegative')
    report = run(args.mode, args.output, args.delay_seconds, suite=args.suite, resume=args.resume, selected_ids=args.selected_ids)
    print(json.dumps(report['summary']))
    if not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
