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
from unittest.mock import patch

from PIL import Image, ImageStat
from .holdout_fixture import create_database
from .run import configured_graph, equivalent
from src.agent.sql_tools import run_sql


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    return json.loads(Path(__file__).with_name('holdout_cases.json').read_text(encoding='utf-8'))


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


def run(mode, output, delay_seconds=15):
    from dotenv import load_dotenv
    from src.agent import graph
    if mode == 'live':
        load_dotenv()
        if not os.getenv('GROQ_API_KEY'):
            raise ValueError('Live evaluation requires local Groq credentials.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(graph.__file__).with_name('prompts.py'), Path(__file__).with_name('holdout_cases.json'), Path(__file__).with_name('holdout_fixture.py')]
    frozen = {p.name: fingerprint(p) for p in sources}
    report = dict(mode=mode, fingerprints=frozen, model=os.getenv('GROQ_MODEL'), records=[], narrative_accuracy_measured=False)
    with tempfile.TemporaryDirectory() as temp:
        db = Path(temp) / 'holdout.sqlite'
        create_database(db)
        for index, item in enumerate(cases()):
            if mode == 'live' and index:
                time.sleep(delay_seconds)
            assert frozen == {p.name: fingerprint(p) for p in sources}, 'Evaluation inputs changed during run'
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
            report['records'].append(record)
            report['summary'] = dict(reference_passed=sum(r['reference_valid'] for r in report['records']), total=len(report['records']))
            if mode == 'live':
                report['summary'].update(sql_correct=sum(r.get('sql_correct', False) for r in report['records']),
                    charts_correct=sum(r.get('chart_correct', False) for r in report['records']),
                    charts_total=sum(r['kind'] != 'sql' for r in report['records']),
                    errors=sum('error_type' in r for r in report['records']), cache_hits=sum(r.get('cache_hit', False) for r in report['records']))
            (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(id=item['id'], passed=record.get('passed'), sql_correct=record.get('sql_correct'), chart_correct=record.get('chart_correct'), error_type=record.get('error_type'))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--output', default='reports/holdout')
    parser.add_argument('--delay-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error('Delay must be nonnegative')
    print(json.dumps(run(args.mode, args.output, args.delay_seconds)['summary']))


if __name__ == '__main__':
    main()
