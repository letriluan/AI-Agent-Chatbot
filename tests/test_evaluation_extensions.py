"""Offline regression tests for evidence accounting, never provider quality claims."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from evals import holdout, reliability
from evals.cost import estimate
from evals.grounding import check_numeric_support
from evals.run import configured_graph, equivalent
from evals.transit_fixture import create_database
from src.agent.sql_tools import run_sql


@pytest.mark.parametrize('answer,rows,status', [
    ('The total is 1,000.00.', [[1000]], 'supported'),
    ('The percentage is 62.5%.', [[62.5]], 'supported'),
    ('The average is 20.83.', [[125 / 6]], 'supported'),
    ('The average is 21.', [[125 / 6]], 'unsupported'),
    ('The change is -5.', [[5]], 'unsupported'),
    ('The change is −5.', [[-5]], 'supported'),
    ('There are two orders.', [[2]], 'supported'),
    ('There are two orders.', [[3]], 'unsupported'),
    ('The percentage is 62.5%.', [[0.625]], 'unsupported'),
    ('The value is 1.2e3.', [[1200]], 'supported'),
    ('1. Total: 100.\n2. Other: 200.', [[100, 200]], 'supported'),
    ('The date is 2026-11-03.', [['2026-11-02']], 'unsupported'),
    ('The date is 2026-11-02.', [['2026-11-02']], 'supported'),
    ('The month is 2026\u201101.', [['2026-01']], 'supported'),
    ('At 25:05:00.', [['25:05:00']], 'supported'),
    ('At 25:06:00.', [['25:05:00']], 'unsupported'),
    ('No matching records.', [], 'no_numeric_claims'),
    ('', [[42]], 'no_numeric_claims'),
    ('There are 99 orders.', [[100]], 'unsupported'),
    ('There are twenty-one orders.', [[20, 1]], 'needs_review'),
    ('There are one hundred orders.', [[1]], 'needs_review'),
    ('Not a value.', [[None]], 'no_numeric_claims'),
    ('The value is 1.23.', [[1e100]], 'unsupported'),
])
def test_numeric_support_is_conservative(answer, rows, status):
    result = check_numeric_support(answer, rows)
    assert result['status'] == status
    assert result['passed'] is (status == 'supported')
    assert not result['semantic_factuality_measured']


def test_numeric_support_does_not_imply_correct_row_labels():
    # Intentionally swapped labels expose this check's documented limitation.
    assert check_numeric_support('SA: 80; WA: 100.', [['SA', 100], ['WA', 80]])['passed']


def test_cost_uses_both_token_directions_and_rejects_missing_usage():
    metrics = dict(llm_calls=2, input_tokens=1000, output_tokens=100, token_usage_complete=True)
    assert estimate(metrics, 'openai/gpt-oss-20b')['estimated_usd'] == '0.000105'
    assert not estimate(metrics, 'other-model')['available']
    assert not estimate({k: v for k, v in metrics.items() if k != 'token_usage_complete'}, 'openai/gpt-oss-20b')['available']
    assert not estimate(dict(metrics, token_usage_complete=False), 'openai/gpt-oss-20b')['available']
    assert not estimate(dict(metrics, input_tokens=-1), 'openai/gpt-oss-20b')['available']
    assert estimate(dict(metrics, llm_calls=0, input_tokens=0, output_tokens=0), 'openai/gpt-oss-20b')['estimated_usd'] == '0.000'


def test_missing_usage_is_not_reported_as_a_zero_cost_request(monkeypatch, tmp_path):
    from src.agent import graph
    from evals.fixture import create_database as base_database
    database = tmp_path / 'test.sqlite'
    base_database(database)
    outputs = iter([SimpleNamespace(content='SELECT COUNT(*) FROM orders'),
                    SimpleNamespace(content='There are six orders.', usage_metadata={'input_tokens': 5, 'output_tokens': 3})])
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=lambda _: next(outputs)))
    with configured_graph(database, tmp_path / 'cache.sqlite'):
        result = graph.ask('How many orders are there?')
        cached = graph.ask('How many orders are there?')
    assert result['metrics']['token_usage_complete'] is False
    assert not estimate(result['metrics'], 'openai/gpt-oss-20b')['available']
    assert cached['metrics']['token_usage_complete'] is True
    assert cached['metrics']['llm_calls'] == 0
    assert estimate(cached['metrics'], 'openai/gpt-oss-20b')['available']


@pytest.mark.parametrize('case', holdout.cases('transit'), ids=lambda c: c['id'])
def test_transit_independent_references(tmp_path, case):
    db = tmp_path / 'transit.sqlite'
    create_database(db)
    actual = run_sql(str(db), case['reference_sql'], max_rows=200)
    assert actual['ok'], actual.get('error')
    assert equivalent(actual['rows'], case['expected_rows'], case['ordered'])


@pytest.mark.parametrize('case', holdout.cases('transit'), ids=lambda c: c['id'])
def test_transit_orchestration_with_scripted_model(tmp_path, monkeypatch, case):
    # Reuses real SQL execution and chart rendering, with no provider calls.
    from src.agent import graph
    database = tmp_path / 'transit.sqlite'
    create_database(database)
    outputs = iter([case['reference_sql']] + ([case['reference_sql']] if case['kind'] != 'sql' else []) + ['Results from supplied data.'])
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=next(outputs))))
    with configured_graph(database, tmp_path / 'cache.sqlite'):
        response = graph.ask(case['question'])
    assert response['ok']
    actual = run_sql(str(database), response['sql'])
    assert equivalent(actual['rows'], case['expected_rows'])
    assert bool(response['image_b64']) is (case['kind'] != 'sql')


def test_retrospective_review_records_its_provenance_and_ignores_missing_answers(tmp_path):
    from evals.grounding import review_report
    source = tmp_path / 'old-live.json'
    source.write_text(json.dumps(dict(mode='live', records=[
        dict(id='one', answer='Total: 99.', actual_rows=[[100]]), dict(id='failure'),
    ])))
    report = review_report(source)
    assert report['summary'] == dict(total=2, supported=0, unsupported=1, unscored=1)
    assert report['new_model_calls'] == 0
    assert len(report['source_sha256']) == len(report['checker_sha256']) == 64


def example_report():
    return dict(mode='live', suite='holdout', model='test-model', fingerprints={'prompts.py': 'frozen'},
                records=[dict(item, reference_valid=True, passed=True, sql_correct=True,
                              numeric_support={'status': 'supported'}) for item in holdout.cases()])


def test_reliability_keeps_failures_and_requires_all_runs():
    first = example_report()
    second = deepcopy(first)
    second['records'][0].update(passed=False, sql_correct=False, error_type='RateLimitError')
    report = reliability.summarise([first, second], 2)
    assert report['summary']['always_passed_cases'] == 15
    assert report['summary']['mixed_cases'] == ['h01']
    assert report['summary']['error_observations'] == 1
    assert report['summary']['sql_correct_observations'] == 31
    assert report['cases'][0]['pass_rate'] == 0.5
    assert not report['summary']['cost_coverage_complete']
    partial = reliability.summarise([first], 5)
    assert not partial['complete'] and not partial['reliability_measured']
    assert partial['summary']['always_passed_rate'] is None
    assert partial['cases'][0]['passed_every_run'] is None


@pytest.mark.parametrize('key,value', [('model', 'another'), ('mode', 'offline'), ('configuration', {'temperature': 1}), ('fingerprints', {'prompts.py': 'changed'})])
def test_reliability_rejects_incomparable_reports(key, value):
    first = example_report()
    second = deepcopy(first)
    second[key] = value
    with pytest.raises(ValueError):
        reliability.summarise([first, second], 2)


def test_reliability_rejects_partial_or_duplicate_case_sets():
    report = example_report()
    report['records'].pop()
    with pytest.raises(ValueError):
        reliability.summarise([report], 5)


def test_offline_repeats_never_claim_model_reliability(tmp_path):
    report = reliability.run('offline', tmp_path / 'run', runs=2, delay_seconds=0)
    assert report['complete']
    assert report['reference_checks_passed'] == 32
    assert not report['agent_quality_measured'] and not report['reliability_measured']
    assert report['cases'] == []
    saved = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert saved['completed_runs'] == 2
    assert 'No model accuracy' in (tmp_path / 'run' / 'REPORT.md').read_text()
    with pytest.raises(ValueError, match='new empty output'):
        reliability.run('offline', tmp_path / 'run', runs=2)


def test_holdout_live_cannot_overwrite_previous_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-not-a-key')
    previous = tmp_path / 'results.json'
    previous.write_text('preserve this first result')
    with pytest.raises(ValueError, match='preserve previous live evidence'):
        holdout.run('live', tmp_path, delay_seconds=0)
    assert previous.read_text() == 'preserve this first result'


def test_repeated_runner_rejects_source_changes_before_calling_model(tmp_path, monkeypatch):
    fingerprints = iter([{'prompts.py': 'original'}, {'prompts.py': 'changed'}])
    monkeypatch.setattr(holdout, 'source_fingerprints', lambda _: next(fingerprints))
    with pytest.raises(ValueError, match='changed between runs'):
        reliability.run('offline', tmp_path / 'run', runs=2)
    assert not (tmp_path / 'run' / 'summary.json').exists()


def test_transit_graph_integration_uses_fresh_caches_and_scores_numeric_support(tmp_path, monkeypatch):
    from src.agent import graph
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-not-a-key')
    monkeypatch.setattr(holdout, 'cases', lambda _: [dict(holdout_item)])
    holdout_item = dict(id='t01', kind='sql', question='How many distinct trips are scheduled?',
                        reference_sql='SELECT COUNT(DISTINCT trip_id) FROM fact_stop_events',
                        expected_rows=[[6]], ordered=True)
    calls = []
    def invoke(_):
        calls.append(1)
        return SimpleNamespace(content='SELECT COUNT(DISTINCT trip_id) FROM fact_stop_events' if len(calls) % 2 else 'There are six trips.',
                               usage_metadata={'input_tokens': 5, 'output_tokens': 3})
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=invoke))
    for index in range(2):
        report = holdout.run('live', tmp_path / f'run-{index}', delay_seconds=0, suite='transit')
        record = report['records'][0]
        assert record['passed'] and record['numeric_support']['passed']
        assert record['cached_llm_calls'] == 0 and record['cache_hit']
    assert len(calls) == 4


@pytest.mark.parametrize('http_status', [401, 402, 403, 429])
def test_live_evaluation_stops_and_saves_when_provider_rejects_access(tmp_path, monkeypatch, http_status):
    from src.agent import graph
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-not-a-key')
    selected = holdout.cases()[:2]
    monkeypatch.setattr(holdout, 'cases', lambda _: selected)
    calls = []
    class ProviderRejected(Exception):
        status_code = http_status
    def invoke(_):
        calls.append(1)
        raise ProviderRejected('secret-provider-message-must-not-be-saved')
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=invoke))
    report = holdout.run('live', tmp_path / 'rejected', delay_seconds=0)
    assert len(calls) == 1
    assert not report['complete'] and 'completed_at' not in report
    assert len(report['records']) == 1
    assert report['records'][0]['provider_http_status'] == http_status
    assert report['stopped_reason'] == ('provider_rate_limit' if http_status == 429 else 'provider_auth_or_billing_restriction')
    saved = (tmp_path / 'rejected' / 'results.json').read_text()
    assert 'secret-provider-message' not in saved


def test_repeated_evaluation_retains_partial_run_and_does_not_start_next_one(tmp_path, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-not-a-key')
    calls = []
    def stopped(mode, output, delay, suite):
        calls.append(1)
        return dict(mode=mode, suite=suite, model='openai/gpt-oss-20b',
                    fingerprints=holdout.source_fingerprints(suite), records=[dict(id='h01')],
                    complete=False, stopped_reason='provider_rate_limit')
    monkeypatch.setattr(holdout, 'run', stopped)
    result = reliability.run('live', tmp_path / 'batch', runs=5, delay_seconds=0)
    assert calls == [1]
    assert result['completed_runs'] == 0 and result['partial_run_observations'] == 1
    assert not result['complete'] and not result['reliability_measured']
    assert (tmp_path / 'batch' / 'summary.json').is_file()
    assert 'No paid fallback' in (tmp_path / 'batch' / 'REPORT.md').read_text()


def test_live_resume_preserves_limit_failure_and_only_runs_remaining_questions(tmp_path, monkeypatch):
    from src.agent import graph
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-not-a-key')
    selected = holdout.cases()[:2]
    monkeypatch.setattr(holdout, 'cases', lambda _: selected)
    class RateLimited(Exception):
        status_code = 429
    def rejected(_):
        raise RateLimited('private diagnostic')
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=rejected))
    first = holdout.run('live', tmp_path / 'run', delay_seconds=0)
    assert not first['complete']
    outputs = iter([selected[1]['reference_sql'], 'The IDs are 5 and 6.'])
    calls = []
    def respond(_):
        calls.append(1)
        return SimpleNamespace(content=next(outputs), usage_metadata={'input_tokens': 5, 'output_tokens': 3})
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=respond))
    resumed = holdout.run('live', tmp_path / 'run', delay_seconds=0, resume=True)
    assert resumed['complete'] and len(resumed['records']) == 2
    assert resumed['records'][0] == first['records'][0]
    assert resumed['records'][1]['passed']
    assert len(calls) == 2
    assert len(list((tmp_path / 'run' / 'history').glob('*.json'))) == 1
    assert resumed['summary']['errors'] == 1  # Do not erase quota failures from the denominator.


def test_completed_reliability_batch_can_be_read_without_repeating_calls(tmp_path, monkeypatch):
    output = tmp_path / 'batch'
    original = reliability.run('offline', output, runs=2, delay_seconds=0)
    monkeypatch.setattr(holdout, 'run', lambda *a, **k: pytest.fail('No completed case may be repeated'))
    resumed = reliability.run('offline', output, runs=2, delay_seconds=0, resume=True)
    assert resumed == original
