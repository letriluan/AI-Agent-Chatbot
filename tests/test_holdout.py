"""Independent reference and evaluator checks; never calls a provider."""
import pytest
from evals.holdout import cases, run
from evals.holdout_fixture import create_database
from evals.run import equivalent
from src.agent.sql_tools import run_sql


@pytest.mark.parametrize('case', cases(), ids=lambda c: c['id'])
def test_holdout_reference(tmp_path, case):
    db = tmp_path / 'holdout.sqlite'
    create_database(db)
    result = run_sql(str(db), case['reference_sql'])
    assert result['ok']
    assert equivalent(result['rows'], case['expected_rows'], case['ordered'])


def test_holdout_is_separate_and_covers_empty_zero_ties_and_gaps():
    items = {c['id']: c for c in cases()}
    assert items['h02']['expected_rows'] == [[5], [6]]
    assert items['h12']['expected_rows'] == []
    assert ['NT', 0] in items['h13']['expected_rows']
    assert [r[0] for r in items['h14']['expected_rows']] == ['2026-01', '2026-02', '2026-04', '2026-05']
    assert not equivalent([[5, 120], [6, 120]], items['h02']['expected_rows'])
    assert not equivalent([[1, 100], [2, 200]], [[1, 100], [2, 201]])


def test_offline_holdout_does_not_claim_model_quality(tmp_path):
    result = run('offline', tmp_path)
    assert result['summary'] == {'reference_passed': 16, 'total': 16}
    assert all('sql_correct' not in row for row in result['records'])
    assert len(result['fingerprints']['prompts.py']) == 64
