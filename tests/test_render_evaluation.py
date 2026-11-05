"""Offline checks for HTTP-report scoring. Never calls Render or a provider."""
import base64
import io

from PIL import Image, ImageDraw
import pytest

from evals.render import assess_response, run, successful


def test_render_score_compares_executed_sql_with_independent_reference(db, tmp_path):
    item = dict(id='count', reference_sql='SELECT COUNT(*) FROM orders', expected_rows=[[6]], ordered=True, chart=False)
    payload = dict(ok=True, sql='SELECT COUNT(*) FROM orders', answer='There are six orders.', metrics={'llm_calls': 2})
    result = assess_response(item, payload, db, tmp_path)
    assert result['sql_correct'] and result['passed']
    assert result['numeric_support']['passed']
    wrong = assess_response(item, dict(payload, sql='SELECT 7', answer='There are seven orders.'), db, tmp_path)
    assert not wrong['sql_correct'] and not wrong['passed']
    assert wrong['numeric_support']['passed']  # Numeric support alone cannot establish correct SQL.


def test_render_score_separates_unsupported_answer_from_correct_sql(db, tmp_path):
    item = dict(id='count', reference_sql='SELECT COUNT(*) FROM orders', expected_rows=[[6]], ordered=True, chart=False)
    payload = dict(ok=True, sql='SELECT COUNT(*) FROM orders', answer='There are 999 orders.', metrics={})
    result = assess_response(item, payload, db, tmp_path)
    assert result['sql_correct']
    assert result['numeric_support']['status'] == 'unsupported'


def test_render_score_does_not_pass_a_failed_agent_response_with_valid_sql(db, tmp_path):
    item = dict(id='count', reference_sql='SELECT COUNT(*) FROM orders', expected_rows=[[6]], ordered=True, chart=False)
    result = assess_response(item, dict(ok=False, sql='SELECT COUNT(*) FROM orders'), db, tmp_path)
    assert result['sql_correct'] and not result['passed']


@pytest.mark.parametrize('sql', ['DELETE FROM orders', 'SELECT missing FROM orders'])
def test_remote_sql_is_only_run_through_bounded_readonly_executor(db, tmp_path, sql):
    item = dict(id='count', reference_sql='SELECT COUNT(*) FROM orders', expected_rows=[[6]], ordered=True, chart=False)
    result = assess_response(item, dict(ok=True, sql=sql, answer='Six.'), db, tmp_path)
    assert not result['sql_correct']


def test_render_refusal_requires_no_sql_and_zero_model_calls(db, tmp_path):
    item = dict(id='write_refusal', reference_sql=None)
    payload = dict(ok=False, sql='', answer='This is read-only.', metrics={'llm_calls': 0})
    assert assess_response(item, payload, db, tmp_path)['passed']
    assert not assess_response(item, dict(payload, metrics={'llm_calls': 1}), db, tmp_path)['passed']


def test_render_chart_check_saves_real_png_without_claiming_plot_data_or_visual_review(db, tmp_path):
    item = dict(id='chart', reference_sql='SELECT COUNT(*) FROM orders', expected_rows=[[6]], ordered=True, chart=True)
    image = Image.new('RGB', (201, 201), 'white')
    ImageDraw.Draw(image).rectangle((30, 30, 100, 170), fill='blue')
    output = io.BytesIO()
    image.save(output, format='PNG')
    payload = dict(ok=True, sql='SELECT COUNT(*) FROM orders', answer='Six.', image_b64=base64.b64encode(output.getvalue()).decode())
    result = assess_response(item, payload, db, tmp_path)
    assert result['png_valid'] and (tmp_path / result['chart_file']).exists()
    assert not result['plot_query_data_verified'] and not result['chart_visually_reviewed']
    missing = assess_response(item, dict(payload, image_b64='invalid'), db, tmp_path)
    assert not missing['passed'] and not missing['png_valid']


def test_render_run_requires_free_only_confirmation_before_any_work(tmp_path):
    with pytest.raises(ValueError, match='free-only'):
        run(tmp_path / 'missing.sqlite', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def completed_report():
    return dict(completed_at='2026-11-05', summary=dict(completed_cases=10, passed=10),
                cache_checks=[dict(passed=True) for _ in range(3)],
                validation_checks=[dict(passed=True) for _ in range(2)])


def test_render_success_requires_primary_cache_and_input_checks():
    assert successful(completed_report())


@pytest.mark.parametrize('failure', ['cache', 'validation', 'incomplete'])
def test_render_success_is_false_for_secondary_failures_or_partial_run(failure):
    report = completed_report()
    if failure == 'incomplete':
        report.pop('completed_at')
    else:
        report['cache_checks' if failure == 'cache' else 'validation_checks'][0]['passed'] = False
    assert not successful(report)
