"""Identifier context is grounded locally, bounded and independent of the LLM."""
import sqlite3
from types import SimpleNamespace

import pytest

from evals.run import configured_graph
from evals.holdout import cases
from evals.transit_fixture import create_database
from src.agent.identifiers import identifier_matches
from src.agent.schema import get_schema


@pytest.fixture
def transit(tmp_path):
    path = tmp_path / 'transit.sqlite'
    create_database(path)
    return path


def test_code_is_confirmed_against_stop_key_without_exporting_name(transit):
    hints = identifier_matches(str(transit), get_schema(str(transit)), 'Which routes serve stop S4?')
    assert hints == [dict(table='dim_stop', column='stop_key', value='S4')]
    assert 'Beach' not in str(hints)


@pytest.mark.parametrize('question', ["Which routes serve stop named S4?", "Return stop_key where stop_name = 'S4'."])
def test_explicit_name_is_not_reinterpreted_as_identifier(transit, question):
    assert identifier_matches(str(transit), get_schema(str(transit)), question) == []


def test_unknown_key_has_no_fabricated_match(transit):
    assert identifier_matches(str(transit), get_schema(str(transit)), 'Which routes serve stop S99?') == []


def test_identifier_context_uses_related_tables_only_and_exact_case(transit):
    schema = get_schema(str(transit))
    assert identifier_matches(str(transit), schema, 'Which routes serve stop s4?') == []
    assert identifier_matches(str(transit), schema, 'Which routes serve stop R1?') == []
    assert identifier_matches(str(transit), schema, 'Return route_short_name for route R3.') == [
        dict(table='dim_route', column='route_key', value='R3')]


def test_generic_keys_work_and_personal_fields_are_not_probed(tmp_path):
    path = tmp_path / 'codes.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript("""
          CREATE TABLE depots(depot_code TEXT PRIMARY KEY, depot_name TEXT);
          CREATE TABLE contacts(email TEXT PRIMARY KEY, value TEXT);
          INSERT INTO depots VALUES('D7','Private display value');
          INSERT INTO contacts VALUES('D7','Do not disclose');
        """)
    hints = identifier_matches(str(path), get_schema(str(path)), 'Find depot D7 and contact D7.')
    assert hints == [dict(table='depots', column='depot_code', value='D7')]


def test_case_insensitive_primary_key_does_not_create_false_exact_match(tmp_path):
    path = tmp_path / 'nocase.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE depots(depot_code TEXT PRIMARY KEY COLLATE NOCASE); INSERT INTO depots VALUES('D7');")
    schema = get_schema(str(path))
    assert identifier_matches(str(path), schema, 'Return depot d7.') == []
    assert identifier_matches(str(path), schema, 'Return depot D7.') == [dict(table='depots',column='depot_code',value='D7')]


def test_parameter_bound_codes_cannot_modify_database(transit):
    schema = get_schema(str(transit))
    identifier_matches(str(transit), schema, "stop S4'; DROP TABLE dim_stop;--")
    with sqlite3.connect(transit) as conn:
        assert conn.execute('SELECT COUNT(*) FROM dim_stop').fetchone()[0] == 6


def test_confirmed_key_is_in_generation_and_plot_context(transit, tmp_path, monkeypatch):
    from src.agent import graph
    calls = []
    outputs = iter([
        "SELECT route_key,COUNT(*) AS event_count FROM fact_stop_events WHERE stop_key='S4' GROUP BY route_key",
        "SELECT route_key,COUNT(*) AS event_count FROM fact_stop_events WHERE stop_key='S4' GROUP BY route_key",
        'Route R2 has 2 events.',
    ])
    def invoke(messages):
        calls.append(messages)
        return SimpleNamespace(content=next(outputs))
    monkeypatch.setattr(graph, '_make_llm', lambda: SimpleNamespace(invoke=invoke))
    with configured_graph(transit, tmp_path / 'cache.sqlite'):
        result = graph.ask('Draw a bar chart of stop event counts by route for stop S4.')
    assert result['ok'] and result['image_b64']
    for messages in calls[:2]:
        context = '\n'.join(message.content for message in messages)
        assert 'CONFIRMED IDENTIFIER MATCHES' in context and 'stop_key' in context and 'S4' in context


def test_stop_does_not_trigger_top_or_force_empty_result_repairs():
    from src.agent import graph
    assert not graph._expects_non_empty('Return routes serving stop S99.')
    assert graph._expects_non_empty('Return the top three routes.')
    assert graph.decide_retry(dict(question='Return routes serving stop S99.', result={'ok': True, 'rows': []}, attempts=1)) == 'answer'


@pytest.mark.parametrize('case', cases('identifier'), ids=lambda c:c['id'])
def test_fresh_identifier_references(tmp_path, case):
    from evals.identifier_fixture import create_database
    from evals.run import equivalent
    from src.agent.sql_tools import run_sql
    path = tmp_path / 'fresh.sqlite'
    create_database(path)
    result = run_sql(str(path), case['reference_sql'])
    assert result['ok'] and equivalent(result['rows'], case['expected_rows'], case['ordered'])
