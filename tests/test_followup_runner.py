import httpx
import pytest

from evals.followup import FailOnceClient, run
from evals.holdout import run as run_holdout


def test_controlled_failure_happens_before_network_and_only_once():
    received = []
    def respond(request):
        received.append(request)
        return httpx.Response(200, json={'ok': True})
    with FailOnceClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.ConnectError):
            client.get('https://example.test')
        assert not received
        assert client.get('https://example.test').json() == {'ok': True}
        assert client.get('https://example.test').status_code == 200
        assert client.injected_errors == 1 and len(received) == 2


def test_followup_requires_free_only_confirmation_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match='Free organisation'):
        run(tmp_path / 'followup')
    assert not (tmp_path / 'followup').exists()


def test_selected_offline_cases_do_not_claim_full_suite_validation(tmp_path):
    result = run_holdout('offline', tmp_path / 'selected', suite='transit', selected_ids=['t15'])
    assert result['expected_cases'] == 1
    assert result['selected_case_ids'] == ['t15']
    assert result['summary']['reference_passed'] == 1
    assert result['agent_quality_measured'] is False


@pytest.mark.parametrize('ids', [[], ['t15','t15'], ['does-not-exist']])
def test_selected_cases_are_validated_before_any_work(tmp_path, ids):
    with pytest.raises(ValueError, match='unique case IDs'):
        run_holdout('offline', tmp_path / 'selected', suite='transit', selected_ids=ids)
