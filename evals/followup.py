"""Targeted regression, controlled connection recovery and fresh identifier checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import httpx

from . import holdout


class FailOnceClient(httpx.Client):
    """One labelled pre-network ConnectError; subsequent sends use real HTTP."""
    injected_errors = 0

    def send(self, request, *args, **kwargs):
        if self.injected_errors == 0:
            self.injected_errors += 1
            raise httpx.ConnectError('Controlled pre-network connection failure', request=request)
        return super().send(request, *args, **kwargs)


def run(output, delay_seconds=30, free_only_confirmed=False):
    if not free_only_confirmed:
        raise ValueError('Use the confirmed Free organisation; paid usage is not authorised')
    if delay_seconds < 30:
        raise ValueError('Keep the free-only follow-up pacing at 30 seconds or more')
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv('GROQ_API_KEY'):
        raise ValueError('Configure the local Groq key first')
    from src.agent import graph
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; original reports stay unchanged')
    output.mkdir(parents=True, exist_ok=True)
    frozen = {suite: holdout.source_fingerprints(suite) for suite in ['transit', 'holdout', 'identifier']}
    runner = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(mode='live_targeted_followup', free_only_confirmed=True,
                    started_at=datetime.now(timezone.utc).isoformat(), complete=False,
                    runner_sha256=runner, fingerprints=frozen, phases=[])

    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def frozen_check():
        if (frozen != {suite: holdout.source_fingerprints(suite) for suite in frozen}
                or runner != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ValueError('Sources changed during the follow-up')

    def phase(name, suite, ids=None, controlled=False):
        frozen_check()
        if manifest['phases']:
            time.sleep(delay_seconds)
        report = holdout.run('live', output / name, delay_seconds, suite=suite, selected_ids=ids)
        manifest['phases'].append(dict(name=name, suite=suite, selected_ids=ids,
                                      controlled_connection_failure=controlled,
                                      complete=report['complete'], summary=report['summary']))
        save()
        return report

    save()
    for name, suite, ids in [('transit-regression', 'transit', ['t15']),
                             ('holdout-regressions', 'holdout', ['h12','h15','h16'])]:
        report = phase(name, suite, ids)
        if not report['complete']:
            manifest['stopped_reason'] = report.get('stopped_reason')
            save()
            return manifest

    # This is an explicit recovery experiment, not a naturally observed outage.
    original = graph.ChatGroq
    models = []
    with FailOnceClient(timeout=20) as client:
        def connected_model(**kwargs):
            model = original(**kwargs, http_client=client)
            models.append(model)
            return model
        with patch.object(graph, 'ChatGroq', connected_model):
            report = phase('controlled-recovery', 'holdout', ['h12'], controlled=True)
        record = report['records'][0]
        recovered = bool(record.get('passed') and record.get('metrics', {}).get('connection_retries', 0) >= 1
                         and client.injected_errors == 1)
        manifest['phases'][-1]['controlled_recovery_passed'] = recovered
        save()
        if not report['complete']:
            manifest['stopped_reason'] = report.get('stopped_reason')
            save()
            return manifest

    report = phase('fresh-identifiers', 'identifier')
    if not report['complete']:
        manifest['stopped_reason'] = report.get('stopped_reason')
        save()
        return manifest
    frozen_check()
    manifest.update(complete=True, completed_at=datetime.now(timezone.utc).isoformat())
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--free-only-confirmed', action='store_true')
    args = parser.parse_args()
    manifest = run(args.output, free_only_confirmed=args.free_only_confirmed)
    print(json.dumps(manifest['phases'], indent=2))
    if not manifest['complete']:
        raise SystemExit(2)
    if (any(phase['summary']['sql_correct'] != phase['summary']['total']
            or phase['summary'].get('errors')
            or phase['summary'].get('charts_correct', 0) != phase['summary'].get('charts_total', 0)
            for phase in manifest['phases'])
            or not manifest['phases'][2].get('controlled_recovery_passed')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
