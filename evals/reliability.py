"""Repeated fresh-cache evaluation with a frozen implementation and per-case results."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path

from . import holdout


def summarise(reports, planned_runs):
    if not reports:
        raise ValueError('No reports to summarise')
    first = reports[0]
    ids = [r['id'] for r in first['records']]
    expected_ids = [r['id'] for r in holdout.cases(first['suite'])]
    if ids != expected_ids or len(set(ids)) != len(ids) or not ids:
        raise ValueError('Reports must contain every case in the suite exactly once')
    if len(reports) > planned_runs or planned_runs < 2:
        raise ValueError('Invalid planned/completed run count')
    for report in reports:
        if any(report.get(key) != first.get(key) for key in ('mode', 'suite', 'model', 'configuration', 'fingerprints')):
            raise ValueError('Cannot combine different modes, suites, models, configurations or source fingerprints')
        if [r['id'] for r in report['records']] != ids:
            raise ValueError('Every run must contain the same complete ordered case set')
    complete = len(reports) == planned_runs
    live = first['mode'] == 'live'
    result = dict(mode=first['mode'], suite=first['suite'], model=first['model'],
                  fingerprints=first['fingerprints'], configuration=first.get('configuration'), planned_runs=planned_runs,
                  completed_runs=len(reports), complete=complete,
                  agent_quality_measured=live, reliability_measured=live and complete,
                  semantic_factuality_measured=False, cases=[])
    if not live:
        result['reference_checks_passed'] = sum(r['reference_valid'] for report in reports for r in report['records'])
        return result
    records = [r for report in reports for r in report['records']]
    for index, case_id in enumerate(ids):
        outcomes = [report['records'][index] for report in reports]
        passed = [r.get('passed') is True for r in outcomes]
        result['cases'].append(dict(
            id=case_id, question=outcomes[0]['question'], passed_runs=sum(passed),
            observed_runs=len(passed), pass_rate=sum(passed) / len(passed),
            passed_every_run=all(passed) if complete else None,
            mixed_results=any(passed) and not all(passed),
            sql_correct_runs=sum(r.get('sql_correct') is True for r in outcomes),
            numeric_supported_runs=sum(r.get('numeric_support', {}).get('status') == 'supported' for r in outcomes),
            outcomes=[dict(run=i + 1, passed=passed[i], sql_correct=r.get('sql_correct', False),
                           chart_correct=r.get('chart_correct'), error_type=r.get('error_type'),
                           numeric_support=r.get('numeric_support', {}).get('status', 'unavailable'))
                      for i, r in enumerate(outcomes)],
        ))
    costs = [r.get('estimated_cost', {}) for r in records]
    known_costs = [Decimal(c['estimated_usd']) for c in costs if c.get('available')]
    result['summary'] = dict(
        case_count=len(ids), observations=len(records),
        passed_observations=sum(r.get('passed') is True for r in records),
        sql_correct_observations=sum(r.get('sql_correct') is True for r in records),
        always_passed_cases=sum(c['passed_every_run'] is True for c in result['cases']) if complete else None,
        always_passed_rate=sum(c['passed_every_run'] is True for c in result['cases']) / len(ids) if complete else None,
        mixed_cases=[c['id'] for c in result['cases'] if c['mixed_results']],
        never_passed_cases=[c['id'] for c in result['cases'] if c['passed_runs'] == 0],
        error_observations=sum('error_type' in r for r in records),
        numeric_supported=sum(r.get('numeric_support', {}).get('status') == 'supported' for r in records),
        numeric_unsupported=sum(r.get('numeric_support', {}).get('status') == 'unsupported' for r in records),
        numeric_unscored=sum(r.get('numeric_support', {}).get('status') not in {'supported', 'unsupported'} for r in records),
        estimated_cost_observations=len(known_costs),
        cost_coverage_complete=len(known_costs) == len(records),
        known_estimated_usd=str(sum(known_costs, Decimal(0))) if known_costs else None,
        mean_estimated_usd_per_question=str(sum(known_costs, Decimal(0)) / len(records))
            if len(known_costs) == len(records) else None,
        actual_bill_measured=False,
    )
    return result


def markdown(report):
    lines = ['# Repeated-run evaluation', '',
             f"Mode: {report['mode']}; suite: {report['suite']}; completed runs: {report['completed_runs']}/{report['planned_runs']}.", '']
    if not report['agent_quality_measured']:
        return '\n'.join(lines + ['Offline reference validation only. No model accuracy or reliability was measured.', ''])
    summary = report['summary']
    lines += [f"SQL matches: {summary['sql_correct_observations']}/{summary['observations']} observations.",
              'A case passes when both its SQL and any required chart checks pass. Numeric support is scored separately.', '',
              '| Case | Passed runs | All planned runs passed | Numeric-supported runs |',
              '| --- | --- | --- | --- |']
    for case in report['cases']:
        lines.append(f"| {case['id']} | {case['passed_runs']}/{case['observed_runs']} | {case['passed_every_run']} | {case['numeric_supported_runs']} |")
    lines += ['', f"Mixed outcomes: {', '.join(summary['mixed_cases']) or 'none observed'}.",
              f"Numeric support: {summary['numeric_supported']} supported, {summary['numeric_unsupported']} unsupported, {summary['numeric_unscored']} unscored.",
              'Numeric support does not verify labels, units, completeness, or which row a number describes.',
              f"Known standard-list-price estimate (USD): {summary['known_estimated_usd']}; coverage {summary['estimated_cost_observations']}/{summary['observations']}.",
              'This is not a measured bill. Failures without token usage prevent a complete per-question estimate.', '',
              'Reports use synthetic data and fresh caches. The original holdout is an exposed regression suite, not unseen data.', '']
    return '\n'.join(lines)


def run(mode, output, runs=5, delay_seconds=15, suite='holdout', resume=False):
    if runs < 2 or mode not in {'offline', 'live'} or delay_seconds < 0:
        raise ValueError('Use at least two runs, offline/live mode and a nonnegative delay')
    # Load configuration before freezing model identity; never record credentials.
    if mode == 'live':
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv('GROQ_API_KEY'):
            raise ValueError('Live evaluation requires local Groq credentials')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Choose a new empty output directory to preserve previous evidence')
    output.mkdir(parents=True, exist_ok=True)
    frozen = holdout.source_fingerprints(suite)
    runner_fingerprint = holdout.fingerprint(Path(__file__))
    model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b') if mode == 'live' else None
    started = datetime.now(timezone.utc).isoformat()
    if resume:
        summary_path = output / 'summary.json'
        if not summary_path.is_file():
            raise ValueError('No saved batch to resume')
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        if any(previous.get(key) != value for key, value in dict(
            mode=mode, suite=suite, model=model, fingerprints=frozen,
            runner_fingerprint=runner_fingerprint, planned_runs=runs).items()):
            raise ValueError('Cannot resume a changed batch, model, sources or run count')
        if previous.get('complete'):
            return previous
        started = previous['started_at']
    reports = []
    for index in range(runs):
        if frozen != holdout.source_fingerprints(suite) or runner_fingerprint != holdout.fingerprint(Path(__file__)):
            raise ValueError('Evaluation sources changed between runs')
        run_output = output / f'run-{index + 1:02d}'
        saved = run_output / 'results.json'
        if resume and saved.exists():
            report = json.loads(saved.read_text(encoding='utf-8'))
            if not report.get('complete'):
                report = holdout.run(mode, run_output, delay_seconds, suite=suite, resume=True)
        else:
            report = holdout.run(mode, run_output, delay_seconds, suite=suite)
        if report['fingerprints'] != frozen or report['model'] != model:
            raise ValueError('Model or evaluation sources changed between runs')
        if not report.get('complete', True):
            aggregate = (summarise(reports, runs) if reports else
                         dict(mode=mode, suite=suite, model=model, fingerprints=frozen,
                              planned_runs=runs, completed_runs=0, complete=False,
                              agent_quality_measured=mode == 'live', reliability_measured=False,
                              semantic_factuality_measured=False, cases=[]))
            aggregate.update(started_at=started, stopped_at=datetime.now(timezone.utc).isoformat(),
                             runner_fingerprint=runner_fingerprint,
                             stopped_reason=report.get('stopped_reason', 'incomplete_run'),
                             partial_run=index + 1, partial_run_observations=len(report['records']))
            (output / 'summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
            (output / 'REPORT.md').write_text(
                f"# Repeated-run evaluation stopped\n\nCompleted runs: {len(reports)}/{runs}. "
                f"Run {index + 1} has {len(report['records'])} saved observations.\n\n"
                f"Stop reason: {aggregate['stopped_reason']}. No paid fallback or account upgrade was attempted. "
                "Partial results are retained; no all-runs reliability result is claimed.\n", encoding='utf-8')
            return aggregate
        reports.append(report)
        aggregate = dict(summarise(reports, runs), started_at=started, runner_fingerprint=runner_fingerprint,
                         updated_at=datetime.now(timezone.utc).isoformat())
        (output / 'summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
        (output / 'REPORT.md').write_text(markdown(aggregate), encoding='utf-8')
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--suite', choices=['holdout', 'transit'], default='holdout')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--delay-seconds', type=float, default=15)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse completed runs and continue a stopped run')
    args = parser.parse_args()
    if args.runs < 2 or args.delay_seconds < 0:
        parser.error('Use at least two runs and a nonnegative delay')
    report = run(args.mode, args.output, args.runs, args.delay_seconds, args.suite, args.resume)
    print(json.dumps({key: value for key, value in report.items() if key not in {'cases', 'fingerprints'}}, indent=2))
    if not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
