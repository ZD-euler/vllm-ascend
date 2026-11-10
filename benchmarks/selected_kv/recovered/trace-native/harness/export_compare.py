"""Export allowlisted aggregate evidence; prompts and token IDs stay remote."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from statistics import median
import spec
import preflight
from perf_common import atomic
DEFAULT_ROOT = Path('/repro/reference-root/m3v15_q4k_chunk6k_20261107/trace-native')
ARMS = preflight.ARMS
LABELS = {'native': 'native-layerwise-B2', 'v0': 'V0-indexer-layerwise', 'm3v9': 'M3v15'}
TOLERANCE = 1e-05

def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite timing or memory metric')
    return result

def metric(sample, key):
    return [finite(row[key]) for row in sample['rank_rows']]

def validate(sample, expected):
    if sample['case'] != expected['case'] or sample['H'] != expected['H']:
        raise ValueError('sample case/history differs from manifest')
    if sample['request_Qs'] != expected['request_Qs']:
        raise ValueError('sample query lengths differ from manifest')
    if sample['global_BS'] != len(expected['request_Qs']):
        raise ValueError('sample request count differs from manifest')
    rows = sample['rank_rows']
    if {(r['dp_rank'], r['tp_rank']) for r in rows} != {(d, t) for d in range(spec.DP) for t in range(spec.TP)} or len(rows) != spec.DP * spec.TP:
        raise ValueError('four unique TP2/DP2 rank rows required')
    for row in rows:
        qs = expected['request_Qs'][row['dp_rank']::spec.DP]
        if not row['valid'] or row.get('issues'):
            raise ValueError('worker rejected this sample')
        if row['actual_H'] != [expected['H']] * len(qs):
            raise ValueError('prefix miss or actual history mismatch')
        if sorted(row['actual_Q']) != sorted(qs) or sorted(row['scheduled_Q']) != sorted(qs):
            raise ValueError('actual scheduled query mismatch')
        if row['model_forward_calls'] != 1:
            raise ValueError('one model forward per DP is required')
        if finite(row['model_npu_ms']) <= 0 or finite(row['allocated_peak_bytes']) <= 0:
            raise ValueError('nonpositive forward duration or HBM peak')
    count = len(expected['request_Qs'])
    if len(sample['first_tokens']) != count or len(sample['logprobs']) != count:
        raise ValueError('missing generated output records')
    if any((len(values) < spec.LOGP_TOP for values in sample['logprobs'])):
        raise ValueError('top-logprob record has fewer entries than requested')

def load_samples(root, arm):
    cases = spec.case_map()
    values = {}
    for path in sorted((root / 'runs' / arm / 'client' / 'samples').glob('*.json')):
        sample = json.loads(path.read_text())
        validate(sample, cases[sample['case']])
        proofs = [json.loads(p.read_text()) for p in (root / 'runs' / arm / 'client' / 'workers').glob(f"completion-{sample['trial']}-*.json")]
        if len(proofs) != spec.DP * spec.TP or {(p['dp_rank'], p['tp_rank']) for p in proofs} != {(d, t) for d in range(spec.DP) for t in range(spec.TP)}:
            raise ValueError('incomplete rank-local completion proof set')
        if arm == 'm3v9':
            if sample.get('li_transport') != 'mte' or sample.get('li_completion_mode') != 'event' or any((p.get('trial') != sample['trial'] or p.get('completion_mode') != 'event' or p.get('oracle') is not False or (p.get('actual_LI_events') != spec.LI_OWNERS) or (p.get('expected_LI_owners') != spec.LI_OWNERS) or (p.get('joined_layers') != spec.LAYERS) for p in proofs)):
                raise ValueError('21 LI event / 78 layer completion proof rejected')
        elif any((p.get('actual_LI_events') != 0 or p.get('oracle') is not False or p.get('completion_mode') != 'sdk' for p in proofs)):
            raise ValueError('Native SDK completion evidence rejected')
        group = values.setdefault(sample['case'], {})
        identity = (sample['phase'], sample['trial_index'])
        if identity in group:
            raise ValueError('duplicate trial identity')
        group[identity] = sample
    return values

def phase(values, name):
    return [values[key] for key in sorted(values) if key[0] == name]

def summarize(values):
    trials = phase(values, 'measure')
    if len(trials) != spec.MEASURED or len(phase(values, 'warmup')) != spec.WARMUPS:
        return None
    if [t['trial_index'] for t in trials] != list(range(spec.WARMUPS, spec.WARMUPS + spec.MEASURED)):
        raise ValueError('formal repetition indices differ from manifest')
    forward = [max(metric(t, 'model_npu_ms')) for t in trials]
    fastest = [min(metric(t, 'model_npu_ms')) for t in trials]
    first = [finite(t['all_first_tokens_ms']) for t in trials]
    pre, skew, enqueue, after_enqueue = ([], [], [], [])
    for sample in trials:
        enters = metric(sample, 'model_enter_ns')
        returns = metric(sample, 'model_return_ns')
        if any((b < a for a, b in zip(enters, returns))):
            raise ValueError('host model boundary is reversed')
        pre.append((max(enters) - sample['resume_start_ns']) / 1000000.0)
        skew.append((max(enters) - min(enters)) / 1000000.0)
        enqueue.append((max(returns) - max(enters)) / 1000000.0)
        after_enqueue.append((sample['all_first_tokens_returned_ns'] - max(returns)) / 1000000.0)
    hbm = [max(metric(t, 'allocated_peak_bytes')) / 2 ** 30 for t in trials]
    result = dict(formal_repetitions=len(trials), warmup_repetitions=spec.WARMUPS, forward_median_ms=median(forward), forward_trials_ms=forward, fastest_rank_forward_trials_ms=fastest, forward_rank_spread_trials_ms=[a - b for a, b in zip(forward, fastest)], first_token_median_ms=median(first), first_token_trials_ms=first, resume_to_last_rank_model_enter_median_ms=median(pre), model_enter_rank_spread_median_ms=median(skew), last_rank_model_enter_to_last_host_return_median_ms=median(enqueue), last_host_return_to_all_first_tokens_median_ms=median(after_enqueue), frontend_add_calls_median_ms=median((finite(t['frontend_add_calls_ms']) for t in trials)), allocated_peak_GiB=max(hbm), allocated_peak_trials_GiB=hbm, geometry_validated=True, dummy_model_calls_per_rank_per_trial=[[row.get('dummy_model_forward_calls') for row in t['rank_rows']] for t in trials], dummy_model_ms_per_rank_per_trial=[[row.get('dummy_model_npu_ms') for row in t['rank_rows']] for t in trials])
    probes = phase(values, 'probe')
    if len(probes) != 1:
        raise ValueError('one independent diagnostic burst per case required')
    result['diagnostic_probe'] = None
    if probes:
        probe = probes[0]
        result['diagnostic_probe'] = dict(separate_from_formal=True, forward_max_rank_ms=max(metric(probe, 'model_npu_ms')), **{key: max(metric(probe, key)) if all((row.get(key) is not None for row in probe['rank_rows'])) else None for key in ('total_load_ms', 'exposed_load_ms', 'sfa_ms', 'ffn_ms', 'union_ms', 'sdk_load_wall_ms', 'selected_main_copy_ms')}, union_error_flags=sum((int(row['selected_union_error_flags']) for row in probe['rank_rows'])) if all((row.get('selected_union_error_flags') is not None for row in probe['rank_rows'])) else None, ffn_calls_per_rank=[sum(row['ffn_calls'].values()) if isinstance(row.get('ffn_calls'), dict) else row.get('ffn_calls') for row in probe['rank_rows']])
    return result

def compare(reference, candidate):
    if reference['input_digest'] != candidate['input_digest']:
        raise ValueError('candidate and reference inputs differ')
    left_tokens, right_tokens = (reference['first_tokens'], candidate['first_tokens'])
    if len(left_tokens) != len(right_tokens):
        raise ValueError('candidate and reference output counts differ')
    mismatch = sum((a != b for a, b in zip(left_tokens, right_tokens)))
    different_sets, compared, maximum = (0, 0, 0.0)
    finite_values = True
    for before, after in zip(reference['logprobs'], candidate['logprobs']):
        left = {int(v['token_id']): float(v['logprob']) for v in before}
        right = {int(v['token_id']): float(v['logprob']) for v in after}
        if not left or not right or len(left) != len(before) or (len(right) != len(after)):
            raise ValueError('empty or duplicate logprob records')
        different_sets += set(left) != set(right)
        finite_values &= all((math.isfinite(v) for v in (*left.values(), *right.values())))
        for token in left.keys() & right.keys():
            delta = abs(left[token] - right[token])
            if math.isfinite(delta):
                maximum = max(maximum, delta)
            compared += 1
    return dict(requests_compared=len(left_tokens), first_token_mismatches=mismatch, top_logprob_set_mismatches=different_sets, logprob_values_compared=compared, max_abs_logprob_difference=maximum if finite_values else None, finite_logprobs=finite_values, passed=bool(not mismatch and (not different_sets) and finite_values and (maximum <= TOLERANCE)))

def export(root, allow_partial=False):
    samples = {arm: load_samples(root, arm) for arm in ARMS}
    report = dict(schema='m3v15_full78_q4k_chunk6k_v1', layers=spec.LAYERS, tp=spec.TP, dp=spec.DP, ep=spec.EP, configured_devices=list(spec.DEVICES), logprob_top=spec.LOGP_TOP, logprob_tolerance=TOLERANCE, rows=[], numerical=[], requested_performance_rows=len(spec.all_cases()) * len(ARMS), timing_notes={'forward': 'Median over formal bursts of the maximum rank NPU model.forward event duration; parallel ranks are never summed.', 'first_token': "All frontend add_request calls have returned; resume to all requests' first tokens. Cold history creation and collection RPC are excluded.", 'host_segments': 'Host enqueue boundaries are not NPU completion boundaries. After-host-return includes outstanding device work; it is not pure postprocessing or CPU cost.', 'load': 'Separate diagnostic burst, each metric is maximum of per-rank totals. Different ranks and overlapping stage metrics cannot be added to formal forward.', 'HBM': 'Maximum framework allocated peak across ranks and formal repetitions; includes weights, KV, and temporary tensors; excludes allocations outside the framework allocator.', 'dummy': 'Idle-DP/EP synchronization model calls are recorded separately over bench_begin to collect. They are not real request forward calls and this wider observation window is not identical to TTFT.'})
    for case in spec.all_cases():
        name = case['case']
        for arm in ARMS:
            summary = summarize(samples.get(arm, {}).get(name, {}))
            if summary is None:
                continue
            report['rows'].append(dict(case=name, arm=LABELS[arm], H=case['H'], Q=case['request_Qs'][0], global_BS=len(case['request_Qs']), local_BS=len(case['request_Qs']) // spec.DP, total_load_scope='Main selected gather plus LI copy NPU event work' if arm == 'v0' else 'SDK H2D call wall time plus selected main-copy NPU event work' if arm == 'm3v9' else 'SDK H2D call wall time, including SDK overhead', **summary))
        references = phase(samples.get('native', {}).get(name, {}), 'measure')
        for arm in ('v0', 'm3v9'):
            candidates = phase(samples.get(arm, {}).get(name, {}), 'measure')
            if len(references) != spec.MEASURED or len(candidates) != spec.MEASURED:
                continue
            if [x['trial_index'] for x in references] != [x['trial_index'] for x in candidates]:
                raise ValueError('paired trial identities differ')
            checks = [compare(a, b) for a, b in zip(references, candidates)]
            report['numerical'].append(dict(case=name, arm=LABELS[arm], trials=checks, passed=all((c['passed'] for c in checks))))
    report['completion_evidence'] = dict(mode='event', oracle=False, rank_count=spec.DP * spec.TP, LI_events_per_rank=21, joined_layers=spec.LAYERS, validated_bursts=sum((len(values) for arm in samples.values() for values in arm.values())))
    report['completed_performance_rows'] = len(report['rows'])
    expected_numerical = len(spec.all_cases()) * (len(ARMS) - 1) if 'native' in ARMS else 0
    report['complete'] = len(report['rows']) == report['requested_performance_rows'] and len(report['numerical']) == expected_numerical
    report['numerical_passed'] = report['complete'] and all((c['passed'] for c in report['numerical'])) if report['numerical'] else None
    report['diagnostics_complete'] = report['complete'] and all((r['diagnostic_probe'] is not None for r in report['rows']))
    if not report['complete'] and (not allow_partial):
        raise ValueError('not all requested performance rows and comparisons are ready')
    atomic(root / 'public-comparison.json', report)
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = export(args.root, args.allow_partial)
    print(json.dumps({k: result[k] for k in ('complete', 'completed_performance_rows', 'numerical_passed', 'diagnostics_complete')}))
