import pathlib, sys, json, datetime
import argparse
p = argparse.ArgumentParser()
p.add_argument('--native-root', type=pathlib.Path, required=True)
p.add_argument('--m3-root', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
args = p.parse_args()

def require_clean_exit(root, arm):
    result = json.loads((root / 'runs' / arm / 'public-result.json').read_text())
    launch = json.loads((root / 'public-launch-result.json').read_text())
    if (result.get('status') != 'completed' or result.get('client_returncode') != 0
            or result.get('remaining_owned') != 0 or launch.get('remaining_owned') != 0
            or launch.get('controller_exit_code') != 0 or launch.get('cleanup_error') is not None or not launch.get('NPU_admitted')):
        raise ValueError('run lacks successful owned exit: ' + str(root))

require_clean_exit(args.native_root, 'native')
require_clean_exit(args.m3_root, 'm3v9')
sys.path.insert(0, str(args.native_root / 'harness'))
import export_compare as e, spec
for root, arm, revision in [(args.native_root, 'native', 'fbb4d867d22095047473e3c7ca9db053f5f4c7b0'),
                             (args.m3_root, 'm3v9', 'db85ccd37d8fe90b9835903355f3bc92c1220e85')]:
    settings = json.loads((root / 'portable-settings.json').read_text())
    if settings['arm'] != arm or settings['revision'] != revision:
        raise ValueError('trace source/arm identity mismatch')
case = spec.all_cases()[0]['case']
left = e.load_samples(args.native_root, 'native')[case]
right = e.load_samples(args.m3_root, 'm3v9')[case]
required = {('warmup', 0), ('profile', 1)}
if not set(left) == set(right) == required:
    raise ValueError('trace contract validation failed')
checks = [dict(phase=k[0], index=k[1], **e.compare(left[k], right[k])) for k in sorted(required)]
summaries = {}
for (label, values) in (('native', left), ('M3v15', right)):
    s = values['profile', 1]
    rows = s['rank_rows']
    summaries[label] = {'profile_forward_rank0_ms': next((v['model_npu_ms'] for v in rows if v['dp_rank'] == v['tp_rank'] == 0)), 'profile_forward_maxrank_ms': max((v['model_npu_ms'] for v in rows)), 'profile_all_first_tokens_ms': s['all_first_tokens_ms'], 'profile_allocated_peak_GiB': max((v['allocated_peak_bytes'] for v in rows)) / 2 ** 30, 'real_model_forward_calls_per_rank': [v['model_forward_calls'] for v in rows], 'wider_observation_dummy_calls_per_rank': [v.get('dummy_model_forward_calls') for v in rows]}
out = {'status': 'PASS' if all((c['passed'] for c in checks)) else 'FAIL', 'H': 524288, 'Q': 4096, 'layers': 78, 'TP': 4, 'DP': 4, 'EP': 16, 'global_BS': 4, 'local_BS': 1, 'max_num_batched_tokens': 6144, 'warmup_bursts': 1, 'profile_bursts': 1, 'profiled_ranks': [0], 'source_versions': {'native': 'fbb4d867d22095047473e3c7ca9db053f5f4c7b0', 'M3v15': 'db85ccd37d8fe90b9835903355f3bc92c1220e85'}, 'numerical': checks, 'requests_compared': sum((x['requests_compared'] for x in checks)), 'logprob_values_compared': sum((x['logprob_values_compared'] for x in checks)), 'max_abs_logprob_difference': max((x['max_abs_logprob_difference'] for x in checks)), 'profile_diagnostics': summaries, 'timing_scope': 'One diagnostic profile per source after one warmup, with profiler overhead. Parallel ranks are not summed. Dummy counters cover bench_begin to collect, which is wider than the profiler active interval.', 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
out['logprob_tolerance'] = e.TOLERANCE
out['exact_match'] = all(c['passed'] and c['max_abs_logprob_difference'] == 0.0 for c in checks)
args.output.write_text(json.dumps(out, indent=2) + '\n')
for phase in (args.native_root, args.m3_root):
    (phase / 'public-numeric-vs-other-trace.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'status': out['status'], 'requests': out['requests_compared'], 'logprob_values': out['logprob_values_compared'], 'max_abs_logprob_difference': out['max_abs_logprob_difference']}))

raise SystemExit(0 if out['status'] == 'PASS' else 1)
