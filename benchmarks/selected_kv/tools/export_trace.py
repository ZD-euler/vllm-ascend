import pathlib, sys, json, datetime, hashlib

def require_clean_exit(root, arm):
    result = json.loads((root / 'runs' / arm / 'public-result.json').read_text())
    launch = json.loads((root / 'public-launch-result.json').read_text())
    if (result.get('status') != 'completed' or result.get('client_returncode') != 0
            or result.get('remaining_owned') != 0 or launch.get('remaining_owned') != 0
            or launch.get('controller_exit_code') != 0 or launch.get('cleanup_error') is not None or not launch.get('NPU_admitted')):
        raise ValueError('run lacks successful owned exit: ' + str(root))

import argparse
parser = argparse.ArgumentParser(description='Validate and export one recovered rank-0 trace capture.')
parser.add_argument('root', type=pathlib.Path)
parser.add_argument('arm', nargs='?', default='m3v9', choices=('native', 'm3v9'))
args = parser.parse_args()
root = args.root
arm = args.arm
require_clean_exit(root, arm)
sys.path.insert(0, str(root / 'harness'))
import export_compare as e, run_cases, spec, preflight
case = spec.all_cases()[0]
groups = e.load_samples(root, arm)[case['case']]
if not set(groups) == {('warmup', 0), ('profile', 1)}:
    raise ValueError('trace contract validation failed')
sample = groups['profile', 1]
trial = sample['trial']
rows = sample['rank_rows']
if not (len(rows) == 16 and all((v['valid'] and v['model_forward_calls'] == 1 and (v['actual_H'] == [524288]) and (v['actual_Q'] == [4096]) for v in rows))):
    raise ValueError('trace contract validation failed')
starts = [json.loads(p.read_text()) for p in (root / 'runs' / arm / 'client/workers').glob('profile-start-' + trial + '-*.json')]
stops = [json.loads(p.read_text()) for p in (root / 'runs' / arm / 'client/workers').glob('profile-stop-' + trial + '-*.json')]
run_cases.validate_profile_acks(starts, trial, 'start')
run_cases.validate_profile_acks(stops, trial, 'stop')
selected = [v for v in stops if v.get('profiler_selected') is True]
if not (len(selected) == 1 and selected[0]['dp_rank'] == 0 and (selected[0]['tp_rank'] == 0)):
    raise ValueError('trace contract validation failed')
raw = list((root / 'runs' / arm / 'trace').rglob('PROF_*'))
raw = [p for p in raw if p.is_dir()]
if not len(raw) == 1:
    raise ValueError('trace contract validation failed')
profile = preflight.check(arm)
rk = next((v for v in rows if v['dp_rank'] == v['tp_rank'] == 0))
out = {'status': 'NPU_capture_complete_pending_offline_device_review', 'version': 'M3v15' if arm == 'm3v9' else 'native-layerwise', 'source_revision': profile['revision'], 'H': 524288, 'Q': 4096, 'request_Qs': [4096] * 4, 'layers': 78, 'TP': 4, 'DP': 4, 'EP': 16, 'local_BS': 1, 'global_BS': 4, 'max_num_batched_tokens': 6144, 'warmup_bursts': 1, 'profile_bursts': 1, 'profiled_global_ranks': [0], 'start_ACKs': len(starts), 'stop_ACKs': len(stops), 'selected_profile_ranks': len(selected), 'skipped_ranks': sum((v.get('profiler_skipped') is True for v in stops)), 'rank_rows_validated': 32, 'single_real_model_forward_per_worker': True, 'profile_forward_rank0_ms': rk['model_npu_ms'], 'profile_forward_maxrank_ms': max((v['model_npu_ms'] for v in rows)), 'profile_all_first_tokens_ms': sample['all_first_tokens_ms'], 'profile_allocated_peak_GiB': max((v['allocated_peak_bytes'] for v in rows)) / 2 ** 30, 'target_counts': next((v for v in starts if v.get('profiler_selected') is True)).get('target_counts'), 'MSTX_counts': selected[0].get('marker_counts'), 'timing_scope': 'Single diagnostic profile burst after one warmup; profiler overhead is included. Parallel rank durations are never summed. Cold history is excluded.', 'numerical_comparison': 'pending_other_trace_arm', 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
preflight.atomic(root / 'public-Q4K-trace-capture-summary.json', out)
print(json.dumps(out))
