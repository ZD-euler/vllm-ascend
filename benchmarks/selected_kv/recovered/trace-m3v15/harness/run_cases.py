"""Five requested geometries, identical inputs across the three source arms."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import inspect
from itertools import chain
import json
import os
from pathlib import Path
import time
import numpy as np
import spec
from perf_common import atomic

async def maybe(value):
    return await value if inspect.isawaitable(value) else value

def check_time():
    if time.time() >= spec.STOP_NEW:
        raise TimeoutError('finite experiment admission window ended')

def digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<u4').tobytes()).hexdigest()

def source_arrays(manifest):
    cache = {}
    for source in manifest['sources']:
        path = Path(source['token_file'])
        if not path.is_absolute():
            path = Path('/repro/reference-root/real_coding') / path
        raw = np.load(path, allow_pickle=False) if path.suffix == '.npy' else json.loads(path.read_text())
        raw = np.asarray(raw)
        if raw.ndim != 1 or len(raw) != source['glm52_tokens']:
            raise ValueError('original real coding corpus length changed')
        if digest(raw) != source['tokens_sha256_uint32_le']:
            raise ValueError('original real coding corpus digest changed')
        cache[source['task_name']] = raw
    return cache

def _inputs_for_full_history(manifest, arrays, H, selected):
    base = max((c for c in manifest['cases'] if c['history'] == H and c['total_query_tokens'] == 2048), key=lambda c: c['requests'])
    needed = max((len(c['request_Qs']) for c in selected if c['H'] == H))
    if len(base['windows']) < needed:
        raise ValueError('the real source has too few independent windows')
    histories, followups, history_proof = ([], {}, [])
    for ordinal, window in enumerate(base['windows'][:needed]):
        values = arrays[window['task']]
        start = window['start_token']
        original = values[start:start + H + base['query']]
        if digest(original) != window['token_sha256']:
            raise ValueError('base real window digest changed')
        history = original[:H].tolist()
        histories.append(history)
        history_proof.append(digest(history))
        seen = set()
        for c in (c for c in selected if c['H'] == H and ordinal < len(c['request_Qs'])):
            Q = c['request_Qs'][0]
            chunks = []
            for offset in chain(range(start + H, len(values) - Q + 1, 128), range(0, start + H - Q + 1, 128)):
                fragment = values[offset:offset + Q]
                key = digest(fragment[:128])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(fragment.tolist())
                if len(chunks) == 13:
                    break
            if len(chunks) != 13:
                raise ValueError('insufficient genuine distinct followups')
            followups.setdefault(c['case'], []).append(chunks)
    if len(set((digest(h[:128]) for h in histories))) != needed:
        raise ValueError('histories share their first cache block')
    return (histories, followups, history_proof)

def inputs_for_H(manifest, arrays, H, selected):
    return _inputs_for_full_history(manifest, arrays, H, selected)

async def first(collector, started):
    while True:
        output = await maybe(collector.get())
        if isinstance(output, BaseException):
            raise output
        if output is not None and output.outputs and output.outputs[0].token_ids:
            item = output.outputs[0]
            row = dict(ms=(time.monotonic_ns() - started) / 1000000.0, token_id=int(item.token_ids[0]), finished=bool(output.finished), logprobs=[])
            if item.logprobs:
                row['logprobs'] = [{'token_id': int(token), 'logprob': float(value.logprob)} for token, value in sorted(item.logprobs[0].items())]
            return row
        if output is not None and output.finished:
            raise ValueError('request finished without a generated token')

def validate_profile_acks(values, trial, action):
    expected = {(d, t) for d in range(spec.DP) for t in range(spec.TP)}
    if len(values) != spec.DP * spec.TP or {(v['dp_rank'], v['tp_rank']) for v in values} != expected:
        raise ValueError('profile RPC requires all 16 unique ranks')
    if any((v['trial'] != trial or v['action'] != action or v['status'] != ('ack' if (v['dp_rank'], v['tp_rank']) == (0, 0) else 'skipped') for v in values)):
        raise ValueError('profile RPC acknowledgement identity mismatch')
    if action == 'stop' and any((not v['stop_after_existing_save_fence'] or v['fence_completions'] < 1 or v['parsed_in_worker'] or (v.get('sender_unfinished_tasks') != 0) or (v.get('pending_last_save_drain_after') is not False) or (v.get('full_D2H_completion_proven') is not True) for v in values)):
        raise ValueError('profile stop boundary rejected')
    if any((v.get('global_rank') != v['dp_rank'] * 4 + v['tp_rank'] or v.get('profiler_selected') != ((v['dp_rank'], v['tp_rank']) == (0, 0)) or v.get('profiler_skipped') != ((v['dp_rank'], v['tp_rank']) != (0, 0)) for v in values)):
        raise ValueError('invalid rank0 profiler selection ACK')
    if any((v.get('trace_rank_dir') is not None or (action == 'stop' and v.get('raw_capture_dirs')) for v in values if not v['profiler_selected'])):
        raise ValueError('skipped rank unexpectedly retained profiler data')
    return values

async def profile_acks(root, trial, action):
    until = time.monotonic() + 120
    while True:
        paths = list((root / "workers").glob(f"profile-{action}-{trial}-*.json"))
        if len(paths) == spec.DP * spec.TP:
            return validate_profile_acks([json.loads(p.read_text()) for p in paths], trial, action)
        if time.monotonic() >= until:
            raise TimeoutError("missing rank-local profiler acknowledgement")
        await asyncio.sleep(.01)

async def burst(engine, params, root, case_value, phase, number, histories, followups, mode):
    check_time()
    trial = f"{case_value['case']}-{phase}-{number}"
    await maybe(engine.pause_generation(mode='keep', wait_for_inflight_requests=True, clear_cache=False))
    await engine.collective_rpc('bench_set_li_completion', timeout=300, args=(mode, phase == 'oracle'))
    await engine.collective_rpc('bench_begin', timeout=180, args=(case_value, phase, trial))
    if phase == "profile":
        await engine.collective_rpc("bench_profile_start", timeout=300, args=(trial,))
        await profile_acks(root, trial, "start")
    add_begin = time.monotonic_ns()
    collectors = []
    prompts = []
    prefix = f"uc-h{case_value['H']}-" if case_value.get('continuation') else ''
    for i in range(len(case_value['request_Qs'])):
        prompt = histories[i] + followups[i][number]
        prompts.append(prompt)
        collectors.append(await maybe(engine.add_request(f'{prefix}{trial}-{i}', dict(prompt_token_ids=prompt), params, data_parallel_rank=i % spec.DP)))
    started = time.monotonic_ns()
    tasks = [asyncio.create_task(first(collector, started)) for collector in collectors]
    await maybe(engine.resume_generation())
    outputs = await asyncio.wait_for(asyncio.gather(*tasks), min(1200, max(1, spec.DEADLINE - time.time())))
    ended = time.monotonic_ns()
    if not all((item['finished'] for item in outputs)):
        raise ValueError('all single-token requests must finish')
    if phase == "profile":
        await engine.collective_rpc("bench_profile_stop", timeout=600, args=(trial,))
        await profile_acks(root, trial, "stop")
    await maybe(engine.pause_generation(mode='keep', wait_for_inflight_requests=True, clear_cache=False))
    await engine.collective_rpc('bench_collect', timeout=300, args=(trial,))
    await engine.collective_rpc('bench_finish_li_completion', timeout=300, args=(phase == 'oracle',))
    proof_until = time.monotonic() + 60
    while True:
        proofs = list((root / 'workers').glob(f'completion-{trial}-*.json'))
        if len(proofs) == spec.DP * spec.TP:
            break
        if time.monotonic() >= proof_until:
            raise TimeoutError('Missing rank-local completion proof')
        await asyncio.sleep(0.01)
    transport = [json.loads(p.read_text()) for p in proofs]
    if {(v['dp_rank'], v['tp_rank']) for v in transport} != {(d, t) for d in range(spec.DP) for t in range(spec.TP)} or any((v['trial'] != trial or v['completion_mode'] != mode for v in transport)):
        raise ValueError('Rank-local completion proofs do not match this burst')
    if mode == 'event':
        if any((v.get('actual_LI_events') != spec.LI_OWNERS or v.get('expected_LI_owners') != spec.LI_OWNERS or v.get('joined_layers') != spec.LAYERS or (v.get('oracle') is not False) for v in transport)):
            raise ValueError('All 16 ranks must prove 21 LI events and 78 joined layers')
    elif any((p.get('actual_LI_events') != 0 or p.get('oracle') is not False or p.get('completion_mode') != 'sdk' for p in transport)):
        raise ValueError('Native SDK completion evidence rejected')
    rows = [json.loads(p.read_text()) for p in (root / 'workers').glob(f'metric-{trial}-*.json')]
    if len(rows) != spec.DP * spec.TP or len({(r['dp_rank'], r['tp_rank']) for r in rows}) != spec.DP * spec.TP:
        raise ValueError('four unique rank rows are required')
    if any((not r['valid'] for r in rows)):
        if phase == 'warmup':
            await burst(engine, params, root, case_value, 'diagnostic', 12, histories, followups, mode)
        raise AssertionError('rank validation failed; per-step evidence retained')
    value = dict(li_transport='mte' if mode == 'event' else 'sdk', li_completion_mode=mode, oracle=transport if phase == 'oracle' else None, case=case_value['case'], title=case_value['title'], H=case_value['H'], request_Qs=case_value['request_Qs'], global_BS=len(case_value['request_Qs']), continuation=case_value.get('continuation', False), trial=trial, phase=phase, trial_index=number, ttft_ms=[item['ms'] for item in outputs], first_tokens=[item['token_id'] for item in outputs], logprobs=[item['logprobs'] for item in outputs], all_first_tokens_ms=(ended - started) / 1000000.0, frontend_add_calls_ms=(started - add_begin) / 1000000.0, resume_start_ns=started, all_first_tokens_returned_ns=ended, rank_rows=rows, input_digest=digest([token for prompt in prompts for token in prompt]))
    atomic(root / 'samples' / f'{trial}.json', value, private=True)
    return value

async def seed(engine, params, histories):
    for begin in range(0, len(histories), spec.DP):
        await maybe(engine.resume_generation())

        async def one(i):
            async for output in engine.generate(dict(prompt_token_ids=histories[i]), params, f'seed-h{len(histories[i])}-{i}', data_parallel_rank=i % spec.DP):
                if output.finished:
                    break
        await asyncio.gather(*(one(i) for i in range(begin, min(begin + spec.DP, len(histories)))))

async def main(args):
    from c8_bootstrap import apply_c8_frontend_schema
    apply_c8_frontend_schema()
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    root = args.output
    for name in ('workers', 'samples'):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(BENCH_OUTPUT=str(root), BENCH_ADMISSION=str(args.admission))
    cfg = json.loads(args.engine_args.read_text())
    if (cfg['tensor_parallel_size'], cfg['data_parallel_size'], cfg['enable_expert_parallel'], cfg['enforce_eager']) != (spec.TP, spec.DP, True, True):
        raise ValueError('TP4 DP4 EP16 eager is required')
    model = json.loads((Path(cfg['model']) / 'config.json').read_text())
    if model.get('text_config', model)['num_hidden_layers'] != spec.LAYERS:
        raise ValueError('78 actual layers are required')
    if cfg.get('kv_transfer_config'):
        cfg['kv_transfer_config'] = KVTransferConfig(**cfg['kv_transfer_config'])
    cfg['worker_cls'] = 'worker_entry.BenchWorker'
    cfg['enable_prefix_caching'] = True
    cfg['disable_log_stats'] = True
    manifest = json.loads(args.corpus.read_text())
    if manifest.get('synthetic_padding_tokens') is not False:
        raise ValueError('original real coding inputs are required')
    arrays = source_arrays(manifest)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False, seed=0, logprobs=spec.LOGP_TOP)
    seed_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False, seed=0)
    selected_cases = spec.all_cases()
    if args.include:
        wanted = set(args.include.split(','))
        selected_cases = [c for c in selected_cases if c['case'] in wanted]
        if not selected_cases:
            raise ValueError('empty case selection')
    prepared_inputs = {H: inputs_for_H(manifest, arrays, H, selected_cases)
                       for H in sorted({c["H"] for c in selected_cases})}
    engine = await maybe(AsyncLLM.from_engine_args(AsyncEngineArgs(**cfg)))
    completed = []
    try:
        for H in sorted({c['H'] for c in selected_cases}):
            (histories, followups, proofs) = prepared_inputs[H]
            atomic(root / f'private-inputs-{H}.json', dict(history_digests=proofs), private=True)
            seed_histories = [history + [followups[selected_cases[0]['case']][i][0][0]] for i, history in enumerate(histories)]
            await seed(engine, seed_params, seed_histories)
            await asyncio.sleep(31)
            for case_value in (c for c in selected_cases if c['H'] == H):
                count = len(case_value['request_Qs'])
                mode = 'sdk' if args.arm == 'native' else 'event'
                protocol = [("warmup", "event"), ("profile", "event")]
                for number, (phase, mode) in enumerate(protocol):
                    value = await burst(engine, params, root, case_value, phase, number, histories[:count], followups[case_value['case']], mode)
                    if phase == 'oracle':
                        atomic(root / f"private-oracle-{case_value['case']}.json", value['oracle'], private=True)
                    atomic(root / 'public-progress.json', dict(status=phase, mode=mode, case=case_value['case'], number=number, completed_cases=completed))
                completed.append(case_value['case'])
                atomic(root / 'public-progress.json', dict(status='case_complete', case=case_value['case'], completed_cases=completed))
        atomic(root / 'public-progress.json', dict(status='complete', completed_cases=completed))
    finally:
        await maybe(engine.pause_generation(mode='keep', wait_for_inflight_requests=True, clear_cache=False))
        await maybe(engine.shutdown())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ('output', 'engine-args', 'corpus', 'admission'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--arm', choices=('native', 'v0', 'm3v9'), required=True)
    parser.add_argument('--include', default='')
    asyncio.run(main(parser.parse_args()))
