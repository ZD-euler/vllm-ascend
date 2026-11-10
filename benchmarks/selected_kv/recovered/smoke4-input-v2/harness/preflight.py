"""CPU-only profile/config enrollment for the three source arms."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import spec
from pathlib import Path
BASE = Path('/repro/reference-root')
ROOT = Path('/repro/reference-root/m3v15_q4k_chunk6k_20261107/smoke4-input-v2')
MODEL = Path('/repro/reference-root/m2_close_perf_20261027/model6-c8-owned-v1')
CORPUS = BASE / '22layer_t1k_20261021/corpus_dp2_t2048_v1/manifest.json'
SDK = BASE / 'matrix42_15cases_20261031/sdk17-site'
CORE_NATIVE = BASE / 'matrix42_15cases_20261031/core-native'
CORE_V0 = BASE / 'm2_parity_20261027/portable-core-v0'
NATIVE = BASE / 'native_fbb4d867_source_20261019'
V0 = BASE / 'overnight_v0_m2_20261027/v0_li_v2/source'
M3V9 = Path('/repro/reference-root/m3v15_q4k_chunk6k_20261107/source')
M3_REVISION = (ROOT / 'SOURCE_COMMIT').read_text().strip()
TEMPLATE = BASE / 'bench22_oneclick_v1_2_20261028/private-profile.json'
ARMS = ('m3v9', 'native')

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.next')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def runtime_files(root: Path):
    files = {}
    for path in root.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or '.git' in path.parts:
            continue
        if path.suffix in ('.py', '.so', '.json', '.conf', '.o', '.bin') or '.so.' in path.name:
            files[str(path.relative_to(root))] = sha(path)
    return files

def base_configs():
    cfg = json.loads((BASE / 'li_event_handoff_quick8_20261105/engine-m3v9.json').read_text())
    return {'native': {'config': cfg}}

def common(cfg, arm):
    cfg = copy.deepcopy(cfg)
    cfg.pop('kv_cache_memory_bytes', None)
    cfg.update(model=str(MODEL), tensor_parallel_size=spec.TP, data_parallel_size=spec.DP, enable_expert_parallel=True, enforce_eager=True, async_scheduling=False, max_model_len=spec.MAX_MODEL_LEN, max_num_batched_tokens=spec.MAX_BATCHED_TOKENS, max_num_seqs=1, num_gpu_blocks_override=spec.BLOCKS, gpu_memory_utilization=0.88, enable_prefix_caching=True, disable_log_stats=True)
    additional = cfg.setdefault('additional_config', {})
    additional['ascend_log_path'] = str(ROOT / 'runs' / arm / 'logs')
    additional['runtime_config_path'] = str(ROOT / 'runtime/config/runtime_config.json')
    additional['runtime_report_dir'] = str(ROOT / 'runtime/report')
    additional['runtime_dump_dir'] = str(ROOT / 'runtime/dump')
    return cfg

def build(arm: str):
    templates = base_configs()
    if arm == 'v0':
        cfg = common(templates['v0_li']['config'], arm)
        cfg['kv_transfer_config'] = None
        sparse = cfg.setdefault('additional_config', {}).setdefault('sparse_prefill_config', {})
        sparse.update(enabled=True, request_layer_enabled=True, request_layer_device_plan=True, dram_size_per_dp_GB=32, request_layer_pool_tokens=5242880, indexer_layerwise_buffers=2, arena_tokens=131072, stage_tokens=16384)
    else:
        cfg = common(templates['native']['config'], arm)
        transfer = copy.deepcopy(cfg['kv_transfer_config'])
        transfer['engine_id'] = 'm3cmp-' + arm
        extra = transfer.setdefault('kv_connector_extra_config', {})
        extra.update(backend='memcache', use_layerwise=True, load_async=False)
        extra['layerwise_num_shared_buffers'] = 2
        extra['layerwise_independent_layers'] = []
        cfg['kv_transfer_config'] = transfer
        additional = cfg.setdefault('additional_config', {})
        additional.pop('sparse_prefill_config', None)
        if arm == 'm3v9':
            additional['selected_kv_config'] = {'enabled': True}
    if arm == 'native':
        cfg.setdefault('additional_config', {}).pop('selected_kv_config', None)
    cfg['enable_chunked_prefill'] = True
    return cfg

def verify_m3_source():
    marker = ROOT / 'SOURCE_COMMIT'
    if marker.read_text().strip() != M3_REVISION:
        raise ValueError('M3v15 source marker mismatch')
    delta = json.loads((ROOT.parent / 'public-source-delta.json').read_text())
    if delta['code_commit'] != M3_REVISION:
        raise ValueError('M3v15 runtime overlay revision changed')
    for name, proof in delta['production_files'].items():
        if sha(M3V9 / name) != proof:
            raise ValueError('M3v15 frozen runtime file changed: ' + name)
    return dict(status='frozen-files', revision=M3_REVISION, revision_scope='nine-runtime-production-files-only', full_git_checkout_match_claimed=False, production_sha256=delta['production_files'], version='M3v15')

def verify_li_c8_metadata(model, quant_description):
    """Reject incomplete or extra LI C8 owner tags before NPU admission."""
    config = model.get('text_config', model)
    owners = [i for i, role in enumerate(config['indexer_types']) if role == 'full']
    expected = {f'model.layers.{i}.self_attn.indexer.quant_type': 'INT8_DYNAMIC' for i in owners}
    actual = {key: value for key, value in quant_description.items() if key.endswith('.indexer.quant_type')}
    if len(config['indexer_types']) != config['num_hidden_layers'] or not owners or actual != expected:
        raise ValueError('LI C8 tags must match every main indexer owner and exclude MTP')
    return owners

def enroll():
    model = json.loads((MODEL / 'config.json').read_text())
    verify_li_c8_metadata(model, json.loads((MODEL / 'quant_model_description.json').read_text()))
    if model.get('text_config', model).get('num_hidden_layers') != spec.LAYERS:
        raise ValueError('model must expose 78 actual layers')
    if not CORPUS.is_file() or not SDK.is_dir():
        raise ValueError('corpus or SDK is missing')
    for path in (CORE_NATIVE, M3V9):
        if not path.is_dir():
            raise ValueError('missing source/core root: ' + str(path))
    m3_proof = verify_m3_source()
    profiles = {}
    for arm in ARMS:
        source = M3V9 if arm == 'm3v9' else NATIVE if arm == 'native' else V0
        core = CORE_NATIVE
        cfg = build(arm)
        config_path = ROOT / f'engine-{arm}.json'
        atomic(config_path, cfg)
        profiles[arm] = dict(arm=arm, source=str(source), core=str(core), sdk=str(SDK), model=str(MODEL), corpus=str(CORPUS), config=str(config_path), config_sha256=sha(config_path), source_files=runtime_files(source), core_files=runtime_files(core), sdk_files=runtime_files(SDK), model_files={p.name: sha(p) for p in MODEL.glob('*.json')}, corpus_sha256=sha(CORPUS), deployment_source_proof=m3_proof, revision=M3_REVISION if arm == 'm3v9' else 'fbb4d867d22095047473e3c7ca9db053f5f4c7b0' if arm == 'native' else 'ed01efa4961b7e724ed1f9d25636cbba7c645857')
    lock = dict(status='enrolled', arms=profiles, m3v9=m3_proof, devices=list(spec.DEVICES), layers=spec.LAYERS, tp=spec.TP, dp=spec.DP, ep=spec.EP, blocks=spec.BLOCKS, model_sha256=profiles['m3v9']['model_files'], harness_files=runtime_files(ROOT / 'harness'), handoff_sha256=sha(BASE / 'matrix42_15cases_20261031/deploy/early_store_handoff.py'))
    atomic(ROOT / 'private-profile.json', lock)
    return dict(status='PASS', arms=list(profiles), m3v9=m3_proof, devices=lock['devices'])

def check(arm):
    lock = json.loads((ROOT / 'private-profile.json').read_text())
    profile = lock['arms'][arm]
    if sha(BASE / 'matrix42_15cases_20261031/deploy/early_store_handoff.py') != lock['handoff_sha256']:
        raise ValueError('early SDK handoff changed')
    if runtime_files(ROOT / 'harness') != lock['harness_files']:
        raise ValueError('frozen harness changed')
    for base, files in ((Path(profile['source']), profile['source_files']), (Path(profile['core']), profile['core_files']), (Path(profile['sdk']), profile['sdk_files'])):
        for name, expected in files.items():
            if sha(base / name) != expected:
                raise ValueError(f'frozen {arm} runtime changed: {base / name}')
    if sha(Path(profile['config'])) != profile['config_sha256']:
        raise ValueError(f'frozen {arm} config changed')
    if sha(Path(profile['corpus'])) != profile['corpus_sha256']:
        raise ValueError('corpus changed')
    for name, expected in profile['model_files'].items():
        if sha(MODEL / name) != expected:
            raise ValueError('model metadata changed')
    if arm == 'm3v9':
        verify_m3_source()
    return profile
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('enroll', 'check'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'enroll':
        print(json.dumps(enroll(), indent=2))
    else:
        print(json.dumps(check(args.arm), indent=2))
