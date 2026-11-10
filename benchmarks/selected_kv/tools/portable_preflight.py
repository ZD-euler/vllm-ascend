"""CPU-only enrollment for the recovered harness with explicit portable inputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / 'portable-settings.json').read_text()) if (ROOT / 'portable-settings.json').is_file() else {}
DEPLOY = Path(SETTINGS.get('deploy', '.'))
MODEL = Path(SETTINGS.get('model', '.'))
CORPUS = Path(SETTINGS.get('corpus', '.'))
ARMS = (SETTINGS.get('arm'),)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.next')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_identity():
    if SETTINGS.get('identity_mode') == 'source-rebuild':
        from build_receipts import verify
        receipts = SETTINGS.get('build_receipts', [])
        required = {'base_ops', 'sdk17'} | ({'selected_features'} if SETTINGS['arm'] == 'm3v9' else set())
        if {r['recipe'] for r in receipts} != required:
            raise ValueError('missing typed source build receipts')
        for receipt in receipts:
            verify(Path(receipt['path']), Path(SETTINGS['bundle_root']), receipt['recipe'],
                   receipt['arm'], Path(SETTINGS[receipt['scope']]))
    expected = SETTINGS.get('identity_files', {})
    for scope in ('source', 'core', 'sdk', 'deploy'):
        if not expected.get(scope):
            raise ValueError('recorded reference hashes required: ' + scope)
        base = Path(SETTINGS[scope]).resolve()
        for name, digest in expected[scope].items():
            target = (base / name).resolve()
            target.relative_to(base)
            if sha(target) != digest:
                raise ValueError('reference identity mismatch: ' + scope + '/' + name)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def runtime_files(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
            and p.relative_to(root).parts[0] not in ('build', 'dist', '.eggs', '.pytest_cache', '.ruff_cache')}


def verify_li_c8_metadata(model, quant_description):
    config = model.get('text_config', model)
    owners = [i for i, role in enumerate(config['indexer_types']) if role == 'full']
    expected = {f'model.layers.{i}.self_attn.indexer.quant_type': 'INT8_DYNAMIC' for i in owners}
    actual = {k: v for k, v in quant_description.items() if k.endswith('.indexer.quant_type')}
    if len(config['indexer_types']) != config['num_hidden_layers'] or not owners or actual != expected:
        raise ValueError('LI C8 tags must match every main indexer owner and exclude MTP')
    return owners


def enroll():
    import spec
    verify_identity()
    reference = Path(SETTINGS['model_reference'])
    for name in ('config.json', 'quant_model_description.json'):
        if sha(MODEL / name) != sha(reference / name):
            raise ValueError('model recipe differs from reference: ' + name)
    expected_metadata = SETTINGS.get('model_base_metadata', {})
    if not expected_metadata:
        raise ValueError('base model index/tokenizer reference hashes required')
    for name, expected in expected_metadata.items():
        if Path(name).name != name or Path(name).suffix not in ('.json', '.model', '.txt'):
            raise ValueError('only model index/tokenizer metadata filenames allowed')
        if sha(MODEL / name) != expected:
            raise ValueError('base model metadata identity changed: ' + name)
    model = json.loads((MODEL / 'config.json').read_text())
    verify_li_c8_metadata(model, json.loads((MODEL / 'quant_model_description.json').read_text()))
    if model.get('text_config', model)['num_hidden_layers'] != spec.LAYERS:
        raise ValueError('wrong model layer count')
    cfg = json.loads((ROOT / 'engine.json').read_text())
    if (cfg['tensor_parallel_size'], cfg['data_parallel_size'], cfg['max_num_batched_tokens']) != (spec.TP, spec.DP, spec.MAX_BATCHED_TOKENS):
        raise ValueError('engine geometry changed')
    # Input construction and every H/Q digest must pass before a model is instantiated.
    from run_cases import source_arrays, inputs_for_H
    manifest = json.loads(CORPUS.read_text())
    if manifest.get('synthetic_padding_tokens') is not False:
        raise ValueError('real corpus provenance required')
    reference = json.loads(Path(SETTINGS['corpus_reference']).read_text())
    def sources_identity(data):
        return {row['task_name']: (row['glm52_tokens'], row['tokens_sha256_uint32_le'])
                for row in data['sources']}
    def cases_identity(data):
        return [(row['history'], row['query'], row['total_query_tokens'], row['requests'],
                 [(w['task'], w['start_token'], w['token_sha256']) for w in row['windows']])
                for row in data['cases']]
    if sources_identity(reference) != sources_identity(manifest) or cases_identity(reference) != cases_identity(manifest):
        raise ValueError('corpus differs from recorded reference sources/windows')
    arrays = source_arrays(manifest)
    for H in {c['H'] for c in spec.all_cases()}:
        inputs_for_H(manifest, arrays, H, spec.all_cases())
    lock = {name: runtime_files(Path(SETTINGS[name])) for name in ('source', 'core', 'sdk', 'deploy')}
    lock['harness'] = runtime_files(ROOT / 'harness')
    lock['model'] = {p.name: sha(p) for p in MODEL.glob('*.json')}
    lock['corpus'] = sha(CORPUS)
    lock['corpus_reference'] = sha(Path(SETTINGS['corpus_reference']))
    lock['engine'] = sha(ROOT / 'engine.json')
    (ROOT / 'runtime-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    return {'status': 'enrolled', 'arm': SETTINGS['arm']}


def check(arm):
    verify_identity()
    if arm not in ARMS:
        raise ValueError('arm mismatch')
    lock = json.loads((ROOT / 'runtime-lock.json').read_text())
    for name in ('source', 'core', 'sdk', 'deploy'):
        if runtime_files(Path(SETTINGS[name])) != lock[name]:
            raise ValueError('runtime changed: ' + name)
    if runtime_files(ROOT / 'harness') != lock['harness']:
        raise ValueError('harness changed')
    for name, expected in lock['model'].items():
        if sha(MODEL / name) != expected:
            raise ValueError('model metadata changed')
    if sha(Path(SETTINGS['corpus_reference'])) != lock['corpus_reference']:
        raise ValueError('corpus reference changed')
    if sha(CORPUS) != lock['corpus'] or sha(ROOT / 'engine.json') != lock['engine']:
        raise ValueError('corpus or engine changed')
    return dict(SETTINGS, config=str(ROOT / 'engine.json'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('enroll', 'check'))
    args = parser.parse_args()
    print(json.dumps(enroll() if args.action == 'enroll' else check(SETTINGS['arm']), indent=2))
