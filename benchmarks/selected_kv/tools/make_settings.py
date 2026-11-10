#!/usr/bin/env python3
"""Bind local paths to supplied reference identities; never enroll unknown hashes."""
import argparse
import json
from build_receipts import allowed, verify
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('native', 'm3v9'), default='m3v9')
    parser.add_argument('--layers', choices=('6', '78'), required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--identity-mode', choices=('historical-runtime', 'source-rebuild'), default='source-rebuild')
    parser.add_argument('--base-receipt', type=Path)
    parser.add_argument('--sdk-receipt', type=Path)
    parser.add_argument('--feature-receipt', type=Path)
    parser.add_argument('--source', type=Path, help='M3 default: enclosing repository; required for native')
    for name in ('model', 'core', 'sdk', 'cann', 'corpus', 'token-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.source is None:
        if args.arm == 'native':
            parser.error('--source is required for the optional native baseline')
        args.source = HERE.parents[1]
    elif not args.source.is_absolute():
        args.source = HERE.parents[1] / args.source
    locks = json.loads((HERE / 'references/source-identities.json').read_text())
    sdk_lock = HERE / 'references/sdk-runtime-identities.json'
    if not sdk_lock.is_file():
        raise ValueError('reference SDK runtime lock not supplied; cannot bless current machine')
    env = json.loads((HERE / 'references/environment-observed.json').read_text())
    model_reference = HERE / 'references' / ('model' + args.layers)
    import hashlib
    metadata = {r['name']: r['sha256'] for r in env['model_metadata_hashes']
                if r['name'].startswith('tokenizer')}
    for path in model_reference.glob('*index.json'):
        metadata[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(name.endswith('index.json') for name in metadata):
        raise ValueError('recorded model index metadata missing')
    settings = {name: str(getattr(args, name).resolve(strict=True))
                for name in ('model', 'source', 'core', 'sdk', 'cann', 'corpus')}
    settings.update(arm=args.arm, session_id=args.session_id, token_root=str(args.token_root.resolve(strict=True)),
        revision={'m3v9': 'db85ccd37d8fe90b9835903355f3bc92c1220e85',
                  'native': 'fbb4d867d22095047473e3c7ca9db053f5f4c7b0'}[args.arm],
        model_reference=str(model_reference), model_base_metadata=metadata,
        full_model_index_sha256=hashlib.sha256((HERE / 'references/model78/quant_model_weights.safetensors.index.json').read_bytes()).hexdigest(),
        deploy=str(HERE / 'recovered'), corpus_reference=str(HERE / 'references/corpus-manifest.json'),
        identity_files={'source': locks[args.arm], 'core': locks['core'],
                        'sdk': json.loads(sdk_lock.read_text()), 'deploy': locks['deploy']})
    settings['bundle_root'] = str(HERE)
    settings['identity_mode'] = args.identity_mode
    settings['build_receipts'] = []
    if args.identity_mode == 'source-rebuild':
        required = [('base_ops', args.base_receipt, args.arm, args.source.resolve(), 'source'),
                    ('sdk17', args.sdk_receipt, 'shared', args.sdk.resolve(), 'sdk')]
        if args.arm == 'm3v9':
            required.append(('selected_features', args.feature_receipt, 'm3v9', args.source.resolve(), 'source'))
        for recipe, receipt, arm, runtime, scope in required:
            if receipt is None:
                raise ValueError('source-rebuild requires receipt: ' + recipe)
            outputs = verify(receipt, HERE, recipe, arm, runtime)
            settings['identity_files'][scope] = {k: v for k, v in settings['identity_files'][scope].items()
                                                 if not allowed(recipe, k)}
            settings['identity_files'][scope].update(outputs)
            settings['build_receipts'].append({'recipe': recipe, 'arm': arm, 'scope': scope,
                                                'path': str(receipt.resolve(strict=True))})
    args.output.write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    main()
