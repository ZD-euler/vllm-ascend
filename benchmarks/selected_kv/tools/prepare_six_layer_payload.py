#!/usr/bin/env python3
"""Select exact reference six-layer tensors from user-owned full safetensors.

CPU only; preserves dtype and tensor values, verifies reference per-tensor raw
byte SHA256, and writes the reference tensor-name/shard partition. No casts.
Never run this script inside the public source bundle output directory.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--selected-model', type=Path)
    parser.add_argument('--verify-limit', type=int, default=0)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    full = args.full_model.resolve(strict=True)
    output = args.output.resolve() if args.output else None
    if not args.verify_only and output is None:
        raise ValueError('--output required for payload generation')
    if output is not None and output.is_relative_to(HERE):
        raise ValueError('weight output must stay outside the code bundle')
    reference = HERE / 'references/model6'
    selected_index = json.loads((reference / 'model.safetensors.index.json').read_text())
    full_index = json.loads((full / 'quant_model_weights.safetensors.index.json').read_text())
    selected = selected_index['weight_map']
    original = full_index['weight_map']
    expected = json.loads((HERE / 'references/model6-tensor-hashes.json').read_text())['tensor_sha256']
    if set(selected) - set(original) or set(selected) != set(expected):
        raise ValueError('reference tensor name sets do not match supplied full model/hash manifest')
    for name in [*set(selected.values()), *set(original.values())]:
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('unsafe shard filename')
    if args.verify_only:
        from safetensors import safe_open
        import torch
        names = sorted(selected)
        if args.verify_limit:
            if args.verify_limit < 8:
                raise ValueError('sample limit must be at least 8 for layer0/5 norm/scale coverage')
            sample = []
            for prefix, kind in [('model.layers.0.', 'norm'), ('model.layers.0.', 'weight_scale'),
                                 ('model.layers.5.', 'norm'), ('model.layers.5.', 'weight_scale')]:
                sample.extend([n for n in names if n.startswith(prefix) and kind in n][:2])
            names = list(dict.fromkeys(sample + names))[:args.verify_limit]
        checked = []
        for name in names:
            row = {'name': name}
            for label, base, index in [('full', full, original),
                    *([('selected', args.selected_model.resolve(strict=True), selected)] if args.selected_model else [])]:
                with safe_open(base / index[name], framework='pt', device='cpu') as stream:
                    tensor = stream.get_tensor(name).contiguous()
                digest = hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                row[label + '_matches_reference'] = digest == expected[name]
            checked.append(row)
        passed = all(all(value for key, value in row.items() if key.endswith('_matches_reference')) for row in checked)
        report = {'status': 'PASS' if passed else 'FAIL', 'mode': 'read_only_raw_bytes_verification',
                  'explicit_npu_calls': False, 'dtype_conversion': False, 'tensor_count': len(checked), 'checks': checked}
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        if not passed:
            raise SystemExit(1)
        return
    output.mkdir(parents=True, exist_ok=False)
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    groups = defaultdict(list)
    for name, shard in selected.items():
        groups[shard].append(name)
    total = 0
    written = []
    for shard, names in groups.items():
        tensors = {}
        for name in names:
            with safe_open(full / original[name], framework='pt', device='cpu') as stream:
                tensor = stream.get_tensor(name).contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest() != expected[name]:
                raise ValueError('selected tensor hash mismatch; no cast attempted: ' + name)
            total += len(raw)
            tensors[name] = tensor
        save_file(tensors, str(output / shard), metadata={'format': 'pt'})
        written.append(shard)
    if total != selected_index['metadata']['total_size']:
        raise ValueError('selected byte size differs from reference')
    for name in ('config.json', 'quant_model_description.json', 'model.safetensors.index.json'):
        shutil.copyfile(reference / name, output / name)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
                 'special_tokens_map.json', 'generation_config.json', 'merges.txt', 'vocab.json'):
        if (full / name).is_file():
            (output / name).symlink_to((full / name).resolve())
    (output / 'payload-build-receipt.json').write_text(json.dumps({
        'status': 'selected_tensor_payload_hashes_verified', 'tensor_count': len(selected),
        'dtype_conversion': False, 'total_bytes': total, 'shards': written,
        'full_index_sha256': hashlib.sha256((full / 'quant_model_weights.safetensors.index.json').read_bytes()).hexdigest(),
        'selected_reference_sha256': hashlib.sha256((HERE / 'references/model6-tensor-hashes.json').read_bytes()).hexdigest()
    }, indent=2) + '\n')
    print('Created hash-verified six-layer CPU payload; no explicit NPU calls.')


if __name__ == '__main__':
    main()
