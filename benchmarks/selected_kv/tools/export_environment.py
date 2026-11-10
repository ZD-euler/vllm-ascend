#!/usr/bin/env python3
"""Read-only environment inventory. Never imports torch or initializes an NPU."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

MODEL_FILES = {
    'config.json', 'quant_model_description.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
    'vocab.txt', 'merges.txt', 'tokenizer.model', 'added_tokens.json',
    'configuration_deepseek.py',
}
PACKAGES = ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers',
            'numpy', 'triton', 'safetensors', 'tokenizers', 'memcache', 'memfabric',
            'memcache-hybrid', 'memfabric-hybrid')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {'command': argv, 'exit_code': result.returncode,
                'stdout': result.stdout[:65536], 'stderr': result.stderr[:4096]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': argv, 'unavailable': type(exc).__name__}


def package_info(name):
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {'installed': False}
    result = {'installed': True, 'version': dist.version,
              'location': str(dist.locate_file(''))}
    # Keep reproducibility refs, discard URL credentials/query strings and local paths.
    direct = dist.read_text('direct_url.json')
    if direct:
        from urllib.parse import urlsplit, urlunsplit
        data = json.loads(direct)
        url = urlsplit(data.get('url', ''))
        if url.scheme in ('https', 'http', 'ssh', 'git'):
            result['source_url'] = urlunsplit((url.scheme, url.hostname or '', url.path, '', ''))
        result['vcs_info'] = data.get('vcs_info', {})
        result['editable'] = data.get('dir_info', {}).get('editable', False)
    return result


def module_location(name):
    # Top-level find_spec does not import the target package.
    try:
        spec = importlib.util.find_spec(name)
        return None if spec is None else spec.origin
    except (ImportError, ValueError):
        return None


def inventory(model_dir=None, libraries=(), version_files=()):
    info = {'schema_version': 1, 'python': sys.version,
            'platform': platform.platform(), 'machine': platform.machine(),
            'packages': {name: package_info(name) for name in PACKAGES},
            'all_installed_distributions': sorted(
                [{'name': dist.metadata.get('Name', ''), 'version': dist.version}
                 for dist in importlib.metadata.distributions()], key=lambda item: item['name'].lower()),
            'module_paths': {name: module_location(name) for name in
                             ('torch', 'torch_npu', 'vllm', 'vllm_ascend')},
            'hardware': {'cpu': command(['lscpu']), 'npu': command(['npu-smi', 'info'])},
            'libraries': [], 'version_files': [], 'model_metadata_hashes': []}
    for raw in libraries:
        path = Path(raw).resolve(strict=True)
        if not path.is_file() or not re.search(r'\.so(?:\.[0-9.]+)?$', path.name):
            raise ValueError('library must be an explicit shared-library file')
        info['libraries'].append({'path': str(path), 'sha256': digest(path)})
    allowed_versions = {'version.info', 'version.cfg', 'version', 'ascend_toolkit_install.info'}
    for raw in version_files:
        path = Path(raw).resolve(strict=True)
        if path.name not in allowed_versions or path.stat().st_size > 65536:
            raise ValueError('only small, named vendor version files are allowed')
        info['version_files'].append({'path': str(path), 'sha256': digest(path),
                                      'text': path.read_text(errors='replace')})
    if model_dir:
        root = Path(model_dir).resolve(strict=True)
        for name in sorted(MODEL_FILES):
            path = root / name
            if path.is_file() and path.resolve().name in MODEL_FILES:
                info['model_metadata_hashes'].append({'name': name, 'sha256': digest(path)})
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--library', action='append', default=[])
    parser.add_argument('--version-file', action='append', default=[])
    args = parser.parse_args()
    result = inventory(args.model_dir, args.library, args.version_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
