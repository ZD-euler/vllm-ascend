#!/usr/bin/env python3
"""Run the original core egg_info entry while preserving frozen runtime version bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    args = parser.parse_args()
    core = args.core.resolve(strict=True)
    expected = json.loads((HERE / 'references/source-identities.json').read_text())['core']
    for name, digest in expected.items():
        if sha(core / name) != digest:
            raise ValueError('core input differs: ' + name)
    version = core / 'vllm/_version.py'
    original = version.read_bytes()
    argv = [sys.executable, 'setup.py', 'egg_info']
    env = dict(os.environ, VLLM_TARGET_DEVICE='empty',
               VLLM_VERSION_OVERRIDE='0.1.dev1+g8cc9aa5ad.d20261016.empty', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    try:
        subprocess.run(argv, cwd=core, env=env, check=True, timeout=120)
    finally:
        version.write_bytes(original)
    for name, digest in expected.items():
        if sha(core / name) != digest:
            raise ValueError('core runtime/input changed during metadata generation: ' + name)
    print('Generated standard core metadata; restored and verified original runtime version bytes.')


if __name__ == '__main__':
    main()
