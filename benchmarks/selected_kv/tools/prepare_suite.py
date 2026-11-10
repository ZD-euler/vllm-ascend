#!/usr/bin/env python3
"""Prepare M3 smoke/trace by default; native comparison is explicit and optional."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]


def resolve_config_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO / path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-native', action='store_true')
    args = parser.parse_args()
    paths = json.loads(args.paths_config.read_text())
    paths.setdefault('m3_source', str(REPO))
    arms = [('m3v9', 'm3')]
    if args.include_native:
        if not paths.get('native_source') or not paths.get('native_base_receipt'):
            raise ValueError('--include-native requires native_source and native_base_receipt')
        arms.append(('native', 'native'))
    args.output.mkdir(parents=True, exist_ok=False)
    for arm, label in arms:
        for layers, phase in [('6', 'smoke'), ('78', 'trace')]:
            settings = args.output / (phase + '-' + label + '-settings.json')
            command = [sys.executable, str(HERE / 'tools/make_settings.py'), '--arm', arm,
                       '--layers', layers, '--session-id', paths['session_id'],
                       '--model', resolve_config_path(paths['model' + layers]),
                       '--source', resolve_config_path(paths[label + '_source']),
                       '--base-receipt', resolve_config_path(paths[label + '_base_receipt']),
                       '--sdk-receipt', resolve_config_path(paths['sdk_receipt']), '--output', str(settings)]
            for key in ('core', 'sdk', 'cann', 'corpus', 'token_root'):
                command.extend(['--' + key.replace('_', '-'), resolve_config_path(paths[key])])
            if arm == 'm3v9':
                command.extend(['--feature-receipt', resolve_config_path(paths['feature_receipt'])])
            subprocess.run(command, check=True, timeout=600)
            recovered = 'smoke4-input-v2' if phase == 'smoke' else ('trace-native' if arm == 'native' else 'trace-m3v15')
            subprocess.run([sys.executable, str(HERE / 'tools/prepare_run.py'), '--settings', str(settings),
                            '--recovered', str(HERE / 'recovered' / recovered),
                            '--output', str(args.output / (phase + '-' + label))], check=True, timeout=120)
    print(f'Prepared {len(arms) * 2} fresh run directories; no model launched.')


if __name__ == '__main__':
    main()
