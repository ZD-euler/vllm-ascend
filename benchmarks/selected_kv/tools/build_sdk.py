#!/usr/bin/env python3
"""Stage the hash-verified sdk17 patches, then build the official C-ABI pair."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from build_receipts import check_inputs, emit, install_build_signal_handlers

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--build-aicpu', action='store_true')
    args = parser.parse_args()
    install_build_signal_handlers()
    if args.output.resolve().is_relative_to(HERE):
        raise ValueError('build output must stay outside the code bundle')
    if not 1 <= args.jobs <= 32:
        raise ValueError('jobs must be 1..32')
    args.output.mkdir(parents=True, exist_ok=False)
    base = HERE / 'sdk/base'
    addon = HERE / 'sdk/sdk17-optin-delivery-v1'
    source = args.output.resolve() / 'source'
    inputs = check_inputs(HERE, 'sdk17', 'shared', HERE / 'sdk')
    subprocess.run([sys.executable, str(addon / 'unrestricted-v17/stage_sdk17.py'),
                    '--archive', str(base / 'matching_pair_source_v2.tar.gz'),
                    '--source-manifest', str(base / 'matching_pair_source_manifest_v1.json'),
                    '--output', str(source), '--addon-dir', str(addon / 'common')],
                   check=True, timeout=300)
    manifest = json.loads((base / 'matching_pair_source_manifest_v1.json').read_text())
    proof = json.loads((source / 'dsa-matching-stage-proof.json').read_text())
    # This map is produced by the supplied strict stager from its frozen base and patch manifests.
    manifest['files'] = proof['final_source_files']
    final_manifest = args.output / 'sdk17-final-source-manifest.json'
    final_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    original_builder = (base / 'build_matching_pair_v1.py').read_text()
    original_record = 'dict(stage=stage,returncode=result.returncode,wall_seconds=time.monotonic()-begin)'
    if original_builder.count(original_record) != 1:
        raise ValueError('recording adapter does not match frozen SDK builder')
    recorded_builder = args.output / 'build_matching_pair_recorded.py'
    recorded_builder.write_text(original_builder.replace(original_record,
        'dict(stage=stage,returncode=result.returncode,wall_seconds=time.monotonic()-begin,argv=command,cwd=str(cwd) if cwd else None)'))
    command = [sys.executable, str(recorded_builder.resolve()), '--sources', str(source),
               '--manifest', str(final_manifest.resolve()), '--out', str(args.output.resolve() / 'build'),
               '--python', sys.executable, '--jobs', str(args.jobs)]
    if args.build_aicpu:
        command.append('--build-aicpu')
    child = subprocess.Popen(command, start_new_session=True)
    try:
        code = child.wait(timeout=7200)
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        raise
    if code:
        raise RuntimeError('SDK build failed; inspect build/public-build.json')
    inner = json.loads((args.output / 'build/public-build.json').read_text())
    commands = [{'argv': row['argv'], 'exit_code': row['returncode'], 'cwd': row.get('cwd')} for row in inner['stages']]
    emit(HERE, 'sdk17', 'shared', source, args.output.resolve() / 'build/site', inputs,
         commands, args.output / 'receipt.json')
    print('SDK build finished; hardware/lifecycle acceptance still requires fresh smoke.')


if __name__ == '__main__':
    main()
