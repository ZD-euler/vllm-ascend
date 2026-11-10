#!/usr/bin/env python3
"""Build the pinned Ascend base custom operators into an extracted owned source."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from build_receipts import allowed, check_inputs, emit, files, install_build_signal_handlers

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('native', 'm3v9'), default='m3v9')
    parser.add_argument('--source', type=Path, default=HERE.parents[1])
    parser.add_argument('--cann', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    install_build_signal_handlers()
    if args.output.resolve().is_relative_to(HERE):
        raise ValueError('build output must stay outside the code bundle')
    source = args.source.resolve(strict=True)
    os.environ['ASCEND_HOME_PATH'] = str(args.cann.resolve(strict=True))
    inputs = check_inputs(HERE, 'base_ops', args.arm, source)
    if any(allowed('base_ops', p.relative_to(source).as_posix()) for p in source.rglob('*') if p.is_file()):
        raise ValueError('base build requires fresh extracted source without generated base artifacts')
    args.output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, 'setup.py', 'build_aclnn', 'build_ext', '--inplace', 'egg_info']
    env = dict(os.environ, ASCEND_HOME_PATH=str(args.cann.resolve(strict=True)),
               SOC_VERSION='ascend910_9382', MAX_JOBS='8', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    with (args.output / 'build.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
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
        raise RuntimeError('base custom-op build failed; see build.log')
    emit(HERE, 'base_ops', args.arm, source, source, inputs,
         [{'argv': argv, 'exit_code': code, 'cwd': str(source)}], args.output / 'receipt.json')


if __name__ == '__main__':
    main()
