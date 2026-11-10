#!/usr/bin/env python3
"""Explicit finite NPU launch with fresh CPU gates and owned-process admission."""
import argparse
import importlib
import fcntl
import signal
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def arm_for_smoke(settings):
    return settings['arm']


def validate_smoke(smoke_root, settings, current_lock):
    smoke_settings = json.loads((smoke_root / 'portable-settings.json').read_text())
    smoke = json.loads((smoke_root / 'runs' / arm_for_smoke(settings) / 'public-result.json').read_text())
    smoke_launch = json.loads((smoke_root / 'public-launch-result.json').read_text())
    smoke_lock = json.loads((smoke_root / 'runtime-lock.json').read_text())
    import ast
    smoke_spec = ast.parse((smoke_root / 'harness/spec.py').read_text())
    geometry = {}
    for node in smoke_spec.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('TP', 'DP', 'EP', 'LAYERS'):
                geometry[node.targets[0].id] = ast.literal_eval(node.value)
    lineage_ok = (geometry == {'TP': 2, 'DP': 2, 'EP': 4, 'LAYERS': 6}
                  and all(smoke_lock[key] == current_lock[key] for key in ('source', 'core', 'sdk', 'deploy'))
                  and {k: v for k, v in smoke_settings.get('model_base_metadata', {}).items() if k.startswith('tokenizer')} == {k: v for k, v in settings.get('model_base_metadata', {}).items() if k.startswith('tokenizer')}
                  and smoke_settings.get('full_model_index_sha256') == settings.get('full_model_index_sha256')
                  and bool(settings.get('full_model_index_sha256'))
                  and smoke_lock['corpus_reference'] == current_lock['corpus_reference'])
    if (smoke_settings.get('arm') != settings['arm'] or smoke_launch.get('controller_exit_code') != 0
            or smoke_launch.get('cleanup_error') is not None or not lineage_ok or smoke.get('status') != 'completed' or smoke.get('remaining_owned') != 0
            or smoke.get('client_returncode') != 0 or smoke_launch.get('remaining_owned') != 0
            or not smoke_launch.get('NPU_admitted')
            or smoke_settings.get('session_id') != settings.get('session_id')
            or not settings.get('session_id')):
        raise ValueError('current session smoke proof failed')


def run_cpu_gates(root, controller, deadline):
    for name in ('public-unit-gate.json', 'public-cpu-factory-gate.json'):
        (root / name).unlink(missing_ok=True)
    # This gate is generated in this invocation, never copied from historical runs.
    command = [sys.executable, '-m', 'unittest', 'discover', '-s',
               str(HERE.parent / 'tests'), '-p', 'test_*.py', '-v']
    with (root / 'cpu-tests.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError('fresh CPU tests failed; see cpu-tests.log')
    subprocess.run([sys.executable, str(HERE / 'cpu_factory_gate.py'), '--run-root', str(root)],
                   check=True, timeout=min(600, max(1, deadline - time.time())))
    controller.atomic(root / 'public-unit-gate.json',
                      {'status': 'passed', 'remaining_owned': 0, 'command': command,
                       'exit_code': result.returncode, 'epoch': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--execute-npu', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    parser.add_argument('--deadline-seconds', type=int, default=7200)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if not args.execute_npu:
        print('Prepared only. Explicit --execute-npu is required to launch.')
        return
    if not 900 <= args.deadline_seconds <= 7200:
        raise ValueError('deadline must be 900..7200 seconds')
    once = root / 'launch-once'
    with once.open('x'):
        pass
    deadline = time.time() + args.deadline_seconds
    os.environ['BENCH_DEADLINE'] = str(deadline)
    sys.path.insert(0, str(root / 'harness'))
    preflight = importlib.import_module('preflight')
    # Full real input construction and identity checks happen before creating a model.
    spec = importlib.import_module('spec')
    reservation_dir = Path('/run/lock/m3v15-reproduction')
    reservation_dir.mkdir(parents=True, exist_ok=True)
    reservations = []
    for device in sorted(spec.DEVICES):
        handle = (reservation_dir / ('device-' + str(device) + '.lock')).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reservations.append(handle)
    # These shared host locks remain open until this launcher exits.
    subprocess.run([sys.executable, str(root / 'harness/preflight.py'), 'enroll'],
                   check=True, timeout=min(600, max(1, deadline - time.time())))
    settings = preflight.SETTINGS
    controller = importlib.import_module('controller')
    owned = importlib.import_module('owned_process_utils_v1')
    if spec.LAYERS == 78:
        if args.smoke_root is None:
            raise ValueError('formal trace requires this reproduction session smoke root')
        validate_smoke(args.smoke_root.resolve(strict=True), settings,
                       json.loads((root / 'runtime-lock.json').read_text()))
    run_cpu_gates(root, controller, deadline)
    if not controller.idle(controller.hardware()):
        raise RuntimeError('selected devices are not idle')
    ns = root / 'namespace'
    (ns / 'tmp').mkdir(parents=True)
    (ns / 'ascend-log').mkdir()
    (ns / 'tmp').chmod(0o1777)
    wrapper = root / 'namespace.sh'
    # Same isolated /tmp, Ascend log and 2GiB /dev/shm as recovered launcher.
    wrapper.write_text('''#!/usr/bin/env bash
set -eo pipefail
mount --make-rprivate /
mount --bind "$1/tmp" /tmp
mount --bind "$1/ascend-log" /root/ascend/log
mount -t tmpfs -o size=2G,mode=1777 m3v15-repro-shm /dev/shm
shift
unset LD_AUDIT LD_PRELOAD PYTHONPATH
source "$1/set_env.sh"
shift
exec "$@"
''')
    arm = settings['arm']
    guard = "import os,sys; assert os.readlink('/proc/self/ns/mnt') != sys.argv[1]; os.execv('/bin/bash',['bash',*sys.argv[2:]])"
    argv = ['unshare', '--mount', '--fork', '--propagation', 'private', sys.executable,
            '-c', guard, os.readlink('/proc/self/ns/mnt'), str(wrapper), str(ns),
            settings['cann'], sys.executable, str(root / 'harness/controller.py'), arm]
    def terminate(signum, frame):
        raise InterruptedError('launcher received signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    child = None
    scope = {}
    admitted = False
    try:
        with (root / 'controller.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, stdin=subprocess.DEVNULL)
            identity = owned.process(child.pid)
            if identity is None:
                raise RuntimeError('cannot identify launched child')
            scope[child.pid] = identity
            while child.poll() is None:
                live = owned.descendants(scope)
                if time.time() >= deadline:
                    raise TimeoutError('finite run deadline reached')
                out = root / 'runs' / arm
                candidate = out / 'public-ready-to-admit.json'
                if not admitted and candidate.is_file():
                    value = json.loads(candidate.read_text())
                    actual = {(x['pid'], x['startticks']) for x in live}
                    workers = value.get('workers', [])
                    if (value.get('allow_npu') is True and value.get('expires', 0) > time.time()
                            and len(workers) == spec.TP * spec.DP
                            and len({w['pid'] for w in workers}) == len(workers)
                            and all((w['pid'], w['startticks']) in actual and
                                    w.get('initialized') is False for w in workers)):
                        preflight.check(arm)
                        if controller.idle(controller.hardware()):
                            controller.atomic(out / 'admission.json', dict(value, root_authorized=True))
                            admitted = True
                time.sleep(1)
            if child.returncode:
                raise RuntimeError('controller failed; see controller.log')
    finally:
        if child is not None:
            owned.descendants(scope)
            cleanup_error = None
            try:
                if child.poll() is None or owned.descendants(scope):
                    owned.retire(scope)
                child.wait(timeout=30)
            except BaseException as error:
                cleanup_error = type(error).__name__
            finally:
                remaining = len(owned.descendants(scope))
                controller.atomic(root / 'public-launch-result.json',
                                  {'controller_exit_code': child.poll(), 'remaining_owned': remaining,
                                   'cleanup_error': cleanup_error, 'NPU_admitted': admitted,
                                   'deadline_epoch': deadline})
            if cleanup_error or remaining:
                raise RuntimeError('owned cleanup incomplete; inspect public-launch-result.json')
    if not admitted:
        raise RuntimeError('run ended without NPU admission')


if __name__ == '__main__':
    main()
