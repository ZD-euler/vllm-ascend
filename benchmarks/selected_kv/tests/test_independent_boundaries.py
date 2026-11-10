"""Exercise portable boundaries with local data and real OS ownership primitives."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
TOOLS = BUNDLE / 'tools'
HARNESS = BUNDLE / 'recovered/trace-m3v15/harness'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndependentBoundaryTests(unittest.TestCase):
    def test_relocated_entry_after_original_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'original'
            shutil.copytree(TOOLS, original / 'tools')
            relocated = Path(tmp) / 'relocated'
            original.rename(relocated)
            # A fresh interpreter imports sibling tools from the new directory.
            result = subprocess.run([sys.executable, str(relocated / 'tools/make_settings.py'), '--help'],
                                    cwd=tmp, capture_output=True, text=True,
                                    env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--base-receipt', result.stdout)
            self.assertFalse(original.exists())

    def test_reference_identity_rejects_core_tamper_and_escape(self):
        preflight = load('boundary_preflight', TOOLS / 'portable_preflight.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {'identity_files': {}}
            for scope in ('source', 'core', 'sdk', 'deploy'):
                directory = root / scope
                directory.mkdir()
                file = directory / 'identity.txt'
                file.write_text(scope)
                settings[scope] = str(directory)
                settings['identity_files'][scope] = {'identity.txt': preflight.sha(file)}
            with patch.object(preflight, 'SETTINGS', settings):
                preflight.verify_identity()
                (root / 'core/identity.txt').write_text('different core')
                with self.assertRaisesRegex(ValueError, 'core/identity.txt'):
                    preflight.verify_identity()
                (root / 'core/identity.txt').write_text('core')
                outside = root / 'outside'
                outside.write_text('core')
                (root / 'core/identity.txt').unlink()
                (root / 'core/identity.txt').symlink_to(outside)
                with self.assertRaises(ValueError):
                    preflight.verify_identity()

    def test_real_corpus_window_hash_and_source_hash(self):
        try:
            import numpy  # Harness CPU dependency; no accelerator imports.
        except ImportError:
            self.skipTest('numpy required by recovered input construction')
        with patch.dict(sys.modules, {'spec': SimpleNamespace(), 'perf_common': SimpleNamespace(atomic=None)}):
            cases = load('boundary_run_cases', HARNESS / 'run_cases.py')
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / 'tokens.json'
            values = list(range(4096))
            token_file.write_text(json.dumps(values))
            manifest = {'sources': [{'task_name': 'fixture', 'token_file': str(token_file),
                         'glm52_tokens': len(values), 'tokens_sha256_uint32_le': cases.digest(values)}],
                        'cases': [{'history': 128, 'query': 128, 'total_query_tokens': 2048,
                         'requests': 1, 'windows': [{'task': 'fixture', 'start_token': 0,
                         'token_sha256': cases.digest(values[:256])}]}]}
            selected = [{'H': 128, 'request_Qs': [128], 'case': 'fixture'}]
            arrays = cases.source_arrays(manifest)
            histories, followups, proof = cases.inputs_for_H(manifest, arrays, 128, selected)
            self.assertEqual(histories, [values[:128]])
            self.assertEqual(followups['fixture'][0][0], values[128:256])
            self.assertEqual(len(followups['fixture'][0]), 13)
            self.assertEqual(proof, [cases.digest(values[:128])])
            manifest['cases'][0]['windows'][0]['start_token'] = 1
            with self.assertRaisesRegex(ValueError, 'window digest'):
                cases.inputs_for_H(manifest, arrays, 128, selected)
            values[0] = 999
            token_file.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'corpus digest'):
                cases.source_arrays(manifest)

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open') and
                         hasattr(signal, 'pidfd_send_signal'), 'requires Linux procfs/pidfd')
    def test_real_process_startticks_ownership(self):
        owned = load('boundary_owned', HARNESS / 'owned_process_utils_v1.py')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            identity = owned.process(child.pid)
            self.assertIsNotNone(identity)
            stale = dict(identity, startticks=identity['startticks'] + 1)
            owned.retire({child.pid: stale})
            self.assertIsNone(child.poll(), 'stale PID identity must not kill a live process')
            owned.retire({child.pid: identity})
            child.wait(timeout=5)
            self.assertEqual(child.returncode, -signal.SIGTERM)
            self.assertIsNone(unrelated.poll(), 'cleanup must preserve unrelated processes')
        finally:
            for process in (child, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)

    @unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open') and
                         hasattr(signal, 'pidfd_send_signal'), 'requires Linux procfs/pidfd')
    def test_launcher_signal_cleans_owned_process_and_releases_lock(self):
        # Isolate the launcher's process-wide signal handlers in a fresh interpreter.
        driver = r'''
import fcntl, importlib.util, json, os, pathlib, signal, subprocess, sys, threading
from types import SimpleNamespace
from unittest.mock import patch
bundle, root = map(pathlib.Path, sys.argv[1:])
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
launcher = load('launcher', bundle / 'tools/launch_run.py')
owned = load('owned', bundle / 'recovered/trace-m3v15/harness/owned_process_utils_v1.py')
(root / 'harness').mkdir()
settings = dict(arm='m3v9', cann=str(root))
def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
controller = SimpleNamespace(idle=lambda _: True, hardware=lambda: {}, atomic=atomic)
modules = {'preflight': SimpleNamespace(SETTINGS=settings), 'controller': controller,
           'spec': SimpleNamespace(LAYERS=6, DEVICES=[0], TP=1, DP=1),
           'owned_process_utils_v1': owned}
real_popen = subprocess.Popen
children = []
def spawn(*args, **kwargs):
    # Verify the shared lock is held before starting any owned child.
    with (root / 'locks/device-0.lock').open('a') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise AssertionError('launcher did not reserve device')
    child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                       start_new_session=True)
    children.append(child)
    threading.Timer(.2, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    return child
real_path = pathlib.Path
def mapped_path(value):
    return root / 'locks' if value == '/run/lock/m3v15-reproduction' else real_path(value)
try:
    with patch.dict(sys.modules, modules), patch.object(sys, 'argv', ['launch', '--run-root', str(root), '--execute-npu']), patch.object(launcher, 'Path', mapped_path), patch.object(launcher, 'run_cpu_gates'), patch.object(launcher.subprocess, 'run'), patch.object(launcher.subprocess, 'Popen', spawn):
        launcher.main()
except InterruptedError: pass
else: raise AssertionError('SIGTERM did not interrupt launcher')
assert len(children) == 1 and children[0].poll() == -signal.SIGTERM
result = json.loads((root / 'public-launch-result.json').read_text())
assert result['remaining_owned'] == 0 and result['cleanup_error'] is None, result
with (root / 'locks/device-0.lock').open('a') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
print('signal cleanup and shared lock verified')
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-c', driver, str(BUNDLE), tmp],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('signal cleanup and shared lock verified', result.stdout)
