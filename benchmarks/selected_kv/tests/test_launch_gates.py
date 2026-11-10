import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

BUNDLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher', BUNDLE / 'tools/launch_run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LaunchGateTests(unittest.TestCase):
    def test_historical_pass_cannot_replace_failed_current_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('public-unit-gate.json', 'public-cpu-factory-gate.json'):
                (root / name).write_text('{"status":"passed"}')
            controller = SimpleNamespace(atomic=Mock())
            with patch.object(launcher.subprocess, 'run', side_effect=[
                    SimpleNamespace(returncode=0), subprocess.CalledProcessError(1, ['factory'])]):
                with self.assertRaises(subprocess.CalledProcessError):
                    launcher.run_cpu_gates(root, controller, launcher.time.time() + 100)
            controller.atomic.assert_not_called()
            self.assertFalse((root / 'public-unit-gate.json').exists())
            self.assertFalse((root / 'public-cpu-factory-gate.json').exists())

    def test_smoke_accepts_distinct_view_indices_but_rejects_lineage_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'harness').mkdir()
            (root / 'harness/spec.py').write_text('TP=2\nDP=2\nEP=4\nLAYERS=6\n')
            (root / 'runs/m3v9').mkdir(parents=True)
            settings = dict(arm='m3v9', session_id='session', full_model_index_sha256='full',
                            model_base_metadata={'tokenizer.json': 'tokenizer', 'full-index.json': '78'})
            smoke_settings = dict(settings, model_base_metadata={'tokenizer.json': 'tokenizer', 'six-index.json': '6'})
            lock = {key: {'runtime-file': 'digest'} for key in ('source', 'core', 'sdk', 'deploy')}
            lock['corpus_reference'] = 'corpus'
            result = dict(status='completed', client_returncode=0, remaining_owned=0)
            launch = dict(controller_exit_code=0, remaining_owned=0, NPU_admitted=True)
            def write(name, data):
                (root / name).write_text(json.dumps(data))
            write('portable-settings.json', smoke_settings)
            write('runtime-lock.json', lock)
            write('runs/m3v9/public-result.json', result)
            write('public-launch-result.json', launch)
            launcher.validate_smoke(root, settings, lock)
            for key, value in [('session_id', 'wrong'), ('arm', 'native'), ('full_model_index_sha256', 'other')]:
                with self.subTest(key=key):
                    write('portable-settings.json', dict(smoke_settings, **{key: value}))
                    with self.assertRaises(ValueError):
                        launcher.validate_smoke(root, settings, lock)
            write('portable-settings.json', smoke_settings)
            write('public-launch-result.json', dict(launch, remaining_owned=1))
            with self.assertRaises(ValueError):
                launcher.validate_smoke(root, settings, lock)
