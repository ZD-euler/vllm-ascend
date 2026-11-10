import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('prepare_run', TOOLS / 'prepare_run.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_fresh_root_paths_and_protocol_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recovered = root / 'recovered'
            harness = recovered / 'harness'
            harness.mkdir(parents=True)
            for name in ('controller.py', 'export_compare.py', 'export_mte.py', 'import_preflight.py'):
                (harness / name).write_text("ROOT = Path('/old/root')\nDEPLOY = preflight.BASE / \"matrix42_15cases_20261031/deploy\"\n")
            (recovered / 'public-unit-gate.json').write_text('{\"status\":\"passed\"}')
            original = "protocol = [('warmup', 'event'), ('profile', 'event')]\n"
            original += "TOKEN_ROOT = Path('/repro/reference-root/real_coding')\n"
            (harness / 'run_cases.py').write_text(original)
            (recovered / 'engine-m3v9.json').write_text(json.dumps({'model': '/old/model', 'seed': 1024}))
            settings = {key: str(root) for key in prepare.REQUIRED}
            settings.update(arm='m3v9', revision='db85ccd37d8fe90b9835903355f3bc92c1220e85')
            output = root / 'new-output'
            prepare.prepare(settings, recovered, output)
            self.assertIn("protocol = [('warmup', 'event'), ('profile', 'event')]", (output / 'harness/run_cases.py').read_text())
            self.assertNotIn('/repro/reference-root', (output / 'harness/run_cases.py').read_text())
            self.assertIn('preflight.DEPLOY', (output / 'harness/controller.py').read_text())
            cfg = json.loads((output / 'engine.json').read_text())
            self.assertEqual(cfg['seed'], 1024)
            self.assertEqual(cfg['model'], str(root.resolve()))
            self.assertNotIn('/old/root', (output / 'harness/controller.py').read_text())
            self.assertNotIn('public-unit-gate.json', {p.name for p in output.iterdir()})
            with self.assertRaises(ValueError):
                prepare.prepare(settings, recovered, output)

    def test_wrong_revision_fails_before_output_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {key: str(root) for key in prepare.REQUIRED}
            settings.update(arm='m3v9', revision='wrong')
            with self.assertRaises(ValueError):
                prepare.prepare(settings, root, root / 'output')
            self.assertFalse((root / 'output').exists())
