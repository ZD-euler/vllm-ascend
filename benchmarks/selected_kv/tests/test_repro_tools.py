"""CPU-only regressions for reproduction packaging; no accelerator imports."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load('manifest')
environment = load('export_environment')


class ManifestTests(unittest.TestCase):
    def test_relocation_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'original'
            root.mkdir()
            (root / 'script.py').write_text('print(1)\n')
            lock = manifest.create(root, root / 'manifest.json')
            relocated = Path(tmp) / 'relocated'
            shutil.copytree(root, relocated)
            manifest.verify(relocated, lock)
            (relocated / 'script.py').write_text('print(2)\n')
            with self.assertRaises(ValueError):
                manifest.verify(relocated, lock)

    def test_unsafe_paths_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('../outside', '/absolute'):
                with self.assertRaises(ValueError):
                    manifest.safe_file(root, name)
            (root / 'model.safetensors').write_bytes(b'not a real weight')
            with self.assertRaises(ValueError):
                manifest.create(root, root / 'manifest.json')

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            root.mkdir()
            (root / 'escape').symlink_to(Path(tmp))
            with self.assertRaises(ValueError):
                manifest.safe_file(root, 'escape/anything')


class EnvironmentTests(unittest.TestCase):
    def test_only_model_metadata_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'never read')
            (root / 'tokenizer.json').symlink_to(root / 'model.safetensors')
            metadata = root / 'metadata'
            metadata.mkdir()
            (metadata / 'tokenizer_config.json').write_text('{}')
            (root / 'tokenizer_config.json').symlink_to(metadata / 'tokenizer_config.json')
            with patch.object(environment, 'command', return_value={}), patch.object(
                    environment, 'package_info', return_value={}), patch.object(
                    environment, 'module_location', return_value=None):
                result = environment.inventory(root)
            self.assertEqual([entry['name'] for entry in result['model_metadata_hashes']], ['config.json', 'tokenizer_config.json'])
            self.assertNotIn('never read', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
