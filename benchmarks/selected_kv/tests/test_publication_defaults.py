"""Regression checks for Git publication defaults; no model or build execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
TOOLS = BUNDLE / 'tools'
REPO = BUNDLE.parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('publication_' + name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationDefaultsTests(unittest.TestCase):
    def assert_prepared_layout(self, commands, expected):
        self.assertEqual(len(commands), 2 * len(expected))
        for index, (layers, directory, harness) in enumerate(expected):
            setting_command, prepare_command = commands[index * 2:index * 2 + 2]
            self.assertTrue(setting_command[1].endswith('make_settings.py'))
            self.assertTrue(prepare_command[1].endswith('prepare_run.py'))
            self.assertEqual(setting_command[setting_command.index('--layers') + 1], layers)
            self.assertEqual(prepare_command[prepare_command.index('--settings') + 1],
                             setting_command[setting_command.index('--output') + 1])
            self.assertEqual(Path(prepare_command[prepare_command.index('--output') + 1]).name, directory)
            self.assertEqual(Path(prepare_command[prepare_command.index('--recovered') + 1]).name, harness)

    def test_suite_default_has_only_m3_and_resolves_config_against_repo(self):
        suite = load('prepare_suite')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {key: 'relative/' + key for key in ('model6', 'model78', 'core', 'sdk',
                    'cann', 'corpus', 'token_root', 'm3_base_receipt', 'sdk_receipt', 'feature_receipt')}
            paths['session_id'] = 'publication-test'
            config = root / 'paths.json'
            config.write_text(json.dumps(paths))
            with patch.object(sys, 'argv', ['suite', '--paths-config', str(config), '--output', str(root / 'runs')]), patch.object(suite.subprocess, 'run') as run:
                suite.main()
            commands = [call.args[0] for call in run.call_args_list]
            self.assert_prepared_layout(commands, [('6', 'smoke-m3', 'smoke4-input-v2'),
                                                  ('78', 'trace-m3', 'trace-m3v15')])
            settings_commands = [command for command in commands if command[1].endswith('make_settings.py')]
            self.assertEqual(len(settings_commands), 2)
            self.assertEqual([command[command.index('--layers') + 1] for command in settings_commands], ['6', '78'])
            for command in settings_commands:
                self.assertEqual(command[command.index('--arm') + 1], 'm3v9')
                self.assertEqual(command[command.index('--source') + 1], str(REPO))
                self.assertEqual(command[command.index('--core') + 1], str(REPO / 'relative/core'))
                self.assertNotIn('native', command)

    def test_native_requires_explicit_config_and_prepares_four_only_when_enabled(self):
        suite = load('prepare_suite')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'paths.json'
            paths = {'session_id': 'test'}
            config.write_text(json.dumps(paths))
            argv = ['suite', '--include-native', '--paths-config', str(config), '--output', str(root / 'runs')]
            with patch.object(sys, 'argv', argv), patch.object(suite.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'native_source'):
                    suite.main()
                run.assert_not_called()
            paths.update({key: str(root) for key in ('model6', 'model78', 'core', 'sdk', 'cann',
                          'corpus', 'token_root', 'm3_base_receipt', 'sdk_receipt', 'feature_receipt',
                          'native_source', 'native_base_receipt')})
            config.write_text(json.dumps(paths))
            with patch.object(sys, 'argv', argv), patch.object(suite.subprocess, 'run') as run:
                suite.main()
            all_commands = [call.args[0] for call in run.call_args_list]
            self.assert_prepared_layout(all_commands, [('6', 'smoke-m3', 'smoke4-input-v2'),
                ('78', 'trace-m3', 'trace-m3v15'), ('6', 'smoke-native', 'smoke4-input-v2'),
                ('78', 'trace-native', 'trace-native')])
            commands = [command for command in all_commands if command[1].endswith('make_settings.py')]
            self.assertEqual([command[command.index('--arm') + 1] for command in commands],
                             ['m3v9', 'm3v9', 'native', 'native'])

    def test_make_settings_defaults_source_to_repo_and_native_requires_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'settings.json'
            command = [sys.executable, str(TOOLS / 'make_settings.py'), '--layers', '6',
                       '--session-id', 'test', '--identity-mode', 'historical-runtime', '--output', str(output)]
            for key in ('model', 'core', 'sdk', 'cann', 'corpus', 'token-root'):
                command.extend(['--' + key, str(root)])
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            settings = json.loads(output.read_text())
            self.assertEqual(settings['source'], str(REPO.resolve()))
            self.assertEqual(settings['arm'], 'm3v9')
            output.unlink()
            result = subprocess.run(command + ['--arm', 'native'], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--source', result.stderr)
            self.assertFalse(output.exists())

    def test_archive_map_retains_only_core_and_original_pins_exclude_only_documents(self):
        revision = json.loads((BUNDLE / 'artifacts/source/REVISIONS.json').read_text())
        self.assertEqual(set(revision['archives']), {'core-native-runtime-source.tar.gz', 'core-native-install-source.tar.gz'})
        original = json.loads((BUNDLE / 'references/original-pinned-source-identities.json').read_text())
        current = json.loads((BUNDLE / 'references/pinned-source-identities.json').read_text())
        for arm in ('m3v9', 'native'):
            self.assertEqual(set(original[arm]) - set(current[arm]), {'README.md', '.gitignore'})
            self.assertTrue(all(original[arm][key] == value for key, value in current[arm].items()))
            self.assertIn('vllm_ascend/_cann_ops_custom/.gitkeep', current[arm])

    def test_tracked_custom_op_placeholder_is_not_generated_output(self):
        receipts = load('build_receipts')
        self.assertFalse(receipts.allowed('base_ops', 'vllm_ascend/_cann_ops_custom/.gitkeep'))
        self.assertTrue(receipts.allowed('base_ops', 'vllm_ascend/_cann_ops_custom/vendors/custom/op.py'))

    def test_generated_version_namespace_is_exact_and_recipe_specific(self):
        receipts = load('build_receipts')
        self.assertTrue(receipts.allowed('base_ops', 'vllm_ascend/_version.py'))
        for path in ('vllm_ascend/_version.py.bak', 'vllm_ascend/worker/_version.py',
                     'vllm_ascend/_version.py/../worker.py'):
            with self.subTest(path=path):
                self.assertFalse(receipts.allowed('base_ops', path))
        for recipe in ('sdk17', 'selected_features'):
            with self.subTest(recipe=recipe):
                self.assertFalse(receipts.allowed(recipe, 'vllm_ascend/_version.py'))
