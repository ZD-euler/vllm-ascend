import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('corpus_export', TOOLS / 'export_corpus_provenance.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class CorpusTests(unittest.TestCase):
    def test_no_content_or_credentials_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tokens.json').write_text('[1, 2, 3]')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'synthetic_padding_tokens': False,
                'prompt': 'PRIVATE PROMPT', 'sources': [{'task_name': 'a',
                'token_file': 'tokens.json', 'glm52_tokens': 3,
                'repo_url': 'https://username:password@example.com/repo?secret=1',
                'tokens': [1, 2, 3], 'commit': 'abc'}],
                'cases': [{'history': 2, 'windows': [{'task': 'a', 'start_token': 0,
                'text': 'PRIVATE PROMPT'}]}]}))
            result = exporter.export(manifest, root)
            self.assertEqual(result['sources'][0]['repo_url'], 'https://example.com/repo')
            self.assertEqual(result['sources'][0]['token_file_sha256'], exporter.sha(root / 'tokens.json'))
            self.assertEqual(result['cases'][0]['windows'], [{'task': 'a', 'start_token': 0}])
            for secret in ('PRIVATE PROMPT', 'password', 'secret=1', 'username'):
                self.assertNotIn(secret, json.dumps(result))
