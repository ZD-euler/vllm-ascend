import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_receipts', BUNDLE / 'tools/build_receipts.py')
receipts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipts)


class ReceiptTests(unittest.TestCase):
    def test_namespace_boundaries(self):
        self.assertTrue(receipts.allowed('base_ops', 'vllm_ascend/_cann_ops_custom/dynamic/op.py'))
        self.assertFalse(receipts.allowed('base_ops', 'vllm_ascend/worker/model_runner.py'))
        self.assertFalse(receipts.allowed('selected_features', 'vllm_ascend/lib/other.so'))
        self.assertFalse(receipts.allowed('sdk17', 'vllm/model.py'))
        self.assertFalse(receipts.allowed('base_ops', 'vllm_ascend/_cann_ops_custom/../worker.py'))

    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['vllm_ascend/lib/' + name for name in receipts.FEATURES]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'receipt fixture, not compiled code')
            data = dict(schema_version=1, recipe='selected_features', arm='m3v9',
                status='compiled_not_NPU_validated',
                builder_sha256=receipts.sha(BUNDLE / 'tools/build_features.py'),
                helper_sha256=receipts.sha(BUNDLE / 'tools/build_receipts.py'),
                inputs={'source.cpp': 'source-digest'}, commands=[{'exit_code': 0}],
                output_root=str(root), outputs={name: receipts.sha(root / name) for name in names})
            receipt = root / 'receipt.json'
            receipt.write_text(json.dumps(data))
            with patch.object(receipts, 'expected_inputs', return_value=data['inputs']):
                receipts.verify(receipt, BUNDLE, 'selected_features', 'm3v9', root)
                for key, value in [('recipe', 'base_ops'), ('arm', 'native'), ('status', 'PASS'),
                                   ('builder_sha256', 'wrong'), ('helper_sha256', 'wrong'),
                                   ('inputs', {'changed': 'digest'}), ('output_root', str(root / 'other')),
                                   ('commands', [{'exit_code': 1}])]:
                    with self.subTest(key=key):
                        receipt.write_text(json.dumps(dict(data, **{key: value})))
                        with self.assertRaises(ValueError):
                            receipts.verify(receipt, BUNDLE, 'selected_features', 'm3v9', root)
                receipt.write_text(json.dumps(data))
                (root / names[0]).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    receipts.verify(receipt, BUNDLE, 'selected_features', 'm3v9', root)

    def test_required_base_and_sdk_outputs(self):
        base = {'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so': 'digest',
                'vllm_ascend/libvllm_ascend_kernels.so': 'digest',
                'vllm_ascend.egg-info/PKG-INFO': 'digest',
                'vllm_ascend.egg-info/entry_points.txt': 'digest'}
        sdk = {name: 'digest' for name in ('memcache_hybrid/lib/libmf_memcache.so',
               'memfabric_hybrid/lib/libmf_hybm_core.so', 'memfabric_hybrid/lib/libmf_smem.so',
               'memcache_hybrid/config/mmc-local.conf', 'memcache_hybrid/config/mmc-meta.conf')}
        for recipe, outputs in [('base_ops', base), ('sdk17', sdk)]:
            with self.subTest(recipe=recipe):
                receipts.require_outputs(recipe, outputs)
                for missing in outputs:
                    with self.assertRaises(ValueError):
                        receipts.require_outputs(recipe, {k: v for k, v in outputs.items() if k != missing})
