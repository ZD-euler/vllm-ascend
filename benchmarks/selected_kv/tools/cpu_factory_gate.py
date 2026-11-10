#!/usr/bin/env python3
"""Probe actual harness imports, library loading and engine config without engine creation."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    sys.path.insert(0, str(root / 'harness'))
    import preflight
    import controller
    settings = preflight.SETTINGS
    profile = preflight.check(settings['arm'])
    env, _ = controller.environment(profile, root / 'runs' / settings['arm'])
    # Re-exec once in exactly the controller child environment before importing frameworks.
    import os
    if not args.child:
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  '--run-root', str(root), '--child'], env)
    import torch
    import torch_npu
    def initialized():
        return bool(torch.npu.is_initialized())
    if initialized():
        raise RuntimeError('NPU initialized before CPU gate')
    from c8_bootstrap import apply_c8_frontend_schema
    apply_c8_frontend_schema()
    importlib.import_module('early_store_handoff')
    importlib.import_module('worker_entry')
    sdk_module = importlib.import_module('memcache_hybrid')
    Path(sdk_module.__file__).resolve().relative_to(Path(settings['sdk']).resolve())
    if importlib.util.find_spec('vllm_ascend.vllm_ascend_C') is None:
        raise ImportError('base custom-op extension missing')
    from importlib import metadata
    required_plugins = {'vllm.platform_plugins': {'ascend': 'vllm_ascend:register'},
                        'vllm.general_plugins': {'ascend_kv_connector': 'vllm_ascend:register_connector',
                            'ascend_model_loader': 'vllm_ascend:register_model_loader',
                            'ascend_model': 'vllm_ascend:register_model'}}
    for group, required in required_plugins.items():
        entries = {entry.name: entry for entry in metadata.entry_points(group=group)}
        for name, target in required.items():
            if name not in entries or entries[name].value != target:
                raise ValueError('missing actual Ascend plugin registration: ' + name)
            Path(entries[name].dist.locate_file('')).resolve().relative_to(Path(settings['source']).resolve())
    import vllm
    import vllm_ascend
    Path(vllm.__file__).resolve().relative_to(Path(settings['core']).resolve())
    Path(vllm_ascend.__file__).resolve().relative_to(Path(settings['source']).resolve())
    loaded = []
    if settings['arm'] == 'm3v9':
        for name in ('libsparse_prefill_union_pipeline.so', 'libsparse_prefill_fused_fixed_launch.so',
                     'libdsa_packed_inplace.so', 'libdsa_selected_li_pages.so'):
            path = Path(settings['source']) / 'vllm_ascend/lib' / name
            torch.ops.load_library(str(path))
            loaded.append(name)
        if not (Path(settings['source']) / 'vllm_ascend/lib/sparse_prefill_fused_fixed.o').is_file():
            raise ValueError('missing fused device object')
    from vllm.platforms import current_platform
    if not type(current_platform).__module__.startswith('vllm_ascend'):
        raise ValueError('Ascend platform was not selected')
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    cfg = json.loads((root / 'engine.json').read_text())
    if cfg.get('kv_transfer_config'):
        cfg['kv_transfer_config'] = KVTransferConfig(**cfg['kv_transfer_config'])
    cfg.update(worker_cls='worker_entry.BenchWorker', enable_prefix_caching=True, disable_log_stats=True)
    actual = AsyncEngineArgs(**cfg).create_engine_config()
    if actual.speculative_config is not None:
        raise ValueError('unexpected speculative config')
    if initialized():
        raise RuntimeError('CPU factory unexpectedly initialized NPU')
    controller.atomic(root / 'public-cpu-factory-gate.json',
                      {'status': 'passed', 'NPU_initialized': False, 'engine_constructed': False,
                       'loaded_feature_libraries': loaded, 'speculative_config_is_None': True})


if __name__ == '__main__':
    main()
