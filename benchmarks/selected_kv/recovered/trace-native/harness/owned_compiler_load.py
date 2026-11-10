"""Test-only owned compiler initialization before the original weight loader.

This is a combined configuration diagnostic: fresh cache paths, explicit CANN
cache options, internal format enabled and JIT compilation enabled. It does not
attribute a failure to one setting, replace quantization, or skip NZ casts.
Only the finite smoke worker should inherit this mixin.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


def _publish(scope, report):
    path = scope / 'public-compiler-load.json'
    temporary = path.with_suffix('.next.json')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


class OwnedCompilerLoadMixin:
    def load_model(self):
        if getattr(self, '_owned_compiler_load_attempted', False):
            raise RuntimeError('The finite compiler-init diagnostic does not retry weight loading')
        self._owned_compiler_load_attempted = True
        root = Path(os.environ['DSA_M2_SMOKE_OUTPUT']).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('A parent-owned existing smoke output directory is required')
        base = root / 'compiler'
        if base.is_symlink():
            raise ValueError('The owned compiler directory cannot be a symlink')
        base.mkdir(mode=0o700, exist_ok=True)
        scope = base / f'rank{int(self.rank)}-pid{os.getpid()}'
        scope.mkdir(mode=0o700, exist_ok=False)
        cache, work = scope / 'cache', scope / 'work'
        cache.mkdir(mode=0o700); work.mkdir(mode=0o700)
        # Only this worker's process environment changes. Explicit set_option
        # below is also required; changing env alone is not a verified fix.
        os.environ['ASCEND_CACHE_PATH'] = str(cache)
        os.environ['ASCEND_WORK_PATH'] = str(work)
        os.environ['ACL_OP_COMPILER_CACHE_DIR'] = str(cache)
        os.environ['ACL_OP_COMPILER_CACHE_MODE'] = 'enable'
        report = dict(phase='owned_paths_ready', rank=int(self.rank), pid=os.getpid(),
            configuration_scope='fresh_cache_plus_options_plus_internal_format_plus_JIT_true',
            model_loader_calls=0, product_math_changed=False, NZ_conversion_skipped=False,
            JIT_enabled_for_entire_finite_diagnostic=True, NPU_or_PR_accepted=False)
        _publish(scope, report)
        started = time.perf_counter()
        try:
            import torch

            report['phase'] = 'cache_options_enter'
            _publish(scope, report)
            torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE': 'enable',
                                  'ACL_OP_COMPILER_CACHE_DIR': str(cache)})
            report.update(phase='internal_format_enter', cache_options_returned=True)
            _publish(scope, report)
            torch.npu.config.allow_internal_format = True
            report.update(phase='compile_mode_enter', internal_format_setter_returned=True)
            _publish(scope, report)
            torch.npu.set_compile_mode(jit_compile=True)
            report.update(phase='original_load_model_enter', compile_mode_returned=True, model_loader_calls=1)
            _publish(scope, report)
            result = super().load_model()
            report.update(phase='original_load_model_returned', load_model_returned=True)
            _publish(scope, report)
            return result
        except BaseException as error:
            report.update(failed=True, error_type=type(error).__name__)
            # Preserve the original framework/compiler exception if even the
            # diagnostic filesystem write fails during exceptional unwinding.
            try:
                _publish(scope, report)
            except Exception:
                pass
            raise
        finally:
            report['configuration_and_weight_load_wall_ms'] = (time.perf_counter() - started) * 1000
            # Do not turn an observer write into a successful weight load.
            try:
                _publish(scope, report)
            except Exception:
                if report.get('load_model_returned'):
                    raise
