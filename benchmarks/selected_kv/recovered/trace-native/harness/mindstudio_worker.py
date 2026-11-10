"""Test-only per-worker MindStudio capture; production scheduling stays untouched."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import re
import threading
import time

from compare_worker import CompareWorker
from perf_common import atomic


class MindStudioWorker(CompareWorker):
    @contextmanager
    def _mstx(self, label, stream=None):
        import torch_npu
        scope = 'dummy' if getattr(self, '_bench_dummy', False) or label.startswith('dummy.') else 'actual'
        name = 'Native/' + scope + '/' + label
        token = torch_npu.npu.mstx.range_start(name, stream=stream) if self._profile_selected else 0
        if self._profile_selected and (not isinstance(token, int) or token <= 0):
            raise RuntimeError('MSTX range did not start: ' + label)
        try:
            yield
        finally:
            if self._profile_selected:
                torch_npu.npu.mstx.range_end(token)
            with self._profile_count_lock:
                self._profile_counts[name] = self._profile_counts.get(name, 0) + 1

    def _profile_patch(self, obj, name, label, stream=None, fence=False):
        original = getattr(obj, name)
        if not callable(original):
            raise RuntimeError("Missing profiling target: " + name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            selected_stream = stream(*args, **kwargs) if stream is not None else None
            value = label(*args, **kwargs) if callable(label) else label
            with self._mstx(value, selected_stream):
                result = original(*args, **kwargs)
            if fence and not getattr(self, "_bench_dummy", False):
                self._profile_fences += 1
                self._profile_last_fence_ns = time.monotonic_ns()
            return result
        setattr(obj, name, wrapped)
        self._profile_patches.append((obj, name, original))

    def _install_mstx(self):
        import torch
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        group = get_kv_transfer_group()
        worker = group.connector_worker
        compute = lambda *a, **k: torch.npu.current_stream()
        self._profile_patch(self.model_runner, "execute_model", "runner.execute_model", compute)
        self._profile_patch(self.model_runner, "_dummy_run", "dummy._dummy_run", compute)
        self._profile_patch(group, "start_load_kv", "connector.prepare.host")
        self._profile_patch(group, "wait_for_layer_load", "SDK.wait_for_layer_load.host")
        self._profile_patch(worker, "save_kv_layer", "SDK.save_kv_layer.host")
        self._profile_patch(worker, "_wait_for_final_layer_save", "SDK.final_save_fence.host", fence=True)
        for direction, transfer in (("recv", worker.kv_recv_thread), ("send", worker.kv_send_thread)):
            self._profile_patch(transfer, "_batch_copy_with_limits",
                                lambda *a, _direction=direction, **k: "SDK." + _direction + ".batch_copy.host")
        seen, sfa, ffn = set(), 0, 0
        for layer in self.model_runner.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if impl is None or id(impl) in seen or not callable(getattr(impl, "_execute_sparse_flash_attention_process", None)):
                continue
            match = re.search(r"layers\.(\d+)", impl.layer_name)
            if match:
                seen.add(id(impl))
                self._profile_patch(impl, "_execute_sparse_flash_attention_process", "SFA.layer" + match[1], compute)
                sfa += 1
        for name, module in self.model_runner.model.named_modules():
            match = re.search(r"layers\.(\d+)\.(?:mlp|moe)$", name)
            if match:
                self._profile_patch(module, "forward", "FFN.layer" + match[1], compute)
                ffn += 1
        if (sfa, ffn) != (78, 78):
            raise RuntimeError("Profiling requires all 78 SFA and FFN targets")
        self._profile_target_counts = dict(SFA=sfa, FFN=ffn)

    def _profile_ack(self, action, **extra):
        value = dict(status='ack' if self._profile_selected else 'skipped', action=action, trial=self._profile_trial, dp_rank=self._bench_dp, tp_rank=self._bench_tp, pid=os.getpid(), physical_die=self._profile_die, trace_rank_dir=str(self._profile_dir) if self._profile_selected else None, **extra, global_rank=self._bench_dp * 4 + self._bench_tp, profiler_selected=self._profile_selected, profiler_skipped=not self._profile_selected, skip_reason=None if self._profile_selected else 'not_global_rank0')
        path = Path(os.environ['BENCH_OUTPUT']) / 'workers' / f'profile-{action}-{self._profile_trial}-dp{self._bench_dp}-tp{self._bench_tp}.json'
        atomic(path, value, private=True)
        return value

    def bench_profile_start(self, trial):
        self._setup()
        row = getattr(self, '_bench_current', None)
        if row is None or row['phase'] != 'profile' or row['trial'] != trial:
            raise RuntimeError('Profiler start requires this profile burst before resume')
        if getattr(self, '_profile_state', None) is not None:
            raise RuntimeError('Only one profile burst is allowed')
        import torch
        import torch_npu
        from torch_npu.profiler.experimental_config import _ExperimentalConfig, ProfilerLevel, ExportType
        logical = int(torch.npu.current_device())
        visible = os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
        self._profile_die = int(visible[logical])
        self._profile_trial = trial
        self._profile_selected = self._bench_dp == 0 and self._bench_tp == 0
        name = f'dp{self._bench_dp}-tp{self._bench_tp}-die{self._profile_die}-pid{os.getpid()}'
        self._profile_dir = Path(os.environ['BENCH_OUTPUT']).parent / 'trace' / 'raw' / name if self._profile_selected else None
        if self._profile_selected:
            self._profile_dir.mkdir(parents=True, exist_ok=False)
        self._profile_patches = []
        self._profile_counts = {}
        self._profile_count_lock = threading.Lock()
        self._profile_fences = 0
        self._profile_last_fence_ns = None
        if self._profile_selected:
            options = _ExperimentalConfig(profiler_level=ProfilerLevel.Level2, mstx=True, data_simplification=False, record_op_args=False, export_type=[ExportType.Text, ExportType.Db])
            handler = torch_npu.profiler.tensorboard_trace_handler(dir_name=str(self._profile_dir), worker_name=name, analyse_flag=False)
            self._mindstudio = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU], on_trace_ready=handler, record_shapes=False, profile_memory=False, with_stack=False, with_flops=False, with_modules=False, experimental_config=options)
            self._mindstudio.start()
        else:
            self._mindstudio = None
        self._profile_state = 'active'
        self._profile_started_ns = time.monotonic_ns()
        self._install_mstx()
        return self._profile_ack('start', started_ns=self._profile_started_ns, target_counts=self._profile_target_counts, starts_before_resume=True, parsed_in_worker=False)

    def bench_profile_stop(self, trial):
        row = getattr(self, '_bench_current', None)
        if getattr(self, '_profile_state', None) != 'active' or trial != self._profile_trial or row is None or (row['phase'] != 'profile') or (row['trial'] != trial):
            raise RuntimeError('Profiler stop does not match the active profile burst')
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        worker = get_kv_transfer_group().connector_worker
        pending_before = bool(getattr(worker, '_pending_last_save_drain', False))
        if pending_before:
            with self._mstx('PROFILE_DRAIN.host'):
                worker._drain_deferred_last_save()
        pending_after = bool(getattr(worker, '_pending_last_save_drain', False))
        sender = worker.kv_send_thread
        sender.raise_if_failed()
        with sender.request_queue.all_tasks_done:
            unfinished = int(sender.request_queue.unfinished_tasks)
        if pending_after or unfinished:
            raise RuntimeError('Final SDK save queue has not drained')
        if self._profile_fences < 1 or self._profile_last_fence_ns is None:
            raise RuntimeError('Existing final D2H save fence has not completed')
        stop_begin = time.monotonic_ns()
        try:
            if self._profile_selected:
                self._mindstudio.stop()
        finally:
            for obj, name, original in reversed(self._profile_patches):
                setattr(obj, name, original)
        if self._profile_selected and (not self._mindstudio.stopped):
            raise RuntimeError('Profiler did not stop')
        raw = sorted((str(p) for p in self._profile_dir.iterdir() if p.is_dir())) if self._profile_selected else []
        if self._profile_selected and (not raw):
            raise RuntimeError('Profiler did not create a raw capture directory')
        self._profile_state = 'stopped'
        return self._profile_ack('stop', stopped_ns=time.monotonic_ns(), stop_begin_ns=stop_begin, last_existing_save_fence_ns=self._profile_last_fence_ns, stop_after_existing_save_fence=self._profile_last_fence_ns <= stop_begin, fence_completions=self._profile_fences, marker_counts=dict(self._profile_counts), pending_last_save_drain_before=pending_before, pending_last_save_drain_after=pending_after, sender_unfinished_tasks=unfinished, sender_failure_checked=True, full_D2H_completion_proven=True, defer_last_save_env=os.environ.get('VLLM_ASCEND_LW_DEFER_LAST_SAVE'), raw_capture_dirs=raw, parsed_in_worker=False, added_layer_synchronizations=0)
