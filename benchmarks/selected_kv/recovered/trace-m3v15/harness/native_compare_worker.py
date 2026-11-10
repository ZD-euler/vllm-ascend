"""Generic four-rank observer used by all three code paths.

Only one boundary pair is recorded around the real model forward.  Route and
selected-path readback happen after the first-token endpoint in an RPC.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import threading
from functools import wraps

import numpy as np

import spec
from perf_common import atomic, interval_union
from owned_process_utils_v1 import process

_MISSING = object()


def identity():
    fields = Path("/proc/self/stat").read_text().rpartition(")")[2].split()
    return dict(pid=os.getpid(), startticks=int(fields[19]))


class NativeCompareWorker:
    """Mixed into the selected source's NPUWorker at import time."""

    def init_device(self):
        output = Path(os.environ["BENCH_OUTPUT"])
        (output / "workers").mkdir(parents=True, exist_ok=True)
        proof = dict(identity(), rank=int(self.rank), initialized=False)
        atomic(output / "workers" / f"import-ready-{os.getpid()}.json", proof)
        gate = Path(os.environ["BENCH_ADMISSION"])
        wait_s = max(0.0, float(os.environ["BENCH_DEADLINE"]) - time.time()) if "BENCH_DEADLINE" in os.environ else 1800
        until = time.monotonic() + wait_s
        while time.monotonic() < until:
            value = json.loads(gate.read_text()) if gate.exists() else {}
            if (
                value.get("allow_npu") is True
                and value.get("expires", 0) > time.time()
                and any(
                    w.get("pid") == proof["pid"] and w.get("startticks") == proof["startticks"]
                    for w in value.get("workers", [])
                )
            ):
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("NPU admission gate expired")
        result = super().init_device()
        self._early_handoff = None
        if os.environ["BENCH_ARM"] != "v0":
            import torch
            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            from vllm.distributed import get_dp_group, get_world_group
            from early_store_handoff import EarlyStoreHandoff
            self._bench_hold = None
            if os.environ.get("BENCH_LATE_SDK") == "1":
                return result
            try: transfer = get_kv_transfer_group()
            except AssertionError: transfer = None
            existing = getattr(getattr(transfer, "connector_worker", None), "m_store", None)
            if not (existing is not None and getattr(existing, "_init_bm", False) and getattr(existing, "store", None) is not None):
                if "__del__" in vars(MemcacheBackend): raise RuntimeError("unsupported backend destructor")
                extra = self.vllm_config.kv_transfer_config.kv_connector_extra_config
                owner = MemcacheBackend(self.parallel_config, device_id=int(torch.npu.current_device()), init_bm=True, lazy_init=False, extra_config=extra)
                torch.distributed.barrier(group=get_world_group().cpu_group)
                self._early_handoff = EarlyStoreHandoff(MemcacheBackend, owner, barrier=lambda: torch.distributed.barrier(group=get_world_group().cpu_group), pause=lambda: time.sleep(.1))
        return result

    def load_model(self):
        import torch

        # Preserve the established startup probe: weights load while a large
        # temporary allocation is held, then the hold is released immediately
        # before the ordinary automatic KV-memory measurement.
        if os.environ["BENCH_ARM"] == "v0":
            self._bench_hold = None
        return super().load_model()

    def determine_available_memory(self, *args, **kwargs):
        import torch

        self._bench_hold = None
        torch.npu.empty_cache()
        return super().determine_available_memory(*args, **kwargs)

    def _setup(self):
        if hasattr(self, "_bench_current"):
            return
        import torch
        from vllm.distributed import get_tp_group

        self._bench_dp = int(self.parallel_config.data_parallel_rank)
        self._bench_tp = int(get_tp_group().rank_in_group)
        self._bench_current = None
        self._bench_rows = []
        runner = self.model_runner
        self._bench_dummy = False
        if callable(getattr(runner, "_dummy_run", None)):
            dummy_original = runner._dummy_run
            @wraps(dummy_original)
            def dummy(*args, **kwargs):
                self._bench_dummy = True
                try: return dummy_original(*args, **kwargs)
                finally: self._bench_dummy = False
            runner._dummy_run = dummy
        original = runner.model.forward

        @wraps(original)
        def body(*args, **kwargs):
            row = self._bench_current
            if row is None:
                return original(*args, **kwargs)
            if self._bench_dummy:
                begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                entered = time.monotonic_ns()
                begin.record()
                try: return original(*args, **kwargs)
                finally:
                    end.record(); row["_dummy_events"].append((begin, end))
                    row["_dummy_host"].append((entered, time.monotonic_ns()))
            n = runner.input_batch.num_reqs
            history = np.asarray(runner.input_batch.num_computed_tokens_cpu[:n]).tolist()
            starts = runner.query_start_loc
            if hasattr(starts, "np"):
                starts = starts.np
            elif getattr(starts, "device", None) is not None and starts.device.type == "cpu":
                starts = starts.detach().numpy()
            else:
                raise RuntimeError("query offsets require a CPU shadow; refusing timed device readback")
            query = np.diff(starts[:n + 1]).tolist()
            row["_actual_H"].extend(history)
            row["_actual_Q"].extend(query)
            row["_scheduled"].extend(query)
            def shape(value):
                return list(value.shape) if hasattr(value, "shape") else None
            row["_geometries"].append(dict(H=history, Q=query, request_count=n, token_count=sum(query), enter_ns=time.monotonic_ns(), execute_index=len(row["_executions"])-1, dummy=self._bench_dummy, positional_shapes=[shape(v) for v in args[:4]], keyword_shapes={k: shape(kwargs.get(k)) for k in ("input_ids", "inputs_embeds", "positions")}))
            row["_forward_calls"] += 1
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            if row["_enter_ns"] is None: row["_enter_ns"] = time.monotonic_ns()
            begin.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                row["_return_ns"] = time.monotonic_ns()
                row["_geometries"][-1]["return_ns"] = row["_return_ns"]
                row["_events"].append((begin, end))
                _, step, routes, enabled = self._selected_state()
                row["_route_steps"].append(dict(present=step is not None, routes=routes, enabled=enabled, li_event_count=len(getattr(step, "_indexer_events", {})), indexer_mte=bool(getattr(step, "indexer_mte", False))))

        runner.model.forward = body

    def _continuation_is_final(self, row, output):
        cached = getattr(output, "scheduled_cached_reqs", None)
        computed = {}
        if cached is not None:
            ids = getattr(cached, "req_ids", ())
            values = getattr(cached, "num_computed_tokens", ())
            computed.update(zip(ids, values))
        for request in getattr(output, "scheduled_new_reqs", ()):
            computed[request.req_id] = request.num_computed_tokens
        scheduled = dict(output.num_scheduled_tokens)
        actual = [computed.get(key, -1) for key in scheduled]
        return (
            len(actual) == row["_local_BS"]
            and actual == [row["H"]] * row["_local_BS"]
            and sorted(scheduled.values()) == sorted(row["_local_Qs"])
        )

    def execute_model(self, *args, **kwargs):
        row = getattr(self, "_bench_current", None)
        if row is None:
            return super().execute_model(*args, **kwargs)
        output = args[0] if args else kwargs["scheduler_output"]
        scheduled = list(output.num_scheduled_tokens.values())
        row["_executions"].append(dict(scheduled_counts=scheduled, request_count=len(scheduled), token_count=sum(scheduled), enter_ns=time.monotonic_ns()))
        if not row.get("_continuation"):
            return super().execute_model(*args, **kwargs)
        if self._continuation_is_final(row, output):
            return super().execute_model(*args, **kwargs)
        # The 1/127-token cold prefix continuation is preparation, not ROI.
        self._bench_current = None
        try:
            return super().execute_model(*args, **kwargs)
        finally:
            self._bench_current = row

    def _selected_state(self):
        runtime = getattr(self.model_runner, "selected_kv_runtime", None)
        step = getattr(runtime, "source_step", None)
        requests = tuple(step.requests) if step is not None else ()
        routes = [bool(request.selected) for request in requests]
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            selected = getattr(get_ascend_config(), "selected_kv_config", None)
            enabled = bool(getattr(selected, "enabled", False))
        except Exception:
            runtime, step, routes, enabled = None, None, [], False
        return runtime, step, routes, enabled

    def _v0_state(self):
        manager = getattr(self.model_runner, "prefill_selection_manager", None)
        config = getattr(manager, "config", None)
        enabled = bool(
            manager is not None
            and getattr(config, "enabled", False) is True
            and getattr(config, "request_layer_enabled", False) is True
        )
        return enabled

    def _probe_install(self):
        import torch
        self._probe_patches = []
        self._probe_events = []
        self._probe_counts = {"sfa": {}, "ffn": {}, "load": {}, "selected_copy": {}, "union": {}}
        self._probe_statuses = []
        self._probe_sdk_ms = []
        self._probe_li_mte_ms = []
        def patch(obj, name, kind, layer):
            original = getattr(obj, name)
            @wraps(original)
            def wrapped(*args, **kwargs):
                if self._bench_dummy: return original(*args, **kwargs)
                counts = self._probe_counts[kind]
                counts[layer] = counts.get(layer, 0) + 1
                a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                a.record()
                try:
                    return original(*args, **kwargs)
                finally:
                    b.record()
                    self._probe_events.append((kind, a, b))
            setattr(obj, name, wrapped)
            self._probe_patches.append((obj, name, original))
        seen = set()
        for layer in self.model_runner.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if impl is None or id(impl) in seen or not callable(getattr(impl, "_execute_sparse_flash_attention_process", None)):
                continue
            match = re.search(r"layers\.(\d+)", impl.layer_name)
            if match:
                seen.add(id(impl)); patch(impl, "_execute_sparse_flash_attention_process", "sfa", int(match[1]))
        arm = os.environ["BENCH_ARM"]
        self._sdk_copy_observed = False
        if arm != "v0":
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            connector = get_kv_transfer_group().connector_worker
            if not callable(getattr(connector, "wait_for_layer_load", None)):
                raise RuntimeError("The native connector must expose wait_for_layer_load")
            patch(connector, "wait_for_layer_load", "load", 0)
            recv = getattr(connector, "kv_recv_thread", None)
            original_handle = getattr(recv, "_handle_request", None)
            original_copy = getattr(recv, "_batch_copy_with_limits", None)
            if callable(original_handle) and callable(original_copy):
                self._sdk_copy_observed = True
                local = threading.local()
                @wraps(original_handle)
                def handle(*args, **kwargs):
                    local.active = True
                    try: return original_handle(*args, **kwargs)
                    finally: local.active = False
                recv._handle_request = handle
                self._probe_patches.append((recv, "_handle_request", original_handle))
                @wraps(original_copy)
                def copy(gvas, addresses, sizes, direction, *args, **kwargs):
                    if not getattr(local, "active", False) or direction != 1:
                        return original_copy(gvas, addresses, sizes, direction, *args, **kwargs)
                    started = time.perf_counter()
                    result = original_copy(gvas, addresses, sizes, direction, *args, **kwargs)
                    self._probe_sdk_ms.append((time.perf_counter() - started) * 1000)
                    return result
                recv._batch_copy_with_limits = copy
                self._probe_patches.append((recv, "_batch_copy_with_limits", original_copy))
        if arm == "m3v9":
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_plan import SelectedKVPlanner
            original = SelectedKVPlanner.build
            @wraps(original)
            def build(planner, *args, **kwargs):
                a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                a.record()
                try:
                    value = original(planner, *args, **kwargs)
                    self._probe_statuses.append(value[1].clone())
                    return value
                finally:
                    b.record(); self._probe_events.append(("union", a, b))
            SelectedKVPlanner.build = build
            self._probe_patches.append((SelectedKVPlanner, "build", original))
            patch(SelectedKVPlanner, "load", "selected_copy", 1)
        for name, module in self.model_runner.model.named_modules():
            match = re.search(r"layers\.(\d+)\.(?:mlp|moe)$", name)
            if match: patch(module, "forward", "ffn", int(match[1]))

    def _probe_collect(self, anchor):
        for obj, name, original in reversed(self._probe_patches): setattr(obj, name, original)
        values = {"sfa": 0., "ffn": 0., "load": 0., "selected_copy": 0., "union": 0.}
        spans = []
        for kind, begin, end in self._probe_events:
            end.synchronize(); values[kind] += float(begin.elapsed_time(end))
            if kind in ("load", "selected_copy"): spans.append((anchor.elapsed_time(begin), anchor.elapsed_time(end)))
        values["exposed"] = interval_union(spans)
        values["union_errors"] = sum(int((s[:, 3:5] != 0).sum().item()) for s in self._probe_statuses)
        if os.environ["BENCH_ARM"] == "v0":
            main = self.model_runner.prefill_selection_manager
            li = self.model_runner.prefill_indexer_layerwise
            loads, waits = [], []
            for record in main.records.values():
                for tile in record.get("tiles", ()):
                    for key, out in (("load", loads), ("wait", waits)):
                        a, b = tile[key]; b.synchronize(); out.append((anchor.elapsed_time(a), anchor.elapsed_time(b)))
            summary = li.summarize(li.snapshot(), anchor)
            loads += [tuple(v["load_span_ms"]) for v in summary["layers"]]
            waits += [tuple(v["wait_span_ms"]) for v in summary["layers"]]
            values["v0_total"] = sum(b-a for a,b in loads)
            values["exposed"] = interval_union(waits)
            values["union_errors"] = sum(int((p["buffers"][1][:len(main.batch_queries),3] != 0).sum().item()) for p in main.batch_plans.values())
        return values

    def bench_set_li_transport(self, mode, oracle=False):
        """Test-only transition while frontend is paused between completed bursts."""
        if mode not in ("sdk", "mte") or (oracle and mode != "mte"):
            raise ValueError("invalid LI transport request")
        if getattr(self, "_bench_current", None) is not None:
            raise RuntimeError("transport cannot change during a burst")
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        connector = get_kv_transfer_group().connector_worker
        source = connector.selected_kv_source
        if source is None:
            raise RuntimeError("selected source not initialized")
        if source.step is not None:
            source.step.wait_completion()
        recv = connector.kv_recv_thread
        if getattr(self, "_li_oracle_original", None) is not None:
            raise RuntimeError("previous oracle callback remains installed")
        source.indexer_mte_enabled = mode == "mte"
        self._li_mode = mode
        self._li_oracle_records = []
        if oracle:
            from li_mte_oracle import compare_one_layer
            original = recv.selected_indexer_load
            if not callable(original):
                raise RuntimeError("native indexer callback is absent")
            def callback(layer_id):
                original(layer_id)
                result = compare_one_layer(source, layer_id)
                if result is not None:
                    self._li_oracle_records.append(result)
            self._li_oracle_original = original
            recv.selected_indexer_load = callback
        return dict(mode=mode, oracle=oracle, source_step_waited=True)

    def bench_finish_li_transport(self, oracle=False):
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        recv = get_kv_transfer_group().connector_worker.kv_recv_thread
        original = getattr(self, "_li_oracle_original", None)
        if original is not None:
            recv.selected_indexer_load = original
            self._li_oracle_original = None
        records = getattr(self, "_li_oracle_records", [])
        source = get_kv_transfer_group().connector_worker.selected_kv_source
        expected = {i for i, item in enumerate(source.indexers) if item is not None}
        if oracle and (len(records) != len(expected) or {r["layer"] for r in records} != expected or any(not r["passed"] for r in records)):
            raise AssertionError("LI byte oracle did not verify every owner")
        return dict(mode=getattr(self, "_li_mode", None), oracle=oracle, dp_rank=self._bench_dp, tp_rank=self._bench_tp, records=records)

    def bench_begin(self, case_value, phase, trial):
        self._setup()
        if self._early_handoff is not None and not self._early_handoff.claimed:
            raise RuntimeError("original connector did not consume early SDK store")
        case_value = spec.geometry(case_value)
        if self._bench_current is not None:
            raise RuntimeError("previous benchmark burst was not collected")
        local_qs = list(case_value["request_Qs"][self._bench_dp::spec.DP])
        if len(local_qs) != 1:
            raise ValueError("local request count must be one or eight")
        import torch

        if phase == "probe":
            self._probe_install()
        torch.npu.reset_peak_memory_stats(self.device)
        self._bench_current = dict(
            trial=trial,
            phase=phase,
            case=case_value["case"],
            H=case_value["H"],
            _local_BS=len(local_qs),
            _local_Qs=local_qs,
            _continuation=bool(case_value.get("continuation", False)),
            _events=[],
            _dummy_events=[],
            _dummy_host=[],
            _geometries=[],
            _executions=[],
            _route_steps=[],
            _actual_H=[],
            _actual_Q=[],
            _scheduled=[],
            _forward_calls=0,
            _enter_ns=None,
            _return_ns=None,
        )
        return {"dp": self._bench_dp, "tp": self._bench_tp, "trial": trial}

    def bench_collect(self, trial):
        import torch

        row = self._bench_current
        if row is None:
            raise RuntimeError("benchmark collect without begin")
        self._bench_current = None
        if row["trial"] != trial:
            raise ValueError("benchmark trial identity changed")
        if not row["_events"]:
            raise AssertionError("no measured model forward")
        begin = row["_events"][0][0]
        steps = []
        for start, end in row["_events"]:
            end.synchronize()
            steps.append(float(start.elapsed_time(end)))
        model_ms = sum(steps)
        dummy_steps = []
        for a, b in row["_dummy_events"]:
            b.synchronize(); dummy_steps.append(float(a.elapsed_time(b)))
        actual_H = row["_actual_H"] or []
        actual_Q = row["_actual_Q"] or []
        scheduled = row["_scheduled"] or []
        issues = []
        if actual_H != [row["H"]] * row["_local_BS"]:
            issues.append("actual_history_mismatch")
        if sorted(actual_Q) != sorted(row["_local_Qs"]):
            issues.append("actual_query_mismatch")
        if sorted(scheduled) != sorted(row["_local_Qs"]):
            issues.append("scheduled_query_mismatch")
        if row["_forward_calls"] != 1:
            issues.append("model_forward_count")
        runtime, step, routes, selected_enabled = self._selected_state()
        if not routes:
            routes = [False] * len(actual_Q)
        expected_feature = spec.FEATURE_ENABLED
        route_verified = (not any(routes) and len(row["_route_steps"]) == len(steps)
            and all(not any(value["routes"]) for value in row["_route_steps"]))
        observed_li_events = sum(value.get("li_event_count", 0) for value in row["_route_steps"])
        if selected_enabled != expected_feature:
            issues.append("selected_feature_configuration_mismatch")
        if not route_verified or observed_li_events or any(value.get("indexer_mte", False) for value in row["_route_steps"]):
            issues.append("unexpected_selected_or_LI_MTE_on_native")
        v0_enabled = False
        arm = os.environ["BENCH_ARM"]
        peak_bytes = int(torch.npu.max_memory_allocated(self.device))
        probe = self._probe_collect(begin) if row["phase"] == "probe" else None
        if probe is not None and arm == "v0" and probe["union_errors"]:
            issues.append("v0_union_error")
        if probe is not None and arm == "m3v9" and (not self._probe_statuses or probe["union_errors"]):
            issues.append("selected_union_unverified_or_error")
        if probe is not None and (self._probe_counts["ffn"] != {i: len(steps) for i in range(spec.LAYERS)} or self._probe_counts["sfa"] != {i: len(steps) for i in range(spec.LAYERS)}):
            issues.append("probe_layer_call_count")
        result = dict(
            **{k: v for k, v in row.items() if not k.startswith("_")},
            pid=os.getpid(),
            dp_rank=self._bench_dp,
            tp_rank=self._bench_tp,
            actual_H=actual_H,
            actual_Q=actual_Q,
            scheduled_Q=scheduled,
            model_forward_calls=row["_forward_calls"],
            model_npu_ms=model_ms,
            model_steps_ms=steps,
            scheduled_steps=len(steps),
            dummy_model_forward_calls=len(dummy_steps),
            dummy_model_npu_ms=sum(dummy_steps),
            dummy_host_intervals_ns=row["_dummy_host"],
            step_geometry=row["_geometries"],
            execute_schedule=row["_executions"],
            model_enter_ns=row["_enter_ns"],
            model_return_ns=row["_return_ns"],
            allocated_peak_bytes=peak_bytes,
            selected_enabled=selected_enabled,
            selected_route=routes,
            selected_route_count=sum(routes),
            selected_route_verified=route_verified,
            observed_LI_events=observed_li_events,
            selected_feature_expected=expected_feature,
            selected_step_present=step is not None,
            v0_request_layer_enabled=v0_enabled,
            li_transport=getattr(self, "_li_mode", None),
            sfa_ms=probe["sfa"] if probe else None,
            ffn_ms=probe["ffn"] if probe else None,
            sfa_calls=self._probe_counts["sfa"] if probe else None,
            ffn_calls=self._probe_counts["ffn"] if probe else None,
            total_load_ms=(probe["v0_total"] if arm == "v0" else sum(self._probe_sdk_ms) + (probe["selected_copy"] if arm == "m3v9" else 0.)) if probe and getattr(self, "_sdk_copy_observed", False) else None,
            sdk_load_wall_ms=sum(self._probe_sdk_ms) if probe and arm != "v0" and getattr(self, "_sdk_copy_observed", False) else None,
            li_mte_callback_wall_ms=sum(self._probe_li_mte_ms) if probe and arm == "m3v9" else None,
            li_mte_callback_count=len(self._probe_li_mte_ms) if probe and arm == "m3v9" else None,
            selected_main_copy_ms=probe["selected_copy"] if probe and arm == "m3v9" else None, exposed_load_ms=probe["exposed"] if probe else None,
            union_ms=probe["union"] if probe and arm == "m3v9" else None,
            selected_union_error_flags=probe["union_errors"] if probe and arm in ("m3v9", "v0") else None,
            valid=not issues,
            issues=issues,
        )
        self._sdk_last_result = result
        path = Path(os.environ["BENCH_OUTPUT"]) / "workers" / f"metric-{trial}-{self._bench_dp}-{self._bench_tp}.json"
        atomic(path, result, private=True)
        if issues and row["phase"] not in ("warmup", "diagnostic"):
            raise AssertionError("rank acceptance failed: " + ",".join(issues))
        return result

    def bench_set_li_completion(self, mode, oracle=False):
        if mode != "sdk" or oracle:
            raise ValueError("Native-only cases require the original SDK path")
        self._setup()
        if self._bench_current is not None:
            raise RuntimeError("Cannot configure transport during a burst")
        self._li_mode = "sdk"
        return dict(mode="sdk", oracle=False, transport_modified=False)

    def bench_finish_li_completion(self, oracle=False):
        row = self._sdk_last_result
        if oracle or not row["valid"] or any(row["selected_route"]):
            raise AssertionError("The native-only SDK route was not verified")
        result = dict(mode="sdk", completion_mode="sdk", oracle=False, records=[],
            actual_LI_events=row.get("observed_LI_events", 0), expected_LI_owners=None, joined_layers=None,
            native_wait_callback_count=(sum(self._probe_counts["load"].values()) if row["phase"] == "probe" else None),
            dp_rank=self._bench_dp, tp_rank=self._bench_tp, trial=row["trial"])
        path = Path(os.environ["BENCH_OUTPUT"]) / "workers" / f"completion-{row['trial']}-dp{self._bench_dp}-tp{self._bench_tp}.json"
        atomic(path, result, private=True)
        return result
