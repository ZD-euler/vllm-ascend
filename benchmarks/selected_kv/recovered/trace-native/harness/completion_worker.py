"""Test-only completion ablation; mode changes happen between drained bursts."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import time

from compare_worker import CompareWorker
from perf_common import atomic


def baseline_methods(path, module):
    tree = ast.parse(path.read_text())
    wanted = {
        "SelectedKVSource": {"load_indexer"},
        "SelectedLoadStep": {"native_layer_ready", "finish"},
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            methods = [item for item in node.body if isinstance(item, ast.FunctionDef)
                       and item.name in wanted[node.name]]
            if {item.name for item in methods} != wanted[node.name]:
                raise ValueError("Baseline methods are incomplete")
            classes.append(ast.ClassDef(name=node.name, bases=[], keywords=[],
                                        body=methods, decorator_list=[]))
    if len(classes) != 2:
        raise ValueError("Baseline source classes are incomplete")
    body = ast.Module(body=[ast.ImportFrom(module="__future__",
                      names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[])
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(body), str(path), "exec"), namespace)
    return tuple((getattr(module, cls), name, getattr(namespace[cls], name))
                 for cls, names in wanted.items() for name in sorted(names))


class CompletionWorker(CompareWorker):
    def _selected_state(self):
        state = super()._selected_state()
        if (getattr(self, "_bench_current", None) is not None
                and not getattr(self, "_bench_dummy", False) and state[1] is not None):
            # The base observer calls this after the real forward's end event.
            # Later DP dummy steps may retire/clear source.step before the RPC.
            self._completion_observed_step = state[1]
            self._completion_observed_trial = self._bench_current["trial"]
        return state

    def bench_set_li_completion(self, mode, oracle=False):
        if mode != "event" or oracle:
            raise ValueError("Full78 uses production event completion, with byte oracle disabled")
        if getattr(self, "_bench_current", None) is not None:
            raise RuntimeError("Cannot prepare a new mode during a burst")
        self._completion_mode = "event"
        self._completion_observed_step = None
        self._completion_observed_trial = None
        result = super().bench_set_li_transport("mte", oracle=False)
        result["completion_mode"] = "event"
        return result

    def bench_finish_li_completion(self, oracle=False):
        result = super().bench_finish_li_transport(oracle=oracle)
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        source = get_kv_transfer_group().connector_worker.selected_kv_source
        step = getattr(self, "_completion_observed_step", None)
        if step is None or not step.indexer_mte:
            raise RuntimeError("Actual LI MTE step was not observed")
        expected = sum(item is not None for item in source.indexers)
        count = len(step._indexer_events)
        if count != (expected if self._completion_mode == "event" else 0):
            raise AssertionError("Actual completion path differs from requested mode")
        if len(step._native_layers) != len(source.components):
            raise AssertionError("Not every real layer joined its load dependency")
        result.update(completion_mode=self._completion_mode, actual_LI_events=count,
                      expected_LI_owners=expected, joined_layers=len(step._native_layers),
                      dp_rank=self._bench_dp, tp_rank=self._bench_tp,
                      trial=self._completion_observed_trial)
        # Some core versions return utility results from DP0 only. Each worker
        # persists its own post-ROI proof; the frontend requires all four.
        path = Path(os.environ["BENCH_OUTPUT"]) / "workers" / (
            f"completion-{self._completion_observed_trial}-dp{self._bench_dp}-tp{self._bench_tp}.json")
        atomic(path, result, private=True)
        return result
