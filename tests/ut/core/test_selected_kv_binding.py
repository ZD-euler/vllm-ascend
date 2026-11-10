# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; no vLLM, CANN, SDK or device runtime is required."""

import ast
import builtins
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch


def load_modules():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    loaded = []
    with patch.dict(sys.modules):
        for name, path in (
            ("vllm_ascend.config_utils", root / "config_utils.py"),
            ("vllm_ascend.core.selected_kv_config", root / "core/selected_kv_config.py"),
            ("vllm_ascend.core.selected_kv_binding", root / "core/selected_kv_binding.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded[1:]


CONFIG, BINDING = load_modules()


class SelectedBindingImportTest(unittest.TestCase):
    def test_disabled_configuration_needs_no_worker_or_device_imports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if (
                name == "torch"
                or name == "vllm"
                or name.startswith(("vllm.", "vllm_ascend.worker.", "vllm_ascend.distributed."))
            ):
                self.fail(f"Disabled selected KV imported {name}")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", guarded_import):
            config, binding = load_modules()
            runner = NS(ascend_config=NS(selected_kv_config=config.SelectedKVConfig()))
            worker = NS(model_runner=runner)
            self.assertEqual(binding.reserve_selected_kv_workspace(worker, 4096), 4096)
            self.assertIsNone(binding.initialize_selected_kv(runner, None, {}))


class SchedulerConfigInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        scheduler_path = root / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
        tree = ast.parse(scheduler_path.read_text())
        scheduler = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KVPoolScheduler"
        )
        constructor = next(
            node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        # Execute the complete native constructor and its existing MagicMock
        # config fixture, substituting only backend/cache dependencies.
        fixture_tree = ast.parse((root / "tests/ut/distributed/ascend_store/test_pool_scheduler.py").read_text())
        fixture = next(
            node for node in fixture_tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_config"
        )
        body = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                ast.ClassDef(name="KVPoolScheduler", bases=[], keywords=[], body=[constructor], decorator_list=[]),
                fixture,
            ],
            type_ignores=[],
        )
        cls.global_config = Mock(side_effect=RuntimeError("global Ascend config is not initialized"))
        namespace = {
            "math": math,
            "envs": NS(),
            "MagicMock": MagicMock,
            "get_ascend_config": cls.global_config,
            "get_layerwise_protocol": lambda *_: None,
            "get_layerwise_data_plane": lambda *_: None,
            "validate_layerwise_topology": lambda *_: None,
            "validate_layerwise_runtime": lambda *_, **__: None,
            "uses_hybrid_kv_cache": lambda *_: False,
            "infer_group_cache_families": lambda *_: ["default"],
            "infer_group_block_sizes": lambda *_: [16],
            "infer_cacheable_group_ids": lambda *_: [0],
            "infer_cache_transfer_granularity": lambda *_: 16,
            "infer_tp_mismatch_info": lambda *_, **__: NS(enabled=False),
            "backend_map": {"mooncake": {"path": "unused", "name": "Backend"}},
            "importlib": NS(import_module=lambda _: NS(Backend=NS(create_scheduler_client=lambda _: NS()))),
        }
        exec(compile(ast.fix_missing_locations(body), str(scheduler_path), "exec"), namespace)
        cls.scheduler_type = namespace["KVPoolScheduler"]
        cls.scheduler_type._infer_mamba_groups = lambda _: {}
        cls.scheduler_type._build_cache_coordinator = lambda _: None
        cls.make_config = staticmethod(namespace["make_config"])

    def test_bare_native_magicmock_needs_no_global_initialization(self):
        scheduler = self.scheduler_type(self.make_config(), use_layerwise=False)
        self.assertFalse(hasattr(scheduler, "_selected_kv_enabled"))
        self.assertEqual(scheduler.grouped_block_size, [16])
        self.assertEqual(scheduler.num_layers, 2)
        self.global_config.assert_not_called()

    def test_selected_configuration_does_not_change_native_scheduler_initialization(self):
        for additional in (
            None,
            {},
            {"selected_kv_config": {}},
            {"selected_kv_config": {"enabled": False}},
            {"selected_kv_config": {"enabled": 1}},
            {"selected_kv_config": {"enabled": "true"}},
            {"selected_kv_config": {"enabled": MagicMock()}},
            {"selected_kv_config": MagicMock()},
            {"selected_kv_config": CONFIG.SelectedKVConfig()},
            {"selected_kv_config": {"enabled": True}},
            {"selected_kv_config": CONFIG.SelectedKVConfig(enabled=True)},
        ):
            with self.subTest(additional=additional):
                config = self.make_config()
                config.additional_config = additional
                scheduler = self.scheduler_type(config, use_layerwise=False)
                self.assertFalse(hasattr(scheduler, "_selected_kv_enabled"))
        self.global_config.assert_not_called()


def deployment():
    return NS(
        kv_transfer_config=NS(
            kv_role="kv_producer",
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"backend": "memcache", "use_layerwise": True},
        ),
        scheduler_config=NS(
            async_scheduling=False, scheduler_reserve_full_isl=True, max_num_seqs=64, max_num_batched_tokens=8192
        ),
        model_config=NS(enforce_eager=True, max_model_len=1048576),
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=None,
        cache_config=NS(kv_cache_memory_bytes=None),
    )


class SelectedConfigTest(unittest.TestCase):
    def test_default_is_disabled_without_new_deployment_constraints(self):
        self.assertFalse(CONFIG.SelectedKVConfig().enabled)
        CONFIG.SelectedKVConfig().validate_deployment(None)

    def test_payload_partition_is_no_longer_a_configuration(self):
        with self.assertRaises(ValueError):
            CONFIG.SelectedKVConfig(partition_rows=524288)
        self.assertEqual(set(CONFIG.SelectedKVConfig.__dataclass_fields__), {"enabled"})

    def test_native_full_isl_and_long_context_are_allowed(self):
        CONFIG.SelectedKVConfig(enabled=True).validate_deployment(deployment())

    def test_explicit_bytes_are_rejected_without_changing_the_value(self):
        config = deployment()
        config.cache_config.kv_cache_memory_bytes = 123456789
        with self.assertRaisesRegex(ValueError, "explicit bytes"):
            CONFIG.SelectedKVConfig(enabled=True).validate_deployment(config)
        self.assertEqual(config.cache_config.kv_cache_memory_bytes, 123456789)

    def test_unsupported_concurrency_is_rejected(self):
        for target, key, value in (
            ("scheduler_config", "async_scheduling", True),
            ("scheduler_config", "max_num_batched_tokens", 8193),
            ("scheduler_config", "max_num_seqs", 257),
            ("parallel_config", "pipeline_parallel_size", 2),
            ("parallel_config", "enable_dbo", True),
            ("model_config", "enforce_eager", False),
        ):
            config = deployment()
            setattr(getattr(config, target), key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONFIG.SelectedKVConfig(enabled=True).validate_deployment(config)

    def test_async_loading_is_rejected(self):
        config = deployment()
        config.kv_transfer_config.kv_connector_extra_config["load_async"] = True
        with self.assertRaises(ValueError):
            CONFIG.SelectedKVConfig(enabled=True).validate_deployment(config)


class StartupGeometryTest(unittest.TestCase):
    def setUp(self):
        self.names = ("owner", "follower", "next_owner", "next_follower")
        self.impls = {
            name: NS(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                local_num_heads=8,
                skip_topk="follower" in name,
                indexer=None if "follower" in name else NS(topk_output_width=2048),
            )
            for name in self.names
        }
        self.runner = NS(
            compilation_config=NS(static_forward_context={name: NS(impl=impl) for name, impl in self.impls.items()})
        )

    def test_valid_owners_and_followers_use_actual_owner_topk(self):
        # A follower's own indexer is unused, including when one exists.
        self.impls["next_follower"].indexer = NS(topk_output_width=128)
        owners = BINDING._topk_owners(self.runner, self.names)
        self.assertEqual(
            owners,
            {"owner": "owner", "follower": "owner", "next_owner": "next_owner", "next_follower": "next_owner"},
        )

    def test_query_head_count_is_not_consumed_by_selected_loading(self):
        self.impls["follower"].local_num_heads = 16
        del self.impls["next_owner"].local_num_heads
        self.impls["next_follower"].local_num_heads = None
        self.assertEqual(
            BINDING._topk_owners(self.runner, self.names),
            {"owner": "owner", "follower": "owner", "next_owner": "next_owner", "next_follower": "next_owner"},
        )

    def test_query_dimensions_rejected_for_owners_and_followers(self):
        for name in self.names:
            for field, value in (("kv_lora_rank", 256), ("qk_rope_head_dim", 0), ("qk_rope_head_dim", None)):
                with self.subTest(name=name, field=field, value=value):
                    impl = self.impls[name]
                    previous = getattr(impl, field)
                    setattr(impl, field, value)
                    with self.assertRaisesRegex(ValueError, "kv_lora_rank=512.*qk_rope_head_dim=64"):
                        BINDING._topk_owners(self.runner, self.names)
                    setattr(impl, field, previous)

    def test_invalid_or_missing_owner_topk_is_rejected(self):
        for name in ("owner", "next_owner"):
            for indexer in (None, NS(), NS(topk_output_width=1024)):
                with self.subTest(name=name, indexer=indexer):
                    previous = self.impls[name].indexer
                    self.impls[name].indexer = indexer
                    with self.assertRaisesRegex(ValueError, "actual TopK owner.*2048"):
                        BINDING._topk_owners(self.runner, self.names)
                    self.impls[name].indexer = previous

    def test_follower_cannot_precede_its_owner(self):
        self.impls["owner"].skip_topk = True
        with self.assertRaisesRegex(ValueError, "earlier actual TopK owner"):
            BINDING._topk_owners(self.runner, self.names)


class PhysicalBudgetTest(unittest.TestCase):
    def test_worker_normalizes_the_real_float_memory_budget(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/worker.py"
        tree = ast.parse(path.read_text())
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reserve_selected_kv_workspace"
        )
        expression = compile(ast.Expression(call.args[1]), str(path), "eval")
        for utilization in (0.5, 0.7, 0.88, 0.9, 0.95):
            for used in (0, 1, 2**30, 16 * 2**30, 20 * 2**30):
                available = 64 * 2**30 * utilization - used
                value = eval(expression, {"self": NS(available_kv_cache_memory_bytes=available)})
                self.assertIs(type(value), int)
                self.assertEqual(value, math.floor(available))
                plan = CONFIG.reserve_workspace_bytes(value, 100, 1000, 10)
                self.assertLessEqual(plan.available_bytes, available)

    def test_fixed_and_n_dependent_bytes_precede_native_factor(self):
        plan = CONFIG.reserve_workspace_bytes(10000, 100, 1000, 10, 200)
        self.assertEqual(plan.maximum_blocks, 98)
        self.assertEqual(plan.native_budget_bytes, 7820)
        # An ordinary core deriving N from native physical bytes still fits.
        plan.validate(plan.native_budget_bytes // 100)

    def test_override_cannot_bypass_workspace_charge(self):
        plan = CONFIG.reserve_workspace_bytes(10000, 100, 1000, 10, 200)
        with self.assertRaisesRegex(ValueError, "physical memory"):
            plan.validate(81)
        with self.assertRaisesRegex(ValueError, "block count"):
            plan.validate(99)
        plan.validate(80)

    def test_actual_padding_is_counted_once_and_oversize_rejected(self):
        plan = CONFIG.reserve_workspace_bytes(10000, 100, 1000, 10, 200)
        plan.validate(80, native_storage_bytes=8200)
        with self.assertRaisesRegex(ValueError, "physical memory"):
            plan.validate(80, native_storage_bytes=8201)

    def test_insufficient_or_invalid_budgets_fail_before_allocation(self):
        for args in (
            (1000, 100, 1000, 1),
            (100, 100, 1, 0),
            (100, 0, 1, 0),
            (-1, 100, 0, 0),
            (1000, 100, -1, 1),
            (True, 100, 0, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CONFIG.reserve_workspace_bytes(*args)

    def test_native_aliases_do_not_multiply_storage(self):
        first = Mock()
        first.untyped_storage.return_value = NS(data_ptr=lambda: 7, nbytes=lambda: 100)
        alias = Mock()
        alias.untyped_storage.return_value = NS(data_ptr=lambda: 7, nbytes=lambda: 100)
        second = Mock()
        second.untyped_storage.return_value = NS(data_ptr=lambda: 8, nbytes=lambda: 20)
        self.assertEqual(BINDING._unique_storage_bytes({"main": (first,), "alias": (alias, second)}), 120)


class WorkspaceIntegrationTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"
        self.modules = {}
        for name in ("prefill_selected_plan", "prefill_selected_workspace"):
            spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules["vllm_ascend.distributed.kv_transfer.sparse_kv_offload." + name] = module
        self.modules.update(
            {
                "vllm.v1.core.kv_cache_utils": NS(get_kv_cache_groups=lambda *_: [NS()]),
                "vllm_ascend.device.device_config": NS(get_ascend_device_type=lambda: "a3"),
                "vllm_ascend.device.hardware": NS(AscendDeviceType=NS(A3="a3")),
            }
        )
        vc = deployment()
        vc.scheduler_config.max_num_batched_tokens = 128
        vc.scheduler_config.max_num_seqs = 4
        vc.model_config.dtype = "bf16"
        self.worker = NS(vllm_config=vc, cache_config=vc.cache_config, get_kv_cache_spec=lambda: {})
        self.worker.model_runner = NS(
            vllm_config=vc,
            model_config=vc.model_config,
            ascend_config=NS(
                selected_kv_config=CONFIG.SelectedKVConfig(enabled=True),
                enable_dsa_cp=False,
                enable_sparse_sfa_c8=True,
                enable_sparse_li_c8=True,
                enable_sparse_li_c4=False,
                kvpp_config=NS(size=1),
            ),
        )

    def reserve(self, owners=None):
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(BINDING, "_native_geometry", return_value=(("a", "b"), 200000, 8388608)),
            patch.object(BINDING, "_topk_owners", return_value=owners or {"a": "a", "b": "b"}),
        ):
            return BINDING.reserve_selected_kv_workspace(self.worker, 2**30)

    def test_worker_reserves_actual_planner_executor_and_source_bytes(self):
        planner = self.modules["vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_plan"]
        executor = self.modules["vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_workspace"]
        for tokens, requests, selected in ((1, 4, 1), (127, 4, 4), (128, 4, 4), (255, 4, 4), (8192, 256, 256)):
            with self.subTest(tokens=tokens, requests=requests):
                self.worker.vllm_config.scheduler_config.max_num_batched_tokens = tokens
                self.worker.vllm_config.scheduler_config.max_num_seqs = requests
                result = self.reserve()
                expected = planner.selected_plan_storage_bytes(
                    executor.selected_union_capacity(tokens), tokens, selected
                )
                expected += executor.selected_runtime_storage_bytes(tokens, requests)
                plan = self.worker.model_runner._selected_kv_budget
                self.assertEqual(plan.fixed_bytes, expected)
                self.assertEqual(plan.bytes_per_block, 8 * 2)
                self.assertEqual(result, 2**30 - 8388608 - expected - 16 * plan.maximum_blocks)
                plan.validate(result // plan.native_page_bytes)

    def test_worker_override_is_checked_before_publishing_budget(self):
        self.worker.cache_config.num_gpu_blocks_override = 10**9
        with self.assertRaisesRegex(ValueError, "block count"):
            self.reserve()
        self.assertFalse(hasattr(self.worker.model_runner, "_selected_kv_budget"))

    def test_dsa_cp_local_topk_is_rejected_before_workspace_reservation(self):
        self.worker.model_runner.ascend_config.enable_dsa_cp = True
        with self.assertRaisesRegex(ValueError, "whole-batch TopK.*DSA-CP"):
            self.reserve()
        self.assertFalse(hasattr(self.worker.model_runner, "_selected_kv_budget"))

    def test_query_dtype_does_not_add_a_selected_only_constraint(self):
        self.worker.model_runner.model_config.dtype = "fp16"
        fp16_bytes = self.reserve()
        self.worker.model_runner.model_config.dtype = "bf16"
        self.assertEqual(self.reserve(), fp16_bytes)

    def test_owner_count_does_not_multiply_shared_output_budget(self):
        one_owner = self.reserve({"a": "a", "b": "a"})
        one_fixed = self.worker.model_runner._selected_kv_budget.fixed_bytes
        two_owners = self.reserve({"a": "a", "b": "b"})
        self.assertEqual(one_owner, two_owners)
        self.assertEqual(one_fixed, self.worker.model_runner._selected_kv_budget.fixed_bytes)


class ForwardBindingTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Mock()
        self.batch = object()
        self.metadata, self.attention = NS(), object()
        self.step = NS(connector_metadata=self.metadata, finish=Mock())
        self.runtime.source = NS(step=self.step, prepare=Mock(return_value=self.step))
        self.runtime.source_step = None

        def bind(*_args):
            self.runtime.source_step = self.step
            return self.batch

        self.runtime.bind_step.side_effect = bind
        self.runner = NS(
            selected_kv_runtime=self.runtime,
            input_batch=NS(num_reqs=2, req_ids=["b", "a", None], num_computed_tokens_cpu=[65536, 128, 0]),
            query_start_loc=NS(cpu=[0, 128, 256, 999]),
        )
        self.output = NS(kv_connector_metadata=self.metadata, num_scheduled_tokens={"a": 128, "b": 128})

    def test_bind_uses_actual_worker_order_and_cpu_chunk_boundaries(self):
        with BINDING.selected_kv_forward(self.runner, self.output, self.attention) as batch:
            self.assertIs(batch, self.batch)
            self.runtime.finish_step.assert_not_called()
        self.runtime.bind_step.assert_called_once_with(
            self.step, ["b", "a"], [65536, 128], [0, 128, 256], self.attention
        )
        self.runtime.finish_step.assert_called_once_with()
        self.runtime.source.prepare.assert_called_once_with(self.metadata, self.output.num_scheduled_tokens)

    def test_bind_failure_does_not_finish_a_previous_batch(self):
        previous = self.runtime.source_step = object()
        self.runtime.bind_step.side_effect = ValueError("stale epoch")
        with (
            self.assertRaisesRegex(ValueError, "stale epoch"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.runtime.finish_step.assert_not_called()
        self.assertIs(self.runtime.source_step, previous)
        self.step.finish.assert_called_once_with(failed=True)

    def test_cpu_view_error_retains_new_source_before_runtime_binding(self):
        class BrokenView:
            def __getitem__(self, key):
                raise ValueError("CPU view unavailable")

        for target, field in (
            (self.runner.input_batch, "req_ids"),
            (self.runner.input_batch, "num_computed_tokens_cpu"),
            (self.runner.query_start_loc, "cpu"),
        ):
            with (
                self.subTest(field=field),
                patch.object(target, field, BrokenView()),
                self.assertRaisesRegex(ValueError, "CPU view unavailable"),
                BINDING.selected_kv_forward(self.runner, self.output, self.attention),
            ):
                self.fail("Must not run the model")
            self.runtime.bind_step.assert_not_called()
            self.runtime.finish_step.assert_not_called()
            self.step.finish.assert_called_once_with(failed=True)
            self.assertIs(self.runtime.source.step, self.step)
            self.step.finish.reset_mock()

    def test_prepare_error_does_not_finish_a_previous_runtime_source(self):
        previous = self.runtime.source_step = object()
        self.runtime.source.prepare.side_effect = ValueError("publication failed")
        with (
            self.assertRaisesRegex(ValueError, "publication failed"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.runtime.finish_step.assert_not_called()
        self.step.finish.assert_not_called()
        self.assertIs(self.runtime.source_step, previous)

    def test_unbound_source_retention_error_preserves_original(self):
        original = ValueError("binding failed before acquisition")
        self.runtime.bind_step.side_effect = original
        self.step.finish.side_effect = RuntimeError("source retention failed")
        with (
            self.assertRaises(ValueError) as caught,
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.assertIs(caught.exception, original)
        self.assertIn("source retention failed", original.__notes__[0])
        self.runtime.finish_step.assert_not_called()

    def test_unprepared_metadata_cannot_reuse_an_identical_previous_request_set(self):
        # Equal contents do not establish that native preparation ran this step.
        self.output.kv_connector_metadata = NS(**vars(self.metadata))
        self.assertEqual(self.output.kv_connector_metadata, self.metadata)
        with (
            self.assertRaisesRegex(RuntimeError, "different connector metadata"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model with another step's filtered transfers")
        self.runtime.bind_step.assert_not_called()
        self.runtime.finish_step.assert_not_called()
        self.assertIs(self.runtime.source.step, self.step)
        self.step.finish.assert_called_once_with(failed=True)

    def test_missing_connector_metadata_retires_without_binding_the_stale_source(self):
        self.output.kv_connector_metadata = None
        self.runtime.source_step = self.step
        self.runtime.bind_step.side_effect = None
        self.runtime.bind_step.return_value = None
        del self.runner.input_batch
        del self.runner.query_start_loc
        with BINDING.selected_kv_forward(self.runner, self.output, self.attention) as batch:
            self.assertIsNone(batch)
        self.runtime.bind_step.assert_called_once_with(None, (), (), (), self.attention)
        self.runtime.source.prepare.assert_not_called()
        self.runtime.finish_step.assert_not_called()

    def test_bind_failure_marks_only_its_newly_acquired_source_failed(self):
        self.runtime.source_step = None

        def acquire_then_fail(*_args):
            self.runtime.source_step = self.step
            raise ValueError("request mismatch")

        self.runtime.bind_step.side_effect = acquire_then_fail
        self.runtime.finish_step.side_effect = RuntimeError("reader failed")
        with (
            self.assertRaisesRegex(ValueError, "request mismatch") as caught,
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.runtime.finish_step.assert_called_once_with(failed=True)
        self.assertIn("reader failed", caught.exception.__notes__[0])

    def test_repeated_failed_binding_does_not_finish_same_source_again(self):
        self.runtime.source_step = self.step
        self.runtime.bind_step.side_effect = ValueError("previous failure")
        with (
            self.assertRaisesRegex(ValueError, "previous failure"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.runtime.finish_step.assert_not_called()
        self.step.finish.assert_not_called()

    def test_model_error_is_not_replaced_by_cleanup_error(self):
        self.runtime.finish_step.side_effect = RuntimeError("cleanup failure")
        original = ValueError("model failure")
        with (
            self.assertRaises(ValueError) as caught,
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            raise original
        self.assertIs(caught.exception, original)
        self.assertIn("cleanup failure", original.__notes__[0])
        self.runtime.finish_step.assert_called_once_with(failed=True)

    def test_cleanup_failure_after_success_is_visible(self):
        self.runtime.finish_step.side_effect = RuntimeError("unretired reader")
        with (
            self.assertRaisesRegex(RuntimeError, "unretired reader"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            pass
        self.runtime.finish_step.assert_called_once_with()

    def test_python310_error_without_notes_preserves_original(self):
        class LegacyError(RuntimeError):
            @property
            def add_note(self):
                raise AttributeError("Python 3.10 has no exception notes")

        original = LegacyError("original")
        self.runtime.finish_step.side_effect = ValueError("cleanup failed")
        with (
            self.assertRaises(LegacyError) as caught,
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            raise original
        self.assertIs(caught.exception, original)

    def test_pure_full_forward_retires_without_reading_request_metadata(self):
        self.output.kv_connector_metadata = NS()
        self.runtime.source.step = None
        self.runtime.source.prepare.return_value = None
        del self.runner.input_batch
        del self.runner.query_start_loc
        self.runtime.bind_step.side_effect = None
        self.runtime.bind_step.return_value = None
        with BINDING.selected_kv_forward(self.runner, self.output, self.attention) as batch:
            self.assertIsNone(batch)
        self.runtime.bind_step.assert_called_once_with(None, (), (), (), self.attention)
        self.runtime.finish_step.assert_not_called()

    def test_pure_full_retirement_failure_keeps_previous_source(self):
        self.output.kv_connector_metadata = NS()
        self.runtime.source.step = None
        self.runtime.source.prepare.return_value = None
        previous = self.runtime.source_step = object()
        self.runtime.bind_step.side_effect = RuntimeError("previous source has live readers")
        with (
            self.assertRaisesRegex(RuntimeError, "previous source has live readers"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.fail("Must not run the model")
        self.assertIs(self.runtime.source_step, previous)
        self.runtime.finish_step.assert_not_called()

    def test_pure_full_model_error_has_no_selected_cleanup(self):
        self.output.kv_connector_metadata = NS()
        self.runtime.source.step = None
        self.runtime.source.prepare.return_value = None
        self.runtime.bind_step.side_effect = None
        self.runtime.bind_step.return_value = None
        original = ValueError("native model failed")
        with (
            self.assertRaises(ValueError) as caught,
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            raise original
        self.assertIs(caught.exception, original)
        self.runtime.finish_step.assert_not_called()

    def test_forward_error_never_finishes_another_source(self):
        with (
            self.assertRaisesRegex(ValueError, "model failed"),
            BINDING.selected_kv_forward(self.runner, self.output, self.attention),
        ):
            self.runtime.source_step = object()
            raise ValueError("model failed")
        self.runtime.finish_step.assert_not_called()
        self.step.finish.assert_called_once_with(failed=True)


class StartupBindingTest(unittest.TestCase):
    def setUp(self):
        self.source = Mock()
        self.runtime_factory = Mock(return_value=object())
        self.worker = NS(selected_kv_source=None)
        self.runner = NS(
            ascend_config=NS(selected_kv_config=CONFIG.SelectedKVConfig(enabled=True)),
            _selected_kv_budget=Mock(),
            selected_kv_runtime=None,
        )
        self.modules = {
            "vllm.distributed.kv_transfer": NS(get_kv_transfer_group=lambda: NS(connector_worker=self.worker)),
            "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_runtime": NS(
                SelectedKVRuntime=self.runtime_factory
            ),
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout": NS(
                get_layerwise_kv_cache_specs=lambda _: {}
            ),
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.selected_kv_source": NS(
                SelectedKVSource=Mock(return_value=self.source)
            ),
        }

    def initialize(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch.object(BINDING, "_native_geometry", return_value=(("layer",), 1, 0)),
            patch.object(BINDING, "_topk_owners", return_value={"layer": "layer"}),
        ):
            BINDING.initialize_selected_kv(self.runner, NS(num_blocks=4), {})

    def test_runtime_failure_rolls_back_installed_source_hooks(self):
        original = RuntimeError("missing kernel library")
        self.runtime_factory.side_effect = original
        with self.assertRaises(RuntimeError) as caught:
            self.initialize()
        self.assertIs(caught.exception, original)
        self.source.close.assert_called_once_with()
        self.assertIsNone(self.worker.selected_kv_source)
        self.assertIsNone(self.runner.selected_kv_runtime)

    def test_source_rollback_error_does_not_hide_initialization_failure(self):
        self.runtime_factory.side_effect = RuntimeError("runtime failed")
        self.source.close.side_effect = ValueError("source rollback failed")
        with self.assertRaisesRegex(RuntimeError, "runtime failed") as caught:
            self.initialize()
        self.assertIn("source rollback failed", caught.exception.__notes__[0])

    def test_success_publishes_both_owners_only_after_construction(self):
        self.initialize()
        self.assertIs(self.worker.selected_kv_source, self.source)
        self.assertIs(self.runner.selected_kv_runtime, self.runtime_factory.return_value)
        self.runner._selected_kv_budget.validate.assert_called_once_with(4, 0)
        self.source.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
