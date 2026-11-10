# SPDX-License-Identifier: Apache-2.0
"""Exercise setup's real path statements without running CANN or setup()."""

import ast
import os
import re
import unittest
import warnings
from pathlib import Path

from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

ROOT = Path(__file__).resolve().parents[3]


def selected_package_path(command, *, editable_mode=False, inplace=False):
    distribution = Distribution({"script_args": [command]})
    distribution.parse_command_line()
    builder = build_ext(distribution)
    builder.build_lib = "build/lib.cpu-test"
    # PEP 660 sets this flag before the build_ext subcommand. No build runs here.
    builder.editable_mode, builder.inplace = editable_mode, inplace
    tree = ast.parse((ROOT / "setup.py").read_text())
    configure = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "configure")
    start = next(
        i
        for i, node in enumerate(configure.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "install_path"
    )
    finish = next(
        i
        for i, node in enumerate(configure.body)
        if isinstance(node, ast.AugAssign) and "-DVLLM_ASCEND_PACKAGE_PATH=" in ast.unparse(node)
    )
    selected = ast.Module(body=configure.body[start : finish + 1], type_ignores=[])
    context = {"self": builder, "ROOT_DIR": str(ROOT), "os": os, "develop": develop, "cmake_args": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(selected, str(ROOT / "setup.py"), "exec"), context)
    arguments = context["cmake_args"]
    return next(
        argument.split("=", 1)[1] for argument in arguments if argument.startswith("-DVLLM_ASCEND_PACKAGE_PATH=")
    ), distribution


class SelectedPackagingTest(unittest.TestCase):
    def test_wheel_uses_build_package_even_though_develop_object_exists(self):
        path, distribution = selected_package_path("bdist_wheel")
        self.assertIsInstance(distribution.get_command_obj("develop"), develop)
        self.assertEqual(path, str(ROOT / "build/lib.cpu-test/vllm_ascend"))

    def test_pep660_build_ext_uses_source_package(self):
        path, _ = selected_package_path("editable_wheel", editable_mode=True)
        self.assertEqual(path, str(ROOT / "vllm_ascend"))

    def test_legacy_develop_uses_source_package(self):
        path, _ = selected_package_path("develop")
        self.assertEqual(path, str(ROOT / "vllm_ascend"))

    def test_explicit_inplace_build_uses_source_package(self):
        path, _ = selected_package_path("build_ext", inplace=True)
        self.assertEqual(path, str(ROOT / "vllm_ascend"))

    def test_runtime_loads_only_reachable_native_targets(self):
        source = ROOT / "csrc/sparse_prefill_native"
        cmake = (source / "CMakeLists.txt").read_text()
        targets = set(re.findall(r"(?:ascendc_library|add_library)\((\w+) SHARED", cmake))
        self.assertEqual(
            targets,
            {
                "sparse_prefill_union_pipeline",
                "sparse_prefill_fused_fixed",
                "sparse_prefill_fused_fixed_launch",
                "dsa_packed_inplace",
                "dsa_selected_li_pages",
            },
        )
        for filename in re.findall(r'\$\{SELECTED_KV_SOURCE_DIR\}/([^"\s]+)', cmake):
            self.assertTrue((source / filename).is_file(), filename)
        plan = (ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/prefill_selected_plan.py").read_text()
        loaded = set(re.findall(r'root / "lib([^"/]+)\.so"', plan))
        self.assertEqual(loaded, {"sparse_prefill_fused_fixed_launch", "dsa_packed_inplace"})
        self.assertTrue(loaded.issubset(targets))
        host_target = cmake.split("add_library(sparse_prefill_fused_fixed_launch SHARED", 1)[1].split(")", 1)[0]
        for name in (
            "sparse_prefill_fused_fixed_launch.cpp",
            "sparse_prefill_batch_queued.cpp",
            "li_page_gather_queued.cpp",
            "hybm_address_export.cpp",
        ):
            self.assertIn(name, host_target)
        address = (ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/selected_kv_address.py").read_text()
        self.assertIn("lib/libsparse_prefill_fused_fixed_launch.so", address)
        self.assertIn('"sparse_prefill_fused_fixed.o"', plan)
        self.assertIn('"${SELECTED_KV_LIBRARY_DIR}/sparse_prefill_fused_fixed.o"', cmake)
        self.assertIn("ascendcl sparse_prefill_union_pipeline dsa_selected_li_pages", cmake)

    def test_obsolete_wave_entries_are_removed(self):
        source = ROOT / "csrc/sparse_prefill_native"
        for filename in ("device_wave.cpp", "device_queued.cpp", "packed_page_wave.cpp"):
            self.assertFalse((source / filename).exists())
        self.assertFalse(
            (ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/prefill_selected_waves.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
