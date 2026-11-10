# SPDX-License-Identifier: Apache-2.0
"""Compile and exercise the real host launcher against a recording ACL stub.

These tests check the host ABI and ownership paths. They do not compile the
AscendC kernels or establish NPU numerical correctness.
"""

import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

ACL_HEADER = r"""
#pragma once
#include <cstddef>
#include <cstdint>
using aclrtBinHandle = void*;
using aclrtFuncHandle = void*;
constexpr int32_t ACL_SUCCESS = 0;
constexpr int32_t ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 9;
struct aclrtLaunchKernelAttr {
    int32_t id;
    union { uint32_t schemMode; } value;
};
struct aclrtLaunchKernelCfg { aclrtLaunchKernelAttr* attrs; size_t numAttrs; };
extern "C" {
int32_t aclrtBinaryLoadFromFile(const char*, const void*, aclrtBinHandle*);
int32_t aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle*);
int32_t aclrtGetHardwareSyncAddr(void**);
int32_t aclrtBinaryUnLoad(aclrtBinHandle);
int32_t aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, void*,
    const aclrtLaunchKernelCfg*, const void*, size_t, const void*, size_t);
}
"""

ACL_STUB = r"""
#include <acl/acl.h>
#include <array>
#include <cstring>

namespace {
std::array<int32_t, 7> status{};
uint64_t sequence = 0;
std::array<uint8_t, 256> arguments{};
size_t argumentSize = 0;
std::array<uint64_t, 6> prefix{};
uint32_t cores = 0, attributeId = 0, schema = 0, configCount = 0;
char functionName[128]{};
int32_t Record(size_t operation) {
    sequence = sequence * 10 + operation;
    return status[operation];
}
}
extern "C" {
void test_reset() {
    status.fill(0); sequence = 0; arguments.fill(0); prefix.fill(0);
    argumentSize = 0; cores = attributeId = schema = configCount = 0;
    functionName[0] = '\0';
}
void test_status(uint32_t operation, int32_t value) { status.at(operation) = value; }
uint64_t test_sequence() { return sequence; }
size_t test_argument_size() { return argumentSize; }
const void* test_arguments() { return arguments.data(); }
const char* test_function_name() { return functionName; }
uint64_t test_prefix(uint32_t index) { return prefix.at(index); }
uint32_t test_cores() { return cores; }
uint32_t test_attribute_id() { return attributeId; }
uint32_t test_schema() { return schema; }
uint32_t test_config_count() { return configCount; }
int32_t aclrtBinaryLoadFromFile(const char*, const void*, aclrtBinHandle* output) {
    const auto rc = Record(1); if (!rc) *output = reinterpret_cast<void*>(0x111);
    return rc;
}
int32_t aclrtBinaryGetFunction(aclrtBinHandle, const char* name, aclrtFuncHandle* output) {
    std::strncpy(functionName, name, sizeof(functionName) - 1);
    const auto rc = Record(2); if (!rc) *output = reinterpret_cast<void*>(0x222);
    return rc;
}
int32_t aclrtGetHardwareSyncAddr(void** output) {
    const auto rc = Record(3); if (!rc) *output = reinterpret_cast<void*>(0x333);
    return rc;
}
int32_t aclrtBinaryUnLoad(aclrtBinHandle) { return Record(4); }
int32_t launch_sparse_prefill_adaptive_prefix(void* stream, void* topk, void* lists,
    void* descriptors, uint32_t history, uint32_t query) {
    prefix = {reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(topk),
        reinterpret_cast<uintptr_t>(lists), reinterpret_cast<uintptr_t>(descriptors),
        history, query};
    return Record(5);
}
int32_t launch_dsa_selected_li_pages(void*, void*, void*, void*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
int32_t aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t n, void* stream,
    const aclrtLaunchKernelCfg* config, const void* args, size_t size,
    const void* extra, size_t extraSize) {
    if (function != reinterpret_cast<void*>(0x222) || !stream ||
        extra || extraSize || size > arguments.size()) return -99;
    cores = n; attributeId = config->attrs[0].id;
    schema = config->attrs[0].value.schemMode; configCount = config->numAttrs;
    argumentSize = size; std::memcpy(arguments.data(), args, size);
    return Record(6);
}
}
"""


class KernelArguments(ctypes.Structure):
    _fields_ = (
        [
            (name, ctypes.c_void_p)
            for name in (
                "ffts",
                "lists",
                "descriptors",
                "membership",
                "counters",
                "groups",
                "bitmap",
                "sources",
                "lengths",
            )
        ]
        + [(name, ctypes.c_uint32) for name in ("history", "query", "width", "capacity")]
        + [("overflow", ctypes.c_void_p)]
    )


class SelectedKernelLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        directory = tempfile.TemporaryDirectory(prefix="selected-kv-launch-")
        cls.addClassCleanup(directory.cleanup)
        build = Path(directory.name)
        (build / "acl").mkdir()
        (build / "acl/acl.h").write_text(ACL_HEADER)
        (build / "stub.cpp").write_text(ACL_STUB)
        library = build / "liblaunch.so"
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-Werror",
                f"-I{build}",
                str(ROOT / "csrc/sparse_prefill_native/sparse_prefill_fused_fixed_launch.cpp"),
                str(build / "stub.cpp"),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        cls.library = ctypes.CDLL(str(library))
        cls.create = cls.library.create_sparse_prefill_fixed_handle
        cls.create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        cls.create.restype = ctypes.c_int32
        cls.destroy = cls.library.destroy_sparse_prefill_fused_handle
        cls.destroy.argtypes = [ctypes.c_void_p]
        cls.destroy.restype = ctypes.c_int32
        cls.launch = cls.library.launch_sparse_prefill_fused_plan
        cls.launch.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_uint32] * 4
        cls.launch.restype = ctypes.c_int32
        cls.library.test_reset.argtypes = []
        cls.library.test_reset.restype = None
        cls.library.test_status.argtypes = [ctypes.c_uint32, ctypes.c_int32]
        cls.library.test_status.restype = None
        cls.library.test_sequence.restype = ctypes.c_uint64
        cls.library.test_argument_size.restype = ctypes.c_size_t
        cls.library.test_arguments.restype = ctypes.c_void_p
        cls.library.test_function_name.restype = ctypes.c_char_p
        cls.library.test_prefix.argtypes = [ctypes.c_uint32]
        cls.library.test_prefix.restype = ctypes.c_uint64

    def setUp(self):
        self.library.test_reset()
        self.handles = []

    def tearDown(self):
        self.library.test_status(4, 0)
        for handle in self.handles:
            self.assertEqual(self.destroy(handle), 0)

    def make_handle(self, capacity=524288):
        handle = ctypes.c_void_p()
        self.assertEqual(self.create(b"device.o", capacity, ctypes.byref(handle)), 0)
        self.handles.append(handle)
        return handle

    @staticmethod
    def launch_values(handle):
        return [handle, 0x100, *(0x200 + 0x100 * i for i in range(9)), 65536, 256, 2048, 524288]

    def test_fixed_create_selects_hardware_sync_kernel(self):
        self.make_handle()
        self.assertEqual(self.library.test_sequence(), 123)
        self.assertEqual(self.library.test_function_name(), b"sparse_prefill_partition_finish_fixed_0")

    def test_prefix_precedes_launch_and_argument_bytes_match_fixed_abi(self):
        handle = self.make_handle()
        self.library.test_reset()
        values = self.launch_values(handle)
        self.assertEqual(self.launch(*values), 0)
        self.assertEqual(self.library.test_sequence(), 56)
        self.assertEqual([self.library.test_prefix(i) for i in range(6)], [0x100, 0x200, 0x300, 0x400, 65536, 256])
        expected = KernelArguments(0x333, *values[3:11], *values[11:], None)
        size = self.library.test_argument_size()
        self.assertEqual(size, ctypes.sizeof(expected))
        self.assertEqual(ctypes.string_at(self.library.test_arguments(), size), bytes(expected))
        self.assertEqual(self.library.test_cores(), 32)
        self.assertEqual(self.library.test_attribute_id(), 9)
        self.assertEqual((self.library.test_schema(), self.library.test_config_count()), (1, 1))

    def test_maximum_history_and_query_keep_exact_pointer_and_scalar_offsets(self):
        handle = self.make_handle()
        values = self.launch_values(handle)
        values[-4:] = [524288, 8192, 16384, 524288]
        self.assertEqual(self.launch(*values), 0)
        actual = KernelArguments.from_buffer_copy(
            ctypes.string_at(self.library.test_arguments(), ctypes.sizeof(KernelArguments))
        )
        self.assertEqual((actual.history, actual.query, actual.width, actual.capacity), tuple(values[-4:]))
        self.assertIsNone(actual.overflow)
        self.assertEqual((KernelArguments.history.offset, KernelArguments.overflow.offset), (9 * 8, 9 * 8 + 4 * 4))

    def test_irregular_history_and_query_reach_prefix_and_fused_abi(self):
        handle = self.make_handle()
        for history in (65537, 65661, 65663, 524287):
            for query in (1, 2, 125, 127, 129, 1023, 1025, 4097, 7984):
                with self.subTest(history=history, query=query):
                    values = self.launch_values(handle)
                    width = ((history + 4095) // 4096) * 128
                    values[-4:] = [history, query, width, 524288]
                    self.assertEqual(self.launch(*values), 0)
                    self.assertEqual([self.library.test_prefix(i) for i in (4, 5)], [history, query])
                    actual = KernelArguments.from_buffer_copy(
                        ctypes.string_at(self.library.test_arguments(), ctypes.sizeof(KernelArguments))
                    )
                    self.assertEqual((actual.history, actual.query, actual.width), (history, query, width))

    def test_prefix_error_stops_fused_launch(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.library.test_status(5, 71)
        self.assertEqual(self.launch(*self.launch_values(handle)), 71)
        self.assertEqual(self.library.test_sequence(), 5)
        self.assertEqual(self.library.test_argument_size(), 0)

    def test_launch_error_is_returned(self):
        handle = self.make_handle()
        self.library.test_reset()
        self.library.test_status(6, 72)
        self.assertEqual(self.launch(*self.launch_values(handle)), 72)
        self.assertEqual(self.library.test_sequence(), 56)

    def test_failed_create_cleans_binary_without_publishing_handle(self):
        for operation, sequence in ((1, 1), (2, 124), (3, 1234)):
            with self.subTest(operation=operation):
                self.library.test_reset()
                self.library.test_status(operation, 70 + operation)
                handle = ctypes.c_void_p(0xBAD)
                self.assertEqual(self.create(b"device.o", 524288, ctypes.byref(handle)), 70 + operation)
                self.assertEqual(handle.value, 0xBAD)
                self.assertEqual(self.library.test_sequence(), sequence)

    def test_destroy_error_preserves_handle_for_retry(self):
        handle = self.make_handle()
        self.library.test_status(4, 74)
        self.assertEqual(self.destroy(handle), 74)
        self.assertEqual(self.launch(*self.launch_values(handle)), 0)
        self.library.test_status(4, 0)
        self.assertEqual(self.destroy(handle), 0)
        self.handles.remove(handle)
        self.assertEqual(self.destroy(None), 0)

    def test_invalid_create_never_calls_acl(self):
        for base in (0, 1, 127, 524416):
            with self.subTest(base=base):
                handle = ctypes.c_void_p()
                self.assertEqual(self.create(b"device.o", base, ctypes.byref(handle)), -1)
                self.assertIsNone(handle.value)
        self.assertEqual(self.create(None, 128, ctypes.byref(ctypes.c_void_p())), -1)
        self.assertEqual(self.create(b"device.o", 128, None), -1)
        self.assertEqual(self.library.test_sequence(), 0)

    def test_invalid_launch_never_submits_prefix(self):
        handle = self.make_handle(capacity=65536)
        for index, invalid in (
            (1, None),
            (2, None),
            (11, 524289),
            (12, 0),
            (12, 8193),
            (13, 129),
            (13, 16385),
            (14, 65537),
        ):
            with self.subTest(index=index, value=invalid):
                self.library.test_reset()
                values = self.launch_values(handle)
                values[14] = 65536
                values[index] = invalid
                self.assertEqual(self.launch(*values), -1)
                self.assertEqual(self.library.test_sequence(), 0)


class InplaceAddressTest(unittest.TestCase):
    """Compile the exact device address helpers; memcpy is a CPU oracle only."""

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A host C++ compiler is required")
        temporary = tempfile.TemporaryDirectory(prefix="selected-inplace-address-")
        cls.addClassCleanup(temporary.cleanup)
        build = Path(temporary.name)
        (build / "address.cpp").write_text(r"""
#define __aicore__
#include "packed_inplace_address.h"
#include <cstring>
using namespace SparsePrefillInplace;
extern "C" {
int decode(int32_t id, uint32_t h, int32_t p, uint32_t sourcePages,
           uint32_t destinationPages, uint64_t base, uint64_t* source, uint64_t* destination) {
    return int(DecodeRow(id, h, p, sourcePages, destinationPages, base, *source, *destination));
}
int adjacent(uint64_t source, uint64_t destination, uint64_t nextSource,
             uint64_t nextDestination, uint32_t run) {
    return Adjacent(source, destination, nextSource, nextDestination, run);
}
int copy_row(int32_t id, uint32_t h, int32_t p, uint32_t n, uint64_t base, void* cache) {
    uint64_t source=0, destination=0;
    auto state=DecodeRow(id,h,p,n,n,base,source,destination);
    if (state==RowState::Selected) std::memcpy(static_cast<char*>(cache)+destination,
                                            reinterpret_cast<void*>(source),ROW_BYTES);
    return int(state);
}
uint32_t row_bytes() { return ROW_BYTES; }
uint32_t history_pages(uint32_t h) { return HistoryPages(h); }
}
""")
        library = build / "libaddress.so"
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-std=c++17",
                "-Werror",
                f"-I{ROOT / 'csrc/sparse_prefill_native'}",
                str(build / "address.cpp"),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.decode.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        cls.lib.decode.restype = ctypes.c_int
        cls.lib.adjacent.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_uint32]
        cls.lib.adjacent.restype = ctypes.c_int
        cls.lib.copy_row.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        cls.lib.copy_row.restype = ctypes.c_int

    def decode(self, token, page, base, *, history=512, source_pages=8, destination_pages=8):
        source, destination = ctypes.c_uint64(), ctypes.c_uint64()
        state = self.lib.decode(
            token, history, page, source_pages, destination_pages, base, ctypes.byref(source), ctypes.byref(destination)
        )
        return state, source.value, destination.value

    def test_partial_history_page_is_included_and_append_rows_are_rejected(self):
        self.lib.history_pages.argtypes = [ctypes.c_uint32]
        self.lib.history_pages.restype = ctypes.c_uint32
        for history in (65536, 65537, 65661, 65663, 524287, 524288):
            with self.subTest(history=history):
                self.assertEqual(self.lib.history_pages(history), (history + 127) // 128)
                self.assertEqual(self.decode(history - 1, 7, 4096, history=history)[0], 2)
                self.assertEqual(self.decode(history - 1, 7, 0, history=history)[0], 1)
                for token in (history, history + 124, history + 126):
                    self.assertEqual(self.decode(token, 7, 4096, history=history)[0], 0)

    def test_original_physical_page_and_separate_ddr_base(self):
        state, source, destination = self.decode(129, 6, 0x100000)
        self.assertEqual(state, 2)
        self.assertEqual(source, 0x100000 + 656)
        self.assertEqual(destination, (6 * 128 + 1) * 656)

    def test_cross_page_run_requires_both_addresses_contiguous(self):
        _, source, destination = self.decode(127, 2, 0x100000)
        _, next_source, next_destination = self.decode(128, 3, 0x100000 + 128 * 656)
        self.assertEqual(self.lib.adjacent(source, destination, next_source, next_destination, 1), 1)
        _, _, scattered_destination = self.decode(128, 6, 0x100000 + 128 * 656)
        self.assertEqual(self.lib.adjacent(source, destination, next_source, scattered_destination, 1), 0)
        self.assertEqual(self.lib.adjacent(source, destination, next_source + 4096, next_destination, 1), 0)

    def test_full_zero_is_skip_and_still_requires_domain_coverage(self):
        self.assertEqual(self.decode(128, 7, 0)[0], 1)
        self.assertEqual(self.decode(128, 7, 0, source_pages=7)[0], 0)
        self.assertEqual(self.decode(128, 8, 0)[0], 0)
        # A run sees FULL before considering adjacency, so no zero source read.
        self.assertNotEqual(self.decode(128, 7, 0)[0], 2)

    def test_invalid_history_page_and_uint64_overflow(self):
        for token, page, base in ((-1, 1, 4096), (512, 1, 4096), (0, -1, 4096), (127, 1, 2**64 - 1)):
            with self.subTest(token=token, page=page, base=base):
                self.assertEqual(self.decode(token, page, base)[0], 0)
        self.assertEqual(self.decode(0, 1, 2**64 - 656)[0], 2)
        self.assertEqual(self.decode(0, 1, 2**64 - 655)[0], 0)
        self.assertEqual(self.lib.adjacent(2**64 - 1, 0, 655, 656, 1), 0)

    def test_cpu_copy_preserves_unselected_bytes_and_writes_656_not_672(self):
        self.assertEqual(self.lib.row_bytes(), 656)
        page_bytes = 128 * 656
        source = ctypes.create_string_buffer(bytes((i * 7) % 256 for i in range(page_bytes)), page_bytes)
        cache = ctypes.create_string_buffer(bytes([0xA5]) * (3 * page_bytes), 3 * page_bytes)
        expected = bytearray(cache.raw)
        for token, page in ((0, 2), (127, 2), (128, 1), (129, 1)):
            self.assertEqual(self.lib.copy_row(token, 512, page, 3, ctypes.addressof(source), cache), 2)
            offset = (page * 128 + token % 128) * 656
            start = token % 128 * 656
            expected[offset : offset + 656] = source.raw[start : start + 656]
        before = cache.raw
        self.assertEqual(self.lib.copy_row(256, 512, 0, 3, 0, cache), 1)
        self.assertEqual(cache.raw, before)
        self.assertEqual(cache.raw, bytes(expected))


class SelectedHostQueueSyntaxTest(unittest.TestCase):
    def test_merged_host_library_links_and_registers_with_torch_and_npu_stubs(self):
        # Link all real host translation units, then exercise their exported
        # entries and queue registration. This does not execute an NPU kernel.
        import torch
        from torch.utils.cpp_extension import include_paths

        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        with tempfile.TemporaryDirectory(prefix="selected-host-queue-") as directory:
            build = Path(directory)
            root = build / "torch_npu/csrc"
            (root / "core/npu").mkdir(parents=True)
            (root / "framework").mkdir()
            (root / "core/npu/NPUStream.h").write_text(
                "#pragma once\nnamespace c10_npu { struct StubStream { void* stream(bool) { return nullptr; } }; "
                "inline StubStream getCurrentNPUStream() { return {}; } }\n"
            )
            (root / "core/npu/NPUGuard.h").write_text(
                "#pragma once\nnamespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device) {} }; }\n"
            )
            (root / "framework/OpCommand.h").write_text(
                "#pragma once\nnamespace at_npu::native { struct OpCommand { template<class F> "
                "static void RunOpApi(const char*, F fn) { fn(); } }; }\n"
            )
            (build / "acl").mkdir()
            (build / "acl/acl.h").write_text(ACL_HEADER)
            (build / "acl_stub.cpp").write_text(ACL_STUB)
            flags = [f"-I{path}" for path in include_paths()]
            flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}")
            sources = ROOT / "csrc/sparse_prefill_native"
            library_dir = Path(torch.__file__).parent / "lib"
            library = build / "libselected_host.so"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    f"-I{build}",
                    *flags,
                    str(sources / "sparse_prefill_fused_fixed_launch.cpp"),
                    str(sources / "sparse_prefill_batch_queued.cpp"),
                    str(sources / "li_page_gather_queued.cpp"),
                    str(sources / "hybm_address_export.cpp"),
                    str(build / "acl_stub.cpp"),
                    f"-L{library_dir}",
                    f"-Wl,-rpath,{library_dir}",
                    "-ltorch_cpu",
                    "-ltorch",
                    "-lc10",
                    "-o",
                    str(library),
                ],
                check=True,
                capture_output=True,
            )
            # Use an isolated process so registration lifetime cannot affect
            # another test loading the same operator namespace.
            script = r"""
import ctypes
import sys
import torch
if sys.argv[2] == "ctypes_first":
    owner = ctypes.CDLL(sys.argv[1])
    torch.ops.load_library(sys.argv[1])
else:
    torch.ops.load_library(sys.argv[1])
    owner = ctypes.CDLL(sys.argv[1])
assert "int[] queries" in str(torch.ops.dsa_prefill_queue.batch_plan.default._schema)
assert "Tensor table" in str(torch.ops.dsa_prefill_queue.inplace_copy.default._schema)
assert "Tensor(a!) target_k" in str(torch.ops.dsa_prefill_queue.li_page_gather.default._schema)
assert "Tensor(b!) target_scale" in str(torch.ops.dsa_prefill_queue.li_page_gather.default._schema)
for name in ("batch_plan", "inplace_copy", "li_page_gather"):
    assert torch._C._dispatch_has_kernel_for_dispatch_key("dsa_prefill_queue::" + name, "PrivateUse1")
create = owner.create_sparse_prefill_fixed_handle
create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
create.restype = ctypes.c_int32
destroy = owner.destroy_sparse_prefill_fused_handle
destroy.argtypes = [ctypes.c_void_p]
destroy.restype = ctypes.c_int32
handle = ctypes.c_void_p()
assert create(b"device.o", 524288, ctypes.byref(handle)) == 0
assert handle.value and destroy(handle) == 0
array = ctypes.POINTER(ctypes.c_uint64)
export = owner.dsa_export_hybm_components
export.argtypes = [ctypes.c_void_p, array, array, ctypes.c_uint64, ctypes.c_uint64,
                   ctypes.c_uint64, array, array, ctypes.POINTER(ctypes.c_int32)]
export.restype = ctypes.c_int32
mapper_type = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, array)
def mapped(address, memory_type, output):
    assert memory_type == 1
    output[0] = address + 4096
    return 0
mapper = mapper_type(mapped)
values = ctypes.c_uint64 * 1
output, failed, status = values(), ctypes.c_uint64(), ctypes.c_int32()
assert export(ctypes.cast(mapper, ctypes.c_void_p), values(8192), values(4096), 1, 0, 656,
              output, ctypes.byref(failed), ctypes.byref(status)) == 0
assert output[0] == 12288 and status.value == 0
print("merged host contracts passed")
"""
            for order in ("ctypes_first", "torch_first"):
                with self.subTest(order=order):
                    result = subprocess.run(
                        [sys.executable, "-c", script, str(library), order],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.stdout, "merged host contracts passed\n")

    def test_real_host_checks_and_nonadjacent_pointer_offsets(self):
        # CPU-owned from_blob storage carries a PrivateUse1 device descriptor.
        # Only host shape checks and a recording function run; no device kernel
        # or allocator is invoked. This tests the real C++ wrapper, not a copy.
        import torch
        from torch.utils.cpp_extension import include_paths

        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        with tempfile.TemporaryDirectory(prefix="selected-host-contract-") as directory:
            build = Path(directory)
            (build / "acl").mkdir()
            (build / "acl/acl.h").write_text(ACL_HEADER)
            root = build / "torch_npu/csrc"
            (root / "core/npu").mkdir(parents=True)
            (root / "framework").mkdir()
            (root / "core/npu/NPUStream.h").write_text(
                "#pragma once\nnamespace c10_npu { struct StubStream { void* stream(bool) { return (void*)1; } }; "
                "inline StubStream getCurrentNPUStream() { return {}; } }\n"
            )
            (root / "core/npu/NPUGuard.h").write_text(
                "#pragma once\nnamespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device) {} }; }\n"
            )
            (root / "framework/OpCommand.h").write_text(
                "#pragma once\n#include <stdexcept>\nextern int opCommands;\n"
                "namespace at_npu::native { struct OpCommand { template<class F> "
                "static void RunOpApi(const char*, F fn) { ++opCommands; "
                'if(fn()) throw std::runtime_error("launch failure"); '
                "} }; }\n"
            )
            source = build / "contracts.cpp"
            source.write_text(
                f'#include "{ROOT / "csrc/sparse_prefill_native/sparse_prefill_batch_queued.cpp"}"\n'
                + r"""
#include <cassert>
#include <cstdlib>
#include <iostream>
int unionCalls=0, copyCalls=0, opCommands=0;
int32_t* topkBase=nullptr;
void* scratchPointers[6]{};
int32_t* sourceOutput=nullptr;
int32_t* statusOutput=nullptr;
at::Tensor Make(at::ScalarType dtype, std::vector<int64_t> shape, std::vector<int64_t> stride={},
                c10::Device device=c10::Device(c10::DeviceType::PrivateUse1,0)) {
    int64_t bytes=8;
    for(auto n:shape) bytes*=n;
    if(!stride.empty()) bytes+=8*shape[0]*stride[0];
    auto data=std::malloc(bytes);
    auto options=at::TensorOptions().dtype(dtype).device(device);
    auto maker=at::for_blob(data,shape);
    maker.options(options).target_device(options.device()).deleter([](void* p){std::free(p);});
    if(!stride.empty()) maker.strides(stride);
    return maker.make_tensor();
}
int32_t Union(void*,void*,void* topk,void* lists,void* descriptors,void* membership,void* counters,
              void* groups,void* bitmap,void* sources,void* lengths,
              uint32_t h,uint32_t q,uint32_t width,uint32_t capacity) {
    assert(h==65536 && q==128 && width==2048 && capacity==128);
    assert(static_cast<int32_t*>(topk)==topkBase+(unionCalls==0?128:640)*2048);
    void* actual[]={lists,descriptors,membership,counters,groups,bitmap};
    for(int i=0;i<6;++i) assert(actual[i]==scratchPointers[i]);
    assert(sources==sourceOutput+unionCalls*capacity && lengths==statusOutput+unionCalls*8);
    ++unionCalls; return 0;
}
int32_t CopyEntry(void*,void*,void*,void*,void*,void*,void*,uint32_t n,uint32_t c,uint32_t sourcePages,
                  uint32_t rows,uint32_t columns,uint32_t stride,uint32_t pages) {
    assert(n==2 && c==128 && sourcePages==8 && rows==4 && columns==8 && stride==12 && pages==8);
    ++copyCalls; return 0;
}
int32_t LargeUnion(void*,void*,void* topk,void*,void*,void*,void*,void*,void*,void*,void*,
                   uint32_t h,uint32_t q,uint32_t width,uint32_t capacity) {
    assert(h==524288 && q>=1025 && q<=8192 && width==16384 && capacity==128);
    assert(static_cast<int32_t*>(topk)==topkBase+(8192-q)*2048);
    ++unionCalls; return 0;
}
template<class F> void Reject(F fn) {
    bool rejected=false;
    const auto beforeEnqueue=opCommands, beforeUnion=unionCalls, beforeCopy=copyCalls;
    try { fn(); } catch(const c10::Error&) { rejected=true; }
    assert(rejected);
    assert(opCommands==beforeEnqueue && unionCalls==beforeUnion && copyCalls==beforeCopy);
}
int main() {
    auto topk=Make(at::kInt,{1024,1,2048}); topkBase=topk.data_ptr<int32_t>();
    TensorList scratch;
    for(auto n:{64*4096,64*8,524288,32*8,32*8,16384}) scratch.push_back(Make(at::kInt,{n}));
    TensorList outputs{Make(at::kInt,{2,128}),Make(at::kInt,{2,8})};
    for(int i=0;i<6;++i) scratchPointers[i]=scratch.get(i).data_ptr();
    sourceOutput=outputs.get(0).data_ptr<int32_t>(); statusOutput=outputs.get(1).data_ptr<int32_t>();
    const auto unionFn=static_cast<int64_t>(reinterpret_cast<uintptr_t>(&Union));
    Plan(topk,scratch,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);
    assert(unionCalls==2 && opCommands==1);
    const auto otherDevice=c10::Device(c10::DeviceType::PrivateUse1,1);
    Reject([&]{Plan(topk,scratch,outputs,{65536},{0},{128},128,0,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{0},{128},128,unionFn,0);});
    Reject([&]{Plan(Make(at::kInt,{1024,1,2048},{},c10::Device(c10::DeviceType::CPU)),
                   scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(Make(at::kLong,{1024,1,2048}),scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(Make(at::kInt,{1024,2048}),scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(Make(at::kInt,{1024,1,2047}),scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(Make(at::kInt,{1024,2,2048}),scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(Make(at::kInt,{1024,1,2048},{4096,2048,1}),
                   scratch,outputs,{65536},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},IntList{},{128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{0},IntList{},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,IntList{},IntList{},IntList{},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{-1},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{524289},{0},{128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{0},{0},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{0},{1025},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{-1},{128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536,65536},{256,128},{128,128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536,65536},{0,64},{128,128},128,unionFn,1);});
    auto bad=scratch.copy(); bad.set(5,Make(at::kInt,{16383}));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=scratch.copy(); bad.set(2,Make(at::kInt,{524287}));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=scratch.copy(); bad.push_back(Make(at::kInt,{1}));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=scratch.copy(); bad.set(0,Make(at::kLong,{64*4096}));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=scratch.copy(); bad.set(1,Make(at::kInt,{64*8-1}));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=scratch.copy(); bad.set(3,Make(at::kInt,{32*8},{},otherDevice));
    Reject([&]{Plan(topk,bad,outputs,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=outputs.copy(); bad.set(0,Make(at::kInt,{2,127}));
    Reject([&]{Plan(topk,scratch,bad,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    bad=outputs.copy(); bad.set(1,Make(at::kInt,{2,7}));
    Reject([&]{Plan(topk,scratch,bad,{65536,65536},{128,640},{128,128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536,65536},{128,200},{128,128},128,unionFn,1);});
    Reject([&]{Plan(topk,scratch,outputs,{65536},{1024},{1},128,unionFn,1);});
    assert(unionCalls==2);
    auto ids=Make(at::kInt,{2,128}), state=Make(at::kInt,{2,8}), pages=Make(at::kUInt64,{8});
    auto metadata=Make(at::kInt,{2,4}), table=Make(at::kInt,{4,8},{12,1});
    auto cache=Make(at::kChar,{8,128,1,656});
    const auto copyFn=static_cast<int64_t>(reinterpret_cast<uintptr_t>(&CopyEntry));
    InplaceCopy(ids,state,pages,metadata,table,cache,copyFn);
    assert(copyCalls==1 && opCommands==2);
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,cache,0);});
    Reject([&]{InplaceCopy(Make(at::kLong,{2,128}),state,pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(Make(at::kInt,{2,128},{256,1}),state,pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,Make(at::kInt,{2,7}),pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,Make(at::kLong,{2,8}),pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,Make(at::kInt,{16}),pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,Make(at::kInt,{2,8},{16,1}),pages,metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kUInt64,{9}),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kLong,{8}),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kUInt64,{2,4}),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kUInt64,{0}),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kUInt64,{8},{2}),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,Make(at::kInt,{2,5}),table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,Make(at::kLong,{2,4}),table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,Make(at::kInt,{2,4},{8,1}),table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kInt,{4,8},{1,4}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kInt,{4,8},{7,1}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kInt,{4,0}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kInt,{257,8}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kLong,{4,8}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,Make(at::kInt,{4,8},{16,2}),cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,Make(at::kChar,{8,128,1,672}),copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,Make(at::kInt,{8,128,1,656}),copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,Make(at::kChar,{1024,656}),copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,
                          Make(at::kChar,{8,128,1,656},{128*656*2,656,656,1}),copyFn);});
    Reject([&]{InplaceCopy(ids,state,Make(at::kUInt64,{8},{},otherDevice),metadata,table,cache,copyFn);});
    Reject([&]{InplaceCopy(ids,state,pages,metadata,table,Make(at::kChar,{8,128,1,656},{},otherDevice),copyFn);});
    assert(copyCalls==1);
    auto largeTopk=Make(at::kInt,{8192,1,2048}); topkBase=largeTopk.data_ptr<int32_t>();
    const auto largeFn=static_cast<int64_t>(reinterpret_cast<uintptr_t>(&LargeUnion));
    for(int64_t q:{1025,2048,4097,7984,8192}) {
        const int64_t cohorts=(q+1)/2;
        auto expanded=scratch.copy();
        expanded.set(0,Make(at::kInt,{cohorts*4096}));
        expanded.set(1,Make(at::kInt,{cohorts*8}));
        const auto before=unionCalls;
        Plan(largeTopk,expanded,outputs,{524288},{8192-q},{q},128,largeFn,1);
        assert(unionCalls==before+1);
        Reject([&]{Plan(largeTopk,scratch,outputs,{524288},{0},{q},128,largeFn,1);});
        auto undersized=expanded.copy();
        undersized.set(1,Make(at::kInt,{cohorts*8-1}));
        Reject([&]{Plan(largeTopk,undersized,outputs,{524288},{0},{q},128,largeFn,1);});
        Reject([&]{Plan(largeTopk,expanded,outputs,{524288},{8193-q},{q},128,largeFn,1);});
    }
    Reject([&]{Plan(largeTopk,scratch,outputs,{524288},{0},{8193},128,largeFn,1);});
    std::cout << "host contracts passed\n";
}
"""
            )
            library_dir = Path(torch.__file__).parent / "lib"
            executable = build / "contracts"
            flags = [f"-I{path}" for path in include_paths()]
            flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    f"-I{build}",
                    *flags,
                    str(source),
                    f"-L{library_dir}",
                    f"-Wl,-rpath,{library_dir}",
                    "-ltorch_cpu",
                    "-ltorch",
                    "-lc10",
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "host contracts passed\n")


class SelectedAssertionBuildContractTest(unittest.TestCase):
    """Compile the real guard, not the SDK assertion implementation or device code."""

    def compile_guard(self, flags, sdk="#define ASSERT_MSG(...) ((void)0)\n"):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        with tempfile.TemporaryDirectory(prefix="selected-kv-assert-") as temporary:
            build = Path(temporary)
            (build / "kernel_operator.h").write_text(sdk)
            source = build / "guard.cpp"
            source.write_text('#include "selected_kv_assert.h"\nint main() { return 0; }\n')
            return subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsyntax-only",
                    *flags,
                    f"-I{build}",
                    f"-I{ROOT / 'csrc/sparse_prefill_native'}",
                    str(source),
                ],
                capture_output=True,
                text=True,
            )

    def test_release_flags_reenable_required_assertions(self):
        result = self.compile_guard(["-DNDEBUG", "-UNDEBUG", "-DASCENDC_DUMP=1"])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_release_without_undefining_ndebug_is_rejected(self):
        result = self.compile_guard(["-DNDEBUG", "-DASCENDC_DUMP=1"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compile with -UNDEBUG", result.stderr)

    def test_disabled_or_implicit_dump_is_rejected_before_sdk_include(self):
        for flags in ([], ["-DASCENDC_DUMP=0"]):
            with self.subTest(flags=flags):
                # Even an SDK default cannot bypass the pre-include contract.
                result = self.compile_guard(
                    flags,
                    "#undef ASCENDC_DUMP\n#define ASCENDC_DUMP 1\n#define ASSERT_MSG(...) ((void)0)\n",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("require ASCENDC_DUMP=1", result.stderr)

    def test_sdk_without_assert_msg_is_rejected(self):
        result = self.compile_guard(["-DASCENDC_DUMP=1"], sdk="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require the SDK device ASSERT_MSG API", result.stderr)

    def test_flags_and_first_include_are_local_to_three_device_targets(self):
        source = ROOT / "csrc/sparse_prefill_native"
        targets = ("sparse_prefill_union_pipeline", "sparse_prefill_fused_fixed", "dsa_packed_inplace")
        cmake = (source / "CMakeLists.txt").read_text()
        expected = f"foreach(selected_kv_target IN ITEMS {' '.join(targets)})"
        body = cmake.split(expected, 1)[1].split("endforeach()", 1)[0]
        self.assertIn("ascendc_compile_definitions(${selected_kv_target} PRIVATE ASCENDC_DUMP=1)", body)
        self.assertIn("-UNDEBUG\n        -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1", body)
        li_flags = (
            "ascendc_compile_definitions(dsa_selected_li_pages PRIVATE ASCENDC_DUMP=1)\n"
            "ascendc_compile_options(dsa_selected_li_pages PRIVATE\n"
            "    -UNDEBUG\n"
            "    -forward-options-to-host-compiler -UNDEBUG -DASCENDC_DUMP=1)"
        )
        self.assertIn(li_flags, cmake)
        self.assertNotIn("-UNDEBUG", cmake.replace(body, "").replace(li_flags, ""))
        for filename in (
            "sparse_prefill_union_pipeline.cpp",
            "sparse_prefill_fused_fixed.cpp",
            "packed_inplace.cpp",
            "li_page_gather.cpp",
        ):
            with self.subTest(filename=filename):
                first_include = next(
                    line for line in (source / filename).read_text().splitlines() if "#include" in line
                )
                self.assertEqual(first_include, '#include "selected_kv_assert.h"')


if __name__ == "__main__":
    unittest.main()
