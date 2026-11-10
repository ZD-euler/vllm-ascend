# SPDX-License-Identifier: Apache-2.0
"""Run the exact copy kernel body with scalar CPU memory models, not NPU codegen."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class SelectedInplaceKernelTest(unittest.TestCase):
    def test_partial_history_table_tail_full_skip_and_append_preservation(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required")
        kernel = ROOT / "csrc/sparse_prefill_native"
        with tempfile.TemporaryDirectory(prefix="selected-inplace-kernel-") as directory:
            build = Path(directory)
            stub = Path(__file__).with_name("selected_bitmap_host_stub.h").read_text()
            stub += r"""
#define __global__
#define KERNEL_TASK_TYPE_DEFAULT(...)
namespace AscendC {
using TPosition = QuePosition;
constexpr int PIPE_ALL = 1;
inline uint32_t GetBlockNum() { return 32; }
template<class T> using DataCopyPadExtParams = PaddingParams;
}
"""
            (build / "kernel_operator.h").write_text(stub)
            # The SDK launch expression is host-incompatible. Preserve the entire
            # real device body; only omit the host launch function after it.
            body = (
                (kernel / "packed_inplace.cpp")
                .read_text()
                .split('extern "C" int32_t launch_sparse_prefill_packed_inplace', 1)[0]
            )
            (build / "test.cpp").write_text(
                body
                + r"""
#include <cassert>
#include <vector>
int main() {
    using namespace AscendC;
    for(uint32_t h : {65537U, 65661U, 65663U, 524287U}) {
        for(int32_t q : {1, 125, 127, 1023, 1025, 2048, 4097, 7984, 8192}) {
            regions.clear();
            const uint32_t columns = HistoryPages(h), pages = columns + 3;
            std::vector<int32_t> table(columns);
            for(uint32_t i=0;i<columns;++i) table[i]=i;
            // Highest physical page is the partial history/append boundary.
            table.back()=pages-1;
            std::vector<uint8_t> source(128*ROW_BYTES, 0x35), cache(pages*128*ROW_BYTES, 0xa5);
            std::vector<uint64_t> bases(pages, 0);
            bases[0]=reinterpret_cast<uintptr_t>(source.data());
            int32_t ids[128]={0,127,static_cast<int32_t>(h-1)};
            int32_t status[8]={3,q,3+q,0,0,0,0,0};
            int32_t meta[4]={0,static_cast<int32_t>(h),8192-q,q};
            Register(ids,sizeof(ids)); Register(status,sizeof(status)); Register(meta,sizeof(meta));
            Register(table.data(),table.size()*4); Register(bases.data(),bases.size()*8);
            Register(source.data(),source.size()); Register(cache.data(),cache.size());
            for(blockIndex=0;blockIndex<32;++blockIndex)
                sparse_prefill_packed_inplace(reinterpret_cast<uint8_t*>(ids),reinterpret_cast<uint8_t*>(status),
                    reinterpret_cast<uint8_t*>(bases.data()),reinterpret_cast<uint8_t*>(meta),
                    reinterpret_cast<uint8_t*>(table.data()),cache.data(),1,128,pages,1,columns,columns,pages,8);
            for(size_t i=0;i<cache.size();++i) {
                bool selected = i<ROW_BYTES || (i>=127*ROW_BYTES && i<128*ROW_BYTES);
                assert(cache[i]==(selected?0x35:0xa5));
            }
        }
    }
}
"""
            )
            executable = build / "copy"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-DASCENDC_DUMP=1",
                    f"-I{build}",
                    f"-I{kernel}",
                    str(build / "test.cpp"),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
