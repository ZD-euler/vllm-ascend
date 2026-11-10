import os
from vllm_ascend.worker.worker import NPUWorker
from owned_compiler_load import OwnedCompilerLoadMixin
if os.environ.get("BENCH_ARM") == "native":
    from native_compare_worker import NativeCompareWorker as Observer
else:
    from completion_worker import CompletionWorker as Observer
class BenchWorker(Observer, OwnedCompilerLoadMixin, NPUWorker):
    pass
