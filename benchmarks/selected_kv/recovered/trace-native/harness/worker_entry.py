"""Concrete worker class for all three source trees."""
from vllm_ascend.worker.worker import NPUWorker

from mindstudio_worker import MindStudioWorker
from owned_compiler_load import OwnedCompilerLoadMixin


class BenchWorker(MindStudioWorker, OwnedCompilerLoadMixin, NPUWorker):
    pass
