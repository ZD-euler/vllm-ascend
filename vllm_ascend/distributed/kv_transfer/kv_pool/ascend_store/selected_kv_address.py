# SPDX-License-Identifier: Apache-2.0
"""Export native-store source addresses from the existing SDK domain.

This module does not initialize a store, allocate DDR, or manufacture a DVA.
The caller supplies the already-loaded HyBM library and retains source leases.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class HyBMAddressExporter:
    """Explicit capability over the initialized SDK's existing address domain."""

    def __init__(self, mapper: Any, owner: ctypes.CDLL | None = None) -> None:
        # ctypes resolves this SDK function dynamically; retain its loaded owner.
        self._mapper = mapper
        self._owner = owner
        self._batch: Callable[..., int] | None = None
        self._batch_owner: ctypes.CDLL | None = None

    @classmethod
    def from_loaded_library(cls, path: str | os.PathLike[str]) -> HyBMAddressExporter:
        # NOLOAD prevents creation of another registry with unrelated mappings.
        library = ctypes.CDLL(str(Path(path).resolve(strict=True)), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        mapper = library.hybm_gva_to_va
        mapper.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        mapper.restype = ctypes.c_int32
        exporter = cls(mapper, library)
        # The host helper calls this exact NOLOAD owner's function pointer;
        # loading it creates no second HyBM registry or address domain.
        helper = ctypes.CDLL(str(Path(__file__).resolve().parents[4] / "lib/libsparse_prefill_fused_fixed_launch.so"))
        batch = helper.dsa_export_hybm_components
        array = ctypes.POINTER(ctypes.c_uint64)
        batch.argtypes = [
            ctypes.c_void_p,
            array,
            array,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            array,
            array,
            ctypes.POINTER(ctypes.c_int32),
        ]
        batch.restype = ctypes.c_int32
        exporter._batch, exporter._batch_owner = batch, helper
        return exporter

    def export_component_arrays(
        self,
        blob_gvas: NDArray[np.uint64],
        blob_sizes: NDArray[np.uint64],
        offset: int,
        page_bytes: int,
    ) -> NDArray[np.uint64]:
        """Use Source's owned contiguous uint64 buffers without Python boxing.

        The host helper performs the same per-object extent and endpoint checks.
        Both input arrays and the unpublished output stay alive for the complete
        synchronous call, including while ctypes releases the Python GIL.
        """
        if type(page_bytes) is not int or page_bytes <= 0:
            raise ValueError("page_bytes must be a positive integer")
        if type(offset) is not int or not 0 <= offset < 1 << 64 or page_bytes >= 1 << 64:
            raise ValueError("Component offset and byte count must fit uint64")
        for values in (blob_gvas, blob_sizes):
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.uint64
                or values.ndim != 1
                or not values.flags.c_contiguous
                or not values.flags.aligned
            ):
                raise ValueError("Source address buffers must be aligned contiguous uint64 vectors")
        if blob_gvas.shape != blob_sizes.shape:
            raise ValueError("GVA and allocation-size buffers must have equal length")
        if self._batch is None:
            raise RuntimeError("Selected KV requires its batch address-export helper")
        output = np.empty(blob_gvas.size, dtype=np.uint64)
        if not output.size:
            return output
        array = ctypes.POINTER(ctypes.c_uint64)
        failed, status = ctypes.c_uint64(), ctypes.c_int32()
        code = self._batch(
            ctypes.cast(self._mapper, ctypes.c_void_p),
            blob_gvas.ctypes.data_as(array),
            blob_sizes.ctypes.data_as(array),
            blob_gvas.size,
            offset,
            page_bytes,
            output.ctypes.data_as(array),
            ctypes.byref(failed),
            ctypes.byref(status),
        )
        if code == 1:
            raise ValueError("Component exceeds its leased allocation")
        if code == 2:
            raise RuntimeError("No live device mapping in the supplied HyBM domain")
        if code != 0:
            raise RuntimeError("Component has no contiguous device byte span")
        return output
