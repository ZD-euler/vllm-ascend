"""Test-only programmatic equivalent of the existing CLI dtype leaf patches.

Do not import patch.platform or patch.worker packages here: their __init__
imports many unrelated operator extensions. Execute the existing leaf files
unchanged before constructing programmatic EngineArgs instead.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

_PREFIX = "_dsa_m2_smoke_c8_leaf_"


def _package_root(package_root=None):
    if package_root is not None:
        return Path(package_root).resolve(strict=True)
    loaded = sys.modules.get("vllm_ascend")
    if loaded is not None and getattr(loaded, "__file__", None):
        return Path(loaded.__file__).resolve(strict=True).parent
    # Finding a top-level spec does not execute the package's __init__. Avoid
    # find_spec on a dotted patch name, which would import all of its parents.
    spec = importlib.util.find_spec("vllm_ascend")
    roots = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(roots) != 1:
        raise RuntimeError("The smoke requires one unambiguous installed Ascend package")
    return Path(roots[0]).resolve(strict=True)


def _load_leaf(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_file() or root not in path.parents:
        raise ValueError("The existing dtype patch must be a file inside the selected package")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = _PREFIX + relative.replace("/", "_").replace(".", "_")
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path or previous._smoke_sha256 != digest:
            raise RuntimeError("A different dtype patch is already bound in this interpreter")
        return dict(relative_file=relative, sha256=digest)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("No loader for the existing dtype patch")
    module = importlib.util.module_from_spec(spec)
    module._smoke_sha256 = digest
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return dict(relative_file=relative, sha256=digest)


def apply_c8_frontend_schema(package_root=None):
    """Widen the original indexer/main cache schemas before EngineArgs."""
    root = _package_root(package_root)
    return tuple(_load_leaf(root, name) for name in (
        "patch/platform/patch_indexer_kv_dtype.py",
        "patch/platform/patch_kv_cache_dtype.py",
    ))


def apply_c8_worker_schema(package_root=None):
    """Optional early worker leaf; normal NPUWorker applies this itself.

    This is deliberately not run by the frontend: its original worker patch
    may import the attention selector. No worker package __init__ is executed.
    """
    return _load_leaf(_package_root(package_root), "patch/worker/patch_kv_cache_dtype.py")
