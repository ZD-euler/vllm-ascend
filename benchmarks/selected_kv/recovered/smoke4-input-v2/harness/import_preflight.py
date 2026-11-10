"""Check the actual controller child environment before any NPU admission."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import controller
import preflight

profile=preflight.check('m3v9')
out=preflight.ROOT/'runs'/'m3v9'
env,_=controller.environment(profile,out)
expected=str(preflight.BASE/'matrix42_15cases_20261031/deploy')
paths=env['PYTHONPATH'].split(':')
if expected not in paths: raise ValueError('controller deploy path differs from preflight')
script='''import importlib.util, json
mods=("early_store_handoff","worker_entry","li_mte_oracle","c8_bootstrap","owned_compiler_load","memcache_hybrid","vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.selected_kv_source","vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_runtime")
for name in mods:
 if importlib.util.find_spec(name) is None: raise ImportError(name)
from early_store_handoff import EarlyStoreHandoff
from worker_entry import BenchWorker
import li_mte_oracle, c8_bootstrap, torch, torch_npu
if hasattr(torch.npu,"is_initialized") and torch.npu.is_initialized(): raise RuntimeError("CPU preflight unexpectedly initialized NPU")
print(json.dumps({"status":"PASS","module_count":len(mods),"npu_initialized":False}))'''
subprocess.run([sys.executable,'-c',script],env=env,cwd=preflight.ROOT,check=True)
