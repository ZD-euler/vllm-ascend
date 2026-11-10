"""One full78 M3v15 event-completion performance contract."""
from __future__ import annotations
import os
import time
FEATURE_ENABLED = False
TP = 4
DP = 4
EP = 16
LAYERS = 78
DEVICES = tuple(range(16))
MAX_BATCHED_TOKENS = 6144
MAX_MODEL_LEN = 528385
BLOCKS = 4609
LOGP_TOP = 20
WARMUPS = 1
MEASURED = 3
DEADLINE = float(os.environ.get('BENCH_DEADLINE', str(time.time() + 7200)))
STOP_NEW = DEADLINE - 600
CASES = [{'case': 'h524288-q4096-bs4', 'title': 'M3v15/native H512K Q4K full16', 'H': 524288, 'request_Qs': [4096, 4096, 4096, 4096], 'continuation': False, 'expected_selected': [1, 1, 1, 1]}]

def case_map():
    return {c['case']: c for c in CASES}

def case(name):
    return case_map()[name]

def geometry(value):
    expected = case(value['case'])
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError('geometry mismatch: ' + key)
    return expected

def local_requests(value):
    return list(value['request_Qs'][0::DP])

def local_query_count(value):
    return sum(local_requests(value))

def global_BS(value):
    return len(value['request_Qs'])

def all_cases():
    return list(CASES)

LI_OWNERS = 21

WARMUPS = 1
MEASURED = 1
