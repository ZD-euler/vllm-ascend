"""Private local measurements; public export is separately allowlisted."""
import json
from pathlib import Path


def atomic(path, value, *, private=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.next')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def interval_union(intervals):
    total, end = 0., None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('completion precedes start')
        total += stop-start if end is None or start > end else max(0., stop-end)
        end = stop if end is None else max(end, stop)
    return total


def classify_bytes(*args):
    raise RuntimeError('descriptor classification is not part of the formal path')
