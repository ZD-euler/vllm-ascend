"""Exact authored PID/startticks ownership helpers; no model/NPU entry point."""
import os
from pathlib import Path
import signal
import time


def process(pid):
    try:
        f = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=f[0], ppid=int(f[1]), startticks=int(f[19]))
    except (OSError, ValueError, IndexError):
        return None

def descendants(owned):
    live = {int(p.name): info for p in Path('/proc').iterdir() if p.name.isdigit()
            and (info := process(int(p.name))) is not None}
    parents = {pid for pid, info in owned.items() if pid in live and info['startticks'] == live[pid]['startticks']}
    while True:
        children = {pid for pid, info in live.items() if info['ppid'] in parents}
        if children <= parents:
            break
        parents |= children
    for pid in parents:
        if pid in live:
            owned[pid] = live[pid]
    return [live[pid] for pid, info in owned.items() if pid in live
            and info['startticks'] == live[pid]['startticks'] and live[pid]['state'] != 'Z']

def retire(owned):
    for sig, seconds in [(signal.SIGTERM, 15), (signal.SIGKILL, 5)]:
        descendants(owned)
        for pid, expected in reversed(list(owned.items())):
            actual = process(pid)
            if not actual or actual['startticks'] != expected['startticks'] or actual['state'] == 'Z':
                continue
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                actual = process(pid)
                if actual and actual['startticks'] == expected['startticks'] and actual['state'] != 'Z':
                    signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        end = time.monotonic() + seconds
        while descendants(owned) and time.monotonic() < end:
            time.sleep(.25)
        if not descendants(owned):
            return
    raise RuntimeError('Owned model has not exited; never start another arm')
