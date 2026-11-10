"""Bounded single-device rescue probe contracts; no SDK/framework imports."""
import json
import os
from pathlib import Path
import time

PHYSICAL_DEVICE = 8
POOL_BYTES = 10 << 30
PAYLOAD_BYTES = 16 << 20
LEASE_TTL_MS = 600_000


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def identity(pid=None, proc=Path('/proc')):
    pid = os.getpid() if pid is None else pid
    raw = (proc/str(pid)/'stat').read_text()
    return dict(pid=pid, startticks=int(raw[raw.rindex(')')+2:].split()[19]),
                boot_id=(proc/'sys/kernel/random/boot_id').read_text().strip())


def observed_scope(sessions, known, proc=Path('/proc')):
    """Capture only own fresh sessions and their descendants, retaining identity."""
    rows = {}
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path/'stat').read_text()
            fields = raw[raw.rindex(')')+2:].split()
            rows[int(path.name)] = (int(fields[1]), int(fields[3]), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    changed = True
    while changed:
        changed = False
        for pid, (parent, session, ticks) in rows.items():
            if pid not in known and (session in sessions or parent in known):
                known[pid] = dict(pid=pid, startticks=ticks,
                    boot_id=(proc/'sys/kernel/random/boot_id').read_text().strip())
                changed = True
    return [token for pid,token in known.items()
            if pid in rows and rows[pid][2] == token['startticks']]


def qualifies(worker_rc, value, scope_empty, forced, meta_rc):
    copies = value.get('copies', [])
    return (worker_rc == 0 and value.get('status') == 'candidate_pass_pending_real_exit'
        and value.get('pool_bytes') == POOL_BYTES and value.get('physical_device') == PHYSICAL_DEVICE
        and value.get('close_returned') is True and value.get('close_returncode') == 0
        and value.get('write_finish_confirmed') is True and value.get('read_lease_released') is True
        and value.get('pending_writer_count') == value.get('pending_reader_count') == 0
        and len(copies) == 2 and {x.get('scattered') for x in copies} == {False, True}
        and all(x.get('byte_equal') is True and x.get('bytes') == PAYLOAD_BYTES for x in copies)
        and (not value.get('combined_api') or value.get('combined_info_equals_independent_query') is True)
        and scope_empty and not forced and meta_rc in (0, -15))


def validate_admission(value, ready, now=None):
    now = time.time() if now is None else now
    for name in ('pid', 'startticks', 'boot_id', 'run_id', 'node'):
        if value.get(name) != ready[name]:
            raise ValueError('admission identity differs')
    if value.get('physical_devices') != [PHYSICAL_DEVICE]:
        raise ValueError('admission must be physical die8 only')
    stamp = value.get('issued_unix')
    if type(stamp) not in (int, float) or not 0 <= now-stamp <= 60:
        raise ValueError('admission is not fresh')
    if value.get('prior_model_scope_empty_verified_by_parent') is not True:
        raise ValueError('parent has not confirmed prior model scope exit')


def check_scalar(value, name):
    if type(value) is not int or value != 0:
        raise RuntimeError(name+' did not return integer zero')
    return value


def check_statuses(values, count, name):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise RuntimeError(name+' returned an incomplete vector')
    for value in values:
        check_scalar(value, name)


def info_snapshot(infos, sizes):
    if not isinstance(infos, (list, tuple)) or len(infos) != len(sizes):
        raise RuntimeError('incomplete KeyInfo response')
    rows = []
    for info, size in zip(infos, sizes, strict=True):
        addresses = list(info.gva_list())
        if int(info.size()) != size or len(addresses) != 1 or type(addresses[0]) is not int or addresses[0] <= 0:
            raise RuntimeError('KeyInfo does not describe the whole valid object')
        rows.append((size, tuple(info.loc_list()), tuple(info.type_list()), tuple(addresses)))
    return rows


def descriptors(base, scattered):
    if not scattered:
        return [base], [0], [PAYLOAD_BYTES]
    # Same 128-token packed-C8 page scale as real sparse KV traffic; gaps in DDR.
    page_bytes = 128*656
    offsets = list(range(0, PAYLOAD_BYTES, page_bytes))
    return ([base+2*offset for offset in offsets], offsets,
            [min(page_bytes, PAYLOAD_BYTES-offset) for offset in offsets])


def acquire(store, keys, sizes, combined):
    # The caller publishes pending ownership before this one logical acquire.
    if combined:
        response = store.batch_add_lease_with_info(keys, lease_ttl_ms=LEASE_TTL_MS)
        if type(response) is not tuple or len(response) != 2:
            raise RuntimeError('combined response is malformed; never repeat Add')
        statuses, infos = response
        check_statuses(statuses, len(keys), 'combined Add')
        first = info_snapshot(infos, sizes)
    else:
        check_statuses(store.batch_add_lease(keys, LEASE_TTL_MS), len(keys), 'Add')
        first = None
    # An independent query is intentionally retained for this correctness probe.
    queried = info_snapshot(store.batch_get_key_info(keys, flag=0), sizes)
    if first is not None and first != queried:
        raise RuntimeError('Add response KeyInfo differs from independent flag0 query')
    return queried, first is not None
