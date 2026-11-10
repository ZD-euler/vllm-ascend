"""Read-only NUMA/buddy snapshot; never configures memory or selects a node."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time


def scan(proc=Path('/proc'), sysfs=Path('/sys/devices/system/node'), page_bytes=None):
    page = os.sysconf('SC_PAGE_SIZE') if page_bytes is None else page_bytes
    rows = {}
    for line in (proc/'buddyinfo').read_text().splitlines():
        match = re.fullmatch(r'Node\s+(\d+),\s+zone\s+(\S+)\s+([\d\s]+)', line)
        if not match:
            raise ValueError('unrecognized buddyinfo row')
        node, zone, counts = int(match[1]), match[2], list(map(int, match[3].split()))
        row = rows.setdefault(node, dict(node=node, zones=[]))
        eligible = lambda target: sum(n*(1 << order)*page for order, n in enumerate(counts)
                                     if (1 << order)*page >= target)
        row['zones'].append(dict(zone=zone, free_blocks_by_order=counts,
            buddy_free_bytes=sum(n*(1 << i)*page for i,n in enumerate(counts)),
            free_in_blocks_ge_2MiB_bytes=eligible(2 << 20),
            free_in_blocks_ge_1GiB_bytes=eligible(1 << 30)))
    for node, row in rows.items():
        root = sysfs/f'node{node}'
        row['cpulist'] = (root/'cpulist').read_text().strip()
        row['distance'] = list(map(int, (root/'distance').read_text().split()))
        row['meminfo_kB'] = {m[1]: int(m[2]) for line in (root/'meminfo').read_text().splitlines()
                            if (m := re.search(r'\b(MemTotal|MemFree|MemUsed):\s+(\d+)\s+kB', line))}
    return dict(schema='ddr_probe_readonly_numa_v1', captured_unix=time.time(), page_bytes=page,
        nodes=[rows[n] for n in sorted(rows)], automatically_selected_node=None,
        note='Buddy blocks are a snapshot, not a guarantee of driver/P2P allocation or free hugepages. '
             'Parent must choose an explicit current node; never inherit an old NUMA6 choice.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = scan()
    payload = json.dumps(value, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(payload)
    else:
        print(payload, end='')
