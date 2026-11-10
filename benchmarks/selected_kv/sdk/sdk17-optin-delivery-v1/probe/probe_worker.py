"""Real SDK 10GiB DDR probe. Explicit parent admission precedes NPU calls."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

from probe_contract import (POOL_BYTES, PAYLOAD_BYTES, LEASE_TTL_MS, PHYSICAL_DEVICE,
    atomic, identity, validate_admission, check_scalar, check_statuses, descriptors, acquire)


def wait_admission(root, config):
    sys.path.insert(0, config['mf_site'])
    import memfabric_hybrid as mf
    sys.path.pop(0)
    sys.path.insert(0, config['sdk_site'])
    import memcache_hybrid as sdk
    import torch
    import torch_npu  # noqa: F401
    if not Path(sdk.__file__).resolve().is_relative_to(Path(config['sdk_site']).resolve()):
        raise RuntimeError('SDK site mismatch')
    if not Path(mf.__file__).resolve().is_relative_to(Path(config['mf_site']).resolve()):
        raise RuntimeError('MF site mismatch')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != str(PHYSICAL_DEVICE):
        raise RuntimeError('single physical die8 visibility required')
    if config['mode']=='unrestricted':
        if os.environ.get('LD_AUDIT') or os.environ.get('DSA_MMC_DDR_NUMA'):
            raise RuntimeError('ordinary DDR mode forbids audit and a NUMA override')
        library=Path(config['sdk_site'])/'memcache_hybrid/lib/libmf_memcache.so'
        if b'ock.mmc.local_service.dram.unrestricted.enabled' not in library.read_bytes():
            raise RuntimeError('qualified opt-in SDK library required')
    ready = dict(identity(), run_id=config['run_id'], node=config['node'],
        physical_devices=[PHYSICAL_DEVICE], explicit_NPU_calls=False)
    atomic(root/'public-import-ready.json', ready)
    while time.time() < config['deadline_unix']:
        path = root/'admission.json'
        if path.exists():
            validate_admission(json.loads(path.read_text()), ready)
            return torch, sdk
        time.sleep(.1)
    raise TimeoutError('parent did not admit within deadline')


def check_payload(torch, store, root, config, value):
    sizes = [PAYLOAD_BYTES, PAYLOAD_BYTES*2]
    keys = ['owned-ddr-probe-'+config['run_id']+'-'+str(i) for i in range(2)]
    src = torch.arange(PAYLOAD_BYTES//4, dtype=torch.int32, device='npu')
    dst = torch.empty_like(src)
    registered = []
    try:
        for tensor in (src, dst):
            check_scalar(store.register_buffer(tensor.data_ptr(), PAYLOAD_BYTES), 'register')
            registered.append(tensor)
        value['pending_writer_count'] = len(keys)
        gvas = store.batch_alloc(keys, sizes, 1, LEASE_TTL_MS)
        if len(gvas) != 2 or any(type(x) is not int or x <= 0 for x in gvas):
            raise RuntimeError('Alloc response incomplete; never retry unknown writers')
        plans = [descriptors(base, bool(index)) for index,base in enumerate(gvas)]
        value['D2H_returncodes'] = []
        for addresses, offsets, lengths in plans:
            torch.npu.synchronize()
            rc = store.batch_copy(addresses, [src.data_ptr()+x for x in offsets], lengths, 0)
            value['D2H_returncodes'].append(rc); check_scalar(rc, 'D2H')
            torch.npu.synchronize()
        check_statuses(store.batch_write_finish(keys, [0, 0]), 2, 'write_finish')
        value['pending_writer_count'] = 0
        value['write_finish_confirmed'] = True
        value['pending_reader_count'] = len(keys)
        infos, combined_equal = acquire(store, keys, sizes, config['combined'])
        value['combined_info_equals_independent_query'] = combined_equal if config['combined'] else None
        value['key_info_sizes'] = [x[0] for x in infos]
        if [x[3][0] for x in infos] != gvas:
            raise RuntimeError('leased KeyInfo does not match allocated objects')
        value['copies'] = []
        for index, info in enumerate(infos):
            addresses, offsets, lengths = descriptors(info[3][0], bool(index))
            dst.zero_(); torch.npu.synchronize()
            rc = store.batch_copy(addresses, [dst.data_ptr()+x for x in offsets], lengths, 1)
            check_scalar(rc, 'H2D')
            torch.npu.synchronize()
            equal = bool(torch.equal(src, dst))
            value['copies'].append(dict(scattered=bool(index), bytes=PAYLOAD_BYTES,
                descriptors=len(lengths), byte_equal=equal))
            value['copies'][-1].update(d2h_returncode=value['D2H_returncodes'][index], h2d_returncode=rc)
            if not equal:
                raise RuntimeError('real SDK payload mismatch')
        check_scalar(store.batch_remove_lease(keys), 'remove_read_lease')
        value['pending_reader_count'] = 0
        value['read_lease_released'] = True
    finally:
        torch.npu.synchronize()
        for tensor in reversed(registered):
            check_scalar(store.unregister_buffer(tensor.data_ptr(), PAYLOAD_BYTES), 'unregister')


def main(root):
    config = json.loads((root/'private-config.json').read_text())
    value = dict(schema='ddr_probe_worker_v2', status='not_complete',
        pool_bytes=POOL_BYTES, physical_device=PHYSICAL_DEVICE, node=config['node'],
        combined_api=config['combined'], pending_writer_count=0, pending_reader_count=0,
        sdk_site=config['sdk_site'], mf_site=config['mf_site'], audit_sha256=config['audit_sha256'],
        allocation_mode=config['mode'], actual_NUMA=None,
        unrestricted_library_key_present=False,
        global_memory_configuration_changed=False, original_sdk_and_MF_files_modified=False)
    store = None
    try:
        torch, sdk = wait_admission(root, config)
        if config['mode']=='unrestricted':
            library=Path(config['sdk_site'])/'memcache_hybrid/lib/libmf_memcache.so'
            value['unrestricted_library_key_present']=True
            value['SDK_library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
        import setproctitle
        setproctitle.setproctitle('dsa-ddr-10g-probe')
        value['setproctitle_before_init'] = True
        torch.set_num_threads(1)
        torch.npu.set_device(0)
        store = sdk.DistributedObjectStore()
        atomic(root/'public-stage.json', dict(stage='init_enter', **identity(), node=config['node'], pool_bytes=POOL_BYTES))
        started = time.monotonic()
        rc = store.init(0, init_bm=True); value['init_returncode'] = rc; check_scalar(rc, 'init')
        value['init_ms'] = (time.monotonic()-started)*1000
        atomic(root/'public-stage.json', dict(stage='init_returned', **identity(), pool_bytes=POOL_BYTES))
        check_payload(torch, store, root, config, value)
        value['status'] = 'payload_and_lease_passed_pending_close_and_exit'
    except BaseException as error:
        value.update(status='failed', exception_type=type(error).__name__)
    finally:
        if store is not None:
            try:
                rc = store.close(); value['close_returncode'] = rc
                check_scalar(rc, 'close')
                value['close_returned'] = True
            except BaseException as error:
                value.update(status='failed', close_exception_type=type(error).__name__)
        if value['status'] == 'payload_and_lease_passed_pending_close_and_exit' and value.get('close_returned'):
            value['status'] = 'candidate_pass_pending_real_exit'
        atomic(root/'public-worker-result.json', value)
    # Do not use os._exit: natural interpreter/SDK/framework teardown is part of acceptance.
    return 0 if value['status'] == 'candidate_pass_pending_real_exit' else 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]).resolve(strict=True)))
