"""One finite private SDK probe, no external process cleanup or NPU reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

from probe_contract import POOL_BYTES, PHYSICAL_DEVICE, atomic, identity, observed_scope, qualifies

HERE = Path(__file__).resolve().parent


def config_text(template, updates):
    output, found = [], set()
    for line in template.read_text().splitlines():
        key = line.partition('=')[0].strip()
        if key in updates:
            output.append(key+'='+str(updates[key])); found.add(key)
        else:
            output.append(line)
    output.extend(key+'='+str(value) for key,value in updates.items() if key not in found)
    return '\n'.join(output)+'\n'


def ports():
    sockets = [socket.socket() for _ in range(3)]
    for value in sockets:
        value.bind(('127.0.0.1', 0))
    result = [value.getsockname()[1] for value in sockets]
    for value in sockets:
        value.close()
    return result


def environment(args, root):
    env = dict(os.environ)
    env.pop('LD_PRELOAD', None); env.pop('LD_AUDIT', None)
    env.pop('DSA_MMC_DDR_NUMA', None)
    env.update(PYTHONPATH=':'.join(map(str, [args.sdk_site, args.mf_site, HERE]))+':'+env.get('PYTHONPATH', ''),
        LD_LIBRARY_PATH=':'.join(map(str, [args.sdk_site/'memcache_hybrid/lib', args.mf_site/'memfabric_hybrid/lib']))+':'+env.get('LD_LIBRARY_PATH', ''),
        MEMFABRIC_HYBRID_EXTEND_LIB_PATH=str(args.mf_site/'memfabric_hybrid/lib'),
        ASCEND_RT_VISIBLE_DEVICES=str(PHYSICAL_DEVICE), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MMC_META_CONFIG_PATH=str(root/'meta.conf'),
        MMC_LOCAL_CONFIG_PATH=str(root/'local.conf'), ASCEND_PROCESS_LOG_PATH=str(root/'logs'))
    return env


def launch(command, env, logfile):
    with logfile.open('xb') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    return process, identity(process.pid)


def stop_owned(process, token):
    if process.poll() is not None:
        return True
    try:
        if identity(process.pid) != token:
            raise RuntimeError('own process identity changed; refuse signal')
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if identity(process.pid) != token:
                raise RuntimeError('own process identity changed; refuse kill')
            process.kill()
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return process.poll() is not None


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir()
    meta_port, store_port, metrics_port = ports()
    sdk_config = args.sdk_site/'memcache_hybrid/config'
    common = {'ock.mmc.meta_service_url':f'tcp://127.0.0.1:{meta_port}', 'ock.mmc.log_level':'error'}
    (root/'meta.conf').write_text(config_text(sdk_config/'mmc-meta.conf', dict(common, **{
        'ock.mmc.meta_service.config_store_url':f'tcp://127.0.0.1:{store_port}',
        'ock.mmc.meta_service.metrics_url':f'http://127.0.0.1:{metrics_port}', 'ock.mmc.log_path':str(root/'logs')})))
    local_updates = dict(common, **{
        'ock.mmc.local_service.config_store_url':f'tcp://127.0.0.1:{store_port}',
        'ock.mmc.local_service.world_size':'64', 'ock.mmc.local_service.protocol':'device_sdma',
        'ock.mmc.local_service.dram.size':'10GB', 'ock.mmc.local_service.max.dram.size':'10GB',
        'ock.mmc.local_service.hbm.size':'0GB', 'ock.mmc.local_service.max.hbm.size':'0GB'})
    if args.unrestricted:
        local_updates['ock.mmc.local_service.dram.unrestricted.enabled']='true'
    (root/'local.conf').write_text(config_text(sdk_config/'mmc-local.conf', local_updates))
    atomic(root/'private-config.json', dict(sdk_site=str(args.sdk_site), mf_site=str(args.mf_site),
        run_id=uuid.uuid4().hex, node=args.node, combined=args.combined,
        mode='unrestricted' if args.unrestricted else 'audit_p2p_fallback',
        audit_sha256=None if args.unrestricted else hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        deadline_unix=min(time.time()+args.timeout_seconds, args.deadline_unix)))
    return supervise(root, args, meta_port)


def supervise(root, args, meta_port):
    env = environment(args, root)
    scopes, processes, known, sessions = [], [], {}, set()
    deadline = min(time.time()+args.timeout_seconds, args.deadline_unix)
    result = dict(schema='ddr_probe_real_exit_v2', status='failed', physical_devices=[PHYSICAL_DEVICE],
        node=args.node, pool_bytes=POOL_BYTES, worker_natural_exit=False, owned_scope_empty=False,
        force_stop_used=False, global_memory_configuration_changed=False,
        allocation_mode='unrestricted' if args.unrestricted else 'audit_p2p_fallback',
        LD_AUDIT_set=not args.unrestricted, actual_NUMA_observed=None)
    try:
        meta_code = ('import sys; sys.path.insert(0,'+repr(str(args.mf_site))+'); '
            'import memfabric_hybrid; sys.path.pop(0); '
            'from memcache_hybrid import MetaService; raise SystemExit(MetaService.main())')
        meta, token = launch([sys.executable, '-c', meta_code], env, root/'private-meta.log')
        scopes.append(token); processes.append(meta); known[token['pid']] = token; sessions.add(meta.pid)
        while time.time() < deadline:
            if meta.poll() is not None:
                raise RuntimeError('private meta exited before probe')
            try:
                with socket.create_connection(('127.0.0.1', meta_port), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise TimeoutError('meta startup deadline')
        worker_env = dict(env)
        if not args.unrestricted:
            worker_env.update(LD_AUDIT=str(args.audit), DSA_MMC_DDR_NUMA=str(args.node))
        worker, token = launch([sys.executable, str(HERE/'probe_worker.py'), str(root)], worker_env, root/'private-worker.log')
        scopes.append(token); processes.append(worker); known[token['pid']] = token; sessions.add(worker.pid)
        atomic(root/'public-owned-processes.json', dict(controller=identity(), processes=scopes))
        while worker.poll() is None and time.time() < deadline:
            observed_scope(sessions, known)
            if meta.poll() is not None:
                raise RuntimeError('private meta exited during probe')
            time.sleep(.2)
        if worker.poll() is None:
            raise TimeoutError('finite probe deadline')
        result['worker_returncode'] = worker.returncode
        result['worker_exit_observed'] = True
        result['worker_natural_exit'] = worker.returncode == 0
        value = json.loads((root/'public-worker-result.json').read_text())
        result['worker'] = value
    except BaseException as error:
        result.update(status='failed', exception_type=type(error).__name__)
    finally:
        for process, token in reversed(list(zip(processes, scopes, strict=True))):
            if process.poll() is None:
                if process is processes[-1] and len(processes) > 1:
                    result['force_stop_used'] = True; result['status'] = 'failed'
                if not stop_owned(process, token):
                    result['status'] = 'failed'
        result['owned_scope_empty'] = not observed_scope(sessions, known)
        atomic(root/'public-owned-processes.json', dict(controller=identity(), processes=list(known.values())))
        result['process_returncodes'] = [process.returncode for process in processes]
        log = root/'private-worker.log'
        rows = re.findall(r'DSA_MMC_PAGE_FALLBACK page_type=(\d+) numa=(\d+) bytes=(\d+) rc=(-?\d+) ms=([\d.]+)',
            log.read_text(errors='replace') if log.exists() else '')
        result['allocation_observations'] = [dict(page_type=int(page),node=int(node),bytes=int(size),
            returncode=int(rc),wall_ms=float(ms)) for page,node,size,rc,ms in rows]
        actual = any(row['node'] == args.node and row['bytes'] == POOL_BYTES and row['page_type'] == 1
            and row['returncode'] == 0 for row in result['allocation_observations'])
        result['actual_2MiB_10GiB_placement_confirmed'] = None if args.unrestricted else actual
        allocation_gate=actual
        if args.unrestricted:
            result['placement_observation']='unknown: ordinary-DDR mode has no audit; no NUMA promise'
            allocation_gate = result.get('worker',{}).get('unrestricted_library_key_present') is True
        if len(processes) == 2 and qualifies(processes[1].returncode, result.get('worker', {}),
                result['owned_scope_empty'], result['force_stop_used'], processes[0].returncode) and allocation_gate:
            result['status'] = 'passed'
        atomic(root/'public-result.json', result)
    return 0 if result['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-site', type=Path, required=True)
    parser.add_argument('--mf-site', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--node', type=int, choices=range(8))
    parser.add_argument('--unrestricted', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=240)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--combined', action='store_true')
    args = parser.parse_args()
    args.sdk_site = args.sdk_site.resolve(strict=True); args.mf_site = args.mf_site.resolve(strict=True)
    if args.unrestricted:
        if args.audit is not None or args.node is not None:
            parser.error('unrestricted mode cannot claim audit or a NUMA placement')
        library=args.sdk_site/'memcache_hybrid/lib/libmf_memcache.so'
        if b'ock.mmc.local_service.dram.unrestricted.enabled' not in library.read_bytes():
            raise ValueError('selected SDK library does not contain the opt-in configuration key')
    elif args.audit is None or args.node is None:
        parser.error('P2P fallback mode requires explicit audit and current NUMA node')
    else:
        args.audit = args.audit.resolve(strict=True)
    if not 0 < args.timeout_seconds <= 600 or args.deadline_unix <= time.time():
        raise ValueError('finite future deadline within 600 seconds required')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
