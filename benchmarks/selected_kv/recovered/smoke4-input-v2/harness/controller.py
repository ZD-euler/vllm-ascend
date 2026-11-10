"""One finite owned four-card service for one source arm at a time."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import spec
from perf_common import atomic
import preflight
import owned_process_utils_v1 as owned

ROOT = Path("/repro/reference-root/m3v15_q4k_chunk6k_20261107/smoke4-input-v2")
HARNESS = ROOT / "harness"
DEVICES = spec.DEVICES
STOP = False


def hardware():
    proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=25)
    if proc.returncode:
        raise RuntimeError("npu-smi failed")
    pattern = re.compile(
        r"^\|\s+\d+\s+(\d+)\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s+(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s+\|",
        re.M,
    )
    rows = [
        dict(device=int(device), utilization=int(util), used_MiB=int(used), total_MiB=int(total))
        for device, util, used, total in pattern.findall(proc.stdout)
    ]
    health = re.findall(r"^\|\s+\d+\s+Ascend\S*\s+\|\s+(\S+)\s+\|", proc.stdout, re.M)
    if len(rows) != 16 or len(health) != 16 or {r["device"] for r in rows} != set(range(16)):
        raise ValueError("all sixteen dies must be observable")
    processes = re.findall(r"^\|\s+(\d+)\s+(\d+)\s+\|\s+\d+\s+\|", proc.stdout, re.M)
    for row, state in zip(rows, health):
        row["health"] = state
        row["processes"] = sum(2 * int(n) + int(c) == row["device"] for n, c in processes)
    return rows


def idle(rows):
    selected = [r for r in rows if r["device"] in DEVICES]
    return len(selected) == spec.TP * spec.DP and all(
        r["health"] == "OK" and r["processes"] == 0 and r["utilization"] <= 1 and r["used_MiB"] < 5000 for r in selected
    )


def free_ports(count):
    sockets = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    return ports


def config(template, changes, forbidden=()):
    lines, seen = [], set()
    for line in Path(template).read_text().splitlines():
        key = line.partition("=")[0].strip()
        if key in forbidden:
            continue
        if key in changes:
            lines.append(key + "=" + changes[key])
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(key + "=" + value for key, value in changes.items() if key not in seen)
    return "\n".join(lines) + "\n"


def capture(proc):
    identity = owned.process(proc.pid)
    if identity is None:
        raise RuntimeError("could not capture child identity")
    return identity


def environment(profile, out):
    sdk = Path(profile["sdk"])
    source = Path(profile["source"])
    core = Path(profile["core"])
    env = dict(os.environ)
    env.pop("BENCH_LATE_SDK", None)
    for name in ("LD_AUDIT", "LD_PRELOAD", "DSA_MMC_DDR_NUMA"):
        env.pop(name, None)
    env.update(
        PYTHONPATH=":".join(map(str, (HARNESS, preflight.BASE / "matrix42_15cases_20261031/deploy", source, core, sdk / "memcache_hybrid", sdk))) + ":" + env.get("PYTHONPATH", ""),
        LD_LIBRARY_PATH=":".join(
            map(
                str,
                (
                    source / "vllm_ascend/lib",
                    sdk / "memfabric_hybrid/lib",
                    sdk / "memcache_hybrid/lib",
                ),
            )
        ) + ":" + env.get("LD_LIBRARY_PATH", ""),
        MEMFABRIC_HYBRID_EXTEND_LIB_PATH=str(sdk / "memfabric_hybrid/lib"),
        DSA_BENCH_SOURCE_ROOT=str(source),
        DSA_BENCH_CORE_ROOT=str(core),
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, DEVICES)),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_USE_V1="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENGINE_READY_TIMEOUT_S="28800",
        HCCL_BUFFSIZE="256",
        HCCL_CONNECT_TIMEOUT="1800",
        HCCL_EXEC_TIMEOUT="3600",
        HCCL_NPU_SOCKET_PORT_RANGE="auto",
        VLLM_NO_USAGE_STATS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TMPDIR=str(Path("/tmp") / ("m3li-v4-" + out.name)),
        XDG_CACHE_HOME=str(out / "cache/xdg"),
        XDG_CONFIG_HOME=str(out / "cache/config"),
        VLLM_CONFIG_ROOT=str(out / "cache/config"),
        VLLM_CACHE_ROOT=str(out / "cache/vllm"),
        TRITON_CACHE_DIR=str(out / "cache/triton"),
        TORCH_EXTENSIONS_DIR=str(out / "cache/torch"),
        ASCEND_CACHE_PATH=str(out / "cache/ascend"),
        ASCEND_WORK_PATH=str(out / "cache/work"),
        ASCEND_PROCESS_LOG_PATH=str(out / "logs"),
        DSA_SHARED_DDR_ROOT=str(out / "shared-ddr"),
        DSA_M2_LI_V0_ORACLE="0",
        VLLM_ASCEND_LW_DEFER_LAST_SAVE="0",
    )
    return env, sdk


def run(arm, include=""):
    global STOP
    if arm not in preflight.ARMS:
        raise ValueError("unsupported arm")
    lock = (ROOT / "service.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if time.time() >= spec.STOP_NEW:
        raise TimeoutError("no new experiment after cutoff")
    profile = preflight.check(arm)
    out = ROOT / "runs" / arm
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("arm output already exists; use a fresh root")
    for name in (
        "logs",
        "cache/ascend",
        "cache/work",
        "cache/xdg",
        "cache/config",
        "cache/vllm",
        "cache/triton",
        "cache/torch",
        "client/workers",
        "shared-ddr",
    ):
        (out / name).mkdir(parents=True, exist_ok=True)
    (Path("/tmp") / ("m3li-v4-" + arm)).mkdir(mode=0o700, parents=True, exist_ok=True)
    env, sdk = environment(profile, out)
    ports = free_ports(4)
    env["MASTER_PORT"] = str(ports[-1])
    env["BENCH_ARM"] = arm
    env["DSA_M2_SMOKE_OUTPUT"] = str(out / "client/workers")
    env["BENCH_OUTPUT"] = str(out / "client")
    env["BENCH_ADMISSION"] = str(out / "admission.json")
    if arm != "v0":
        sdk_config = sdk / "memcache_hybrid/config"
        common = {"ock.mmc.meta_service_url": "tcp://127.0.0.1:" + str(ports[0]), "ock.mmc.log_level": "error"}
        (out / "meta.conf").write_text(
            config(
                sdk_config / "mmc-meta.conf",
                dict(
                    common,
                    **{
                        "ock.mmc.log_path": str(out / "logs"),
                        "ock.mmc.meta_service.config_store_url": "tcp://127.0.0.1:" + str(ports[1]),
                        "ock.mmc.meta_service.metrics_url": "http://127.0.0.1:" + str(ports[2]),
                        "ock.mmc.evict_threshold_high": "90",
                        "ock.mmc.evict_threshold_low": "89",
                    },
                ),
            )
        )
        (out / "local.conf").write_text(
            config(
                sdk_config / "mmc-local.conf",
                dict(
                    common,
                    **{
                        "ock.mmc.local_service.config_store_url": "tcp://127.0.0.1:" + str(ports[1]),
                        "ock.mmc.local_service.world_size": "64",
                        "ock.mmc.local_service.protocol": "device_sdma",
                        "ock.mmc.local_service.dram.size": "10GB",
                        "ock.mmc.local_service.max.dram.size": "10GB",
                        "ock.mmc.local_service.hbm.size": "0GB",
                        "ock.mmc.local_service.max.hbm.size": "0GB",
                        "ock.mmc.local_service.dram.unrestricted.enabled": "true",
                    },
                ),
                forbidden=("ock.mmc.log_path",),
            )
        )
        env["MMC_META_CONFIG_PATH"] = str(out / "meta.conf")
        env["MMC_LOCAL_CONFIG_PATH"] = str(out / "local.conf")
    result = dict(
        arm=arm,
        status="starting",
        NPU_admitted=False,
        source_revision=profile["revision"],
        devices=list(DEVICES),
        force_model_cleanup_used=False,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )
    atomic(out / "private-controller-identity.json", owned.process(os.getpid()), private=True)
    meta = client = None
    meta_scope, client_scope = {}, {}
    try:
        if arm != "v0":
            with (out / "private-meta.log").open("x") as log:
                meta = subprocess.Popen(
                    [sys.executable, "-c", "from memcache_hybrid import MetaService; raise SystemExit(MetaService.main())"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=out,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            meta_scope[meta.pid] = capture(meta)
            until = time.time() + 60
            while True:
                if meta.poll() is not None:
                    raise RuntimeError("MetaService startup failed")
                try:
                    with socket.create_connection(("127.0.0.1", ports[0]), timeout=0.2):
                        break
                except OSError:
                    if time.time() > until:
                        raise TimeoutError("MetaService startup timeout")
                    time.sleep(0.2)
        command = [
            sys.executable,
            str(HARNESS / "run_cases.py"),
            "--arm",
            arm,
            "--engine-args",
            profile["config"],
            "--corpus",
            profile["corpus"],
            "--output",
            str(out / "client"),
            "--admission",
            str(out / "admission.json"),
        ]
        if include:
            command.extend(["--include", include])
        with (out / "private-client.log").open("x") as log:
            client = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=out,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        client_scope[client.pid] = capture(client)
        idle_count, observations = 0, []
        while client.poll() is None:
            live = owned.descendants(client_scope)
            if meta is not None:
                owned.descendants(meta_scope)
                if meta.poll() is not None:
                    raise RuntimeError("MetaService exited during model lifetime")
            if STOP or time.time() >= spec.DEADLINE:
                raise TimeoutError("finite experiment deadline reached")
            gate_path = out / "admission.json"
            if not result["NPU_admitted"] and gate_path.exists():
                gate_value = json.loads(gate_path.read_text())
                gate_ids = {(w["pid"], w["startticks"]) for w in gate_value.get("workers", [])}
                live_ids = {(w["pid"], w["startticks"]) for w in live}
                if (gate_value.get("allow_npu") is True and gate_value.get("expires", 0) > time.time()
                        and len(gate_ids) == spec.TP * spec.DP and gate_ids.issubset(live_ids)):
                    result.update(NPU_admitted=True, admitted_epoch=time.time(), admission_owner="root")
            if not result["NPU_admitted"]:
                if time.time() >= spec.STOP_NEW:
                    raise TimeoutError("NPU admission cutoff reached")
                ready = [json.loads(p.read_text()) for p in (out / "client/workers").glob("import-ready-*.json")]
                actual = {(p["pid"], p["startticks"]) for p in live}
                prepared = (
                    len(ready) == spec.TP * spec.DP
                    and len({r["pid"] for r in ready}) == spec.TP * spec.DP
                    and all((r["pid"], r["startticks"]) in actual for r in ready)
                )
                result["CPU_workers_ready"] = len(ready)
                if prepared:
                    atomic(out / "public-CPU-ready.json", dict(workers=ready, controller=owned.process(os.getpid()), NPU_admitted=False))
                unit_gate = ROOT / "public-unit-gate.json"
                unit_status = json.loads(unit_gate.read_text()) if unit_gate.exists() else {}
                unit_ok = unit_status.get("status") == "passed" and unit_status.get("remaining_owned") == 0
                result["unit_gate_passed"] = unit_ok
                if prepared and unit_ok:
                    rows = hardware()
                    idle_count = idle_count + 1 if idle(rows) else 0
                    observations = (
                        observations + [dict(time=time.time(), hardware=rows)] if idle_count else []
                    )[-2:]
                    result["hardware"] = rows
                    if idle_count >= 2:
                        preflight.check(arm)
                        live = owned.descendants(client_scope)
                        if not all(
                            (r["pid"], r["startticks"]) in {(x["pid"], x["startticks"]) for x in live}
                            for r in ready
                        ):
                            raise RuntimeError("worker identity changed during admission")
                        atomic(
                            out / "public-ready-to-admit.json",
                            dict(allow_npu=True, workers=ready, expires=time.time() + 600, idle_snapshots=observations),
                        )
                        result["root_gate_candidate_ready"] = True
            progress = out / "client/public-progress.json"
            result["progress"] = json.loads(progress.read_text()) if progress.exists() else None
            result["status"] = "running" if result["NPU_admitted"] else "CPU_preimport_waiting_idle"
            atomic(out / "private-owned.json", {str(k): v for k, v in client_scope.items()}, private=True)
            atomic(out / "public-status.json", result)
            time.sleep(2)
        result["client_returncode"] = client.returncode
        until = time.time() + 120
        while owned.descendants(client_scope) and time.time() < until:
            time.sleep(0.5)
        result["remaining_model_processes"] = len(owned.descendants(client_scope))
        result["status"] = "completed" if client.returncode == 0 and result["remaining_model_processes"] == 0 else "failed"
        preflight.check(arm)
    except BaseException as error:
        import traceback

        (out / "private-controller-error.txt").write_text(traceback.format_exc())
        result.update(status="failed", error_class=type(error).__name__)
    finally:
        if client is not None and (client.poll() is None or owned.descendants(client_scope)):
            result["force_model_cleanup_used"] = True
            owned.retire(client_scope)
        if meta is not None:
            owned.descendants(meta_scope)
            if meta.poll() is None:
                meta.send_signal(signal.SIGTERM)
                try:
                    meta.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    owned.retire(meta_scope)
            if owned.descendants(meta_scope):
                owned.retire(meta_scope)
            result["Meta_returncode"] = meta.poll()
        result["remaining_owned"] = len(owned.descendants(client_scope)) + len(owned.descendants(meta_scope))
        atomic(out / "public-result.json", result)
        atomic(out / "public-status.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=preflight.ARMS)
    parser.add_argument("--include", default="")
    args = parser.parse_args()
    result = run(args.arm, args.include)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("status") == "completed" and result.get("remaining_owned") == 0 else 1)
