# Reproduce M3v15 from this Git branch

Clone the confirmed fork and feature branch:

```bash
git clone --branch feat/dsa-prefill-ondemand-m3v15 https://github.com/ZD-euler/vllm-ascend.git
cd vllm-ascend/benchmarks/selected_kv
```

The enclosing repository is the default M3 source. Native comparison is optional;
it uses an external Git worktree pinned to baseline
`fbb4d867d22095047473e3c7ca9db053f5f4c7b0`. There are no duplicate M3/native
source archives in this repository. Core and patched SDK source archives remain
because they are independent runtime dependencies.

The 78-layer trace uses GLM-5.2 W8A8/SFA C8/LI C8, TP4/DP4/EP16,
H524288/Q4096, local BS1/global BS4 and chunk6144. There is one warmup and
one profile burst; all sixteen ranks execute and only rank 0 profiles. Functional
smoke uses six layers, TP2/DP2/EP4, global BS2, one warmup, three measure bursts
and one probe. Preserve these contracts when comparing results.

## Environment and local directories

Supply matching A3 Linux/aarch64 hardware, CANN 9.1.0 and licensed GLM-5.2
W8A8 weights. Neither weights nor corpus contents are in Git. The reference
private image is provenance only and is not required or distributed.

```bash
export REPRO_M3="$(git rev-parse --show-toplevel)"
export REPRO_CODE=/data/m3v15-reproduction/code
export REPRO_BUILD=/data/m3v15-reproduction/build
export REPRO_DATA=/data/m3v15-reproduction/data
export REPRO_RUNS=/data/m3v15-reproduction/runs
export REPRO_WEIGHTS=/models/GLM-5.2-w8a8
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/9.1.0
mkdir -p "$REPRO_BUILD" "$REPRO_RUNS" /data/m3v15-reproduction/tmp
export TMPDIR=/data/m3v15-reproduction/tmp
source "$ASCEND_HOME_PATH/set_env.sh"
```

Create a Python 3.11 environment. The full observed software lock is
`references/requirements-observed-aarch64.txt`; it excludes source-built core,
Ascend and patched SDK packages. Platform-specific wheels may require vendor
indexes or a prepared Linux/aarch64 wheelhouse. For offline installation:

```bash
python3 -m venv /data/m3v15-reproduction/venv
source /data/m3v15-reproduction/venv/bin/activate
python3 -m pip install --no-index --find-links /data/aarch64-wheelhouse --no-deps \
  -r references/requirements-observed-aarch64.txt \
  -r references/requirements-corpus-tools.txt
```

Pyarrow 21.0.0 is an explicit corpus-tool dependency, verified in the earlier
Linux/aarch64 offline rebuild; it is not claimed as the historical target venv
version. For corpus-only work without changing a shared environment:

```bash
mkdir -p /data/m3v15-reproduction/cpu-deps
python3 -m pip install --no-deps --target /data/m3v15-reproduction/cpu-deps \
  -r references/requirements-corpus-tools.txt
export PYTHONPATH=/data/m3v15-reproduction/cpu-deps${PYTHONPATH:+:$PYTHONPATH}
```

Keep `TMPDIR` on task-owned storage with enough space; the reference attempt
initially failed because the default `/tmp` was too small.

## Build the M3 arm

```bash
python3 tools/unpack_sources.py --output "$REPRO_CODE"
python3 tools/prepare_core_metadata.py --core "$REPRO_CODE/core-native"
python3 tools/build_base_ops.py --cann "$ASCEND_HOME_PATH" --output "$REPRO_BUILD/base-m3"
python3 tools/build_features.py --cann "$ASCEND_HOME_PATH" \
  --output "$REPRO_BUILD/m3-features" --replace-feature-targets
python3 tools/build_sdk.py --output "$REPRO_BUILD/sdk17" --build-aicpu
```

Both Ascend builders default to the enclosing repository. The base recipe uses
the original setup entries for custom ACLNN operators, extension and egg-info
registration. The selected recipe builds four shared libraries and one fused
device object. Never substitute a private-install `fused_fixed.so` wrapper for
`libsparse_prefill_fused_fixed_launch.so`.

The SDK recipe verifies the supplied C-ABI base and common/AddInfo/build plus
opt-in DDR patches. Stock pip SDK 1.3.0 is insufficient. All three recipes emit
typed receipts with fixed input hashes, commands, toolchain and scoped output
hashes. Receipts mean `compiled_not_NPU_validated`. Actual CPU load/factory and
same-build smoke checks remain mandatory. Base metadata can legitimately be
0.0.0 without original Git metadata; the receipt records it. Production identity
comes from fixed file hashes, not a version string.

Only publication `README.md` and `.gitignore` are excluded from original
source hash gates. Their complete original map remains in
`references/original-pinned-source-identities.json`; all other original pinned
files remain checked. The feature branch's Git commit differs from the original
M3 code label because it adds reproduction materials. See `artifacts/source/REVISIONS.json`.

## Prepare model and public input

```bash
python3 tools/prepare_six_layer_payload.py --full-model "$REPRO_WEIGHTS" \
  --output "$REPRO_BUILD/model6"
python3 tools/create_model_view.py --weights "$REPRO_WEIGHTS" \
  --reference references/model78 --output "$REPRO_BUILD/model78"
python3 tools/regenerate_corpus.py --provenance references/corpus-manifest.json \
  --tokenizer "$REPRO_WEIGHTS" --output "$REPRO_DATA"
```

Six-layer preparation selects 7153 recorded tensors and reference shard
partitions, preserving dtype and checking raw-byte fingerprints. Original
full/six-layer CPU comparisons matched all 7153 names. Full 78-layer tensor
values have not all been hashed; model index identity alone is insufficient to
claim complete numerical weight identity.

For a target without public DNS/network, download the fixed public archives on
a networked machine and make those archives locally available:

```bash
python3 tools/download_artifacts.py --output /data/public-artifact-archives
# Optional --official-hf uses huggingface.co with the identical recorded path.
python3 tools/regenerate_corpus.py --provenance references/corpus-manifest.json \
  --tokenizer "$REPRO_WEIGHTS" --artifact-dir /data/public-artifact-archives \
  --output "$REPRO_DATA"
```

Both paths check archive, canonical-text and final token hashes. Do not skip a
hash after a download failure. The original recorded windows are retained.

## Default: M3 smoke and trace

Copy `configs/paths.example.json` to your local paths file and edit it. A missing
`m3_source`, or `"m3_source": "."`, means the enclosing Git repository. Relative
paths in this JSON are resolved against the repository root, not your shell's
current directory. Keep data/build/run paths outside Git.

```bash
python3 tools/prepare_suite.py --paths-config /data/my-paths.json --output "$REPRO_RUNS/m3-suite"
export REPRO_RUNS="$REPRO_RUNS/m3-suite"
python3 tools/launch_run.py --run-root "$REPRO_RUNS/smoke-m3" --execute-npu
python3 tools/launch_run.py --run-root "$REPRO_RUNS/trace-m3" --execute-npu \
  --smoke-root "$REPRO_RUNS/smoke-m3"
python3 tools/export_trace.py "$REPRO_RUNS/trace-m3"
```

Preparation does not start a model. Explicit launch requires root mount namespace
capability, idle selected devices, fresh CPU gates, shared host device locks and
own-worker admission. A finite deadline and PID/startticks ownership bound
cleanup; no foreign workload is stopped or device reset. Trace requires clean
same-build six-layer smoke. M3-only execution validates that arm's captures; it
**does not establish a new M3/native numerical comparison**.

## Optional fixed-baseline native comparison

Use a second fresh suite/output root. Obtain the fixed baseline commit and
create an external worktree; do not put it inside the M3 repository:

```bash
git -C "$REPRO_M3" fetch origin fbb4d867d22095047473e3c7ca9db053f5f4c7b0
git -C "$REPRO_M3" worktree add --detach /data/m3v15-reproduction/native-baseline \
  fbb4d867d22095047473e3c7ca9db053f5f4c7b0
python3 tools/build_base_ops.py --arm native --source /data/m3v15-reproduction/native-baseline \
  --cann "$ASCEND_HOME_PATH" --output "$REPRO_BUILD/base-native"
```

Copy/edit `configs/paths.with-native.example.json`; reuse the same core, SDK,
models, corpus and M3 receipts, and supply the external `native_source` plus
`native_base_receipt`. Then:

```bash
python3 tools/prepare_suite.py --include-native --paths-config /data/my-pair-paths.json \
  --output /data/m3v15-reproduction/pair-runs
export REPRO_RUNS=/data/m3v15-reproduction/pair-runs
```

Run the M3 smoke/trace/export commands above for this new suite, then:

```bash
python3 tools/launch_run.py --run-root "$REPRO_RUNS/smoke-native" --execute-npu
python3 tools/launch_run.py --run-root "$REPRO_RUNS/trace-native" --execute-npu \
  --smoke-root "$REPRO_RUNS/smoke-native"
python3 tools/export_trace.py "$REPRO_RUNS/trace-native" native
python3 tools/compare_traces.py --native-root "$REPRO_RUNS/trace-native" \
  --m3-root "$REPRO_RUNS/trace-m3" --output "$REPRO_RUNS/trace-comparison.json"
```

Comparison preserves the original first-token and finite-logprob 1e-5 tolerance
criteria and reports exact equality separately. Single profiled timings include
profiler overhead; ranks are not summed and no speedup ratio is guaranteed.
Device capture still needs offline device review. Historical references and
prior CPU evidence are in `HISTORICAL_REFERENCE.md` and `VALIDATION.md`; neither
is a current PASS gate. Fresh full builds/ABI+factory and NPU smoke/trace have
not been executed during preparation of this publication branch.

## Publication normalization

Recovered reference templates use the non-personal `/repro/reference-root` alias.
`prepare_run.py` replaces that template marker using explicit local paths; no
seed/burst/profiler logic changes. Original-to-publication SHA256 mappings are in
`references/publication-privacy-transform.json`. The SDK archive has normalized
owner/timestamp headers; its member content hashes remain unchanged, and the
strict archive/handoff/payload hash chain is updated. Original production pins
remain unchanged except the two documented publication file exceptions.

The exact SDK unified-diff path has a scoped `.gitattributes` exception for
`blank-at-eol`: its single-space empty-context markers are patch syntax. Patch
bytes and SHA256 remain unchanged. The recovered handoff only loses a redundant
EOF blank line; its publication deploy hash and provenance are updated.
