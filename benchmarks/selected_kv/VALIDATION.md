# Prior ZIP validation — 2026-11-10

The evidence below belongs to the earlier standalone ZIP. This Git publication
changes source layout and default arms; new publication CPU results must be
recorded separately. No fresh full build/ABI factory/NPU result is implied.

These are separate evidence categories. A static review, successful compilation,
a CPU test, a smoke run and a trace comparison do not substitute for one another.

## Completed local CPU checks

- Tooling suite: 17 tests, 15 passed and two Linux-only tests skipped; exit 0.
- The four source archives were actually unpacked into three fresh source
  directories under a different temporary root; archive SHA256 checks and
  extraction completed with exit 0.
- Command-line help checks completed with exit 0.
- Core runtime snapshot: 3155 regular files matched the observed reference
  hashes, with no missing, changed or extra files. This is content identity,
  not a claim that the core is a complete Git checkout.

The original 12-test tooling suite also completed with exit 0 before the final
independent boundary tests were added. Current tests include package integrity,
receipt namespaces/tampering, stale PASS rejection, smoke lineage, input-window
hash behavior and isolated process/relocation boundaries. On the reference Linux host all 17 tests passed with no skips, exit 0, including
real PID/startticks ownership protection, launcher SIGTERM cleanup and shared
flock release. The final delivered snapshot is checked again after packaging.

## Completed reference-host CPU tensor checks

All 7153 six-layer selected tensor names exist in the full-model index.
A read-only CPU check compared each tensor's contiguous raw bytes against the
recorded fingerprint independently in both the full checkpoint and original
six-layer payload. All matched, without dtype conversion; the final report run
exited 0. A 12-tensor sample also passed. An earlier attempt failed only while
writing its report to an incorrect path and is not counted as a passing run.
Published hash-only reports are in `references/public-model6-*-verification.json`.
These are CPU payload checks, not NPU model validation.

## Public corpus reconstruction

The direct URL attempt failed during DNS resolution of the recorded public
host, with curl exit 6, before extraction. This is not a successful input rebuild.
The explicit offline route subsequently completed successfully on the reference
Linux/aarch64 host, exit 0, using 16 retained public archive files whose original
SHA256 hashes were checked. It rebuilt 16 sources and 16 cases, totalling
2,055,318 tokens. All 16 archive, canonical-text and uint32 token hashes matched;
no synthetic padding was used. No model or NPU test was launched.

Pyarrow 21.0.0 was installed into an isolated task-owned `--target` directory,
without changing the shared environment. Its ARM CPython 3.11 wheel SHA256 was
`479ee41399fcddc46159a551705b89c05f11e8b8cb8e968f7fec64f62d91985e`.
An initial installation attempt exited 1 because the default `/tmp` filesystem
had insufficient space. Setting a task-owned `TMPDIR` allowed installation to
complete with exit 0. The successful corpus command used that `TMPDIR`, the
isolated dependency directory on `PYTHONPATH`, and
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`:

```bash
PYTHONPATH="$TASK_ROOT/cpu-deps" TMPDIR="$TASK_ROOT/tmp" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 tools/regenerate_corpus.py --provenance references/corpus-manifest.json \
  --tokenizer "$USER_MODEL" --artifact-dir "$TASK_ROOT/offline-public-artifacts-cache-v1" \
  --output "$TASK_ROOT/data-corpus-rebuild-offline-v2"
```

Task/model-specific absolute paths are represented by portable aliases above;
flags and environment settings retain the actual successful invocation. Direct
fresh URL downloading remains unverified because of the earlier DNS failure;
the complete offline archive-to-token reconstruction is verified.

## Static review and pending runtime work

Astra reviewed non-test implementation code and reported no remaining concrete
static delivery blocker. Sol independently reviewed test code. These reviews
are not execution results.

This delivery preparation has not run fresh full source builds, target ABI and
factory checks using newly built outputs, or new NPU smoke/trace tests. The
launcher's fresh gates must still run on the reproducer's machine. No historical
PASS is reused as current acceptance. Full 78-layer weight values have not been
completely hashed; index identity alone does not prove full numerical identity.
Captured traces require offline device review before a device-correctness claim.


## Git publication revision: local CPU validation

The publication layout was checked separately from the earlier ZIP:

- Local suite: 23 tests, 21 passed, two Linux-only tests skipped, no failures;
  exit 0.
- All 21 Python tools passed AST parsing and command-line help checks.
- Both retained core source archives passed SHA256 checks and actual unpacking;
  exit 0. No duplicate Ascend source archives are required.
- `git diff --check` exited 0. The production-only commit remains
  `721e1e985f477e9f8bb785c47fdbd3b4dcb6ef94`, whose tree is exactly
  `19989cdf87d70f6a3874374fb4cdfe0ecbbace89`, the original M3v15 tree.

This publication revision's Linux suite was **not executed**. Automatic approval
review rejected the attempted bulk upload because its references scope included
corpus provenance and environment metadata. The rejected upload was not retried
or bypassed. The earlier ZIP's 17/17 Linux pass is historical evidence and does
not establish a Linux pass for these publication changes.

No fresh complete build, actual rebuilt-artifact ABI/config factory, or new NPU
smoke/trace result is claimed. Public GitHub privacy review is still pending;
these changes have not been committed or pushed.
