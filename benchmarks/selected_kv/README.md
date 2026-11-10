# M3v15 reproduction on this feature branch

中文入口：[START_HERE_ZH.md](START_HERE_ZH.md)。完整命令：[REPRODUCE.md](REPRODUCE.md)。

Default execution uses the enclosing Git repository as M3 source and prepares
only M3 smoke and trace. Edit `configs/paths.example.json`; config-relative paths
resolve against the repository root. Native comparison is explicit with
`--include-native`, a fixed-baseline external Git worktree and its build receipt.
The M3/native source archives from the earlier ZIP are not duplicated here.

Core source and patched SDK sources remain bundled. Model weights and corpus
contents stay external. Typed source-verified build receipts, fresh CPU gates,
same-build smoke and owned finite execution remain required. A version string
or historical PASS cannot substitute for current validation.

Prior ZIP CPU evidence and historical trace references are preserved and labelled
in `VALIDATION.md` and `HISTORICAL_REFERENCE.md`. This publication changes path
and arm defaults; its CPU checks must be rerun. Full fresh builds/ABI+factory and
new NPU smoke/trace were not executed during this branch preparation. M3-only
execution does not establish a new cross-arm numerical result.

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
