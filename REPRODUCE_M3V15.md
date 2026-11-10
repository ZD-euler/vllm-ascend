# M3v15 复现入口

中文说明：[benchmarks/selected_kv/START_HERE_ZH.md](benchmarks/selected_kv/START_HERE_ZH.md)。
完整命令：[benchmarks/selected_kv/REPRODUCE.md](benchmarks/selected_kv/REPRODUCE.md)。

默认 M3 源码就是当前仓库，默认只做 M3 smoke/trace。
native 对照需显式启用，并使用固定 baseline 的仓库外 Git worktree。
不把模型、语料、env、编译产物或 runs/results 提交 Git。

本分支增加复现材料，原始 M3v15 生产源码标签与本分支发布 commit 是不同身份；
原始文件哈希与 commit 来源见 `benchmarks/selected_kv/artifacts/source/REVISIONS.json`。
保留上游说明和许可证。原有 ZIP 保持不变；其 CPU 验证是历史证据，不冒充本分支新结果。
