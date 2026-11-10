# 从这里开始：默认只复现 M3v15

克隆本特性分支后进入 `benchmarks/selected_kv`，按
[REPRODUCE.md](REPRODUCE.md) 的命令顺序执行。M3 源码默认就是仓库根目录；
不需要下载重复的 M3/native 源码 tar，也不需要原机器目录或历史对话。

正式 trace：GLM-5.2 W8A8、SFA C8、LI C8，78 层、TP4/DP4/EP16，
H524288/Q4096、local BS1/global BS4、chunk6144。
一次 warmup 加一次 profile，16 个 rank 执行，仅 rank 0 采集。
先做 6 层 TP2/DP2/EP4 的四卡功能 smoke（一 warmup、三 measure、一 probe）。

另备匹配的 A3 Linux/aarch64 硬件、CANN 9.1.0、Python 3.11 环境和有权使用的
GLM-5.2 W8A8 完整权重。仓库不含权重、prompt、token 数组或语料正文。
软件版本锁、独立 core 源码和带补丁 SDK17 的完整来源在本目录中。
不要使用普通 pip SDK 1.3 代替带补丁 SDK17。

顺序如下，完整可执行命令均在英文 runbook：

1. 安装锁定的依赖，使用任务专属 `TMPDIR`；解压 core 并生成其原生包元数据。
2. 在当前仓库构建基础算子和 M3 的四库一对象，构建带补丁 SDK，保留三类 build receipt。
3. 从用户完整模型生成 6 层选定 payload，保留 dtype 并校验 7153 个 tensor；准备 78 层 loader view。
4. 按固定公开 archive、正文和 token 哈希重建语料。可用 `--artifact-dir` 完全离线读取预下载 archive。
5. 复制并修改唯一的 `configs/paths.example.json`，然后执行：

```bash
python3 tools/prepare_suite.py --paths-config /data/my-paths.json --output /data/fresh-m3-runs
export REPRO_RUNS=/data/fresh-m3-runs
```

默认只生成 `smoke-m3` 和 `trace-m3` 两个目录，不启动模型。
JSON 中 `m3_source` 可省略或写 `.`；相对路径都相对于 Git 仓库根目录，不相对于 shell cwd。
按 runbook 显式使用 `--execute-npu` 先 smoke 再 trace，最后 `export_trace.py`。
单臂输出只能证明该臂的验证结果，不能宣称完成了新的 M3/native 数值对照。

需要 native 对照时，使用 `configs/paths.with-native.example.json`，
按 runbook 在仓库外创建固定 commit `fbb4d867d22095047473e3c7ca9db053f5f4c7b0`
的 Git worktree，构建其基础算子并填写 native receipt；显式添加 `--include-native`。
完成两臂本次 smoke/trace 后才运行 `compare_traces.py`。

历史 ZIP 版本已验证 CPU 工具测试、Linux ownership/信号清理、16 源语料离线重建和
7153×2 模型 tensor 字节哈希。本 Git 发布版本调整了默认路径和可选对照结构，
这些改动的本次验证应另行记录，不拿历史 PASS 代替。
尚未完成全新完整源码构建/实际 ABI+factory/新 NPU smoke 或 trace。
历史 78 层数值一致与单 profile 时延只是参照，不承诺加速比；parallel ranks 不求和。

检查每次 gate、退出码和 `remaining_owned`，失败时不要绕过验证。
模型、数据、build、env、runs/results 一律放在仓库外；`.gitignore` 是额外防线。
顶层保留上游 README 和许可证。本分支发布 commit 不等于原始 M3v15 commit；
生产源码身份仍由固定原始文件哈希追踪，只对发布 README/.gitignore 作明确文档例外。
