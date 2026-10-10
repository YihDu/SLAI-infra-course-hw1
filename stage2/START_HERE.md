# Stage 2 启动与提交指南

更新：2026-10-10。此文整理现有材料和待执行步骤，不是已完成实验的报告。
启动资料已齐全。权威依据为 [BACKGROUND](BACKGROUND.md)、[README](README.md)
及随包评测器，本指南不替代完整技术契约。

## 阅读顺序与材料

1. [BACKGROUND.md](BACKGROUND.md)：学习目标、MIP 与多智能体框架要求。
2. [README.md](README.md)：五个负载、硬件、时序、评分和提交。
3. [WALKTHROUGH.md](WALKTHROUGH.md)：构建、小例子、导出与评测。
4. [ISA.md](ISA.md)、[ABI.md](ABI.md)：静态程序、输入输出和存储契约。
5. [Stage 1 状态](../stage1/project/STAGE1_STATUS.md)：已有方案、实验和 MILP 证据。

包内提供 `source/`、`source/Cargo.lock`、`tools/`、十类 `configs/`、
`examples/`、五个 `programs/reference/*.jsonl` 和 `release-contract.json`。
Python 工具需要 3.11+，Rust 编译需要 1.90+；旧 Python/NumPy 精确版本限制
只属于 Stage 1。当前机器为 x86_64，发布方尚未宣称该平台已验证，需本地验证。

## 要做什么

产出一套公共硬件和五个静态程序：`w-p`、`a-p`、`w-d1`、`w-d16`、`w-d4l`。
两个 Prefill 各占评分权重 1/4，三个 Decode 各占 1/6；所有数值、面积、功率
与资源检查必须通过。模型 `phase-two-compact-v2`，契约 `phase-two-static-v1`。

同时产出可运行的多智能体框架和具有实质决策变量的 MIP。职责覆盖知识提取、
建模、求解执行、验证修订；说明信息交换、分歧处理和知识保留。
只从已完成实验选最高分，或让多个 agent 独立暴力搜索，不满足建模目标。

至少展示一条真实链条：

**Stage 1 证据 → 优化变量 → 剪枝与建模 → 求解提案 → 仿真验证 → 模型修订。**

区分不可行剪枝、可证明支配和启发式剪枝，说明后者风险及重访方式。
区分模型内最优与机器全局最优。用相同新版模拟器和可比预算对照原搜索方法。
报告评测次数、失败/超时/重试、墙钟时间、可用 token 统计、求解时间/status/gap、
预测与实测误差。未知计数标明缺失，不填零。

## 启动顺序

以下均为待执行步骤。本次没有运行这些命令。工作目录为 `stage2/`；
每次尝试使用新的输出目录，不覆盖旧报告。

先构建并验证小例子：

```sh
cargo build --release --manifest-path source/Cargo.toml --features compact --bin vnext-concurrent
mkdir -p runs
source/target/release/vnext-concurrent evaluate examples/small-prefill.jsonl 7 runs/small-check-001
```

小例子验证执行链，不具有官方五场景形状，不能作为正式成绩。
然后直接使用随包 compact 参考程序建立本地基线：

```sh
python3 tools/grade_v09.py programs/reference runs/reference-001 \
  --binary source/target/release/vnext-concurrent --seed 7 \
  --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 600
```

随后在本目录建立自己的 `project/` 框架与知识库，提取旧证据，定义首版 MIP，
生成候选并记录预测，评测后修订。参考生成器用 `export` 产生程序，
再由 `tools/compact_program.py` 压缩表达，具体命令见 walkthrough。
算法不限于参考生成器 policy 菜单，但生成结果必须符合静态契约。

## 最终自测

用 `tools/grade_v09.py` 评测最终五个程序，保存以下机器生成原件：

```text
<grade-dir>/grade.json
<grade-dir>/contract/result.json
<grade-dir>/workers/summary.json
<grade-dir>/workers/w-p/evaluation/result.json
<grade-dir>/workers/a-p/evaluation/result.json
<grade-dir>/workers/w-d1/evaluation/result.json
<grade-dir>/workers/w-d16/evaluation/result.json
<grade-dir>/workers/w-d4l/evaluation/result.json
<grade-dir>/programs/<case>.jsonl
```

记录命令、seed、模型/契约/源码哈希、程序哈希和主机环境。
改动最终程序必须重测；不得手改 `grade.json` 或只提供截图。
本地报告是复现证据，最终成绩由组织方用私有 seed 重评。

## 提交材料

| 材料 | 内容 |
| --- | --- |
| 可执行结果 | 五个最终 JSONL、对应公共硬件配置 |
| 自测 | 汇总、契约、worker 汇总与五场景原始报告 |
| 可复现框架 | 多智能体实现、生成器、配置、依赖与运行命令 |
| 数学模型 | 变量范围、约束、目标、假设、求解设置/status/gap |
| 知识与实验 | Stage 1 证据、剪枝依据、预测/实测、失败与模型修订 |
| 分析报告 | 公平对照、搜索成本、结论与自身贡献 |
| 原生过程记录 | 相关会话与材料索引；说明缺失范围，隐去敏感凭证 |

官方要求相关失败、重启和子 agent 记录；旧用户指令只授权收集当前主会话。
差异尚未统一，本指南不擅自改变用户指令。保留已有历史，启动额外会话收集
流程前统一策略，不将较窄的记录范围宣称为完整覆盖官方要求。

## 打包与提交

命令模板如下，占位路径需替换为实际成果：

```sh
python3 tools/package_phase_two.py runs/FINAL_GRADE submission.zip \
  --materials PATH_TO_FRAMEWORK_AND_CONFIGS \
  --materials PATH_TO_REPORT_AND_EVIDENCE \
  --materials PATH_TO_NATIVE_TRACES
```

工具自动加入已评分程序快照和所需自测文件；源码、单独硬件配置、依赖、报告
与原生记录须通过 `--materials` 加入。提供短材料索引，检查 ZIP 内材料齐全。
打包成功本身不证明五场景评测通过。

限制：ZIP ≤256 MiB；解压 ≤1 GiB；条目 ≤10000；每个程序 ≤128 MiB。
不要打包依赖/构建缓存；不允许软链接、不安全路径、重复条目或加密 ZIP。
保留相关原始证据，去掉重复中间产物与凭证。

提交到 [Phase Two 独立入口](https://linux-slai.tail6d76d1.ts.net:8443/phase-two/submit/)。
文档截止：**2026-10-12 12:00 北京时间（04:00 UTC）**。
网站登录、在线可达性和最新公告尚未核验；本地运行无需 portal 访问。
