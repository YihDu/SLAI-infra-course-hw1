# P1 end-to-end multi-SM：实现、完整对照与验证阻塞

## 结论

实现了完整四 SM P1 skeleton（cand-099），包括两个 prompt、三层 Transformer、prompt commit 和两个首个生成位置。D1 与硬件保持 cand-098 不变。

**尚未获得 correctness-pass 的完整并行程序，也没有并行 P1 cycles/score。** 完整 grade 在 frozen HBM race checker 阶段耗尽 6 GiB 预算。不能据此宣称并行路线失败，更不能宣称达到 >1.5×。

同一分片运算串行执行的完整对照 cand-100 通过 correctness、面积、功耗、周期门槛。它确认数值与 ABI 分片没有错误，但不证明跨 SM 同步正确。默认程序保持 cand-098；并行实现作为 opt-in recipe 保留。

## 1. 完整 execution plan

四个持久 workgroup g0..g3，固定在 SM0..3，每个 batch member 的每一层执行：

| 阶段 | 分片 / 每 SM 工作 | 依赖与驻留 |
|---|---|---|
| LN1 | 64 prompt token / 4 = 16 行；行内完整 256 维 | 独立行归约；输出 HBM，barrier |
| QKV | 768 输出列 / 4 = 192 列；全部 64 行 | 复用 nkm_lanes GEMM；完整输入 stride；barrier |
| Attention | 8 heads / 4 = 2 heads；该 head 的全部 queries/KV | KV 写入各自 ABI 分区；scores/probs 在 RF；context HBM，barrier |
| WO | 256 输出列 / 4 = 64 列 | 每片读取完整 context；barrier |
| residual1 → LN2 | 16 token 行 | 保持 row owner；中间 HBM；两阶段间无跨 SM barrier，结束后 barrier |
| W1 → bias/GELU | 1024 FFN 列 / 4 = 256 列 | 相同 column owner；W1→GELU 无跨 SM barrier；激活 HBM，之后 barrier |
| W2 | 256 输出列 / 4 = 64 列 | 读取完整 FFN 激活；barrier |
| residual2 → 下一层 LN1 | 16 token 行 | 保持 row owner，无额外跨 SM barrier |

不切 K/reduction 维度，不需要跨 SM 求和。第一版共 84 个 barrier、1 个 STEP.COMMIT、0 个显式 WAIT。RF/SH 私有，因此跨 workgroup 中间值保留 HBM 交换。HBM scratch 地址、公开输出和 KV 地址由生成器管理。

首个生成 token：LN/residual/output 由 SM0 执行；QKV/WO/W1/W2 仍四路列分片，Attention 仍四路 head 分片。两个 batch member 顺序安排；之后可研究 batch overlap，但本次未引入。

## 2. 完整实验结果

| 候选 | 实际执行模式 | P1 cycles | D1 cycles | Score | 结论 |
|---|---|---:|---:|---:|---|
| cand-098 | 原始单 WG | 8,522,411 | 718,132 | 2567.9541 | 当前可提交最优 |
| cand-099 | 完整四 SM | 未取得 | 未取得 | 未取得 | `not_verified / memory_limit`，不是 incorrect |
| cand-100 | 四份分片运算，单 WG 串行对照 | 8,544,671 | 718,132 | 2564.6069 | 完整 correctness + grade eligible；不晋升 |

cand-100 相对 cand-098 的 P1 慢 0.261%；不是四 SM 加速结果。两者面积均 16.25838297088 mm²，P1/D1 峰值滚动功耗为 2.9710608543 / 2.3857896063 W。

`control-verification.json` 证明：cand-099 和 cand-100 除 WG 归属/生命周期及 barrier 参与组外，运算、地址、事件、循环完全相同；D1 字节相同；硬件相同；新 compiler 默认分支生成的两份 asm 与 cand-098 字节相同。

证据：`project/experiments/cand-099/{metrics.json,reports/grade.heartbeat.json}`；`project/experiments/cand-100/reports/grade.json`。实验均已进入现有 DAG、registry、iteration-log；没有修改搜索基础设施或 frozen evaluator。

## 3. 四个 SM 是否实际工作

**并行候选未完成 timing，实际 utilization 未知。** 不用静态工作量冒充利用率。

| SM | cand-099 静态分配指令 | MMA FLOPs |
|---|---:|---:|
| 0 | 41,732 | 156,648,960 |
| 1 | 41,488 | 156,648,960 |
| 2 | 41,488 | 156,648,960 |
| 3 | 41,488 | 156,648,960 |

四路均分全部 MMA，SM0 多承担单 token 向量阶段。静态请求读/写合计约 112 MB / 8.12 MB；这不是经过 cache 后的真实 HBM traffic，详细字节数见 `static-schedule.json`。

cand-098 实测只有 SM0 活跃，SM1..3 无指令。SM0 TC utilization 6.876%，RF read/write utilization 33.069% / 32.717%；全芯片平均 TC utilization 约 1.719%。这些指标是资源占用率，不是核心算力效率或已证明的 critical-path 瓶颈。

## 4. 完整 baseline Amdahl attribution

使用 frozen `estimate_pipeline(..., trace={})` 原生接口，逐事件观察完全不变的 cand-098 程序；重现 8,522,411 cycles。脚本 `profile.py` 不 patch evaluator，也不绕过 race checker。

归因口径：单 WG 按每条事件 issue 到下一条 issue 的区间归给该事件所属阶段，末事件延伸至总 cycles。含数据依赖、调度、访存等待；这不是各 kernel 的独立运行延迟。总和加最初 1 cycle 等于完整 P1 周期。该口径不能直接用作多 WG 的无重叠阶段占比。

| 阶段 | 归因 cycles | 占比 |
|---|---:|---:|
| LN1 | 171,148 | 2.01% |
| QKV | 1,339,482 | 15.72% |
| Attention | 1,811,460 | 21.26% |
| WO | 448,674 | 5.26% |
| residual1 | 47,214 | 0.55% |
| LN2 | 171,180 | 2.01% |
| W1 | 1,784,946 | 20.94% |
| bias/GELU | 904,854 | 10.62% |
| W2 | 1,761,882 | 20.67% |
| residual2 | 73,434 | 0.86% |
| 输出 | 8,136 | 0.10% |

四个 dense GEMM 合计 62.60%。假设仅它们都达到 microbenchmark 3.28×、其余阶段完全不变，Amdahl 推算约 **1.77× P1**。它说明路线有价值，**不是**完整多 SM 实测；cache、NoC 竞争、同步、不同 GEMM 形状均可能改变它。

Baseline HBM read/write：47,165,440 / 8,120,320 B；NoC/DMA bytes：120,074,240 B；RF read/write：357,728,608 / 177,223,456 B。SM0 issue_wait=8,360,828 cycles 是全部发射间隙，不能叫作 WAIT/barrier 开销。并行候选同步开销、真实 traffic、RF/TC 利用率均尚未取得。

证据：`project/experiments/cand-098/reports/p1-profile.json`、原始 `grade.json`。

## 5. 验证器阻塞的完整程序证据

- cand-099 P1 展开 166,289 条指令，完整 grade 在约 69.3 秒时达到 6,214 MiB physical footprint，被现有 runtime 的 6,144 MiB 上限终止。
- 独立只读观察对同一完整 P1 调用原始 `validate_hbm_races`。Python trace 只读取局部变量，不修改代码、输入、知识集合或检查结果。
- 处理 10,000 条时，保存的 frozenset 对象合计 1,066,128,280 B；16,000 条时为 2,525,337,920 B，尚不足完整程序的 10%。该次观察在 3,108 MiB physical footprint 触及 3 GiB 上限。
- 15,000→16,000 条跨越阶段 barrier 后，最大依赖集合从 5,080 增至 15,460 项。
- frozen `hbm_race.py` 的 `_join` 对历史 event 集合求并；`_completed` 加 prefix，但不移除已被 prefix 覆盖的 event；`event_knowledge` 保存每个事件的集合快照。RF 重用及阶段 barrier 会持续传播历史集合。增加 barrier 不会把这些集合压缩为 prefix。
- 当前宿主只有 16 GiB RAM。未盲目扩大内存/超时，也没有禁用检查、替换 checker、修改 frozen 文件或产生非标准分数。

这证明当前受阻于**完整程序验证的可扩展性**，尚不能判定真实多 SM 的性能收益。用户指定的两项成功条件均未完全满足；本阶段停在这个明确阻塞，而非把未验证实现报告为成功。

## 6. 下一步判断

继续 multi-SM 主线的价值仍高于优先改 D1 persistent：dense GEMM 占 P1 62.6%，另有 Attention/GELU 31.9% 可并行。当前最大的三项 baseline 性能阶段是 **Attention 21.3%、W1 20.9%、W2 20.7%**；RF/访存依赖和共享 NoC 是待验证的限制来源。

第一前提是获得允许完整 race validation 的执行条件，或由 evaluator 维护方解决依赖集合存储问题；本仓库保持 frozen 不动。之后直接重跑 cand-099，测四 SM 的真实 makespan、资源利用率和 barrier 等待，再决定 fusion；现在没有证据支持转向 D1，也没有必要继续 tile/hardware sweep。

本次没有连续三个完成的多 SM 候选，也没有启动结构参数搜索：完整并行 skeleton 尚未通过门槛，不能将内存失败当作性能回退样本。

## 7. 复现

```sh
# 默认提交仍为 cand-098 程序；新实现通过候选快照运行。
.venv/bin/python -m project.campaigns.p1-multism.verify_controls
# 每次新实验使用新的 candidate，避免覆盖既有证据。
.venv/bin/python -m project.campaigns.p1-multism.materialize
.venv/bin/python -m project.search.controller run <new-id> --seed 7 \
  --timeout-seconds 900 --wall-time-seconds 1000 --memory-mib 6144 --max-grades 1
```

Gemini 的归因研究原稿带有数值/单位错误，已标注，不能作为结果依据；另一个研究 worker 被中断。两者 traces 曾随实验提交；按用户后续要求，当前工作区只保留主对话的一份完整导出，额外导出仍可在 Git 历史查阅。最终结论来自主 agent 对 frozen 源码、原始 grade、完整程序 trace 和内存进度的直接核验。
