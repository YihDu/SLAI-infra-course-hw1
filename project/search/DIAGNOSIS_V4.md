# 搜索空间扩展与逐层剪枝（2026-09-28）

本轮从 cand-083（2245.039 分）继续搜索，先实现可生成的新布局，再用资源约束、评测成本和完整 grade 淘汰候选。冻结评测器没有修改。

## 瓶颈及已实施的改变

原 nkm 把全部 M×N 累加器放进 RF 的一个分区。P1 M=64、每分区 1024 words，因此 N 被实现限制为 ≤16。新 `nkm_lanes` 将每个 M tile 的累加器放入独立 RF 分区，保留 B 对全部 M tiles 的复用；分区 0/1 用作 A/B，分区 3 留给 B 预取，累加器使用 4 起的分区。跨阶段的临时值在 GEMM 进入前已经写回，因此可以复用这些分区。

对一个阶段，设实际行数 R、M tile 为 m、N tile 为 n、K chunk 为 k、vector_lanes 为 v、每分区容量 L=16384/v FP32 words。静态检查是：

- A：min(m,R)×k ≤ L；B：k×n ≤ L。
- 新布局的单个累加器：min(m,R)×n ≤ L。
- 累加器分区数量：4+ceil(R/m) ≤ v。
- 旧 nkm 仍需 R×n ≤ min(L,1024)，对应旧生成器实际限制。

每个阶段分别使用其有效参数，避免用被覆盖的全局 prompt 参数剪掉合法 decode 配置。搜索器枚举已实现的 mnk/nkm/nkm_lanes 和完整 tile 菜单；cand-083 硬件下 P1 prompt 的 840 个组合，经静态检查留下 218 个，其中 55 个采用新布局。再应用本机评测成本策略，留下 168 个可优先执行的组合。这是当前已实现程序族的空间，不代表任意程序或所有算法。

另一个可扩展方向是联合硬件/程序变换：减少闲置 SM 释放面积，再扩大 TC；或者减少 vector 分区数量以增大每个 RF 分区，配合 N=128/K=16。此类组合直接进入提案模板，不再要求每个中间点先提高分数。正确但暂时不提高分数的桥接候选保留为分支。

## 用实测决定剪枝

cand-086 的共享内存 panel 没有减少 RF 读写需求：P1 仍为 425.82/204.75 MB，却增加了 shared_read/shared_write 服务量。DMA 从 157.82 MB 降到 138.95 MB，P1 周期反而增加。服务统计允许重叠，不能把 busy cycles 相加当作关键路径。该实现失败并不意味着所有共享内存方案失败。

cand-087（m16,n32,k32）将 P1 DMA 降到 120.07 MB、RF 写降到 177.22 MB，P1 从 10,750,199 降到 9,484,805 周期。cand-088（m16,n64,k16）DMA 进一步下降，但 RF 读写上升，周期为 9,986,555。这为优先搜索 N32 附近提供了证据；N64 及联合硬件分支仍可扩展。

cand-089（n128,k8）在完整功能检查中长时间占用 CPU。源码 `codesign/challenge/micro.py` 的 `_mma` 在 k≥16 时使用 NumPy 向量运算，k<16 时改用逐元素 Python 循环。保存运行采样和 `reports/cost-prune.json` 后停止该进程，正确性标记为未验证。`evaluation_cost_reasons` 是评测成本策略，独立于 ISA 合法性检查；自动提案优先排除 P1 k_chunk<16，直接 controller 仍可运行此类实验。N128 没有整体封禁，可用 8 个 vector 分区提供 2048-word 容量，让 B=16×128 保持在快速检查路径。

## 搜索与证据

- `propose.stage_space` 枚举布局及参数，再做实际资源剪枝；提案器根据实际 recipe/layout 识别新结构。
- 六候选轮次的探索配额随 explore_ratio 变化，例如 0.54 对应 2 exploit / 1 expand / 3 novelty。
- 含新布局的候选固定当前编译器快照，并将编译器身份纳入指纹；历史候选输入不修改。
- `prepared=True` 的评测路径现在保留静态展开指标。
- 完整评测维持两个进程并发，每个 3 GiB / 600 秒限制，避免把 Gemini 提案并发直接等同于模拟器并发。
- Gemini 本轮只返回了继续查看源码的说明，没有提供可用结论；其 session trace 已保存，本轮判断来自主 agent 的源码和实测检查。

可复用命令：

```sh
.venv/bin/python -m project.search.structural_campaign --max-evals 8 --jobs 2
.venv/bin/python -m project.search.structural_campaign --refine --max-evals 3 --jobs 2
```

首次 campaign 的 K8 尝试形成了成本剪枝规则；当前脚本重跑时会跳过它并记录原因。每个已执行候选的输入、grade、失败证据和 iteration-log 是结果依据。汇总见 `project/campaigns/structural-*/summary.json` 与本轮结果报告。
