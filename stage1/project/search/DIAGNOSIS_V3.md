# 当前搜索空间诊断与扩展顺序（cand-083 后）

本文件只分析现有源码、冻结模型与已保存的公开评测；没有启动新的 `grade`。最佳已验证候选是 cand-083：P1 10,750,199 cycles，D1 744,864 cycles，score 2245.039，面积 18.756 mm²，两个案例的峰值窗口功率为 3.034/2.310 W。

## 核心判断

硬件菜单已经覆盖冻结模型的字段，但程序生成器仍只覆盖原始实现附近。`project/compiler.py` 的 `Builder` 始终发出一个位于 SM 0 的 workgroup，`shared_bytes=0`；P1 两个 batch 成员与 D1 八步都在该程序流中顺序执行。每层 LN、QKV、attention、FFN 等阶段在 HBM scratch 上分配中间张量。DSL 可以改变 GEMM 顺序、分块、预取、attention scores 的 HBM/RF 驻留、bias/GELU 融合，以及硬件字段；`workgroups` 算子仍直接报“not implemented”。所以扩展硬件枚举与增加 Gemini 提案数量都不会自动产生新的程序结构。

现有 `nkm` GEMM 把完整的 `M × n_tile` 累加器放进一个 RF lane。P1 prompt 的 `M=64`、每 lane 1024 FP32 words，因此 `n_tile>16` 被排除。当前编译器也没有 M panel 外层循环、SH weight panel、跨阶段驻留或多 workgroup 输出分片。扩展数值 tile 网格无法穿过这些实现约束。

## 量化证据

| 指标 | P1 | D1 |
| --- | ---: | ---: |
| 周期 | 10,750,199 | 744,864 |
| 展开后指令 | 264,815 | 19,866 |
| HBM 读 / 写 | 47.17 / 8.12 MB | 2.01 / 0.17 MB |
| RF 读 / 写需求 | 425.82 / 204.75 MB | 18.33 / 16.67 MB |
| DMA 搬运需求 | 157.82 MB | 15.50 MB |
| SM 0 RF 读 / 写 busy | 3.353 / 3.218 M cycles | 0.144 / 0.263 M cycles |
| SM 0 DMA busy | 2.466 M cycles | 0.244 M cycles |

这些 busy 时间可以重叠，不能相加，也不是关键路径占比。`issue_wait` 是等待的表象，不能指认具体资源。数据支持针对 RF 服务、DMA/NoC 搬运和 P1 的串行执行提出控制变量实验；单纯减少一处 HBM scratch 的收益上限需要按它实际删除的 RF/DMA/NoC 指令量估算。

上一轮 12 次真实评测主要只改变硬件；所有正常硬件对照的 P1/D1 指令数与 HBM 读写量不变。cand-085 将 reduction units 从 0 改到 1 后周期完全相同；冻结 `reduction_cycles()` 对这两个值都采用 `max(1, units)`。cand-081 将 cache 从 2 MiB 改到 4 MiB，两个案例的周期也不变。这说明合法菜单包含当前程序的等效或低价值维度；需要按程序资源使用状态过滤提案。

当前仅 SM 0 有 workgroup，8 SM 配置面积 18.756 mm²。静态面积公式给出：相同配置改为 4 SM 是 10.172 mm²；再把 TC array 改为 8×16 且保持 K parallelism=4 是 16.258 mm²，低于 24 mm²。8 SM 直接改为 8×16/K=4 则为 30.930 mm²。**这些只有面积计算，没有功能或周期评测。** 减少闲置 SM 可作为通向新硬件组合的中间节点；当前只从改善 score 且两个周期不退化的候选提升 head，可能丢掉这类桥接状态。

## 搜索策略的剩余问题

1. 六候选轮次固定分配 2 exploit、2 expand、2 novelty；`explore_ratio` 对 `count=6` 不起作用。名为 novelty 的模板仍由旧的 GEMM 顺序、scores 驻留、融合和分块组成，并不生成新算法。
2. 资源统计只保存案例级汇总。提案器没有根据 P1/D1 的 RF、DMA 或具体 GEMM 阶段归因来选择算子。`prepare()` 生成的 `static.json` 在 `prepared=True` 的评测路径没有合并进 `metrics.json`，使搜索状态更难使用静态计数。
3. 档案保留正确候选，但搜索缺少“为了释放面积，先走一个暂时没有分数收益的中间点”的组合路径。应允许附有明确后继和面积证明的两步提案，分别验证每一步。
4. 两路同时运行真实评测已接近这台机器的资源边界；Gemini 可并行提出、审查独立假设，冻结模拟器的并发仍由内存和实测吞吐决定。

## 应真正加入的程序空间

| 优先级 | 算子与首个原型 | 提案前的必要检查 | 主要不确定性 |
| --- | --- | --- | --- |
| 1 | **P1 W1 的二维 GEMM panel + SH weight 驻留**。从当前单 workgroup 的一个 GEMM 开始；试 `M_panel=16, N_panel=64`。W1 的 `K=256`，一个完整 weight panel 为 `256×64×4=64 KiB`，能装入现有 128 KiB SH。 | RF lane：A 的 `16×48=768` words、单个 B tile `48×16=768`、panel 累加器 `16×64=1024`；检查 SH 容量、尾块、同步、exactly-once 输出。当前 `N_tile=16` 使 A 对 W1 的 64 个 N tile 重复加载；64 列 panel 可将逻辑 A 加载轮数降到 16。 | SH→RF B 重载和 RF 写入可能抵消 A/NoC 节省；先比较静态字节与服务量，再用 grade 验证。 |
| 2 | **producer-consumer 驻留**。先试 D1 的 LN1→QKV 或 P1 一个 row panel 的 LN→GEMM，随后再扩展到 FFN。 | 按单个 live panel 检查 RF/SH 占用、覆盖 HBM 输出和消费者顺序；记录删除的 LD/ST、RF 写、DMA、NoC 需求。 | 当前 RF 写和 DMA 都忙；局部 HBM 流量减少不等于同等周期减少。 |
| 3 | **P1 两个 batch 分到两个 SM**，随后才考虑 GEMM 列分片、attention head 分片。 | 独立 scratch 和最终输出区、唯一事件、prompt 全部完成后的单个 commit；先对小程序测 race 校验的时间/物理内存随指令数增长，再尝试完整程序。 | 完整 P1 的旧尝试 cand-068 在 race 校验中被人工中断，无正确性结论。 |
| 较后 | blocked/online attention、更多细 tile | 目前两例 scores 已在 RF，D1 最长 136 个值；先证明新算法能减少关键服务量并满足 FP32 容差。 | 额外归约和同步可能比省下的流量贵。 |

`cand-068` 的 P1 只有约 1,917 行原始汇编，却在展开到大量指令后的双 workgroup race 校验中达到采样报告所示的约 48.2 G 物理占用，运行约 11 分钟后中断。冻结 race checker 对单 workgroup 有快速返回路径；多 workgroup 会构建整图并维护/合并事件知识集合。采样栈的大量时间位于 Python 的 `set_update`；集合增长是合理原因推断，但其精确复杂度需要小规模曲线测定。不要把这次资源中止记成 ISA 不支持并发或功能错误。冻结校验器不得修改。

## 实施与验收顺序

1. **低成本硬件桥接**：在 cand-083 的程序上独立测试 `sm_count=4`，然后从该分支测试 `tc_array=8×16`。两次真实 grade 分开记录；只有第二次通过两例正确性、面积/功率并改善 score 才提升最终配置。
2. **先做第一个真实结构算子**：为 W1 panel 建立可再生的 recipe 字段、源码实现和静态资源证明。局部小程序先检查 SH/RF 寻址与同步；完整候选用单次 grade 验证。记录预测与实测的指令数、逻辑传输量、RF 服务量和周期差，不预设它一定加速。
3. **再扩展融合和多 SM**：针对仍占主要服务量的阶段逐个加入算子；多 workgroup 原型先在 3 GiB / 600 s 的条件下测 race 校验曲线。若完整 P1 不可验证，则保留单 workgroup 的结构改进路径。
4. **改提案器**：按“结构模板→合法 tile/hardware 邻域”两层搜索；只开放已发出不同汇编且通过小程序验证的算子。让预算分配随实际成功率和每次 grade 的成本调整，过滤无效果资源，把面积桥接状态保存在可扩展档案，而非只看单步 score。

这样扩大的是可生成、可验证、会改变资源需求的程序族。下一个目标是形成至少一个经过 grade 的新内核，而非提高同一内核上的随机提案数量。
