# 搜索效率诊断（2026-09-28）

审查范围：当前 compiler、DSL、propose、engine、controller、evaluator、archives，
以及 cand-000 至 cand-072 的历史输入、生成汇编与报告。cand-073 是审查前已启动的
缓存实验，审查期间完成：P1=12,347,819、D1=776,636、分数=2051.477、面积=7.853 mm²，
两项功能检查与资格门槛均通过；现已同步为最佳候选。以下瓶颈表固定使用 cand-072，
便于复现对比。本次诊断没有启动新候选或修改冻结评测器。

## 结论

效率问题同时来自可复现的实现缺陷、重复评测、缺少因果反馈，以及过窄的程序空间。
继续扩大轮数或增加 Gemini 数量不能自动解决这些问题。此前将 issue_wait 当成具体
瓶颈、将 stage tile 扩展描述为充分的结构搜索，以及一次资源中止后暂停整个多 SM
方向，都缺少足够证据，应调整。

## 已确认的问题

| 优先级 | 问题 | 代码与证据 | 后果 |
| --- | --- | --- | --- |
| P0 | 有价值的硬件提案无法进入搜索 | dsl.HW_MENU 缺 tc_array、tc_k_parallel；propose.hw_toggles 却包含它们。apply_ops 实际返回 unknown/unsupported field，emit 捕获后直接丢弃。SFU、共享内存子字段等也不在 DSL 菜单中。 | 搜索没有覆盖声称支持的硬件空间，TC 服务时间相关变量长期不可达。 |
| P0 | novelty 名额被截掉 | propose 的 exploit 循环可额外插入硬件提案，最后 return proposals[:count]；第 11、12 轮均实际得到 3 exploit + 3 expand、0 novelty。 | 探索比例与实际行为不一致；增加 explore_ratio 不保证新结构得到评测。 |
| P0 | 手动路径重复昂贵工作 | runner.evaluate_submission 对 check、estimate、grade 都调用相同时序路径；evaluator.evaluate 顺序执行三者。 | 每个成功候选进行 3 遍时序、2 遍功能检查。自动路径已有单次 grade 实现，但新结构实验又回到了重复路径。 |
| P1 | “静态筛选”没有筛掉候选 | evaluator.static_screen 仅统计未展开文本行数和 opcode 数；调用者只把返回值存入 metrics。 | 无指令展开量、访存量或验证成本门槛。函数名称和注释夸大实际能力。 |
| P1 | estimate 不是廉价预测器 | 它运行真实调度器；可选筛选阈值仅为 incumbent score 的 0.6；近期未启用。 | 直接加 estimate 通常只是增加一轮昂贵计算，不构成有效的多层评测。 |
| P1 | 缺少实际执行产物去重 | 指纹仅包含 recipe 与 hardware，未含 compiler；生成后也不比较硬件和两份汇编哈希。 | 历史有 9 个额外候选与此前候选的硬件和两份汇编完全相同，且均有 grade.json。反过来，换 compiler 而 recipe/hardware 不变可能被错误去重。 |
| P1 | 新 stage 空间仍基本靠手工 | exploit 只读取 recipe['gemm']，没有对 gemm_stages 的有效配置生成邻域；新增两个 stage 模板只进入 novelty。 | cand-071/072 的阶段优化并未形成完整的自动局部搜索机制。 |
| P1 | 没有用测量结果指导提案 | _bottleneck 只取 P1 busy_cycles 最大项；propose 不读取 bottleneck。_family_fail_streak 未被调用，need_new_operator 只设标记。 | 所谓反馈主要是分数排序和随机扰动，尚无基于瓶颈或失败原因的策略更新。 |
| P1 | 状态与结果分成两条不一致路径 | cmd_run 写 baseline_cycles，不更新 archives/search；keep 仅更新 head。cmd_run 的 evaluating 不同步注册表，异常无统一收尾。 | 曾需手工同步 cand-071/072；执行中的 cand-073 在 registry 仍显示 created。中断可留下半完成候选。 |
| P1 | 无运行成本和资源预算 | subprocess.run 无 timeout、进度、峰值内存记录；异常笼统标 generate_error；失败在自动路径不扣 grade budget。 | cand-068 检查约 11 分钟、采样 footprint 约 48 GiB 后靠人工中止。预算不能约束总时间和失败成本。 |
| P2 | 档案标签没有准确描述整个程序 | struct_sig 取最后一组 mutation；dominated 按比分低于父候选标记，但 archives 仍能收纳它。 | 同一最终结构被分到多个单字段标签，名称与实际 Pareto 语义不一致。不能把该状态当成严格支配剪枝。 |
| P2 | 共用可写工作区 | 每个候选覆盖根目录 compiler、hardware、recipe、programs；registry 无并发协调。 | 当前评测不能直接开几十路并行。应先隔离候选运行目录、集中写回状态，再按内存预算控制并发。 |

## 重复执行证据

以下分组按规范化 hardware JSON 加两份汇编原始字节的 SHA-256 核对，组内完全一致：

- cand-007、cand-013
- cand-014、cand-038
- cand-019、cand-034、cand-035、cand-040、cand-041、cand-042
- cand-020、cand-037
- cand-036、cand-043

共 14 个候选占 5 个执行点，多出 9 次 grade。部分后来标记为 harness_noop，
表示历史问题已被识别，不代表目前还会以完全相同方式重现；生成产物哈希门槛仍缺失。

## 程序性能的实际证据

以 cand-072 为固定参考：分数 2004.671，面积 6.894 mm²，P1 峰值功率约 2.630 W。

| 指标 | P1 | D1 |
| --- | ---: | ---: |
| 总周期 | 12,522,857 | 801,958 |
| TC busy cycles | 5,625,504 | 243,264 |
| RF read busy cycles | 4,533,128 | 144,288 |
| RF write busy cycles | 3,218,344 | 263,232 |
| DMA busy cycles | 2,465,984 | 244,416 |
| HBM read bytes | 149,702,656 | 15,474,688 |
| HBM write bytes | 8,120,320 | 167,936 |

这些 busy 时间会重叠，不能相加或直接作为关键路径占比。issue_wait 是下一条指令
能发射之前累计的等待时间，不能区分 TC、RF、DMA 或数据依赖。当前数据支持优先
检查 P1 的 TC/RF 服务代价和 D1 的 RF/DMA/TC 组合，但具体因果必须靠控制变量实验
或更细的调度分析确认。面积和功率还有余量，不代表增加任意资源都会加速。

最近 cand-065→cand-072 分数仅增加约 1.97%，P1 周期完全不变。P1/D1 使用同一
硬件，但程序参数需要按阶段选择；继续只改 D1 的 tile，难以产生新的大幅提升。

FFN 融合也必须先估算上限。仅删除 w1 输出的 HBM 写入和随后读取，P1 可去掉
3,194,880 bytes，约为当前总 HBM 读写的 2.02%；D1 为 65,536 bytes，约 0.42%。
这是该中间张量的流量减少量，不是总性能加速预测；还要计算指令和其他访存变化。
不应在缺少估算时将小 epilogue 融合宣传为大幅提升的主路径。

多 workgroup 的 cand-068 未完成冻结 HBM race 检查，没有功能错误或分数结论。
应该先做小规模并发原型，测量验证时间与内存随指令数增长的趋势，再决定完整程序
如何分块。一次资源中止不能证明 ISA 不支持多 SM 或该方向没有价值。

## 修复顺序与验收

1. 先恢复搜索器声称具备的能力：补全合法硬件菜单及依赖关系；确保每轮配额精确且
   novelty 不会被截断；对拒绝的提案记录原因。验收：关键硬件字段可达、固定轮数的
   各类提案数符合配置，不能靠静默丢弃隐藏错误。
2. 统一评测和状态写回：自动探索使用既有单次 grade 路径，明确保留需要独立 check
   的验证模式；已有 check 时不再无条件重复 estimate。功能结果必须为 true 且资格
   通过才可入选。统一保存 archives、head、search 和终态，异常保存 stderr。
3. 增加生成后精确去重与可复用结果键：hardware、两份 ASM、冻结评测器版本、模式、
   seeds。另保留 compiler/recipe provenance；不能因“只是硬件变化”而略过功能检查，
   TC K 分组可改变浮点计算。
4. 设置 wall-time/内存上限、阶段计时、失败计费和运行目录隔离。先有限并发，按实测
   内存成本调度。验收：无无限等待、无 stale evaluating、无跨候选工作区覆盖。
5. 做有预算的硬件灵敏度实验，优先 TC array/K 并行、TC 数量、RF 和 DMA/cache。
   以两项周期与成本差异决定下一步，避免在未诊断前直接投入大生成器改写。
6. 再开发能改变主要服务量的程序结构：更好的 A/B 驻留、分块复用、经过小规模验证的
   多 SM 任务划分。先写预计减少的 bytes/指令/服务量与资源约束，评测后核对预测。

搜索效果同时报告“每个有效独立候选的成本、重复率、各阶段耗时、失败资源成本、
单位时间的新最佳提升”，不能只用轮数或候选总数衡量进展。
