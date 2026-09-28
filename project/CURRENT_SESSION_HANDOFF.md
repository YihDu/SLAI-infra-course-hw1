# 新 session 接手：P1 multi-SM 与 frozen checker 内存阻塞

更新：2026-09-28。用户最后要求：**把进度写入文档并提交 Git，之后由新 session 接手。当前停止实验。**

最新记录规则：**只保留当前主对话的一份完整导出；不再收集各 worker/session。** 此要求覆盖此前保存所有 agent traces 的指示。

## 1. 先读这些文件

1. 本文：任务、结论、未完成事项、接手顺序。
2. `project/campaigns/validation-efficient/REPORT.md`：本轮详细证据。
3. `project/campaigns/p1-multism/REPORT.md`：上一轮完整 multi-SM skeleton、cand-099/100 与完整 baseline 阶段归因。
4. `AGENTS.md`、`project/PROBLEM.md`：项目约束。frozen evaluator 不得修改，asm 只能由 generator 生成。

仓库：`/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter`，分支 `dev-yh`。
Python 使用 `.venv/bin/python`（3.12.13 / NumPy 2.5.3）。

## 2. 当前最优与默认工作区

- registry head 是 **cand-095**；**cand-098** 是其完整 grade 复验，作为本阶段比较基准。
- P1 = **8,522,411**，D1 = **718,132**，score = **2567.954051821853**。
- 面积 16.25838297088 mm²；P1/D1 峰值滚动功耗 2.9710608543 / 2.3857896063 W。
- 硬件：4 SM，TC 2×8x16，K parallel 4，vector 16，SFU 8，RF 8R4W，SH 128 KiB/SM，DMA depth2/engines2，cache2MiB。
- 默认 `project/recipe.json` 与 root `programs/*.asm` 保持 cand-098；新 compiler 含 opt-in 多 SM 路径，默认生成两份 asm **与 cand-098 字节一致**。
- 没有新的完整 multi-SM correctness-pass 或 timing/score；不要把未验证候选晋升或打包为最优。

## 3. 用户当前目标与约束

用户要求：

- 不把 cand-099 checker 内存失败解释成 multi-SM 性能失败，不再微调它的 tile/barrier。
- 比较 cand-098/099/100 的展开指令、事件、WG、访存和同步边；建立 validation cost model，grade 前剪枝。
- 从 cand-098 依次试 only W1、only W2、W1+W2、only QKV、QKV+W1+W2，Attention/LN/residual 保持成熟路径，每个完整 grade。
- 首个通过者成为新 structural parent；随后分析跨算子 residency、D1 persistent、Attention 算法重写。
- outer search = skeleton/fusion/residency/attention/persistent；inner = tiles/shards/WG/placement/hardware。连续低收益应停止 inner。
- 成功标准包括完整 multi-SM grade、内存受控、真实 end-to-end speedup、下一项 >1.5× 潜力结构。
- 不上传官方 server，不修改 frozen checker。

**这些成功标准尚未实现。** 五个完整增量 grade 均在本地内存上限中断，没有取得 correctness 或 timing。

用户后来问“是否本地 Mac 内存限制”“如何推荐”。已解释：本机 16 GiB，实际终止由我们 runtime 设置的 3/6 GiB 上限触发；不是证明任何机器上都不能跑。建议优先使用现有大内存服务器校准/验证，再决定算法改写。

### 尚未得到用户答复的选择

- 是否有现成大内存服务器，容量、SSH host 别名、远程目录是什么？**未提供连接信息，未授权/执行任何具体远程操作。**
- 是否允许提前放宽“Attention/LN 保持成熟路径”，转向算法级实际指令压缩？**未获明确选择。**
- 256–512 GiB 只是先前提出的资源级别，不保证足够，也不建议直接购买。最后的仅 W1 前缀外推约 478 GiB **逻辑事件集合**，并非实测全程峰值，远程也必须先校准。

用户最终要求停下并交接，故不要在接手时自动继续五种已经失败的本地 grade。

## 4. 本轮完成了什么

### 实现

- `project/compiler.py` 新增 `IncrementalP1Builder`：四个 persistent WG，指定 **prompt** GEMM 输出列分片，每个并行 GEMM 前后两个 coarse barrier；第一生成 token 和其他阶段继续成熟单主 WG 路径。未改 D1/hardware。
- `project/search/validation_cost.py`：只读展开图统计、RF/SH 长依赖链的事件集合逻辑存储下界。不是 race validator，不替代 frozen grade。
- `project/search/evaluator.py`：多 WG 程序在 generate/static 后运行成本模型，默认预测过预算则 `pruned / validation_cost_pruned`。
- 本轮五候选的 proposal 明确带 `validation_cost_calibration=true`，允许一次受控完整 grade 校准。这不是模型通过，也不是通用运行设置。
- `project/search/hierarchy.py` + propose/engine：outer identity 不随 tile/shards/placement/hardware 改变；连续三个 eligible 候选对父候选 score 增益 <10% 时停止该 outer 的 inner proposals。内存失败、不正确或未测量不作为低性能样本。
- registry 已启用 hierarchy，`start_after_id=100`，objective=score，minimum_gain=.10，patience=3。
- 这些是身份/预算策略；尚未实现未来 persistent/online-attention 的 generator，不能声称它们已经可自动搜索。

### 实验

| Candidate | Prompt 并行部分 | 展开指令 | events | barriers | grade 峰值 MiB | 状态 |
|---|---|---:|---:|---:|---:|---|
| 101 | W1 | 161,537 | 161,516 | 12 | 3079.02 | memory_limit |
| 102 | W2 | 161,537 | 161,516 | 12 | 3072.35 | memory_limit |
| 103 | W1+W2 | 161,549 | 161,516 | 24 | 3085.69 | memory_limit |
| 104 | QKV | 161,537 | 161,516 | 12 | 3078.22 | memory_limit |
| 105 | QKV+W1+W2 | 161,561 | 161,516 | 36 | 3080.00 | memory_limit |

全部 4 WG、0 WAIT，D1 相同。每个调用了完整 `grade --seed 7`，均未完成；**没有 grade.json，没有分数，没有各阶段 timing**。日志里通用模板提及 grade.json 是预期输出路径，实际失败证据是 metrics、grade heartbeat/stdout/stderr。

候选 snapshots/reports/proposals 均在 `project/experiments/cand-101..105`；禁止覆盖。下一编号由 `state.next_id` 获取，不写死。

### 验证

- 51 个 search tests 通过，含新成本下界与真实小程序 frozen checker 集合大小对照，以及 outer/inner budget 测试。
- `python -m project.campaigns.p1-multism.verify_controls` 通过；默认两份 asm 与 cand-098 字节一致。
- 完整自动 preflight 拒绝且不启动 grade 的集成路径尚无专门测试；calibration override 的五次实际运行已覆盖准备/模型/grade 链。
- prepared=True 的 evaluate 会重新创建 metrics，因此 calibration/preflight 的一些字段不在最终 metrics 中；独立 `reports/validation-cost.json` 和 proposal 完整保留。后续可改善报告整合，但不应因此扩大 infra 工作。

## 5. 最关键的技术结论

1. **1 WG→多 WG 会触发全程序 checker。** cand-098/100 只有一个 WG，因此 checker early return；多 WG 后会为成熟单 WG 阶段也保存每个事件的依赖快照。
2. **没有“免费粗化 event/loop”的 ISA 写法。** LD/ST/MMA/VEC/REDUCE/SFU 都要求唯一 event；FOR 进入 checker 前展开。当前 cand-099 已是 4 个长驻 WG、84 个 coarse barrier、0 WAIT。
3. **RF 重用带来历史集合复制；barrier 只合并，不压缩。** `_join` 保留 event 并集，`_completed` 加 prefix 但不删除已覆盖事件，`event_knowledge` 持有全部快照。同名 WG 重开继承历史；换名字并正确同步也需传递依赖。
4. **only W1 在到达 W1 前已经膨胀。** 第一并行入口 barrier 的 PC=17,412；观察到 PC=9,999，仍在第一 prompt 的 Attention，9995 个事件集合已占 2,048,903,752 B；2.5 GiB 有界观察随后终止。
5. cand-099 事件只比 baseline 多 2.9%，并非 WG 把计算笛卡尔积复制四次。增量五候选事件数完全与 baseline 相同。

## 6. 下次接手的推荐步骤

1. 先确认用户希望用大内存服务器，还是允许提前算法压缩；不要把上一条建议当成已有远程访问授权。
2. 若有服务器：核对 pinned interpreter/NumPy、源码冻结 hash、RAM/可用内存；先对完整程序做有界 prefix 校准，明确逻辑存储、RSS、耗时，再尝试完整 grade。保留所有报告，不上传官方 server。
3. 优先考虑 cand-099 的完整 skeleton（其模型下界约29.7 GiB，低于 only-W1 的约108 GiB）；用户之前的“不微调 cand-099”不妨碍在不同资源条件下复验相同输入，但需要明确新 attempt，不覆盖旧报告。
4. 若选择本地算法压缩：目标必须是减少**实际展开指令/依赖传播**，而不是静态源码行数或 barrier 数。它会超出本轮“成熟 Attention/LN 不变”条件，需要用户确认方向。
5. 第一个真实完整并行 grade 通过后，再收集 trace、per-SM utilization、sync cost、HBM traffic，建立新 parent。未通过前不报并行 speedup。
6. 源码上最大的候选机会是 row-block persistent MLP/residency，详见本轮报告。独立 Attention 加速本身无法带来1.5× P1；D1已经单WG跨8个commit，机会在 weight驻留/NoC/RF，而不是重新把WG叫作persistent。

## 7. 复现入口

```sh
.venv/bin/python -m unittest discover -s project/search/tests -v
.venv/bin/python -m project.campaigns.p1-multism.verify_controls
.venv/bin/python -m project.search.validation_cost \
  project/experiments/cand-101/programs/M1_P1.asm \
  --report project/<new-report-name>.json --memory-mib 3072
```

`materialize.py` 会创建新的候选，**不要把它当状态查询重复执行**。已保存成本报告无需重跑。

本轮 Gemini worker：`01a0e7ca-3de6-76e0-a4a8-40458a990b77`，只读审计；输出并不简洁，仅作为辅助，最终结论由主 agent 核验源码和实测。用户最新澄清：只需当前主对话的一份完整导出，不再逐个保存 worker/session。`agent-trace/` 已按该要求清理；后续交接依靠本文和实验报告。
