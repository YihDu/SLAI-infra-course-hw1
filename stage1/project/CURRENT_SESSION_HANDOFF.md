# 当前进度与接手说明：cand-143

更新日期：2026-09-29。本文是新 session 的当前状态入口。更详细的最终分析见
`project/campaigns/current-design-limit/REPORT.md`，完整实验历史见
`project/iteration-log.md`。旧交接中的 cand-106、cand-114、cand-118 等“当前最优”
描述均已失效。

## 1. 当前可提交状态

- Git 分支：`dev-yh`
- registry head / best：`cand-143`
- 完整 frozen grade：正确且 eligible
- 未修改 frozen evaluator，未手改最终 asm，未上传官方 server
- Python：`.venv/bin/python`（3.12.13 / NumPy 2.5.3）
- 当前没有 evaluator 或搜索进程在后台运行

| 指标 | cand-143 | 相对 frozen baseline |
|---|---:|---:|
| Score | **7013.610245** | 7.014x baseline score |
| P1 cycles | **4,897,135** | 6.33x speedup |
| D1 cycles | **167,539** | 7.77x speedup |
| Area | **22.859813 mm²** | ≤ 24 mm² |
| P1 peak power | **4.018955 W** | ≤ 20 W |
| D1 peak power | **7.494538 W** | ≤ 20 W |
| Grade peak memory | **4069.86 MiB** | 本机完成 |

权威结果：

- `project/experiments/cand-143/reports/grade.json`
- `project/experiments/cand-143/reports/p1-profile.json`
- `project/experiments/cand-143/reports/d1-profile.json`

当前硬件为 4 SM、2 个 8x16 TC/SM、K parallel 4、16 vector lanes、16 SFU
lanes、2 reduction units、RF 8R4W、128 KiB SH/SM、2 DMA engines、8 HBM
channels、16 MiB cache、multicast enabled。完整配置以 `hardware.json` 为准。

## 2. 当前程序如何工作

### P1

P1 仍是单个长驻 WG 在 SM0 上执行。现有收益主要来自单 SM 内的数据驻留、融合和
issue pipeline：

- Prompt Attention 使用分块实现，避免完整 scores/probs 的 HBM materialization。
- Prompt MLP 将 W1/GELU/W2 组织成 RF 驻留流水；W2 accumulator 直接完成 bias、
  residual epilogue，移除 `ff2` HBM 往返。
- residual1 直接流入 LN2；LayerNorm gamma/beta 缓存在 RF，LN 输入复用现有 64 KiB SH。
- QKV、WO 和 MLP panel 使用下一列/下一 neuron 的 seed load lookahead。
- 第一生成 token 的 GEMM 使用已验证的 N64/K16 tile。
- P1 的 expanded instructions 已从 cand-124 的 89,579 降到后续融合前的 81,947；
  cand-143 profile 有 78,032 个 trace events。

### D1

D1 使用四个 persistent WG，占用四个 SM，跨 8 个 decode step 保持固定 owner：

- QKV、WO、W1、W2 按输出列/神经元分片。
- GELU 分成四个 owner，并复用相邻投影已有 barrier。
- projection 双缓冲 B panel，同时把完整单 token A vector 保持在 RF，跨 N panel 复用。
- Attention 对 history K/V panel 做双缓冲。
- 当前 D1 有 12,816 个 trace events；相对 cand-124，D1 从 207,673 降到 167,539 cycles。

## 3. cand-124 到 cand-143 的有效实验

本阶段从 cand-124 的 score 6189.6015 推进到 cand-143 的 7013.6102，综合增益
13.31%。每一项均经过完整 frozen grade，除明确标记为 preflight 的候选。

| Candidate | 改动 | 结果/结论 |
|---|---|---|
| 126 | P1 first-token GEMM N64/K16 | kept，score 6210.63；N128/K8、N256/K4 更慢 |
| 130 | P1 W2 RF-resident residual epilogue | kept，score 6229.40；cand-129 的 stride-zero 方案不正确 |
| 131 | LN affine RF cache；residual1→LN2 fusion | kept，score 6257.60 |
| 132 | LN2→W1 activation 放入已有 64 KiB SH | kept，score 6259.02，收益很小 |
| 133 | multicast | kept，score 6264.42 |
| 135 | 四路 D1 GELU | kept，D1 195,174，score 6456.32 |
| 136 | D1 projection operand double buffer | kept，D1 171,256，score 6892.44 |
| 137 | D1 history K/V pipeline | kept，D1 169,003，score 6938.23 |
| 139 | D1 A vector 跨 N panel RF resident | kept，D1 167,539，score 6969.54 |
| 142 | P1 MLP next-neuron seed lookahead | kept，P1 4,935,427，score 6986.35 |
| 143 | P1 QKV/WO next-column seed lookahead | kept，P1 4,897,135，score 7013.61 |

## 4. 当前瓶颈

### P1 阶段归因

| 阶段 | Cycles | 占 P1 |
|---|---:|---:|
| RF-persistent prompt MLP | 2,486,327 | 50.77% |
| QKV | 758,838 | 15.50% |
| Attention | 506,946 | 10.35% |
| fused output epilogue | 325,105 | 6.64% |
| first-token W1 | 274,050 | 5.60% |
| first-token W2 | 272,490 | 5.56% |
| WO | 253,842 | 5.18% |

P1 主要受 RF 服务限制：profile 中 RF read/write busy cycles 约 2.64M/2.36M，
硬件 RF ports 已为合法上限 8R4W。P1 仍只使用 SM0，因此多 SM 在架构上仍有潜力，
但 frozen checker 的验证表示目前阻塞了该路线。

### D1 阶段归因

| 阶段 | 占 D1 |
|---|---:|
| W2 | 23.56% |
| W1 | 23.51% |
| QKV | 18.71% |
| Attention | 17.54% |
| WO | 6.37% |
| GELU | 2.97% |

D1 已经真实使用四个 SM。剩余时间分散在多个 dense projection 和 Attention；现有
panel pipeline 的最后几步收益已经低于 1%。

## 5. P1 multi-SM 的验证边界

不要把 multi-SM preflight 失败解释为性能失败。目前没有可引用的 multi-SM P1
cycle 数。

cand-141 对最小 two-WG、batch-parallel end-to-end P1 skeleton 做了重新预检：

- 78,032 events
- 156,360 dependency edges
- 78,136 global commit incidences
- retained event-set logical lower bound：13,773,311,648 bytes
- 本机 validation budget：6,144 MiB
- 预测差距约 2.24x，因此在 grade 前正确剪枝

报告位于 `project/experiments/cand-141/reports/validation-cost.json`。根因是 frozen
race checker 为 multi-WG 程序保存展开后事件的依赖集合；静态 FOR 也会先展开，
barrier 合并依赖但不会压缩已经保留的历史集合。增加 coarse barrier 或只减少源码
行数不能解决问题，必须减少实际事件和跨 WG 依赖传播。

## 6. 已测量且不应重复的局部搜索

- P1 first-token N128/K8、N256/K4 均不如 N64/K16。
- 128 KiB SH activation 路径会损失 pipeline overlap。
- D1 W1 weight shard 放入 private SH 正确但更慢；W2 的形状和访问方式相同，不应
  重复同一实验。
- layer-major batch ordering 基本中性且略慢。
- TC3/cache4 和 TC3/cache2 交换均比当前 TC2/cache16 慢；TC4/8x16 超面积，
  TC4/8x8 对齐 tile 的吞吐等价。
- 增加 SH banks/ports 不改变 cycles；更大 NoC、更多 DMA engine 也已无收益。
- 当前 design family 内继续做 tile、SH 容量、带宽字段或同类 panel prefetch sweep，
  预期只会得到微小或负收益。

## 7. 搜索和实验做法

候选目录是不可变证据。不要覆盖已有 `cand-NNN`，不要直接修改其中 snapshot。
所有 generator 修改发生在 `project/compiler.py` 或新的生成器源文件中；asm 通过编译器
生成。每次评价后必须保留 reports 并追加 `project/iteration-log.md`。

搜索分两层：

- Outer structural search：parallel skeleton、fusion partition、residency plan、
  attention algorithm、persistent scheduling、RF layout/event compression。
- Inner parameter search：tile M/N/K、shard/WG 数、placement 和 hardware category。

同一 outer structure 连续三个 eligible candidate 相对 parent 增益低于 10% 时，应停止
inner sweep，回到结构搜索。MILP 只负责在受约束的离散空间提出候选，frozen grade 才是
真实性能和正确性的唯一依据。当前 registry 已实现该 hierarchy policy 和 revision CAS；
候选创建、registry 保存、checkout/keep/revert 必须串行执行。

标准流程：

```sh
.venv/bin/python -m project.search.controller new cand-NNN --parent cand-143 \
  --hypothesis "..." --summary "..." --from-worktree
.venv/bin/python -m project.search.controller run cand-NNN --seed 7
.venv/bin/python -m project.search.controller keep cand-NNN --note "..."
# 或：
.venv/bin/python -m project.search.controller revert cand-NNN --note "..."
```

对 multi-WG P1，必须先运行 validation-cost preflight。预测超过预算时保留报告并剪枝，
不要启动必然耗尽本机内存的 grade。

## 8. 下一 session 应从哪里继续

当前实现族已经达到局部设计边界。下一项工作应选择一个新的 outer structure：

1. **新的 P1 RF layout/tile family。** 当前 vector16 和每个 RF partition 的布局限制了
   更大的 M/N/K tile。可研究 vector8 释放面积/端口压力后重新设计 tile 和 SFU/VEC
   平衡。这需要成组改写 generator，不能只改一个 tile 参数。
2. **真正压缩 P1 multi-SM 事件图。** 目标是把实际 expanded event 和跨 WG dependency
   propagation 至少降低到 cand-141 下界的 1/2.24 以下，再做 frozen grade。只调 barrier、
   WG 名称或 FOR 写法无效。
3. **使用能容纳现有 multi-SM checker 状态的验证环境。** 若环境改变，先用 cand-141
   同一输入做受控内存校准；成功前不要声称 multi-SM speedup。

优先级建议：先做一页纸的 RF layout 候选容量和服务周期模型，同时为事件压缩提出至少
两个具体 generator skeleton。只有模型显示可能带来大于约 10% 的完整模型收益时才生成
新候选。不要重新启动当前已饱和的参数 sweep。

## 9. 接手和复现命令

先确认仓库状态，默认不需要重新跑六分钟级 full grade：

```sh
git status --short
git log -3 --oneline
.venv/bin/python -m project.search.controller status
.venv/bin/python -m project.search.controller checkout cand-143
.venv/bin/python -m project.compiler
```

需要复核 profile 或测试时：

```sh
PYTHONPATH=. .venv/bin/python project/campaigns/p1-multism/profile.py cand-143
PYTHONPATH=. .venv/bin/python project/campaigns/p1-multism/profile_d1.py cand-143
.venv/bin/python -m unittest discover -s project/search/tests -q
```

只有发现报告损坏或准备正式打包时，才重复完整 grade：

```sh
.venv/bin/python challenge.py grade \
  --hardware hardware.json \
  --program-p1 programs/M1_P1.asm \
  --program-d1 programs/M2_D1.asm \
  --baseline baseline_manifest.json \
  --seed 7 \
  --report project/grade-handoff-verify.json
```

继续实验时以 `cand-143` 为 parent 创建新 candidate。不要修改 frozen evaluator，
不要自动向官方 server 提交。按用户最新要求，`agent-trace/` 只保存当前主对话的一份完整
导出，不收集 Gemini worker 的独立 trace。
