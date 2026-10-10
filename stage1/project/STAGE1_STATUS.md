# Stage 1 成果与 Stage 2 可用证据

更新：2026-10-10。结果读取已有报告，本次没有重新评测。

## 最终验证结果

当前 Stage 1 目录方案为 **cand-155（privsh256_hbm4）**。

| 指标 | 已保存结果 |
| --- | ---: |
| M1_P1 cycles | 493984 |
| M2_D1 cycles | 48838 |
| 本地实验分数 | 40901.124578635245 |
| 面积 mm² | 23.372983198720004 |
| P1 / D1 峰值功率 W | 11.122408978516123 / 10.793015843962335 |
| 正确性与资格检查 | 两项均通过 |

证据：[迭代日志](iteration-log.md)、[cand-155 指标](experiments/cand-155/metrics.json)、
[提交审计](submission-audit-20261007.json)。本地通过不等于服务器正式成绩。
已有最终包为Stage 1 目录 `submission-v073-20261007-final-100mib.zip`。

Git 关键节点：`03bd2c3` 提升 cand-155；`5f480c1` 整理历史和提交审计；
`1490829` 增加解压大小限制；`20440ee`、`0746071` 清理冗余报告并保留审计。

## 已经做过什么

历程包括 B-stationary GEMM、recipe/结构化搜索、RF/SH 驻留、多 SM 划分、
DMA 预取与流水、MILP 调度和仿真反馈修订。旧 briefing 中的 baseline、
cand-001 和“SM0 单工作组”等描述是历史起点，不是最终程序的状态。

| 可复用材料 | Stage 2 用途与边界 |
| --- | --- |
| [搜索实现](search/) 与 [候选记录](experiments/) | 候选溯源、预算、失败回退；需适配新评测器 |
| [D1 MILP 实验](campaigns/milp-d1/README.md) | 变量、约束、代理成本和实测差异的真实案例 |
| [自动 MILP 反馈报告](campaigns/auto-milp-20260929-005803/REPORT.md) | 检查已有反馈链，提取适用条件 |
| [迭代日志](iteration-log.md) | 形成带候选 ID 和路径的知识条目 |
| [原有 agent briefing](STAGE1_AGENT_BRIEFING.md) | 保留 Stage 1 规范与历史上下文；文内路径现在相对 stage1 目录 |

Stage 2 改变了模型、规模和 ISA。保留强方案及依据，迁移算法思想并生成合法
JSONL 后，在同一新版评测器与可比预算下建立对照。不得把旧分数、旧时序结论
或代理模型最优性直接当作 Stage 2 实测结果或全局最优证明。
历史日志保持原样，新状态集中记录于本文件。
