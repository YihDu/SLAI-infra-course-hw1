# 多 SM 审查结论（主 agent 复核）

Gemini 提供了独立列分片 GEMM 原型。主 agent 已用固定 `.venv/bin/python` 复跑，原始 worker 报告、命令与失败尝试保存于 `project/campaigns/gemini-multism-review/`。

- M8/K48/N32 的双 WG 小内核通过 race 和功能检查，最大绝对误差 5.722046e-06，235 cycles。
- M64/K256/N256 的孤立内核，1/2/4 SM timing 分别为 81,325 / 43,460 / 24,809 cycles。4 SM 相对同一单 SM 内核加速 3.28 倍；这不是完整 Transformer 的提速或 grade 分数。
- 扩到 8 SM 的 worker 配置面积约 30.930 mm²，超出 24 mm²，不能作为合法候选。原报告只强调功耗合格的说法不完整。
- 独立列分片输出没有重叠，允许共享只读输入 A；接入后续消费者时仍需显式同步。不能把“孤立列分片无竞争”泛化到整张图。
- 多活跃 WG 的 WAIT/BARRIER 应明确 wg/wgs。STEP.COMMIT 由解析器分发到所有活跃 tape，并处理已结束 tape 的依赖；原报告称“有 WG 提前结束就必然死锁”不正确，见 pipeline_events.py 的 group_program。
- SH 为 WG 私有，不可直接跨 WG 交换值。

下一步：复用保存的 `scaling_four_sm.py`，先测更大指令规模下的 race-check 时间与内存，再把列分片接入一个完整 stage。当前只验证了独立小内核，完整 P1/D1 的正确性、提交时序和验证成本尚未确立。
