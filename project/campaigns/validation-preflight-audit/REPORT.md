# 本机继续执行前的预筛选复核

## 当前约束

用户已明确继续在同一台16GiB Mac上运行，不转移服务器；frozen checker仍不允许修改。用户再次给出的Part A–E要求继续有效，包括Part C中暂时保留成熟Attention/LN/residual路径。

既有cand-101..105已经分别做过一次完整grade尝试，均在checker内存阶段中断。本次没有重跑这些候选，没有新建完整模型候选，也没有新的correctness/timing/score。

## 本次实际完成

1. 复核frozen ISA/race checker关于events、loop expansion、WG lifecycle、WAIT/BARRIER的语义。未发现能在保持既有计算路径的同时清除历史event集合的ISA操作。
2. 补充`project/search/tests/test_validation_preflight.py`：
   - 真正启动generation、static parser、cost model三个子进程；构造两个WG的合法数据依赖链。
   - 成本预测超预算后停止，grade启动次数为0。
   - 已prepared输入也不能忽略现有成本拒绝报告后继续grade。
   - calibration override及说明正确保留；预算耗尽时没有评测子进程启动。
3. 修复`evaluate(prepared=True)`：在最终metrics中保留validation-cost报告、成本阶段heartbeat以及明确的calibration说明；对没有override的已拒绝成本报告执行同样的prune。

验证命令：

```sh
.venv/bin/python -m unittest project.search.tests.test_validation_preflight -v
```

结果：2个新增测试通过，1.043秒。证据：本目录`tests.stderr`、`tests.stdout`。这是预筛选测试，不是完整P1 grade。

## 结论边界

- 不能把逻辑event集合的存储下界直接当作Mac实测物理内存下界。
- 不能从未找到表示方法推导出“任何未来multi-SM算法都不可能在16GiB完成”。只说明现有成熟路径和已试五种增量结构尚无可行的本机验证方案。
- 当前唯一events不可合并、静态FOR必须展开、barrier不会压缩集合；减少源码行数/同步数不是已验证的解法。
- 五个候选的既有结果、phase归因、traffic分析及成本模型见`../validation-efficient/REPORT.md`。

## 待用户明确的下一步

已询问是否允许提前改Attention实际分块，以降低真实展开事件数，再重新尝试only-W1。这会放宽Part C中“保持cand-098成熟Attention路径”的约束，但不改变工作负载或checker。

未获得该选择前，不实施依赖这一授权的算法改写。不启动服务器、不改frozen checker、不把成功标准标记完成。
