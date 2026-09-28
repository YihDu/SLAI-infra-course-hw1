# Validation-efficient multi-SM：本轮证据与未完成项

## 状态

用户要求本轮写文档、提交 Git、交给新 session。实验已停止。接手入口：`project/CURRENT_SESSION_HANDOFF.md`。

**没有新的完整 multi-SM grade pass，没有实测多 SM speedup。** 五种增量结构都进行了有界完整 grade 尝试，在 frozen race checker 内存阶段终止。未改 frozen evaluator，未向官方 server 提交。

## A. cand-098 / 099 / 100 程序对照

| 指标 | cand-098 | cand-099 | cand-100 |
|---|---:|---:|---:|
| 展开指令 | 161,519 | 166,289 | 166,283 |
| events | 161,516 | 166,196 | 166,196 |
| LD | 59,224 | 60,004 | 60,004 |
| ST | 13,948 | 14,338 | 14,338 |
| WG | 1 | 4 | 1 |
| BARRIER | 0 | 84 | 84 |
| WAIT | 0 | 0 | 0 |
| STEP.COMMIT | 1 | 1 | 1 |
| 展开依赖图边 | 323,036 | 332,828 | 332,564 |
| barrier 跨 WG 有向组对关系 | 0 | 1,008 | 0 |
| 显式 named-event 跨 WG sync edges | 0 | 0 | 0 |
| frozen checker 单 WG 快速返回 | 是 | 否 | 是 |

“组对关系”是4组 barrier 的4×3个逻辑排序关系，不等同于图中的点对点 WAIT 边。同步图采用 shared barrier vertex，其 incidence 边不同。baseline/control 图是成本分析器为比较而构造；真实 checker 因快速返回不会构造它。

各 phase 的 LD/ST、events、graph/commit edges、静态 loop 数、源代码行数在 `cand-098-cost.json`、`cand-099-cost.json`、`cand-100-cost.json`。

P1 全部 prompt + 首 token 的阶段 events：

| 阶段 | 098 | 099 | 100 |
|---|---:|---:|---:|
| LN1 | 5,850 | 5,850 | 5,850 |
| QKV | 12,096 | 12,096 | 12,096 |
| Attention | 84,960 | 84,960 | 84,960 |
| WO | 4,032 | 4,032 | 4,032 |
| LN2 | 5,850 | 5,850 | 5,850 |
| W1 | 16,128 | 16,128 | 16,128 |
| GELU | 14,040 | 18,720 | 18,720 |
| W2 | 14,400 | 14,400 | 14,400 |

其余是 residual/output。增加的4,680个事件来自 GELU 1024维分成4×256：原来每行3个384上限块，分片后每行4块。不是所有计算乘了WG数。

Attention 重复来源是 batch×layer×query×head×ceil(valid_keys/16)，GEMM 是 batch×layer×Ntiles×Ktiles×Mtiles；分片时N/head域被分割，未与WG做重复笛卡尔积。

### 前三个结构原因

1. 多 WG 激活全程序检查，连成熟单主 WG Attention/LN 也进入依赖集合保存路径。
2. RF buffer反复读写使 `_Knowledge.events` 单调积累；每条event又保留一个独立frozenset快照。
3. barrier/commit/predecessor 合并历史，但 prefix 不消除 events；加粗 barrier或重开WG不能回收历史快照。

唯一 event 是 ISA 强制要求，FOR 在 checker 中展开。不能省略event、重复event、绕过racecheck，或靠更短源码使验证图变小。

## B. Validation cost model 与剪枝

模型实现：`project/search/validation_cost.py`。

```
source + loops → frozen iter_parse 展开指令
→ 与 checker 相同的 tape/sync graph
→ RF/SH写依赖、barrier、commit、predecessor 的最长事件链 d(e)
→ 每个真实事件依赖集合大小至少 d(e)
→ 多WG retained event-set逻辑存储下界 ≈ 152E + 16Σd(e) bytes
```

该下界省略 HBM 依赖与多路径并集，省略 graph/prefix/tuple/dict 的附加内存。常数适用于当前64位CPython set布局；152项保守扣除了small-set内联空间。单WG实际快速返回，因此 checker retained-set预测为0。

**逻辑堆下界不是物理内存/RSS下界**，尤其Mac有内存压缩。不能把模型值宣称为实测全程峰值。模型的安全用途是提前拒绝在当前预算下明显不值得启动的实验；通过模型不代表合法或内存一定够。

- cand-099 retained-set逻辑存储下界约29.7 GiB。
- 五个incremental候选约102–108 GiB；长主WG链使只并行部分算子未必更省checker内存。
- only-W1前10,000条的实测set bytes / 模型下界 = 4.43；若比例不变，完整事件集合约478 GiB。这是**单前缀外推**，不是峰值预测保证，更不是机器采购依据。
- `only-w1-race-phases.json`：第一并行入口PC=17,412；最后观测PC=9,999仍在prompt Attention，事件集合已2.049GB。因此膨胀发生在真正并行W1之前。

harness默认在generate/static后跑模型，超预算标记`validation_cost_pruned`，不消耗grade budget。五个指定结构是明确记录的calibration override，各只运行一次bounded grade。未来不要继承该override来进行参数扫点。

## C. 五种完整增量 grade

| ID | Prompt并行GEMM | instructions | events | BARRIER | 峰值MiB | grade秒 | 逻辑下界GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 101 | W1 | 161,537 | 161,516 | 12 | 3079.02 | 40.45 | 107.85 |
| 102 | W2 | 161,537 | 161,516 | 12 | 3072.35 | 36.13 | 108.11 |
| 103 | W1+W2 | 161,549 | 161,516 | 24 | 3085.69 | 36.62 | 104.97 |
| 104 | QKV | 161,537 | 161,516 | 12 | 3078.22 | 40.49 | 108.18 |
| 105 | QKV+W1+W2 | 161,561 | 161,516 | 36 | 3080.00 | 37.37 | 102.15 |

每个4WG、0WAIT，D1/硬件未改。结果均`not_verified / memory_limit`，P1/D1 cycles、score、各阶段cycles和实际speedup均**不可用**，没有新的structural parent。这不属于连续低收益的inner search。

证据在各候选`reports/validation-cost.json`、`reports/grade.heartbeat.json`、`metrics.json`；完整grade没有生成最终grade.json。

## D. 更大结构方向：已量化的机会，尚未实现/测量

`traffic.py` 对字节相同的cand-098程序按真实ABI/scratch区间统计，结果在`traffic.json`。这里是HBM操作数的**逻辑请求字节**，并非cache miss后物理HBM流量。

### Cross-op residency

| 中间值/边界 | 写入bytes | 后续读取bytes |
|---|---:|---:|
| LN1 → QKV（ln1） | 399,360 | 9,732,096 |
| Attention → WO（ctx） | 399,360 | 3,244,032 |
| WO → residual（att） | 399,360 | 399,360 |
| residual保留至LN2/最终残差（res） | 399,360 | 798,720 |
| LN2 → W1（ln2） | 399,360 | 12,976,128 |
| W1 → GELU（ff1） | 1,597,440 | 1,597,440 |
| GELU → W2（gelu） | 1,597,440 | 12,976,128 |
| W2 → residual（ff2） | 399,360 | 399,360 |

同一SM/WG内的row-block persistent MLP是有依据的候选：16个token的hidden为16KiB、FFN激活64KiB，可考虑SH128KiB内的复用安排；RF算子需逐tile搬运，SH私有，不能把跨WG列分片输入直接当共享驻留。

W1+GELU+W2占完整baseline约52.24%。若该整个区域经并行/驻留达到4×、其他不变，Amdahl约1.64× P1；要超过1.5×，该区域约需2.76×。这是**潜力门槛**，不是实现或性能证据。复制weights、NoC、SH/RF端口可能抵消收益。

### D1 persistent execution

- 现有D1已经只有一个WG，跨8个STEP.COMMIT；commit并不删除RF/SH，WG.END才释放。
- 全部weights/参数unique = 1,582,080 bytes，8step请求12,656,640 bytes，恰好8遍。
- 实际完整D1 HBM read仅2,012,160 bytes（cache2MiB），NoC=15,302,656 bytes。因此“weights读8遍”主要说明cache→RF/NoC重复，不能宣称HBM每步都重新读1.58MB。
- 单WG RF64KiB + SH128KiB放不下全部1.58MB weights；完整全芯片驻留也受SM/WG私有空间和中间值容量限制。
- 可研究跨step保留热点weight shard、固定head/FFN owner、tile weights留SH/RF，不能跨step提前读未释放输入，不能跨step重排有因果依赖的计算。
- 多WG D1也会触发同一个checker问题；单WG局部persistent可继续保持快速路径。未据孤立1.1% fusion否定此路线，亦未实现新D1候选。

### Attention算法

- 当前scores/probs已经RF-resident，没有scores/probs HBM materialization；“消除scores写HBM”不是新的收益。
- 完整P1 Attention = 84,960 events，约52.6%计算/访存event；归因cycles=1,811,460，约21.26%。只加速Attention，即使无限快，P1上限也约1.27×。
- new_K与new_V逻辑读取各13,178,880 bytes（合计26,357,760）；写入各399,360 bytes。每个query重复扫描已出现KV。
- block query streaming可在多个query间复用同一KV块、将M=1小MMA合为M>1，同时减少LD/MMA初始化事件。在线softmax可避免完整score向量，但增加running max/sum/rescale工作；当前RF scores很短，不能先验认为online更快。
- KV驻留/块复用可减少逻辑请求和事件，但真实收益要扣除SH↔RF、mask/softmax和跨WG依赖，且需要用户放宽本阶段成熟Attention路径约束后才能提前实施。

尚未找到经实验确认的>1.5×新结构；以上给出的是可检验的容量/流量/阶段占比依据。

## E. Search分层

- Outer身份：parallel skeleton、fusion partition、residency、attention algorithm、persistent scheduling。
- Inner变动：tile M/N/K、shard数、WG数、placement、hardware；这些不能重置outer低收益计数。
- registry开启：从cand-101起，连续3个eligible候选对parent score增益<10%则阻止该outer的inner proposals；失败/无分数不计入。
- 这是预算/身份机制；未来结构的代码生成仍由明确hypothesis实现，不能只加一个recipe标签就当成已支持的新算法。

## 测试与交接限制

51个search tests通过，记录在`tests.stderr`。成本模型下界在小程序上对照原始checker实际集合大小通过；没有修改checker。默认compiler输出与cand-098字节一致。

成本模型默认prune路径尚缺专门的完整集成测试；prepared=True时部分preflight字段不进入最终metrics，但独立cost报告和proposal保留。下一session在进一步扩展前可补这个小范围验证，不必重跑全部五个OOM实验。

当前没有运行中的grade或worker。用户要新session接手，不自动开启更多实验。
