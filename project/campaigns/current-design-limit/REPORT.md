# Current design limit: cand-143

## Verified head

`cand-143` is the highest fully graded candidate in this campaign.

| Metric | cand-124 start | cand-143 | Change |
|---|---:|---:|---:|
| Score | 6189.6015 | **7013.6102** | **+13.31%** |
| P1 cycles | 5,072,661 | **4,897,135** | -3.46% |
| D1 cycles | 207,673 | **167,539** | -19.33% |
| P1 baseline speedup | 6.11x | **6.33x** | |
| D1 baseline speedup | 6.27x | **7.77x** | |
| Area | 22.700 mm² | **22.860 mm²** | eligible |
| Peak power P1 / D1 | 4.015 / 6.707 W | **4.019 / 7.495 W** | eligible |

Both cases pass frozen functional validation. The final grade process used
4,069.9 MiB peak memory. No official server submission was made.

## Changes that survived full grade

- P1 first-token GEMMs use N64/K16, reducing expanded P1 instructions from
  89,579 at cand-124 to 81,947 before later fusion.
- Prompt W2 accumulators remain in RF through residual+bias, removing the
  `ff2` HBM round trip.
- Prompt residual1 flows directly into LN2; LayerNorm affine parameters are
  cached in RF.
- LN1 and LN2 prompt activations reuse the existing 64 KiB SH allocation.
- P1 MLP looks ahead across neuron panels; QKV/WO look ahead across output
  columns. These move seed loads without changing arithmetic or event count.
- D1 GELU is partitioned across four persistent owners.
- D1 projections double-buffer B panels and keep the complete single-token A
  vector resident in RF across output panels.
- D1 history K/V panels are double-buffered inside each attention owner.
- Multicast is enabled; it was the only positive remaining single hardware
  field in the bounded post-pipeline screen.

## Rejected boundaries

- P1 step N128/K8 and N256/K4 were both slower than N64/K16.
- A 128 KiB SH prompt activation path was correct but lost pipeline overlap.
- Persistent D1 W1 shards in SH were correct but slower than the 16 MiB cache
  path (cand-140: score 6931.38).
- Layer-major batch ordering was neutral on cache16 and did not make the legal
  TC3/cache4 exchange competitive (proxy score 6139.90).
- TC3/8x16 with cache4 or cache2 was legal but slower (about 6104.6 proxy).
- Increasing SH banks or ports changed no cycles. Larger NoC and more DMA
  engines had already shown no benefit. Relevant bandwidth/resource menu
  fields are now at their legal maxima in cand-143.

## Final bottleneck attribution

P1 (`4,897,135` cycles, single WG on SM0):

| Stage | Cycles | Share |
|---|---:|---:|
| RF-persistent prompt MLP | 2,486,327 | 50.77% |
| QKV | 758,838 | 15.50% |
| Attention | 506,946 | 10.35% |
| Fused output epilogue | 325,105 | 6.64% |
| first-token W1 / W2 | 274,050 / 272,490 | 5.60% / 5.56% |
| WO | 253,842 | 5.18% |

D1 (`167,539` cycles, four persistent WGs): W1 23.51%, W2 23.56%, QKV
18.71%, Attention 17.54%, WO 6.37%, and all remaining stages individually
below 3% except no stage.

The P1 resource lower bound is now dominated by RF service (`rf_read`
2.64M and `rf_write` 2.36M busy cycles); RF ports are already at the legal
8R4W maximum. TC3 requires reducing cache and loses overall. Further material
single-SM improvement therefore requires a new RF layout/tile family rather
than another parameter sweep.

## Multi-SM validation boundary

After all event reductions, cand-141 re-ran the minimal two-WG batch-parallel
P1 preflight: 78,032 events, 156,360 graph edges, and a 13.773 GB retained
event-set logical lower bound against the 6,144 MiB local validation budget.
It was pruned before grade. This is not a performance failure; no multi-SM
cycles are claimed. It proves that barrier removal plus the current event
compression is still insufficient for frozen-checker validation on this Mac.

## Stop decision

The implemented design family is saturated:

- three bounded P1 step tile points establish the local optimum;
- SH residency and persistent-weight variants are measured and rejected;
- remaining legal hardware exchanges are measured or analytically throughput
  equivalent;
- the last P1/D1 pipeline candidates yield sub-1% gains;
- multi-SM P1 remains validation-cost pruned with a quantified 2.24x gap.

A new campaign should begin only with a different RF layout/event-compressed
algorithm, or with an execution environment that can validate the existing
multi-SM skeleton. Repeating tiles, SH capacity, hardware bandwidth, or the
current point-to-point panel pipelines is not justified by the measurements.
