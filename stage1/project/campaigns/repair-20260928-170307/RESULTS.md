# Search repair campaign results

Baseline: cand-073 (2051.477). Selected incumbent: cand-083 (2245.039, +9.44%).

Actual evaluator launches: 12/12; cache hits during this campaign: 0; failed evaluations: 0.

| Candidate | Change | P1 cycles | D1 cycles | Score | Wall seconds |
|---|---|---:|---:|---:|---:|
| cand-074 | Reproduce cand-073 with isolated single-grade runner | 12347819 | 776636 | 2051.47741118116 | 125.6 |
| cand-075 | Hardware sensitivity: tc_k_parallel=2 against cand-073 | 13129883 | 759924 | 2011.1993372800794 | 249.8 |
| cand-076 | Hardware sensitivity: tc_k_parallel=4 against cand-073 | 12093131 | 753564 | 2104.4624234432395 | 247.8 |
| cand-077 | Hardware sensitivity: tc_array=8x8 against cand-073 | 12064091 | 776636 | 2075.4609403356135 | 246.3 |
| cand-078 | Hardware sensitivity: tc_array=8x16 against cand-073 | 11412395 | 762444 | 2153.6655079126076 | 246.0 |
| cand-079 | Hardware sensitivity: tc_count=4 against cand-073 | 13207115 | 767372 | 1995.554860534254 | 237.7 |
| cand-080 | Hardware sensitivity: dma_engines=2 against cand-073 | 12161525 | 768440 | 2078.124801537899 | 218.3 |
| cand-081 | Hardware sensitivity: cache_mib=4 against cand-073 | 12347819 | 776636 | 2051.47741118116 | 206.9 |
| cand-082 | Compose beneficial control cand-080 | 11929847 | 744864 | 2131.1533261851882 | 134.0 |
| cand-083 | Compose beneficial control cand-077 | 10750199 | 744864 | 2245.0389561884695 | 174.7 |
| cand-084 | Stage sensitivity: d1 k_chunk=16 | 10750199 | 771287 | 2206.2481983505695 | 167.3 |
| cand-085 | Archive mechanism expansion: reduction_units=1 | 10750199 | 744864 | 2245.0389561884695 | 130.7 |

## Verification

- 39 mechanism tests passed (proposal quotas, bounded processes, budget, cache, state and archives).
- cand-074 reproduced cand-073 exactly: both assembly files, cycles and score.
- All completed grades retain frozen evaluator reports and logs. No evaluator/manifests were edited.
- Root hardware, compiler, recipe and both programs match the selected candidate.
- 8x16 with K parallelism 4 requires 30.5297 mm² and was excluded by the 24 mm² limit.
- K parallelism 2 and TC count 4 regressed score; cache 4 MiB gave no cycle improvement.
- Exact historical cache import verifies raw report provenance, runtime, seed, baseline and artifacts.
- Busy-resource times overlap; they are not additive bottleneck percentages.

## Remaining opportunities

Only SM 0 is currently active. Fewer SMs can free area for larger TC configurations; useful multi-SM programs require another compiler change and bounded validation. This campaign expands existing operators; it does not implement a new fused or distributed compiler backend.
