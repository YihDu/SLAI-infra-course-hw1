# search_state (auto-generated; do not edit)

- best: cand-095 score=2567.954051821853 (P1=8522411, D1=718132)
- phase=2 round=12 plateau_rounds=0 explore_ratio=0.54 need_new_operator=False
- bottleneck: M1_P1 sm0.rf_read=2818280, sm0.rf_write=2788264, sm0.dma=1876160; M2_D1 sm0.rf_write=261248, sm0.dma=239104, shared.noc=226521 (overlapping service counts)

## elite (score desc)
- cand-095 score=2567.95 parent=cand-094 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes hyp=Larger M tile reduces repeated B reads
- cand-094 score=2547.92 parent=cand-093 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes hyp=Expand decode N tile after prompt/hardware controls
- cand-093 score=2512.22 parent=cand-087 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes hyp=Compose best RF dataflow with measured area bridge
- cand-097 score=2479.58 parent=cand-094 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+hw:vector_lanes=8+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes hyp=Joint RF expansion: 8 vector partitions allow N128 with K16,
- cand-087 score=2390.11 parent=cand-083 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes hyp=RF accumulator partitions: P1 prompt M=16, N=32, K=32; prese

## failure classes
- dry_run: 27
- evaluation_resource_limit: 1
- evaluator_assertion: 1
- evaluator_error: 1
- generate_error: 3
- harness_noop: 9

## pareto front
- cand-095 {'M1_P1': 8522411, 'M2_D1': 718132}

## diversity cells
- attention|balanced: cand-019
- compute|balanced: cand-095
- compute|d1: cand-072
- compute|p1: cand-070
- fusion|balanced: cand-009
- hardware|balanced: cand-093
- memory|balanced: cand-073
- memory|p1: cand-086
- sig:fuse: cand-009
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:reduction_units=1+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-085
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+hw:vector_lanes=8+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes: cand-097
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm: cand-092
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes: cand-095
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_count=4+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm: cand-091
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+kernel:p1_w1_shared+nkm+nopf+s:rf+stage:d1: cand-086
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-083
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1:nkm+stage:p1_prompt:nkm_lanes: cand-087
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-082
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+nkm+nopf+s:rf+stage:d1: cand-080
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+nkm+nopf+s:rf+stage:d1: cand-078
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+nkm+nopf+s:rf+stage:d1: cand-077
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_count=4+nkm+nopf+s:rf+stage:d1: cand-079
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_k_parallel=2+nkm+nopf+s:rf+stage:d1: cand-075
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-076
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+nkm+nopf+s:rf+stage:d1: cand-074
- sig:fuse+hw:cache_mib=4+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+nkm+nopf+s:rf+stage:d1: cand-081
- sig:fuse+hw:hbm_channels=1+nkm+pf+s:rf: cand-049
- sig:fuse+mnk+nopf+s:hbm: cand-021
- sig:fuse+nkm: cand-010
- sig:fuse+nkm+pf+s:rf: cand-033
- sig:fuse+nkm+s:rf: cand-062
- sig:hw:cache_mib=2: cand-073
- sig:hw:dma_depth=2: cand-065
- sig:hw:hbm_channels=8: cand-067
- sig:hw:noc_bytes_per_cycle=128: cand-063
- sig:hw:reduction_units=1: cand-044
- sig:hw:rf_ports=8R4W: cand-047
- sig:hw:sm_noc_bytes_per_cycle=128: cand-059
- sig:hw:sm_noc_bytes_per_cycle=32: cand-057
- sig:hw:tc_count=1: cand-064
- sig:hw:vector_lanes=8: cand-045
- sig:mnk: cand-000
- sig:mnk+nofuse+nopf+s:hbm: cand-027
- sig:nkm: cand-001
- sig:nkm+pf: cand-003
- sig:nkm+pf+s:rf: cand-015
- sig:nkm+s:rf: cand-017
- sig:pf: cand-066
- sig:s:rf: cand-019
- sig:stage:d1: cand-072
- sig:stage:p1_prompt: cand-070
