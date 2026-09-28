# search_state (auto-generated; do not edit)

- best: cand-083 score=2245.0389561884695 (P1=10750199, D1=744864)
- phase=2 round=12 plateau_rounds=0 explore_ratio=0.54 need_new_operator=False
- bottleneck: M1_P1 sm0.rf_read=3353480, sm0.rf_write=3218344, sm0.dma=2465984; M2_D1 sm0.rf_write=263232, sm0.dma=244416, shared.noc=227385 (overlapping service counts)

## elite (score desc)
- cand-083 score=2245.04 parent=cand-082 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1 hyp=Compose beneficial control cand-077
- cand-085 score=2245.04 parent=cand-083 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:reduction_units=1+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1 hyp=Archive mechanism expansion: reduction_units=1
- cand-084 score=2206.25 parent=cand-083 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1 hyp=Stage sensitivity: d1 k_chunk=16
- cand-078 score=2153.67 parent=cand-073 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x16+nkm+nopf+s:rf+stage:d1 hyp=Hardware sensitivity: tc_array=8x16 against cand-073
- cand-082 score=2131.15 parent=cand-076 sig=fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1 hyp=Compose beneficial control cand-080

## failure classes
- dry_run: 27
- evaluation_resource_limit: 1
- evaluator_assertion: 1
- generate_error: 3
- harness_noop: 9

## pareto front
- cand-083 {'M1_P1': 10750199, 'M2_D1': 744864}

## diversity cells
- attention|balanced: cand-019
- compute|balanced: cand-084
- compute|d1: cand-072
- compute|p1: cand-070
- fusion|balanced: cand-009
- hardware|balanced: cand-083
- memory|balanced: cand-073
- sig:fuse: cand-009
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:reduction_units=1+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-085
- sig:fuse+hw:cache_mib=2+hw:dma_depth=2+hw:dma_engines=2+hw:rf_ports=8R4W+hw:sm_noc_bytes_per_cycle=128+hw:tc_array=8x8+hw:tc_k_parallel=4+nkm+nopf+s:rf+stage:d1: cand-083
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
