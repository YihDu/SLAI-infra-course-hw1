# search_state (auto-generated; do not edit)

- best: cand-072 score=2004.670977306224 (P1=12522857, D1=801958)
- phase=2 round=12 plateau_rounds=0 explore_ratio=0.54 need_new_operator=False
- bottleneck: P1 sm0.issue_wait=12257978

## elite (score desc)
- cand-072 score=2004.67 parent=cand-071 sig=stage:d1 hyp=D1 stage-specific k_chunk 32 to 24 with n_tile 32
- cand-071 score=1996.48 parent=cand-065 sig=stage:d1 hyp=D1-only GEMM tile n32 k32 with stage-specific recipe
- cand-065 score=1965.99 parent=cand-059 sig=hw:dma_depth=2 hyp=expand cand-059: dma_depth=2 to make prefetch actually overl
- cand-067 score=1894.93 parent=cand-056 sig=hw:hbm_channels=8 hyp=expand cand-056: double HBM channels [hbm_channels=8]
- cand-070 score=1875.68 parent=cand-059 sig=stage:p1_prompt hyp=P1 prompt m_tile 16 with stage-specific GEMM policy on dma_d

## failure classes
- dry_run: 27
- evaluation_resource_limit: 1
- evaluator_assertion: 1
- generate_error: 3
- harness_noop: 9

## pareto front
- cand-072 {'M1_P1': 12522857, 'M2_D1': 801958}

## diversity cells
- attention|balanced: cand-019
- compute|balanced: cand-046
- compute|d1: cand-072
- compute|p1: cand-070
- fusion|balanced: cand-009
- hardware|balanced: cand-047
- memory|balanced: cand-065
- sig:fuse: cand-009
- sig:fuse+hw:hbm_channels=1+nkm+pf+s:rf: cand-049
- sig:fuse+mnk+nopf+s:hbm: cand-021
- sig:fuse+nkm: cand-010
- sig:fuse+nkm+pf+s:rf: cand-033
- sig:fuse+nkm+s:rf: cand-062
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
