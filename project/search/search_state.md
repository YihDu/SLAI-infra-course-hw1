# search_state (auto-generated; do not edit)

- best: cand-047 score=1874.814304538389 (P1=12917099, D1=888914)
- phase=2 round=10 plateau_rounds=1 explore_ratio=0.54 need_new_operator=False
- bottleneck: P1 sm0.issue_wait=12712892

## elite (score desc)
- cand-047 score=1874.81 parent=cand-020 sig=hw:rf_ports=8R4W hyp=expand cand-020: widen RF ports to relieve MMA read-port bou
- cand-052 score=1860.47 parent=cand-019 sig=hw:rf_ports=8R4W hyp=expand cand-019: widen RF ports to relieve MMA read-port bou
- cand-051 score=1761.56 parent=cand-047 sig=fuse+nkm+s:rf hyp=exploit cand-047: tiles (16, 16, 48)->(16, 16, 24)
- cand-050 score=1296.92 parent=cand-047 sig=fuse+nkm+s:rf hyp=exploit cand-047: tiles (16, 16, 48)->(16, 8, 48)
- cand-053 score=1245.03 parent=cand-044 sig=hw:dma_depth=2 hyp=expand cand-044: dma_depth=2 to make prefetch actually overl

## failure classes
- dry_run: 27
- generate_error: 3
- harness_noop: 9

## pareto front
- cand-047 {'M1_P1': 12917099, 'M2_D1': 888914}

## diversity cells
- attention|balanced: cand-019
- compute|balanced: cand-046
- fusion|balanced: cand-009
- hardware|balanced: cand-047
- memory|balanced: cand-053
- sig:fuse: cand-009
- sig:fuse+hw:hbm_channels=1+nkm+pf+s:rf: cand-049
- sig:fuse+mnk+nopf+s:hbm: cand-021
- sig:fuse+nkm: cand-010
- sig:fuse+nkm+pf+s:rf: cand-033
- sig:fuse+nkm+s:rf: cand-051
- sig:hw:dma_depth=2: cand-053
- sig:hw:reduction_units=1: cand-044
- sig:hw:rf_ports=8R4W: cand-047
- sig:hw:vector_lanes=8: cand-045
- sig:mnk: cand-000
- sig:mnk+nofuse+nopf+s:hbm: cand-027
- sig:nkm: cand-001
- sig:nkm+pf: cand-003
- sig:nkm+pf+s:rf: cand-015
- sig:nkm+s:rf: cand-017
- sig:pf: cand-025
- sig:s:rf: cand-019
