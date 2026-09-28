# Iteration Log

Chronological record of every evaluated candidate: changes, local checks,
scores/observations, rejected ideas, and submission rationale. Entry template
is in `AGENTS.md` §7. Append newest entries at the bottom.

Baselines to beat (frozen manifest, seed 7, 500 MHz):
- M1_P1: 30,996,995 cycles
- M2_D1: 1,302,032 cycles
- Eligibility: both correct, area ≤ 24 mm², peak power ≤ 20 W, ≤ 2× baseline each

---

## 2026-09-28 — Repo setup and baseline import

- Change: none (unmodified starter). Initialized git repo, branch `dev-yh`,
  remote `origin` → github.com/YihDu/SLAI-infra-course-hw1. Added `AGENTS.md`
  briefing + this log.
- Verify: not yet run — environment pending (need Python 3.12.13 + NumPy 2.5.3;
  system Python is 3.14.3 without NumPy).
- Result: n/a
- Decision: next step is environment setup, then reproduce baseline
  check/estimate/grade before any optimization.

## 2026-09-28 — Baseline reproduction (env ready)

- Change: none. Created `.venv` via uv (Python 3.12.13 + numpy 2.5.3).
- Verify: `python -m project.compiler` -> asm sha256 == MANIFEST.json;
  `check-1.json` PASS both cases (max err ~4.7e-6); `grade-1.json`.
- Result: P1 = 30,996,995 / D1 = 1,302,032 cycles, score 1000.0, eligible.
  area 5.60 mm^2, peak 1.77 W. SM0 issue_wait = 98.9% of cycles.
- Decision: baseline confirmed; start Phase A software experiments.

## 2026-09-28 — Candidate 1: B-stationary GEMM loop order (KEEP)

- Hypothesis: `Builder.gemm` order m->n->k re-loads each weight tile once per
  8-row m-tile (~8x per GEMM, x2 batches). Reordering to n->k->m keeps the B
  tile in RF lane 1 across all m-tiles; all m-tile accumulators for one n-tile
  live in lane 2 (8 tiles x 8x16 = 1024 words = exactly one lane).
- Change: `project/compiler.py` `Builder.gemm` only; single workgroup,
  baseline hardware, everything else untouched. Guard added:
  `m_extent * tile_n <= 1024`.
- Static check: expanded P1 LDs with weight-region src 63,600 -> 16,560
  (prompt GEMMs exactly /8); total LDs 165,130 -> 118,090. D1 identical (m=1).
- Verify: `check-2.json` PASS (seed 7, both cases, max err ~4.7e-6);
  `estimate-2.json`; `grade-2.json` eligible.
- Result:

  | metric | P1 base -> new | D1 base -> new |
  |---|---|---|
  | cycles | 30,996,995 -> 22,373,717 (1.385x) | 1,302,032 -> 1,310,832 (0.993x) |
  | HBM read | 284.8 -> 152.7 MB (-46%) | 16.1 MB (unchanged) |
  | instr | 336,275 -> 289,235 | ~same |
  | rf_read util | 29.3% -> 40.5% (busy unchanged 9.07M) | 21.8% -> 21.7% |
  | rf_write busy | 10.62M -> 6.49M | unchanged |
  | tc busy | 5.63M (unchanged; util 18.1% -> 25.1%) | unchanged |
  | dma busy | 4.62M -> 2.55M | unchanged |
  | hbm channel util | ~7.5% -> ~5.7% each | ~9.9% unchanged |

- Score: **1173.1** (baseline 1000). D1 tiny regression (~+0.7% cycles) likely
  from hoisted acc-init ordering; acceptable.
- Learning: hypothesis CONFIRMED for prefill. Removing 47k weight LDs cut
  ~8.6M cycles ~ traffic+RF-write service time. issue_wait still ~98.7%:
  single-WG serial dependency chain is now the dominant limiter, and
  rf_read (MMA operand reads, 40.5% util) is the busiest resource.
- Decision: KEEP. Next: (a) parallelism across workgroups/SMs, (b) cut the
  remaining scratch round-trips (scores/probs/intermediates), (c) revisit MMA
  operand staging since rf_read is now top consumer.

## 2026-09-28 10:46:22 — cand-000: frozen baseline reference point (score=1000)
- Parent: (root); Summary: unmodified starter compiler + baseline hardware
- Inputs: compiler@9c706d56 hardware@a37b1544
- Hypothesis: frozen baseline reference point (score=1000)
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=30996995 (1.000x), M2_D1=1302032 (1.000x); score=1000.0, eligible=True
  - M1_P1: hbm_r=284.8MB hbm_w=10.5MB instr=336275 peakW=1.77; top busy: sm0.issue_wait=30660640, sm0.rf_write=10623464, sm0.rf_read=9069448, sm0.tc=5625504
  - M2_D1: hbm_r=16.1MB hbm_w=0.3MB instr=22362 peakW=1.45; top busy: sm0.issue_wait=1279478, sm0.rf_write=527808, sm0.rf_read=284064, sm0.dma=257210
- Decision: pending keep/revert

## 2026-09-28 10:46:22 — cand-001: B-stationary GEMM loop order n->k->m removes ~8x repeated weight loads
- Parent: cand-000; Summary: gemm() loop reorder; accs live in lane2; single WG unchanged
- Inputs: compiler@57e341f0 hardware@a37b1544
- Hypothesis: B-stationary GEMM loop order n->k->m removes ~8x repeated weight loads
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=22373717 (1.385x), M2_D1=1310832 (0.993x); score=1173.0811006016247, eligible=True
  - M1_P1: hbm_r=152.7MB hbm_w=10.5MB instr=289235 peakW=1.77; top busy: sm0.issue_wait=22084402, sm0.rf_read=9069448, sm0.rf_write=6494696, sm0.tc=5625504
  - M2_D1: hbm_r=16.1MB hbm_w=0.3MB instr=22362 peakW=1.43; top busy: sm0.issue_wait=1288278, sm0.rf_write=527808, sm0.rf_read=284064, sm0.dma=257210
- Decision: pending keep/revert

### cand-000 verdict: KEEP — reference root

### cand-001 verdict: KEEP — P1 1.385x, correctness pass; serial chain still dominant

### cand-000 verdict: KEPT — reference root

### cand-001 verdict: KEPT — P1 1.385x, correctness pass; serial chain still dominant

## 2026-09-28 11:35:17 — drun-001: exploit cand-001: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:35:17 — drun-002: exploit cand-001: tiles (8, 16, 48)->(2, 16, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 2, "n_tile": 16, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:35:17 — drun-003: exploit cand-001: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:35:17 — drun-004: expand cand-000: fuse bias+gelu epilogue
- Parent: cand-000; kind=expand family=fusion sig=fuse
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:35:17 — drun-005: novelty: untried structure nkm_pf on cand-001
- Parent: cand-001; kind=novelty family=memory sig=nkm+pf
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_prefetch", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

### round 1 summary: best=cand-001 score=1173.081 gain=1.0000 plateau=1 explore=0.34

## 2026-09-28 11:37:16 — drun-006: exploit cand-001: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-007: exploit cand-001: tiles (8, 16, 48)->(2, 16, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 2, "n_tile": 16, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-008: exploit cand-001: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-009: expand cand-000: fuse bias+gelu epilogue
- Parent: cand-000; kind=expand family=fusion sig=fuse
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-010: novelty: untried structure nkm_pf on cand-001
- Parent: cand-001; kind=novelty family=memory sig=nkm+pf
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_prefetch", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

### round 1 summary: best=cand-001 score=1173.081 gain=1.0000 plateau=1 explore=0.34

## 2026-09-28 11:37:16 — drun-011: exploit cand-001: tiles (8, 16, 48)->(8, 16, 24)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 24}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-012: exploit cand-001: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-013: exploit cand-001: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-014: expand cand-000: fuse bias+gelu epilogue
- Parent: cand-000; kind=expand family=fusion sig=fuse
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:37:16 — drun-015: novelty: untried structure scores_rf on cand-001
- Parent: cand-001; kind=novelty family=attention sig=s:rf
- Ops: [{"op": "scores_residency", "space": "rf"}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

### round 2 summary: best=cand-001 score=1173.081 gain=1.0000 plateau=2 explore=0.34

## 2026-09-28 11:38:47 — drun-016: exploit cand-001: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:38:47 — drun-017: exploit cand-001: tiles (8, 16, 48)->(2, 16, 48)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 2, "n_tile": 16, "k_chunk": 48}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:38:47 — drun-018: exploit cand-001: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:38:47 — drun-019: expand cand-000: fuse bias+gelu epilogue
- Parent: cand-000; kind=expand family=fusion sig=fuse
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:38:47 — drun-020: novelty: untried structure nkm_pf on cand-001
- Parent: cand-001; kind=novelty family=memory sig=nkm+pf
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_prefetch", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 11:38:47 — drun-021: exploit cand-001: tiles (8, 16, 48)->(8, 16, 32)
- Parent: cand-001; kind=exploit family=memory sig=nkm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 32}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

### round 1 summary: best=cand-001 score=1173.081 gain=1.0000 plateau=1 explore=0.34


## 2026-09-28 11:46:21 — cand-002: exploit cand-001: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-001; kind=exploit family=memory bias=balanced sig=nkm fp=a9631f7a313e736a
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27996777 M2_D1=1737288 cycles; area=5.59764068864mm2 peakW=1.931752833217892; score=910.920963143551 eligible=True
  - M1_P1: hbm_r=267.5MB top busy: sm0.issue_wait=27557510, sm0.rf_read=9077032, sm0.rf_write=8890856
  - M2_D1: hbm_r=29.7MB top busy: sm0.issue_wait=1698990, sm0.rf_write=552384, sm0.dma=470330
- Decision: status=evaluated failure_class=none


## 2026-09-28 11:48:49 — cand-003: novelty: untried structure nkm_pf on cand-001
- Parent: cand-001; kind=novelty family=memory bias=balanced sig=nkm+pf fp=c72def168724372d
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22494677 M2_D1=1453168 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1111.1503617194342 eligible=True
  - M1_P1: hbm_r=154.5MB top busy: sm0.issue_wait=22203634, sm0.rf_read=9069448, sm0.rf_write=6549992
  - M2_D1: hbm_r=18.5MB top busy: sm0.issue_wait=1429462, sm0.rf_write=601536, sm0.dma=294074
- Decision: status=evaluated failure_class=none

### round 1 summary: best=cand-001 score=1173.081 gain=1.0000 plateau=1 explore=0.34


## 2026-09-28 11:53:12 — cand-004: exploit cand-001: tiles (8, 16, 48)->(2, 16, 48)
- Parent: cand-001; kind=exploit family=memory bias=balanced sig=nkm fp=5e60de2f5066b4dc
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 2, "n_tile": 16, "k_chunk": 48}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=26522537 M2_D1=1310832 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1077.4312710849626 eligible=True
  - M1_P1: hbm_r=152.7MB top busy: sm0.issue_wait=25869190, sm0.rf_read=13788040, sm0.tc=10666656
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1288278, sm0.rf_write=527808, sm0.rf_read=284064
- Decision: status=dominated failure_class=none


## 2026-09-28 11:55:10 — cand-005: exploit cand-001: tiles (8, 16, 48)->(8, 16, 24)
- Parent: cand-001; kind=exploit family=memory bias=balanced sig=nkm fp=56a05cecf41c0696
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 24}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=24445817 M2_D1=1351504 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1105.2478525385854 eligible=True
  - M1_P1: hbm_r=177.0MB top busy: sm0.issue_wait=24039382, sm0.rf_read=9450088, sm0.rf_write=7255976
  - M2_D1: hbm_r=16.3MB top busy: sm0.issue_wait=1315510, sm0.rf_write=536768, sm0.rf_read=288544
- Decision: status=dominated failure_class=none


## 2026-09-28 11:57:04 — cand-006: exploit cand-001: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-001; kind=exploit family=memory bias=balanced sig=nkm fp=0d868c77dfb8fd32
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=23262277 M2_D1=1317280 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1147.6394407423866 eligible=True
  - M1_P1: hbm_r=152.6MB top busy: sm0.issue_wait=23015202, sm0.rf_read=8932168, sm0.rf_write=6220136
  - M2_D1: hbm_r=16.0MB top busy: sm0.issue_wait=1298950, sm0.rf_write=524992, sm0.rf_read=282656
- Decision: status=dominated failure_class=none


## 2026-09-28 12:00:19 — cand-007: expand cand-000: fuse bias+gelu epilogue
- Parent: cand-000; kind=expand family=fusion bias=balanced sig=fuse fp=c6152b1c5747f74a
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=30851135 M2_D1=1298904 cycles; area=5.59764068864mm2 peakW=1.7715864571802942; score=1003.5673543192571 eligible=True
  - M1_P1: hbm_r=283.2MB top busy: sm0.issue_wait=30517120, sm0.rf_write=10573544, sm0.rf_read=9044488
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1276414, sm0.rf_write=526784, sm0.rf_read=283552
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:03:02 — cand-008: expand cand-003: fuse bias+gelu epilogue
- Parent: cand-003; kind=expand family=fusion bias=balanced sig=fuse fp=850a9ef1390e1f3e
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22348817 M2_D1=1450040 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1115.972170751516 eligible=True
  - M1_P1: hbm_r=152.9MB top busy: sm0.issue_wait=22060114, sm0.rf_read=9044488, sm0.rf_write=6500072
  - M2_D1: hbm_r=18.4MB top busy: sm0.issue_wait=1426398, sm0.rf_write=600512, sm0.dma=293002
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:05:42 — cand-009: novelty: untried structure fuse_gelu on cand-001
- Parent: cand-001; kind=novelty family=fusion bias=balanced sig=fuse fp=33c61a7d000396c1
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22227857 M2_D1=1307704 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1178.3304547853895 eligible=True
  - M1_P1: hbm_r=151.1MB top busy: sm0.issue_wait=21940882, sm0.rf_read=9044488, sm0.rf_write=6444776
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1285214, sm0.rf_write=526784, sm0.rf_read=283552
- Decision: status=evaluated failure_class=none

### round 2 summary: best=cand-009 score=1178.330 gain=1.0045 plateau=2 explore=0.34


## 2026-09-28 12:07:40 — cand-010: exploit cand-009: tiles (8, 16, 48)->(16, 16, 48)
- Parent: cand-009; kind=exploit family=fusion bias=balanced sig=fuse+nkm fp=c3a05a00e8785bef
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 16, "k_chunk": 48}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22382453 M2_D1=1307704 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1174.2540289231054 eligible=True
  - M1_P1: hbm_r=151.1MB top busy: sm0.issue_wait=22156150, sm0.rf_read=9044488, sm0.rf_write=6444776
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1285214, sm0.rf_write=526784, sm0.rf_read=283552
- Decision: status=dominated failure_class=none


## 2026-09-28 12:09:25 — cand-011: exploit cand-009: tiles (8, 16, 48)->(8, 16, 32)
- Parent: cand-009; kind=exploit family=fusion bias=balanced sig=fuse+nkm fp=3661bda877c83bf0
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 32}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=24210373 M2_D1=1323688 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1122.2175843408888 eligible=True
  - M1_P1: hbm_r=151.2MB top busy: sm0.issue_wait=23873478, sm0.rf_read=9206728, sm0.rf_write=6769256
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1296206, sm0.rf_write=530112, sm0.rf_read=285216
- Decision: status=dominated failure_class=none


## 2026-09-28 12:12:15 — cand-012: exploit cand-009: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-009; kind=exploit family=fusion bias=balanced sig=fuse+nkm fp=f314351fb378b9e5
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27850917 M2_D1=1734160 cycles; area=5.59764068864mm2 peakW=1.931752833217892; score=914.1264889348438 eligible=True
  - M1_P1: hbm_r=265.9MB top busy: sm0.issue_wait=27413990, sm0.rf_read=9052072, sm0.rf_write=8840936
  - M2_D1: hbm_r=29.6MB top busy: sm0.issue_wait=1695926, sm0.rf_write=551360, sm0.dma=469258
- Decision: status=dominated failure_class=none


## 2026-09-28 12:15:55 — cand-013: expand cand-007: double-buffer B tiles
- Parent: cand-007; kind=expand family=memory bias=balanced sig=pf fp=15dcc438f76ab5b8
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=30851135 M2_D1=1298904 cycles; area=5.59764068864mm2 peakW=1.7715864571802942; score=1003.5673543192571 eligible=True
  - M1_P1: hbm_r=283.2MB top busy: sm0.issue_wait=30517120, sm0.rf_write=10573544, sm0.rf_read=9044488
  - M2_D1: hbm_r=16.1MB top busy: sm0.issue_wait=1276414, sm0.rf_write=526784, sm0.rf_read=283552
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:18:03 — cand-014: expand cand-001: keep attention scores in RF
- Parent: cand-001; kind=expand family=attention bias=balanced sig=s:rf fp=19039ac31e1e14e2
- Ops: [{"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22100837 M2_D1=1284834 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1192.1825329196229 eligible=True
  - M1_P1: hbm_r=151.3MB top busy: sm0.issue_wait=21833602, sm0.rf_read=9053608, sm0.rf_write=6466184
  - M2_D1: hbm_r=15.8MB top busy: sm0.issue_wait=1263560, sm0.rf_write=525632, sm0.rf_read=282912
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:20:23 — cand-015: novelty: untried structure nkm_pf_rf on cand-009
- Parent: cand-009; kind=novelty family=memory bias=balanced sig=nkm+pf+s:rf fp=46b7b004577c04fd
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_prefetch", "enabled": 1}, {"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22075937 M2_D1=1424034 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1133.0546914360318 eligible=True
  - M1_P1: hbm_r=151.5MB top busy: sm0.issue_wait=21809314, sm0.rf_read=9028648, sm0.rf_write=6471560
  - M2_D1: hbm_r=18.1MB top busy: sm0.issue_wait=1401672, sm0.rf_write=598336, sm0.dma=285184
- Decision: status=dominated failure_class=none

### round 3 summary: best=cand-014 score=1192.183 gain=1.0118 plateau=0 explore=0.34


## 2026-09-28 12:22:53 — cand-016: exploit cand-014: tiles (8, 16, 48)->(8, 16, 32)
- Parent: cand-014; kind=exploit family=attention bias=balanced sig=nkm+s:rf fp=3ed983a6780dce16
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 32}, {"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=24079193 M2_D1=1298658 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1136.0626036140557 eligible=True
  - M1_P1: hbm_r=151.3MB top busy: sm0.issue_wait=23762038, sm0.rf_read=9215848, sm0.rf_write=6790664
  - M2_D1: hbm_r=15.8MB top busy: sm0.issue_wait=1272392, sm0.rf_write=528960, sm0.rf_read=284576
- Decision: status=dominated failure_class=none


## 2026-09-28 12:24:45 — cand-017: exploit cand-014: tiles (8, 16, 48)->(16, 16, 48)
- Parent: cand-014; kind=exploit family=attention bias=balanced sig=nkm+s:rf fp=b3c122bf8a968893
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 16, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22255433 M2_D1=1284834 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1188.0346056227518 eligible=True
  - M1_P1: hbm_r=151.3MB top busy: sm0.issue_wait=22048870, sm0.rf_read=9053608, sm0.rf_write=6466184
  - M2_D1: hbm_r=15.8MB top busy: sm0.issue_wait=1263560, sm0.rf_write=525632, sm0.rf_read=282912
- Decision: status=dominated failure_class=none


## 2026-09-28 12:28:36 — cand-018: exploit cand-014: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-014; kind=exploit family=attention bias=balanced sig=nkm+s:rf fp=c16f062eaf2ae286
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27716789 M2_D1=1707234 cycles; area=5.59764068864mm2 peakW=1.9026077852199403; score=923.5334757991376 eligible=True
  - M1_P1: hbm_r=265.7MB top busy: sm0.issue_wait=27299602, sm0.rf_read=9061192, sm0.rf_write=8862344
  - M2_D1: hbm_r=29.1MB top busy: sm0.issue_wait=1670216, sm0.rf_write=550208, sm0.dma=459392
- Decision: status=dominated failure_class=none


## 2026-09-28 12:30:43 — cand-019: expand cand-009: keep attention scores in RF
- Parent: cand-009; kind=expand family=attention bias=balanced sig=s:rf fp=2e4feb59f56b0539
- Ops: [{"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:32:34 — cand-020: expand cand-010: keep attention scores in RF
- Parent: cand-010; kind=expand family=attention bias=balanced sig=s:rf fp=6de9256553383d73
- Ops: [{"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22109573 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1193.4042900831387 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21905350, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 12:36:49 — cand-021: novelty: random mnk tiles=(8, 16, 24)
- Parent: cand-014; kind=novelty family=memory bias=balanced sig=fuse+mnk+nopf+s:hbm fp=5bd0cc22e122250c
- Ops: [{"op": "gemm_order", "order": "mnk"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 24}, {"op": "gemm_prefetch", "enabled": 0}, {"op": "scores_residency", "space": "hbm"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=33021791 M2_D1=1339272 cycles; area=5.59764068864mm2 peakW=1.7715864571802942; score=955.2915117902414 eligible=True
  - M1_P1: hbm_r=307.5MB top busy: sm0.issue_wait=32529664, sm0.rf_write=11334824, sm0.rf_read=9425128
  - M2_D1: hbm_r=16.3MB top busy: sm0.issue_wait=1303342, sm0.rf_write=535744, sm0.rf_read=288032
- Decision: status=dominated failure_class=none

### round 4 summary: best=cand-019 score=1197.599 gain=1.0045 plateau=1 explore=0.34


## 2026-09-28 12:39:35 — cand-022: exploit cand-019: tiles (8, 16, 48)->(8, 16, 24)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=ac572e82f17a050f
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 24}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=24026741 M2_D1=1323298 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1126.6638611979947 eligible=True
  - M1_P1: hbm_r=174.1MB top busy: sm0.issue_wait=23644726, sm0.rf_read=9409288, sm0.rf_write=7177544
  - M2_D1: hbm_r=16.0MB top busy: sm0.issue_wait=1288648, sm0.rf_write=533568, sm0.rf_read=286880
- Decision: status=dominated failure_class=none


## 2026-09-28 12:41:59 — cand-023: exploit cand-019: tiles (8, 16, 48)->(8, 16, 32)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=4db087d2863f570d
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 32}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=23933333 M2_D1=1295522 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1140.8975242069364 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=23618518, sm0.rf_read=9190888, sm0.rf_write=6740744
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1269320, sm0.rf_write=527936, sm0.rf_read=284064
- Decision: status=dominated failure_class=none


## 2026-09-28 12:44:42 — cand-024: exploit cand-019: tiles (8, 16, 48)->(4, 16, 48)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=52b8df7fb6990c4f
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 4, "n_tile": 16, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22281209 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1188.7989042667773 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21894970, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=dominated failure_class=none


## 2026-09-28 12:46:39 — cand-025: expand cand-020: double-buffer B tiles
- Parent: cand-020; kind=expand family=memory bias=balanced sig=pf fp=30fd2cc2cedd6be3
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22182149 M2_D1=1424034 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1130.3388047148876 eligible=True
  - M1_P1: hbm_r=151.5MB top busy: sm0.issue_wait=21976198, sm0.rf_read=9028648, sm0.rf_write=6471560
  - M2_D1: hbm_r=18.1MB top busy: sm0.issue_wait=1401672, sm0.rf_write=598336, sm0.dma=285184
- Decision: status=dominated failure_class=none


## 2026-09-28 12:49:01 — cand-026: expand cand-014: double-buffer B tiles
- Parent: cand-014; kind=expand family=memory bias=balanced sig=pf fp=8a7c84d9a7be1c97
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22221797 M2_D1=1427170 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1128.088530457202 eligible=True
  - M1_P1: hbm_r=153.1MB top busy: sm0.issue_wait=21952834, sm0.rf_read=9053608, sm0.rf_write=6521480
  - M2_D1: hbm_r=18.1MB top busy: sm0.issue_wait=1404744, sm0.rf_write=599360, sm0.dma=286208
- Decision: status=dominated failure_class=none


## 2026-09-28 12:51:41 — cand-027: novelty: random mnk tiles=(8, 64, 16)
- Parent: cand-019; kind=novelty family=memory bias=balanced sig=mnk+nofuse+nopf+s:hbm fp=d366bee507975620
- Ops: [{"op": "gemm_order", "order": "mnk"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 64, "k_chunk": 16}, {"op": "gemm_prefetch", "enabled": 0}, {"op": "scores_residency", "space": "hbm"}, {"op": "fuse_bias_gelu", "enabled": 0}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27592005 M2_D1=1237679 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1087.11373917888 eligible=True
  - M1_P1: hbm_r=227.2MB top busy: sm0.issue_wait=27324338, sm0.rf_write=10348904, sm0.rf_read=9830728
  - M2_D1: hbm_r=15.4MB top busy: sm0.issue_wait=1221077, sm0.rf_write=524992, sm0.rf_read=291872
- Decision: status=dominated failure_class=none

### round 5 summary: best=cand-019 score=1197.599 gain=1.0000 plateau=2 explore=0.34


## 2026-09-28 12:55:41 — cand-028: exploit cand-019: tiles (8, 16, 48)->(2, 16, 48)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=7c136c16167cb951
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 2, "n_tile": 16, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=26103797 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1098.3125106615971 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=25474870, sm0.rf_read=13747240, sm0.tc=10666656
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=dominated failure_class=none


## 2026-09-28 12:57:39 — cand-029: exploit cand-019: tiles (8, 16, 48)->(8, 16, 64)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=7ed8a97e522117f4
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 16, "k_chunk": 64}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22850897 M2_D1=1293474 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1168.5307558154386 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=22628242, sm0.rf_read=8891368, sm0.rf_write=6141704
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1276488, sm0.rf_write=521792, sm0.rf_read=280992
- Decision: status=dominated failure_class=none


## 2026-09-28 13:01:45 — cand-030: exploit cand-019: tiles (8, 16, 48)->(8, 8, 48)
- Parent: cand-019; kind=exploit family=attention bias=balanced sig=fuse+nkm+s:rf fp=703e7311b450d7db
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27570929 M2_D1=1704098 cycles; area=5.59764068864mm2 peakW=1.9026077852199403; score=926.8247918622404 eligible=True
  - M1_P1: hbm_r=264.1MB top busy: sm0.issue_wait=27156082, sm0.rf_read=9036232, sm0.rf_write=8812424
  - M2_D1: hbm_r=29.1MB top busy: sm0.issue_wait=1667144, sm0.rf_write=549184, sm0.dma=458368
- Decision: status=dominated failure_class=none


## 2026-09-28 13:04:31 — cand-031: expand cand-027: keep attention scores in RF
- Parent: cand-027; kind=expand family=attention bias=balanced sig=s:rf fp=7c58a9e1521ae859
- Ops: [{"op": "scores_residency", "space": "rf"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27321035 M2_D1=1213810 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1103.1808039751736 eligible=True
  - M1_P1: hbm_r=225.9MB top busy: sm0.issue_wait=27075448, sm0.rf_write=10320392, sm0.rf_read=9814888
  - M2_D1: hbm_r=15.2MB top busy: sm0.issue_wait=1198488, sm0.rf_write=522816, sm0.rf_read=290720
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:07:24 — cand-032: expand cand-024: double-buffer B tiles
- Parent: cand-024; kind=expand family=memory bias=balanced sig=pf fp=7044d4568cdc1820
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22431545 M2_D1=1424034 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1124.0376356259756 eligible=True
  - M1_P1: hbm_r=151.5MB top busy: sm0.issue_wait=22043578, sm0.rf_read=9028648, sm0.rf_write=6471560
  - M2_D1: hbm_r=18.1MB top busy: sm0.issue_wait=1401672, sm0.rf_write=598336, sm0.dma=285184
- Decision: status=dominated failure_class=none


## 2026-09-28 13:25:24 — cand-033: novelty: random nkm tiles=(1, 8, 32)
- Parent: cand-019; kind=novelty family=memory bias=balanced sig=fuse+nkm+pf+s:rf fp=f0ee74fb076a59aa
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 1, "n_tile": 8, "k_chunk": 32}, {"op": "gemm_prefetch", "enabled": 1}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=50660093 M2_D1=1926050 cycles; area=5.59764068864mm2 peakW=1.8243357268374292; score=643.1375334157768 eligible=True
  - M1_P1: hbm_r=271.2MB top busy: sm0.issue_wait=47884030, sm0.tc=41225760, sm0.rf_read=24001864
  - M2_D1: hbm_r=33.8MB top busy: sm0.issue_wait=1876808, sm0.rf_write=626240, sm0.dma=532096
- Decision: status=dominated failure_class=none

### round 6 summary: best=cand-019 score=1197.599 gain=1.0000 plateau=3 explore=0.44

### autonomous batch summary (rounds 1-6, 2026-09-28, seed=7)
- 29 evaluated + 21 dry-run nodes; 1 pre-eval constraint prune in dry-run path; no incorrect candidates.
- Best: cand-019 score=1197.60 (P1=21954977, D1=1281698) = nkm B-stationary + scores in RF + fused bias+gelu.
- Effective ops: scores_residency(rf) (+1.6% on nkm, +0.45% combined w/ fuse); fuse_bias_gelu (+0.45%); nkm order (+17% from phase-0 manual step).
- Negative ops: prefetch (no dma_depth -> redundant LD, -3~-5%); n_tile=8 (B traffic x2); m_tile<=2 (instr explosion); mnk anywhere.
- Bottleneck unchanged: P1 sm0.issue_wait ~98.8% of cycles — serial single-WG chain saturated; dataflow-ops exhausted -> unlock phase 2 (multi-WG / hardware).
- Head: cand-019 (kept). Pareto: {cand-019 (P1-min), cand-031 (D1-min 1213810)}.

## 2026-09-28 13:36:33 — drun-022: exploit cand-019: hw rf_ports 4R2W->8R4W
- Parent: cand-019; kind=exploit family=hardware sig=hw:rf_ports=8R4W
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 13:36:33 — drun-023: exploit cand-019: hw sm_count 8->12
- Parent: cand-019; kind=exploit family=hardware sig=hw:sm_count=12
- Ops: [{"op": "hw_set", "path": "sm_count", "value": 12}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 13:36:33 — drun-024: expand cand-031: fuse bias+gelu epilogue
- Parent: cand-031; kind=expand family=fusion sig=fuse
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 13:36:33 — drun-025: expand cand-020: widen RF ports to relieve MMA read-port bound [rf_ports=8R4W]
- Parent: cand-020; kind=expand family=hardware sig=hw:rf_ports=8R4W
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 13:36:33 — drun-026: expand cand-014: wider SM NoC link [sm_noc_bytes_per_cycle=128]
- Parent: cand-014; kind=expand family=memory sig=hw:sm_noc_bytes_per_cycle=128
- Ops: [{"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 128}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

## 2026-09-28 13:36:33 — drun-027: novelty: random nkm tiles=(16, 8, 64)
- Parent: cand-019; kind=novelty family=memory sig=hw:sm_count=32+nkm+nofuse+nopf+s:hbm
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 8, "k_chunk": 64}, {"op": "gemm_prefetch", "enabled": 0}, {"op": "scores_residency", "space": "hbm"}, {"op": "fuse_bias_gelu", "enabled": 0}, {"op": "hw_set", "path": "sm_count", "value": 32}]
- Result: PRUNED (dry_run) mechanism test
- Decision: pruned before/at evaluation stage

### round 7 summary: best=cand-019 score=1197.599 gain=1.0000 plateau=4 explore=0.54


## 2026-09-28 13:38:46 — cand-034: exploit cand-019: hw rf_ports 4R2W->8R4W
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:rf_ports=8R4W fp=bc9f80dc971510a2
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:40:58 — cand-035: exploit cand-019: hw sm_count 8->12
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:sm_count=12 fp=fe9942186646c82e
- Ops: [{"op": "hw_set", "path": "sm_count", "value": 12}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:43:00 — cand-036: expand cand-031: fuse bias+gelu epilogue
- Parent: cand-031; kind=expand family=fusion bias=balanced sig=fuse fp=062742f6b8fbbde1
- Ops: [{"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27175175 M2_D1=1210674 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1107.569130887403 eligible=True
  - M1_P1: hbm_r=224.3MB top busy: sm0.issue_wait=26931928, sm0.rf_write=10270472, sm0.rf_read=9789928
  - M2_D1: hbm_r=15.1MB top busy: sm0.issue_wait=1195416, sm0.rf_write=521792, sm0.rf_read=290208
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:44:22 — cand-037: expand cand-020: widen RF ports to relieve MMA read-port bound [rf_ports=8R4W]
- Parent: cand-020; kind=expand family=hardware bias=balanced sig=hw:rf_ports=8R4W fp=32db6f8fc708a8cc
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22109573 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1193.4042900831387 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21905350, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:45:55 — cand-038: expand cand-014: wider SM NoC link [sm_noc_bytes_per_cycle=128]
- Parent: cand-014; kind=expand family=memory bias=balanced sig=hw:sm_noc_bytes_per_cycle=128 fp=790882511f0a7043
- Ops: [{"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 128}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22100837 M2_D1=1284834 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1192.1825329196229 eligible=True
  - M1_P1: hbm_r=151.3MB top busy: sm0.issue_wait=21833602, sm0.rf_read=9053608, sm0.rf_write=6466184
  - M2_D1: hbm_r=15.8MB top busy: sm0.issue_wait=1263560, sm0.rf_write=525632, sm0.rf_read=282912
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:49:21 — cand-039: novelty: random nkm tiles=(16, 8, 64)
- Parent: cand-019; kind=novelty family=memory bias=balanced sig=hw:sm_count=32+nkm+nofuse+nopf+s:hbm fp=1560e00fa1dfaaf4
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 8, "k_chunk": 64}, {"op": "gemm_prefetch", "enabled": 0}, {"op": "scores_residency", "space": "hbm"}, {"op": "fuse_bias_gelu", "enabled": 0}, {"op": "hw_set", "path": "sm_count", "value": 32}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27433687 M2_D1=1751464 cycles; area=5.59764068864mm2 peakW=2.0154056390369264; score=916.4904172822608 eligible=True
  - M1_P1: hbm_r=267.3MB top busy: sm0.issue_wait=27166452, sm0.rf_read=8937640, sm0.rf_write=8616296
  - M2_D1: hbm_r=29.5MB top busy: sm0.issue_wait=1721614, sm0.rf_write=549568, sm0.dma=468218
- Decision: status=dominated failure_class=none

### round 7 summary: best=cand-019 score=1197.599 gain=1.0000 plateau=4 explore=0.54


## 2026-09-28 13:51:34 — cand-040: exploit cand-019: hw reduction_units 0->1
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:reduction_units=1 fp=94bc9c7b5d9d25cf
- Ops: [{"op": "hw_set", "path": "reduction_units", "value": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:53:52 — cand-041: exploit cand-019: hw hbm_channels 4->2
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:hbm_channels=2 fp=4b40a64f4b60265f
- Ops: [{"op": "hw_set", "path": "hbm_channels", "value": 2}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 13:56:06 — cand-042: exploit cand-019: hw noc_bytes_per_cycle 256->512
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:noc_bytes_per_cycle=512 fp=765f765265610450
- Ops: [{"op": "hw_set", "path": "noc_bytes_per_cycle", "value": 512}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.59764068864mm2 peakW=1.7715864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none

### harness fix 2026-09-28: hw_set materialization bug
- Bug: engine._materialize copied parent hardware.json instead of writing the
  mutated prop['hardware']; 9 phase-2 hw candidates (cand-034..043) therefore
  re-evaluated their parent's design point. Fixed; those nodes marked
  failure_class=harness_noop, fingerprints released for re-proposal.
- Evidence of the bug: identical cycles/area/score to parents. Discarded from
  archives; the underlying design points remain untested.


## 2026-09-28 14:01:05 — cand-044: exploit cand-019: hw reduction_units 0->1
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:reduction_units=1 fp=94bc9c7b5d9d25cf
- Ops: [{"op": "hw_set", "path": "reduction_units", "value": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21954977 M2_D1=1281698 cycles; area=5.8376406886400005mm2 peakW=1.7775864571831552; score=1197.5985984332774 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21690082, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1260488, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none


## 2026-09-28 14:04:05 — cand-045: exploit cand-019: hw vector_lanes 16->8
- Parent: cand-019; kind=exploit family=hardware bias=balanced sig=hw:vector_lanes=8 fp=1055104233338e90
- Ops: [{"op": "hw_set", "path": "vector_lanes", "value": 8}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=22265321 M2_D1=1288802 cycles; area=5.50164068864mm2 peakW=1.7691864571831553; score=1185.9408933569387 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=22000426, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1267592, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=dominated failure_class=none


## 2026-09-28 14:07:29 — cand-046: expand cand-036: reduction unit for LN/softmax reduces [reduction_units=1]
- Parent: cand-036; kind=expand family=compute bias=balanced sig=hw:reduction_units=1 fp=195ec88606acde60
- Ops: [{"op": "hw_set", "path": "reduction_units", "value": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=27175175 M2_D1=1210674 cycles; area=5.8376406886400005mm2 peakW=1.7775864571831552; score=1107.569130887403 eligible=True
  - M1_P1: hbm_r=224.3MB top busy: sm0.issue_wait=26931928, sm0.rf_write=10270472, sm0.rf_read=9789928
  - M2_D1: hbm_r=15.1MB top busy: sm0.issue_wait=1195416, sm0.rf_write=521792, sm0.rf_read=290208
- Decision: status=evaluated failure_class=none


## 2026-09-28 14:09:36 — cand-047: expand cand-020: widen RF ports to relieve MMA read-port bound [rf_ports=8R4W]
- Parent: cand-020; kind=expand family=hardware bias=balanced sig=hw:rf_ports=8R4W fp=32db6f8fc708a8cc
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=12917099 M2_D1=888914 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1874.814304538389 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12712892, sm0.tc=5571744, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867736, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none

## 2026-09-28 14:11:47 — cand-048: expand cand-014: dma_depth=2 to make prefetch actually overlap [dma_depth=2]
- Parent: cand-014; kind=expand family=memory sig=hw:dma_depth=2
- Ops: [{"op": "hw_set", "path": "dma_depth", "value": 2}]
- Result: PRUNED (generate_error) grade produced no report (rc=1):

Traceback (most recent call last):
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py", line 6, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-code
- Decision: pruned before/at evaluation stage


## 2026-09-28 14:15:20 — cand-049: novelty: random nkm tiles=(32, 8, 32)
- Parent: cand-019; kind=novelty family=memory bias=balanced sig=fuse+hw:hbm_channels=1+nkm+pf+s:rf fp=b3503608c839b8de
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 32, "n_tile": 8, "k_chunk": 32}, {"op": "gemm_prefetch", "enabled": 1}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}, {"op": "hw_set", "path": "hbm_channels", "value": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=28400621 M2_D1=2250698 cycles; area=5.44764068864mm2 peakW=1.5061085769694176; score=794.5991478350456 eligible=True
  - M1_P1: hbm_r=271.2MB top busy: sm0.issue_wait=28124374, sm0.rf_write=9247496, sm0.rf_read=9200968
  - M2_D1: hbm_r=33.8MB top busy: sm0.issue_wait=2201424, shared.hbm_channel_0=1064192, sm0.rf_write=626240
- Decision: status=dominated failure_class=none

### round 9 summary: best=cand-047 score=1874.814 gain=1.5655 plateau=0 explore=0.54


## 2026-09-28 14:19:55 — cand-050: exploit cand-047: tiles (16, 16, 48)->(16, 8, 48)
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=fuse+nkm+s:rf fp=c0c82030ca9ad779
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 8, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=18493067 M2_D1=1297490 cycles; area=6.671639413760001mm2 peakW=2.6023709853516293; score=1296.9228571905599 eligible=True
  - M1_P1: hbm_r=264.1MB top busy: sm0.issue_wait=18199580, sm0.tc=5933856, sm0.rf_read=4533992
  - M2_D1: hbm_r=29.1MB top busy: sm0.issue_wait=1260568, sm0.dma=458368, shared.noc=458368
- Decision: status=dominated failure_class=none


## 2026-09-28 14:22:03 — cand-051: exploit cand-047: tiles (16, 16, 48)->(16, 16, 24)
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=fuse+nkm+s:rf fp=aa0e97af5d268445
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 16, "k_chunk": 24}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=14111249 M2_D1=921682 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1761.5596139891823 eligible=True
  - M1_P1: hbm_r=174.1MB top busy: sm0.issue_wait=13836770, sm0.tc=5630304, sm0.rf_read=4720520
  - M2_D1: hbm_r=16.0MB top busy: sm0.issue_wait=887064, sm0.rf_write=271616, sm0.dma=252800
- Decision: status=dominated failure_class=none


## 2026-09-28 14:24:30 — cand-052: expand cand-019: widen RF ports to relieve MMA read-port bound [rf_ports=8R4W]
- Parent: cand-019; kind=expand family=hardware bias=balanced sig=hw:rf_ports=8R4W fp=bc9f80dc971510a2
- Ops: [{"op": "hw_set", "path": "rf_ports", "value": "8R4W"}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13116989 M2_D1=888914 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1860.4743007003888 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12852110, sm0.tc=5625504, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867736, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none


## 2026-09-28 14:26:28 — cand-053: expand cand-044: dma_depth=2 to make prefetch actually overlap [dma_depth=2]
- Parent: cand-044; kind=expand family=memory bias=balanced sig=hw:dma_depth=2 fp=d0ae7c9cd4adadcd
- Ops: [{"op": "hw_set", "path": "dma_depth", "value": 2}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=21298223 M2_D1=1222470 cycles; area=5.957640688640001mm2 peakW=1.9649001931822625; score=1245.0300554926546 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=21033328, sm0.rf_read=9028648, sm0.rf_write=6416264
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1201260, sm0.rf_write=524608, sm0.rf_read=282400
- Decision: status=evaluated failure_class=none

## 2026-09-28 14:26:28 — cand-054: expand cand-020: second DMA engine [dma_engines=2]
- Parent: cand-020; kind=expand family=memory sig=hw:dma_engines=2
- Ops: [{"op": "hw_set", "path": "dma_engines", "value": 2}]
- Result: PRUNED (generate_error) [Errno 32] Broken pipe
- Decision: pruned before/at evaluation stage

## 2026-09-28 14:26:28 — cand-055: novelty: random mnk tiles=(8, 8, 48)
- Parent: cand-047; kind=novelty family=memory sig=mnk+nofuse+nopf+s:rf
- Ops: [{"op": "gemm_order", "order": "mnk"}, {"op": "gemm_tiles", "m_tile": 8, "n_tile": 8, "k_chunk": 48}, {"op": "gemm_prefetch", "enabled": 0}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 0}]
- Result: PRUNED (generate_error) [Errno 32] Broken pipe
- Decision: pruned before/at evaluation stage

### round 10 summary: best=cand-047 score=1874.814 gain=1.0000 plateau=1 explore=0.54
