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


## 2026-09-28 15:15:35 — cand-056: exploit cand-047: tiles (16, 16, 48)->(4, 16, 48)
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=fuse+nkm+s:rf fp=5581718e90a6e16f
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 4, "n_tile": 16, "k_chunk": 48}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13475549 M2_D1=888914 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1835.555492983374 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=13089326, sm0.tc=5733024, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867736, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=dominated failure_class=none


## 2026-09-28 15:16:45 — cand-057: exploit cand-047: hw sm_noc_bytes_per_cycle 64->32
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=hw:sm_noc_bytes_per_cycle=32 fp=0bf5c7085bee0f67
- Ops: [{"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 32}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13787089 M2_D1=1053450 cycles; area=6.62043941376mm2 peakW=1.7199026653547953; score=1666.9684198231182 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=13582856, sm0.tc=5571744, shared.noc=4931968
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1032232, shared.noc=496640, sm0.noc_in=491392
- Decision: status=dominated failure_class=none


## 2026-09-28 15:18:48 — cand-058: exploit cand-047: tiles (16, 16, 48)->(16, 16, 32)
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=fuse+nkm+s:rf fp=124a216bdb041a89
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 16, "k_chunk": 32}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13422437 M2_D1=903378 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1824.400503955544 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=13188278, sm0.tc=5596704, sm0.rf_read=4615496
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=877208, sm0.rf_write=264320, sm0.dma=248320
- Decision: status=dominated failure_class=none


## 2026-09-28 15:20:32 — cand-059: expand cand-052: wider SM NoC link [sm_noc_bytes_per_cycle=128]
- Parent: cand-052; kind=expand family=memory bias=balanced sig=hw:sm_noc_bytes_per_cycle=128 fp=f3c21a8095cf8e04
- Ops: [{"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 128}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13109117 M2_D1=888634 cycles; area=6.774039413760001mm2 peakW=2.3305919133597084; score=1861.325996168528 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12844238, sm0.tc=5625504, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867456, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none


## 2026-09-28 15:22:50 — cand-060: expand cand-051: double-buffer B tiles
- Parent: cand-051; kind=expand family=memory bias=balanced sig=pf fp=3c52c7255de1758c
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=14202833 M2_D1=953682 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1726.161168219366 eligible=True
  - M1_P1: hbm_r=175.8MB top busy: sm0.issue_wait=13926626, sm0.tc=5630304, sm0.rf_read=4720520
  - M2_D1: hbm_r=16.6MB top busy: sm0.issue_wait=917912, sm0.rf_write=280832, sm0.dma=262016
- Decision: status=dominated failure_class=none


## 2026-09-28 15:26:13 — cand-061: expand cand-050: double HBM channels [hbm_channels=8]
- Parent: cand-050; kind=expand family=memory bias=balanced sig=hw:hbm_channels=8 fp=88a1c49c01abc017
- Ops: [{"op": "hw_set", "path": "hbm_channels", "value": 8}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=17370443 M2_D1=1297490 cycles; area=6.871639413760001mm2 peakW=2.6128686893463797; score=1338.1757908815193 eligible=True
  - M1_P1: hbm_r=264.1MB top busy: sm0.issue_wait=17076956, sm0.tc=5933856, sm0.rf_read=4533992
  - M2_D1: hbm_r=29.1MB top busy: sm0.issue_wait=1260568, sm0.dma=458368, shared.noc=458368
- Decision: status=evaluated failure_class=none

### round 11 summary: best=cand-047 score=1874.814 gain=1.0000 plateau=2 explore=0.54


## 2026-09-28 15:28:32 — cand-062: exploit cand-047: tiles (16, 16, 48)->(16, 16, 64)
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=fuse+nkm+s:rf fp=cd40d6a34502fa0f
- Ops: [{"op": "gemm_order", "order": "nkm"}, {"op": "gemm_tiles", "m_tile": 16, "n_tile": 16, "k_chunk": 64}, {"op": "scores_residency", "space": "rf"}, {"op": "fuse_bias_gelu", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13224203 M2_D1=901330 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1840.1107341264512 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=13045340, sm0.tc=5550624, sm0.rf_read=4463432
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=884376, sm0.rf_write=261248, sm0.dma=248320
- Decision: status=dominated failure_class=none


## 2026-09-28 15:31:44 — cand-063: exploit cand-047: hw noc_bytes_per_cycle 256->128
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=hw:noc_bytes_per_cycle=128 fp=ed7d495f64a422b4
- Ops: [{"op": "hw_set", "path": "noc_bytes_per_cycle", "value": 128}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=12917099 M2_D1=888914 cycles; area=6.607639413760001mm2 peakW=2.322768657411642; score=1874.814304538389 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12712892, sm0.tc=5571744, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867736, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none


## 2026-09-28 15:35:11 — cand-064: exploit cand-047: hw tc_count 2->1
- Parent: cand-047; kind=exploit family=hardware bias=balanced sig=hw:tc_count=1 fp=45ecec5211e523f3
- Ops: [{"op": "hw_set", "path": "tc_count", "value": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=15038603 M2_D1=1050082 cycles; area=5.8362054937600005mm2 peakW=2.3070954813190667; score=1598.6569175780114 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=14834396, sm0.tc=10993920, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=1028904, sm0.tc=473216, sm0.rf_write=262656
- Decision: status=dominated failure_class=none


## 2026-09-28 15:37:42 — cand-065: expand cand-059: dma_depth=2 to make prefetch actually overlap [dma_depth=2]
- Parent: cand-059; kind=expand family=memory bias=balanced sig=hw:dma_depth=2 fp=d084471571e40788
- Ops: [{"op": "hw_set", "path": "dma_depth", "value": 2}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=12522857 M2_D1=833822 cycles; area=6.89403941376mm2 peakW=2.629855153361387; score=1965.9942357374484 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12257978, sm0.tc=5625504, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=812644, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none


## 2026-09-28 15:39:41 — cand-066: expand cand-052: double-buffer B tiles
- Parent: cand-052; kind=expand family=memory bias=balanced sig=pf fp=5aa16f9bbbaaa000
- Ops: [{"op": "gemm_prefetch", "enabled": 1}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=13216349 M2_D1=993746 cycles; area=6.671639413760001mm2 peakW=2.3279813293190665; score=1752.9808817511307 eligible=True
  - M1_P1: hbm_r=151.5MB top busy: sm0.issue_wait=12949742, sm0.tc=5625504, sm0.rf_read=4533128
  - M2_D1: hbm_r=18.1MB top busy: sm0.issue_wait=971416, sm0.rf_write=299520, sm0.dma=285184
- Decision: status=dominated failure_class=none


## 2026-09-28 15:42:09 — cand-067: expand cand-056: double HBM channels [hbm_channels=8]
- Parent: cand-056; kind=expand family=memory bias=balanced sig=hw:hbm_channels=8 fp=87c855f7a1b7380e
- Ops: [{"op": "hw_set", "path": "hbm_channels", "value": 8}]
- Verify: functional={'M1_P1': True, 'M2_D1': True}; M1_P1=12644297 M2_D1=888914 cycles; area=6.871639413760001mm2 peakW=2.3329813293190664; score=1894.931033551575 eligible=True
  - M1_P1: hbm_r=149.7MB top busy: sm0.issue_wait=12258074, sm0.tc=5733024, sm0.rf_read=4533128
  - M2_D1: hbm_r=15.7MB top busy: sm0.issue_wait=867736, sm0.rf_write=262656, sm0.dma=248320
- Decision: status=evaluated failure_class=none

### round 12 summary: best=cand-065 score=1965.994 gain=1.0486 plateau=0 explore=0.54

## 2026-09-28 15:58:25 — cand-068: P1 batch workgroups on SM0/SM1
- Change: schedule.p1_batch_partition serial→batch; D1 and hardware inherited from cand-065.
- Verify: controller run cand-068 --seed 7 → check produced no report; interrupted after ~11m25s at codesign/challenge/hbm_race.py:462 with ~48 GiB physical footprint (sample: project/experiments/cand-068/check-process-sample.txt).
- Result: correctness unknown; no cycles, score, area/power result.
- Decision: reject this emitted structure as unverified under practical evaluator resources; preserve candidate and investigate lower-cost parallel partitioning.

## 2026-09-28 15:59:49 — cand-069: P1 prompt-only m_tile 16
- Change: stage-specific GEMM policy; P1 prompt m_tile 8→16, P1 step and D1 retain 8, hardware inherited from cand-065 (dma_depth=2).
- Verify: controller run cand-069 --seed 7 → no check report; evaluator raised AssertionError: DMA slot changed after issue preview (pipeline_events.py:701).
- Result: correctness and score unknown; no valid timing report.
- Decision: reject this combination pending separate run on dma_depth=1; preserve failed candidate.

## 2026-09-28 15:59:49 — cand-070: P1 prompt m_tile 16 with stage-specific GEMM policy on dma_depth 1
- Parent: cand-059; Summary: p1_prompt.m_tile 8->16; P1 step/D1 remain 8; hardware from cand-059
- Inputs: compiler@caef84c6 hardware@9b093b55
- Hypothesis: P1 prompt m_tile 16 with stage-specific GEMM policy on dma_depth 1
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=12909227 (2.401x), M2_D1=888634 (1.465x); score=1875.6812796394786, eligible=True
  - M1_P1: hbm_r=149.7MB hbm_w=8.1MB instr=204143 peakW=2.33; top busy: sm0.issue_wait=12705020, sm0.tc=5571744, sm0.rf_read=4533128, sm0.rf_write=3218344
  - M2_D1: hbm_r=15.7MB hbm_w=0.2MB instr=21018 peakW=1.95; top busy: sm0.issue_wait=867456, sm0.rf_write=262656, sm0.dma=248320, sm0.tc=243264
- Decision: pending keep/revert

## 2026-09-28 16:06:20 — cand-071: D1-only GEMM tile n32 k32 with stage-specific recipe
- Parent: cand-065; Summary: D1 n_tile 16->32 and k_chunk 48->32; P1 unchanged
- Inputs: compiler@caef84c6 hardware@5cb52af0
- Hypothesis: D1-only GEMM tile n32 k32 with stage-specific recipe
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=12522857 (2.475x), M2_D1=808550 (1.610x); score=1996.4823457755592, eligible=True
  - M1_P1: hbm_r=149.7MB hbm_w=8.1MB instr=264815 peakW=2.63; top busy: sm0.issue_wait=12257978, sm0.tc=5625504, sm0.rf_read=4533128, sm0.rf_write=3218344
  - M2_D1: hbm_r=15.3MB hbm_w=0.2MB instr=15642 peakW=2.15; top busy: sm0.issue_wait=792748, sm0.rf_write=258176, sm0.dma=242176, sm0.tc=240448
- Decision: pending keep/revert

### cand-071 verdict: KEPT — new best score 1996.48; D1 stage tile n32/k32 improves decode, P1 unchanged

## 2026-09-28 16:16:46 — cand-072: D1 stage-specific k_chunk 32 to 24 with n_tile 32
- Parent: cand-071; Summary: Only D1 k_chunk 32->24; all other recipe and hardware fields unchanged
- Inputs: compiler@caef84c6 hardware@5cb52af0
- Hypothesis: D1 stage-specific k_chunk 32 to 24 with n_tile 32
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=12522857 (2.475x), M2_D1=801958 (1.624x); score=2004.670977306224, eligible=True
  - M1_P1: hbm_r=149.7MB hbm_w=8.1MB instr=264815 peakW=2.63; top busy: sm0.issue_wait=12257978, sm0.tc=5625504, sm0.rf_read=4533128, sm0.rf_write=3218344
  - M2_D1: hbm_r=15.5MB hbm_w=0.2MB instr=19866 peakW=2.17; top busy: sm0.issue_wait=781932, sm0.rf_write=263232, sm0.dma=244416, sm0.tc=243264
- Decision: pending keep/revert

### cand-072 verdict: KEPT — new best score 2004.67; D1-only k_chunk 24 improves decode

## 2026-09-28 16:26:10 — cand-073: 2 MiB cache reuses weights and activations across decode steps and P1 GEMM tiles
- Parent: cand-072; Summary: hardware.cache_mib 0->2; programs and recipe unchanged
- Inputs: compiler@caef84c6 hardware@b8a39a17
- Hypothesis: 2 MiB cache reuses weights and activations across decode steps and P1 GEMM tiles
- Verify: check PASS (max err 4.72e-06); estimate -> reports/estimate.json; grade -> reports/grade.json
- Result: M1_P1=12347819 (2.510x), M2_D1=776636 (1.677x); score=2051.47741118116, eligible=True
  - M1_P1: hbm_r=47.2MB hbm_w=8.1MB instr=264815 peakW=2.39; top busy: sm0.issue_wait=12082940, sm0.tc=5625504, sm0.rf_read=4533128, sm0.rf_write=3218344
  - M2_D1: hbm_r=2.0MB hbm_w=0.2MB instr=19866 peakW=1.98; top busy: sm0.issue_wait=756610, sm0.rf_write=263232, sm0.dma=244416, sm0.tc=243264
- Decision: pending keep/revert

### cand-073 verdict: KEPT — 2 MiB cache improves both cases; completed during framework diagnosis; no further candidates queued

## 2026-09-28 17:03:07 — cand-074: Reproduce cand-073 with isolated single-grade runner
- Change: parent=cand-073; []
- Evidence: project/experiments/cand-074/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-074/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-074/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-074/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-074/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=125.5738903330639; peak MiB=2246.7520599365234
- Inputs: compiler.py@caef84c6 hardware.json@b8a39a17 recipe.json@70cf2ca5
- Verify: PASS elapsed=125.58s
- Result: M1_P1=12347819 (2.510x), M2_D1=776636 (1.677x); score=2051.477, eligible=True
  - deltas vs head (cand-073): {'M1_P1': 0, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-075: Hardware sensitivity: tc_k_parallel=2 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "tc_k_parallel", "value": 2}]
- Evidence: project/experiments/cand-075/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-075/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-075/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-075/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-075/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=249.77853466698434; peak MiB=2230.2520599365234
- Inputs: compiler.py@caef84c6 hardware.json@a0b215ed recipe.json@70cf2ca5
- Verify: PASS elapsed=249.78s
- Result: M1_P1=13129883 (2.361x), M2_D1=759924 (1.713x); score=2011.199, eligible=True
  - deltas vs head (cand-073): {'M1_P1': 782064, 'M2_D1': -16712}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-076: Hardware sensitivity: tc_k_parallel=4 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "tc_k_parallel", "value": 4}]
- Evidence: project/experiments/cand-076/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-076/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-076/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-076/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-076/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=247.75014808296692; peak MiB=2223.53328704834
- Inputs: compiler.py@caef84c6 hardware.json@a39f6185 recipe.json@70cf2ca5
- Verify: PASS elapsed=247.75s
- Result: M1_P1=12093131 (2.563x), M2_D1=753564 (1.728x); score=2104.462, eligible=True
  - deltas vs head (cand-073): {'M1_P1': -254688, 'M2_D1': -23072}
- Decision: KEPT — new incumbent head (score=2104.462)

## 2026-09-28 17:05:18 — cand-077: Hardware sensitivity: tc_array=8x8 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "tc_array", "value": "8x8"}]
- Evidence: project/experiments/cand-077/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-077/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-077/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-077/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-077/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=246.25728795898613; peak MiB=2235.9083099365234
- Inputs: compiler.py@caef84c6 hardware.json@25e44bf3 recipe.json@70cf2ca5
- Verify: PASS elapsed=246.26s
- Result: M1_P1=12064091 (2.569x), M2_D1=776636 (1.677x); score=2075.461, eligible=True
  - deltas vs head (cand-076): {'M1_P1': -29040, 'M2_D1': 23072}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-078: Hardware sensitivity: tc_array=8x16 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "tc_array", "value": "8x16"}]
- Evidence: project/experiments/cand-078/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-078/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-078/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-078/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-078/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=245.99945983302314; peak MiB=2239.1895599365234
- Inputs: compiler.py@caef84c6 hardware.json@e5568781 recipe.json@70cf2ca5
- Verify: PASS elapsed=246.00s
- Result: M1_P1=11412395 (2.716x), M2_D1=762444 (1.708x); score=2153.666, eligible=True
  - deltas vs head (cand-076): {'M1_P1': -680736, 'M2_D1': 8880}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-079: Hardware sensitivity: tc_count=4 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "tc_count", "value": 4}]
- Evidence: project/experiments/cand-079/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-079/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-079/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-079/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-079/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=237.71395124995615; peak MiB=2229.2676849365234
- Inputs: compiler.py@caef84c6 hardware.json@1c292ac3 recipe.json@70cf2ca5
- Verify: PASS elapsed=237.72s
- Result: M1_P1=13207115 (2.347x), M2_D1=767372 (1.697x); score=1995.555, eligible=True
  - deltas vs head (cand-076): {'M1_P1': 1113984, 'M2_D1': 13808}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-080: Hardware sensitivity: dma_engines=2 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "dma_engines", "value": 2}]
- Evidence: project/experiments/cand-080/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-080/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-080/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-080/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-080/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=218.29056779108942; peak MiB=2202.5801391601562
- Inputs: compiler.py@caef84c6 hardware.json@d8a32c76 recipe.json@70cf2ca5
- Verify: PASS elapsed=218.30s
- Result: M1_P1=12161525 (2.549x), M2_D1=768440 (1.694x); score=2078.125, eligible=True
  - deltas vs head (cand-076): {'M1_P1': 68394, 'M2_D1': 14876}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:05:18 — cand-081: Hardware sensitivity: cache_mib=4 against cand-073
- Change: parent=cand-073; [{"op": "hw_set", "path": "cache_mib", "value": 4}]
- Evidence: project/experiments/cand-081/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-081/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-081/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-081/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-081/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=206.8949515420245; peak MiB=2248.3301849365234
- Inputs: compiler.py@caef84c6 hardware.json@7e797d21 recipe.json@70cf2ca5
- Verify: PASS elapsed=206.90s
- Result: M1_P1=12347819 (2.510x), M2_D1=776636 (1.677x); score=2051.477, eligible=True
  - deltas vs head (cand-076): {'M1_P1': 254688, 'M2_D1': 23072}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:21:31 — cand-082: Compose beneficial control cand-080
- Change: parent=cand-076; [{"op": "hw_set", "path": "dma_engines", "value": 2}]
- Evidence: project/experiments/cand-082/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-082/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-082/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-082/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-082/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=134.01188812498003; peak MiB=2300.767707824707
- Inputs: compiler.py@caef84c6 hardware.json@042d93a1 recipe.json@70cf2ca5
- Verify: PASS elapsed=134.02s
- Result: M1_P1=11929847 (2.598x), M2_D1=744864 (1.748x); score=2131.153, eligible=True
  - deltas vs head (cand-076): {'M1_P1': -163284, 'M2_D1': -8700}
- Decision: KEPT — new incumbent head (score=2131.153)

## 2026-09-28 17:23:51 — cand-083: Compose beneficial control cand-077
- Change: parent=cand-082; [{"op": "hw_set", "path": "tc_array", "value": "8x8"}]
- Evidence: project/experiments/cand-083/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-083/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-083/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-083/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-083/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=174.7084431250114; peak MiB=2181.7676391601562
- Inputs: compiler.py@caef84c6 hardware.json@08e9950e recipe.json@70cf2ca5
- Verify: PASS elapsed=174.71s
- Result: M1_P1=10750199 (2.883x), M2_D1=744864 (1.748x); score=2245.039, eligible=True
  - deltas vs head (cand-082): {'M1_P1': -1179648, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=2245.039)

## 2026-09-28 17:26:51 — cand-084: Stage sensitivity: d1 k_chunk=16
- Change: parent=cand-083; [{"op": "stage_gemm_tiles", "stage": "d1", "k_chunk": 16}]
- Evidence: project/experiments/cand-084/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-084/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-084/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-084/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-084/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=167.24951037496794; peak MiB=2208.75203704834
- Inputs: compiler.py@caef84c6 hardware.json@08e9950e recipe.json@485e65e1
- Verify: PASS elapsed=167.25s
- Result: M1_P1=10750199 (2.883x), M2_D1=771287 (1.688x); score=2206.248, eligible=True
  - deltas vs head (cand-083): {'M1_P1': 0, 'M2_D1': 26423}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:29:47 — cand-085: Archive mechanism expansion: reduction_units=1
- Change: parent=cand-083; [{"op": "hw_set", "path": "reduction_units", "value": 1}]
- Evidence: project/experiments/cand-085/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-085/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-085/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-085/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-085/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=130.74203354201745; peak MiB=2207.45516204834
- Inputs: compiler.py@caef84c6 hardware.json@ffe8ec90 recipe.json@70cf2ca5
- Verify: PASS elapsed=130.75s
- Result: M1_P1=10750199 (2.883x), M2_D1=744864 (1.748x); score=2245.039, eligible=True
  - deltas vs head (cand-083): {'M1_P1': 0, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 17:33 — Search framework repair and campaign summary
- Change: isolated grade-only evaluation; exact cache; full hardware menu; 2/2/2 quotas; stage mutations; bounded processes and single-writer state.
- Verify: `.venv/bin/python -m unittest discover -s project/search/tests -q` -> 39 passed; reports `project/campaigns/repair-20260928-170307/summary.json` and candidate grade reports.
- Result: 12/12 actual calls; selected cand-083, score=2245.038956 (+9.44% vs cand-073); cycles={'M1_P1': 10750199, 'M2_D1': 744864}; area/power in summary.
- Rejected without grade: 8x16 + K=4 requires 30.5297 mm² >24.
- Decision: keep cand-083; root submission snapshots verified byte-for-byte; retain all eligible tradeoffs and worker traces.

### Campaign area/power evidence supplement

| Candidate | Area mm² | P1 peak W | D1 peak W |
|---|---:|---:|---:|
| cand-074 | 7.853101 | 2.386783 | 1.979213 |
| cand-075 | 9.325101 | 2.423583 | 2.041073 |
| cand-076 | 12.269101 | 2.497183 | 2.125514 |
| cand-077 | 9.523969 | 2.428554 | 2.031819 |
| cand-078 | 12.865704 | 2.512098 | 2.157010 |
| cand-079 | 9.523969 | 2.428554 | 2.048956 |
| cand-080 | 8.253101 | 2.770929 | 2.012421 |
| cand-081 | 8.692162 | 2.407759 | 2.000190 |
| cand-082 | 12.669101 | 2.881329 | 2.145594 |
| cand-083 | 18.755969 | 3.033500 | 2.310412 |
| cand-084 | 18.755969 | 3.033500 | 2.151741 |
| cand-085 | 18.995969 | 3.039500 | 2.316412 |

## 2026-09-28 17:58:32 — cand-086: Stage each P1 prompt W1 64-column weight panel in shared memory and reuse A across four subtiles
- Change: parent=cand-083; [{"op": "w1_shared_panel", "enabled": 1}]
- Evidence: project/experiments/cand-086/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-086/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-086/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-086/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-086/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=105.69506037503015; peak MiB=2056.626922607422
- Inputs: compiler.py@5e139b9e hardware.json@08e9950e recipe.json@12e71e12
- Verify: PASS elapsed=105.70s
- Result: M1_P1=11817029 (2.623x), M2_D1=744864 (1.748x); score=2141.302, eligible=True
  - deltas vs head (cand-083): {'M1_P1': 1066830, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-086 verdict: REVERTED — Correct, but P1 rose 10,750,199 to 11,817,029 cycles; D1 unchanged, shared-panel transfer overhead outweighed RF reuse

## 2026-09-28 18:06:35 — cand-087: RF accumulator partitions: P1 prompt M=16, N=32, K=32; preserve B reuse
- Change: parent=cand-083; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 16, "n_tile": 32, "k_chunk": 32}]
- Evidence: project/experiments/cand-087/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-087/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-087/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-087/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-087/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=84.15203645895235; peak MiB=1809.7204895019531
- Inputs: compiler.py@cb8bf1a2 hardware.json@08e9950e recipe.json@fdf6e2dd
- Verify: PASS elapsed=84.15s
- Result: M1_P1=9484805 (3.268x), M2_D1=744864 (1.748x); score=2390.110, eligible=True
  - deltas vs head (cand-083): {'M1_P1': -1265394, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=2390.110)

## 2026-09-28 18:06:35 — cand-088: RF accumulator partitions: P1 prompt M=16, N=64, K=16; preserve B reuse
- Change: parent=cand-083; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 16, "n_tile": 64, "k_chunk": 16}]
- Evidence: project/experiments/cand-088/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-088/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-088/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-088/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-088/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=77.21067199995741; peak MiB=1623.4235000610352
- Inputs: compiler.py@cb8bf1a2 hardware.json@08e9950e recipe.json@670b9fea
- Verify: PASS elapsed=77.21s
- Result: M1_P1=9986555 (3.104x), M2_D1=744864 (1.748x); score=2329.294, eligible=True
  - deltas vs head (cand-087): {'M1_P1': 501750, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 18:06:35 — cand-089: RF accumulator partitions: P1 prompt M=8, N=128, K=8; preserve B reuse
- Change: parent=cand-083; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 8, "n_tile": 128, "k_chunk": 8}]
- Evidence: project/experiments/cand-089/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-089/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-089/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-089/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-089/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=198.10664862510748; peak MiB=178.00059509277344
- Inputs: compiler.py@cb8bf1a2 hardware.json@08e9950e recipe.json@33efd04e
- Verify: FAILED — failure_class=evaluator_error; evidence=pipeline elapsed=198.11s
- Result: status=not_verified, failure_class=evaluator_error
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 18:06:35 — cand-090: RF accumulator partitions: P1 prompt M=8, N=64, K=16; preserve B reuse
- Change: parent=cand-083; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 8, "n_tile": 64, "k_chunk": 16}]
- Evidence: project/experiments/cand-090/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-090/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-090/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-090/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-090/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=93.7008055410115; peak MiB=1679.3766479492188
- Inputs: compiler.py@cb8bf1a2 hardware.json@08e9950e recipe.json@00f6827e
- Verify: PASS elapsed=93.70s
- Result: M1_P1=10354943 (2.993x), M2_D1=744864 (1.748x); score=2287.485, eligible=True
  - deltas vs head (cand-087): {'M1_P1': 870138, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 18:11:24 — cand-091: Area bridge: remove four idle SMs before widening TC
- Change: parent=cand-083; [{"op": "hw_set", "path": "sm_count", "value": 4}]
- Evidence: project/experiments/cand-091/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-091/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-091/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-091/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-091/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=133.4245033749612; peak MiB=2035.0956497192383
- Inputs: compiler.py@cb8bf1a2 hardware.json@74084fc2 recipe.json@70cf2ca5
- Verify: PASS elapsed=133.43s
- Result: M1_P1=10750199 (2.883x), M2_D1=744864 (1.748x); score=2245.039, eligible=True
  - deltas vs head (cand-087): {'M1_P1': 1265394, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 18:13:43 — cand-092: Area bridge successor: wider TC with four SMs
- Change: parent=cand-091; [{"op": "hw_set", "path": "tc_array", "value": "8x16"}]
- Evidence: project/experiments/cand-092/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-092/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-092/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-092/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-092/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=126.49249624996446; peak MiB=2188.7832641601562
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@70cf2ca5
- Verify: PASS elapsed=126.50s
- Result: M1_P1=10127399 (3.061x), M2_D1=738688 (1.763x); score=2322.689, eligible=True
  - deltas vs head (cand-087): {'M1_P1': 642594, 'M2_D1': -6176}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 18:15:57 — cand-093: Compose best RF dataflow with measured area bridge
- Change: parent=cand-087; [{"op": "hw_set", "path": "sm_count", "value": 4}, {"op": "hw_set", "path": "tc_array", "value": "8x16"}]
- Evidence: project/experiments/cand-093/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-093/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-093/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-093/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-093/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=92.48771350004245; peak MiB=1816.5798645019531
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@fdf6e2dd
- Verify: PASS elapsed=92.49s
- Result: M1_P1=8656949 (3.581x), M2_D1=738688 (1.763x); score=2512.220, eligible=True
  - deltas vs head (cand-087): {'M1_P1': -827856, 'M2_D1': -6176}
- Decision: KEPT — new incumbent head (score=2512.220)

## 2026-09-28 18:17:32 — cand-094: Expand decode N tile after prompt/hardware controls
- Change: parent=cand-093; [{"op": "stage_gemm_tiles", "stage": "d1", "n_tile": 64, "k_chunk": 16}]
- Evidence: project/experiments/cand-094/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-094/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-094/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-094/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-094/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=103.43378408299759; peak MiB=1817.8298645019531
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@0278fd0c
- Verify: PASS elapsed=103.44s
- Result: M1_P1=8656949 (3.581x), M2_D1=718132 (1.813x); score=2547.922, eligible=True
  - deltas vs head (cand-093): {'M1_P1': 0, 'M2_D1': -20556}
- Decision: KEPT — new incumbent head (score=2547.922)

## 2026-09-28 18:19:35 — cand-095: Larger M tile reduces repeated B reads
- Change: parent=cand-094; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 32, "n_tile": 32, "k_chunk": 32}]
- Evidence: project/experiments/cand-095/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-095/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-095/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-095/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-095/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=145.9133948329836; peak MiB=1776.0485916137695
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@de007d24
- Verify: PASS elapsed=145.92s
- Result: M1_P1=8522411 (3.637x), M2_D1=718132 (1.813x); score=2567.954, eligible=True
  - deltas vs head (cand-094): {'M1_P1': -134538, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=2567.954)

## 2026-09-28 18:19:35 — cand-096: Full-M tile trades N reuse for B reuse
- Change: parent=cand-094; [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 64, "n_tile": 16, "k_chunk": 16}]
- Evidence: project/experiments/cand-096/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-096/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-096/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-096/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-096/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=167.49343241704628; peak MiB=1890.8924560546875
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@b1ea161c
- Verify: PASS elapsed=167.50s
- Result: M1_P1=9957359 (3.113x), M2_D1=718132 (1.813x); score=2375.726, eligible=True
  - deltas vs head (cand-095): {'M1_P1': 1434948, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 18:19:35 — cand-097: Joint RF expansion: 8 vector partitions allow N128 with K16, avoiding scalar checker
- Change: parent=cand-094; [{"op": "hw_set", "path": "vector_lanes", "value": 8}, {"op": "stage_gemm_tiles", "stage": "p1_prompt", "order": "nkm_lanes", "m_tile": 16, "n_tile": 128, "k_chunk": 16}]
- Evidence: project/experiments/cand-097/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-097/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-097/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-097/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-097/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=82.39935512503143; peak MiB=1463.5327835083008
- Inputs: compiler.py@cb8bf1a2 hardware.json@95d6433f recipe.json@e47e8b42
- Verify: PASS elapsed=82.40s
- Result: M1_P1=9043631 (3.427x), M2_D1=725840 (1.794x); score=2479.584, eligible=True
  - deltas vs head (cand-095): {'M1_P1': 521220, 'M2_D1': 7708}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### 2026-09-28 18:26 — structural campaign summary
- Decision: keep cand-095 at 2567.954052; 10 eligible passes / 11 launches.
- cand-089: stopped scalar K<16 checker path based on source inspection and CPU sampling; correctness remains unverified. Original runtime report preserved.
- Evidence: project/campaigns/structural-20260928-180635/RESULTS.md and space-audit.json; structural-refine campaign summary.
- Policy: promote by eligible composite score; permit per-case tradeoffs. Next: multi-SM prototypes and RF-resident operator chains; see project/search/STRATEGY_NEXT.md.

## 2026-09-28 18:38 — Gemini multi-SM isolated-kernel experiments
- Change: independent GEMM columns split across 1/2/4 SMs; tiny 2-WG functional/race check, plus worker-only 8-SM timing exploration. No submission candidate changed.
- Verify: pinned Python reruns in project/campaigns/gemini-multism-review/verified-results.json; worker commands, outputs and failed attempts in worker-experiments.json.
- Result: tiny 2-WG=235 cycles; M64/K256/N256: 1/2/4 SM=81325/43460/24809 cycles. These are isolated kernels, not full-grade scores.
- Decision: retain column-split prototype for integration; reject the worker's 8-SM hardware as over-area (30.930 mm²). Full workload correctness and race-check scaling remain unverified.

## 2026-09-28 18:39 — Gemini isolated D1 FFN fusion prototypes
- Change: full activation RF residency and streaming FFN residency, tested as isolated synthetic kernels; no submission triple changed.
- Verify: pinned Python reruns in project/campaigns/gemini-fusion-review/verified-results.json; failed attempts and worker commands retained in worker-experiments.json.
- Result: baseline/A/B=28970/28646/28639 cycles; synthetic functional checks pass. Savings about 1.1% of this FFN kernel; no full-grade score claimed.
- Decision: retain prototypes for later composition; prioritize multi-SM integration. Reject worker claim of universal RF capacity: weight tile is 1024 words and does not fit 32/64 vector-lane hardware.

## 2026-09-28 18:42:31 — cand-098: Fresh full-grade score verification of current worktree submission
- Change: parent=cand-095; Reproduce current hardware/compiler/recipe; bypass evaluation cache
- Evidence: project/experiments/cand-098/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-098/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-098/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-098/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-098/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=85.38173337490298; peak MiB=1773.0954895019531
- Inputs: compiler.py@cb8bf1a2 hardware.json@b92be26a recipe.json@de007d24
- Verify: PASS elapsed=85.38s
- Result: M1_P1=8522411 (3.637x), M2_D1=718132 (1.813x); score=2567.954, eligible=True
  - deltas vs head (cand-095): {'M1_P1': 0, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 19:16:08 — cand-099: P1 end-to-end column GEMMs, head attention and row vector stages on four SMs; D1/hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "column_head", "shards": 4}]
- Evidence: project/experiments/cand-099/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-099/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-099/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-099/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-099/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=69.2560170419747; peak MiB=6214.3955001831055
- Inputs: compiler.py@e9b475ad hardware.json@b92be26a recipe.json@74fdd7d6
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=69.26s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:19:49 — cand-100: Numerical control: identical four-shard full P1 computation serialized in one WG; isolates arithmetic/ABI correctness from multi-WG validation memory
- Change: parent=cand-099; [{"op": "p1_schedule", "kind": "column_head", "shards": 4, "placement": "serial"}]
- Evidence: project/experiments/cand-100/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-100/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-100/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-100/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-100/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=145.83691100007854; peak MiB=1778.7829666137695
- Inputs: compiler.py@d10d296a hardware.json@b92be26a recipe.json@555d37ea
- Verify: PASS elapsed=145.84s
- Result: M1_P1=8544671 (3.628x), M2_D1=718132 (1.813x); score=2564.607, eligible=True
  - deltas vs head (cand-095): {'M1_P1': 22260, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-28 19:26 — P1 multi-SM complete-model attribution and validation diagnosis
- Change: Observe existing cand-098 timing and cand-099 race-validation progress; no evaluator or candidate input changes. New compiler supports column/head/row P1 partitioning; numerical control cand-100 uses the same arithmetic/addresses in a single WG.
- Verify: `.venv/bin/python -m project.campaigns.p1-multism.profile cand-098` -> `project/experiments/cand-098/reports/p1-profile.json`; reproduced P1=8522411. `observe_race.py` under existing runtime -> `project/campaigns/p1-multism/race-observation.json`, `race-progress.jsonl`; memory_limit at 3107.96 MiB, no validation pass claimed.
- Verify: `.venv/bin/python -m project.campaigns.p1-multism.verify_controls` -> `project/campaigns/p1-multism/control-verification.json`: serial/parallel arithmetic and address equivalence, unchanged D1/hardware, default byte identity to cand-098 all passed.
- Result: cand-099 full grade not_verified/memory_limit (6214.40 MiB); cand-100 full grade eligible, P1=8544671 D1=718132 score=2564.60694, area=16.25838297088, power=2.9710608543/2.3857896063 W. Baseline dense GEMM=62.60%, Attention=21.26%, GELU=10.62%; no actual parallel cycles/utilization available.
- Decision: Keep cand-098 as default generated submission; retain opt-in multi-SM source/candidate for follow-up. Do not promote control or interpret validation failure as poor architectural scaling. Frozen checker event-set retention blocks the complete parallel experiment on this 16 GiB host. User success criteria not yet met. See `project/campaigns/p1-multism/REPORT.md`.

### cand-100 verdict: REVERTED — Numerical control passed but score regresses; preserve cand-098 default. Parallel cand-099 remains not_verified due frozen checker memory growth.

## 2026-09-28 19:37:24 — cand-101: Incremental full P1: w1 prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "incremental_gemm", "gemms": ["w1"], "shards": 4}]
- Evidence: project/experiments/cand-101/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-101/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-101/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-101/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-101/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=40.452517415978946; peak MiB=3079.0182342529297
- Inputs: compiler.py@77923d93 hardware.json@b92be26a recipe.json@07415acd
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=40.45s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:37:24 — cand-102: Incremental full P1: w2 prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "incremental_gemm", "gemms": ["w2"], "shards": 4}]
- Evidence: project/experiments/cand-102/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-102/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-102/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-102/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-102/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=36.130835209041834; peak MiB=3072.346336364746
- Inputs: compiler.py@77923d93 hardware.json@b92be26a recipe.json@4845157c
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=36.13s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:37:24 — cand-103: Incremental full P1: w1,w2 prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "incremental_gemm", "gemms": ["w1", "w2"], "shards": 4}]
- Evidence: project/experiments/cand-103/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-103/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-103/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-103/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-103/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=36.624174332944676; peak MiB=3085.6901321411133
- Inputs: compiler.py@77923d93 hardware.json@b92be26a recipe.json@3a876d88
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=36.62s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:37:24 — cand-104: Incremental full P1: qkv prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "incremental_gemm", "gemms": ["qkv"], "shards": 4}]
- Evidence: project/experiments/cand-104/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-104/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-104/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-104/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-104/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=40.49411379103549; peak MiB=3078.2213592529297
- Inputs: compiler.py@77923d93 hardware.json@b92be26a recipe.json@99a3aeb9
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=40.49s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:37:24 — cand-105: Incremental full P1: qkv,w1,w2 prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged
- Change: parent=cand-098; [{"op": "p1_schedule", "kind": "incremental_gemm", "gemms": ["qkv", "w1", "w2"], "shards": 4}]
- Evidence: project/experiments/cand-105/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-105/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-105/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-105/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-105/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=37.36664620798547; peak MiB=3080.0026092529297
- Inputs: compiler.py@77923d93 hardware.json@b92be26a recipe.json@5b200d7b
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=37.37s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 19:54 — Validation-cost diagnosis and session handoff
- Change: Read-only expanded-program cost/traffic analysis; incremental prompt GEMM skeleton; validation-cost preflight; outer/inner plateau guard. Default emitted P1/D1 remain byte-identical to cand-098; hardware and frozen evaluator unchanged.
- Verify: `.venv/bin/python -m unittest discover -s project/search/tests -v` -> `project/campaigns/validation-efficient/tests.stderr` -> 51 passed. `.venv/bin/python -m project.campaigns.p1-multism.verify_controls` -> passed.
- Verify: Full grade attempts cand-101..105 all `not_verified / memory_limit`, each one attempt at 3072 MiB. No grade.json was produced; earlier generic log references denote intended report paths, not completed reports. Actual evidence: each metrics.json and grade.heartbeat.json.
- Verify: Frozen race checker observation on complete cand-101 at 2560 MiB -> `project/campaigns/validation-efficient/only-w1-race-observation.json`, peak 2568.94 MiB; at PC 9999 still prompt Attention, before first parallel entry at PC 17412. No correctness/timing pass inferred.
- Result: Incremental event counts all 161516, retained-set logical allocation lower bounds 102–108 GiB. Logical heap estimates are not measured peak RSS. Default score remains 2567.95405 (P1=8522411, D1=718132).
- Decision: User explicitly requested documentation, Git commit, and new-session handoff. Stop experiments; no official submission. See `project/CURRENT_SESSION_HANDOFF.md` and `project/campaigns/validation-efficient/REPORT.md`. Remote resources and permission to advance algorithm compression remain unconfirmed.

## 2026-09-28 22:59 — Local validation preflight integration audit
- Change: Preserve validation-cost/heartbeat/calibration metadata for prepared inputs; reject a retained cost-prune decision before grade unless explicitly calibrated. Frozen checker, algorithms, hardware and emitted submission unchanged.
- Verify: `.venv/bin/python -m unittest project.search.tests.test_validation_preflight -v` -> `project/campaigns/validation-preflight-audit/tests.stderr` -> 2 passed; actual generation/static/cost subprocess chain prunes with zero grade launches.
- Result: No new full-model evaluation or score. cand-098 remains P1=8522411 D1=718132 score=2567.95405. Existing cand-101..105 memory failures were not repeated.
- Decision: Continue on this Mac as requested. Await explicit choice on advancing Attention algorithm blocking before a multi-SM pass; do not modify frozen checker, expand memory limits or assert an unmeasured speedup. See `project/campaigns/validation-preflight-audit/REPORT.md`.
