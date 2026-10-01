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

## 2026-09-28 23:03:11 — cand-106: Block 16 prompt queries, stream 32-key/value panels, coalesce public KV export; keep row-local causal softmax and D1 unchanged
- Change: parent=cand-098; [{"op": "attention_algorithm", "algorithm": "blocked_prompt"}]
- Evidence: project/experiments/cand-106/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-106/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-106/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-106/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-106/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=75.70147025003098; peak MiB=1513.2515563964844
- Inputs: compiler.py@36184949 hardware.json@b92be26a recipe.json@70dcde79
- Verify: PASS elapsed=75.70s
- Result: M1_P1=7292285 (4.251x), M2_D1=718132 (1.813x); score=2776.110, eligible=True
  - deltas vs head (cand-095): {'M1_P1': -1230126, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=2776.110)

## 2026-09-28 23:06:24 — cand-107: Integrate only W1 four-SM GEMM after verified blocked prompt attention; enforce validation memory preflight.
- Change: parent=cand-106; []
- Evidence: project/experiments/cand-107/metrics.json; reports/grade.json; attempts=0
- Inputs: compiler.py@36184949 hardware.json@b92be26a recipe.json@c23fd461
- Verify: FAILED — failure_class=validation_cost_pruned; evidence=pipeline
- Result: status=not_verified, failure_class=validation_cost_pruned
- Decision: reject/preserve failed candidate (status=not_verified)

### cand-106 verdict: KEPT — Full frozen grade eligible; blocked prompt attention reduces P1 cycles 14.43%, D1 unchanged. Multi-SM child cand-107 cost-pruned, no claim of multi-SM pass.

## 2026-09-28 — cand-106 full-program attribution and cand-107 decision
- Change: profile verified blocked prompt Attention via native trace; no evaluator edits.
- Verify: PYTHONPATH=. .venv/bin/python project/campaigns/p1-multism/profile.py cand-106 -> cand-106/reports/p1-profile.json; reproduced P1 7292285 cycles; D1 byte-identical to cand-098.
- Result: Attention attribution 1811460 -> 581682 cycles, full P1 1.1687x; HBM read/write unchanged. W1/GELU/W2 now 61.05% of P1.
- Decision: keep cand-106; cand-107 preflight-only prune (41.31 GiB logical retained sets, not measured RSS), no full multi-SM result. Stop same-structure variations; next structural hypothesis persistent MLP, requiring ~2.203x region speedup for 1.5x full P1. See campaigns/blocked-attention/REPORT.md.

## 2026-09-28 23:14:00 — cand-108: Persistent row-block W1/GELU/W2: activations and all W2 accumulators stay in RF; eliminate ff1/gelu HBM round trips. Trade additional weight reads for no materialization.
- Change: parent=cand-106; []
- Evidence: project/experiments/cand-108/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-108/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-108/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-108/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-108/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=18.275015250081196; peak MiB=207.00061798095703
- Inputs: compiler.py@d9bf0464 hardware.json@b92be26a recipe.json@208993ce
- Verify: FAILED — in())
                     ^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/cli.py", line 344, in main
    result = _evaluate(
             ^^^^^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/cli.py", line 234, in _evaluate
    report = evaluate_submission(mode, hardware, programs, seeds, baseline)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/runner.py", line 323, in evaluate_submission
    report["cases"][case]["timing"] = estimate_case(case, hardware, programs[case])
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/runner.py", line 274, in estimate_case
    timing = estimate_pipeline(
             ^^^^^^^^^^^^^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/pipeline.py", line 178, in estimate_pipeline
    return estimate_global_events(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/pipeline_events.py", line 1173, in estimate_global_events
    ).run()
      ^^^^^
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/pipeline_events.py", line 1063, in run
    self._issue(next_instruction)
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/pipeline_events.py", line 594, in _issue
    self._issue_transfer(tape, pc, ins, begin, sequence)
  File "/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/codesign/challenge/pipeline_events.py", line 701, in _issue_transfer
    raise AssertionError("DMA slot changed after issue preview")
AssertionError: DMA slot changed after issue preview
; evidence=pipeline elapsed=18.28s
- Result: status=not_verified, failure_class=evaluator_error
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 23:15:57 — cand-109: D1 four persistent SM owners: parallel projection columns and attention heads over eight steps; P1 unchanged. Preflight checker cost before complete grade.
- Change: parent=cand-106; []
- Evidence: project/experiments/cand-109/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-109/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-109/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-109/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-109/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=15.568582749925554; peak MiB=3094.7213821411133
- Inputs: compiler.py@33d047d8 hardware.json@b92be26a recipe.json@7a69e229
- Verify: FAILED — failure_class=memory_limit; evidence=pipeline elapsed=15.57s
- Result: status=not_verified, failure_class=memory_limit
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-28 23:16:30 — cand-110: RF persistent MLP with completion waits on final output store burst; cand-108 hit frozen DMA issue-preview assertion on eighth ST. No evaluator change.
- Change: parent=cand-108; []
- Evidence: project/experiments/cand-110/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-110/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-110/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-110/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-110/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=120.50940462492872; peak MiB=1379.3764419555664
- Inputs: compiler.py@3d4023cb hardware.json@b92be26a recipe.json@208993ce
- Verify: PASS elapsed=120.52s
- Result: M1_P1=7049767 (4.397x), M2_D1=718132 (1.813x); score=2823.457, eligible=True
  - deltas vs head (cand-106): {'M1_P1': -242518, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=2823.457)

## 2026-09-29 — Recover cand-111 registration before grade
- cand-110 evaluation saved its earlier registry snapshot after cand-111 creation, removing cand-111 registry entry. Immutable cand-111 input/proposal files survived.
- Restored only the missing registry record from candidate.json/proposal.json; no experiment evidence overwritten. cand-111 grade had not started. Future candidate creation/evaluation registry mutations are serialized.

## 2026-09-28 23:17:26 — cand-111: D1 persistent parallel owners preserve native64-column panels: QKV3SM, WO/W2 2SM, W1/attention4SM. Avoid 65% event inflation seen in cand-109. Full grade bounded6GiB on local16GiB Mac.
- Change: parent=cand-106; []
- Evidence: project/experiments/cand-111/metrics.json; reports/grade.json; attempts=0
- Inputs: compiler.py@35b8cfd3 hardware.json@b92be26a recipe.json@fd1c7264
- Verify: FAILED — Recovered incomplete evaluation; evidence=pipeline
- Result: status=not_verified, failure_class=interrupted
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-29 — Persistent MLP bottleneck attribution and DMA diagnosis
- Verify: native estimate traceback diagnostic for cand-108 locates frozen assertion at PC8777, eighth final ST; diagnostic saved in campaigns/structural-mlp/cand-108-dma-diagnostic.txt. No evaluator edits.
- Result: cand-110 complete grade passes after final-store completion waits: P1=7049767 D1=718132 score=2823.45652. HBM writes fall38.74%, but RF read/write busy cycles barely change.
- Decision: retain validated improvement; materialization elimination alone is insufficient for large gains. Investigate actual SM parallelism and RF operand reuse; see campaigns/structural-mlp/REPORT.md.

## 2026-09-29 00:40:06 — cand-112: D1 persistent parallel owners preserve native64-column panels: QKV3SM, WO/W2 2SM, W1/attention4SM. Avoid 65% event inflation seen in cand-109. Full grade bounded6GiB on local16GiB Mac. Recovery attempt after cand-111 registry interruption; no prior grade result.
- Change: parent=cand-106; []
- Evidence: project/experiments/cand-112/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-112/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-112/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-112/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-112/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=124.74917254201137; peak MiB=5393.301544189453
- Inputs: compiler.py@35b8cfd3 hardware.json@b92be26a recipe.json@fd1c7264
- Verify: PASS elapsed=124.75s
- Result: M1_P1=7292285 (4.251x), M2_D1=278743 (4.671x); score=4455.914, eligible=True
  - deltas vs head (cand-110): {'M1_P1': 242518, 'M2_D1': -439389}
- Decision: KEPT — new incumbent head (score=4455.914)

## 2026-09-29 — MILP D1 partition model (no score claim)
- Change: bounded HiGHS1.15.1 multiple-choice integer model over four D1 projection stages; owner count1..4, RF-legal N/K panels, event cap15500. Solver environment /tmp/transformer-milp-env is separate from frozen grade environment.
- Verify: PYTHONPATH=. /tmp/transformer-milp-env/bin/python project/campaigns/milp-d1/solve.py -> solution.json, options.json, model.lp. Exhaustive enumeration independently matches optimal service-proxy objective122624.
- Result: selected QKV/WO/W2 four owners with N32/K32; W1 four owners with N64/K16. Predicted expanded compute/transfer events15376. Proxy objective is not simulated cycles; candidate113 goes through full cost screen and grade.
- Decision: test solver proposal against verified heuristic cand-112; no broad hardware sweep or evaluator edits.

## 2026-09-29 00:44:30 — cand-113: HiGHS MILP chooses D1 projection owner counts and RF panels under15500-event cap; compare actual full grade against heuristic cand-112. P1/hardware unchanged.
- Change: parent=cand-112; []
- Evidence: project/experiments/cand-113/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-113/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-113/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-113/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-113/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=122.78160633298103; peak MiB=5178.754554748535
- Inputs: compiler.py@09b90e14 hardware.json@b92be26a recipe.json@77ff6593
- Verify: PASS elapsed=122.78s
- Result: M1_P1=7292285 (4.251x), M2_D1=212659 (6.123x); score=5101.489, eligible=True
  - deltas vs head (cand-112): {'M1_P1': 0, 'M2_D1': -66084}
- Decision: KEPT — new incumbent head (score=5101.489)

## 2026-09-29 — MILP controlled comparison confirmed
- Verify: cand-113 full frozen grade eligible; P1/hardware byte-identical to cand-112. D1=212659 versus278743 cycles; score5101.48865 versus4455.91424.
- Result: D1 1.31075x, score+14.488%; peak5178.75MiB within6144MiB cap. Exact events15376 match MILP constraint count.
- Decision: retain MILP proposal; its service objective is an approximate ranking model, not a cycle prediction or universal optimum. Compose with verified P1 fusion in fresh cand-114.

## 2026-09-29 00:47:16 — cand-114: Compose independently verified P1 RF persistent MLP (cand-110) with MILP D1 projection plan (cand-113); complete frozen grade confirms shared hardware and both programs.
- Change: parent=cand-113; []
- Evidence: project/experiments/cand-114/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-114/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-114/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-114/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-114/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=110.82400549994782; peak MiB=5230.457633972168
- Inputs: compiler.py@09b90e14 hardware.json@b92be26a recipe.json@dba49006
- Verify: PASS elapsed=110.83s
- Result: M1_P1=7049767 (4.397x), M2_D1=212659 (6.123x); score=5188.494, eligible=True
  - deltas vs head (cand-113): {'M1_P1': -242518, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=5188.494)

### cand-114 verdict: KEPT — Complete grade passes: RF resident P1 MLP plus MILP-balanced four-SM decode. Preserve full experiment and solver evidence; no official submission.

## 2026-09-29 00:58:04 — cand-115: Automatic MILP D1 projection proposal with measured feedback and event-budget trust region.
- Change: parent=cand-114; []
- Evidence: project/experiments/cand-115/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-115/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-115/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-115/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-115/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=118.64900533400942; peak MiB=4982.707450866699
- Inputs: compiler.py@09b90e14 hardware.json@b92be26a recipe.json@37efd95c
- Verify: PASS elapsed=118.66s
- Result: M1_P1=7049767 (4.397x), M2_D1=210849 (6.175x); score=5210.717, eligible=True
  - deltas vs head (cand-114): {'M1_P1': 0, 'M2_D1': -1810}
- Decision: KEPT — new incumbent head (score=5210.717)

## 2026-09-29 01:00:06 — cand-116: Automatic MILP D1 projection proposal with measured feedback and event-budget trust region.
- Change: parent=cand-115; []
- Evidence: project/experiments/cand-116/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-116/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-116/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-116/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-116/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=123.42704683402553; peak MiB=5218.129623413086
- Inputs: compiler.py@09b90e14 hardware.json@b92be26a recipe.json@63148e3b
- Verify: PASS elapsed=123.43s
- Result: M1_P1=7049767 (4.397x), M2_D1=215937 (6.030x); score=5148.962, eligible=True
  - deltas vs head (cand-115): {'M1_P1': 0, 'M2_D1': 5088}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-29 01:02:13 — cand-117: Automatic MILP D1 projection proposal with measured feedback and event-budget trust region.
- Change: parent=cand-115; []
- Evidence: project/experiments/cand-117/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-117/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-117/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-117/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-117/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=115.40498912497424; peak MiB=5524.629737854004
- Inputs: compiler.py@09b90e14 hardware.json@b92be26a recipe.json@47560b03
- Verify: PASS elapsed=115.41s
- Result: M1_P1=7049767 (4.397x), M2_D1=228289 (5.703x); score=5007.728, eligible=True
  - deltas vs head (cand-115): {'M1_P1': 0, 'M2_D1': 17440}
- Decision: evaluated; retained in Pareto/archives (head preserved)

## 2026-09-29 01:04:57 — cand-118: After D1 MILP plateau, change outer P1 MLP issue schedule: alternate RF0/1 and RF13/14 operands so next loads overlap current MMA; same arithmetic, loads and stores, D1 unchanged.
- Change: parent=cand-115; []
- Evidence: project/experiments/cand-118/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-118/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-118/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-118/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-118/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=109.23450200003572; peak MiB=5459.535873413086
- Inputs: compiler.py@79d78164 hardware.json@b92be26a recipe.json@64177290
- Verify: PASS elapsed=109.24s
- Result: M1_P1=6201341 (4.998x), M2_D1=210849 (6.175x); score=5555.742, eligible=True
  - deltas vs head (cand-115): {'M1_P1': -848426, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=5555.742)

## 2026-09-29 — Automated MILP campaign and registry consistency
- Change: sequential MILP provider, historical no-good exclusions, bounded grade, durable feedback and plateau stop; registry compare-and-swap rejects stale snapshots.
- Verify: 16 state/feedback/records tests pass; 2 isolated-HiGHS solver tests pass, including independent enumeration and infeasible/no-good constraints.
- Result: auto-milp-20260929-005803 evaluated cand-115/116/117 at scores5210.7168/5148.9623/5007.7282 and stopped after3 valid low-gain candidates with outer_structure_required; best115 staged.
- Decision: stop this D1 inner sweep. New outer P1 operand-pipeline candidate118 enters full grade. Local single-projection feedback calibrates subsequent MILP proposals; no global optimum claim. Gemini review returned provider503/no capacity and supplied no usable findings.

### cand-118 verdict: KEPT — Full grade passes: alternate MLP operand buffers overlap loads and MMA; P1 improvement after D1 automatic plateau.

## 2026-09-29 01:14:03 — cand-119: P1 QKV only: double-buffer A row tiles and B K panels so next HBM-to-RF loads overlap current 32x32 MMA; preserve arithmetic order and D1.
- Change: parent=cand-118; []
- Evidence: project/experiments/cand-119/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-119/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-119/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-119/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-119/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=112.92825416696724; peak MiB=5126.254486083984
- Inputs: compiler.py@aa8440fd hardware.json@b92be26a recipe.json@40ddbe94
- Verify: PASS elapsed=112.93s
- Result: M1_P1=5737061 (5.403x), M2_D1=210849 (6.175x); score=5776.172, eligible=True
  - deltas vs head (cand-118): {'M1_P1': -464280, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=5776.172)

### cand-119 verdict: KEPT — Full grade passes: QKV A/B operand double buffering reduces P1 while preserving instruction count and D1.

## 2026-09-29 01:16:23 — cand-120: Compose verified P1 QKV operand pipeline with identical WO A/B double buffering; preserve arithmetic order and D1.
- Change: parent=cand-119; []
- Evidence: project/experiments/cand-120/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-120/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-120/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-120/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-120/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=113.5413031670032; peak MiB=5155.832565307617
- Inputs: compiler.py@aa8440fd hardware.json@b92be26a recipe.json@5952b2cd
- Verify: PASS elapsed=113.55s
- Result: M1_P1=5580809 (5.554x), M2_D1=210849 (6.175x); score=5856.475, eligible=True
  - deltas vs head (cand-119): {'M1_P1': -156252, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=5856.475)

### cand-120 verdict: KEPT — Full grade passes: compose QKV and WO operand double buffering; instruction counts and D1 unchanged.

## 2026-09-29 01:18:45 — cand-121: P1 blocked Attention: double-buffer two K and V panels so next HBM-to-RF load overlaps current QK/PV MMA; preserve softmax, causal mask, MMA order and D1.
- Change: parent=cand-120; []
- Evidence: project/experiments/cand-121/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-121/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-121/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-121/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-121/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=128.46293570799753; peak MiB=4994.066482543945
- Inputs: compiler.py@aa8440fd hardware.json@b92be26a recipe.json@8208bd47
- Verify: PASS elapsed=128.47s
- Result: M1_P1=6810587 (4.551x), M2_D1=210849 (6.175x); score=5301.424, eligible=True
  - deltas vs head (cand-120): {'M1_P1': 1229778, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-121 verdict: REVERTED — Rejected: candidate snapshot predates blocked_prompt_pipeline implementation; recipe label fell back to old attention path, so result is not evidence about panel pipelining.

## 2026-09-29 01:23:59 — cand-122: Corrected generator snapshot: blocked P1 Attention double-buffers two K/V panels while preserving softmax, causal mask, MMA order and D1.
- Change: parent=cand-120; []
- Evidence: project/experiments/cand-122/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-122/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-122/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-122/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-122/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=113.68067525001243; peak MiB=5181.629440307617
- Inputs: compiler.py@751e9357 hardware.json@b92be26a recipe.json@8208bd47
- Verify: PASS elapsed=113.68s
- Result: M1_P1=5571113 (5.564x), M2_D1=210849 (6.175x); score=5861.569, eligible=True
  - deltas vs head (cand-120): {'M1_P1': -9696, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=5861.569)

### cand-122 verdict: KEPT — Corrected Attention panel pipeline passes full grade; small positive P1 gain with unchanged instruction count and D1.

## 2026-09-29 01:26:27 — cand-123: P1 prompt MLP: keep complete 32x256 LN2 input block in RF for W1, store 32x1024 GELU activation in 128KiB SH, then consume it in W2; preserve reduction order and D1.
- Change: parent=cand-122; []
- Evidence: project/experiments/cand-123/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-123/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-123/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-123/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-123/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=108.59989662503358; peak MiB=5139.801338195801
- Inputs: compiler.py@751e9357 hardware.json@b92be26a recipe.json@7cf7e99d
- Verify: PASS elapsed=108.61s
- Result: M1_P1=6358501 (4.875x), M2_D1=210849 (6.175x); score=5486.653, eligible=True
  - deltas vs head (cand-122): {'M1_P1': 787388, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-123 verdict: REVERTED — Correct but slower: SH activation residency and LN2 RF reuse lose current W1/W2 overlap; reject despite fewer instructions.

## 2026-09-29 — Post-pipeline single-field hardware screen
- Change: Hold cand-122 programs fixed and estimate current hardware plus NoC512, DMA4, SFU16, reduction2, HBM8, cache4/8/16, multicast, and four bounded combinations. Exclude higher TC-count variants before simulation because their modeled area exceeds 24 mm².
- Verify: `.venv/bin/python project/campaigns/hardware-post-pipeline/screen.py` -> `project/campaigns/hardware-post-pipeline/results.json` and per-variant `estimate-*.json` -> timing estimates completed; estimate is ranking evidence only.
- Result: current=5861.569 proxy; NoC512 unchanged; DMA4=5859.803; SFU16=5964.877; reduction2=5885.171; HBM8=5900.641; cache4/8/16=5861.821/5891.097/6031.966. Best nested combination cache16+SFU16+HBM8+reduction2 estimates P1=5,072,661, D1=207,673, score=6189.601, area=22.700 mm², peak power P1/D1=4.015/6.707 W.
- Decision: stop this hardware inner neighborhood after low single-field gains; send only the best legal combination to complete frozen grade as cand-124.

## 2026-09-29 01:46:59 — cand-124: Compose the only measured post-pipeline hardware gains: 16 MiB cache, 16 SFU lanes, 8 HBM channels, and 2 reduction units; preserve verified cand-122 programs.
- Change: parent=cand-122; Bounded hardware combination selected by fixed-program estimate screen.
- Evidence: project/experiments/cand-124/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-124/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-124/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-124/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-124/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=103.71696650004014; peak MiB=5379.973373413086
- Inputs: compiler.py@751e9357 hardware.json@c324ea0d recipe.json@8208bd47
- Verify: PASS elapsed=103.72s
- Result: M1_P1=5072661 (6.111x), M2_D1=207673 (6.270x); score=6189.601, eligible=True
  - deltas vs head (cand-122): {'M1_P1': -498452, 'M2_D1': -3176}
- Decision: KEPT — new incumbent head (score=6189.601)

### cand-124 verdict: KEPT — Full grade passes exactly at estimate: best bounded hardware composition; new incumbent after post-pipeline screen.

## 2026-09-29 01:51:59 — cand-125: Assign each independent P1 batch member end-to-end to one persistent WG/SM, with no per-layer cross-WG barriers and only the required STEP.COMMIT join.
- Change: parent=cand-124; Validation-efficient two-WG batch-parallel P1 skeleton.
- Evidence: project/experiments/cand-125/metrics.json; reports/grade.json; attempts=0
- Inputs: compiler.py@f18b4b7f hardware.json@c324ea0d recipe.json@7de42f47
- Verify: FAILED — failure_class=validation_cost_pruned; evidence=pipeline
- Result: status=not_verified, failure_class=validation_cost_pruned
- Decision: reject/preserve failed candidate (status=not_verified)

### cand-125 verdict: REVERTED — Validation-cost pruned: two end-to-end batch WGs remove per-layer barriers but still require a 17.46 GB retained-event-set logical lower bound under frozen checker; no grade launched.

## 2026-09-29 — cand-124 post-hardware stage attribution
- Change: Profile the kept cand-124 P1 trace without changing candidate inputs.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/p1-multism/profile.py cand-124` -> `project/experiments/cand-124/reports/p1-profile.json` -> reproduced P1=5,072,661 cycles.
- Result: prompt MLP=2,468,852 cycles (48.67%), QKV=794,736 (15.67%), Attention=506,960 (9.99%), WO=264,912 (5.22%); P1 still executes on SM0 only. D1 uses all four SMs.
- Decision: stop the saturated hardware inner neighborhood. The next outer search must reduce prompt MLP/QKV event count and work before multi-WG validation; cand-125 proves barrier removal alone is insufficient.

## 2026-09-29 01:55:58 — cand-126: Retile only P1 first-token GEMMs from N16/K48 to RF-legal N64/K16, reducing small-M QKV/W1/W2 panel count and event overhead; prompt path and D1 unchanged.
- Change: parent=cand-124; P1 step projection retile N64 K16.
- Evidence: project/experiments/cand-126/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-126/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-126/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-126/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-126/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=103.42327679193113; peak MiB=5398.692123413086
- Inputs: compiler.py@751e9357 hardware.json@c324ea0d recipe.json@069653fb
- Verify: PASS elapsed=103.43s
- Result: M1_P1=5038371 (6.152x), M2_D1=207673 (6.270x); score=6210.628, eligible=True
  - deltas vs head (cand-124): {'M1_P1': -34290, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6210.628)

### cand-126 verdict: KEPT — Full grade passes; small but certain P1 step retile gain with reduced expanded instructions and unchanged D1.

## 2026-09-29 01:58:49 — cand-127: Retile only P1 first-token GEMMs from N64/K16 to the next RF-capacity-equivalent point N128/K8, reducing output-panel initialization/stores and repeated A traffic.
- Change: parent=cand-126; P1 step projection retile N128 K8.
- Evidence: project/experiments/cand-127/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-127/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-127/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-127/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-127/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=106.00069658295251; peak MiB=5396.3952713012695
- Inputs: compiler.py@751e9357 hardware.json@c324ea0d recipe.json@8febb4c1
- Verify: PASS elapsed=106.01s
- Result: M1_P1=5045751 (6.143x), M2_D1=207673 (6.270x); score=6206.085, eligible=True
  - deltas vs head (cand-126): {'M1_P1': 7380, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-127 verdict: REVERTED — Correct full grade but slightly slower than cand-126; larger N128/K8 does not improve the step schedule.

## 2026-09-29 02:01:06 — cand-128: Test the final RF-capacity-equivalent P1 first-token boundary N256/K4; minimize N panels while keeping B tiles at 1024 words and TC K groups exact.
- Change: parent=cand-126; P1 step projection retile N256 K4; final point in bounded inner sweep.
- Evidence: project/experiments/cand-128/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-128/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-128/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-128/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-128/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=107.06090270797722; peak MiB=5400.520225524902
- Inputs: compiler.py@751e9357 hardware.json@c324ea0d recipe.json@02b69b16
- Verify: PASS elapsed=107.07s
- Result: M1_P1=5063367 (6.122x), M2_D1=207673 (6.270x); score=6195.279, eligible=True
  - deltas vs head (cand-126): {'M1_P1': 24996, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-128 verdict: REVERTED — Correct full grade but slower than cand-126; stop the three-point P1-step tile neighborhood and return to outer structural search.

## 2026-09-29 02:04:38 — cand-129: Keep W2 accumulators in RF through residual2 and b2, broadcasting bias into RF and storing final prompt output directly; remove ff2 HBM materialization and row-wise residual epilogue.
- Change: parent=cand-126; Fuse RF-persistent prompt MLP W2 with residual+bias output epilogue.
- Evidence: project/experiments/cand-129/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-129/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-129/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-129/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-129/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=58.83156325004529; peak MiB=5046.6602783203125
- Inputs: compiler.py@5db9cf0b hardware.json@c324ea0d recipe.json@38f6b0da
- Verify: FAILED — failure_class=incorrect; evidence=reports/check.json elapsed=58.84s
- Result: status=incorrect, failure_class=incorrect
- Decision: reject/preserve failed candidate (status=incorrect)

### cand-129 verdict: REVERTED — Rejected: invalid stride-zero HBM bias broadcast; repair with verified K=1 outer-product bias epilogue in cand-130.

## 2026-09-29 02:06:29 — cand-130: Repair fused RF-resident W2 residual epilogue using the existing legal K=1 ones-by-bias MMA after ff2+residual VEC; eliminate ff2 round trip and row-wise residual2.
- Change: parent=cand-126; Legal RF-persistent W2 residual+bias epilogue.
- Evidence: project/experiments/cand-130/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-130/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-130/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-130/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-130/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=102.83081037492957; peak MiB=5434.5202713012695
- Inputs: compiler.py@52a36d05 hardware.json@c324ea0d recipe.json@38f6b0da
- Verify: PASS elapsed=102.84s
- Result: M1_P1=5008053 (6.189x), M2_D1=207673 (6.270x); score=6229.399, eligible=True
  - deltas vs head (cand-126): {'M1_P1': -30318, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6229.399)

### cand-130 verdict: KEPT — Full grade passes: W2 accumulators remain in RF through residual+bias; removes ff2 HBM round trip and row residual loop, small positive cycles and lower event count.

## 2026-09-29 02:09:37 — cand-131: Cache prompt LN gamma/beta in RF and carry WO residual rows directly into LN2 while still writing the residual once; remove repeated affine loads and residual HBM reload.
- Change: parent=cand-130; Fuse prompt residual1-LN2 and cache both prompt LayerNorm affine parameters.
- Evidence: project/experiments/cand-131/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-131/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-131/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-131/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-131/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=102.18015029199887; peak MiB=5421.582748413086
- Inputs: compiler.py@0c3dc51c hardware.json@c324ea0d recipe.json@59f06ab6
- Verify: PASS elapsed=102.19s
- Result: M1_P1=4963013 (6.246x), M2_D1=207673 (6.270x); score=6257.601, eligible=True
  - deltas vs head (cand-130): {'M1_P1': -45040, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6257.601)

### cand-131 verdict: KEPT — Full grade passes: prompt affine parameters stay in RF and residual1 flows directly into LN2; lower cycles, traffic, and expanded events.

## 2026-09-29 02:13:56 — cand-132: Place each prompt LN2 activation in 64 KiB shared memory and feed the existing double-buffered RF MLP from SH, eliminating repeated cache/NoC source loads without changing GELU/W2 scheduling.
- Change: parent=cand-131; LN2-to-W1 shared-resident source with existing RF-pipelined fused MLP.
- Evidence: project/experiments/cand-132/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-132/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-132/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-132/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-132/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=98.06765970797278; peak MiB=5341.520133972168
- Inputs: compiler.py@97a2189c hardware.json@c324ea0d recipe.json@b65ba68d
- Verify: PASS elapsed=98.08s
- Result: M1_P1=4960771 (6.248x), M2_D1=207673 (6.270x); score=6259.015, eligible=True
  - deltas vs head (cand-131): {'M1_P1': -2242, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6259.015)

### cand-132 verdict: KEPT — Full grade passes with a marginal positive result; retain best score but stop this SH-input neighborhood because 16 MiB cache already captures most reuse.

## 2026-09-29 02:21:48 — cand-133: Enable multicast, the only positive remaining legal hardware field in the fixed cand-132 screen; preserve programs and all other hardware.
- Change: parent=cand-132; Final bounded hardware refinement: multicast only.
- Evidence: project/experiments/cand-133/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-133/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-133/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-133/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-133/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=123.59193879098166; peak MiB=5411.020202636719
- Inputs: compiler.py@97a2189c hardware.json@d362f61f recipe.json@b65ba68d
- Verify: PASS elapsed=123.60s
- Result: M1_P1=4960771 (6.248x), M2_D1=207315 (6.280x); score=6264.417, eligible=True
  - deltas vs head (cand-132): {'M1_P1': 0, 'M2_D1': -358}
- Decision: KEPT — new incumbent head (score=6264.417)

### cand-133 verdict: KEPT — Full grade matches screen: multicast gives the only remaining positive hardware delta; close bounded hardware search.

## 2026-09-29 02:31:34 — cand-134: Execute both P1 batch members layer-major so each layer weight set is reused immediately across the two independent members; preserve arithmetic and single-WG validation fast path.
- Change: parent=cand-133; Layer-major batch ordering for cache reuse.
- Evidence: project/experiments/cand-134/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-134/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-134/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-134/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-134/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=98.30359158292413; peak MiB=5363.223304748535
- Inputs: compiler.py@ed0038ab hardware.json@d362f61f recipe.json@f583700d
- Verify: PASS elapsed=98.31s
- Result: M1_P1=4960924 (6.248x), M2_D1=207315 (6.280x); score=6264.321, eligible=True
  - deltas vs head (cand-133): {'M1_P1': 153, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-134 verdict: REVERTED — Correct but marginally slower on cache16/TC2; layer-major+TC3/cache4 estimate scores 6139.90, so immediate weight reuse does not recover lost cache capacity.

## 2026-09-29 02:38:25 — cand-135: Shard each single-row D1 GELU across the four persistent owners, reusing the existing W1 trailing and W2 leading barriers; remove the SM0-only GELU bottleneck without new sync boundaries.
- Change: parent=cand-133; Four-way persistent D1 GELU column partition.
- Evidence: project/experiments/cand-135/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-135/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-135/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-135/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-135/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=100.37200987502001; peak MiB=5549.957748413086
- Inputs: compiler.py@afae0815 hardware.json@d362f61f recipe.json@5f490ed2
- Verify: PASS elapsed=100.38s
- Result: M1_P1=4960771 (6.248x), M2_D1=195174 (6.671x); score=6456.320, eligible=True
  - deltas vs head (cand-133): {'M1_P1': 0, 'M2_D1': -12141}
- Decision: KEPT — new incumbent head (score=6456.320)

### cand-135 verdict: KEPT — Full grade passes: four-way D1 GELU partition removes the SM0-only segment and gives the largest gain in this continuation.

## 2026-09-29 02:42:05 — cand-136: Double-buffer A/B K panels for every sharded single-row D1 projection while preserving owner counts, tile shapes, event count, and ascending K arithmetic.
- Change: parent=cand-135; Four-owner D1 GEMM operand pipeline.
- Evidence: project/experiments/cand-136/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-136/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-136/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-136/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-136/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=107.64860445901286; peak MiB=5562.879600524902
- Inputs: compiler.py@6dfa0f02 hardware.json@d362f61f recipe.json@8e8a84cd
- Verify: PASS elapsed=107.66s
- Result: M1_P1=4960771 (6.248x), M2_D1=171256 (7.603x); score=6892.442, eligible=True
  - deltas vs head (cand-135): {'M1_P1': 0, 'M2_D1': -23918}
- Decision: KEPT — new incumbent head (score=6892.442)

### cand-136 verdict: KEPT — Full grade passes: double-buffered D1 projection operands overlap LD and MMA, producing a material end-to-end decode improvement with the same arithmetic.

## 2026-09-29 02:46:39 — cand-137: Double-buffer only the 128-token D1 history K/V panels inside each persistent attention head owner; preserve RF softmax, generated-tail path, event count, and arithmetic.
- Change: parent=cand-136; D1 history K/V panel pipeline.
- Evidence: project/experiments/cand-137/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-137/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-137/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-137/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-137/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=105.46414175000973; peak MiB=5592.223442077637
- Inputs: compiler.py@94110064 hardware.json@d362f61f recipe.json@ab2388f7
- Verify: PASS elapsed=105.47s
- Result: M1_P1=4960771 (6.248x), M2_D1=169003 (7.704x); score=6938.232, eligible=True
  - deltas vs head (cand-136): {'M1_P1': 0, 'M2_D1': -2253}
- Decision: KEPT — new incumbent head (score=6938.232)

### cand-137 verdict: KEPT — Full grade passes: history K/V double buffering gives a small positive D1 gain; stop attention panel tuning.

## 2026-09-29 02:50:39 — cand-138: Reuse the existing 64 KiB prompt shared allocation for LN1 output, and feed the unchanged QKV operand pipeline from SH instead of repeated cache/NoC activation reads.
- Change: parent=cand-137; LN1-to-QKV shared-resident activation.
- Evidence: project/experiments/cand-138/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-138/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-138/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-138/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-138/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=99.85473783290945; peak MiB=5509.348419189453
- Inputs: compiler.py@136be636 hardware.json@d362f61f recipe.json@8a06dffb
- Verify: PASS elapsed=99.86s
- Result: M1_P1=4959259 (6.250x), M2_D1=169003 (7.704x); score=6939.290, eligible=True
  - deltas vs head (cand-137): {'M1_P1': -1512, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6939.290)

### cand-138 verdict: KEPT — Full grade passes with a marginal positive LN1-to-QKV SH result; retain best score and stop SH residency expansion.

## 2026-09-29 02:54:11 — cand-139: Keep each single-token D1 activation vector resident in one RF partition across all output panels; retain B double buffering and identical ascending-K arithmetic.
- Change: parent=cand-138; D1 A-resident projection pipeline.
- Evidence: project/experiments/cand-139/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-139/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-139/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-139/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-139/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=76.88162470795214; peak MiB=4066.6442642211914
- Inputs: compiler.py@d0d6c993 hardware.json@d362f61f recipe.json@6a8b7871
- Verify: PASS elapsed=76.89s
- Result: M1_P1=4959259 (6.250x), M2_D1=167539 (7.772x); score=6969.543, eligible=True
  - deltas vs head (cand-138): {'M1_P1': 0, 'M2_D1': -1464}
- Decision: KEPT — new incumbent head (score=6969.543)

### cand-139 verdict: KEPT — Full grade passes: RF-resident D1 activation removes repeated A loads, improves decode and lowers validation work.

## 2026-09-29 02:58:35 — cand-140: Preload each owner’s two 128x128 D1 W1 weight shards into its 128 KiB private SH once, then reuse them across all eight committed decode steps.
- Change: parent=cand-139; Persistent two-layer D1 W1 weight shards in SH.
- Evidence: project/experiments/cand-140/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-140/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-140/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-140/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-140/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=75.13695241592359; peak MiB=4080.2223205566406
- Inputs: compiler.py@77d2e3e5 hardware.json@d362f61f recipe.json@a1b7ade1
- Verify: PASS elapsed=75.14s
- Result: M1_P1=4959259 (6.250x), M2_D1=169389 (7.687x); score=6931.379, eligible=True
  - deltas vs head (cand-139): {'M1_P1': 0, 'M2_D1': 1850}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-140 verdict: REVERTED — Correct but slower: persistent W1 SH service plus preload loses to the 16 MiB cache path; reject and do not repeat the same-sized W2 shard experiment.

## 2026-09-29 03:00:46 — cand-141: Re-evaluate the minimal two-WG end-to-end batch-parallel P1 skeleton after reducing the mature program to roughly 78k events; no per-layer barriers, validation preflight only unless within budget.
- Change: parent=cand-139; Compressed-head batch-parallel validation-cost checkpoint.
- Evidence: project/experiments/cand-141/metrics.json; reports/grade.json; attempts=0
- Inputs: compiler.py@fc709d21 hardware.json@d362f61f recipe.json@622d3349
- Verify: FAILED — failure_class=validation_cost_pruned; evidence=pipeline
- Result: status=not_verified, failure_class=validation_cost_pruned
- Decision: reject/preserve failed candidate (status=not_verified)

### cand-141 verdict: REVERTED — Validation-cost pruned at 13.77 GB logical retained-event lower bound versus 6 GiB budget; compressed two-WG P1 remains locally unverifiable.

## 2026-09-29 03:03:11 — cand-142: Prefetch the next prompt MLP neuron panel’s first W1 A/B operands into idle RF0/RF15 during the current neuron’s W2 phase; preserve events and arithmetic.
- Change: parent=cand-139; Cross-neuron W1 seed lookahead inside RF-persistent prompt MLP.
- Evidence: project/experiments/cand-142/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-142/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-142/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-142/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-142/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=76.68808820901904; peak MiB=4113.800537109375
- Inputs: compiler.py@f8e92de2 hardware.json@d362f61f recipe.json@51179836
- Verify: PASS elapsed=76.70s
- Result: M1_P1=4935427 (6.281x), M2_D1=167539 (7.772x); score=6986.349, eligible=True
  - deltas vs head (cand-139): {'M1_P1': -23832, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=6986.349)

### cand-142 verdict: KEPT — Full grade passes: cross-neuron W1 seed lookahead gives a final small P1 gain; close the current MLP pipeline neighborhood.

## 2026-09-29 03:05:51 — cand-143: Prefetch each next P1 QKV/WO output column’s K0 B/A operands into released parity-0 RF buffers before the current column’s final stores; preserve events and arithmetic.
- Change: parent=cand-142; Cross-column seed lookahead for pipelined prompt projections.
- Evidence: project/experiments/cand-143/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/.venv/bin/python', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/challenge.py', 'grade', '--hardware', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-143/inputs/hardware.json', '--program-p1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-143/programs/M1_P1.asm', '--program-d1', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-143/programs/M2_D1.asm', '--report', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/project/experiments/cand-143/reports/grade.json', '--seed', '7', '--baseline', '/Users/yihdu/Codefield/SLAI-infra-Code/transformer-codesign-starter/baseline_manifest.json']; elapsed=76.33950995793566; peak MiB=4069.862991333008
- Inputs: compiler.py@65f5e878 hardware.json@d362f61f recipe.json@68e04bd1
- Verify: PASS elapsed=76.34s
- Result: M1_P1=4897135 (6.330x), M2_D1=167539 (7.772x); score=7013.610, eligible=True
  - deltas vs head (cand-142): {'M1_P1': -38292, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=7013.610)

### cand-143 verdict: KEPT — Full grade passes: cross-column prompt projection lookahead yields another positive P1 gain and raises the verified score above 7000.

## 2026-09-29 — Post-cand132 hardware and TC/cache exchange screens
- Change: Hold verified programs fixed while testing SH banks8, SH 2R1W, multicast, their combination, and legal TC3/cache4-or-cache2 area exchanges. Also test layer-major batch ordering with TC3/cache4.
- Verify: `project/campaigns/hardware-post-pipeline/cand132/results.json`, `tc-trade/results.json`, and `tc-trade/estimate-layer-major-tc3-cache4.json`; timing-only estimates, followed by full cand-133 grade for the only positive field.
- Result: SH banks/ports changed no cycles; multicast improved D1 207,673→207,315 and passed full grade. TC3/cache4 and TC3/cache2 scored about6104.6 proxy; layer-major+TC3/cache4 scored6139.90, all below the incumbent. TC4/8x8 is service-equivalent for the active aligned tiles; TC4/8x16 is area-ineligible.
- Decision: keep multicast only. Close the current hardware neighborhood; compute/cache area exchanges do not beat cache16+TC2.

## 2026-09-29 — cand-143 current-design saturation audit
- Change: Re-profile the final kept head and re-run the minimal two-WG batch-parallel P1 validation preflight after all event compression. No evaluator changes and no official submission.
- Verify: `project/experiments/cand-143/reports/{p1-profile.json,d1-profile.json,grade.json}`; cand-141 `reports/validation-cost.json`; full evidence summarized in `project/campaigns/current-design-limit/REPORT.md`.
- Result: cand-143 P1=4,897,135, D1=167,539, score=7013.610245, area=22.8598 mm², peak power=4.019/7.495 W, both functional passes. P1 is 50.77% prompt MLP; D1 W1/W2 are23.51%/23.56%. Compressed batch-parallel P1 still has78,032 events and13.773 GB retained-set logical lower bound versus6,144 MiB budget, so no grade was launched.
- Decision: current implemented design family is saturated. Stop tile/SH/hardware/panel sweeps; a future campaign needs a different RF layout/event-compressed algorithm or a validation environment capable of the multi-SM skeleton.

## 2026-09-29 11:18:40 — cand-144: New host has 2.8TiB RAM; re-run cand-141 minimal 2-WG P1 skeleton under raised memory budget as controlled calibration before any multi-SM speedup claims
- Change: parent=cand-141; cand-141 inputs cloned verbatim; memory_mib 6144->131072, timeout 600->7200
- Evidence: project/experiments/cand-144/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-144/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-144/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-144/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-144/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=3050.721168628894; peak MiB=88961.94140625
- Inputs: compiler.py@fc709d21 hardware.json@d362f61f recipe.json@622d3349
- Verify: PASS elapsed=3050.74s
- Result: M1_P1=2484678 (12.475x), M2_D1=167539 (7.772x); score=9846.403, eligible=True
  - deltas vs head (cand-143): {'M1_P1': -2412457, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=9846.403)

### cand-144 verdict: KEPT — First verified multi-WG P1: 2-WG batch-parallel skeleton, score 9846.40, P1 2484678 (1.97x vs cand-143). Race-checker calibration: real peak 86.9GiB = 6.5x retained-LB; 51min. New-host feasibility confirmed.

## 2026-09-29 13:04:57 — cand-145: 4-WG batch x team(2) shard: each member owned by a 2-WG team; row-sharded LN/residual, head-aligned QKV column split + head-sharded attention, column-sharded WO/W1/W2, teams independent until STEP.COMMIT. Preflight LB 9.95GiB ~ est 68GiB peak.
- Change: parent=cand-144; new p1_schedule kind=batch_shard team=2 mlp=sharded
- Evidence: project/experiments/cand-145/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-145/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-145/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-145/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-145/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=6037.860476180445; peak MiB=132569.88671875
- Inputs: compiler.py@c622a7bb hardware.json@d362f61f recipe.json@7984112a
- Verify: PASS elapsed=6037.87s
- Result: M1_P1=1586728 (19.535x), M2_D1=167539 (7.772x); score=12321.439, eligible=True
  - deltas vs head (cand-144): {'M1_P1': -897950, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=12321.439)

## 2026-09-29 16:10 — REJECTED (no candidate): batch_shard team=4 with 2 WGs per SM on 4 SMs
- Change: worktree-only; recipe p1_schedule team 2->4, Builder sm_map round-robin (8 WGs on 4 SMs). P1 95,155 events.
- Verify: `project.search.dev_functional` (race check skipped, dev only) -> passed, max err 3.0e-6. `challenge.py estimate` -> /tmp/estimate-team4.log -> CRASHED after ~40 min / ~120 GiB inside frozen `pipeline_events._issue_transfer`: `AssertionError: DMA slot changed after issue preview`.
- Result: no cycles. Frozen timing scheduler cannot evaluate this co-resident-WG DMA pattern, so any grade would fail.
- Motivation that remains valid: cand-145 busy stats show issue_wait 1.557M of 1.587M cycles per SM; rf_read 670K (42%), rf_write 606K, dma 371K, tc 102K (6%), vec 115K. TC is not the bottleneck; RF/DMA per SM are.
- Decision: drop co-residence. Next: 8 WGs on 8 SMs (one WG/SM). Area check: sm_count=8 + shared_kib=0 (neither program uses SH) + tc_count=1 + cache_mib=8 = 21.94 mm².

### cand-145 note: full grade score 12321.439 (P1 1,586,728 / D1 167,539), peak 129.5 GiB, 6038 s. cand-146 (cand-145 + 1024-col elementwise chunks, 85.7K P1 events) registered, not graded yet.

## 2026-09-29 16:30 — Analysis: frozen-timing microbenchmarks for a 10x bound model (no candidate)
- Change: none to submission. New dev tool `project/campaigns/bound-model/microbench.py` runs hand-built micro-programs through frozen `estimate_pipeline` (with trace).
- Verify: `PYTHONPATH=. pixi run python project/campaigns/bound-model/microbench.py` -> `project/campaigns/bound-model/microbench.json`.
- Result: cross-SM ST->BARRIER->LD handoff 18 cycles (1 float) / 51 (1 KiB) / 147 (4 KiB). MMA 32^3 per SM: single dependent chain 101 MAC/cycle; 2-4 interleaved chains 142-144 MAC/cycle (2tc kp4), 142 (1tc kp2), 123 (1tc kp1) -> TC over-provisioned ~7x, RF read port is the limit. GEMV m=1 MMA: 29 MAC/cycle. LD into RF: ~50-55 B/cycle per SM from HBM or SH (RF write port 64 B/cycle), single 4 KiB LD latency ~130-150 cycles.
- Decision: bound model says 10x needs ~16 SMs with small TC (1tc kp1/kp2), 2D weight/activation partition for P1, SH/RF-resident weights for D1. Retract vector_lanes=8 plan (<=7% RF-read gain).

## 2026-09-29 16:45 — Analysis: critical-path attribution of cand-145 on the frozen timing model (no candidate)
- Change: none to submission. New dev tool `project/campaigns/bound-model/critpath.py`: subclasses frozen GlobalEventPipeline with a read-only `_issue` observer (same readiness helpers), race validation skipped in-process. Asserts identical cycles.
- Verify: D1 167,539 == frozen (29 s, <1 GiB); P1 1,586,728 == frozen (419 s, ~1 GiB). Reports `project/campaigns/bound-model/critpath-c145-{d1,p1}.json`.
- Finding (memory): estimate/grade timing memory (~130 GiB) is the race validator called inside `GlobalEventPipeline.__init__` (pipeline_events.py:293), not the scheduler; scheduler alone ~1 GiB.
- Result D1 critical path: LD HBM->RF 59.7%, MMA 29.3%, VEC 6.5%, barriers 0.1%. Per-tape stalls: 83% RAW on LD. => weight streaming latency dominates; weights must become resident.
- Result P1 critical path: MMA 47.9%, LD 27.5%, VEC 15.2%, ST 4.9%, SFU 3.2%, barriers ~0. By stage: prompt W1 22.4%, W2 22.0%, GELU 12.7%, QKV 9.3%, attention 7.4%; step phase (2 rows) 16.5% total. Per-tape stalls: RAW<-LD 42%, WAR 30% (buffer reuse), RAW<-VEC 13%.
- Decision: fixable inside current structure: fuse bias+GELU into W1 epilogue (12.7%), batch the step phase across all WGs (16.5%), deeper RF buffering for WAR (30% stall). 10x still needs more SMs + small TC + resident weights.

## 2026-09-29 17:20 — Measured estimate (no candidate yet): 8 SMs, batch2 x team4, one WG per SM
- Change: `project/campaigns/p1-8sm/hardware-8sm.json` (sm_count 8, tc_count 1, kp4, cache 16->8 MiB, shared 0; 21.94 mm²), `recipe-8sm.json` p1_schedule team=4 sm_count=8. P1 95,155 events.
- Verify: dev_functional P1 passed (3.0e-6); frozen check_case D1 passed; frozen estimate_case P1 -> `project/campaigns/p1-8sm/p1-estimate.json` (3182 s); critpath observer reproduces 951,345.
- Result: P1 951,345 (1.67x vs cand-145), D1 168,363; projected score ~15.9K (estimate only, full race-validated grade not run). P1 peak window power 18.33 W (limit 20), HBM read 39.3 MB (cache 8 MiB -> misses 371K->614K lines), energy 9.0 mJ.
- Critical path: MMA 41%, LD 38% (LD did not scale: 436K->362K), step phase (2 rows) ~21% and did not scale (s_w1 102K->91K).
- Decision: power is now a hard wall (T_P1 >= E/20 W; 9.0 mJ -> >=225K cycles). Next: cut HBM energy (bytes x 150 pJ) and LD latency chains before adding SMs.

## 2026-09-29 15:40:40 — cand-146: Elementwise helper chunks 384->1024 (RF lane capacity at vector_lanes=16) cut P1 events 95.1K->85.7K; fewer events should lower race/timing validation cost and remove per-chunk issue overhead in bias_gelu/add_rows/add_bias
- Change: parent=cand-145; cand-145 + 1024-col elementwise chunks; Builder sm_map plumbing (team=2 placement unchanged). dev_functional passed max err 3.0e-6
- Evidence: project/experiments/cand-146/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-146/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-146/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-146/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-146/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=4331.682301051915; peak MiB=114770.65625
- Inputs: compiler.py@772e9676 hardware.json@d362f61f recipe.json@7984112a
- Verify: PASS elapsed=4331.69s
- Result: M1_P1=1582025 (19.593x), M2_D1=167539 (7.772x); score=12339.740, eligible=True
  - deltas vs head (cand-145): {'M1_P1': -4703, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=12339.740)

## 2026-09-29 18:40 — Dev loop (critpath observer + dev_functional): W1/W2 operand pipeline, bias+GELU epilogue fusion, B triple buffer
- Change: compiler.py adds `epilogue_bias` to generic + operand-pipeline GEMM (MMA k=1 ones x bias, in-place tanh-GELU, same op order as bias_gelu), `p1_schedule.fuse_gelu`, `gemm_stages.p1_prompt.b_buffers` (2 = byte-identical expanded stream to before; 3 rotates B via RF1/7/13). Variants in `project/campaigns/p1-fuse/<v>/` (recipe.json, M1_P1.asm, critpath.json, devfunc.json).
- Verify: all variants dev_functional pass (max err 3.0e-6). v0 critpath == cand-146 grade (1,582,025).
- Result (P1 cycles, 4 SM cand-146 hardware): v0 1,582,025; v1 W1/W2 operand pipeline 1,257,064 (-20.5%); v2 fuse GELU 1,521,116 (-3.8%); v3 both 1,198,655; v4 +step prefetch 1,111,753 (-29.7%); v5 +B triple buffer 1,106,913 (-0.4%). 8 SM (hardware-8sm.json): v4_8sm 835,415; v5_8sm 811,610 (vs 951,345 before).
- Finding: after v5 the remaining g0 stalls are MMA waiting on MMA/operands (RF-read-port-bound GEMM throughput), not buffer WAR; B-buffer WAR diagnosis was only ~0.4%.
- Decision: v4 promoted to cand-147 (grade running). Next: power check on v5_8sm, step-phase merge, D1 weight residency.

## 2026-09-29 19:40 — Dev loop: D1 8 column owners + SH-resident weights (no candidate yet)
- Change: compiler.py ParallelD1Builder: shards 8 allowed; `d1_plan` factored out; `d1_schedule.weights="sh"` copies each owner's weight panel HBM->SH once (`preload` upfront | first_use) and D1 GEMV B tiles then load SH->RF; attention owners = min(heads, shards). shards=4 output byte-identical to cand-146. Hardware `project/campaigns/d1-sh/hardware-8sm-sh256.json` (8 SM, 1 TC kp4, cache 8 MiB, shared 256 KiB x4 banks 1R1W, 22.865 mm²).
- Verify: dev_functional D1 pass for all; critpath (frozen scheduler) -> `project/campaigns/d1-sh/<v>/critpath.json`.
- Result (D1 cycles): d0 current 168,363; d8 (8 owners, HBM weights) 134,230; d8sh (SH weights, upfront) 115,019; d8shf (first_use) 114,706, peak 11.2 W. P1 v5_8sm on same SMs 811,610, peak 17.3 W, avg 5.19 W.
- Remaining D1 critical path: MMA waiting on SH->RF tiles (RF write port ~55 B/cycle), attention 25.6% (history K/V LD), preload ~11%.
- Decision: combine P1 v5_8sm + D1 d8shf + hardware-8sm-sh256 as cand-148 (projected ~20.8K from estimates; must pass full grade).

## 2026-09-29 17:16:27 — cand-147: critpath attribution: P1 W1/W2 used the non-prefetching generic GEMM (LD-RAW/WAR stalls) and GELU round-tripped HBM. Enable operand_pipeline for w1/w2, fuse bias+GELU into W1 epilogue (MMA k=1 ones x bias, then in-place tanh-GELU), prefetch step GEMMs.
- Change: parent=cand-146; cand-146 + gemm_stages.p1_prompt.operand_pipeline+=[w1,w2], p1_schedule.fuse_gelu=1, p1_step.prefetch=1. Dev: critpath P1 1,111,753 (-29.7%), dev_functional pass 3.0e-6; 81,227 events
- Evidence: project/experiments/cand-147/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-147/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-147/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-147/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-147/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=4072.812298295088; peak MiB=101581.546875
- Inputs: compiler.py@fc4e6a87 hardware.json@d362f61f recipe.json@d2df42ab
- Verify: PASS elapsed=4072.82s
- Result: M1_P1=1111753 (27.881x), M2_D1=167539 (7.772x); score=14720.024, eligible=True
  - deltas vs head (cand-146): {'M1_P1': -470272, 'M2_D1': 0}
- Decision: KEPT — new incumbent head (score=14720.024)

## 2026-09-29 18:43 — Log correction + cand-147 verdict
- Correction: the three preceding analysis entries were stamped with estimated times (16:30/16:45/17:20/18:40/19:40); their actual order is as written but the wall-clock stamps are approximate. Entries are append-only, so they are corrected here rather than edited.
- cand-147 full grade: score 14,720.024, P1 1,111,753 (identical to critpath dev prediction), D1 167,539, eligible, P1 peak 11.49 W. Kept by controller.
- cand-146 full grade: score 12,339.740, P1 1,582,025, peak RSS 112.1 GiB / 72 min (cand-145: 129.5 GiB / 100 min). Kept.

## 2026-09-29 19:13 — DMA-slot assertion root cause + dmafix; v6 merged-step results
- Analysis: `DMA slot changed after issue preview` (pipeline_events.py:701) root-caused via instrumented critpath run. Preview samples `dma_slice` calendar at *dependency-ready* time R (line 534), but issue happens at `begin = max(R, current_cycle)`. With 16 tapes, a dependency-ready LD can wait while other tapes' events issue; if pending HBM transfers' slice reservations then cover `begin+4` on every engine, the recomputed slot_ready > begin and the assertion fires. Confirmed crash: g5 LD in `pb0_resln2`, begin=196395.
- Fix (hardware menu only, frozen code untouched): `dma_depth=1` makes the assertion structurally unreachable — a free slot's engine has no pending transfer, so its slice calendar has no future reservations. Paired `dma_engines=4` to keep 4 in-flight transfers per SM (same as 2x2). Area: 23.73 mm2 (16-SM config), well under 24.
- Verify (dev): d16rf_split D1 re-timed on dmafix hw = 87,400 cycles (vs 86,560 on depth=2, +1%). P1 v5_16sm retest in progress.
- v6_8sm (merged generated step, both batch rows fused m=2): P1 critpath = 730,816 cycles (v5_8sm: 951,345), -23%. Events 81,307 -> 74,374. devfunc pass.
- v6_16sm generated (team=8 + step_merge): 74,534 events, devfunc pass. critpath running.
- cand-148 (8-SM hw + P1 v5_8sm + D1 d8shf) grade in progress, ~42 min, ~98 GiB peak.

## 2026-09-29 18:31:48 — cand-148: Hardware rebalance + dataflow: 8 SMs with 1 TC (TC ~7x over-provisioned vs RF read port), 256 KiB SH per SM; P1 batch2 x team4 one WG/SM with W1/W2 operand pipeline, GELU epilogue, B triple buffer; D1 8 column owners with weights SH-resident after first use.
- Change: parent=cand-147; hardware: sm 8, tc_count 1 kp4, cache 8 MiB, shared 256 KiB 4 banks 1R1W (22.86 mm2). Dev critpath: P1 811,610 (peak 17.3 W), D1 114,706 (peak 11.2 W); dev_functional both pass. Projected ~20.8K (estimate).
- Evidence: project/experiments/cand-148/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-148/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-148/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-148/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-148/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=4654.410382003989; peak MiB=100715.8046875
- Inputs: compiler.py@3772d72b hardware.json@5585ac1d recipe.json@643c9689
- Verify: PASS elapsed=4654.43s
- Result: M1_P1=811610 (38.192x), M2_D1=114706 (11.351x); score=20821.109, eligible=True
  - deltas vs head (cand-147): {'M1_P1': -300143, 'M2_D1': -52833}
- Decision: KEPT — new incumbent head (score=20821.109)

## 2026-09-29 19:55 — cand-148 verdict + 16-SM dmafix/power campaign
- cand-148 full grade: **score 20,821.109**, P1 811,610 / D1 114,706, eligible, peaks 17.33W/11.21W. Kept — new head.
- 16-SM (dma_depth=1, engines=4, cache8, SH128, 1x8x16 kp1 TC, 23.73mm²): v5_16sm P1=640,202 and v6_16sm P1=558,696 both run assertion-free — the dma_depth=1 fix is confirmed. Step-merge saves 12.7% at 16 SM.
- Power wall found: 8 HBM channels saturate at ~19.2W dynamic → both 16-SM P1 variants hit 20.02W (>20 limit). hbm_channels menu has no 6; **4 channels caps HBM dynamic at ~9.6W** — D1 re-timed 87,779 cyc/10.5W (+0.4%), P1 v6_16sm 563,776 cyc/10.9W (+0.9%). Power solved at ~1% cost.
- Energy attribution added to critpath (per-resource energy + peak-window composition): confirmed P1 peak window = p_qkv/p_a weight-load burst; D1 peak = initial weight preload (dlw_pre).
- cand-149 = hardware-16sm-4ch (23.53mm²) + merged recipe (P1 v6_16sm + D1 d16rf_split), programs byte-identical to validated variants. Grade launched. Projected ~28.6K (dev numbers, unverified).

## 2026-09-29 20:55 — D1 KV-history SH residency + dual accumulator; params_sh rejected
- Change (compiler.py): `ParallelD1Builder.attention_decode_split` preloads each part-WG's K/V history slice (span×hd each) into SH once; per-step reads switch HBM→RF to SH→RF. K reload preserves the transposed gather (`shape=[hd,n], strides=[1,hd]` — a flat copy produced err≈1.03, caught by bisect: V-only passed, K-only failed). New recipe keys: `d1_schedule.kv_sh` (1/"k"/"v"), `d1_schedule.acc_split` (dual accumulators break the serial MMA->acc chain, lanes 3/4), `d1_schedule.params_sh` (replicate LN/bias params into SH, ld() intercept rewrites in-region flat HBM sources to SH).
- Verify (dev): d16rf_split_kv devfunc pass (max err 2.6e-6); d16_kv_accs devfunc pass (1.4e-6). Both on hardware-16sm-4ch.
- Result (dev critpath): D1 87,779 -> **84,084** (kv_sh) -> **81,540** (kv_sh + acc_split), peak ~10.5W. Stall mix after: LD:SH->RF 332K (per-step weight streaming, RF-ingest bound), MMA.ACC 261K, dma_slot 222K, LD:HBM->RF 110K.
- REJECTED: `params_sh` (LN/bias SH residency) = 84,371 vs 84,084 — preload burst offsets per-step savings; flag left available but off.
- Structural notes: dma_engines menu caps at 4, dma_depth must stay 1 (slot-assertion) -> max 4 in-flight HBM transfers/SM; dma_slot stalls (~222K summed) are slot-capacity waits on per-step HBM ops (qkv/partials/intermediates). RF ingest is per-SM (64B/c shared by resident WGs) so 2WG/SM would not add bandwidth.
- Decision: fold d16_kv_accs into next candidate after cand-149; projected P1 563,776 / D1 81,540 -> ~29.6K.

## 2026-09-29 13:15 UTC — D1 screen d16_load_ahead
- Change: base=project/campaigns/d1-sh/d16_kv_accs/recipe.json; patch={"d1_schedule":{"load_ahead":1}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py d16_load_ahead --patch {"d1_schedule":{"load_ahead":1}}`; reports `project/campaigns/d1-sh/d16_load_ahead`. Development checks only; full frozen race validation still required.
- Result: D1=76169; P1 unchanged by recipe; area=23.53006052352; peak W=10.50048748108938; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:16 UTC — D1 screen d16_ahead_gelu
- Change: base=project/campaigns/d1-sh/d16_load_ahead/recipe.json; patch={"d1_schedule":{"fuse_gelu":1}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py d16_ahead_gelu --base project/campaigns/d1-sh/d16_load_ahead/recipe.json --patch {"d1_schedule":{"fuse_gelu":1}}`; reports `project/campaigns/d1-sh/d16_ahead_gelu`. Development checks only; full frozen race validation still required.
- Result: D1=75434; P1 unchanged by recipe; area=23.53006052352; peak W=10.50048748108938; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:22 UTC — Frozen D1 verification d16_ahead_gelu
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/d16_ahead_gelu` -> `project/campaigns/d1-sh/d16_ahead_gelu/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 13:25 UTC — D1 screen layer_rf_v1
- Change: base=project/campaigns/d1-sh/d16_ahead_gelu/recipe.json; patch={"d1_schedule":{"layer_groups":1,"shared_bytes":65536,"rf_resident":["l0wo","l0w1","l0w2","l1wo","l1w1","l1w2"]}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_v1 --base project/campaigns/d1-sh/d16_ahead_gelu/recipe.json --patch {"d1_schedule":{"layer_groups":1,"shared_bytes":65536,"rf_resident":["l0wo","l0w1","l0w2","l1wo","l1w1","l1w2"]}}`; reports `project/campaigns/d1-sh/layer_rf_v1`. Development checks only; full frozen race validation still required.
- Result: D1=74180; P1 unchanged by recipe; area=23.53006052352; peak W=10.491617081089254; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:27 UTC — M1_P1 screen merged_n16
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule":{"kind":"merged_batch"},"gemm_stages":{"p1_prompt":{"n_tile":16}}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_n16 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --patch {"p1_schedule":{"kind":"merged_batch"},"gemm_stages":{"p1_prompt":{"n_tile":16}}}`; reports `project/campaigns/structural/merged_n16`. Development checks only; full frozen race validation still required.
- Result: M1_P1=failed; area=None; peak W=None; error=ValueError('Read before write').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:28 UTC — M1_P1 screen merged_n32
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule":{"kind":"merged_batch"}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_n32 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --patch {"p1_schedule":{"kind":"merged_batch"}}`; reports `project/campaigns/structural/merged_n32`. Development checks only; full frozen race validation still required.
- Result: M1_P1=failed; area=None; peak W=None; error=ValueError('Read before write').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:31 UTC — M2_D1 screen layer_rf_tc4x8kp4
- Change: base=project/campaigns/d1-sh/layer_rf_v1/recipe.json; patch={}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={"tc_array":"4x8","tc_k_parallel":4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc4x8kp4 --base project/campaigns/d1-sh/layer_rf_v1/recipe.json --hardware-patch {"tc_array":"4x8","tc_k_parallel":4}`; reports `project/campaigns/d1-sh/layer_rf_tc4x8kp4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=62672; area=22.93345700352; peak W=10.476701993089254; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:32 UTC — M2_D1 screen layer_rf_all
- Change: base=project/campaigns/d1-sh/layer_rf_v1/recipe.json; patch={"d1_schedule":{"compact_rf":1,"rf_resident":["l0qkv","l0wo","l0w1","l0w2","l1qkv","l1wo","l1w1","l1w2"],"projections":{"qkv":{"n_tile":8,"k_chunk":128}}}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_all --base project/campaigns/d1-sh/layer_rf_v1/recipe.json --patch {"d1_schedule":{"compact_rf":1,"rf_resident":["l0qkv","l0wo","l0w1","l0w2","l1qkv","l1wo","l1w1","l1w2"],"projections":{"qkv":{"n_tile":8,"k_chunk":128}}}}`; reports `project/campaigns/d1-sh/layer_rf_all`. Development checks only; full frozen race validation still required.
- Result: M2_D1=79384; area=23.53006052352; peak W=10.49776191308942; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:33 UTC — M2_D1 screen layer_rf_all_tc4x8kp4
- Change: base=project/campaigns/d1-sh/layer_rf_all/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_all_tc4x8kp4 --base project/campaigns/d1-sh/layer_rf_all/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json`; reports `project/campaigns/d1-sh/layer_rf_all_tc4x8kp4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=64560; area=22.93345700352; peak W=10.48284682508942; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:33 UTC — M2_D1 screen layer_rf_all_qkvk32
- Change: base=project/campaigns/d1-sh/layer_rf_all/recipe.json; patch={"d1_schedule":{"projections":{"qkv":{"k_chunk":32}}}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_all_qkvk32 --base project/campaigns/d1-sh/layer_rf_all/recipe.json --patch {"d1_schedule":{"projections":{"qkv":{"k_chunk":32}}}}`; reports `project/campaigns/d1-sh/layer_rf_all_qkvk32`. Development checks only; full frozen race validation still required.
- Result: M2_D1=failed; area=None; peak W=None; error=ValueError('RF-resident tiles pack 1 or 2 per lane').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:35 UTC — M2_D1 screen layer_rf_all_qkvk32_v2
- Change: base=project/campaigns/d1-sh/layer_rf_all/recipe.json; patch={"d1_schedule":{"projections":{"qkv":{"k_chunk":32}}}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_all_qkvk32_v2 --base project/campaigns/d1-sh/layer_rf_all/recipe.json --patch {"d1_schedule":{"projections":{"qkv":{"k_chunk":32}}}}`; reports `project/campaigns/d1-sh/layer_rf_all_qkvk32_v2`. Development checks only; full frozen race validation still required.
- Result: M2_D1=76852; area=23.53006052352; peak W=10.491617081089254; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:37 UTC — M2_D1 screen layer_rf_shpre
- Change: base=project/campaigns/d1-sh/layer_rf_v1/recipe.json; patch={"d1_schedule":{"rf_preload_sh":1}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_shpre --base project/campaigns/d1-sh/layer_rf_v1/recipe.json --patch {"d1_schedule":{"rf_preload_sh":1}}`; reports `project/campaigns/d1-sh/layer_rf_shpre`. Development checks only; full frozen race validation still required.
- Result: M2_D1=75327; area=23.53006052352; peak W=10.460384953089202; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 13:37 UTC — M2_D1 screen layer_rf_shpre_tc
- Change: base=project/campaigns/d1-sh/layer_rf_v1/recipe.json; patch={"d1_schedule":{"rf_preload_sh":1}}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_shpre_tc --base project/campaigns/d1-sh/layer_rf_v1/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json --patch {"d1_schedule":{"rf_preload_sh":1}}`; reports `project/campaigns/d1-sh/layer_rf_shpre_tc`. Development checks only; full frozen race validation still required.
- Result: M2_D1=63795; area=22.93345700352; peak W=10.445469865089201; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 20:11:29 — cand-149: 16-SM rebalance: 1x8x16 kp1 TC, cache8, SH128, dma_depth1+engines4 (frozen-scheduler DMA-slot fix), hbm_channels=4 (power cap); P1 v6_16sm merged-step team8, D1 d16rf_split 16-owner split-K + RF resident
- Change: parent=cand-148; dev: P1 563776 cyc/10.9W, D1 87779 cyc/10.5W, projected score ~28.6K
- Evidence: project/experiments/cand-149/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=5147.903912968002; peak MiB=78148.6328125
- Inputs: compiler.py@651e1ab3 hardware.json@bf2d0822 recipe.json@bf54e8e4
- Verify: PASS elapsed=5147.93s
- Result: M1_P1=563776 (54.981x), M2_D1=87779 (14.833x); score=28557.621, eligible=True
  - deltas vs head (cand-148): {'M1_P1': -247834, 'M2_D1': -26927}
- Decision: KEPT — new incumbent head (score=28557.621)

## 2026-09-29 13:39 UTC — Frozen D1 verification layer_rf_tc4x8kp4
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/layer_rf_tc4x8kp4` -> `project/campaigns/d1-sh/layer_rf_tc4x8kp4/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 13:39 UTC — Frozen D1 verification layer_rf_v1
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/layer_rf_v1` -> `project/campaigns/d1-sh/layer_rf_v1/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 13:46 UTC — Structural campaign decisions and cand-150 launch
- Change: D1 per-layer RF ownership; compact all-weight RF and SH-assisted RF preload alternatives; P1 merged-batch and neuron-sharded RF MLP families. Isolated inputs and status: `project/campaigns/structural/STATUS.md`.
- Verify: D1 layer_rf_v1 and layer_rf_tc4x8kp4 both pass frozen check_case seeds7/19 and estimate_case. Five P1 variants pass development checks seeds7/19; timing ongoing. `traffic.py` -> `project/campaigns/structural/traffic.json` is a static requested-byte audit, not a timing result.
- Result: same-hardware D1=74180; alternative-TC D1=62672. All-weight RF variants (79384/76852; TC variant64560) and SH-assisted preload (75327; TC variant63795) rejected against their matched hardware reference. P1 timing results pending.
- Prior pending experiment recovered: NoC512 P1=550728 peak11.2723W area23.6581; reports moved into `project/campaigns/p1-fuse/v6_16sm/critpath-4ch-noc512.json` and `project/campaigns/d1-sh/d16_kv_accs/critpath-4ch-noc512.json` (D1=78040). Development timing only.
- Decision: cand-149 full grade completed eligible score28557.620577, and root hardware/recipe/programs now reproduce its exact assembly while retaining new compiler capabilities. Made KV SH residency opt-in (missing flag means off), restoring cand-149 source compatibility; all new candidates explicitly set kv_sh=1. Frozen files unchanged. Registered cand-150 with byte-identical parent P1 and frozen-verified layer_rf_v1 D1; controller full grade running, projected31065.18. Alternative TC awaits matched P1 timing before acceptance.

## 2026-09-29 13:49 UTC — M1_P1 screen merged_n32_v2
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule":{"kind":"merged_batch"}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_n32_v2 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --patch {"p1_schedule":{"kind":"merged_batch"}}`; reports `project/campaigns/structural/merged_n32_v2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=608136; area=23.53006052352; peak W=11.02602170099784; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:00 UTC — M1_P1 screen merged_neuron_mlp
- Change: base=project/campaigns/structural/merged_n16_v2/recipe.json; patch={"p1_schedule":{"neuron_mlp":1}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_neuron_mlp --case M1_P1 --out-root project/campaigns/structural --base project/campaigns/structural/merged_n16_v2/recipe.json --patch {"p1_schedule":{"neuron_mlp":1}}`; reports `project/campaigns/structural/merged_neuron_mlp`. Development checks only; full frozen race validation still required.
- Result: M1_P1=716850; area=23.53006052352; peak W=10.567552425661813; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:02 UTC — M1_P1 screen parent_tc4x8kp4
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py parent_tc4x8kp4 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json`; reports `project/campaigns/structural/parent_tc4x8kp4`. Development checks only; full frozen race validation still required.
- Result: M1_P1=561565; area=22.93345700352; peak W=10.987744390420774; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:09 UTC — M1_P1 screen merged_neuron_rfpanel
- Change: base=project/campaigns/structural/merged_neuron_mlp/recipe.json; patch={"p1_schedule":{"neuron_mlp":2}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_neuron_rfpanel --case M1_P1 --out-root project/campaigns/structural --base project/campaigns/structural/merged_neuron_mlp/recipe.json --patch {"p1_schedule":{"neuron_mlp":2}}`; reports `project/campaigns/structural/merged_neuron_rfpanel`. Development checks only; full frozen race validation still required.
- Result: M1_P1=753279; area=23.53006052352; peak W=10.567552425666104; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:15 UTC — M1_P1 screen merged_n16_v2
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule":{"kind":"merged_batch"},"gemm_stages":{"p1_prompt":{"n_tile":16}}}; hardware=project/campaigns/d1-sh/hardware-16sm-4ch.json
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py merged_n16_v2 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --patch {"p1_schedule":{"kind":"merged_batch"},"gemm_stages":{"p1_prompt":{"n_tile":16}}}`; reports `project/campaigns/structural/merged_n16_v2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=764661; area=23.53006052352; peak W=10.56755242566205; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:19 UTC — M2_D1 screen layer_rf_tc_noc512
- Change: base=project/campaigns/d1-sh/layer_rf_tc4x8kp4/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={"noc_bytes_per_cycle":512}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc_noc512 --base project/campaigns/d1-sh/layer_rf_tc4x8kp4/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json --hardware-patch {"noc_bytes_per_cycle":512}`; reports `project/campaigns/d1-sh/layer_rf_tc_noc512`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58770; area=23.06145700352; peak W=10.77603965708452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:21 UTC — M2_D1 screen layer_rf_tc_params
- Change: base=project/campaigns/d1-sh/layer_rf_tc4x8kp4/recipe.json; patch={"d1_schedule":{"rf_params":1}}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc_params --base project/campaigns/d1-sh/layer_rf_tc4x8kp4/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json --patch {"d1_schedule":{"rf_params":1}}`; reports `project/campaigns/d1-sh/layer_rf_tc_params`. Development checks only; full frozen race validation still required.
- Result: M2_D1=62714; area=22.93345700352; peak W=10.476701993089254; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:24 UTC — Structural round verdicts and next targeted tests
- Change: close cross-batch/MLP round; retain common-hardware TC4x8 kp4; add isolated P1 generated-step and prompt-W1 RF-input residency tests, plus NoC512. Reports/status under `project/campaigns/structural/`.
- Verify: all four P1 restructures passed dev seeds7/19 but lost in timing. P1 parent on TC4x8 kp4 =561565, D1 frozen verified=62672; same shared hardware area22.93346, peak10.98774/10.47670 W. Joint CLI grade launched under `project/campaigns/d1-sh/layer_rf_tc4x8kp4/joint/`, predicted score33863.655 (not yet joint verified). Cand-150 remains independently in full grade.
- Result: reject merged N16=764661, merged N32=608136, neuron SH-stream=716850, neuron RF-panel=753279 versus baseline P1=563776. Keep code as experimental options and retain all reports. RF-parameter residency D1=62714 vs62672 is also rejected. NoC512 D1=58770 is promising, frozen verification launched; P1 timing pending.
- Next: step_a_resident_tc and prompt_w1_a_resident_tc both pass dev seeds7/19 and are timing. Static `traffic-resident.json`: requested HBM->RF bytes 74.48MB/72.34MB vs75.46MB; no cycles claimed from byte counts. Latest compiler still reproduces both cand-149 programs byte-identically with its recipe; `git diff --check` passes; frozen evaluator unchanged.

## 2026-09-29 14:27 UTC — Frozen D1 verification layer_rf_tc_noc512
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/layer_rf_tc_noc512` -> `project/campaigns/d1-sh/layer_rf_tc_noc512/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 14:45 UTC — M1_P1 screen prompt_w1_a_resident_tc
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages":{"p1_prompt":{"a_resident":["w1"]}}}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py prompt_w1_a_resident_tc --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json --patch {"gemm_stages":{"p1_prompt":{"a_resident":["w1"]}}}`; reports `project/campaigns/structural/prompt_w1_a_resident_tc`. Development checks only; full frozen race validation still required.
- Result: M1_P1=600392; area=22.93345700352; peak W=10.987744390420774; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:46 UTC — M1_P1 screen parent_tc_noc512
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py parent_tc_noc512 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json`; reports `project/campaigns/structural/parent_tc_noc512`. Development checks only; full frozen race validation still required.
- Result: M1_P1=560845; area=23.06145700352; peak W=11.323344161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 14:47 UTC — M1_P1 screen step_a_resident_tc
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages":{"p1_step":{"a_resident":1}}}; hardware=project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py step_a_resident_tc --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc4x8kp4/hardware.json --patch {"gemm_stages":{"p1_step":{"a_resident":1}}}`; reports `project/campaigns/structural/step_a_resident_tc`. Development checks only; full frozen race validation still required.
- Result: M1_P1=558176; area=22.93345700352; peak W=10.987744390420774; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 21:41:36 — cand-150: D1 per-layer persistent RF groups with WO/W1/W2 resident in each layer; KV SH, dual accumulators, load-ahead and fused GELU. Frozen D1 seeds7/19 and estimate pass:74180 cycles. P1 and hardware byte-identical to graded cand-149.
- Change: parent=cand-149; Isolated structural search; sources in proposal.json
- Evidence: project/experiments/cand-150/metrics.json; reports/grade.json; attempts=1
- Command (grade): ['/home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python', '/home/jovyan/projects/SLAI-infra-course-hw1/challenge.py', 'grade', '--hardware', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-150/inputs/hardware.json', '--program-p1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-150/programs/M1_P1.asm', '--program-d1', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-150/programs/M2_D1.asm', '--report', '/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-150/reports/grade.json', '--seed', '7', '--baseline', '/home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json']; elapsed=5187.866576676723; peak MiB=78147.64453125
- Inputs: compiler.py@9a853643 hardware.json@f243db04 recipe.json@74670c77
- Verify: PASS elapsed=5187.90s
- Result: M1_P1=563776 (54.981x), M2_D1=74180 (17.552x); score=31065.183, eligible=True
  - deltas vs head (cand-149): {'M1_P1': 0, 'M2_D1': -13599}
- Decision: KEPT — new incumbent head (score=31065.183)

## 2026-09-29 23:15:46 — cand-151: Shared TC4x8 kp4 and NoC512: screened P1 560845, frozen D1 58770; validate joint score and all gates.
- Change: parent=cand-150; Isolated structural search; sources in proposal.json
- Evidence: project/experiments/cand-151/metrics.json; reports/grade.json; attempts=0
- Inputs: compiler.py@c423ffce hardware.json@9431f1d4 recipe.json@74670c77
- Verify: FAILED — failure_class=validation_cost_pruned; evidence=pipeline
- Result: status=not_verified, failure_class=validation_cost_pruned
- Decision: reject/preserve failed candidate (status=not_verified)

## 2026-09-29 15:16 UTC — M2_D1 screen layer_rf_tc_noc512_vec32
- Change: base=project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"vector_lanes":32}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc_noc512_vec32 --base project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --hardware-patch '{"vector_lanes":32}'`; reports `project/campaigns/d1-sh/layer_rf_tc_noc512_vec32`. Development checks only; full frozen race validation still required.
- Result: M2_D1=failed; area=None; peak W=None; error=ValueError('RF address/scope invalid').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:16 UTC — M2_D1 screen layer_rf_tc_noc512_sh2r
- Change: base=project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_ports":"2R1W"}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc_noc512_sh2r --base project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --hardware-patch '{"shared_ports":"2R1W"}'`; reports `project/campaigns/d1-sh/layer_rf_tc_noc512_sh2r`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57794; area=23.419857003519997; peak W=10.784999657084503; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:17 UTC — M2_D1 screen layer_rf_tc_noc512_sh8
- Change: base=project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_banks":8}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py layer_rf_tc_noc512_sh8 --base project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --hardware-patch '{"shared_banks":8}'`; reports `project/campaigns/d1-sh/layer_rf_tc_noc512_sh8`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57658; area=23.57345700352; peak W=10.78883965708451; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:22 UTC — Frozen D1 verification layer_rf_tc_noc512_sh2r
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/layer_rf_tc_noc512_sh2r` -> `project/campaigns/d1-sh/layer_rf_tc_noc512_sh2r/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 15:23 UTC — Launch aggressive-r1 bounded campaign
- Change: 26 pinned hypotheses, four evaluation workers; hardware resource reallocations and D1/P1 structural/tiling branches. Add pinned compiler option and area prefilter to isolated screen runner. No evaluator changes.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/aggressive.py aggressive-r1 --jobs 4`; syntax compilation and plan-only checks passed. Live evidence: `project/campaigns/structural/aggressive-r1/state.json`, per-candidate reports and stdout/stderr.
- Result: running; no new joint score claimed. Formal registry incumbent remains cand-150 score31065.183.
- Decision: automatically continue through top-three frozen D1 checks, at most nine same-hardware P1 measurements, then at most two sequential full joint grades. Retain all failed cases; no unverified promotion.

## 2026-09-29 15:23 UTC — M2_D1 screen cache0
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 0}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 0}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57057; area=19.58521098752; peak W=10.36964627468937; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:23 UTC — M2_D1 screen cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache4 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58770; area=21.38333399552; peak W=10.73408658188452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:23 UTC — M2_D1 screen tc2_kp2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_count": 2, "tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc2_kp2 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_count": 2, "tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/tc2_kp2`. Development checks only; full frozen race validation still required.
- Result: M2_D1=61278; area=23.26032484352; peak W=10.78101135308454; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:23 UTC — M2_D1 screen hbm8_cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"hbm_channels": 8, "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_cache4 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"hbm_channels": 8, "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_cache4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58250; area=21.58333399552; peak W=18.78186744588858; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:24 UTC — Frozen D1 verification layer_rf_tc_noc512_sh8
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py project/campaigns/d1-sh/layer_rf_tc_noc512_sh8` -> `project/campaigns/d1-sh/layer_rf_tc_noc512_sh8/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 15:24 UTC — M2_D1 screen tc4_kp1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_count": 4, "tc_k_parallel": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc4_kp1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_count": 4, "tc_k_parallel": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/tc4_kp1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=66422; area=23.65806052352; peak W=10.790954745084594; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:24 UTC — M2_D1 screen tc8x8_kp2_hbm8
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_array": "8x8", "tc_k_parallel": 2, "hbm_channels": 8, "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x8_kp2_hbm8 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x8", "tc_k_parallel": 2, "hbm_channels": 8, "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/tc8x8_kp2_hbm8`. Development checks only; full frozen race validation still required.
- Result: M2_D1=64026; area=21.78220183552; peak W=18.786839141888592; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:24 UTC — M2_D1 screen tc8x8_kp2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_array": "8x8", "tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x8_kp2 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x8", "tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/tc8x8_kp2`. Development checks only; full frozen race validation still required.
- Result: M2_D1=64566; area=23.26032484352; peak W=10.781011353084265; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:24 UTC — M2_D1 screen sh8_2r_cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=56978; area=22.61213399552; peak W=10.76480658188449; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen hbm8_sh8_cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"hbm_channels": 8, "shared_banks": 8, "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_sh8_cache4 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"hbm_channels": 8, "shared_banks": 8, "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57084; area=22.09533399552; peak W=18.794667445888592; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen qkv_k16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"qkv": {"k_chunk": 16}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py qkv_k16 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"qkv": {"k_chunk": 16}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/qkv_k16`. Development checks only; full frozen race validation still required.
- Result: M2_D1=59291; area=23.06145700352; peak W=10.776039657082642; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen w1_k16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"w1": {"k_chunk": 16}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py w1_k16 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"w1": {"k_chunk": 16}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/w1_k16`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58575; area=23.06145700352; peak W=10.776039657084274; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen qkv_k8
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"qkv": {"k_chunk": 8}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py qkv_k8 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"qkv": {"k_chunk": 8}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/qkv_k8`. Development checks only; full frozen race validation still required.
- Result: M2_D1=59646; area=23.06145700352; peak W=10.776039657082627; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen w2_k64
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"w2": {"k_chunk": 64}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py w2_k64 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"w2": {"k_chunk": 64}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/w2_k64`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58431; area=23.06145700352; peak W=10.563939017087021; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen wo_k128
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"wo": {"k_chunk": 128}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py wo_k128 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"wo": {"k_chunk": 128}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/wo_k128`. Development checks only; full frozen race validation still required.
- Result: M2_D1=59052; area=23.06145700352; peak W=10.77603965708452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:25 UTC — M2_D1 screen wo_k32
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"projections": {"wo": {"k_chunk": 32}}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py wo_k32 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"projections": {"wo": {"k_chunk": 32}}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/wo_k32`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58770; area=23.06145700352; peak W=10.776039657084489; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:26 UTC — M2_D1 screen kv_hbm
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"kv_sh": 0}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py kv_hbm --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"kv_sh": 0}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/kv_hbm`. Development checks only; full frozen race validation still required.
- Result: M2_D1=62322; area=23.06145700352; peak W=10.776039657084564; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:26 UTC — M2_D1 screen rf_w2_only
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"rf_resident": ["l0w2", "l1w2"]}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py rf_w2_only --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"rf_resident": ["l0w2", "l1w2"]}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/rf_w2_only`. Development checks only; full frozen race validation still required.
- Result: M2_D1=62837; area=23.06145700352; peak W=10.776039657084398; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:26 UTC — M2_D1 screen kv_k_only
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"kv_sh": "k"}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py kv_k_only --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"kv_sh": "k"}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/kv_k_only`. Development checks only; full frozen race validation still required.
- Result: M2_D1=60924; area=23.06145700352; peak W=10.77603965708454; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:26 UTC — M2_D1 screen kv_v_only
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"kv_sh": "v"}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py kv_v_only --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"kv_sh": "v"}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/kv_v_only`. Development checks only; full frozen race validation still required.
- Result: M2_D1=60934; area=23.06145700352; peak W=10.77603965708454; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:26 UTC — M1_P1 screen p1_k16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages": {"p1_prompt": {"k_chunk": 16}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_k16 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"k_chunk": 16}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_k16`. Development checks only; full frozen race validation still required.
- Result: M1_P1=failed; area=None; peak W=None; error=ValueError('P1 operand pipeline requires M64/128 Mtile32 N16/32 K32').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:27 UTC — M2_D1 screen attention_unsplit
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={"d1_schedule": {"attention_split": 0}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py attention_unsplit --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"d1_schedule": {"attention_split": 0}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/attention_unsplit`. Development checks only; full frozen race validation still required.
- Result: M2_D1=75208; area=23.06145700352; peak W=10.776039657084825; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:39 UTC — M1_P1 screen p1_team4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule": {"team": 4}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_team4 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"p1_schedule": {"team": 4}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_team4`. Development checks only; full frozen race validation still required.
- Result: M1_P1=875592; area=23.06145700352; peak W=10.523687286426135; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:43 UTC — Joint grade layer_rf_tc4x8kp4
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/d1-sh/layer_rf_tc4x8kp4/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/d1-sh/layer_rf_tc4x8kp4/joint` -> `project/campaigns/d1-sh/layer_rf_tc4x8kp4/joint/grade.json`.
- Result: status=completed; score=33863.65511607333; eligible=True; cycles={'M1_P1': 561565, 'M2_D1': 62672}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-29 15:44 UTC — M1_P1 screen step_a_resident_tc_noc512
- Change: base=project/campaigns/structural/step_a_resident_tc/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py step_a_resident_tc_noc512 --case M1_P1 --out-root project/campaigns/structural --base project/campaigns/structural/step_a_resident_tc/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json`; reports `project/campaigns/structural/step_a_resident_tc_noc512`. Development checks only; full frozen race validation still required.
- Result: M1_P1=556412; area=23.06145700352; peak W=11.323344161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:45 UTC — M1_P1 screen parent_tc_noc512_sh2r
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512_sh2r/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py parent_tc_noc512_sh2r --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512_sh2r/hardware.json`; reports `project/campaigns/structural/parent_tc_noc512_sh2r`. Development checks only; full frozen race validation still required.
- Result: M1_P1=560845; area=23.419857003519997; peak W=11.332304161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:46 UTC — M1_P1 screen parent_tc_noc512_sh8
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512_sh8/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py parent_tc_noc512_sh8 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512_sh8/hardware.json`; reports `project/campaigns/structural/parent_tc_noc512_sh8`. Development checks only; full frozen race validation still required.
- Result: M1_P1=560845; area=23.57345700352; peak W=11.336144161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:54 UTC — M1_P1 screen p1_step_rf
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages": {"p1_step": {"a_resident": 1}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_step_rf --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages": {"p1_step": {"a_resident": 1}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf`. Development checks only; full frozen race validation still required.
- Result: M1_P1=556412; area=23.06145700352; peak W=11.323344161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 15:57 UTC — M1_P1 screen p1_b2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages": {"p1_prompt": {"b_buffers": 2}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_b2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"b_buffers": 2}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=587568; area=23.06145700352; peak W=11.184294958662221; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:08 UTC — M1_P1 screen p1_n16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages": {"p1_prompt": {"n_tile": 16}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_n16 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"n_tile": 16}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_n16`. Development checks only; full frozen race validation still required.
- Result: M1_P1=864402; area=23.06145700352; peak W=10.683927591876843; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:12 UTC — M1_P1 screen p1_step_unmerged
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={"p1_schedule": {"step_merge": 0}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_step_unmerged --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"p1_schedule": {"step_merge": 0}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_unmerged`. Development checks only; full frozen race validation still required.
- Result: M1_P1=641065; area=23.06145700352; peak W=11.323344161093543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:17 UTC — Frozen D1 verification sh8_2r_cache4
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4` -> `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 16:23 UTC — Frozen D1 verification cache0
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0` -> `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 16:28 UTC — Frozen D1 verification hbm8_sh8_cache4
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4` -> `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 16:41 UTC — M1_P1 screen cache0__parent
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0__parent --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0__parent`. Development checks only; full frozen race validation still required.
- Result: M1_P1=715822; area=19.58521098752; peak W=11.088668194682114; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:44 UTC — Joint grade layer_rf_tc_noc512
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/d1-sh/layer_rf_tc_noc512/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/d1-sh/layer_rf_tc_noc512/joint` -> `project/campaigns/d1-sh/layer_rf_tc_noc512/joint/grade.json`.
- Result: status=completed; score=34992.20854977672; eligible=True; cycles={'M1_P1': 560845, 'M2_D1': 58770}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-29 16:45 UTC — Launch P1 paired W1 input reuse
- Change: opt-in `gemm_stages.p1_prompt.paired_n=["w1"]` emits two adjacent W1 output panels per LN2 input load, with RF accumulators for both 32-row blocks and alternating B weight prefetch. The default recipe still reproduces cand-149 P1/D1 byte for byte.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_paired_w1_noc512 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages":{"p1_prompt":{"paired_n":["w1"]}}}'` -> `project/campaigns/structural/p1_paired_w1_noc512/`; development functional seeds 7/19 passed, timing pending. Static audit `project/campaigns/structural/p1-paired-traffic.json`.
- Result: requested HBM->RF 69.17 MB versus 75.46 MB parent; 72998 versus 74534 instructions. No cycle claim yet.
- Decision: retain as a live experiment; compare timing with P1=560845 on the exact NoC512 hardware, then run frozen joint validation only if beneficial.

## 2026-09-29 16:47 UTC — Stage best fully graded triple
- Change: set root `hardware.json` to NoC512, TC4x8 kp4; set `project/recipe.json` to layer-group D1 with RF-resident WO/W1/W2, SH KV and fused GELU; regenerate both assembly programs. Preserve the experimental paired W1 generator as an opt-in branch.
- Verify: frozen joint grade `project/campaigns/d1-sh/layer_rf_tc_noc512/joint/grade.json` eligible=True; root hardware JSON semantically matches its snapshot, recipe and both emitted programs match it byte for byte. `git diff --check` passed. Area=23.061457 mm².
- Result: P1=560845, D1=58770, score=34992.208550, peak powers=11.3233/10.7760 W.
- Decision: keep as current verified submission while more aggressive P1 dataflow and joint search run separately.

## 2026-09-29 16:54 UTC — M1_P1 screen cache0__p1_step_rf
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0__p1_step_rf --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0__p1_step_rf`. Development checks only; full frozen race validation still required.
- Result: M1_P1=705418; area=19.58521098752; peak W=11.088668194682114; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:56 UTC — M1_P1 screen sh8_2r_cache4__parent
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4__parent --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4__parent`. Development checks only; full frozen race validation still required.
- Result: M1_P1=560968; area=22.61213399552; peak W=11.312111085893543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:56 UTC — M1_P1 screen sh8_2r_cache4__p1_step_rf
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4__p1_step_rf --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4__p1_step_rf`. Development checks only; full frozen race validation still required.
- Result: M1_P1=556591; area=22.61213399552; peak W=11.312111085893543; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:59 UTC — M1_P1 screen p1_paired_w1_noc512
- Change: base=project/experiments/cand-149/inputs/recipe.json; patch={"gemm_stages":{"p1_prompt":{"paired_n":["w1"]}}}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_paired_w1_noc512 --case M1_P1 --out-root project/campaigns/structural --base project/experiments/cand-149/inputs/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages":{"p1_prompt":{"paired_n":["w1"]}}}'`; reports `project/campaigns/structural/p1_paired_w1_noc512`. Development checks only; full frozen race validation still required.
- Result: M1_P1=552651; area=23.06145700352; peak W=11.236964777086685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 16:59 UTC — M1_P1 screen sh8_2r_cache4__p1_b2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4__p1_b2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/sh8_2r_cache4__p1_b2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=587691; area=22.61213399552; peak W=11.173061883462221; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 17:02 UTC — Joint frozen grade launched for paired W1
- Change: combine paired W1 P1=552651 (development timing) with frozen verified D1=58770 under one exact NoC512 hardware. Snapshot at `project/campaigns/structural/p1_paired_w1_noc512/joint/` asserts both generated programs equal their measured source programs.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/p1_paired_w1_noc512/joint` -> `grade-run.json`, running. Full grade will append its result to this log.
- Result: projected score 35250.664; eligibility pending frozen validation.
- Decision: await full grade; keep root on previously eligible 34992.209 triple. Concurrent paired W1 + step RF screen passed dev seeds7/19 and is timing.

## 2026-09-29 17:07 UTC — M1_P1 screen cache0__p1_b2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0__p1_b2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/cache0__p1_b2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=737309; area=19.58521098752; peak W=11.088931938685768; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 17:24 UTC — M1_P1 screen p1_paired_w1_step_rf_noc512
- Change: base=project/campaigns/structural/p1_paired_w1_noc512/recipe.json; patch={"gemm_stages":{"p1_step":{"a_resident":1}}}; hardware=project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py p1_paired_w1_step_rf_noc512 --case M1_P1 --out-root project/campaigns/structural --base project/campaigns/structural/p1_paired_w1_noc512/recipe.json --hardware project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{"gemm_stages":{"p1_step":{"a_resident":1}}}'`; reports `project/campaigns/structural/p1_paired_w1_step_rf_noc512`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548218; area=23.06145700352; peak W=11.236964777086685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 17:25 UTC — M1_P1 screen hbm8_sh8_cache4__parent
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_sh8_cache4__parent --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-149/inputs/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4__parent`. Development checks only; full frozen race validation still required.
- Result: M1_P1=565766; area=22.09533399552; peak W=20.32819710358599; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 17:26 UTC — M1_P1 screen hbm8_sh8_cache4__p1_step_rf
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_sh8_cache4__p1_step_rf --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_step_rf/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4__p1_step_rf`. Development checks only; full frozen race validation still required.
- Result: M1_P1=560209; area=22.09533399552; peak W=20.32819710358599; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 17:33 UTC — M1_P1 screen hbm8_sh8_cache4__p1_b2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_sh8_cache4__p1_b2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/p1_b2/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4/hardware.json --patch '{}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/aggressive-r1/hbm8_sh8_cache4__p1_b2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=566879; area=22.09533399552; peak W=19.530123083949924; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:05 UTC — M1_P1 screen step_k16_n64
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_step": {"k_chunk": 16, "n_tile": 64}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py step_k16_n64 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_step": {"k_chunk": 16, "n_tile": 64}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/step_k16_n64`. Development checks only; full frozen race validation still required.
- Result: M1_P1=failed; area=None; peak W=None; error=ValueError('merged step needs head-aligned QKV column tiles').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:06 UTC — Launch paired W1 large search
- Change: pin the paired W1 + step RF generator and 27 hypotheses under `project/campaigns/structural/p1-pair-r1/`; vary W1 preload order/K tile, prompt/step program settings, cache/TC/DMA/NoC/SH hardware within the legal menu. Six worker processes screen candidates.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/p1_pair_campaign.py p1-pair-r1 --workers 6` -> live `state.json` plus per-candidate reports. Compiler syntax, 27-proposal plan and byte-exact reproduction of the screened parent passed before launch.
- Result: running; current best fully graded score remains 34992.209. The first N64 step proposal was rejected by head-alignment validation and logged above.
- Decision: complete all screens, frozen-verify top three D1 configurations, then full-grade at most two joint finalists. Keep complete evidence and promote only an eligible grade.

## 2026-09-29 18:12 UTC — User-directed stopping rule for current round
- Change: freeze search scope to the already running aggressive-r1 and p1-pair-r1 campaigns plus two existing standalone paired-W1 joint grades. Do not propose another round. Launch `project/campaigns/structural/finalize_round.py` to wait for terminal reports, compare all eligible complete grades, and stage the highest score.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/finalize_round.py` -> `project/campaigns/structural/current-round-finalization/status.json`. The root's four submission artifact hashes are pinned to prevent overwriting later user edits; a temporary-file apply_patch probe succeeded.
- Result: waiting; highest fully eligible grade so far is 34992.208550. Final result will be appended to this log after selection.
- Decision: finish this round and stop; update root only from a byte-reproducible, eligible frozen grade.

## 2026-09-29 18:16 UTC — Joint grade p1_paired_w1_noc512
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/p1_paired_w1_noc512/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/p1_paired_w1_noc512/joint` -> `project/campaigns/structural/p1_paired_w1_noc512/joint/grade.json`.
- Result: status=completed; score=35250.66385124824; eligible=True; cycles={'M1_P1': 552651, 'M2_D1': 58770}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-29 18:30 UTC — M1_P1 screen paired_late_b
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_prompt": {"paired_b_prefetch": 0}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py paired_late_b --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"paired_b_prefetch": 0}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/paired_late_b`. Development checks only; full frozen race validation still required.
- Result: M1_P1=526494; area=23.06145700352; peak W=11.157735073089668; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:30 UTC — M1_P1 screen paired_k16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_prompt": {"paired_k_chunk": 16}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py paired_k16 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"paired_k_chunk": 16}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/paired_k16`. Development checks only; full frozen race validation still required.
- Result: M1_P1=541622; area=23.06145700352; peak W=11.280452105089136; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:30 UTC — M1_P1 screen step_k16_n32
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_step": {"k_chunk": 16, "n_tile": 32}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py step_k16_n32 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_step": {"k_chunk": 16, "n_tile": 32}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/step_k16_n32`. Development checks only; full frozen race validation still required.
- Result: M1_P1=545922; area=23.06145700352; peak W=11.236964777086685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:31 UTC — M1_P1 screen paired_preload_a
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_prompt": {"paired_a_preload": 1}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py paired_preload_a --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"paired_a_preload": 1}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/paired_preload_a`. Development checks only; full frozen race validation still required.
- Result: M1_P1=545365; area=23.06145700352; peak W=11.140463990180322; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:31 UTC — M1_P1 screen step_k32_n16
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_step": {"k_chunk": 32, "n_tile": 16}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py step_k32_n16 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_step": {"k_chunk": 32, "n_tile": 16}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/step_k32_n16`. Development checks only; full frozen race validation still required.
- Result: M1_P1=610952; area=23.06145700352; peak W=11.236964777086685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:32 UTC — M1_P1 screen prompt_b2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_prompt": {"b_buffers": 2}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py prompt_b2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"b_buffers": 2}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/prompt_b2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=549503; area=23.06145700352; peak W=10.952510649088927; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:44 UTC — M1_P1 screen cache0
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"cache_mib": 0}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 0}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache0`. Development checks only; full frozen race validation still required.
- Result: M1_P1=654584; area=19.58521098752; peak W=11.331714562667601; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:44 UTC — M2_D1 screen cache0__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 0}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache0__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 0}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache0__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57057; area=19.58521098752; peak W=10.36964627468937; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:54 UTC — M1_P1 screen w2_no_operand_pipeline
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={"gemm_stages": {"p1_prompt": {"operand_pipeline": ["qkv", "wo", "w1"]}}}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py w2_no_operand_pipeline --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{"gemm_stages": {"p1_prompt": {"operand_pipeline": ["qkv", "wo", "w1"]}}}' --hardware-patch '{}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/w2_no_operand_pipeline`. Development checks only; full frozen race validation still required.
- Result: M1_P1=611064; area=23.06145700352; peak W=11.128816158426114; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:55 UTC — M1_P1 screen cache2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"cache_mib": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=549185; area=20.54427249152; peak W=11.174035164286686; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:55 UTC — M1_P1 screen cache1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"cache_mib": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache1 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache1`. Development checks only; full frozen race validation still required.
- Result: M1_P1=549159; area=20.12474173952; peak W=11.089792599487993; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:55 UTC — M1_P1 screen cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache4 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache4`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548397; area=21.38333399552; peak W=11.195011701886687; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:55 UTC — M2_D1 screen cache2__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache2__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache2__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58770; area=20.54427249152; peak W=10.71311004428452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:55 UTC — M2_D1 screen cache1__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache1__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache1__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58808; area=20.12474173952; peak W=10.70262177548452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:56 UTC — M2_D1 screen cache4__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py cache4__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/cache4__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58770; area=21.38333399552; peak W=10.73408658188452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:58 UTC — M1_P1 screen noc256
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"noc_bytes_per_cycle": 256}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py noc256 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"noc_bytes_per_cycle": 256}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/noc256`. Development checks only; full frozen race validation still required.
- Result: M1_P1=563573; area=22.93345700352; peak W=11.093703721086094; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 18:58 UTC — Joint grade aggressive-r1
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/aggressive-r1/joint-0`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/aggressive-r1/joint-0` -> `project/campaigns/structural/aggressive-r1/joint-0/grade.json`.
- Result: status=completed; score=35673.764039986585; eligible=True; cycles={'M1_P1': 556591, 'M2_D1': 56978}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-29 18:59 UTC — M2_D1 screen noc256__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"noc_bytes_per_cycle": 256}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py noc256__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"noc_bytes_per_cycle": 256}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/noc256__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=62672; area=22.93345700352; peak W=10.476701993089254; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:10 UTC — M1_P1 screen hbm8_cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"hbm_channels": 8, "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_cache4 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"hbm_channels": 8, "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/hbm8_cache4`. Development checks only; full frozen race validation still required.
- Result: M1_P1=527556; area=21.58333399552; peak W=20.31539710358599; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:19 UTC — Joint grade p1_paired_w1_step_rf_noc512
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/p1_paired_w1_step_rf_noc512/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/p1_paired_w1_step_rf_noc512/joint` -> `project/campaigns/structural/p1_paired_w1_step_rf_noc512/joint/grade.json`.
- Result: status=completed; score=35392.89884148437; eligible=True; cycles={'M1_P1': 548218, 'M2_D1': 58770}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-29 19:20 UTC — M1_P1 screen sh2r
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"shared_ports": "2R1W"}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh2r --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_ports": "2R1W"}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh2r`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548218; area=23.419857003519997; peak W=11.245924777086685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:20 UTC — M1_P1 screen sh8
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"shared_banks": 8}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 8}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548218; area=23.57345700352; peak W=11.249764777086686; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:20 UTC — M1_P1 screen hbm8_cache2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"hbm_channels": 8, "cache_mib": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py hbm8_cache2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"hbm_channels": 8, "cache_mib": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/hbm8_cache2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=523760; area=20.74427249152; peak W=20.294420565985988; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:20 UTC — M1_P1 screen sh8_2r_cache4
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8_2r_cache4`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548397; area=22.61213399552; peak W=11.225731701886685; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:20 UTC — M2_D1 screen sh2r__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_ports": "2R1W"}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh2r__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_ports": "2R1W"}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh2r__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57794; area=23.419857003519997; peak W=10.784999657084503; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:21 UTC — M2_D1 screen sh8__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_banks": 8}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 8}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=57658; area=23.57345700352; peak W=10.78883965708451; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:21 UTC — M1_P1 screen dma_depth2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"dma_depth": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py dma_depth2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"dma_depth": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/dma_depth2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=failed; area=None; peak W=None; error=ValueError('area exceeds 24 mm2').
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:21 UTC — M2_D1 screen sh8_2r_cache4__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh8_2r_cache4__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 8, "shared_ports": "2R1W", "cache_mib": 4}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8_2r_cache4__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=56978; area=22.61213399552; peak W=10.76480658188449; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:25 UTC — M1_P1 screen tc8x8_kp2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"tc_array": "8x8", "tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x8_kp2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x8", "tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc8x8_kp2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=516781; area=23.26032484352; peak W=11.428945474451274; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:25 UTC — M2_D1 screen tc8x8_kp2__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_array": "8x8", "tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x8_kp2__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x8", "tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc8x8_kp2__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=64566; area=23.26032484352; peak W=10.781011353084265; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:36 UTC — M1_P1 screen tc8x16_kp1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"tc_array": "8x16", "tc_k_parallel": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x16_kp1 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x16", "tc_k_parallel": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc8x16_kp1`. Development checks only; full frozen race validation still required.
- Result: M1_P1=518406; area=23.65806052352; peak W=11.150765728961359; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:36 UTC — M2_D1 screen tc8x16_kp1__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_array": "8x16", "tc_k_parallel": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc8x16_kp1__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_array": "8x16", "tc_k_parallel": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc8x16_kp1__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=70278; area=23.65806052352; peak W=10.790954745084258; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:44 UTC — M1_P1 screen rf4r2w
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"rf_ports": "4R2W"}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py rf4r2w --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"rf_ports": "4R2W"}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/rf4r2w`. Development checks only; full frozen race validation still required.
- Result: M1_P1=839683; area=20.913459553280003; peak W=10.88909245610501; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:45 UTC — M2_D1 screen rf4r2w__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"rf_ports": "4R2W"}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py rf4r2w__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"rf_ports": "4R2W"}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/rf4r2w__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=79016; area=20.913459553280003; peak W=10.721164680828492; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:45 UTC — M1_P1 screen tc4x8_kp2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc4x8_kp2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc4x8_kp2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=594767; area=20.11745700352; peak W=11.026837013516257; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:45 UTC — M2_D1 screen tc4x8_kp2__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"tc_k_parallel": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py tc4x8_kp2__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"tc_k_parallel": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/tc4x8_kp2__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=64566; area=20.11745700352; peak W=10.702439657084549; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:45 UTC — M1_P1 screen sh1bank
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"shared_banks": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh1bank --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh1bank`. Development checks only; full frozen race validation still required.
- Result: M1_P1=548218; area=22.67745700352; peak W=11.227364777086686; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:46 UTC — M2_D1 screen sh1bank__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"shared_banks": 1}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sh1bank__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"shared_banks": 1}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh1bank__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=74511; area=22.67745700352; peak W=10.76643965708452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:46 UTC — M1_P1 screen dma2
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"dma_engines": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py dma2 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"dma_engines": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/dma2`. Development checks only; full frozen race validation still required.
- Result: M1_P1=593051; area=21.941457003520004; peak W=10.974517001090716; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:47 UTC — M2_D1 screen dma2__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"dma_engines": 2}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py dma2__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"dma_engines": 2}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/dma2__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=59262; area=21.941457003520004; peak W=10.53076276108713; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:52 UTC — M1_P1 screen no_multicast
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"multicast": false}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py no_multicast --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"multicast": false}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/no_multicast`. Development checks only; full frozen race validation still required.
- Result: M1_P1=624407; area=22.42145700352; peak W=10.66612360358555; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 19:53 UTC — M2_D1 screen no_multicast__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"multicast": false}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py no_multicast__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"multicast": false}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/no_multicast__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=72361; area=22.42145700352; peak W=10.53654525710235; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 20:01 UTC — M1_P1 screen sm_noc64
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json; hardware_patch={"sm_noc_bytes_per_cycle": 64}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sm_noc64 --case M1_P1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1_paired_w1_step_rf_noc512/hardware.json --patch '{}' --hardware-patch '{"sm_noc_bytes_per_cycle": 64}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sm_noc64`. Development checks only; full frozen race validation still required.
- Result: M1_P1=546331; area=22.856657003520002; peak W=11.186841713090251; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 20:02 UTC — M2_D1 screen sm_noc64__d1
- Change: base=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json; patch={}; hardware=/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json; hardware_patch={"sm_noc_bytes_per_cycle": 64}
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py sm_noc64__d1 --case M2_D1 --out-root /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1 --base /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/recipe.json --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/d1-sh/layer_rf_tc_noc512/hardware.json --patch '{}' --hardware-patch '{"sm_noc_bytes_per_cycle": 64}' --compiler /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/compiler.py`; reports `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sm_noc64__d1`. Development checks only; full frozen race validation still required.
- Result: M2_D1=58978; area=22.856657003520002; peak W=10.77091965708452; error=None.
- Decision: screen only, no incumbent promotion; compare before full frozen validation.

## 2026-09-29 20:07 UTC — Frozen D1 verification sh8_2r_cache4__d1
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8_2r_cache4__d1` -> `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8_2r_cache4__d1/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 20:13 UTC — Frozen D1 verification sh8__d1
- Change: none; verify screened variant using unmodified check_case and estimate_case.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py /home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8__d1` -> `/home/jovyan/projects/SLAI-infra-course-hw1/project/campaigns/structural/p1-pair-r1/sh8__d1/frozen-verification.json`.
- Result: check seeds=[7, 19]; timing completed=True.
- Decision: D1 evidence only; joint grade required before promotion.

## 2026-09-29 20:23 UTC — Joint grade aggressive-r1
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/aggressive-r1/joint-1`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/aggressive-r1/joint-1` -> `project/campaigns/structural/aggressive-r1/joint-1/grade.json`.
- Result: status=completed; score=35534.31772478464; eligible=True; cycles={'M1_P1': 560968, 'M2_D1': 56978}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 rf
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf` -> `project/gap_analysis/M2_D1--rf.json`.
- Result: 56978 -> 40863 cycles; removable=16115; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 rf_read
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf_read` -> `project/gap_analysis/M2_D1--rf_read.json`.
- Result: 56978 -> 50352 cycles; removable=6626; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 baseline
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 baseline` -> `project/gap_analysis/M2_D1--baseline.json`.
- Result: 56978 -> 56978 cycles; removable=0; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 rf_write
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf_write` -> `project/gap_analysis/M2_D1--rf_write.json`.
- Result: 56978 -> 46911 cycles; removable=10067; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 hbm
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 hbm` -> `project/gap_analysis/M2_D1--hbm.json`.
- Result: 56978 -> 54640 cycles; removable=2338; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 hbm+cache` -> `project/gap_analysis/M2_D1--hbm+cache.json`.
- Result: 56978 -> 43492 cycles; removable=13486; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:33 UTC — Diagnostic oracle M2_D1 cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 cache` -> `project/gap_analysis/M2_D1--cache.json`.
- Result: 56978 -> 54799 cycles; removable=2179; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:34 UTC — Diagnostic oracle M2_D1 tc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 tc` -> `project/gap_analysis/M2_D1--tc.json`.
- Result: 56978 -> 53578 cycles; removable=3400; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:34 UTC — Diagnostic oracle M2_D1 vec
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 vec` -> `project/gap_analysis/M2_D1--vec.json`.
- Result: 56978 -> 54218 cycles; removable=2760; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:34 UTC — Diagnostic oracle M2_D1 reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 reduce` -> `project/gap_analysis/M2_D1--reduce.json`.
- Result: 56978 -> 55874 cycles; removable=1104; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:34 UTC — Diagnostic oracle M2_D1 sfu
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 sfu` -> `project/gap_analysis/M2_D1--sfu.json`.
- Result: 56978 -> 56306 cycles; removable=672; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:34 UTC — Diagnostic oracle M2_D1 vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 vec+sfu+reduce` -> `project/gap_analysis/M2_D1--vec+sfu+reduce.json`.
- Result: 56978 -> 52490 cycles; removable=4488; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:38 UTC — Diagnostic oracle M2_D1 noc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 noc` -> `project/gap_analysis/M2_D1--noc.json`.
- Result: 56978 -> 56009 cycles; removable=969; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:38 UTC — Diagnostic oracle M2_D1 shared
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 shared` -> `project/gap_analysis/M2_D1--shared.json`.
- Result: 56978 -> 56466 cycles; removable=512; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:40 UTC — Diagnostic oracle M2_D1 dma
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dma` -> `project/gap_analysis/M2_D1--dma.json`.
- Result: 56978 -> 66393 cycles; removable=-9415; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:40 UTC — Diagnostic oracle M2_D1 rf+shared+dma+noc+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+shared+dma+noc+hbm+cache` -> `project/gap_analysis/M2_D1--rf+shared+dma+noc+hbm+cache.json`.
- Result: 56978 -> 16076 cycles; removable=40902; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:41 UTC — Diagnostic oracle M2_D1 rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce` -> `project/gap_analysis/M2_D1--rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce.json`.
- Result: 56978 -> 2204 cycles; removable=54774; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:43 UTC — Diagnostic oracle M2_D1 resident_warm_weights
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_warm_weights` -> `project/gap_analysis/M2_D1--resident_warm_weights.json`.
- Result: 56978 -> 43097 cycles; removable=13881; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:43 UTC — Diagnostic oracle M2_D1 resident_activations
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_activations` -> `project/gap_analysis/M2_D1--resident_activations.json`.
- Result: 56978 -> 46126 cycles; removable=10852; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:43 UTC — Diagnostic oracle M2_D1 resident_all
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_all` -> `project/gap_analysis/M2_D1--resident_all.json`.
- Result: 56978 -> 25020 cycles; removable=31958; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:43 UTC — Diagnostic oracle M2_D1 resident_kv
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_kv` -> `project/gap_analysis/M2_D1--resident_kv.json`.
- Result: 56978 -> 52274 cycles; removable=4704; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:43 UTC — Diagnostic oracle M2_D1 resident_local_weights
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_local_weights` -> `project/gap_analysis/M2_D1--resident_local_weights.json`.
- Result: 56978 -> 54604 cycles; removable=2374; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:44 UTC — Diagnostic oracle M2_D1 rf+shared+dma_unlimited+noc+hbm+cache+tc+vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+shared+dma_unlimited+noc+hbm+cache+tc+vec+sfu+reduce` -> `project/gap_analysis/M2_D1--rf+shared+dma_unlimited+noc+hbm+cache+tc+vec+sfu+reduce.json`.
- Result: 56978 -> 2204 cycles; removable=54774; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:44 UTC — Diagnostic oracle M2_D1 rf+shared+dma_unlimited+noc+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+shared+dma_unlimited+noc+hbm+cache` -> `project/gap_analysis/M2_D1--rf+shared+dma_unlimited+noc+hbm+cache.json`.
- Result: 56978 -> 16076 cycles; removable=40902; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 dma_unlimited
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dma_unlimited` -> `project/gap_analysis/M2_D1--dma_unlimited.json`.
- Result: 56978 -> 66393 cycles; removable=-9415; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 resident_warm_weights+activations
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_warm_weights+activations` -> `project/gap_analysis/M2_D1--resident_warm_weights+activations.json`.
- Result: 56978 -> 31868 cycles; removable=25110; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 resident_local_weights+activations+kv
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_local_weights+activations+kv` -> `project/gap_analysis/M2_D1--resident_local_weights+activations+kv.json`.
- Result: 56978 -> 39312 cycles; removable=17666; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 rf+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+hbm+cache` -> `project/gap_analysis/M2_D1--rf+hbm+cache.json`.
- Result: 56978 -> 26340 cycles; removable=30638; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 rf+tc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+tc` -> `project/gap_analysis/M2_D1--rf+tc.json`.
- Result: 56978 -> 33425 cycles; removable=23553; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 rf+hbm+cache+tc+vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+hbm+cache+tc+vec+sfu+reduce` -> `project/gap_analysis/M2_D1--rf+hbm+cache+tc+vec+sfu+reduce.json`.
- Result: 56978 -> 14312 cycles; removable=42666; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:45 UTC — Diagnostic oracle M2_D1 rf+vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 rf+vec+sfu+reduce` -> `project/gap_analysis/M2_D1--rf+vec+sfu+reduce.json`.
- Result: 56978 -> 36605 cycles; removable=20373; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:48 UTC — Diagnostic oracle M1_P1 resident_all
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 resident_all` -> `project/gap_analysis/M1_P1--resident_all.json`.
- Result: 556591 -> 367959 cycles; removable=188632; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:56 UTC — Diagnostic oracle M1_P1 resident_activations
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 resident_activations` -> `project/gap_analysis/M1_P1--resident_activations.json`.
- Result: 556591 -> 481139 cycles; removable=75452; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:58 UTC — Diagnostic oracle M1_P1 resident_weights
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 resident_weights` -> `project/gap_analysis/M1_P1--resident_weights.json`.
- Result: 556591 -> 419621 cycles; removable=136970; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 02:59 UTC — Diagnostic oracle M1_P1 rf
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf` -> `project/gap_analysis/M1_P1--rf.json`.
- Result: 556591 -> 494418 cycles; removable=62173; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:01 UTC — Diagnostic oracle M1_P1 rf_write
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf_write` -> `project/gap_analysis/M1_P1--rf_write.json`.
- Result: 556591 -> 537297 cycles; removable=19294; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:02 UTC — Diagnostic oracle M1_P1 baseline
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 baseline` -> `project/gap_analysis/M1_P1--baseline.json`.
- Result: 556591 -> 556591 cycles; removable=0; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:02 UTC — Diagnostic oracle M1_P1 rf_read
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf_read` -> `project/gap_analysis/M1_P1--rf_read.json`.
- Result: 556591 -> 548485 cycles; removable=8106; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:02 UTC — Diagnostic oracle M1_P1 rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce` -> `project/gap_analysis/M1_P1--rf+shared+dma+noc+hbm+cache+tc+vec+sfu+reduce.json`.
- Result: 556591 -> 4998 cycles; removable=551593; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:03 UTC — Diagnostic oracle M1_P1 rf+shared+dma+noc+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf+shared+dma+noc+hbm+cache` -> `project/gap_analysis/M1_P1--rf+shared+dma+noc+hbm+cache.json`.
- Result: 556591 -> 207510 cycles; removable=349081; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:03 UTC — Diagnostic oracle M2_D1 resident_all__wide
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 resident_all__wide` -> `project/gap_analysis/M2_D1--resident_all__wide.json`.
- Result: 56978 -> 37884 cycles; removable=19094; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:04 UTC — Diagnostic oracle M1_P1 hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 hbm+cache` -> `project/gap_analysis/M1_P1--hbm+cache.json`.
- Result: 556591 -> 405945 cycles; removable=150646; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:04 UTC — Diagnostic oracle M1_P1 resident_all__wide
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 resident_all__wide` -> `project/gap_analysis/M1_P1--resident_all__wide.json`.
- Result: 556591 -> 286170 cycles; removable=270421; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:06 UTC — Diagnostic oracle M1_P1 cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 cache` -> `project/gap_analysis/M1_P1--cache.json`.
- Result: 556591 -> 494691 cycles; removable=61900; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:08 UTC — Diagnostic oracle M1_P1 dma
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 dma` -> `project/gap_analysis/M1_P1--dma.json`.
- Result: 556591 -> 597888 cycles; removable=-41297; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:09 UTC — Diagnostic oracle M2_D1 dag_no_barriers__resident_local
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dag_no_barriers__resident_local` -> `project/gap_analysis/M2_D1--dag_no_barriers__resident_local.json`.
- Result: 56978 -> 44563 cycles; removable=12415; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:09 UTC — Diagnostic oracle M2_D1 dag_no_barriers
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dag_no_barriers` -> `project/gap_analysis/M2_D1--dag_no_barriers.json`.
- Result: 56978 -> 62428 cycles; removable=-5450; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:10 UTC — Diagnostic oracle M1_P1 reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 reduce` -> `project/gap_analysis/M1_P1--reduce.json`.
- Result: 556591 -> 531871 cycles; removable=24720; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:10 UTC — Diagnostic oracle M1_P1 noc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 noc` -> `project/gap_analysis/M1_P1--noc.json`.
- Result: 556591 -> 547404 cycles; removable=9187; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:11 UTC — Diagnostic oracle M1_P1 sfu
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 sfu` -> `project/gap_analysis/M1_P1--sfu.json`.
- Result: 556591 -> 554829 cycles; removable=1762; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:11 UTC — Diagnostic oracle M1_P1 tc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 tc` -> `project/gap_analysis/M1_P1--tc.json`.
- Result: 556591 -> 546838 cycles; removable=9753; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:11 UTC — Diagnostic oracle M1_P1 rf+tc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf+tc` -> `project/gap_analysis/M1_P1--rf+tc.json`.
- Result: 556591 -> 490635 cycles; removable=65956; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:11 UTC — Diagnostic oracle M1_P1 vec+sfu+reduce
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 vec+sfu+reduce` -> `project/gap_analysis/M1_P1--vec+sfu+reduce.json`.
- Result: 556591 -> 524186 cycles; removable=32405; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:12 UTC — Diagnostic oracle M1_P1 vec
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 vec` -> `project/gap_analysis/M1_P1--vec.json`.
- Result: 556591 -> 542370 cycles; removable=14221; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:12 UTC — Diagnostic oracle M1_P1 hbm
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 hbm` -> `project/gap_analysis/M1_P1--hbm.json`.
- Result: 556591 -> 558445 cycles; removable=-1854; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:12 UTC — Diagnostic oracle M1_P1 shared
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 shared` -> `project/gap_analysis/M1_P1--shared.json`.
- Result: 556591 -> 556591 cycles; removable=0; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:17 UTC — Diagnostic oracle M1_P1 rf+tc+shared+dma+noc+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf+tc+shared+dma+noc+hbm+cache` -> `project/gap_analysis/M1_P1--rf+tc+shared+dma+noc+hbm+cache.json`.
- Result: 556591 -> 44118 cycles; removable=512473; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:25 UTC — Diagnostic oracle M1_P1 rf+hbm+cache
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 rf+hbm+cache` -> `project/gap_analysis/M1_P1--rf+hbm+cache.json`.
- Result: 556591 -> 237284 cycles; removable=319307; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:26 UTC — Joint grade paired_late_b
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/p1-pair-r1/paired_late_b/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/p1-pair-r1/paired_late_b/joint` -> `project/campaigns/structural/p1-pair-r1/paired_late_b/joint/grade.json`.
- Result: status=completed; score=36115.70251935997; eligible=True; cycles={'M1_P1': 526494, 'M2_D1': 58770}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-30 03:26 UTC — Joint grade sh8_2r_cache4
- Change: measured P1/D1 combined with one shared hardware; snapshots `project/campaigns/structural/p1-pair-r1/sh8_2r_cache4/joint`.
- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/structural/grade_joint.py project/campaigns/structural/p1-pair-r1/sh8_2r_cache4/joint` -> `project/campaigns/structural/p1-pair-r1/sh8_2r_cache4/joint/grade.json`.
- Result: status=completed; score=35939.28974242999; eligible=True; cycles={'M1_P1': 548397, 'M2_D1': 56978}.
- Decision: preserve full grade; registry promotion pending controlled import after writer lock is free.

## 2026-09-30 03:26 UTC — Diagnostic oracle M2_D1 dag_no_barriers_precise__resident_local
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dag_no_barriers_precise__resident_local` -> `project/gap_analysis/M2_D1--dag_no_barriers_precise__resident_local.json`.
- Result: 56978 -> 44563 cycles; removable=12415; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:26 UTC — Diagnostic oracle M1_P1 hbm+cache+noc
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M1_P1 hbm+cache+noc` -> `project/gap_analysis/M1_P1--hbm+cache+noc.json`.
- Result: 556591 -> 395730 cycles; removable=160861; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:27 UTC — Diagnostic oracle M2_D1 dag_no_barriers_precise
- Change: process-local resource ablation only; frozen files and submission unchanged.
- Verify: `PYTHONPATH=. .venv/bin/python project/gap_analysis/oracle.py M2_D1 dag_no_barriers_precise` -> `project/gap_analysis/M2_D1--dag_no_barriers_precise.json`.
- Result: 56978 -> 62428 cycles; removable=-5450; no valid score/power.
- Decision: analysis evidence only; never promote an oracle as submission.

## 2026-09-30 03:28 UTC — Oracle gap analysis / DAG audit
- Change: diagnostic scripts only; pinned joint-0 candidate; no sweep or submission mutation.
- Verify: baseline oracles and observers reproduce 556591/56978; RF/HBM accounting reconciles exactly; completion-chain lengths conserve both cycles.
- Result: `project/gap_analysis/REPORT.md`, `summary.json`, per-case DAG/lineage/reservation artifacts. Initial zero-duration calendar / NoC timeline diagnostics failed; fixed in process-local instrumentation and rerun. Observer event-id parsing was corrected and final audits rerun. No failed attempt was scored.
- Decision: stop after analysis; structural hypotheses and falsification gates documented, no new large search.

## 2026-09-30 03:29 UTC — Oracle gap analysis / DAG audit
- Change: diagnostic scripts only; pinned joint-0 candidate; no sweep or submission mutation.
- Verify: baseline oracles and observers reproduce 556591/56978; RF/HBM accounting reconciles exactly; completion-chain lengths conserve both cycles.
- Result: `project/gap_analysis/REPORT.md`, `summary.json`, per-case DAG/lineage/reservation artifacts. Initial zero-duration calendar / NoC timeline diagnostics failed; fixed in process-local instrumentation and rerun. Observer event-id parsing was corrected and final audits rerun. No failed attempt was scored.
- Decision: stop after analysis; structural hypotheses and falsification gates documented, no new large search.

## 2026-09-30 15:30:56 — cand-152: Promote frozen-graded paired_late_b joint triple to root: P1 paired W1 kernel (paired_n=[w1], paired_b_prefetch=0) + step a_resident on NoC512/16SM/1TC hardware; D1 layer-RF owners. Byte-identical artifacts.
- Change: parent=cand-150; Manual promotion of completed joint grade; sources in proposal.json
- Evidence: project/experiments/cand-152/metrics.json; reports/grade.json; attempts=1
- Inputs: compiler.py@94da97ef hardware.json@e3ce1318 recipe.json@8713643e
- Verify: PASS
- Result: M1_P1=526494 (58.874x), M2_D1=58770 (22.155x); score=36115.703, eligible=True
  - deltas vs head (cand-150): {'M1_P1': -37282, 'M2_D1': -15410}
- Decision: KEPT — new incumbent head (score=36115.703)

## 2026-09-30 15:35 UTC — CHECKPOINT: Task 0 closeout before structural rewrite campaign
- Change: promoted graded paired_late_b joint triple to root as cand-152 via `project/campaigns/structural/promote_paired_late_b.py`; root `hardware.json` (semantic-equal), `project/recipe.json`, `programs/*.asm` now byte-identical to `project/campaigns/structural/p1-pair-r1/paired_late_b/joint/`; `python -m project.compiler` regenerates both programs byte-identically (sha256 verified).
- Verify: promotion script precondition asserts (compiler equality, program regeneration) -> PASS; registry head=search.best_id=cand-152; `project/experiments/cand-152/` holds inputs/programs/reports/metrics evidence.
- Result: incumbent score=36115.702519 (P1=526494 58.874x, D1=58770 22.155x), area=23.061mm2, peak power 11.158W/10.776W; prior cand-150 (31065.183) retained unchanged.
- Decision: checkpoint sealed. Finalizer's waiting state superseded by this manual promotion (user instruction 2026-09-30). Next: structural rewrite campaign Task 1 (persistent dual-batch MLP), Task 2 (D1 persistent layer pipeline), conditional Task 3 (RF-service-aware accumulator DAG). No official-server submission.

## 2026-09-30 16:47:03 — cand-153: Promote frozen-graded paired_late_b joint triple to root: P1 paired W1 kernel (paired_n=[w1], paired_b_prefetch=0) + step a_resident on NoC512/16SM/1TC hardware; D1 layer-RF owners. Byte-identical artifacts.
- Change: parent=cand-150; Manual promotion of completed joint grade; sources in proposal.json
- Evidence: project/experiments/cand-153/metrics.json; reports/grade.json; attempts=1
- Inputs: compiler.py@94da97ef hardware.json@e3ce1318 recipe.json@8713643e
- Verify: PASS
- Result: M1_P1=526494 (58.874x), M2_D1=58770 (22.155x); score=36115.703, eligible=True
  - deltas vs head (cand-152): {'M1_P1': 0, 'M2_D1': 0}
- Decision: evaluated; retained in Pareto/archives (head preserved)

### cand-153 verdict: duplicate
- DUPLICATE of cand-152: promote_paired_late_b.py was re-run after context compaction on 2026-09-30 16:47; inputs/programs/grade byte-identical to cand-152. Kept for provenance, not a new result; head remains cand-152.

## 2026-09-30 18:16:04 — cand-154: D1 token-local LN handoff + centred LN fold (beta W built in assembly at step 0, kept in RF) + attention K/V prefetch + b1 fold + chunked A; hardware shared_ports 2R1W. P1 unchanged.
- Change: parent=cand-152; structural-rewrite Task 2 Stage A/B; frozen D1 check seeds 7/19 pass, frozen D1 estimate 48952
- Evidence: project/experiments/cand-154/metrics.json; reports/grade.json; attempts=1
- Command (grade): /home/jovyan/projects/SLAI-infra-course-hw1/.venv/bin/python /home/jovyan/projects/SLAI-infra-course-hw1/challenge.py grade --hardware /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-154/inputs/hardware.json --program-p1 /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-154/programs/M1_P1.asm --program-d1 /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-154/programs/M2_D1.asm --baseline /home/jovyan/projects/SLAI-infra-course-hw1/baseline_manifest.json --seed 7 --report /home/jovyan/projects/SLAI-infra-course-hw1/project/experiments/cand-154/reports/grade.json; elapsed=4704.1; peak MiB=None
- Inputs: compiler.py@ebeda39d hardware.json@0619510b recipe.json@5dcf9315
- Verify: PASS elapsed=4704.05s
- Result: M1_P1=526494 (58.874x), M2_D1=48952 (26.598x); score=39572.063, eligible=True
  - deltas vs head (cand-152): {'M1_P1': 0, 'M2_D1': -9818}
- Decision: KEPT — new incumbent head (score=39572.063)

### cand-154 verdict: PROMOTED to root
- Full frozen joint grade rc=0 score=39572.063052975514 eligible=True cycles={'M1_P1': 526494, 'M2_D1': 48952} elapsed=4704s
- Root files staged from candidate inputs/programs.

## 2026-09-30 — P1 wavefront family: falsified at microbenchmark
- Diagnosis (gap_analysis M1_P1 DAG, 556k proxy): 1,163,776 LD line-queries, ~500-775k cross-WG duplicates: rep-8 intra-team A-tiles ~305k, rep-2/rep-3 weights shared pb0<->pb1<->step ~110-160k. Theoretical budget ~60-110k (borderline >=20% gate).
- Implemented gemm_wavefront (team-lockstep same-cycle A-tile issue -> frozen multicast coalescing); fixed program-order constraint (WAIT may only reference earlier events -> emitted in interleaved wavefront order).
- MLP microbench (3-layer dual-batch LN2->W1->GELU->W2->residual), same fixture:
  - current: 253,233 cyc, cache 202,534/373,344, hbm_r 10.93MB
  - wave (W2 wavefront): 261,877 cyc (-3.4%), cache 163,672/338,976, hbm_r 11.22MB, functional PASS
  - wave2 (W1 4-pass + W2): 275,337 cyc (-8.7%), cache 125,367/311,072, hbm_r 11.89MB, functional PASS
- Verdict: multicast coalescing works (queries -9%/-17%) but per-chunk lockstep WAITs serialize MMA issue more than the saved bank slots. Family stopped per gate; primitive removed by user. Evidence: project/campaigns/structural-rewrite/mlp_wave.asm, mlp_wave2.asm, /tmp/mlpw_*.log.

## 2026-10-01 — P1 structural sprint: gjoin / step_sh / wavefront falsifications
- Experiment (structural sprint): p1_schedule.global_join=1 (16-WG barriers across both prompt teams).
  Screen (dev-patched estimate, conservative span-overlap, race-check off): cycles=878,500 hbm_r=38.17MB cache=255,210/851,680 vs patched base 533,009 (frozen 526,494). VERDICT: FALSIFIED (-65%). Mechanism confirmed: lockstepping teams puts rep-2 weight queries at identical cycles -> in-flight misses duplicate (no merge in timed_cache) -> hit rate 48%->30%. Team read-timing directly controls weight hit rate.
- Experiment: p1_schedule.step_sh=['l0w1','l1w1'] shared_bytes=131072 (pre-commit step weights into idle SH): 517,443 (-15.6k vs base). stepsh_order (consumption-order SH fill): 527,017 (-6k). Under 10% gate so far; composing with deeper B pipeline next.
- Note: patched-estimate baseline = 533,009 vs frozen 526,494 (conservative patch inflates ~1.3%); all screens compared against patched base.
- Experiment: p1_schedule.team_stagger (pb1 WGs WAIT on pb0 anchor LD at each weight stage -> convert in-flight misses to hits): cycles=556,018 (+4.3% vs base), cache hit rate flat ~48%. VERDICT: FALSIFIED at stage granularity — anchor fires at issue-time not fill-completion, pb1 still lands inside the in-flight window and serializes its own vector work.
- Experiment: gemm_stages.p1_step.b_bufs=4 (deeper B pipeline in step GEMM, acc->lane10): 531,497 (-0.3%). Negligible alone.
- Composition stepsh_pipe (step_sh + b_bufs4): 516,243 (-3.1% vs base) — best dev screen so far; still below 10% gate.
- Experiment: p1_schedule.act_sh=1 (stage ln1/ln2 activation operands into idle 128KiB SH per WG, kill n-tile A re-reads): 528,878 (-0.8%), queries 845k->640k (-24%) but cycles barely moved. KEY FINDING: P1 is NOT cache-bank-service-bound — query removal without latency removal yields ~0. Critical path = raw<-LD dependency chains + dma_slot in-flight cap (4/SM).
- In flight: hw {cache_mib:4, dma_depth:2} (22.70mm2) and {cache_mib:0, dma_depth:2} — doubles outstanding transfers per SM at the cost of cache capacity; tests whether dma_slot binds.
- Experiment: hw {cache_mib:0, dma_depth:2}: 663,779 (+24.6%) — cache capacity is essential; hbm_r doubled to 59.7MB.
- Experiment: hw {cache_mib:4, dma_depth:2} (22.70mm2): 523,478 (-1.8%) — doubling in-flight slots helps modestly.
- Composition stepsh_pipe_dd2 (step_sh + b_bufs4 + cache4/dd2): 511,178 (-4.0% vs patched base) — best dev screen; roughly additive. Projected frozen P1 ~505k, joint ~40.4k (+2%) — below 10% full-grade gate.
- Critpath attribution (project/campaigns/structural-rewrite/critpath-p1base.json, 526,494==frozen): critical path = 3,669-instruction serial service chain, gaps=0. service_by_kind: MMA.ACC 278,326 (53%), LD 125,019, ST 59,036, VEC 46,711, SFU 12,450, REDUCE 4,907. by_stage: p_w1 130k, p_w2 84k, sm_ws 77k, p_qkv 73k, p_a 45k. Per-SM RF-read calendar busy 282k ~= chain MMA 278k -> chain is RF-read-service-bound (reads fixed by tc_array only: B-amp=m/pm, A-amp=n/pn).
- Model decoded: MMA span = rf_read(reads/128B/c) -> tc compute -> rf_write(writes/64B/c), reserved serially; 4x8/kp4 MMA 32x32x32 = 416+130+64=610cyc. 8x16/kp2 span=419 (-31%) area 29.9; 8x16/kp1 span=546 area 24.02; 8x8/kp4 span=482 (-21%) area 29.51.
- Earlier tc8x8/kp2 516,781 explained: reads -31% but compute/MMA 130->259 = span-neutral (610), hence -2% only. NOT evidence against RF-bound model.
- Screening: tc8x8k4 (8x8/kp4 + cache4+shared0+dma2/d2+mc0+red0+sfu8+noc256+smnoc64 = 23.20mm2) and tc8x16k1 (8x16/kp1+noc256 = 23.89mm2).
- Experiment: tc8x16k1 (tc_array 8x16,kp1 + noc256 = 23.89mm2): 516,671 (-2%) — RF reads -46% but chain re-composes around serial LD<->MMA interleave. tc8x8k4 (gutted support hw): 605,319 (+13.6%) — resource cuts starve LD delivery. VERDICT: TC-array upgrades dead under 24mm2 cap.
- Experiment: step_warm (pre-commit consumption-order cache warming of all step weight slices through scratch RF lanes): 583,179 (+9.4%), hbm_r +6.7MB — consumers duplicated in-flight warm fills (no in-flight miss merge). VERDICT: FALSIFIED; entire cache-timing/dedup direction now conclusively closed (wavefront, gjoin, stagger, act_sh, stepwarm all negative).
- Experiment: hbm_channels 4->8 (+0.2mm2): 514,232 (-3.5%) — aggregate HBM channel service is on the critical path. peakW 19.62W (tight vs 20W cap).
- Composition full_stack = stepsh_pipe (step_sh l0w1+l1w1 + b_bufs4) + hw {cache4, dd2, hbm8} = 22.90mm2: **498,010 cycles (-6.6% vs patched base), first sub-500k screen**, peakW 19.95W. D1 under same hw = 48,299 (-1.3%). Frozen functional check running.
- vec32 probe (vector_lanes 16->32): fails generation — RF lane repacking breaks program views. Not a free hw knob.
- Experiment: combo_all (actsh + stepsh + pipe + dd2c4): 519,442 — actsh STAGING LDs cost more than query savings; actsh excluded from stack.
- New mechanism priv_sh (same-WG res/out row-slab residency in aliased 2048-word SH buffer; removes res/out HBM round trips): helpers + generator wired; screens running under full-stack hw.
- New structural axis team=16 (32 WGs, 2 co-resident per SM; qkv/attention keep 8 head-owners, wo/w2 stay 8-wide, w1+row stages go 16-wide): hypothesis — sibling tapes interleave different-resource stalls (one's MMAs fill the other's LD waits); per-SM rf_read floor unchanged at ~278k. Screen running.
- Experiment: team=16 (32 WGs, 2 co-resident/SM; qkv/attention 8 head-owners, wo/w2 8 owners, w1+rows 16-wide): 638,675 (+19.8%) — sibling tapes contend on the same per-SM calendars (rf_read is one shared resource); latency-hiding does not materialize. VERDICT: FALSIFIED.
- Composition privsh_stepsh (priv_sh aliased res/out SH buffer + step_sh l0w1 + b_bufs4, hw {cache4,dd2,hbm8}): **493,745** — new best dev screen (-4.3k vs full_stack); peakW 19.34W (priv_sh removes HBM traffic -> power headroom improved).

## 2026-10-01 — priv_sh residency + step_sh_at + power constraint discovery
- Frozen check full_stack (stepsh_pipe+cache4/dd2/hbm8, 22.90mm2): P1 functional PASS, cycles=493,701, peakW=**20.39W -> INELIGIBLE** (>20W). D1 PASS 48,299/18.82W. Dev screen's 19.95W underestimated by ~0.4W. Peak window ~cyc 397k = pre-commit prefetch+GEMM overlap; hbm_byte=150pJ/B dominates window energy.
- Frozen check privsh_stepsh (priv_sh + l0w1 staging): PASS, 493,640, peakW=**20.50W -> INELIGIBLE**.
- New mechanism priv_sh (res/out same-WG SH residency in one aliased 2048-word buffer): dev screens — privsh_full 499,953; privsh_stepsh 493,745; privsh192 (shared_kib=192, l0w1+l1w1+l0wo) 489,230; privsh256 (shared_kib=256, all-3-w1+l0wo) **480,826**; privsh256_dd1 481,358.
- New mechanism step_sh_at (prefetch emitted at layer-1 boundary instead of pre-commit — spreads transfer energy & overlap): privsh256_early = **477,110** (-3.7k vs late-prefetch), dev peakW 19.35.
- Frozen check privsh256 (pre-commit variant): CRASHED at pipeline_events._issue_transfer assert "DMA slot changed after issue preview" — scheduler fragility when a previewed transfer's begin slides into a busy dma_slice window (long prefetch transfers widen the window). Frozen evaluator unmodified; avoid via timing change (early emit / fewer big transfers).
- team=16 (2 co-resident WGs/SM): 638,675 (+19.8%) FALSIFIED — sibling tapes contend on shared per-SM calendars; no latency-hiding win.
- Fallback screened: privsh256_hbm4 = 499,209 dev, peakW 11.16W — legal but loses the hbm8 ~18k.
- Power diagnosis: hbm_byte=150pJ/B dominates; 20W <=> ~270B/cyc avg in a 1000-cyc window. hbm8 doubles burst rate; prefetch concentrates step-weight HBM energy into the commit window. Mitigations in flight: step_sh_at=1 spread (early emit), hbm4 fallback.

## 2026-10-01 12:43:20 — cand-155: Promote frozen-graded privsh256_hbm4: P1 priv_sh res/out SH residency + step_sh 4-slice weight prefetch + 256KiB SH on hbm4/cache4/dd2 hardware (power-legal: peak 11.1W); D1 root program unchanged.
- Change: parent=cand-154; Manual promotion of completed joint grade; sources in proposal.json
- Evidence: project/experiments/cand-155/metrics.json; reports/grade.json; attempts=1
- Inputs: compiler.py@57149b1d hardware.json@c4da83b2 recipe.json@321bd0fd
- Verify: PASS
- Result: M1_P1=493984 (62.749x), M2_D1=48838 (26.660x); score=40901.125, eligible=True
  - deltas vs head (cand-154): {'M1_P1': -32510, 'M2_D1': -114}
- Decision: KEPT — new incumbent head (score=40901.125)

## 2026-10-01 12:50 — sprint-2 close-out: all remaining structural families falsified; submission frozen
- Branch B P1 microbenchmarks (all FALSIFIED vs 526,494): wstage (prompt weight SH residency + layer-boundary prefetch) = 660,249 (+24%); wstage_combo (+step_sh) = 619,645 (+18%); ares (RF A-residency) = 597,237 (+13.5%); merged_batch generation-blocked. Pattern: added transfers always lose; work removal wins only ~4-9%.
- Branch C D1 (FALSIFIED): upfront preload = 60,758 (+24% — streams serialize at t=0); nogroups = 52,754 (+8% — worse steady residency). D1 kept at 48,838.
- Frozen check noc256 (hbm8 + noc 256B/c throttle): P1 485,106 PASS func but peakW=20.09W -> INELIGIBLE. hbm8+256KiB structurally over-caps power.
- Winner: cand-155 = privsh256_hbm4 (priv_sh + step_sh[l0w1..l2w1,l0wo] + shared_kib=256 on hbm4/cache4/dd2/16SM): frozen grade eligible, P1=493,984 / D1=48,838, peak 11.1W/10.8W, area 23.37mm2, **score=40,901.125 (+3.35% vs cand-154)**.
- Structural conclusion: critical WG tape is 100% service-serialized (service_sum==end_cycle==526,494, gaps=0); rf_read calendar ~278k is menu-capped (8R4W max). Remaining gap needs a new primitive (decoupled producer-consumer issue, cross-WG ownership handoff) not expressible in current ISA/DSL.
