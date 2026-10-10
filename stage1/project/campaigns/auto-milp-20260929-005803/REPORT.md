# First automatic MILP iteration

Goal: iterate toward the current design's practical limit; this campaign alone
is not proof of an ISA-wide or design-wide upper bound.

Entry: `.venv/bin/python -m project.search.milp_campaign --max-grades 6 --memory-mib 6144 --wall-seconds 2400`.
Starting incumbent cand-114 score5188.49449.

| Candidate | Changed D1 projection | D1 cycles | Score | Peak MiB |
|---|---|---:|---:|---:|
| 115 | W1 N128/K8 | 210849 | 5210.71681 | 4982.71 |
| 116 | additionally WO N16/K64 | 215937 | 5148.96230 | 5218.13 |
| 117 | from115, W2 N16/K64 | 228289 | 5007.72822 | 5524.63 |

All pass complete frozen grade. P1 stays7049767. Best115 is automatically
promoted/staged; 116/117 do not replace it. Three consecutive eligible gains
below10% trigger `outer_structure_required` after3/6 grades, rather than spending
the remainder on similar parameter variants. Events remain within15500.
Each request, solution, solver heartbeat, launch and feedback record is saved.

The first loop uses uncalibrated model v1. Its first alternate ranks worse in the
proxy but improves measured performance. Controlled parent/child comparisons
change only one D1 projection, enabling local delta corrections in model v2:
`feedback-calibration.json` records evidence; `feedback-request.json` and
`feedback-solution.json` demonstrate that calibrated ranking now selects the
verified115 plan as best in the original event region. This retrospective check
is not an independent performance result or a global model-validation claim.

Registry stale-snapshot protection, failed-memory feedback, plateau behavior,
historical exclusions and controlled delta attribution have16 passing tests
including existing records/archive/cache tests. Two optional HiGHS tests check
no-good constraints, infeasibility and an independent finite optimum.

The goal remains active: inner plateau caused a structural change to P1 MLP
operand pipelining (cand-118), which subsequently improved score to5555.74170.
Further P1 QKV/WO issue scheduling remains an untested opportunity.
