# Bounded MILP experiment: D1 projection owners and RF panels

This experiment uses HiGHS integer optimization to propose an executable schedule.
It does not replace the frozen evaluator and does not establish a global optimum
of the full hardware/program problem.

## Model

Binary x[stage, option]; choose one option for each QKV, WO, W1, W2 stage.
Each option specifies owners (1–4), N tile (16/32/64/128), K chunk (8/16/32/64).
Only exact column partitions and RF operands <=1024 words enter the option set.
All sixteen layer/step instances share a stage's choice. Attention keeps four
head owners. LN/residual/GELU stay on SM0. All stages use coarse barriers.

Objective: minimize the sum of per-stage critical-owner service proxies over
sixteen invocations. The proxy includes MMA RF read/compute/write service,
operand loads, initialization and output stores, with a shared NoC floor.
It omits exact DMA/cache overlap, cross-stage interactions and synchronization
latency. Its numerical objective must not be labeled actual D1 cycles.

Constraint: total computation/transfer events <=15500, versus15056 for the
verified cand-112 schedule. This empirical budget limits event growth; it is not
a guarantee of physical checker memory. Exact validation-cost preflight and a
6144MiB bounded frozen grade are still mandatory.

## Reproduce

Use a separate environment to preserve the evaluator's pinned Python/NumPy:

```sh
uv venv --python .venv/bin/python /tmp/transformer-milp-env
uv pip install --python /tmp/transformer-milp-env/bin/python -r project/campaigns/milp-d1/requirements-solver.txt
PYTHONPATH=. /tmp/transformer-milp-env/bin/python project/campaigns/milp-d1/solve.py
```

The script saves the LP, legal option table and selected solution, and checks the
MILP objective against an independent exhaustive enumeration of this small model.
Compiler support is `d1_schedule.projections` in `ParallelD1Builder`.

API reference: [HiGHS Python modelling documentation](https://ergo-code.github.io/HiGHS/dev/interfaces/python/model-py/).

## Selected proposal

| Stage | SM owners | N | K |
|---|---:|---:|---:|
| QKV | 4 | 32 | 32 |
| WO | 4 | 32 | 32 |
| W1 | 4 | 64 | 16 |
| W2 | 4 | 32 | 32 |

Exact generated D1 events15376 agree with the model count; instructions15552.
Candidate113 is a child of cand-112, with byte-identical P1 and hardware, so its
full grade provides a controlled comparison against the heuristic partition.

## Full-grade result

| Case | Heuristic cand-112 | MILP cand-113 |
|---|---:|---:|
| P1 cycles | 7292285 | 7292285 |
| D1 cycles | 278743 | 212659 |
| Score | 4455.9142 | 5101.4887 |
| Grade process peak MiB | 5393.30 | 5178.75 |

Both cases pass correctness and all eligibility gates. D1 improves1.31075x;
score improves14.488%. The service proxy predicts1.5078x improvement for the
projection-only objective; whole D1 improves less because other stages and
shared-resource interactions remain. This is evidence for using the MILP as a
proposal generator, not treating its objective as a full-model prediction.
Frozen reports live in `project/experiments/cand-113/reports/grade.json`.

No broad parameter sweep or hardware change was used. The optimizer selects
among118 legal stage options (730080 combinations before the global event cap).
The exhaustive oracle only verifies this finite proxy model. It does not prove
global optimality of assembly, fusion, residency, placement, or hardware.

## Final composition

cand-114 combines the verified RF resident P1 MLP with this D1 MILP plan.
Full frozen grade passes: P1=7049767, D1=212659, score=5188.494494259971.
P1 bytes equal cand-110; D1 bytes equal cand-113. Hardware remains unchanged.
Grade elapsed110.82s, peak5230.46MiB. Area16.25838mm²; P1/D1 peak rolling
power3.20909/6.55440W. Seed7 functional max absolute errors2.7930e-6/1.1210e-6.
Default inputs and generated programs are now cand-114.

Relative to the start-of-work cand-106 (score2776.11), final score improves1.869x;
relative to baseline1000 it is5.188x. A10x score improvement has not been achieved.

## Remaining bottlenecks and next use of MILP

1. P1 still uses SM0 only; matrix RF service and dependencies dominate, and simple
   activation residency gave only3.33% P1 improvement. P1 full multi-SM remains
   blocked by checker history storage. Do not transplant this D1 result as proof
   that P1 multi-SM now fits.
2. D1 now has balanced TC work (12056 cycles on each SM), but primary-WG serial
   LN/residual/GELU and shared transfer service remain. RF-write busy fractions
   are33.63% on SM0 and29.42% on the others; these are not overall occupancies.
3. Validation memory is still material: the actual process peak is about5.1GiB,
   far above the longest-chain-only logical model. Restrict event growth and
   retain bounded full grade. No universal memory guarantee is inferred.

Next MILP extension should model a concrete implementable residency/issue-order
choice with measured coefficients. It should not spend a large budget repeatedly
optimizing the same N/K menu or claim that this finite partition optimum solves
the outer algorithm problem.
