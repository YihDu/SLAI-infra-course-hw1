# Agent + MILP automatic iteration

Current entry point:

```sh
.venv/bin/python -m project.search.milp_campaign --max-grades 6 --memory-mib 6144 --wall-seconds 2400
```

Requires the isolated HiGHS environment documented in
`project/campaigns/milp-d1/README.md`; it does not add packages to the frozen
evaluator environment. The solver executable is configurable with
`--solver-python`.

## Implemented boundary

The current provider optimizes D1 projection owner counts and RF panels under a
15500-event trust region on the verified fixed-column/head structure. It carries
P1 and hardware forward unchanged. It cannot generate new fusion or residency
algorithms by changing labels. Codex owns outer algorithm implementation and
must respond to `outer_structure_required` with a concrete new structure or an
evidence-backed saturation report.

- MILP excludes historical projection plans on identical hardware.
- Compiler inputs are immutable snapshots of the verified structural parent.
- Candidate creation, evaluation, publication and default staging are serial
  under the controller writer lock; grade concurrency is one.
- Registry save now compares revisions under a separate short write lock. A
  stale in-memory snapshot raises instead of overwriting newer candidates.
- Existing prepare/cost/grade/cache/archive/log code is reused.
- Successful grade score decides promotion; solver objective never does.
- Every attempt records predicted events/proxy, actual cycles/score, process
  memory and failure class in a durable campaign checkpoint.
- Memory failures restrict the event trust region for this skeleton. They are
  excluded from performance and plateau statistics.
- Three consecutive eligible candidates improving the incumbent by less than
  10% stop this inner search and request outer work. Solver exhaustion and
  resource/budget stops are recorded separately.

Performance feedback supports local corrections from controlled comparisons
where exactly one D1 projection changes. The observed whole-D1 delta, divided by
sixteen invocations, corrects that option relative to its parent. Multi-projection
changes are not used to infer per-stage coefficients. The request saves these
corrections and campaign evidence identifies each source candidate. Corrections
are conditional on the existing context and may miss interactions; grade remains
the ground truth. Memory event caps remain empirical, not proof of RSS bounds.


## Recovery

`--resume <campaign directory>` reuses exclusions, observations and pending
attempt metadata. Recovery occurs under the existing controller lock. A new
invocation can replenish an exhausted bounded budget. A structural plateau or
solver failure remains explicit and does not silently restart the same search.
Completed or interrupted attempts not yet checkpointed are reconciled from
immutable candidate reports; existing report directories are never rerun.

## Verification

- `test_milp_loop`: stale snapshot protection, failed-memory feedback, valid-score
  plateau logic, historical plan recognition.
- Existing `test_records` verifies registry/archive/cache/promotion compatibility.
- `test_milp_solver` in the HiGHS environment checks no-good exclusions, infeasible
  budgets, and the exact optimum against independent finite enumeration.
- Campaign `auto-milp-20260929-005803` exercises the actual solver-to-grade path.

These boundaries define the implemented inner optimizer; reaching its stopping
rule does not prove a global score ceiling of the ISA or hardware design space.
