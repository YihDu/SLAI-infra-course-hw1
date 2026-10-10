# Bounded search execution

The controller proposes candidates, then executes this chain:

`immutable inputs → isolated generation → expanded ISA screen → exact artifact cache → grade → records / archives / iteration log`

- Default grade includes functional checks and timing. `--validation full` explicitly requests check followed by grade. There is no redundant estimate pass.
- Candidate generation and challenge subprocesses use the candidate directory, with absolute input/report paths. They never stage inputs into the root worktree.
- Only the coordinator writes registry, archives, cache, and the iteration log. A process lock excludes concurrent controllers; registry/cache writes use atomic replacement.
- Evaluation concurrency defaults to two. Each subprocess has a 600-second timeout and 3072 MiB physical-memory limit. macOS accounting includes compressed memory. Termination kills its process group.
- `reports/*.heartbeat.json` records PID, elapsed time, peak memory and completion status once per second; stdout/stderr and final run statistics remain beside the raw reports.
- The evaluation budget counts actual check/grade launches, including crashes, memory failures and timeouts. Generation, static rejection and cache hits consume no evaluator calls. The default batch wall budget is 7200 seconds.
- Interrupted candidates are recovered as `not_verified`; resource/evaluator failures never count as functional failures or enter the archives. Never rerun a completed candidate in place: create a child attempt.
- Cache identity includes hardware, both generated programs, evaluator source, baseline, Python/NumPy, seeds, and validation mode. Compiler identity also participates in input proposal deduplication. Historical imports require matching raw report provenance, artifact hashes, baseline, runtime and seeds. Cache reuse links its source; it creates no synthetic grade report.
- Six proposals allocate 2 exploit / 2 archive expansion / 2 novelty, with at most 128 proposal attempts and explicit rejection/shortfall diagnostics. Stage overrides participate in mutation and validation. The hardware menu mirrors the frozen evaluator.
- Score improvement promotes the incumbent only if both case cycle counts also do not regress. Eligible tradeoffs remain available in Pareto archives. A six-round plateau stops automatic search for a new operator.

```sh
.venv/bin/python -m unittest discover -s project/search/tests -v
.venv/bin/python -m project.search.controller search-plan --rounds 1 --per-round 6
.venv/bin/python -m project.search.controller search-run --phase 2 --rounds 2 \
  --per-round 6 --max-evals 12 --jobs 2 --seed 7
.venv/bin/python -m project.search.controller run cand-NNN --validation full --max-evals 2
```

`search-plan` does not create candidates or update the registry. `checkout` stages an existing candidate's inputs and generated programs. Root files are not used by running evaluations.

## Repair validation campaign

`python -m project.search.campaign --max-evals 12 --jobs 2` reproduces cand-073 before proceeding to seven isolated hardware controls (K parallelism 2/4, TC array 8x8/8x16, TC count 4, DMA engines 2, cache 4 MiB). Remaining budget permits at most two beneficial combinations, one stage neighbor and one archive expansion. The campaign restores the best complete submission triple and writes a launch ledger plus summary under `project/campaigns/`.

# P1 W1 shared panel operator

`w1_shared_panel` enables a distinct GEMM schedule for prompt W1 only. It stages
one 256×64 FP32 weight panel (64 KiB) in workgroup shared memory, accumulates
16 prompt rows across four 16-column subtiles, and reuses the A tile within
each K chunk. The compiler reserves 64 KiB of shared memory only for P1;
decode remains on the previous path. The DSL requires at least 64 KiB of
shared memory and 1024 RF words per lane. Candidates using this operator pin
the current `project/compiler.py` in their immutable input snapshots, because
older candidate compilers do not implement it. Its first measured candidate
is `cand-086`. It passed correctness but raised P1 from 10,750,199 to
11,817,029 cycles while D1 stayed at 744,864, so the candidate was reverted.
HBM bytes were unchanged; extra shared memory service and the new scheduling
order are the likely sources of the regression. The full verdict is in
`project/iteration-log.md`.

## RF accumulator layouts and structural campaigns

`stage_gemm_tiles` now accepts `order`, including `nkm_lanes`. This layout
spreads live accumulators across RF partitions, allowing wider N tiles while
retaining B reuse. `propose.stage_space` enumerates all implemented layouts
and tile choices, with stage-specific capacity checks. P1 K chunks below 16
remain legal but automatic proposals skip their scalar functional-check path.
Joint templates include an SM/TC area bridge and a vector-width/RF-capacity
change. See `DIAGNOSIS_V4.md` for the equations, observations and limits.

Run a bounded expansion with `python -m project.search.structural_campaign
--max-evals 8 --jobs 2`, then adapt around its verified winner with
`python -m project.search.structural_campaign --refine --max-evals 3 --jobs 2`.
Both commands record snapshots, grades, pruning and launch budgets, then
restore the best submission while retaining the current compiler capability.

## Evaluation slot refill

The coordinator refills an evaluator slot as soon as its process completes,
instead of waiting for an entire two-candidate wave. Completed metrics are
buffered and published in proposal order. Registry/cache updates remain on the
coordinator thread; identical in-flight artifacts share one evaluation future,
and the locked launch budget is unchanged. A regression test holds the first
candidate until the third starts, proving a fast second candidate frees a slot
without changing publication order. This verifies scheduling behavior; no new
hardware score is claimed from the scheduler change alone.
