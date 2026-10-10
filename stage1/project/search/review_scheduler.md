> Primary-agent review: this worker reviewed the pre-change wave scheduler while
> the coordinator was being updated. The implementation now uses slot refill,
> proposal-order publication and a call-scoped execution-key/future map.
> The budget warning below overstates an existing risk: `evaluator.evaluate`
> already catches a rejected `before_launch` claim and returns `budget_exhausted`.
> `validation="full"` intentionally charges both actual check and grade launches.
> No extra reservation counter was adopted; adding active jobs to already claimed
> budget would double-count them. The archive API is `archives.update`, not
> `update_archives`. Serial preparation still costs time, so utilization is not
> guaranteed to be 100%. The slow-first/refill regression test and existing
> duplicate/budget tests passed; this report is advisory, not evaluator evidence.

# Architectural Review: Work-Conserving Scheduling for Search Coordinator

## 1. Executive Summary & Existing Bottlenecks

### Current State: Rigid Wave-Based Execution vs. Refill Draft
In `project/search/coordinator.py`, `evaluate_many` historically scheduled candidate evaluations in discrete, fixed-size batches ("waves") of size `jobs`:
```python
for start in range(0, len(candidates), jobs):
    wave = []
    for cand in candidates[start:start + jobs]:
        ...
        fut = pool.submit(...)
        wave.append((cand, fut, key, False))
    for cand, fut, key, duplicate in wave:
        metrics = copy.deepcopy(fut.result())
        finish(cand, metrics, None if duplicate else key)
```

### The Bottleneck: Head-of-Line Blocking
- **Worker Starvation:** In a wave of `jobs=2`, if Candidate A requires 300s (full cycle simulation) while Candidate B requires 15s (early functional rejection or lighter case), Worker 2 sits idle for 285s. Candidate C cannot start preparation or execution until Candidate A finishes and the entire wave barrier completes.
- **Goal:** Implement **work-conserving** scheduling where a free worker slot is immediately dispatched work from the pending candidate queue, while strictly maintaining:
  1. Deterministic proposal-order result integration.
  2. In-flight duplicate reuse across asynchronous workers.
  3. Single-writer registry and cache invariants.
  4. Exact grade budget bounds without race conditions.

---

## 2. Safety Requirement 1: Deterministic Proposal-Order Result Integration

### Problem & Invariants
- `records.record_result()` performs critical stateful side-effects:
  - Updates candidate entries in `reg["candidates"]`.
  - Promotes incumbents and updates Pareto/elite archives (`archives.update_archives`).
  - Appends Markdown entries to `project/iteration-log.md`.
- If candidates complete out-of-order and are committed as they finish, run results become nondeterministic across identical seeds. Archive contents, log chronology, and search policy feedback diverge depending on microsecond OS thread scheduling.
- The returned `results` list from `evaluate_many` must strictly match input proposal order.

### Required Architecture: Reorder Buffer (ROB) / Retirement Queue
1. **Sequence Tagging:** Every candidate entering `evaluate_many` is assigned a monotonic sequence index `seq = 0, 1, ..., N-1`.
2. **Decoupled Completion vs. Commit:**
   - Worker threads/futures complete out-of-order and place their raw metrics into a pending buffer indexed by `seq` (e.g., `entries` list).
   - A single-threaded **Retirement Head** (`next_publish` / `commit_seq`, initialized to 0) checks if candidate `commit_seq` has resolved.
   - When candidate `commit_seq` is ready, `finish(cand, metrics)` is called, committing:
     - `cand.dir / 'metrics.json'` write.
     - `record_result(reg, cand, metrics)` (archive updates and `iteration-log.md` append).
     - `cache.store(key, cand, metrics)`.
     - `state.save(reg)`.
     - Append to return list `results`.
   - `commit_seq` increments by 1, cascading through any subsequently completed contiguous candidates.
3. **Verification in Tests:** As verified in `tests/test_coordinator.py`, if Candidate 0 runs slowly while Candidate 1 finishes early, Candidate 2 can launch, but publication must stall until Candidate 0 finishes, emitting results in exact `['0', '1', '2', '3']` order.

---

## 3. Safety Requirement 2: In-Flight Duplicate Reuse

### Problem & Invariants
- In a dynamic work-conserving queue, an incoming candidate `C_dup` may match an execution key currently executing on a worker (`C_lead`), or already finished by a worker but not yet committed by the ROB, or already committed in persistent cache.
- Redundant simulator invocations waste evaluation budget and hardware compute.

### Critical Analysis of In-Flight Tracking (`by_key`)
In the prototype work-conserving draft:
```python
elif key in by_key:
    future, duplicate = by_key[key], True
```
While this correctly avoids launching duplicates, several edge-case hazards exist:
1. **Memory & Lifecycle Leakage:** `by_key` is never pruned. Completed futures remain in memory across the entire evaluation session.
2. **Bypassing Cache Qualification Gates:** If a lead candidate finishes with an error or ineligible result, `cache.store()` rejects it. If `by_key[key]` persists indefinitely, later candidates matching `key` will link to the uncacheable failed run via `by_key` rather than being handled through standard screening.
3. **Atomic Window & Hand-off:** An entry must remain in `in_flight_keys` while running, and upon completion must transition directly to persistent disk cache before removal from the in-flight table.

### Required Architecture: In-Flight Execution Map & Lifecycle
1. **In-Flight Registry:**
   - Maintain an active table: `in_flight_keys: dict[str, Future]`.
2. **Three-Tier Execution Resolution:**
   - **Tier 1 (Persistent Cache):** Check `cache.lookup(key)`. If hit, resolve immediately as `cache_hit=True`.
   - **Tier 2 (In-Flight Leader):** If `key in in_flight_keys`, do NOT submit a new task to `ThreadPoolExecutor`. Do NOT call `budget.claim()`. Attach `C_dup` to `in_flight_keys[key]`.
   - **Tier 3 (New Evaluation):** Register `key -> future` in `in_flight_keys`, reserve budget, and submit to pool.
3. **Failure Propagation Isolation:**
   - If `C_lead` fails (e.g., timeout, syntax error, simulation failure, or unhandled exception):
     - `C_dup` must not crash or trigger retry loops.
     - Per existing protocol: `C_dup` must record `verdict="not_verified"`, `failure_class="duplicate_unverified"`, `cache_hit=True`, `cache_source=C_lead.id`, `evaluation_attempts=0`.
   - If `C_lead` encountered an unhandled exception in `future.result()`, `C_dup` must safely catch the exception and assign `cache_source=C_lead.id` (avoiding `None` when `metrics` has no `'candidate'` key).
   - Only `C_lead` may call `cache.store()`; duplicates must never attempt to store duplicate artifacts.

---

## 4. Safety Requirement 3: Single-Writer Registry & Cache Invariants

### Problem & Invariants
- Multiple worker threads running concurrently must never concurrently mutate:
  - `reg` dictionary.
  - `experiments/registry.json` via `state.save(reg)`.
  - `experiments/execution_cache.json` via `cache.store()`.
  - `project/iteration-log.md` via `records.record_result()`.
  - `cand.status` on disk.
- Worker threads writing concurrently cause torn JSON writes, race-corrupted cache files, and lost updates.

### Premature State Mutation Hazard
- In the prototype:
  ```python
  cand.status = 'evaluating'; cand.save()
  reg['candidates'][cand.id]['status'] = 'evaluating'; state.save(reg)
  ```
  This is executed **before** `prepare()`, cache lookup, duplicate lookup, or budget check.
- **Hazards:**
  1. **Redundant Disk I/O:** Pruned, cached, or budget-exhausted candidates write `registry.json` to disk repeatedly.
  2. **Corrupted Crash Recovery:** If the process is interrupted, `coordinator.recover()` inspects candidates with status `'evaluating'` and marks them as interrupted failures (`Recovered incomplete evaluation`). A candidate that was merely cached or pruned would be falsely recorded as a failed evaluation!

### Required Architecture: Strict Thread Boundaries
1. **Subprocess Isolation:** Worker threads must run pure functional jobs (`evaluator.evaluate`). They read candidate inputs and write strictly to private candidate directories (`cand.reports/*`).
2. **No Shared State in Workers:** Worker threads must NEVER touch `reg`, call `state.save()`, call `cache.store()`, or call `record_result()`.
3. **Lazy `'evaluating'` Status Transition:**
   - A candidate must only be marked `'evaluating'` and saved to `reg` if and when it is actively submitted to a worker thread for simulator execution.
   - Candidates resolved via cache, static pruning, or budget exhaustion should transition directly to their final status upon retirement.

---

## 5. Safety Requirement 4: Exact Grade Budget Enforcement

### Problem & Invariants
- `Budget` has hard limits: `maximum` (e.g., 12 or 2) and `wall_seconds`.
- `budget.claim()` increments `used` under a lock and raises `RuntimeError('evaluation_budget_exhausted')` if exceeded.
- In `evaluator.py`, `budget.claim()` is called via callback `before_launch` inside `evaluator.evaluate`.

### Failure Modes of Naive Concurrency
1. **Race to Claim (Over-subscription Exception):**
   - Suppose `budget.maximum = 2`, `budget.used = 1`.
   - Worker slot 1 and Worker slot 2 become free simultaneously.
   - Candidate A checks `budget.available()` -> `True` (`used=1 < 2`).
   - Candidate B checks `budget.available()` -> `True` (`used=1 < 2`).
   - Both tasks submit. Candidate A calls `budget.claim()` (`used=2`). Candidate B calls `budget.claim()` -> raises `RuntimeError('evaluation_budget_exhausted')` inside worker thread! Candidate B fails with `evaluator_error` or uncaught exception rather than clean budget handling.
2. **Premature Speculative Budget Drain:**
   - If the scheduler claims budget before `evaluator.prepare()` or cache check, candidates that fail static screening or hit cache will permanently burn grade budget slots.
3. **Multi-Mode Accounting (`validation='full'` vs `'grade'`):**
   - In `evaluator.py`, `validation="full"` runs `check` and then `grade`, triggering `before_launch` twice. The coordinator must ensure that budget accounting correctly reflects grade evaluations and does not prematurely reject due to pre-grade checks.

### Required Architecture: Budget Reservation Protocol
1. **Phase Separation:**
   - Candidate preparation (`evaluator.prepare`) and cache lookup do NOT consume budget.
2. **Active Evaluation Accounting:**
   - Track `reserved_evals` on the coordinator thread:
     `budget.used + len(active_workers) < budget.maximum`
   - When a candidate passes preparation and cache check, and requires simulation:
     - Check available budget including currently active/running simulations.
     - Dispatch only when: `len(active_workers) < jobs` **AND** `budget.used + len(active_workers) < budget.maximum`.
3. **Clean Budget Exhaustion:**
   - When `budget.used + len(active_workers) >= budget.maximum`, any subsequent non-cached candidate in the proposal queue must not be dispatched.
   - It is immediately resolved at the coordinator level with:
     `{'reports': {}, 'verdict': 'not_verified', 'failure_class': 'budget_exhausted', 'evaluation_attempts': 0}`.
   - It enters the ROB and is retired in proposal order.

---

## 6. Concrete Failure Modes of Naive Implementations

| # | Failure Mode | Trigger / Mechanism | Consequence |
|---|---|---|---|
| 1 | **Race on `budget.claim()`** | Dispatching `N` tasks when remaining budget `< N` because `available()` checked before worker claims. | Worker throws unhandled `RuntimeError`, candidate marked `evaluator_error` instead of clean `budget_exhausted`. |
| 2 | **Cache Invalidation Gap** | In-flight key removed upon worker future completion, but before retirement / `cache.store()`. | Incoming identical candidate misses both in-flight map and disk cache, running a duplicate expensive simulation. |
| 3 | **Archive Inversion** | Fast candidate (5s) completes before slow candidate (200s) and writes directly to archives. | Nondeterministic Pareto frontier and incumbent selection across runs with identical seed. |
| 4 | **Premature Evaluation Marking** | Marking all queued candidates `evaluating` before prepare/cache check. | Cached/pruned candidates converted to `interrupted` if process restarts or encounters keyboard interrupt. |
| 5 | **Duplicate Leader Crash** | Leader candidate fails with timeout or memory limit; duplicate attempts to read absent reports. | Duplicate crashes coordinator with `FileNotFoundError` or unhandled exception. |
| 6 | **Unbounded In-Flight Leak** | `by_key` retains completed/failed futures indefinitely across search rounds. | Memory leak; subsequent duplicate candidates bind to old uncacheable failures. |

---

## 7. Recommended Actionable Architecture

### State Transition per Candidate
```
[PROPOSED]
    │
    ▼ (coordinator thread)
[PREPARE] ──(fail/prune)──► [RESOLVED (pruned/error)] ────┐
    │                                                     │
    ▼                                                     │
[CACHE / IN-FLIGHT CHECK] ──(hit)──► [RESOLVED (cached)] ─┤
    │                                                     │
    ▼ (needs grade evaluation)                            │
[BUDGET CHECK] ──(exhausted)──► [RESOLVED (exhausted)] ───┤
    │                                                     │
    ▼ (slot free & budget token available)                │
[DISPATCH TO WORKER]                                      │
    │                                                     │
    ▼ (worker completes)                                  │
[RESOLVED (evaluated)] ───────────────────────────────────┤
                                                          │
                                                          ▼
                                              [REORDER BUFFER (ROB)]
                                                          │ (in proposal order)
                                                          ▼
                                             [COMMIT & RETIRE (finish)]
```

### Coordinator Event-Loop Pattern
Structure `coordinator.evaluate_many` around an event-driven loop:
1. **Queues & State:**
   - `entries = []`: ROB sequence `(cand, future, key, duplicate, immediate)`.
   - `next_publish = 0`: Monotonic publication pointer.
   - `active = set()`: Currently executing worker futures.
   - `by_key = {}`: Active in-flight keys mapped to `(leader_cand, future)`.
2. **Refill & Dispatch Step:**
   - While `next_candidate < len(candidates)` and `len(active) < jobs`:
     - Inspect next candidate without setting status `'evaluating'` prematurely.
     - Execute `evaluator.prepare(cand)`. If failure/pruned, append to `entries` with `immediate = prepared`, continue.
     - Check `cache.lookup(key)`. If hit, append to `entries` with `immediate = hit`, continue.
     - Check `key in by_key`. If present, append to `entries` with `future = by_key[key]`, `duplicate = True`, continue.
     - Check budget reservation: `if (budget.used + len(active)) >= budget.maximum: immediate = budget_exhausted`, append to `entries`, continue.
     - Candidate requires simulation: set `cand.status = 'evaluating'`, submit task to pool, register in `active` and `by_key[key]`.
3. **Wait & Publish Step:**
   - If `active`: wait on `wait(active, return_when=FIRST_COMPLETED)` and remove completed futures from `active`.
   - Drain `publish_ready()`: while `next_publish < len(entries)` and entry is done:
     - Finalize metrics with duplicate handling and error isolation.
     - Call `finish(cand, metrics, key)`.
     - Evict `key` from `by_key` once stored or resolved.
     - Advance `next_publish += 1`.
4. **Guarantees:**
   - **Full Utilization:** Slots refill immediately upon any worker completion.
   - **Determinism:** Results, archives, and logs publish strictly in proposal order.
   - **Zero Redundancy:** In-flight duplicates and cache hits consume 0 simulator runs and 0 budget.
   - **Exact Budget:** Grade launches never exceed `budget.maximum`.
