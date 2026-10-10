# Agent Briefing: Transformer Hardware–Program Co-design

For the abstract problem formulation, see `project/PROBLEM.md`.

Read this file first. It formulates the whole assignment for an incoming agent
session: what the problem is, what the rules are, how to iterate, and what must
be logged. Authoritative sources: `README.md` (statement), `ISA.md` (instruction
set), `ABI.md` (HBM layout), `ASSEMBLY_GUIDE.md` (program walkthrough), and the
frozen evaluator under `codesign/challenge/`.

## 1. The problem in one paragraph

This is a **joint hardware/software optimization** task for a teaching
accelerator. You must submit **one** accelerator configuration
(`hardware.json`) that runs **two** fixed assembly programs: a prompt-prefill
program (`programs/M1_P1.asm`) and an autoregressive-decode program
(`programs/M2_D1.asm`). A deterministic simulator scores correctness, cycle
count, chip area, and power. You may change hardware menu choices *and* freely
redesign the programs (tiling, fusion, workgroup placement, scheduling, data
movement). The submission unit is the triple
`(H, P_prefill, P_decode)`.

## 2. Objective and constraints

Simulated clock: **500 MHz**. Frozen baseline cycles:

- `M1_P1` (prefill): **30,996,995** cycles
- `M2_D1` (decode): **1,302,032** cycles

```
score = 1000 * sqrt( (30996995 / T_P1) * (1302032 / T_D1) )
```

Baseline scores 1000; there is no maximum. A submission is **eligible** only if:

- both programs are functionally correct
  (`abs(cand - ref) <= 1e-3 + 1e-3 * abs(ref)` on every required hidden/KV
  element; nonfinite values fail);
- chip area ≤ **24 mm²** (`Hardware.area_mm2()` in
  `codesign/challenge/hardware.py`, coefficients in `cost_v04.json`);
- each case's max rolling 1000-cycle average power ≤ **20 W**;
- each case's cycles ≤ 2× its own baseline.

## 3. The workload (fixed, not chosen)

FP32 decoder-only Transformer inference. Per layer:
`LN1 → QKV projection → causal attention (scale 1/sqrt(32)) → output proj +
residual → LN2 → w1+b1 → tanh-approx GELU → w2+b2 → residual`.
LayerNorm ε = 1e-5. No embedding, vocab head, sampling, or quantization.

| Case | Model | Scenario contract |
| --- | --- | --- |
| `M1_P1` | M1: 3 layers, width 256, 8 heads × 32, FFN 1024 (~2.37M params) | Batch 2 × 64-position prompts. Compute both prompts' hidden + per-layer K/V, then `STEP.COMMIT {"step":0}`, then compute the first new position for each batch element (attends to own prompt + itself). |
| `M2_D1` | M2: 2 layers, width 128, 4 heads × 32, FFN 512 (~0.40M params) | Batch 1, 128-position KV history already in HBM. Eight sequential decode steps; step `s` attends to 128 history + `s+1` generated positions; `STEP.COMMIT {"step":s}` for every `s=0..7` (later inputs release only after earlier commits). |

Inputs are teacher-forced hidden states — you do not pick tokens. Every
required output element must be written **exactly once** to its published ABI
region (`ABI.md`); weights/history are read-only; future inputs are unreadable
until released.

## 4. The machine and the hardware menu

GPU-inspired single chip: identical SMs (tensor-core-like `MMA.ACC` units,
vector lanes, SFUs, optional reduction units, a fixed 256 KiB RF per SM split
into private 64 KiB workgroup slots, optional shared memory, DMA engines),
a shared NoC, HBM channels, and an optional chip-wide cache. See the full menu
in `README.md` §2 — tunables include `sm_count`, `tc_count`/`tc_array`/
`tc_k_parallel`, `vector_lanes`, `sfu_lanes`, `reduction_units`, `rf_ports`,
`shared_kib`/`shared_banks`/`shared_ports`, `dma_depth`/`dma_engines`,
`sm_noc_bytes_per_cycle`, `multicast`, `noc_bytes_per_cycle`, `hbm_channels`,
`cache_mib`.

The timing model is a **deterministic resource scheduler**, not a GPU. Intuitions
from real GPUs are hypotheses to test, not rules. One hardware JSON serves both
cases.

## 5. ISA essentials (see `ISA.md` for the contract)

One `OPCODE {json}` per line. Opcodes: `WG.BEGIN`/`WG.END` (named workgroup on an
explicit SM; ≤4 resident per SM), `LD`/`ST` (HBM↔RF↔SH copies with optional
`shape`/`strides` tiles), `MMA.ACC` (`acc += a@b` on RF tiles, needs `tc_count>0`),
`VEC` (add/sub/mul/div/fma/gt/select/max), `REDUCE` (sum/max → 1 element),
`SFU` (exp/rsqrt/tanh), `WAIT`, `BARRIER`, `STEP.COMMIT`, static `FOR`/`END.FOR`
with compile-time integer expressions. Memory addresses are **FP32 element
offsets**, not bytes. Key traps: initialize RF/accumulators before reading;
events must be unique after loop expansion; cross-workgroup HBM ordering needs
explicit `WAIT`/`BARRIER`/commit; an `ST` may still be reading its RF/SH source
after issue.

## 6. The iteration loop (how an agent should work)

The assembly files are **generated**, not hand-edited. Edit
`project/compiler.py` (a small Python emitter with `Builder`, `gemm`,
`layernorm`, `attention`, `attention_decode`, helpers) or write your own
generator, then regenerate.

Environment: **`grade` refuses to run unless the interpreter is Python 3.12.13
with NumPy 2.5.3** — the frozen baseline manifest pins them. `check`/`estimate`
need NumPy installed. Set up such an env first (e.g. pyenv/uv).

```sh
python -m project.compiler                     # regenerate programs/*.asm
python challenge.py check    --hardware hardware.json \
    --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm \
    --seed 7 --report project/check-N.json     # correctness FIRST
python challenge.py estimate ... --report project/estimate-N.json  # timing only
python challenge.py grade    ... --baseline baseline_manifest.json \
    --report project/grade-N.json              # public experimental score
```

Discipline: change **one idea at a time**; keep one `hardware.json` valid for
both programs; `estimate` is the fast exploration loop but only `check`
establishes correctness; keep every report file under `project/` as evidence.

## 7. Mandatory experiment log

After **every** experiment (a hardware or program change you evaluate), append
to `project/iteration-log.md` — this file is a **required submission artifact**
and proves the result was reached by genuine iteration. Entry template:

```markdown
## YYYY-MM-DD HH:MM — <one-line title>
- Change: <what changed, e.g. hardware.json field a→b, compiler tile 8→16>
- Verify: <command> -> <report path> -> passed/failed (+ why)
- Result: P1=… D1=… cycles (baseline 30996995/1302032), score ~…, area/power …
- Decision: keep/revert; next step and why
```

Also record rejected ideas — they are part of the evidence. Do not wait until
submission day; log as you go.

## 8. Frozen files — do not modify

`codesign/` (the evaluator), `MANIFEST.json`, `baseline_manifest.json`, and the
generated `programs/*.asm` (edit the *generator* instead). The grader executes
only the emitted assembly — never rely on Python scripts at grading time.

## 9. Submission packaging

ZIP (≤25 MiB) with these paths **at its root**:

```
hardware.json
programs/M1_P1.asm
programs/M2_D1.asm
project/iteration-log.md
project/<generator sources, scripts, experiment reports>
agent-trace/<one complete primary conversation export, native format>
```

**User override (2026-09-28):** Keep only one complete native export of the
current primary conversation in `agent-trace/`. Do not separately collect worker
or additional session exports. This supersedes the earlier all-session trace
collection instruction. Keep experiment reports and iteration logs as usual.

Deadlines: milestone 1 = **2026-10-07
12:00 Beijing**; milestone 2 (final) = **2026-10-12 12:00 Beijing**.

## 10. Repo map and current state

| Path | Role |
| --- | --- |
| `hardware.json` | Current shared accelerator config (baseline: 8 SM, 2×4x8 TC, 16 vec lanes, 128 KiB SHM, 4 HBM ch, no cache) |
| `project/compiler.py` | Editable program generator — the main thing you change |
| `programs/*.asm` | Generated output; regenerate, don't hand-edit |
| `codesign/challenge/` | Frozen parser/executor/scheduler/cost model — read-only |
| `baseline_manifest.json` | Frozen baseline metrics for `grade` |
| `ABI.md` | Every HBM symbol, offset, shape, release step |
| `ASSEMBLY_GUIDE.md` | Map of the generated programs' block structure |

Git: branch `dev-yh`, remote `origin` = GitHub `YihDu/SLAI-infra-course-hw1`.

### Experiment harness (project/search/, built 2026-09-28)

Lightweight deterministic controller — the agent proposes hypotheses and edits
candidate inputs; the harness only executes the fixed chain
`stage inputs → generate → check → estimate → grade → metrics.json →
append iteration-log.md`. Evaluator (challenge.py) is the sole ground truth;
correctness is a hard gate (failed check → status `incorrect`, no timing).

```sh
python -m project.search.controller status           # registry table
python -m project.search.controller new cand-NNN --parent cand-001 \
    --hypothesis "..." --summary "..." [--from-worktree | --compiler F --hardware F]
python -m project.search.controller run cand-NNN --seed 7   # ~6 min
python -m project.search.controller keep|cand-NNN --note "..."
python -m project.search.controller revert cand-NNN --note "..."  # restores parent inputs
python -m project.search.controller checkout cand-NNN      # stage inputs to worktree
```

Each `project/experiments/cand-NNN/` holds `candidate.json` (id, parent,
hypothesis, status), `inputs/` (compiler.py + hardware.json + recipe.json
snapshots — the source of truth, not the worktree), `programs/`, `reports/`,
`metrics.json`, `proposal.json`. Branching = `new --parent <non-head>`;
`checkout` realizes any branch. The log is append-only; verdicts append a
`### cand-NNN verdict:` entry.

Current registry root: cand-000 = baseline (score 1000.0, kept root),
cand-001 = B-stationary GEMM (score 1173.1).

### Autonomous search (project/search/, built 2026-09-28)

Operator-graph evolutionary search on top of the harness. Modules:

| File | Role |
| --- | --- |
| `search/dsl.py` | Structured operators (`gemm_order`, `gemm_tiles`, `gemm_prefetch`, `scores_residency`, `fuse_bias_gelu`; `hw_set`/`workgroups` locked at phase 1), recipe validation, fingerprints |
| `search/propose.py` | Proposal policy: 3 exploit incumbent + 2 expand archives + 1 novelty per round; structural templates (`STRUCTURES`) separate from the numeric tile grid (`legal_tiles`) |
| `search/engine.py` | `run_search(rounds, per_round, seeds, dry_run, max_grades)`; pruning cascade `DSL validate -> duplicate -> constraint -> generate -> static screen -> grade`; archives update; plateau policy; `search_state.md` |
| `search/archives.py` | elite (top-8 score) / pareto (P1,D1 cycles front) / diversity (family,bias cells) — correct candidates only |
| `search/evaluator.py` | `evaluate_search` = generate + static screen + one `grade` run (which already contains functional_passed + timing + score) |

Recipe-driven compiler: `project/compiler.py` loads `project/recipe.json`
(defaults = `dsl.DEFAULT_RECIPE`, byte-identical output to cand-001). DSL-only
candidates inherit `project/experiments/_shared/compiler.py`; a proposal may
pin a different compiler via `compiler_path`.

Commands:

```sh
python -m project.search.controller search-plan --rounds 2 --per-round 6   # dry run
python -m project.search.controller search-run  --rounds 5 --per-round 6 \
    --max-grades 30 --seed 7                                              # real run
python -m project.search.controller graph     # candidate DAG
python -m project.search.controller state-md  # regenerate search_state.md
```

Search state lives in `registry.json["search"]` (round, plateau_rounds,
explore_ratio, need_new_operator, phase). Plateau policy: 3 flat rounds ->
explore_ratio rises; 6 -> `need_new_operator` (agent must register a new
structural template, not keep tuning). Phase 1 searches recipe only; phase 2
unlocks `hw_set`/`workgroups`.

Known trap fixed 2026-09-28: `next_id` collided with manually created ids and
once overwrote `cand-001`'s `candidate.json`/`inputs/compiler.py`; restored
with a `provenance_note`. Never write into an existing `cand-NNN/` dir.

Dry-run candidates use `drun-NNN` ids (mechanism tests, no evaluator).

## 11. Known baseline weaknesses → first hypotheses to test

The shipped baseline is deliberately naive; all of these are *measured* targets:

1. **One workgroup on SM 0** — 7 of 8 SMs idle. P1's batch=2, per-head, and
   N-column parallelism are free concurrency; D1 is sequential across steps but
   parallel across heads/layers' weights prefetch.
2. **All intermediates round-trip HBM scratch** — fuse LN→QKV, attention, and
   FFN stages in RF/SHM where capacity allows.
3. **Fixed 8×16×48 GEMM tiles, N-tile restricted to 8/16** — larger/other tiles,
   tail tiles, and better K splits may raise TC utilization.
4. **Attention materializes scores/probs in HBM** — online softmax or RF-resident
   attention removes traffic.
5. **`shared_kib=128` paid for but `shared_bytes=0`** — unused area; either use
   SHM for staged weights/KV or drop it to save area.
6. **No DMA pipelining** (`dma_depth=1`, 1 engine) and no cache — prefetch and
   `cache_mib`/`multicast` may hide HBM latency.

Measure each against the simulator; keep what measurably helps on **both** cases.
