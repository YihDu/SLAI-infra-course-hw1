# Problem Formulation: Transformer Hardware–Program Co-design

This document abstracts the assignment into a self-contained optimization
problem. It is written so that any agent (or reader) can understand *what* is
being optimized, *over which space*, *under which oracle*, and *why* the
solution method is an agent-guided search loop — without first reading the
whole starter package.

## 1. Decision variables

The artifact to produce is a triple:

```
(H, P_prefill, P_decode)
```

- **H** — one accelerator configuration drawn from a fixed hardware menu
  (~120.7 billion raw combinations). Each SM may carry matrix engines (TCs),
  vector lanes, SFUs, reduction units, a fixed 256 KiB register file, optional
  shared memory, and DMA engines; chip-wide choices are the NoC bandwidth,
  per-SM NoC links, HBM channel count, multicast, and an optional shared cache.
  H is described by `hardware.json` and is **shared by both programs**.

- **P_prefill** — an assembly program for case `M1_P1`: batch 2 × 64-position
  prompt prefill through model M1 (3 layers, width 256, 8×32 heads, FFN 1024),
  then one new position per batch element.

- **P_decode** — an assembly program for case `M2_D1`: eight sequential decode
  steps through model M2 (2 layers, width 128, 4×32 heads, FFN 512) against a
  128-position KV history.

The program space is open-ended: tiling and tail tiles, loop order, RF/shared
memory residency, fusion, recomputation, attention algorithm (e.g. online
softmax), workgroup count and SM placement, DMA scheduling, synchronization.
Programs are emitted by an editable generator (`project/compiler.py`), not
hand-written.

## 2. The computation (fixed workload)

Both cases run the same layer function, in FP32:

```
x ──> LN1 ──> QKV proj ──> causal attention (scale 1/√32) ──> WO proj ──> +x ──>
    ──> LN2 ──> W1+b1 ──> tanh-GELU ──> W2+b2 ──> +residual ──> x'
```

Per layer, newly computed K/V must be exported to the ABI regions. Final-layer
hidden states go to `output/hidden`. LayerNorm ε = 1e-5. There is no
embedding, vocab projection, sampling, or quantization.

The interface contract that makes the two cases different:

- **P1**: must write *both* prompts' outputs and K/V, then
  `STEP.COMMIT {"step":0}` releases the two `input/step0` vectors; the two
  first-new-position results are checked at case end.
- **D1**: `input/step{s}` is readable only after steps `< s` commit; every step
  `s=0..7` must produce output+K/V then `STEP.COMMIT {"step":s}`. Attention at
  step `s` covers 128 history + `s+1` generated positions.

## 3. Objective function

Deterministic simulated execution at a fixed 500 MHz clock. Frozen baseline
cycle counts: `T_P1^0 = 30,996,995`, `T_D1^0 = 1,302,032`.

```
score(H, P1, P2) = 1000 · sqrt( T_P1^0 / T_P1  ·  T_D1^0 / T_D1 )   if eligible
                 = ineligible → no score                          otherwise
```

The geometric mean weights both cases equally in log-space: a 2× speedup on
one case is worth the same as √2× on both.

## 4. Constraints (eligibility gates)

1. **Correctness** — FP32 functional executor compares every required hidden
   and new-K/V element: `|cand − ref| ≤ 1e-3 + 1e-3·|ref|`; nonfinite fails.
   Instruction-level rules: initialize before read, write required outputs
   exactly once, never read unreleased inputs, unique events, legal RF/SH/HBM
   views, all workgroups ended, commits in order.
2. **Area** — `area(H) ≤ 24 mm²`, per the formula in `hardware.py` with
   coefficients `cost_v04.json`.
3. **Power** — per case, max rolling 1000-cycle average ≤ 20 W.
4. **Latency floor** — each case ≤ 2× its own baseline cycles.

## 5. The oracle

`challenge.py` provides three evaluation modes over `(hardware.json, asm,
asm)`:

- `estimate` — timing only (fast exploration signal; does not prove
  correctness);
- `check` — functional correctness against public seed inputs;
- `grade` — correctness + timing + gates → public experimental score.

Timing comes from a deterministic resource scheduler
(`pipeline_events.py`, `service.py`): instruction issue, TC/vector/SFU
service times, RF/SH port contention, DMA queues, NoC and HBM channel
contention, cache behavior. **This model — not GPU intuition — is ground
truth.** Properties that matter on real GPUs may not hold here and vice versa.

## 6. Why this is a co-design problem

H and P interact: more TCs only pay off if the program emits tiles that fill
them; more SMs only help if workgroups are placed on them; shared memory/cache
only pay if the schedule creates reuse; area spent on unused resources is pure
cost. The space is far too large for enumeration (~1.2×10¹¹ hardware configs
× effectively unbounded program space), and the objective is only measurable
through the simulator — so this is a **black-box, constrained optimization /
program-synthesis** problem.

## 7. The agentic solution method

The intended method is an explicit search loop driven by an AI agent, with
measured feedback:

```
repeat:
    1. PROPOSE  — pick a joint change (H field(s) and/or generator change),
                  informed by the accumulated experiment history
    2. GENERATE — run the compiler to emit both .asm programs
    3. VERIFY   — `check` must pass; reject on any correctness violation
    4. MEASURE  — `estimate`/`grade` → cycles, area, power, score
    5. LEARN    — record the outcome in project/iteration-log.md;
                  keep, revert, or branch the search
until diminishing returns or deadline
```

Design rules for the search: one idea per iteration; `estimate` for cheap
screening, `check` before trusting any candidate; keep `hardware.json` legal
for both programs; never hand-edit `.asm` (edit the generator); never modify
the frozen evaluator (`codesign/`, `baseline_manifest.json`).

Optional aids are allowed but not required: MILP for bounded subproblems,
heuristic cost models, compiler passes — the judge only scores the emitted
hardware+assembly.

## 8. Process constraints (part of the deliverable)

Because submissions are graded partly on demonstrated process:

- every evaluated candidate → an entry in `project/iteration-log.md`;
- every agent session used on the project → its native trace file under
  `agent-trace/` (all tools, all sessions, including dead ends);
- generator sources and experiment reports stay under `project/`.

Final ZIP (root paths): `hardware.json`, `programs/M1_P1.asm`,
`programs/M2_D1.asm`, `project/…`, `agent-trace/…`.
Deadlines (Beijing): milestone 1 = 2026-10-07 12:00; final = 2026-10-12 12:00.

## 9. Environment prerequisite

`grade` validates that the runtime matches the frozen baseline's pinned
environment: **Python 3.12.13 + NumPy 2.5.3** (`cli.py` `_load_baseline`
compares them). `check`/`estimate` need NumPy. Set up that interpreter before
the first experiment.
