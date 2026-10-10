> PRIMARY REVIEW: Raw Gemini research note, retained as a dead-end/research artifact. Several claims below are inaccurate: cand-098 cache is 2 MiB (not 1); 9.289e9 pJ = 9.289 mJ (not 9,289 mJ); predicted stage durations were not measured. Use project/campaigns/p1-multism/REPORT.md and reports/p1-profile.json for verified conclusions.

# Technical Review: Stage Elapsed Cycles and Overlapping Per-SM Resource Attribution for Multi-SM P1

**Status**: Read-only Architectural Review & Methodology Specification
**Scope**: Frozen evaluator timing APIs (`codesign/challenge/pipeline.py`, `pipeline_events.py`), historical grade inspection (`project/experiments/cand-098/reports/grade.json`), multi-SM execution semantics, and non-invasive stage attribution design.
**Target File**: `project/search/p1_attribution_review.md`

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Co-design Optimization Dilemma
In the transformer hardware–software co-design challenge, prompt prefill (`M1_P1`) is the dominant component of the objective function:
$$\text{score} = 1000 \times \sqrt{ \left( \frac{30,996,995}{T_{\text{P1}}} \right) \times \left( \frac{1,302,032}{T_{\text{D1}}} \right) }$$
The shipped baseline executes both programs with a single workgroup pinned to SM 0 (`sm_count=8`, 7 SMs idle). For `M1_P1`, scaling from 1 SM to multi-SM (e.g. 2, 4, or 8 SMs) via tensor-parallel column splitting, batch parallelism, or head distribution is the single largest algorithmic opportunity.

However, moving to multi-SM introduces severe execution overlap:
1. Multiple SMs execute independent workgroups asynchronously.
2. Background DMA engines pipeline HBM weight/activation transfers concurrently with tensor-core compute.
3. Stages (LayerNorm, QKV projection, Causal Attention, FFN W1/W2) exhibit producer-consumer dependencies, partial concurrency, and synchronization barriers.

### 1.2 The Diagnostic Gap in Standard Reports
The standard evaluator outputs macro-level summary metrics (total cycles, total energy, 128-bin timeline histograms). In `project/experiments/cand-098/reports/grade.json`, `M1_P1` achieved 8,522,411 cycles. While this certifies functional correctness and total execution time, it provides zero insight into:
- The exact cycle makespan of individual Transformer stages (e.g. How long did Attention take vs. FFN W1?).
- Load imbalance across SMs (Is SM 0 straggling while SM 1–3 wait at a barrier?).
- Resource bottlenecks per stage (Is QKV TC-bound or NoC-bandwidth bound?).
- The degree of temporal overlap between concurrent stages.

### 1.3 Core Recommendation: Non-Invasive Trace Callbacks
The frozen evaluator (`codesign/challenge/`) already contains an unexposed but fully functional tracing hook in `estimate_pipeline(...)`:
```python
def estimate_pipeline(
    hardware: Hardware,
    program,
    hbm_words: int,
    step_requirements: dict[int, list[tuple[int, int]]] | None = None,
    trace: dict[str, dict[str, int]] | None = None,
    energy_trace: list[EnergyEvent] | None = None,
) -> PerfResult
```
By passing a dictionary `trace={}`, the simulator populates exact integer cycle timestamps (`issue` and `finish`) for every event-bearing instruction in the program.

**No evaluator modifications are required or permitted.** By coupling this existing trace callback with a structured event-naming scheme in the compiler, we can derive exact stage makespans, per-SM active times, overlap matrices, and resource attribution.

---

## 2. In-Depth Audit of cand-098 Baseline (`cand-098/reports/grade.json`)

`cand-098` represents a verified, full-grade submission evaluated on 2026-09-28. Inspecting its generated report provides the baseline for what data is currently available and where the attribution gaps lie.

### 2.1 Macro Metrics in cand-098
From `project/experiments/cand-098/reports/grade.json`:
- **Cycles**: 8,522,411 (Baseline: 30,996,995, speedup: 3.637x)
- **Instructions**: 161,519
- **HBM Traffic**: Read = 47,165,440 B (45.0 MiB), Write = 8,120,320 B (7.74 MiB)
- **Cache Traffic** (1 MiB cache): Hits = 1,012,320, Misses = 736,960 (Hit rate: 57.86%)
- **Power & Energy**:
  - Dynamic Energy: 9,289.22 mJ
  - Total Energy: 16,217.25 mJ
  - Average Power: 0.951 W (well below the 20.0 W gate)
  - Peak 1000-cycle Window Power: 2.971 W (instantaneous upper bound: 10.219 W)
- **Chip Area**: 16.258 mm² (Gate: <= 24.0 mm²)

### 2.2 Per-SM and Shared Resource Breakdown in cand-098
In `timing.resource_stats`:
1. **SM Accounting (`sms`)**:
   `cand-098` contains **only key `"0"`** because its program only issued instructions for workgroup `"g"` on SM 0 (`WG.BEGIN {"wg": "g", "sm": 0, ...}`).
   - `busy_cycles`:
     - `rf_read`: 2,818,280
     - `rf_write`: 2,788,264
     - `dma`: 1,876,160
     - `noc_in`: 1,563,058
     - `tc` (Tensor Core): 586,032
     - `vec` (Vector ALUs): 564,984
     - `sfu` (Special Function Unit): 269,964
     - `issue`: 161,519
     - `noc_out`: 126,844
     - `shared_read` / `shared_write`: 0 (Shared memory was allocated but unused)
   - `dma_bytes`: 120,074,240 B
2. **Shared Accounting (`shared`)**:
   - `noc` busy cycles: 1,667,790 (Capacity: 256 B/cyc)
   - `hbm_channel_0..3` busy cycles: 431,920 each (Identical load across 4 channels)
   - `cache_bank_0..3` busy cycles: 621,560 each

### 2.3 Coarse Timeline Histogram (`timeline`)
The timeline field contains compact, binned Riemann sums produced by `codesign/challenge/resource_timeline.py`:
- `edges_cycles`: 129 edges forming **128 uniform bins** from cycle 0 to 8,522,411.
- **Bin Width**: Delta t = ceil(8,522,411 / 128) = 66,582 cycles.
- For each bin, it records integrated `busy_cycles`, `utilization`, `bytes`, and `dynamic_energy_pj`.

### 2.4 Critical Attribution Deficiencies of the Standard Report
1. **Zero Stage Granularity**: The report aggregates the entire 8.52M cycles across 3 layers, prompt batch 2, and step 0 decode into a single monolithic bundle. There is no way to know whether QKV took 2M cycles or 5M cycles.
2. **Temporal Smearing (66.5k Cycle Resolution)**:
   A single LayerNorm operation over 64 tokens of width 256 requires only ~2,000 to 4,000 cycles. Inside a 66,582-cycle bin, this fast stage is completely smeared out and indistinguishable from background idle time or adjacent GEMM operations.
3. **Inability to Attribute Multi-SM Load Imbalance**:
   In a multi-SM run, `resource_stats["sms"]` will report keys `"0"`, `"1"`, ..., `"sm_count-1"`. However, the timeline bins only report the aggregate busy cycles of each SM within each 66k cycle window. If SM 0 finishes its tile at cycle 10,000 and waits 56,000 cycles for SM 3 at a barrier, the timeline reports partial utilization for that bin but cannot attribute the stall to a specific straggler in a specific stage.

---

## 3. Frozen Pipeline Timing Trace APIs (Under-the-Hood Analysis)

### 3.1 Timing Model Architecture
The challenge timing model (`TIMING_MODEL = "pipeline-global-events-v2"`) is implemented in `codesign/challenge/pipeline_events.py` via `GlobalEventPipeline`.
It is a deterministic, event-driven resource calendar simulator.

```
       +---------------------------------------------+
       |   codesign/challenge/pipeline.py            |
       |   def estimate_pipeline(hw, prog, words,    |
       |       step_reqs, trace, energy_trace)       |
       +----------------------+----------------------+
                              | delegates to
                              v
       +---------------------------------------------+
       |   codesign/challenge/pipeline_events.py     |
       |   def estimate_global_events(...)           |
       |       -> GlobalEventPipeline(...).run()     |
       +----------------------+----------------------+
                              | populates
                              v
                 trace[event] = {issue, finish}
```

### 3.2 Exact API Signatures

```python
# Entry point in codesign/challenge/pipeline.py
def estimate_pipeline(
    hardware: Hardware,
    program,  # Iterable[Instruction] from iter_parse(program_str)
    hbm_words: int,
    step_requirements: dict[int, list[tuple[int, int]]] | None = None,
    trace: dict[str, dict[str, int]] | None = None,
    energy_trace: list[EnergyEvent] | None = None,
) -> PerfResult:
    from .pipeline_events import estimate_global_events
    return estimate_global_events(
        hardware, program, hbm_words, step_requirements, trace, energy_trace
    )
```

### 3.3 The `trace` Parameter Mechanics
In `GlobalEventPipeline.__init__` (`pipeline_events.py:298`):
```python
self.trace = trace
```
Whenever an event-bearing instruction completes service, `GlobalEventPipeline._record` is invoked (`pipeline_events.py:380`):
```python
def _record(self, name: str, issue: int, finish: int):
    if name in self.queues.waited_events:
        self.events[name] = finish
    if self.trace is not None:
        self.trace[name] = {"issue": issue, "finish": finish}
    self.last_finish = max(self.last_finish, finish)
```

#### What Gets Recorded in `trace[name]`
1. **Compute Instructions (`MMA.ACC`, `VEC`, `REDUCE`, `SFU`)**:
   - `_issue_compute` (`pipeline_events.py:610-666`):
     - `begin`: Cycle when the instruction issued from the workgroup tape.
     - `read_end`: Cycle when source RF read completed.
     - `compute_end`: Cycle when functional unit service (TC / VEC / SFU / REDUCE) finished.
     - `finish`: Cycle when destination RF write completed (`compute_end + ceil(write_bytes / write_port)`).
     - Records: `self._record(args["event"], begin, finish)`
     - Result: `issue = begin`, `finish = RF write complete`.
2. **Local Transfers (`LD`/`ST` between RF and SH or RF and RF)**:
   - `_issue_transfer` (`pipeline_events.py:730-745`):
     - Synchronous memory copy.
     - Records: `self._record(ins.args["event"], begin, finish)`
3. **Asynchronous HBM Transfers (`LD`/`ST` involving HBM)**:
   - When issued, a `PendingTransfer` struct is created (`pipeline_events.py:755`).
   - Transfer is segmented into 64-byte cache lines, scheduled over `dma_slice`, `noc_in`/`noc_out`, `noc`, and `hbm_channels`.
   - When all lines finish arrival and local RF/SH write or HBM write drain completes, `_finish_transfer` is called (`pipeline_events.py:982`):
     - Records: `self._record(transfer.instruction.args["event"], transfer.issue, finish)`
     - Result: `issue = transfer.issue` (when the instruction was dispatched), `finish = drain/writeback cycle`.

### 3.4 The `energy_trace` Parameter Mechanics
In `GlobalEventPipeline._finish_energy` and `_make_timeline`:
If `energy_trace` is passed as a `list[EnergyEvent]`, the pipeline populates:
```python
EnergyEvent(start: int, finish: int, dynamic_pj: float)
```
Each entry represents an active service interval (coalesced by resource and SM) with its uniform dynamic energy charge. Note: `EnergyEvent` does not store the SM ID or opcode, but its `start` and `finish` correspond to the exact calendar intervals.

### 3.5 Handling Non-Event-Bearing Instructions
The following opcodes in `codesign/challenge/isa.py` do not accept an `event` field:
- `WG.BEGIN`, `WG.END`: Workgroup boundary setup and teardown.
- `FOR`, `END.FOR`: Statically unrolled during parsing/grouping; not seen by simulator.
- `WAIT`: Stalls the workgroup tape until specified events reach their `finish` cycle.
- `BARRIER`: Stalls listed workgroup tapes until all listed groups reach the barrier and prior events complete.
- `STEP.COMMIT`: Global sync barrier for ABI step verification.

**How to measure stall/barrier overhead:**
Because `WAIT` and `BARRIER` do not generate event records, their elapsed delay is captured by the delta between the `finish` of the waited event and the `issue` cycle of the next instruction on that workgroup tape.

---

## 4. Multi-SM P1 Stage Event Mapping Approach

### 4.1 M1_P1 Workload Stage Taxonomy
`M1_P1` executes FP32 inference on Model M1 (Layers $L=3$, Width $D=256$, Heads $H=8$, Head Dimension $D_{\text{head}}=32$, FFN $F=1024$).
- **Phase `prompt` (`p`)**: Batch size $B=2$, sequence length $S=64$.
- **STEP.COMMIT**: Step 0 boundary commit.
- **Phase `step` (`s`)**: Batch size $B=2$, sequence length $S=1$ (first generated decode token).

For each batch member $b \in \{0, 1\}$ and each layer $l \in \{0, 1, 2\}$, the logical stage sequence is:
1. `LN1`: LayerNorm 1 over input activations ($S \times D$).
2. `QKV`: GEMM projection from $D \to 3D$ ($S \times 256 \times 768$).
3. `ATTN_SCORE`: Causal scaled dot-product $Q K^T / \sqrt{32}$.
4. `ATTN_SOFTMAX`: Max reduction, SFU exp, sum reduction, division.
5. `ATTN_CONTEXT`: Attention context GEMM $P V$ ($S \times 64 \times 32$ per head).
6. `ATTN_OUT`: Output projection GEMM $W_O$ ($S \times 256 \times 256$).
7. `RES1`: Elementwise residual addition ($X + \text{Attn}$).
8. `LN2`: LayerNorm 2 over residual ($S \times D$).
9. `FFN_W1`: First feed-forward GEMM projection ($S \times 256 \times 1024$).
10. `FFN_ACT`: Bias add + GELU activation (or fused `bias_gelu`).
11. `FFN_W2`: Second feed-forward GEMM projection ($S \times 1024 \times 256$).
12. `RES2`: Elementwise residual addition ($X_{\text{LN2}} + \text{FFN}_{W2} + b_2$).
13. `OUTPUT_ST`: Store final hidden states to published ABI region.

### 4.2 Structured Event Naming Scheme (Recommended)
The ISA specification states:
> "An `event` is a unique nonempty name for an operation... Strings containing braces interpolate a bound loop index... Give every expanded event-bearing operation a distinct event, including across loop iterations and reused workgroup names."

We can design a canonical naming grammar embedded directly into the compiler generator (`project/compiler.py`).

#### Grammar Specification
```
<event_id> ::= <phase> "." <batch> "." <layer> "." <stage> "." <op> "." <sm> "." <wg> "." <iter_suffix>

Where:
  <phase>       ::= "p" | "s"                    (prompt vs. step0)
  <batch>       ::= "b0" | "b1"                  (batch element)
  <layer>       ::= "l0" | "l1" | "l2"         (transformer layer)
  <stage>       ::= "ln1" | "qkv" | "ascore" | "asoft" | "actx" | "wo"
                  | "res1" | "ln2" | "w1" | "gelu" | "w2" | "res2" | "out"
  <op>          ::= "ld" | "st" | "mma" | "vec" | "red" | "sfu"
  <sm>          ::= "sm" <int>                     (target SM: sm0, sm1, ...)
  <wg>          ::= "wg" <int>                     (workgroup ID)
  <iter_suffix> ::= loop index interpolation strings (e.g. "m{m}_n{n}_k{k}")
```

#### Examples of Expanded Event Names
- `p.b0.l0.qkv.ld.sm0.wg0.m0_k0`: HBM weight load for QKV tile on SM 0.
- `p.b0.l0.qkv.mma.sm1.wg1.m0_n1_k0`: MMA accumulation on SM 1.
- `p.b1.l2.gelu.sfu.sm2.wg2.r15_c0`: SFU evaluation of GELU tanh approximation on SM 2.
- `p.b0.l0.res2.st.sm3.wg3.r63`: Store final layer output to HBM.

### 4.3 Alternative: Compile-Time Metadata Sidecar
If modifying the event naming format inside the assembly is undesirable, the compiler can maintain an internal dictionary:
```python
stage_sidecar = {
    "e123_0_1": {
        "phase": "prompt",
        "batch": 0,
        "layer": 1,
        "stage": "w1",
        "sm": 2,
        "wg": "g2",
        "op": "MMA.ACC"
    },
    ...
}
```
**Tradeoff**: Strategy A (Structured Event Names) is superior because the assembly file is completely self-describing; any post-run analysis script or profiler can parse the event string using a single regular expression without needing auxiliary JSON sidecars.

---

## 5. Attribution Caveats in Multi-SM Overlapping Execution

Accurately attributing cycles in an asynchronous, multi-SM, pipelined accelerator requires addressing several non-trivial mathematical and hardware subtleties.

### 5.1 Caveat 1: Wall-Clock Makespan vs. Sum-of-Stages
In a serial execution model (like cand-000 baseline):
$$T_{\text{total}} = \sum_{\text{stages}} T_{\text{stage}}$$
In a multi-SM or pipelined execution model, **this equation is false**:
1. **Parallel Concurrency**: If 4 SMs run 4 column tiles of QKV simultaneously from cycle 1,000 to cycle 10,000, each SM did 9,000 cycles of work, but the elapsed wall-clock time for the stage is 9,000 cycles, **not** $4 \times 9,000 = 36,000$ cycles.
2. **Inter-Stage Pipelining**: If SM 0 finishes QKV column tile 0 early and initiates Attention score computation while SM 3 is still finishing QKV column tile 3, the QKV stage and Attention stage overlap in wall-clock time:
   $$[t_{\text{start}}^{\text{attn}}, t_{\text{end}}^{\text{attn}}] \cap [t_{\text{start}}^{\text{qkv}}, t_{\text{end}}^{\text{qkv}}] \neq \emptyset$$
3. **Idle Gaps & Barriers**: If all SMs complete QKV at cycle 20,000 and wait for HBM writes to drain before starting Attention at cycle 22,000, the 2,000-cycle gap must be attributed to synchronization / drain stall, not compute.

**Conclusion**: Attribution must clearly separate **Elapsed Wall-Clock Span (Makespan)** from **Cumulative Active Entity-Cycles**.

### 5.2 Caveat 2: Stragglers, Load Imbalance, and Barrier Stalls
Consider 4 SMs executing FFN W1 ($M=64, K=256, N=1024$), where each SM computes $N/4 = 256$ columns:
- SM 0 finishes at cycle 15,200.
- SM 1 finishes at cycle 15,250.
- SM 2 finishes at cycle 15,300.
- SM 3 suffers cache misses or NoC contention and finishes at cycle 18,400.
- A `BARRIER` synchronizes all 4 SMs before FFN W2.

```
Cycle:      10,000                   15,200           18,400
SM 0:       |====== Active Compute =====|------ Idle Stall ------|
SM 1:       |====== Active Compute ======|----- Idle Stall ------|
SM 2:       |====== Active Compute =======|---- Idle Stall ------|
SM 3:       |================ Active Compute ====================|
                                                         ^ Barrier release
```

If we only report average active time, SM 0 appears to take 5,200 cycles. But the **stage elapsed time** is $18,400 - 10,000 = 8,400$ cycles!
SM 0, 1, and 2 spent up to 3,200 cycles stalled.
**Metric Required**:
$$\text{Stage Straggler Penalty} = \max_{\text{sm}} (t_{\text{end}}(S, \text{sm})) - \min_{\text{sm}} (t_{\text{end}}(S, \text{sm}))$$
$$\text{Imbalance Efficiency} = \frac{\sum_{\text{sm}} \text{ActiveCycles}(S, \text{sm})}{\text{sm\_count} \times \text{ElapsedCycles}(S)}$$

### 5.3 Caveat 3: Shared Resource Contention Attribution
In the teaching machine:
- Shared NoC: single ring / crossbar with `noc_bytes_per_cycle` (e.g. 64 or 256 B/cyc).
- HBM Channels: e.g. 4 channels, each serving 16 B/cyc.
- Shared Cache: e.g. 4 banks, each serving 16 B/cyc.

When SM 0 and SM 1 concurrently stream weights from HBM:
- Both transfers are issued at $t=100$.
- If run in isolation, each transfer takes 400 cycles.
- When run concurrently, NoC or channel serialization expands the latency to 780 cycles.
- The trace records:
  - `SM0.LD`: `issue: 100, finish: 880`
  - `SM1.LD`: `issue: 100, finish: 875`
- The profiler must not treat this expanded latency as algorithmic inefficiency; it is **shared resource queueing delay**. Attributing shared resource consumption by byte volume (from HBM/NoC transfer sizes) resolves the contention cost accurately.

### 5.4 Caveat 4: Asynchronous DMA Latency vs. Hidden Latency
In `codesign/challenge/pipeline_events.py:755`:
When an `LD` from HBM to RF is issued:
- The instruction issues at $t_{\text{issue}}$.
- The DMA transfer runs in the background.
- If the compiler scheduled independent compute (e.g. vector math or another tile's MMA) immediately after `LD`, the compute units execute while DMA is in flight.
- `trace[ld_event]` will record `{"issue": t_issue, "finish": t_finish}`.
- If $t_{\text{finish}}$ occurs while the SM is busy computing, **zero stall cycles occurred**.
- The stage elapsed time must not double-count the DMA latency if it was fully hidden. The true stage critical path is determined by the consumer's earliest execution cycle.

### 5.5 Caveat 5: Silent Event Overwrite Trap in `_record()`
In `codesign/challenge/pipeline_events.py:384`:
```python
if self.trace is not None:
    self.trace[name] = {"issue": issue, "finish": finish}
```
`self.trace` is a standard Python `dict`. If two instructions emit the same event name string (for instance, if a loop variable interpolation is omitted or if two workgroups on different SMs use the same event name `"load_tile"`), **the second instruction silently overwrites the first**!
**Hard Requirement**: The event naming scheme must guarantee 100% uniqueness across the entire expanded execution stream (including phase, batch, layer, stage, SM, and loop indices).

---

## 6. Mathematical Framework & Metrics Definitions

Let $\mathcal{E}$ be the set of all event records captured in `trace`:
$$e = (\text{name}, \text{issue}, \text{finish})$$
Let $\mathcal{S}$ be the set of all logical stages (e.g. $S = \text{p.b0.l0.qkv}$).
For any stage $S$, let $\mathcal{E}_S \subseteq \mathcal{E}$ be the subset of events belonging to $S$.

### 6.1 Stage Boundaries and Elapsed Makespan
For stage $S$:
$$t_{\text{start}}(S) = \min_{e \in \mathcal{E}_S} (e.\text{issue})$$
$$t_{\text{end}}(S) = \max_{e \in \mathcal{E}_S} (e.\text{finish})$$
$$\text{Elapsed}(S) = t_{\text{end}}(S) - t_{\text{start}}(S)$$

### 6.2 Per-SM Active Time
For stage $S$ on SM $m$:
$$\mathcal{E}_{S, m} = \{ e \in \mathcal{E}_S \mid \text{sm}(e) = m \}$$
$$t_{\text{start}}(S, m) = \min_{e \in \mathcal{E}_{S, m}} (e.\text{issue})$$
$$t_{\text{end}}(S, m) = \max_{e \in \mathcal{E}_{S, m}} (e.\text{finish})$$
$$\text{ActiveSpan}(S, m) = t_{\text{end}}(S, m) - t_{\text{start}}(S, m)$$

### 6.3 Straggler and Imbalance Metrics
$$\text{StragglerLatency}(S) = \max_m (t_{\text{end}}(S, m)) - \min_m (t_{\text{end}}(S, m))$$
$$\text{ImbalanceLoss}(S) = \sum_m \left( \max_{m'} (t_{\text{end}}(S, m')) - t_{\text{end}}(S, m) \right)$$

### 6.4 Stage Overlap Matrix
For any two stages $S_i$ and $S_j$:
$$\text{OverlapCycles}(S_i, S_j) = \max\left(0, \min(t_{\text{end}}(S_i), t_{\text{end}}(S_j)) - \max(t_{\text{start}}(S_i), t_{\text{start}}(S_j))\right)$$
$$\text{OverlapRatio}(S_i, S_j) = \frac{\text{OverlapCycles}(S_i, S_j)}{\min(\text{Elapsed}(S_i), \text{Elapsed}(S_j))}$$

An overlap ratio near 1.0 indicates perfect pipelined execution between the two stages.

---

## 7. Implementation Blueprint: Standalone Attribution Profiler (`p1_profiler.py`)

Below is the complete, non-invasive architecture for extracting stage attribution using existing simulator APIs without altering `codesign/challenge/`.

```python
"""project/search/p1_profiler.py

Standalone Stage-Level Attribution Profiler for Multi-SM P1.
Executes against frozen codesign evaluator without modifying any challenge files.
"""

import re
import json
from dataclasses import dataclass
from pathlib import Path

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.abi import build_layout, MODELS
from codesign.challenge.runner import required_hbm_words, step_requirements
from codesign.challenge.pipeline import estimate_pipeline


STAGE_REGEX = re.compile(
    r"^(?P<phase>[ps])\.(?P<batch>b\d+)\.(?P<layer>l\d+)\.(?P<stage>[a-z0-9_]+)\.(?P<op>[a-z0-9_]+)\.(?P<sm>sm\d+)"
)


@dataclass
class StageStats:
    phase: str
    batch: str
    layer: str
    stage: str
    start_cycle: int = 1 << 60
    end_cycle: int = 0
    sm_starts: dict[str, int] = None
    sm_ends: dict[str, int] = None
    compute_events: int = 0
    transfer_events: int = 0

    def __post_init__(self):
        if self.sm_starts is None:
            self.sm_starts = {}
        if self.sm_ends is None:
            self.sm_ends = {}


def profile_p1(hardware_path: Path, program_path: Path) -> dict:
    with open(hardware_path) as f:
        hardware = Hardware.from_dict(json.load(f))
    program_str = program_path.read_text(encoding="utf-8")

    model = MODELS["M1"]
    layout = build_layout(model, "P1")
    words = required_hbm_words(program_str, layout)
    step_reqs = step_requirements("M1_P1", layout)

    trace = {}
    perf_result = estimate_pipeline(
        hardware=hardware,
        program=iter_parse(program_str),
        hbm_words=words,
        step_requirements=step_reqs,
        trace=trace,
    )

    # Aggregate events by stage
    stages: dict[str, StageStats] = {}
    unmatched_events = 0

    for name, span in trace.items():
        issue, finish = span["issue"], span["finish"]
        match = STAGE_REGEX.match(name)
        if not match:
            unmatched_events += 1
            stage_key = "unattributed"
            phase = batch = layer = stage = op = sm = "unknown"
        else:
            d = match.groupdict()
            phase, batch, layer, stage, op, sm = (
                d["phase"], d["batch"], d["layer"], d["stage"], d["op"], d["sm"]
            )
            stage_key = f"{phase}.{batch}.{layer}.{stage}"

        if stage_key not in stages:
            stages[stage_key] = StageStats(phase=phase, batch=batch, layer=layer, stage=stage)

        st = stages[stage_key]
        st.start_cycle = min(st.start_cycle, issue)
        st.end_cycle = max(st.end_cycle, finish)
        st.sm_starts[sm] = min(st.sm_starts.get(sm, issue), issue)
        st.sm_ends[sm] = max(st.sm_ends.get(sm, finish), finish)

        if op in {"mma", "vec", "red", "sfu"}:
            st.compute_events += 1
        elif op in {"ld", "st"}:
            st.transfer_events += 1

    # Format report
    stage_table = []
    for key, st in sorted(stages.items(), key=lambda item: item[1].start_cycle):
        elapsed = st.end_cycle - st.start_cycle
        straggle = max(st.sm_ends.values()) - min(st.sm_ends.values()) if st.sm_ends else 0
        stage_table.append({
            "stage_key": key,
            "start": st.start_cycle,
            "end": st.end_cycle,
            "elapsed": elapsed,
            "straggle_penalty": straggle,
            "active_sms": list(st.sm_starts.keys()),
            "compute_events": st.compute_events,
            "transfer_events": st.transfer_events,
            "pct_total": elapsed / perf_result.cycles * 100.0,
        })

    return {
        "total_cycles": perf_result.cycles,
        "instruction_count": perf_result.instruction_count,
        "stages": stage_table,
        "unmatched_events": unmatched_events,
    }
```

---

## 8. Summary of Findings & Next Action Items

| Category | Finding | Recommendation |
| :--- | :--- | :--- |
| **Evaluator Capabilities** | `codesign/challenge/pipeline.py` natively supports `trace: dict[str, dict[str, int]]` callback. | Use directly from Python. Zero changes to `codesign/` required. |
| **Current Baseline** | `cand-098/reports/grade.json` provides only coarse 128-bin histograms ($\Delta t \approx 66.5$k cycles) and single SM data. | Insufficient for stage profiling; trace callback is mandatory. |
| **Stage Attribution** | Event names can encode stage metadata: `<phase>.<batch>.<layer>.<stage>.<op>.<sm>.<wg>.<id>`. | Adopt structured event names in `project/compiler.py` when emitting multi-SM P1. |
| **Attribution Caveats** | Stage spans overlap; multi-SM stragglers inflate makespan; DMA runs asynchronously; shared NoC serializes concurrent loads. | Measure global makespan $[t_{\text{start}}, t_{\text{end}}]$, per-SM active times, and stage overlap matrices. |
| **Safety & Integrity** | ISA requires globally unique event names. Duplicate names overwrite trace records silently. | Ensure loop index interpolation and workgroup ID are included in all event names. |

### Concrete Next Steps
1. **Compiler Event Tagging**: When authoring the multi-SM column-split/head-split P1 compiler in `project/compiler.py`, format all `args["event"]` strings using the structured taxonomy.
2. **Implement Profiler Script**: Place `project/search/p1_profiler.py` using the exact design above to enable single-command attribution profiling (`python -m project.search.p1_profiler`) on candidate programs.
3. **Target High-ROI Stages**: Use the profiler output to pinpoint the largest elapsed-cycle stages in multi-SM P1 (expected: QKV and FFN W1 GEMMs) and guide TC tile selection, NoC scheduling, and barrier placement.
