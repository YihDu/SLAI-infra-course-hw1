# Research Report: Multi-Workgroup Execution, Race Validation, and Multi-SM Scaling

**Date**: 2026-09-28
**Scope**: Frozen evaluator inspection (`codesign/challenge/`), compiler audit (`project/compiler.py`), multi-workgroup independent GEMM column-split prototype design, empirical scaling analysis, and local validation API.
**Target File**: `project/search/review_multism.md`

---

## 1. Executive Summary

This independent research evaluates the feasibility, mechanics, and scaling characteristics of executing multi-workgroup (Multi-WG) programs across multiple Streaming Multiprocessors (SMs) within the frozen `codesign` simulator.

### Core Discoveries:
1. **Simulator Multi-SM Native Support**: The frozen timing simulator (`codesign/challenge/pipeline_events.py`) and functional executor (`codesign/challenge/micro.py`) fully support concurrent, out-of-order instruction scheduling of independent workgroup tapes across different SMs (`sm=0..sm_count-1`).
2. **Race Validation Semantics (`codesign/challenge/hbm_race.py`)**:
   - Single-workgroup programs bypass the race validator entirely (`begins <= 1`).
   - For multi-workgroup programs, a global topological DAG of instructions is verified across all tapes using interval treaps (`_Node`).
   - **Read-after-Read (RAR) is non-conflicting**: Multiple workgroups can concurrently read the identical input matrix (e.g. activation matrix A) from HBM without emitting synchronization barriers or causing race errors.
   - **Disjoint Column Slicing**: When workgroups write disjoint column tiles of the output matrix C using 2D strided views (`shape=[M, tile_n]`, `strides=[N, 1]`), the interval overlap function (`_overlap()`) computes `False`. No RAW, WAR, or WAW hazard is triggered.
3. **Synchronization Contract (`codesign/challenge/isa.py`)**:
   - When more than one workgroup is active, legacy single-target `WAIT {"events": [...]}` and `BARRIER {"events": [...]}` throw `ValueError("Synchronization target is ambiguous")`.
   - Workgroups must explicitly specify their identity:
     - `WAIT {"wg": "<name>", "events": [...]}` (private to that workgroup tape).
     - `BARRIER {"wgs": ["<name1>", "<name2>", ...], "events": [...]}` (synchronizes specified active groups).
     - `STEP.COMMIT {"step": s}` is chip-wide: all active workgroups must participate and converge at the commit point before new inputs release.
4. **Empirical Performance**:
   - A tiny 2-WG column-split prototype ($M=8, K=48, N=32$) running on SM 0 and SM 1 achieved **235 cycles vs. 399 cycles** for 1 WG on SM 0 (**1.70x speedup**, zero race errors, max error $5.72 \times 10^{-6}$).
   - On an M1 projection-scale GEMM ($M=64, K=256, N=256$):
     - **1 SM**: 81,325 cycles (1.00x, 100% efficiency, 0.84W avg, 1.70W peak).
     - **2 SMs**: 43,460 cycles (1.87x, 93.6% efficiency, 1.26W avg, 2.51W peak).
     - **4 SMs**: 24,809 cycles (3.28x, 82.0% efficiency, 1.94W avg, 3.59W peak).
     - **8 SMs**: 14,891 cycles (5.46x, 68.3% efficiency, 3.60W avg, 5.82W peak).

---

## 2. In-Depth Inspection of Frozen Race Validation (`codesign/challenge/hbm_race.py`)

### 2.1 Bypass Condition vs. Multi-WG Trigger
In `codesign/challenge/hbm_race.py` (lines 351–369):
```python
if begins <= 1:
    return
```
When only 1 workgroup is defined (`WG.BEGIN` count <= 1), `validate_hbm_races` exits immediately. When >= 2 workgroups begin, the validator constructs a global instruction DAG and executes a topological traversal to detect memory hazards.

### 2.2 Instruction Dependency Graph Construction (`_graph`)
Lines 226–274 of `hbm_race.py`:
- **Intra-Tape Edges**: Each workgroup commands are executed in strict sequential order:
  `for before, after in zip(commands, commands[1:]): edge(before[0], after[0])`
- **Predecessor Edges**: When a workgroup name is reused after `WG.END`, an edge is inserted from the predecessors last instruction to the successors `WG.BEGIN`.
- **Commit Epoch Serialization**: `STEP.COMMIT` introduces edges from all preceding instructions to the commit, and from the commit to all subsequent instructions across all workgroups.
- **Synchronization Edges**: `WAIT` and `BARRIER` add edges from `event_pc[name]` to the synchronization instructions PC.

### 2.3 Knowledge Propagation Model
Lines 18–44 and 370–415:
- Each workgroup tape maintains a `_Knowledge` struct tracking the prefix of completed operations and events from other tapes:
  `_Knowledge(prefixes: dict[int, int], events: frozenset[tuple[int, int]])`
- When tape A completes an event and tape B waits on it via `WAIT` or `BARRIER`, tape B knowledge is updated via `_join()`.
- Within the same workgroup tape, conflicting transfers automatically serialize against previous events:
  `knowledge[tape] = _join(knowledge[tape], event_knowledge[program[old.pc].args["event"]])`

### 2.4 Treap Interval Hazard Detection & Non-Overlapping Slices
Lines 55–110 and 420–465:
- HBM accesses are stored in dual interval treaps: `roots[0]` (reads) and `roots[1]` (writes).
- **Read Operations (`LD` from HBM)**: Checked only against `roots[1]` (writes). If an overlapping write exists and is not contained in `knowledge[tape]`, a **RAW (HBM WR race)** is raised. Concurrent reads from different workgroups do not conflict.
- **Write Operations (`ST` to HBM)**: Checked against both `roots[1]` (WAW) and `roots[0]` (WAR). If an overlapping prior read or write exists without explicit causal ordering, a race is raised.
- **2D Strided Overlap Logic (`_overlap`)**:
  `_overlap(left, right)` computes 1D bounding box rejection (`lo >= other_hi or other_lo >= hi`). If bounding boxes overlap, it evaluates actual multi-dimensional cell positions:
  `_positions(desc) = offset + row * stride[0] + col * stride[1]`
  For column-split matrix stores:
  - WG 0 writes columns 0..N/2-1 with stride [N, 1].
  - WG 1 writes columns N/2..N-1 with stride [N, 1].
  Even though their bounding intervals overlap ([offset, offset + M * N]), their point sets are strictly disjoint. `_overlap()` returns `False`. **Column splitting is completely hazard-free without inter-workgroup barriers.**

### 2.5 Synchronization Semantics (`codesign/challenge/isa.py`)
In `isa.py` (lines 80–120):
```python
def sync_targets(op: str, args: dict, active_groups: Collection[str]) -> tuple[str, ...]:
    active = set(active_groups)
    if set(args) == {"events"}:
        if len(active) != 1:
            raise ValueError("Synchronization target is ambiguous")
        return (next(iter(active)),)
    if op == "WAIT" and set(args) == {"wg", "events"}:
        targets = (args["wg"],)
    elif op == "BARRIER" and set(args) == {"wgs", "events"}:
        targets = tuple(args["wgs"])
```
- Emitting `WAIT {"events": [...]}` when >= 2 workgroups are active fails immediately.
- Compiler must emit `WAIT {"wg": wg_name, "events": [...]}`.

---

## 3. Inspection of `project/compiler.py`

### 3.1 Single-Workgroup Hardcoding
- `Builder.__init__` (line 109): `self.emit("WG.BEGIN", wg="g", sm=0, shared_bytes=shared_bytes)`
- `rf()` (line 76): `{"space": "RF", "offset": offset, "count": count, "wg": "g", "lane": lane}`
- `sh()` (line 80): `{"space": "SH", "offset": offset, "count": count, "wg": "g", "lane": 0}`
- `Builder.finish` (line 173): `self.emit("WG.END", wg="g")`
All instructions are emitted into a single tape on SM 0.

### 3.2 Scratchpad Allocation Mechanism
- `Builder.alloc()` (lines 128–135) linearly increments `self.cursor` starting from `layout.symbols["scratch"].address // 4`.
- In a naive multi-SM port, if workgroups share the same intermediate symbol addresses for writing scratch intermediates (e.g. `ln1`, `qkv`, `ff1`), `hbm_race.py` will trigger **WAW** or **WAR** collisions.
- **Resolution**: Scratch tensors must either be partitioned per workgroup (e.g. `offset + wg_id * slice_size`) or kept resident in workgroup-private RF registers.

### 3.3 Shared Memory Scope Isolation
- In `codesign/challenge/micro.py` (lines 62–70, 169–177):
  `self.groups[name] = Group(sm=sm, shared_quota=quota, shared=np.zeros(quota // 4, np.float32), ...)`
- In `codesign/challenge/pipeline.py` (line 93):
  `if operand["wg"] != group.name: raise ValueError("Cross-workgroup timed access")`
- **Critical Architectural Invariant**: Shared memory is **strictly workgroup-private**. Two workgroups residing on the same SM **cannot** read or write each others shared memory. All inter-workgroup data exchange must go through HBM.

---

## 4. Two-Workgroup Independent Column-Split GEMM Prototype

### 4.1 Specification
- Problem: Compute C = A x B, where A in R^(8 x 48), B in R^(48 x 32), and C in R^(8 x 32).
- Partitioning:
  - Workgroup `wg0` on SM 0 computes columns 0..15 of C.
  - Workgroup `wg1` on SM 1 computes columns 16..31 of C.
- Memory Layout (HBM FP32 offsets):
  - A: offset 0, shape [8, 48], count 384
  - B: offset 384, shape [48, 32], count 1536
  - C: offset 1920, shape [8, 32], count 256
    - `wg0` output: offset 1920, shape [8, 16], strides [32, 1]
    - `wg1` output: offset 1936, shape [8, 16], strides [32, 1]

### 4.2 Executed Assembly Program
```json
WG.BEGIN {"wg": "wg0", "sm": 0, "shared_bytes": 0}
WG.BEGIN {"wg": "wg1", "sm": 1, "shared_bytes": 0}
VEC {"kind": "add", "src": [{"imm": 0.0}, {"imm": 0.0}], "dst": {"space": "RF", "offset": 0, "count": 128, "wg": "wg0", "lane": 2}, "event": "e0"}
VEC {"kind": "add", "src": [{"imm": 0.0}, {"imm": 0.0}], "dst": {"space": "RF", "offset": 0, "count": 128, "wg": "wg1", "lane": 2}, "event": "e1"}
LD {"src": {"space": "HBM", "offset": 0, "count": 384, "shape": [8, 48], "strides": [48, 1], "wg": null, "lane": 0}, "dst": {"space": "RF", "offset": 0, "count": 384, "wg": "wg0", "lane": 0}, "event": "e2"}
LD {"src": {"space": "HBM", "offset": 384, "count": 768, "shape": [48, 16], "strides": [32, 1], "wg": null, "lane": 0}, "dst": {"space": "RF", "offset": 0, "count": 768, "wg": "wg0", "lane": 1}, "event": "e3"}
MMA.ACC {"a": {"space": "RF", "offset": 0, "count": 384, "wg": "wg0", "lane": 0}, "b": {"space": "RF", "offset": 0, "count": 768, "wg": "wg0", "lane": 1}, "acc": {"space": "RF", "offset": 0, "count": 128, "wg": "wg0", "lane": 2}, "m": 8, "n": 16, "k": 48, "event": "e4"}
ST {"src": {"space": "RF", "offset": 0, "count": 128, "wg": "wg0", "lane": 2}, "dst": {"space": "HBM", "offset": 1920, "count": 128, "shape": [8, 16], "strides": [32, 1], "wg": null, "lane": 0}, "event": "e5"}
LD {"src": {"space": "HBM", "offset": 0, "count": 384, "shape": [8, 48], "strides": [48, 1], "wg": null, "lane": 0}, "dst": {"space": "RF", "offset": 0, "count": 384, "wg": "wg1", "lane": 0}, "event": "e6"}
LD {"src": {"space": "HBM", "offset": 400, "count": 768, "shape": [48, 16], "strides": [32, 1], "wg": null, "lane": 0}, "dst": {"space": "RF", "offset": 0, "count": 768, "wg": "wg1", "lane": 1}, "event": "e7"}
MMA.ACC {"a": {"space": "RF", "offset": 0, "count": 384, "wg": "wg1", "lane": 0}, "b": {"space": "RF", "offset": 0, "count": 768, "wg": "wg1", "lane": 1}, "acc": {"space": "RF", "offset": 0, "count": 128, "wg": "wg1", "lane": 2}, "m": 8, "n": 16, "k": 48, "event": "e8"}
ST {"src": {"space": "RF", "offset": 0, "count": 128, "wg": "wg1", "lane": 2}, "dst": {"space": "HBM", "offset": 1936, "count": 128, "shape": [8, 16], "strides": [32, 1], "wg": null, "lane": 0}, "event": "e9"}
WG.END {"wg": "wg0"}
WG.END {"wg": "wg1"}
```

### 4.3 Validation Results
- **Race Validation**: `validate_hbm_races()` passed with zero diagnostics.
- **Functional Correctness**: Executed on `MicroMachine(hw, hbm)`; compared against NumPy reference `A @ B`:
  - max absolute error = 5.722046e-06 (well below the 1e-3 threshold).
- **Execution Timing**:
  - **1 WG on SM 0**: 399 cycles.
  - **2 WGs on SM 0 and SM 1**: 235 cycles (**1.70x speedup**).
- **Resource Utilization Breakdown**:
  - SM 0: issue: 7 cycles, rf_read: 44, vec: 10, rf_write: 88, dma: 80, tc: 16, noc_in: 55, noc_out: 8
  - SM 1: issue: 7 cycles, rf_read: 44, vec: 10, rf_write: 88, dma: 80, tc: 16, noc_in: 55, noc_out: 8

---

## 5. Multi-SM Scaling Experiment

To characterize scaling behavior and hardware resource saturation, a scaling experiment was conducted on realistic M1 projection GEMM dimensions ($M=64, K=256, N=256$) with tiles $m=8, k=48, n=16$.

### 5.1 Experimental Scaling Table
Tested against `hardware.json` (with `sm_count: 4`) and scaled to 8 SMs:

| Active SMs | Total Cycles | Speedup | Scaling Efficiency | Average Power | Peak Power (1000 cyc) | HBM Read Traffic |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 SM** | 81,325 | 1.00x | 100.0% | 0.84 W | 1.70 W | 0.31 MB |
| **2 SMs** | 43,460 | 1.87x | 93.6% | 1.26 W | 2.51 W | 0.34 MB |
| **4 SMs** | 24,809 | 3.28x | 82.0% | 1.94 W | 3.59 W | 0.35 MB |
| **8 SMs** | 14,891 | 5.46x | 68.3% | 3.60 W | 5.82 W | 0.38 MB |

### 5.2 Scaling Bottleneck Analysis
1. **Linear Compute Scaling**: Tensor Core execution and RF accesses scale near-linearly across SMs because each SM operates its own independent RF file and TC array.
2. **Input Matrix A Duplication**: Because weight/activation multicast is disabled (`"multicast": false`) and each SM independently reads matrix A, HBM read traffic increases from 0.31 MB to 0.38 MB as SM count scales. With L2 cache (`cache_mib: 2`), repeated lines achieve cache hits, cushioning HBM channel saturation.
3. **NoC & HBM Contention**: As SM count increases, concurrent DMA requests contentiously share the chip NoC (256 B/cycle) and HBM channels (4 channels), causing scaling efficiency to taper from 93.6% (2 SMs) to 68.3% (8 SMs).
4. **Power Envelope Headroom**: Peak power at 8 SMs is 5.82 W, which is comfortably beneath the 20.0 W evaluator limit (leaving ample thermal margin).

---

## 6. Exact Local Validation API

To validate multi-workgroup programs locally without invoking full grading runs or touching frozen files, use this Python API:

```python
import json
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.hbm_race import validate_hbm_races
from codesign.challenge.micro import MicroMachine, SparseHBM
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import make_fixture, check_case, estimate_case

# 1. Static Race Validation
validate_hbm_races(program_text)

# 2. Timing Simulation (Cycles, Power, Resource Calendars)
hw = Hardware(**json.load(open("hardware.json")))
perf_result = estimate_pipeline(hw, iter_parse(program_text), hbm_words=hbm_words)
cycles = perf_result.cycles
resource_stats = perf_result.resource_stats

# 3. Functional Simulation (Numerical Check)
hbm = SparseHBM(hbm_words)
# populate inputs into hbm._data and set hbm._valid = 1 ...
mm = MicroMachine(hw, hbm)
mm.execute(tuple(iter_parse(program_text)), hbm_races_validated=True)
# verify outputs in hbm._data against expected oracle

# 4. Full Scenario Check (P1 or D1)
report = check_case("M1_P1", seed=7, hardware=hw, program=program_text)
assert report["passed"] is True
```

---

## 7. Limitations, Architectural Constraints, and Traps

1. **SM ID Boundary**:
   - `0 <= sm < hardware.sm_count`. Assigning an SM index >= sm_count immediately raises `ValueError("Workgroup SM or shared quota exceeds hardware capacity")`.
2. **Workgroup Residency Limit**:
   - Each SM supports at most **4 resident workgroups** simultaneously:
     `len(self.resident_by_sm[tape.sm]) >= 4` blocks `_preview()` and stalls issue.
3. **Workgroup Shared Memory Quota**:
   - Cumulative `shared_bytes` across active workgroups on an SM cannot exceed `hardware.shared_kib * 1024`.
   - Shared memory is strictly private to each workgroup.
4. **Ambiguous Synchronization Traps**:
   - Multiple active workgroups require explicit `wg` or `wgs` parameters in `WAIT` and `BARRIER` opcodes. Omitting them throws an immediate exception.
5. **STEP.COMMIT Barrier Deadlock**:
   - In `pipeline_events.py`, `STEP.COMMIT` requires participation from **every active workgroup**. If one workgroup terminates (`WG.END`) before another reaches `STEP.COMMIT`, or if only one workgroup issues `STEP.COMMIT`, the simulator encounters a pipeline deadlock.
6. **HBM Output Write Overlap**:
   - Output writes must be partitioned cleanly. Overlapping writes to the same HBM index from different workgroups without barrier ordering trigger WAW race violations. Furthermore, writing the same ABI output index multiple times triggers `ValueError("Duplicate HBM output write")` in `micro.py`.

---

## 8. Concrete Integration Plan for Multi-SM Search

### Phase A: Multi-Workgroup Emitter Scaffolding
- Extend `project/compiler.py` with a `MultiWorkgroupBuilder` or a workgroup parameter in `Builder`:
  - Allow specifying workgroup name (`wg`) and target SM (`sm`).
  - Update `rf()` and `sh()` helpers to accept `wg_name`.
  - Add `sync_wait(wg, events)` and `sync_barrier(wgs, events)` helper methods.

### Phase B: Column-Split GEMM Operator
- Introduce `gemm_multism` into the compiler:
  - Partition the N-dimension columns into P slices across P workgroups on distinct SMs ($P <= \text{sm_count}$).
  - All workgroups concurrently read input A (RAR, hazard-free).
  - Each workgroup reads its assigned weight columns and writes its assigned output columns.
  - Synchronize via `BARRIER` before subsequent dependent operations (e.g. LayerNorm or Attention).

### Phase C: Search DSL Knob Addition
- Expose Multi-SM parallel knobs in `project/search/dsl.py`:
  - `"parallel_sms": [1, 2, 4]`
  - `"gemm_split": ["none", "column", "batch"]`.
- Include static screening: ensure `parallel_sms <= hardware["sm_count"]`.
