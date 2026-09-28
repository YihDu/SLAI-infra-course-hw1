# Search space v2: program structure before local tuning

Status: implementation in progress. The v1 candidate records remain valid.
`stage_gemm_tiles` is implemented and has a passing candidate; the other
structural operators below remain proposals.

## Why change the space

The current compiler emits one workgroup on SM 0 and uses HBM scratch between
most layer stages. The search changes one global GEMM tile/order, attention
score residency, bias/GELU fusion, or one hardware field. It cannot express
the largest remaining opportunities: independent batch work, parallel heads
and output columns, and producer-consumer data residency. More numeric tile
samples in the same space have diminishing value.

## Design point

Keep the immutable candidate triple `(hardware, compiler, recipe)`. Extend the
recipe with **case- and stage-specific** choices rather than one global GEMM
setting. Every field must change emitted assembly or be canonicalized away.

```json
{
  "schedule": {
    "p1_prompt": {"partition": "batch", "sm_map": "distinct"},
    "p1_step": {"partition": "batch", "sm_map": "distinct"},
    "d1": {"partition": "serial"}
  },
  "gemm": {
    "p1_prompt": {"order": "nkm", "m_tile": 16, "n_tile": 16,
                  "k_chunk": 48, "prefetch": 0},
    "p1_step": {"order": "nkm", "m_tile": 1, "n_tile": 16,
                "k_chunk": 48, "prefetch": 0},
    "d1": {"order": "nkm", "m_tile": 1, "n_tile": 16,
           "k_chunk": 48, "prefetch": 0}
  },
  "attention": {"p1_prompt": "rf_scores", "p1_step": "rf_scores",
                "d1": "rf_scores"},
  "fusion": {"ffn_epilogue": "bias_gelu", "residual": "hbm"}
}
```

The example is a schema illustration, not an evaluated candidate. Version the
recipe and add a migration that maps every v1 recipe to byte-identical output.

## Structural operators, in implementation order

| Operator | Choices to implement first | Dependency and legality |
| --- | --- | --- |
| `p1_batch_partition` | serial; two workgroups on distinct SMs | Prompt batches have disjoint output/KV/scratch; both must finish before the single `STEP.COMMIT`. Generate unique workgroup and event names. |
| `gemm_column_partition` | serial; 2 or 4 column shards | Shards write disjoint output columns. Add a barrier before any consumer reads the full tensor. Works for both cases, including D1's narrow M dimension. |
| `attention_head_partition` | serial; one workgroup per head | Each head owns disjoint QKV/KV/context slices. Barrier before `wo` projection. Preserve causal length and step release rules. |
| `stage_gemm_policy` | one global policy; separate prompt/step/decode policies | Validate each tile against its own stage's dimensions and RF capacity. |
| `activation_residency` | HBM scratch; RF producer-consumer; shared-memory staging | Only fuse adjacent stages when live ranges fit in a 64 KiB workgroup RF slot or configured shared memory; wait for stores before reusing source storage. |
| `attention_algorithm` | current RF scores; blocked/online softmax | Preserve FP32 accuracy and causal mask. Start with D1's 136-element key length and compare public checks before enabling P1. |

The first P1 batch-parallel attempt (cand-068) emitted two valid workgroup
streams, but `check` did not finish after more than 11 minutes and its HBM race
validation reached roughly 48 GiB of physical memory. It was interrupted and
has no correctness or score verdict. Multi-workgroup operators therefore need
a smaller scoped prototype and a practical validation-cost gate before they
enter automated search. Expose each operator only after its compiler
implementation passes `check`; a DSL placeholder is not searchable.

The stage-specific GEMM operator is now implemented in the compiler and DSL.
The compiler emits byte-identical output for v1 recipes. Cand-070 passed
`check` with P1 prompt `m_tile=16` on `dma_depth=1` hardware and improved its
parent score from 1861.33 to 1875.68. Cand-069 encountered an evaluator DMA
scheduler assertion on `dma_depth=2`, so that combination has no score.
Cand-071 passed `check` and `grade` with D1-only `n_tile=32, k_chunk=32`:
P1=12,522,857, D1=808,550, score=1996.48, area=6.894 mm². Cand-072 then
changed only the D1 `k_chunk` from 32 to 24 and passed the same full chain:
P1=12,522,857, D1=801,958, score=2004.67. It is the current incumbent.

## Hardware is conditional on the program

Search hardware only in combinations where a program can use it. `sm_count`
and multicast are canonical no-ops for a single workgroup; shared memory size
is a cost-only choice until a shared-memory operator emits SH accesses. Add
`tc_k_parallel`, `tc_array`, DMA, cache, RF ports, and NoC choices as bounded
neighborhoods around structurally different programs. Preserve one hardware
configuration for both cases and enforce area/power gates on the combined
candidate.

## Static pruning and search policy

1. Construct a typed operator graph. Nodes carry their preconditions (workgroup
   count, scratch disjointness, barrier placement, RF/SH live bytes, tile
   divisibility, hardware capabilities). Reject invalid compositions before
   generating assembly.
2. Canonicalize equivalent points and fingerprint `(compiler version, recipe,
   hardware)`. A fingerprint based only on recipe and hardware can collide
   after a compiler change.
3. Search in two levels: first compare a few structural templates with safe
   default tiles; then tune tiles and hardware around structurally successful
   templates. Do not spend most grade runs on adjacent numeric tile values.
4. Use `estimate` only as a cheap rejection signal. `grade` remains the
   correctness and eligibility oracle; any evaluated change gets an immutable
   candidate directory and an iteration-log entry.
5. Maintain a Pareto front on P1 and D1 cycles plus a diversity archive keyed
   by actual program structure, not just the last mutation's operator name.
   Allow a candidate that helps one case to survive when it opens a useful
   composition, while the final score still uses both cases.
6. Store per-case cycle/resource deltas and emitted instruction counts with
   each candidate. Propose the next structural mutation from measured stalls,
   not from an unconditional random hardware menu.

## Rollout and acceptance

The v1 reference is cand-065. Next, search a small legal grid around cand-072's
D1 tile while preserving the stage-specific policy. Then prototype one small
producer-consumer HBM elimination in a single workgroup. Revisit D1 head or
column workgroups only after a small multi-workgroup program completes the
frozen `check` within practical time and memory. Keep all earlier candidate
snapshots and reports unchanged.
