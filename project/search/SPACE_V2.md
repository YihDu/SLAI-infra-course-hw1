# Search space v2: program structure before local tuning

Status: design proposal. The current v1 search and its candidate records remain
valid. Do not start v2 evaluation until the in-flight v1 round finishes.

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

The first operator is deliberately small: P1 batch parallelism can be isolated
without changing attention math or the D1 program. The next two make D1's
sequential steps internally parallel. Expose each operator only after its
compiler implementation passes `check`; a DSL placeholder is not searchable.

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

First finish the current v1 round and restore its best candidate to the
worktree. Freeze a v1 reference snapshot. Implement recipe migration and
`p1_batch_partition`, then run one candidate through generate, check, estimate,
and grade using the existing harness. Log both passes and failures. Add D1
column/head partitioning only after the P1 topology is correct. Keep v1
candidates and reports unchanged, so provenance and score comparisons survive
the compiler transition.
