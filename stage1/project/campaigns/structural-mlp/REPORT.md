# Structural bottleneck experiments, 2026-09-29

## Measured diagnosis

cand-106 stage attribution: MLP W1/GELU/W2 occupies 61.05% of P1, QKV 18.37%, Attention 7.98%. Eliminating all remaining Attention would at most yield ~1.087x P1. Score scales as sqrt(P1 speedup × D1 speedup); 10x current score requires a 100x product, which is not supported by the measured opportunities.

Single-SM RF traffic and dependency stalls persist after Attention blocking. cand-106 TC busy 452400 cycles versus RF read2672360/write2456296 and P1 total7292285. These resource demands overlap; they must not be summed as independent time fractions.

## RF resident MLP

Implemented opt-in `fusion.p1_mlp=rf_stream` for M1 prompt only. A 32-token block keeps all 256 W2 output channels in RF4..11; RF2 holds a 32-neuron W1 activation through bias/GELU and into W2; RF3 is GELU scratch; RF0/1 stream operands; RF12 holds ones for the bias outer product. W1 and W2 K chunks retain ascending numerical order. D1 unchanged, hardware unchanged. No evaluator modifications.

- cand-108 hit frozen scheduler assertion `DMA slot changed after issue preview` at expanded PC8777, the eighth final ST (RF11 → ff2). Diagnostic reads the exception traceback locals without modifying evaluator behavior. See `cand-108-dma-diagnostic.txt`.
- cand-110 adds a completion WAIT after each of the eight final stores. Complete frozen grade passes: P1=7049767, D1=718132, score=2823.456520250167. Peak process memory1379.38MiB; elapsed120.51s.
- Versus cand-106: P1 cycles −3.326%, score +1.7055%, instructions99839→89579, HBM read47165440→45626624, write8120320→4974592.
- RF read busy2672360→2637800 and write2456296→2407936 change little; TC452400→455472. Fusion removes intermediate traffic but does not remove the dominant matrix RF demand, and block-local execution repeats weight loads. This is evidence against expecting a large score jump from materialization elimination alone.

## D1 multi-SM

Implemented fixed projection-column and attention-head WG owners across all eight STEP.COMMITs. LN/residual/GELU remain on primary WG; explicit coarse barriers protect shared HBM boundaries. WG persistence does not imply full weight residency.

- cand-109 divides all GEMMs into32-column panels on4SM; 24848 events versus prior15056 computation/transfer events. Preflight lower bound0.560GiB does not include dependency unions. Complete grade attempt hits 3094.72MiB process limit at15.57s, before final report. No performance or correctness verdict.
- cand-111 preserves64-column panels and uses3SM for QKV,2SM for WO/W2,4SM for W1/heads. An older in-flight registry write removed its registration after candidate creation; immutable proposal/inputs survived. Registration was restored, but partial preparation results are retained and no grade was run. Future registry mutations are serialized.
- cand-112 is a fresh immutable retry of cand-111's program, bounded at6144MiB on the same16GiB Mac. Complete frozen grade passes: P1=7292285, D1=278743, score=4455.914242343791; peak process memory5393.30MiB, elapsed124.75s. Both functional checks and all area/power gates pass. D1 speedup2.5763x versus cand-106; score1.6051x. Four SMs have actual TC/DMA/issue work. D1 peak rolling power6.23695W. This establishes full-model multi-SM success for D1, not for P1.

## Research workers

Two bounded read-only Gemini reviews were requested. MLP worker returned an incomplete response; D1 worker hit reconnect failures and was terminated. Neither is used as performance or correctness evidence. No separate worker trace exports were added, following the user's primary-conversation-only rule.

## Final composition

cand-113 MILP D1 plan improves D1 to212659 cycles. cand-114 combines it with cand-110 P1 RF MLP: P1=7049767 D1=212659 score=5188.494494259971, full grade eligible. See `project/campaigns/milp-d1/README.md` for model and controlled comparison. cand-114 is the default kept candidate.
