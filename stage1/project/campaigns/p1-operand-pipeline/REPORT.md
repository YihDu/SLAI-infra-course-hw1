# P1 resident MLP operand pipeline — cand-118

After automatic D1 MILP search stopped at its inner plateau, switch the P1 MLP
issue schedule. Preserve 32x32 arithmetic tiles, ascending reduction order, all
loads/stores, and W2 RF accumulator residency. Alternate W1 A/B buffers between
RF0/1 and RF13/14; alternate W2 B between RF1/14. Enqueue the next panel after
issuing the current MMA so operand overwrite dependencies do not serialize DMA
behind that MMA. RF12 remains the bias-outer-product ones vector; RF4..11 remain
W2 accumulators. Existing final-store completion WAITs are retained.

Opt-in: `fusion.p1_mlp=rf_pipeline`. The compiler still reproduces both cand-114
programs byte-for-byte when `rf_stream` is selected. D1 is unchanged from115.

Full frozen grade, seed7: both functional and eligibility gates pass.

| Metric | Parent115 | Pipeline118 |
|---|---:|---:|
| P1 cycles | 7049767 | 6201341 |
| D1 cycles | 210849 | 210849 |
| Score | 5210.71681 | 5555.74170 |
| P1 expanded instructions | 89579 | 89579 |

P1 cycles fall12.03%; score improves6.62%. This disproves saturation of the
whole current design after the D1 parameter plateau. Default incumbent is118.
P1 still uses a single SM; this is an issue-order improvement, not a P1 multi-SM
pass. No evaluator modification or official submission.

Next hypothesis: QKV/WO GEMMs still reuse their operand RF buffers serially;
apply the same reasoning to their two M-row accumulator partitions, with a
separate full candidate. Do not resume the exhausted D1 tile neighborhood merely
because a P1 change moved the global head.
