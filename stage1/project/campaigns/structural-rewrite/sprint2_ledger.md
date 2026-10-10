# Sprint 2 hypothesis ledger (high-leverage structural search)

Frozen root: cand-154 (score 39,572.063; P1=526,494 / D1=48,952)
P1 critpath (frozen 526,494): MMA.ACC 278k (RF-read floor, tc-fixed),
LD:HBM→RF 125k, ST 59k, VEC 47k, SFU 12k. Stage split: p_w1 130k,
p_w2 84k, sm_ws 77k, p_qkv 73k, p_a 45k.
D1 critpath (48,952): LD:HBM→RF 15.6k, MMA 12.9k, HBM→SH 5.8k, VEC 5.5k.
Profile: cold step0 = 20.1k (2 layers × ~10k serial first-use streams);
steady = 4.07k/step (qkv 1.04 + a 1.11 + w1 1.08 + w2 0.6 + wo 0.24).

## Branch A — power salvage (≤15%)

Peak window located at cycle 28,677–29,677 = **cold start**, not prefetch:
hbm_write 21.3µJ (ln1 result stores) + hbm_read 17.1µJ (weight loads).
All hbm8 variants sit at 20.4–20.5W regardless of program shape.

Attempts (max 3):
1. privsh256_hbm4: dev 499,209 / 11.16W → frozen check running.
2. privsh256e_noc256: dev 497,132 / 18.93W → frozen check running.
3. reserved.

## Branch B — P1 structural hypotheses (≥100k budget gate)

| # | Hypothesis | Budget | Verdict |
|---|-----------|-------:|---------|
| B1 wstage: per-stage weight SH residency, prefetch in previous stage's drain (single 32k-word region, SH→RF B redirect) | ~60-100k | FALSIFIED: 660,249 (+24%) isolated; 619,645 composed — the extra HBM→SH→RF hop adds serial links; staged reads slower than cache-hit HBM→RF |
| B2 merged_batch retest | ~120k | generation blocked: paired_n kernel rejects current W1 dims — family needs new primitives, deferred |
| B16 a_resident for w1/qkv/wo (A panel RF-resident, kills 4x n-pass re-reads) | ~20-40k | TESTING |
| B3 co-resident sibling tapes (team16) | overlap | FALSIFIED +19.8% (barrier-coupled, shared calendars) |
| B4 k-split GEMM + reduce merge | ~40-60k | under gate standalone; revisit inside wstage composition |
| B5 qkv→attention RF handoff | ~30-50k | under gate |
| B6 ln1-recompute per owner (recompute-vs-move) | ~20-50k + kills cold-start ST burst | under gate standalone; candidate composition item |
| B7 persistent row-owner whole layer | dead: weight replication ×8 owners | PRUNED analytically |
| B8 weight-stationary remap | dead: same RF reads | PRUNED analytically |
| B9 fine-grain cross-stage tile events | wavefront falsified; coarse = B1 | folded into B1 |
| B10 tc_array 8x16k1 / 8x8k4 | RF floor −46% | FALSIFIED −2% / +13.6% (chain re-composes, support cuts starve LDs) |
| B11 vector_lanes 32 | VEC 47k | dead: halves words/lane, breaks RF views |
| B12 global team sync / stagger / stepwarm | — | FALSIFIED +65% / +4.3% / +9.4% |
| B13 act_sh query-dedup | — | FALSIFIED −0.8% |
| B14 cache0 | — | FALSIFIED +24.6% (cache essential) |
| B15 whole-layer weight residency | needs 81,920w > 65,536w cap for owners<8 | PRUNED capacity; per-stage B1 is the feasible form |

## Branch C — D1 hypotheses (≥20% gate)

| # | Hypothesis | Budget | Verdict |
|---|-----------|-------:|---------|
| C1 upfront preload (layer-1 team streams during layer-0 compute) | −5-8k | FALSIFIED 60,758 (+24%): t=0 stream serializes ~18k; steady regressed to ~5.0k/step |
| C2 barrier-free steady pipeline | ~1-2k | risky rewrite; deferred |
| C3 hbm8 cold-start halving | −6k cold | FALSIFIED: D1@hbm8 = 48,426 (−1%); cold is per-WG stream-serialized, not channel-bound |
| C4 attention parts>4 | — | already at cap (parts=4/head) |

## Standing conclusions

- Per-SM rf_read calendar ~276-282k is the hard P1 floor at tc=4x8/kp4;
  MMA operands are RF-only; floor only moves with tc_array (area-blocked).
- Serial per-WG tape re-composes around any single-resource cut
  (tc upgrade, dd, cache) — the slack is ~250k of hideable ordering slack.
- Cold-start power peak (~29k window) is an HBM read+write burst;
  hbm8 variants sit 20.4-20.5W; throttle/fallback variants are the legalizers.

## FINAL — sprint closed 2026-10-01 (submission mode)

**Winner: cand-155 = privsh256_hbm4** — frozen-graded joint score **40,901.125**
(P1=493,984 / D1=48,838, peak 11.12W/10.79W, area 23.373mm2, eligible=True).
Promoted to root; iteration-log + registry updated.

Final falsification tally: 6/6 Branch-B P1 microbenchmarks falsified
(wstage 660k, wstage_combo 620k, ares 597k, merged_batch gen-blocked,
team16 639k earlier, plus stage family); 2/2 Branch-C D1 falsified
(upfront 60.8k, nogroups 52.8k); Branch-A salvage won via hbm4 downgrade
(noc256 = 485k but 20.09W ineligible; all hbm8 variants ≥20.09W ineligible).

Residual structural gap: P1 = 493,984 vs ~350k optimistic floor
(278k rf_read-capped MMA + ~70k unavoidable serial residue). Closing it needs
decoupled producer-consumer issue semantics not expressible in current ISA.
