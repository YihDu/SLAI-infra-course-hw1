# Structural search results — 2026-09-28

Best: cand-095, score 2567.954052, +14.383% against cand-083.

11 full-grade launches: 10 eligible passes; cand-089 stopped for measured checker cost, not classified as a correctness failure. All grades used seed 7.

| Candidate | P1 cycles | D1 cycles | Score | Outcome |
| --- | ---: | ---: | ---: | --- |
| cand-087 | 9484805 | 744864 | 2390.1102253707836 | kept |
| cand-088 | 9986555 | 744864 | 2329.2938717469783 | evaluated |
| cand-089 | — | — | — | not_verified |
| cand-090 | 10354943 | 744864 | 2287.4851157230846 | evaluated |
| cand-091 | 10750199 | 744864 | 2245.0389561884695 | evaluated |
| cand-092 | 10127399 | 738688 | 2322.689428598885 | evaluated |
| cand-093 | 8656949 | 738688 | 2512.2200312594855 | kept |
| cand-094 | 8656949 | 718132 | 2547.9215759712715 | kept |
| cand-095 | 8522411 | 718132 | 2567.954051821853 | kept |
| cand-096 | 9957359 | 718132 | 2375.7262879581485 | evaluated |
| cand-097 | 9043631 | 725840 | 2479.583625689317 | evaluated |

Best area: 16.25838297088 mm²; maximum case rolling power: 2.971060854310833 W.

The winning prompt layout is nkm_lanes, M32/N32/K32; decode uses N64/K16. Hardware uses four SMs and 8×16 TC arrays. Root hardware, recipe and generated programs are restored to this candidate.

The K8 trial evidence is in `project/experiments/cand-089/reports/cost-prune.json`; sampler output is adjacent. The frozen evaluator was not changed.

Space audit: 840 P1 combinations → 218 resource-legal → 168 preferred after checker-cost filtering on cand-083 hardware. N128 remains available through joint vector-partition/hardware proposals.

Next strategy: `project/search/STRATEGY_NEXT.md`. Source diagnosis: `project/search/DIAGNOSIS_V4.md`.

Verification: 45 search tests passed; default-off generation remained identical before campaign; all successful candidates passed full grade. Gemini worker returned no usable conclusion; its trace is retained.
