# Iteration Log

Chronological record of every evaluated candidate: changes, local checks,
scores/observations, rejected ideas, and submission rationale. Entry template
is in `AGENTS.md` §7. Append newest entries at the bottom.

Baselines to beat (frozen manifest, seed 7, 500 MHz):
- M1_P1: 30,996,995 cycles
- M2_D1: 1,302,032 cycles
- Eligibility: both correct, area ≤ 24 mm², peak power ≤ 20 W, ≤ 2× baseline each

---

## 2026-09-28 — Repo setup and baseline import

- Change: none (unmodified starter). Initialized git repo, branch `dev-yh`,
  remote `origin` → github.com/YihDu/SLAI-infra-course-hw1. Added `AGENTS.md`
  briefing + this log.
- Verify: not yet run — environment pending (need Python 3.12.13 + NumPy 2.5.3;
  system Python is 3.14.3 without NumPy).
- Result: n/a
- Decision: next step is environment setup, then reproduce baseline
  check/estimate/grade before any optimization.
