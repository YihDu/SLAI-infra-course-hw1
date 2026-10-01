"""Promote the graded paired_late_b joint triple (score 36115.7025) to root.

Manual promotion per user instruction 2026-09-30 (structural-campaign Task 0):
the current-round finalizer was left in "waiting" state and the two best joint
grades were never imported. This script:

1. asserts the root compiler regenerates the joint programs byte-identically
   from the joint recipe (the joint compiler.py already equals the root one);
2. stages recipe/hardware/programs into the worktree root;
3. registers the triple as a new kept candidate under the coordinator writer
   lock via records.record_result (head promotion, archives, search best,
   iteration-log entry all handled there);
4. copies the frozen grade evidence into the candidate reports dir.

It never touches codesign/, baseline_manifest.json, or any prior candidate.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from project import compiler
from project.search import state, records
from project.search.candidate import Candidate, EXPERIMENTS
from project.search.coordinator import writer_lock
from project.search import engine

JOINT = REPO / "project/campaigns/structural/p1-pair-r1/paired_late_b/joint"
GRADE = json.loads((JOINT / "grade.json").read_text())
SCORE = float(GRADE["experimental_score"])
CYCLES = {
    case: int(info["cycles"])
    for case, info in GRADE["gate_diagnostics"]["scenarios"].items()
}
AREA = float(GRADE["gate_diagnostics"]["area_mm2"])
POWER = max(
    float(info["peak_power_w"])
    for info in GRADE["gate_diagnostics"]["scenarios"].values()
)
assert GRADE["eligible"] is True and GRADE["status"] == "candidate"
assert all(c["functional_passed"] for c in GRADE["cases"].values())

recipe = json.loads((JOINT / "recipe.json").read_text())
p1 = compiler.generate_m1_p1(recipe)[0]
d1 = compiler.generate_m1_d1(recipe=recipe)[0]
assert p1 == (JOINT / "programs/M1_P1.asm").read_text(), "P1 regeneration mismatch"
assert d1 == (JOINT / "programs/M2_D1.asm").read_text(), "D1 regeneration mismatch"
assert (JOINT / "compiler.py").read_text() == (REPO / "project/compiler.py").read_text()

# Stage root triple (hardware.json keeps the repo's indent=1 convention).
(REPO / "project/recipe.json").write_text((JOINT / "recipe.json").read_text())
hw = json.loads((JOINT / "hardware.json").read_text())
(REPO / "hardware.json").write_text(json.dumps(hw, indent=1, sort_keys=True) + "\n")
(REPO / "programs/M1_P1.asm").write_text(p1)
(REPO / "programs/M2_D1.asm").write_text(d1)

with writer_lock():
    reg = state.load()
    cid = state.next_id(reg)
    assert not (EXPERIMENTS / cid).exists(), cid
    cand = Candidate(
        cid, "cand-150",
        "Promote frozen-graded paired_late_b joint triple to root: P1 paired W1 "
        "kernel (paired_n=[w1], paired_b_prefetch=0) + step a_resident on "
        "NoC512/16SM/1TC hardware; D1 layer-RF owners. Byte-identical artifacts.",
        "Manual promotion of completed joint grade; sources in proposal.json",
        kind="manual", failure_class="none",
    )
    cand.save()
    (cand.inputs / "compiler.py").write_text((REPO / "project/compiler.py").read_text())
    (cand.inputs / "recipe.json").write_text((JOINT / "recipe.json").read_text())
    (cand.inputs / "hardware.json").write_text((REPO / "hardware.json").read_text())
    (cand.programs / "M1_P1.asm").write_text(p1)
    (cand.programs / "M2_D1.asm").write_text(d1)
    (cand.reports / "grade.json").write_text((JOINT / "grade.json").read_text())
    for extra in ("grade-run.json", "manifest.json"):
        src = JOINT / extra
        if src.exists():
            (cand.reports / extra).write_text(src.read_text())
    (cand.dir / "proposal.json").write_text(json.dumps({
        "sources": {"joint": str(JOINT)},
        "program_regeneration_verified": True,
        "promotion": "manual per user instruction 2026-09-30; finalizer left waiting",
    }, indent=2) + "\n")
    metrics = {
        "candidate": cid,
        "reports": {"grade": GRADE},
        "score": SCORE,
        "cycles": CYCLES,
        "functional_passed": True,
        "eligible": True,
        "area_mm2": AREA,
        "power_w": POWER,
        "evaluation_attempts": 1,
    }
    (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    records.record_result(reg, cand, metrics)
    state.save(reg)
    engine.write_search_state(reg)
print(cid, "score", SCORE, "cycles", CYCLES)
