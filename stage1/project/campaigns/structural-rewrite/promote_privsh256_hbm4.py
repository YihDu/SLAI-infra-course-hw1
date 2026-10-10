"""Promote the frozen-graded privsh256_hbm4 joint triple to root.

Sources:
  hardware:  campaigns/structural-rewrite/hw_privsh256_hbm4.json
  P1 asm:    campaigns/structural-rewrite/p1_privsh256_hbm4.asm
             (recipe = root + priv_sh + step_sh[l0w1,l1w1,l2w1,l0wo]
              + shared_bytes 262144 + p1_step.b_bufs 4)
  D1 asm:    programs/M2_D1.asm (unchanged root program)
  grade:     campaigns/structural-rewrite/grade_privsh256_hbm4.json
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

CAMP = REPO / "project/campaigns/structural-rewrite"
GRADE = json.loads((CAMP / "grade_privsh256_hbm4.json").read_text())
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

recipe = json.loads((CAMP / "recipe_privsh256_hbm4.json").read_text())
p1 = compiler.generate_m1_p1(recipe)[0]
d1 = compiler.generate_m1_d1(recipe=recipe)[0]
assert p1 == (CAMP / "p1_privsh256_hbm4.asm").read_text(), "P1 regen mismatch"
assert d1 == (REPO / "programs/M2_D1.asm").read_text(), "D1 regen mismatch"

(REPO / "project/recipe.json").write_text(json.dumps(recipe, indent=1) + "\n")
hw = json.loads((CAMP / "hw_privsh256_hbm4.json").read_text())
(REPO / "hardware.json").write_text(json.dumps(hw, indent=1, sort_keys=True) + "\n")
(REPO / "programs/M1_P1.asm").write_text(p1)
(REPO / "programs/M2_D1.asm").write_text(d1)

with writer_lock():
    reg = state.load()
    cid = state.next_id(reg)
    assert not (EXPERIMENTS / cid).exists(), cid
    cand = Candidate(
        cid, "cand-154",
        "Promote frozen-graded privsh256_hbm4: P1 priv_sh res/out SH residency "
        "+ step_sh 4-slice weight prefetch + 256KiB SH on hbm4/cache4/dd2 "
        "hardware (power-legal: peak 11.1W); D1 root program unchanged.",
        "Manual promotion of completed joint grade; sources in proposal.json",
        kind="manual", failure_class="none",
    )
    cand.save()
    (cand.inputs / "compiler.py").write_text(
        (REPO / "project/compiler.py").read_text())
    (cand.inputs / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n")
    (cand.inputs / "hardware.json").write_text(
        (REPO / "hardware.json").read_text())
    (cand.programs / "M1_P1.asm").write_text(p1)
    (cand.programs / "M2_D1.asm").write_text(d1)
    (cand.reports / "grade.json").write_text(
        (CAMP / "grade_privsh256_hbm4.json").read_text())
    (cand.dir / "proposal.json").write_text(json.dumps({
        "sources": {
            "hardware": str(CAMP / "hw_privsh256_hbm4.json"),
            "p1_asm": str(CAMP / "p1_privsh256_hbm4.asm"),
            "grade": str(CAMP / "grade_privsh256_hbm4.json"),
        },
        "program_regeneration_verified": True,
        "promotion": "manual; sprint-2 salvage branch winner "
                     "(hbm8 variants power-ineligible)",
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
