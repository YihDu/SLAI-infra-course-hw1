"""Register the D1 Stage A/B candidate, run one full frozen joint grade, and
promote it to root if eligible and better than the head.

Candidate = root compiler.py + root recipe with D1 flags
(fused_ln=fold, attn_prefetch, fold_bias, chunked_a) + hardware.json with
shared_ports 1R1W -> 2R1W. P1 program must be byte-identical to root.
Never touches codesign/ or prior candidates.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from project import compiler
from project.search import engine, records, state
from project.search.candidate import EXPERIMENTS, Candidate
from project.search.coordinator import writer_lock

HERE = Path(__file__).resolve().parent
D1_FLAGS = {"fused_ln": "fold", "attn_prefetch": 1, "fold_bias": 1, "chunked_a": 1}


def main():
    recipe = json.loads((REPO / "project/recipe.json").read_text())
    recipe["d1_schedule"].update(D1_FLAGS)
    hw = json.loads((REPO / "hardware.json").read_text())
    hw["shared_ports"] = "2R1W"
    p1 = compiler.generate_m1_p1(json.loads(json.dumps(recipe)))[0]
    d1 = compiler.generate_m1_d1(recipe=json.loads(json.dumps(recipe)))[0]
    assert p1 == (REPO / "programs/M1_P1.asm").read_text(), "P1 must stay byte-identical"
    assert d1 == (HERE / "d1-stageab/M2_D1.asm").read_text(), "D1 differs from frozen-checked program"

    with writer_lock():
        reg = state.load()
        parent = reg["head"]
        cid = state.next_id(reg)
        assert not (EXPERIMENTS / cid).exists(), cid
        cand = Candidate(
            cid, parent,
            "D1 token-local LN handoff + centred LN fold (beta W built in assembly "
            "at step 0, kept in RF) + attention K/V prefetch + b1 fold + chunked A; "
            "hardware shared_ports 2R1W. P1 unchanged.",
            "structural-rewrite Task 2 Stage A/B; frozen D1 check seeds 7/19 pass, "
            "frozen D1 estimate 48952",
            kind="manual", failure_class="none", status="evaluating",
        )
        cand.save()
        (cand.inputs / "compiler.py").write_text((REPO / "project/compiler.py").read_text())
        (cand.inputs / "recipe.json").write_text(json.dumps(recipe, indent=1) + "\n")
        (cand.inputs / "hardware.json").write_text(json.dumps(hw, indent=1, sort_keys=True) + "\n")
        (cand.programs / "M1_P1.asm").write_text(p1)
        (cand.programs / "M2_D1.asm").write_text(d1)
        reg["candidates"].setdefault(cid, {}).update(
            parent=parent, hypothesis=cand.hypothesis, kind="manual", status="evaluating")
        state.save(reg)
    print(cid, "registered; launching full grade", flush=True)

    report = cand.reports / "grade.json"
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "challenge.py"), "grade",
           "--hardware", str(cand.inputs / "hardware.json"),
           "--program-p1", str(cand.programs / "M1_P1.asm"),
           "--program-d1", str(cand.programs / "M2_D1.asm"),
           "--baseline", str(REPO / "baseline_manifest.json"),
           "--seed", "7", "--report", str(report)]
    start = time.time()
    with (cand.reports / "grade.stdout").open("w") as out, \
            (cand.reports / "grade.stderr").open("w") as err:
        rc = subprocess.run(cmd, cwd=REPO, stdout=out, stderr=err).returncode
    elapsed = time.time() - start
    grade = json.loads(report.read_text()) if report.exists() else {}
    cycles = {c: v["timing"]["cycles"] for c, v in grade.get("cases", {}).items()
              if v.get("timing")}
    (cand.reports / "grade-run.json").write_text(json.dumps(
        {"command": cmd, "returncode": rc, "elapsed_seconds": elapsed,
         "score": grade.get("experimental_score"), "eligible": grade.get("eligible"),
         "cycles": cycles}, indent=1))
    print("grade rc", rc, "score", grade.get("experimental_score"),
          "eligible", grade.get("eligible"), cycles, flush=True)

    power = [v["timing"].get("peak_window_power_w") for v in grade.get("cases", {}).values()
             if v.get("timing")]
    metrics = {
        "candidate": cid,
        "reports": {"grade": grade},
        "score": grade.get("experimental_score"),
        "cycles": cycles,
        "eligible": grade.get("eligible"),
        "area_mm2": grade.get("gate_diagnostics", {}).get("area_mm2"),
        "power_w": max(power) if power else None,
        "evaluation_attempts": 1,
        "elapsed": elapsed,
        "runs": {"grade": {"command": " ".join(cmd), "elapsed": round(elapsed, 1)}},
    }
    if rc != 0 or not grade:
        metrics.update(failure_class="grade_error", status="not_verified",
                       failure_note=f"grade rc={rc}; see reports/grade.stderr")
    (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    with writer_lock():
        reg = state.load()
        prior_head = reg["head"]
        records.record_result(reg, cand, metrics)
        promoted = reg["head"] == cid and prior_head != cid
        state.save(reg)
        engine.write_search_state(reg)
    if promoted:
        cand.stage_to_worktree()
        (REPO / "programs/M1_P1.asm").write_text(p1)
        (REPO / "programs/M2_D1.asm").write_text(d1)
    with (REPO / "project/iteration-log.md").open("a") as log:
        log.write(
            f"\n### {cid} verdict: {'PROMOTED to root' if promoted else 'not promoted'}\n"
            f"- Full frozen joint grade rc={rc} score={grade.get('experimental_score')} "
            f"eligible={grade.get('eligible')} cycles={cycles} elapsed={elapsed:.0f}s\n"
            f"- Root files {'staged from candidate inputs/programs' if promoted else 'unchanged'}.\n")
    print("promoted" if promoted else "not promoted", flush=True)


if __name__ == "__main__":
    main()
