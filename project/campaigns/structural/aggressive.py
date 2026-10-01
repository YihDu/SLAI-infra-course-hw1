"""Bounded broad search: 26 screens, top-3 D1 checks, matched P1, top-2 grades.

No incumbent mutation. Development screens are not correctness certification.
The queued policy and compiler are pinned in the run directory before launch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "project/campaigns/d1-sh/layer_rf_tc_noc512"
P1BASE = ROOT / "project/experiments/cand-149/inputs/recipe.json"
REFINE = ROOT / "project/campaigns/d1-sh/refine.py"
VERIFY = ROOT / "project/campaigns/d1-sh/verify.py"
GRADE = ROOT / "project/campaigns/structural/grade_joint.py"


def proposals():
    jobs = []
    def d(name, patch=None, hw=None):
        jobs.append(dict(name=name, case="M2_D1", patch=patch or {}, hw=hw or {}))
    # Controlled resource reallocations, retaining the same 16 RF lanes.
    for name, hw in [
        ("hbm8_cache4", {"hbm_channels":8,"cache_mib":4}),
        ("cache4", {"cache_mib":4}),
        ("cache0", {"cache_mib":0}),
        ("tc2_kp2", {"tc_count":2,"tc_k_parallel":2}),
        ("tc4_kp1", {"tc_count":4,"tc_k_parallel":1}),
        ("tc8x8_kp2", {"tc_array":"8x8","tc_k_parallel":2}),
        ("tc8x8_kp2_hbm8", {"tc_array":"8x8","tc_k_parallel":2,
                                  "hbm_channels":8,"cache_mib":4}),
        ("sh8_2r_cache4", {"shared_banks":8,"shared_ports":"2R1W","cache_mib":4}),
        ("hbm8_sh8_cache4", {"hbm_channels":8,"shared_banks":8,"cache_mib":4}),
    ]:
        d(name, hw=hw)
    for name, schedule in [
        ("qkv_k16", {"projections":{"qkv":{"k_chunk":16}}}),
        ("qkv_k8", {"projections":{"qkv":{"k_chunk":8}}}),
        ("w1_k16", {"projections":{"w1":{"k_chunk":16}}}),
        ("w2_k64", {"projections":{"w2":{"k_chunk":64}}}),
        ("wo_k128", {"projections":{"wo":{"k_chunk":128}}}),
        ("wo_k32", {"projections":{"wo":{"k_chunk":32}}}),
        ("kv_hbm", {"kv_sh":0}),
        ("kv_k_only", {"kv_sh":"k"}),
        ("kv_v_only", {"kv_sh":"v"}),
        ("rf_w2_only", {"rf_resident":["l0w2","l1w2"]}),
        ("attention_unsplit", {"attention_split":0}),
    ]:
        d(name, {"d1_schedule":schedule})
    for name, patch in [
        ("p1_step_rf", {"gemm_stages":{"p1_step":{"a_resident":1}}}),
        ("p1_b2", {"gemm_stages":{"p1_prompt":{"b_buffers":2}}}),
        ("p1_k16", {"gemm_stages":{"p1_prompt":{"k_chunk":16}}}),
        ("p1_n16", {"gemm_stages":{"p1_prompt":{"n_tile":16}}}),
        ("p1_team4", {"p1_schedule":{"team":4}}),
        ("p1_step_unmerged", {"p1_schedule":{"step_merge":0}}),
    ]:
        jobs.append(dict(name=name, case="M1_P1", patch=patch, hw={}))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--jobs", type=int, default=4, choices=range(1,7))
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    plan = proposals()
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    out = ROOT / "project/campaigns/structural" / args.name
    out.mkdir(exist_ok=False)
    (out/"compiler.py").write_text((ROOT/"project/compiler.py").read_text())
    (out/"driver.py").write_text(Path(__file__).read_text())
    (out/"plan.json").write_text(json.dumps(plan, indent=2))
    spec = importlib.util.spec_from_file_location("pinned_aggressive", out/"compiler.py")
    compiler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compiler
    spec.loader.exec_module(compiler)
    state = {"status":"running", "phase":"screens", "started":datetime.now(timezone.utc).isoformat(),
             "workers":args.jobs,"screen_count":len(plan),"results":[]}
    def save():
        (out/"state.json").write_text(json.dumps(state, indent=2))
    def command(cmd, stem, timeout=7200):
        with (out/f"{stem}.stdout").open("w") as stdout, (out/f"{stem}.stderr").open("w") as stderr:
            try:
                return subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                                      timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                with (ROOT/"project/iteration-log.md").open("a") as log:
                    log.write(f"\n## {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — {stem} timeout\n"
                              f"- Verify: {cmd!r}; logs under `{out.relative_to(ROOT)}`.\n"
                              "- Result: timeout; no verified result. Decision: reject this attempt.\n")
                return 124
    def screen(job):
        cmd=[sys.executable,str(REFINE),job["name"],"--case",job["case"],
             "--out-root",str(out),"--base",str(job.get("base", P1BASE if job["case"]=="M1_P1" else BASE/"recipe.json")),
             "--hardware",str(job.get("hardware",BASE/"hardware.json")),
             "--patch",json.dumps(job["patch"]),"--hardware-patch",json.dumps(job["hw"]),
             "--compiler",str(out/"compiler.py")]
        rc=command(cmd,job["name"])
        report=out/job["name"]/"critpath.json"
        result=dict(job,returncode=rc)
        if rc==0 and report.exists():
            raw=json.loads(report.read_text())
            result.update({k:raw[k] for k in ("cycles","area_mm2","peak_window_power_w")})
            result["screen_passed"]=raw["area_mm2"]<=24 and raw["peak_window_power_w"]<=20
        return result
    def batch(jobs):
        results=[]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in pool.map(screen,jobs):
                results.append(result); state["results"].append(result); save()
                print(json.dumps(result),flush=True)
        return results
    save()
    try:
        results=batch(plan)
        good=[r for r in results if r.get("screen_passed")]
        d1=sorted((r for r in good if r["case"]=="M2_D1"),key=lambda r:r["cycles"])[:3]
        p1=sorted((r for r in good if r["case"]=="M1_P1"),key=lambda r:r["cycles"])[:2]
        # Always include the unchanged parent: a failed P1 search cannot hide
        # a hardware improvement. Rank only same-hardware measured pairs.
        p1=[{"name":"parent","base":str(P1BASE)}]+[
            {"name":r["name"],"base":str(out/r["name"]/"recipe.json")} for r in p1]
        state["phase"]="frozen_d1_and_matched_p1"; save()
        matched=[]
        for d in d1:
            directory=out/d["name"]
            if command([sys.executable,str(VERIFY),str(directory)],d["name"]+"_verify"):
                continue
            for candidate in p1:
                matched.append(dict(name=d["name"]+"__"+candidate["name"],case="M1_P1",
                                    base=candidate["base"],hardware=str(directory/"hardware.json"),
                                    patch={},hw={},d1=d["name"]))
        measured=batch(matched)
        pairs=[]
        for r in measured:
            if not r.get("screen_passed"): continue
            dc=next(d["cycles"] for d in d1 if d["name"]==r["d1"])
            pairs.append(dict(r,predicted_score=1000*(30996995*1302032/(r["cycles"]*dc))**0.5))
        state["pairs"]=sorted(pairs,key=lambda r:-r["predicted_score"])
        state["phase"]="joint_grades"; save()
        for index,pair in enumerate(state["pairs"][:2]):
            joint=out/f"joint-{index}"
            (joint/"programs").mkdir(parents=True,exist_ok=False)
            recipe=json.loads((out/pair["name"]/"recipe.json").read_text())
            recipe["d1_schedule"]=json.loads((out/pair["d1"]/"recipe.json").read_text())["d1_schedule"]
            p1text=compiler.generate_m1_p1(recipe)[0]
            d1text=compiler.generate_m1_d1(recipe=recipe)[0]
            assert p1text==(out/pair["name"]/"M1_P1.asm").read_text()
            assert d1text==(out/pair["d1"]/"M2_D1.asm").read_text()
            (joint/"programs/M1_P1.asm").write_text(p1text)
            (joint/"programs/M2_D1.asm").write_text(d1text)
            (joint/"recipe.json").write_text(json.dumps(recipe,indent=2))
            (joint/"hardware.json").write_text((out/pair["d1"]/"hardware.json").read_text())
            (joint/"compiler.py").write_text((out/"compiler.py").read_text())
            (joint/"proposal.json").write_text(json.dumps(pair,indent=2))
            command([sys.executable,str(GRADE),str(joint)],f"joint-{index}",timeout=11000)
        state["status"]="completed"
    except BaseException as exc:
        state.update(status="failed",error=repr(exc)); raise
    finally:
        save()


if __name__ == "__main__":
    main()
