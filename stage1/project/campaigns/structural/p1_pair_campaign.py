"""Bounded P1 dataflow search with matched D1 timing and frozen finalists."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
REFINE = ROOT / "project/campaigns/d1-sh/refine.py"
VERIFY = ROOT / "project/campaigns/d1-sh/verify.py"
PREPARE = ROOT / "project/campaigns/structural/prepare_pair.py"
GRADE = ROOT / "project/campaigns/structural/grade_joint.py"
BASE_P1 = ROOT / "project/campaigns/structural/p1_paired_w1_step_rf_noc512"
BASE_D1 = ROOT / "project/campaigns/d1-sh/layer_rf_tc_noc512"
BASE_D1_CYCLES = 58770


def proposals():
    result = []
    def add(name, recipe=None, hardware=None):
        result.append({"name":name,"recipe_patch":recipe or {},"hardware_patch":hardware or {}})
    add("paired_preload_a", {"gemm_stages":{"p1_prompt":{"paired_a_preload":1}}})
    add("paired_late_b", {"gemm_stages":{"p1_prompt":{"paired_b_prefetch":0}}})
    add("paired_k16", {"gemm_stages":{"p1_prompt":{"paired_k_chunk":16}}})
    add("step_k16_n64", {"gemm_stages":{"p1_step":{"k_chunk":16,"n_tile":64}}})
    add("step_k16_n32", {"gemm_stages":{"p1_step":{"k_chunk":16,"n_tile":32}}})
    add("step_k32_n16", {"gemm_stages":{"p1_step":{"k_chunk":32,"n_tile":16}}})
    add("prompt_b2", {"gemm_stages":{"p1_prompt":{"b_buffers":2}}})
    add("w2_no_operand_pipeline", {"gemm_stages":{"p1_prompt":{"operand_pipeline":["qkv","wo","w1"]}}})
    for name, patch in [
        ("cache4",{"cache_mib":4}),("cache2",{"cache_mib":2}),
        ("cache1",{"cache_mib":1}),("cache0",{"cache_mib":0}),
        ("noc256",{"noc_bytes_per_cycle":256}),
        ("hbm8_cache4",{"hbm_channels":8,"cache_mib":4}),
        ("hbm8_cache2",{"hbm_channels":8,"cache_mib":2}),
        ("sh8",{"shared_banks":8}),("sh2r",{"shared_ports":"2R1W"}),
        ("sh8_2r_cache4",{"shared_banks":8,"shared_ports":"2R1W","cache_mib":4}),
        ("tc8x8_kp2",{"tc_array":"8x8","tc_k_parallel":2}),
        ("tc8x16_kp1",{"tc_array":"8x16","tc_k_parallel":1}),
        ("tc4x8_kp2",{"tc_k_parallel":2}),
        ("dma2",{"dma_engines":2}),("dma_depth2",{"dma_depth":2}),
        ("rf4r2w",{"rf_ports":"4R2W"}),
        ("sh1bank",{"shared_banks":1}),
        ("no_multicast",{"multicast":False}),
        ("sm_noc64",{"sm_noc_bytes_per_cycle":64}),
    ]:
        add(name, hardware=patch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--workers",type=int,default=6,choices=range(1,9))
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    plan=proposals()
    if args.plan_only:
        print(json.dumps(plan,indent=2)); return
    run=ROOT/"project/campaigns/structural"/args.name
    run.mkdir(exist_ok=False)
    (run/"compiler.py").write_bytes((ROOT/"project/compiler.py").read_bytes())
    (run/"driver.py").write_bytes(Path(__file__).read_bytes())
    (run/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    state={"status":"running","phase":"screens","started_utc":datetime.now(timezone.utc).isoformat(),
           "workers":args.workers,"planned":len(plan),"results":[],"finalists":[]}
    def save():
        (run/"state.json").write_text(json.dumps(state,indent=2)+"\n")
    def call(command,stem,timeout):
        with (run/f"{stem}.stdout").open("w") as stdout,(run/f"{stem}.stderr").open("w") as stderr:
            try:
                return subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                with (ROOT/"project/iteration-log.md").open("a") as log:
                    log.write(f"\n## {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — {stem} timeout\n"
                              f"- Verify: {command!r}; logs in `{run.relative_to(ROOT)}`.\n"
                              "- Result: timed out; no verified score.\n"
                              "- Decision: reject this attempt; keep its snapshots.\n")
                return 124
    def screen(job):
        name=job["name"]
        p1cmd=[sys.executable,str(REFINE),name,"--case","M1_P1","--out-root",str(run),
               "--base",str(BASE_P1/"recipe.json"),"--hardware",str(BASE_P1/"hardware.json"),
               "--patch",json.dumps(job["recipe_patch"]),
               "--hardware-patch",json.dumps(job["hardware_patch"]),
               "--compiler",str(run/"compiler.py")]
        result={"name":name,"recipe_patch":job["recipe_patch"],"hardware_patch":job["hardware_patch"]}
        result["P1_returncode"]=call(p1cmd,name+"__p1",3600)
        p1report=run/name/"critpath.json"
        if result["P1_returncode"] or not p1report.exists(): return result
        p1=json.loads(p1report.read_text())
        result.update(P1=p1["cycles"],area_mm2=p1["area_mm2"],P1_peak_w=p1["peak_window_power_w"])
        if result["area_mm2"]>24 or result["P1_peak_w"]>20: return result
        if not job["hardware_patch"]:
            result.update(D1=BASE_D1_CYCLES,D1_source=str(BASE_D1),D1_verified=True)
        else:
            dname=name+"__d1"
            dcmd=[sys.executable,str(REFINE),dname,"--case","M2_D1","--out-root",str(run),
                  "--base",str(BASE_D1/"recipe.json"),"--hardware",str(BASE_D1/"hardware.json"),
                  "--patch","{}","--hardware-patch",json.dumps(job["hardware_patch"]),
                  "--compiler",str(run/"compiler.py")]
            result["D1_returncode"]=call(dcmd,dname,900)
            d1report=run/dname/"critpath.json"
            if result["D1_returncode"] or not d1report.exists(): return result
            d1=json.loads(d1report.read_text())
            result.update(D1=d1["cycles"],D1_peak_w=d1["peak_window_power_w"],
                          D1_source=str(run/dname),D1_verified=False)
            if result["D1_peak_w"]>20: return result
        result["predicted_score"]=1000*(30996995*1302032/(result["P1"]*result["D1"]))**0.5
        return result
    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(screen,job) for job in plan]
            for future in as_completed(futures):
                result=future.result()
                state["results"].append(result); save()
                print(json.dumps(result),flush=True)
        state["phase"]="frozen_finalists"; save()
        ranked=sorted((r for r in state["results"] if "predicted_score" in r),
                      key=lambda r:-r["predicted_score"])
        for result in ranked[:3]:
            if result["D1_verified"]:
                result["frozen_D1_passed"]=True
            else:
                src=Path(result["D1_source"])
                result["frozen_D1_passed"]=call([sys.executable,str(VERIFY),str(src)],
                                                 result["name"]+"__verify",1800)==0
            state["finalists"].append(result); save()
        state["phase"]="joint_grades"; save()
        launched=0
        for result in state["finalists"]:
            if not result["frozen_D1_passed"] or launched>=2: continue
            p1dir=run/result["name"]
            d1dir=Path(result["D1_source"])
            if call([sys.executable,str(PREPARE),str(p1dir),str(d1dir)],
                    result["name"]+"__prepare",120): continue
            result["joint_grade_returncode"]=call(
                [sys.executable,str(GRADE),str(p1dir/"joint")],
                result["name"]+"__grade",11000)
            launched+=1; save()
        state["status"]="completed"
    except BaseException as exc:
        state.update(status="failed",error=repr(exc)); raise
    finally:
        save()


if __name__=="__main__":
    main()
