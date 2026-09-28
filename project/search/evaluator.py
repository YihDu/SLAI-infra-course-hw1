"""Isolated evaluator: generate -> expanded static screen -> grade.

The only ground truth is challenge.py. This module runs it as a subprocess and
parses the written report files; it never interprets or scores anything itself.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE_MANIFEST = REPO / "baseline_manifest.json"


def python_exe() -> str:
    exe = os.environ.get("CHALLENGE_PYTHON")
    if exe:
        return exe
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def summarize(report: dict) -> dict:
    """Reduce a check/estimate/grade report to the metrics we compare on."""
    out = {"mode": report.get("mode"), "cases": {}}
    if "experimental_score" in report:
        out["score"] = report["experimental_score"]
        out["eligible"] = report.get("eligible")
        out["area_mm2"] = report.get("gate_diagnostics", {}).get("area_mm2")
    for case, block in report.get("cases", {}).items():
        entry = {"functional_passed": block.get("functional_passed")}
        errors = [
            c.get("max_absolute_error")
            for c in block.get("functional", [])
            for c in c.get("checks", {}).values()
        ]
        if errors:
            entry["max_abs_error"] = max(e for e in errors if e is not None)
        timing = block.get("timing")
        if timing:
            entry["cycles"] = timing["cycles"]
            entry["instruction_count"] = timing["instruction_count"]
            entry["hbm_read_bytes"] = timing["hbm_read_bytes"]
            entry["hbm_write_bytes"] = timing["hbm_write_bytes"]
            entry["peak_window_power_w"] = timing["peak_window_power_w"]
            stats = timing.get("resource_stats") or {}
            busy = {}
            for sm, s in (stats.get("sms") or {}).items():
                for res, cyc in (s.get("busy_cycles") or {}).items():
                    busy[f"sm{sm}.{res}"] = cyc
                busy[f"sm{sm}.issue_wait"] = s.get("issue_wait_cycles")
            for res, cyc in (stats.get("shared", {}).get("busy_cycles") or {}).items():
                busy[f"shared.{res}"] = cyc
            entry["busy_cycles"] = busy
        out["cases"][case] = entry
    return out


def prepare(cand, log=print, timeout_seconds=600, memory_mib=3072):
    from . import runtime
    deadline = time.monotonic() + timeout_seconds
    log(f"[{cand.id}] generate", flush=True) if log is print else log(f"[{cand.id}] generate")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    result = runtime.run([python_exe(), str(cand.inputs / "compiler.py")], cand.dir,
                         cand.reports / "generate", timeout_seconds, memory_mib, env)
    metrics = {"candidate": cand.id, "reports": {}, "evaluation_attempts": 0, "generation": result}
    if result["failure_class"] or result["exit_code"]:
        metrics.update(verdict="not_verified", failure_class=result["failure_class"] or "generation_error")
        return metrics
    # Parse in a bounded child: loop expansion itself can consume substantial memory.
    script = "from codesign.challenge.isa import parse; import json,pathlib; out={}; " + \
        "\nfor n in ('M1_P1','M2_D1'):\n p=pathlib.Path('programs')/(n+'.asm'); ins=parse(p.read_text()); out[n]={'expanded_instructions':len(ins),'bytes':p.stat().st_size,'workgroup_lifetimes':sum(i.op=='WG.BEGIN' for i in ins)}\n" + \
        "pathlib.Path('reports/static.json').write_text(json.dumps(out))"
    screen = runtime.run([python_exe(), "-c", script], cand.dir, cand.reports / "static",
                         max(.01, deadline-time.monotonic()), memory_mib, env)
    if screen["failure_class"] or screen["exit_code"]:
        metrics.update(verdict="not_verified", failure_class=screen["failure_class"] or "static_error", static_run=screen)
        return metrics
    metrics["static"] = json.loads((cand.reports / "static.json").read_text())
    if any(v["expanded_instructions"] > 1000000 for v in metrics["static"].values()):
        metrics.update(verdict="pruned", failure_class="screen_pruned")
    if not metrics.get("failure_class") and any(v.get("workgroup_lifetimes", 1) > 1 for v in metrics["static"].values()):
        # This is a cost forecast only. The frozen grade still performs all checks.
        script = ("import json,pathlib; from project.search.validation_cost import analyze; "
                  "stats=json.loads(pathlib.Path('reports/static.json').read_text()); "
                  f"out={{n:analyze((pathlib.Path('programs')/(n+'.asm')).read_text(),{memory_mib}) "
                  "for n,v in stats.items() if v.get('workgroup_lifetimes',1)>1}; "
                  "pathlib.Path('reports/validation-cost.json').write_text(json.dumps(out,indent=2))")
        cost_run = runtime.run([python_exe(), "-c", script], cand.dir, cand.reports / "validation-cost",
                               max(.01, deadline-time.monotonic()), memory_mib, env)
        metrics["validation_cost_run"] = cost_run
        if cost_run["failure_class"] or cost_run["exit_code"]:
            metrics.update(verdict="not_verified", failure_class=cost_run["failure_class"] or "cost_model_error")
        else:
            cost = json.loads((cand.reports / "validation-cost.json").read_text())
            metrics["validation_cost"] = cost
            proposal_path = cand.dir / "proposal.json"
            proposal = json.loads(proposal_path.read_text()) if proposal_path.exists() else {}
            calibration = proposal.get("validation_cost_calibration") is True
            if any(v["decision"] == "prune_validation_cost" for v in cost.values()):
                if calibration:
                    metrics["validation_cost_override"] = proposal.get("calibration_reason", "explicit calibration")
                else:
                    metrics.update(verdict="pruned", failure_class="validation_cost_pruned")
    return metrics


def evaluate(cand, seeds, log=print, *, validation="grade", timeout_seconds=600,
             memory_mib=3072, prepared=False, before_launch=None):
    from . import runtime
    import time
    if validation not in ("grade", "full"):
        raise ValueError(validation)
    started = time.monotonic()
    metrics = {"candidate": cand.id, "reports": {}, "evaluation_attempts": 0, "runs": {}}
    if not prepared:
        metrics.update(prepare(cand, log, timeout_seconds, memory_mib))
        if metrics.get("failure_class"):
            return metrics
    elif (cand.reports / "static.json").exists():
        metrics["static"] = json.loads((cand.reports / "static.json").read_text())
    for mode in (["check", "grade"] if validation == "full" else ["grade"]):
        report = cand.reports / (mode + ".json")
        if report.exists():
            raise FileExistsError(report)
        cmd = [python_exe(), str(REPO / "challenge.py"), mode,
               "--hardware", str(cand.inputs / "hardware.json"),
               "--program-p1", str(cand.programs / "M1_P1.asm"),
               "--program-d1", str(cand.programs / "M2_D1.asm"), "--report", str(report)]
        for seed in seeds:
            cmd += ["--seed", str(seed)]
        if mode == "grade":
            cmd += ["--baseline", str(BASELINE_MANIFEST)]
        try:
            if before_launch:
                before_launch(mode)
        except RuntimeError as exc:
            metrics.update(verdict="not_verified", failure_class="budget_exhausted", failure_note=str(exc)); break
        metrics["evaluation_attempts"] += 1
        log(f"[{cand.id}] {mode}")
        remaining = max(.01, timeout_seconds - (time.monotonic() - started))
        result = runtime.run(cmd, cand.dir, cand.reports / mode, remaining, memory_mib)
        metrics["runs"][mode] = result
        if result["failure_class"] or not report.exists():
            metrics.update(verdict="not_verified", failure_class=result["failure_class"] or "evaluator_error",
                           failure_note=(cand.reports / (mode+".stderr")).read_text()[-2000:]); break
        raw = json.loads(report.read_text())
        reduced = summarize(raw); metrics["reports"][mode] = reduced
        cases = reduced.get("cases", {})
        if set(cases) != {"M1_P1", "M2_D1"} or any(c.get("functional_passed") is not True for c in cases.values()):
            metrics.update(verdict="incorrect", failure_class="incorrect"); break
        if mode == "grade":
            if reduced.get("eligible") is not True:
                metrics.update(verdict="ineligible", failure_class="ineligible")
            elif result["exit_code"]:
                metrics.update(verdict="not_verified", failure_class="evaluator_error")
            else:
                metrics["verdict"] = "evaluated"
    metrics["elapsed"] = time.monotonic() - started
    (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    return metrics


def evaluate_search(cand, seeds, log=print, **kwargs):
    kwargs.pop("screen_floor", None)
    return evaluate(cand, seeds, log, **kwargs)
