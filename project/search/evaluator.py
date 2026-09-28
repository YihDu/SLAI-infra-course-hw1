"""Deterministic evaluator wrapper: generate -> check -> estimate -> grade.

The only ground truth is challenge.py. This module runs it as a subprocess and
parses the written report files; it never interprets or scores anything itself.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE_MANIFEST = REPO / "baseline_manifest.json"


def python_exe() -> str:
    exe = os.environ.get("CHALLENGE_PYTHON")
    if exe:
        return exe
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def generate() -> None:
    """Regenerate programs/*.asm from the staged project/compiler.py."""
    proc = subprocess.run(
        [python_exe(), "-m", "project.compiler"],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"generate failed:\n{proc.stdout}\n{proc.stderr}")


def run_challenge(mode: str, report: Path, seeds: list[int] | None = None) -> dict:
    """Run `challenge.py <mode>` and return the parsed report it wrote."""
    if mode not in ("check", "estimate", "grade"):
        raise ValueError(f"unknown mode {mode}")
    if report.exists():
        raise FileExistsError(f"report exists: {report}")
    cmd = [
        python_exe(), "challenge.py", mode,
        "--hardware", "hardware.json",
        "--program-p1", "programs/M1_P1.asm",
        "--program-d1", "programs/M2_D1.asm",
        "--report", str(report),
    ]
    for seed in seeds or []:
        cmd += ["--seed", str(seed)]
    if mode == "grade":
        cmd += ["--baseline", str(BASELINE_MANIFEST)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if not report.exists():
        raise RuntimeError(
            f"{mode} produced no report (rc={proc.returncode}):\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    result = json.loads(report.read_text())
    result["_exit_code"] = proc.returncode
    return result


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


def static_screen(p1: Path, d1: Path) -> dict:
    """Free pre-eval screen on generated asm: line/opcode counts.

    Returns counts and a rough instruction volume proxy. Used to prune
    absurd candidates before paying simulator time, and to flag borderline
    ones for an optional `estimate` pass.
    """
    def counts(path: Path) -> dict:
        ops = {}
        n = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            op = line.split(" ", 1)[0]
            ops[op] = ops.get(op, 0) + 1
            n += 1
        return {"lines": n, "ops": ops}
    return {"M1_P1": counts(p1), "M2_D1": counts(d1)}


def evaluate_search(cand, seeds: list[int], log=print,
                    screen_floor: float | None = None) -> dict:
    """Search-path evaluation: generate -> static screen -> [estimate] -> grade.

    `grade` already contains per-case functional_passed + timing + score, so a
    single grade run serves as both the correctness gate and the score. The
    optional `estimate` pass only runs when static screening cannot decide and
    a `screen_floor` score was supplied.
    """
    log(f"[{cand.id}] generate")
    generate()
    for name in ("M1_P1.asm", "M2_D1.asm"):
        (cand.programs / name).write_bytes((REPO / "programs" / name).read_bytes())

    metrics = {"candidate": cand.id, "reports": {}}
    metrics["static"] = static_screen(cand.programs / "M1_P1.asm",
                                      cand.programs / "M2_D1.asm")

    if screen_floor is not None:
        est = run_challenge("estimate", cand.reports / "estimate.json")
        metrics["reports"]["estimate"] = summarize(est)
        cyc = {c: m.get("cycles") for c, m in
               metrics["reports"]["estimate"]["cases"].items()}
        if all(cyc.values()):
            est_score = 1000.0 * (
                (30996995 / cyc["M1_P1"]) * (1302032 / cyc["M2_D1"])) ** 0.5
            metrics["est_score"] = est_score
            if est_score < screen_floor:
                metrics["verdict"] = "screen_pruned"
                (cand.dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2) + "\n")
                return metrics

    grade = run_challenge("grade", cand.reports / "grade.json", seeds)
    metrics["reports"]["grade"] = summarize(grade)
    cases = metrics["reports"]["grade"]["cases"]
    if any(m.get("functional_passed") is False for m in cases.values()):
        metrics["verdict"] = "incorrect"
    elif metrics["reports"]["grade"].get("eligible") is False:
        metrics["verdict"] = "ineligible"
    else:
        metrics["verdict"] = "evaluated"
    (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def evaluate(cand, seeds: list[int], log=print) -> dict:
    """Full pipeline for a staged candidate. Returns merged metrics."""
    log(f"[{cand.id}] generate")
    generate()
    for name in ("M1_P1.asm", "M2_D1.asm"):
        (cand.programs / name).write_bytes((REPO / "programs" / name).read_bytes())

    metrics = {"candidate": cand.id, "reports": {}}

    check = run_challenge("check", cand.reports / "check.json", seeds)
    metrics["reports"]["check"] = summarize(check)
    passed = all(
        c.get("functional_passed") for c in check["cases"].values()
    )
    if not passed:
        metrics["verdict"] = "incorrect"
        (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics

    est = run_challenge("estimate", cand.reports / "estimate.json")
    metrics["reports"]["estimate"] = summarize(est)

    grade = run_challenge("grade", cand.reports / "grade.json", seeds)
    metrics["reports"]["grade"] = summarize(grade)

    for case in metrics["reports"]["grade"]["cases"]:
        m = metrics["reports"]["grade"]["cases"][case]
        c = metrics["reports"]["check"]["cases"].get(case, {})
        m["functional_passed"] = c.get("functional_passed")
        m["max_abs_error"] = c.get("max_abs_error")
    (cand.dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
