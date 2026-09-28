"""Central coordinator for candidate evaluation records and state updates.

Owns recording evaluation results, state/archives synchronization,
incumbent promotion, deltas calculation, and iteration-log appending.
Caller is responsible for saving state (state.save).
Inputs and report files of past candidates remain immutable.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from . import archives
from .candidate import REPO

DEFAULT_LOG = REPO / "project" / "iteration-log.md"
BASELINE_CYCLES = {"M1_P1": 30996995, "M2_D1": 1302032}


def _extract_cycles(grade: dict, metrics: dict) -> dict[str, int]:
    cycles: dict[str, int] = {}
    grade_cases = grade.get("cases", {}) if isinstance(grade, dict) else {}
    for case in ("M1_P1", "M2_D1"):
        c_info = grade_cases.get(case, {})
        if isinstance(c_info, dict) and "cycles" in c_info and c_info["cycles"] is not None:
            cycles[case] = int(c_info["cycles"])
        elif "cycles" in metrics and isinstance(metrics["cycles"], dict):
            if case in metrics["cycles"] and metrics["cycles"][case] is not None:
                cycles[case] = int(metrics["cycles"][case])
    return cycles


def _extract_functional(grade: dict, check: dict, metrics: dict) -> tuple[dict[str, bool], bool | None]:
    func_cases: dict[str, bool] = {}
    grade_cases = grade.get("cases", {}) if isinstance(grade, dict) else {}
    check_cases = check.get("cases", {}) if isinstance(check, dict) else {}

    for case in ("M1_P1", "M2_D1"):
        passed = None
        if case in grade_cases and isinstance(grade_cases[case], dict):
            passed = grade_cases[case].get("functional_passed")
        if passed is None and case in check_cases and isinstance(check_cases[case], dict):
            passed = check_cases[case].get("functional_passed")
        if passed is not None:
            func_cases[case] = bool(passed)

    if func_cases:
        all_passed = set(func_cases) == {"M1_P1", "M2_D1"} and all(v is True for v in func_cases.values())
        return func_cases, all_passed

    if "functional_passed" in metrics and metrics["functional_passed"] is not None:
        return {}, bool(metrics["functional_passed"])

    return {}, None


def record_result(reg: dict, cand: Any, metrics: dict, log_path: Path | None = None) -> None:
    """Update candidate and registry state atomically from coordinator perspective.

    Caller saves state.
    Updates:
      - cand.status and cand.failure_class (without touching inputs/reports)
      - registry meta: cycles, score, eligible, functional flags, failure, deltas
      - archives (archives.update)
      - head / search best
      - appends iteration-log entry (including cache hits and failures with evidence)
    """
    reg.setdefault("candidates", {})
    cid = getattr(cand, "id", None) or metrics.get("candidate", "cand-unknown")
    meta = reg["candidates"].setdefault(cid, {})

    # Preserve or initialize basic metadata
    if hasattr(cand, "parent") and cand.parent:
        meta.setdefault("parent", cand.parent)
    if hasattr(cand, "hypothesis") and cand.hypothesis:
        meta.setdefault("hypothesis", cand.hypothesis)
    if hasattr(cand, "kind") and cand.kind:
        meta.setdefault("kind", cand.kind)
    if hasattr(cand, "round") and cand.round is not None:
        meta.setdefault("round", cand.round)
    if hasattr(cand, "family") and cand.family:
        meta.setdefault("family", cand.family)
    if hasattr(cand, "bias") and cand.bias:
        meta.setdefault("bias", cand.bias)
    if hasattr(cand, "struct_sig") and cand.struct_sig:
        meta.setdefault("struct_sig", cand.struct_sig)
    if hasattr(cand, "fingerprint") and cand.fingerprint:
        meta.setdefault("fingerprint", cand.fingerprint)

    reports = metrics.get("reports", {}) if isinstance(metrics, dict) else {}
    grade = reports.get("grade", {}) if isinstance(reports, dict) else {}
    check = reports.get("check", {}) if isinstance(reports, dict) else {}
    verdict = metrics.get("verdict")

    # 1. Cycles
    cycles = _extract_cycles(grade, metrics)
    if cycles:
        meta["cycles"] = cycles

    # 2. Score
    score = None
    if isinstance(grade, dict) and grade.get("score") is not None:
        score = float(grade["score"])
    elif metrics.get("score") is not None:
        score = float(metrics["score"])
    if score is not None:
        meta["score"] = score

    # 3. Eligible flag
    eligible = None
    if isinstance(grade, dict) and "eligible" in grade and grade["eligible"] is not None:
        eligible = bool(grade["eligible"])
    elif "eligible" in metrics and metrics["eligible"] is not None:
        eligible = bool(metrics["eligible"])
    if eligible is not None:
        meta["eligible"] = eligible

    # 4. Functional flags
    func_cases, func_passed = _extract_functional(grade, check, metrics)
    if func_cases:
        meta["functional"] = func_cases
    if func_passed is not None:
        meta["functional_passed"] = func_passed

    # 5. Area and power
    if isinstance(grade, dict) and "area_mm2" in grade and grade["area_mm2"] is not None:
        meta["area_mm2"] = grade["area_mm2"]
    elif "area_mm2" in metrics and metrics["area_mm2"] is not None:
        meta["area_mm2"] = metrics["area_mm2"]

    power_w = None
    if isinstance(grade, dict) and grade.get("cases"):
        pw_list = [m.get("peak_window_power_w") for m in grade["cases"].values() if isinstance(m, dict)]
        pw_valid = [p for p in pw_list if p is not None]
        if pw_valid:
            power_w = max(pw_valid)
    elif "power_w" in metrics:
        power_w = metrics["power_w"]
    if power_w is not None:
        meta["power_w"] = power_w

    # 6. Status and Failure classification
    cand_status = "evaluated"
    failure_class = "none"
    failure_note = metrics.get("failure_note") or meta.get("failure_note")

    if verdict == "incorrect" or func_passed is False:
        cand_status = "incorrect"
        failure_class = "incorrect"
    elif verdict == "screen_pruned" or metrics.get("failure_class") == "screen_pruned":
        cand_status = "pruned"
        failure_class = "screen_pruned"
    elif verdict == "ineligible" or eligible is False:
        cand_status = "evaluated"
        failure_class = "ineligible"
    elif metrics.get("failure_class") and metrics["failure_class"] not in (None, "none"):
        cand_status = metrics.get("status", "not_verified")
        failure_class = metrics["failure_class"]
    elif score is not None and eligible is True and func_passed is True:
        cand_status = "evaluated"
        failure_class = "none"
    else:
        cand_status = metrics.get("status", "evaluated")
        failure_class = metrics.get("failure_class", "none")

    if hasattr(cand, "status"):
        cand.status = cand_status
    if hasattr(cand, "failure_class"):
        cand.failure_class = failure_class
    meta["status"] = cand_status
    meta["failure_class"] = failure_class
    if failure_note:
        meta["failure_note"] = failure_note
    meta["evaluation_attempts"] = metrics.get("evaluation_attempts", 0)
    meta["elapsed"] = metrics.get("elapsed")
    meta["cache_source"] = metrics.get("cache_source")

    # 7. Compute deltas vs prior head and parent
    prior_head = reg.get("head")
    deltas: dict[str, Any] = {}
    if prior_head and prior_head in reg["candidates"]:
        head_meta = reg["candidates"][prior_head]
        head_cycles = head_meta.get("cycles") or {}
        head_score = head_meta.get("score")
        cyc_delta = {}
        for case, c_val in cycles.items():
            if case in head_cycles and head_cycles[case] is not None:
                cyc_delta[case] = c_val - head_cycles[case]
        if cyc_delta:
            deltas["cycles"] = cyc_delta
        if score is not None and head_score is not None:
            deltas["score"] = score - head_score
        deltas["prior_head"] = prior_head

    parent_id = getattr(cand, "parent", None) or meta.get("parent")
    if parent_id and parent_id in reg["candidates"]:
        parent_meta = reg["candidates"][parent_id]
        p_cycles = parent_meta.get("cycles") or {}
        p_score = parent_meta.get("score")
        p_delta: dict[str, Any] = {"parent": parent_id}
        p_cyc_delta = {}
        for case, c_val in cycles.items():
            if case in p_cycles and p_cycles[case] is not None:
                p_cyc_delta[case] = c_val - p_cycles[case]
        if p_cyc_delta:
            p_delta["cycles"] = p_cyc_delta
        if score is not None and p_score is not None:
            p_delta["score"] = score - p_score
        deltas["vs_parent"] = p_delta

    meta["deltas"] = deltas

    # Save candidate metadata without touching inputs or reports
    if hasattr(cand, "save"):
        try:
            cand.save()
        except Exception:
            pass

    # 8. Recompute archives
    archives.update(reg, cid)

    # 9. Incumbent promotion:
    # "Prefer incumbent promotion only score improves and BOTH cycles <= prior head; otherwise retain Pareto."
    is_adm = archives.is_admissible(meta)
    promoted = False
    if is_adm:
        if prior_head is None:
            reg["head"] = cid
            cand_status = "kept"
            meta["status"] = "kept"
            if hasattr(cand, "status"):
                cand.status = "kept"
            if hasattr(cand, "save"):
                try:
                    cand.save()
                except Exception:
                    pass
            promoted = True
        else:
            head_meta = reg["candidates"].get(prior_head, {})
            head_score = head_meta.get("score")
            head_cycles = head_meta.get("cycles") or {}

            score_improves = score is not None and (head_score is None or score > head_score)
            p1_le = (
                "M1_P1" in cycles and "M1_P1" in head_cycles
                and head_cycles["M1_P1"] is not None
                and cycles["M1_P1"] <= head_cycles["M1_P1"]
            )
            d1_le = (
                "M2_D1" in cycles and "M2_D1" in head_cycles
                and head_cycles["M2_D1"] is not None
                and cycles["M2_D1"] <= head_cycles["M2_D1"]
            )

            if score_improves and p1_le and d1_le:
                reg["head"] = cid
                cand_status = "kept"
                meta["status"] = "kept"
                if hasattr(cand, "status"):
                    cand.status = "kept"
                if hasattr(cand, "save"):
                    try:
                        cand.save()
                    except Exception:
                        pass
                promoted = True
            else:
                # Otherwise retain Pareto: head remains prior_head
                pass

    # 10. Update search best
    best_cid = archives.best_id(reg)
    if best_cid and best_cid in reg.get("candidates", {}):
        reg.setdefault("search", {})
        reg["search"]["best_id"] = best_cid
        best_score = reg["candidates"][best_cid].get("score")
        if best_score is not None:
            reg["search"]["best_score"] = best_score

    # 11. Append once; evidence path and command are retained in metrics.
    if not meta.get("result_logged"):
        append_iteration_log(cand, metrics, meta, log_path=log_path, promoted=promoted)
        meta["result_logged"] = True


def append_iteration_log(
    cand: Any,
    metrics: dict,
    meta: dict,
    log_path: Path | None = None,
    promoted: bool = False,
) -> None:
    """Append one iteration-log entry per finalized candidate.

    Includes cache hits and failures with evidence and elapsed.
    Must never throw on missing errors or missing fields.
    """
    try:
        path = log_path or DEFAULT_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = f" — {cand.id}: "
        if path.exists() and any(line.startswith("## ") and marker in line for line in path.read_text().splitlines()):
            return
        lines = _format_log_entry(cand, metrics, meta, promoted=promoted)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
    except Exception:
        raise


def _format_log_entry(cand: Any, metrics: dict, meta: dict, promoted: bool = False) -> list[str]:
    created_at = getattr(cand, "created_at", None) or time.strftime("%Y-%m-%d %H:%M:%S")
    cid = getattr(cand, "id", None) or meta.get("id") or "cand-unknown"
    hyp = getattr(cand, "hypothesis", "") or meta.get("hypothesis", "")
    summary = getattr(cand, "summary", "") or meta.get("summary", "-")
    parent = getattr(cand, "parent", None) or meta.get("parent", "(root)")

    cached = bool(metrics.get("cached") or metrics.get("cache_hit"))
    source_cand = metrics.get("source_candidate") or metrics.get("cached_from")

    elapsed = metrics.get("elapsed") or metrics.get("wall_time")
    elapsed_str = f" elapsed={elapsed:.2f}s" if (isinstance(elapsed, (int, float)) and elapsed > 0) else ""

    cache_tag = f" [CACHE HIT from {source_cand}]" if (cached and source_cand) else ("[CACHE HIT]" if cached else "")
    lines = [f"## {created_at} — {cid}: {hyp}{cache_tag}"]
    lines.append(f"- Change: parent={parent}; {summary}")
    lines.append(f"- Evidence: project/experiments/{cid}/metrics.json; reports/grade.json; attempts={metrics.get('evaluation_attempts', 0)}")
    for mode, run in metrics.get("runs", {}).items():
        lines.append(f"- Command ({mode}): {run.get('command')}; elapsed={run.get('elapsed')}; peak MiB={run.get('peak_memory_mib')}")

    # Candidate input hashes if present
    hashes_str = ""
    try:
        if hasattr(cand, "input_hashes"):
            h = cand.input_hashes()
            hashes_str = " ".join(f"{k}@{v[:8]}" for k, v in sorted(h.items()))
    except Exception:
        pass
    if hashes_str:
        lines.append(f"- Inputs: {hashes_str}")

    status = meta.get("status", "evaluated")
    failure_class = meta.get("failure_class", "none")
    failure_note = meta.get("failure_note") or metrics.get("failure_note")

    reports = metrics.get("reports", {}) if isinstance(metrics, dict) else {}
    grade = reports.get("grade", {}) if isinstance(reports, dict) else {}
    check = reports.get("check", {}) if isinstance(reports, dict) else {}

    # Format verification line
    if failure_class not in (None, "none") or status in ("incorrect", "pruned", "ineligible"):
        err_msg = failure_note
        if not err_msg and isinstance(check, dict) and check.get("cases"):
            failed_cases = [c for c, m in check["cases"].items() if isinstance(m, dict) and m.get("functional_passed") is False]
            if failed_cases:
                err_msg = f"functional check failed on {failed_cases}"
        if not err_msg:
            err_msg = f"failure_class={failure_class}"
        evidence = metrics.get("evidence") or (f"reports/check.json" if failure_class == "incorrect" else "pipeline")
        lines.append(f"- Verify: FAILED — {err_msg}; evidence={evidence}{elapsed_str}")
        lines.append(f"- Result: status={status}, failure_class={failure_class}")
        lines.append(f"- Decision: reject/preserve failed candidate (status={status})")
    else:
        # Success verification line
        max_err = None
        if isinstance(check, dict) and check.get("cases"):
            errs = [m.get("max_abs_error") for m in check["cases"].values() if isinstance(m, dict) and m.get("max_abs_error") is not None]
            if errs:
                max_err = max(errs)
        err_str = f" (max err {max_err:.2e})" if max_err is not None else ""
        if cached:
            verify_str = f"PASS (cached from {source_cand}){err_str}"
        else:
            verify_str = f"PASS{err_str}"
        lines.append(f"- Verify: {verify_str}{elapsed_str}")

        # Result line
        cycles = meta.get("cycles") or {}
        cyc_parts = []
        for case in ("M1_P1", "M2_D1"):
            if case in cycles:
                c_val = cycles[case]
                ratio = BASELINE_CYCLES.get(case, 1) / max(c_val, 1)
                cyc_parts.append(f"{case}={c_val} ({ratio:.3f}x)")
        cyc_str = ", ".join(cyc_parts) if cyc_parts else "cycles=n/a"
        score_val = meta.get("score")
        score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else str(score_val)
        eligible_val = meta.get("eligible")
        lines.append(f"- Result: {cyc_str}; score={score_str}, eligible={eligible_val}")

        # Deltas line
        deltas = meta.get("deltas") or {}
        if deltas.get("cycles"):
            lines.append(f"  - deltas vs head ({deltas.get('prior_head')}): {deltas['cycles']}")

        # Decision line
        if promoted:
            lines.append(f"- Decision: KEPT — new incumbent head (score={score_str})")
        elif status == "kept":
            lines.append(f"- Decision: KEPT")
        else:
            lines.append(f"- Decision: evaluated; retained in Pareto/archives (head preserved)")

    return lines
