"""Autonomous search engine: proposal -> pruning cascade -> evaluate -> archives.

Cascade per candidate:
    DSL/static validation -> duplicate -> constraint/resource
    -> generate -> static estimate screen -> grade (correctness + timing + score)

Correctness is a hard gate: only `functional_passed` candidates enter the
archives. The DAG is append-only; every proposal (including pruned ones)
gets a candidate directory so the graph and its evidence are complete.
"""

import json
import random
import shutil
import time
from pathlib import Path

from . import archives, dsl, propose, state
from .candidate import EXPERIMENTS, INPUT_FILES, REPO, WORKTREE_PATHS, Candidate
from .evaluator import evaluate_search, static_screen

LOG = REPO / "project" / "iteration-log.md"
STATE_MD = REPO / "project" / "search" / "search_state.md"
BASELINE_CYCLES = {"M1_P1": 30996995, "M2_D1": 1302032}
PLATEAU_EPS = 1.005          # <0.5% best-score gain counts as a plateau round
PLATEAU_EXPLORE_AT = 3       # raise exploration after this many flat rounds
PLATEAU_NEW_OP_AT = 6        # require a new structural operator after this many


def _append_log(text: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text)


def _log(text: str = "") -> None:
    print(text, flush=True)


# ---------------------------------------------------------------- candidates

def _shared_compiler() -> Path:
    """Recipe-aware compiler baseline for DSL-only candidates.

    New structural operators arrive as agent-edited compiler snapshots; a
    proposal may then carry `compiler_path` to pin a different compiler.
    """
    shared = EXPERIMENTS / "_shared" / "compiler.py"
    if not shared.exists():
        shared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKTREE_PATHS["compiler.py"], shared)
    return shared


def _materialize(reg: dict, prop: dict, round_no: int, cid: str,
                 dry: bool = False) -> Candidate:
    if (EXPERIMENTS / cid).exists():
        raise RuntimeError(f"refusing to overwrite existing candidate {cid}")
    cand = Candidate(
        id=cid, parent=prop["parent"], hypothesis=prop["hypothesis"],
        summary=json.dumps(prop["ops"]), ops=prop["ops"],
        family=prop["family"], bias=prop["bias"], kind=prop["kind"],
        round=round_no, struct_sig=prop["struct_sig"],
        fingerprint=prop["fingerprint"],
    )
    cand.save()
    parent_dir = EXPERIMENTS / prop["parent"] / "inputs"
    # compiler: proposal override > shared recipe-aware baseline
    if prop.get("compiler_path"):
        shutil.copy2(prop["compiler_path"], cand.inputs / "compiler.py")
    else:
        shutil.copy2(_shared_compiler(), cand.inputs / "compiler.py")
    (cand.inputs / "hardware.json").write_text(
        json.dumps(prop["hardware"], indent=2, sort_keys=True) + "\n")
    (cand.inputs / "recipe.json").write_text(
        json.dumps(prop["recipe"], indent=2, sort_keys=True) + "\n")
    (cand.dir / "proposal.json").write_text(
        json.dumps(prop, indent=2, sort_keys=True) + "\n")
    reg["candidates"][cid] = {
        "parent": prop["parent"], "status": "created",
        "hypothesis": prop["hypothesis"], "ops": prop["ops"],
        "family": prop["family"], "bias": prop["bias"],
        "kind": prop["kind"], "round": round_no,
        "struct_sig": prop["struct_sig"], "fingerprint": prop["fingerprint"],
    }
    if not dry:
        reg.setdefault("fingerprints", {})[prop["fingerprint"]] = cid
    return cand


def _record_failure(reg, cand, failure_class, note=""):
    meta = reg["candidates"][cand.id]
    meta["status"] = "pruned"
    meta["failure_class"] = failure_class
    if note:
        meta["failure_note"] = note
    cand.status = "pruned"
    cand.failure_class = failure_class
    cand.verdict_note = note or None
    cand.save()
    _append_log(
        f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} — {cand.id}: "
        f"{cand.hypothesis}\n"
        f"- Parent: {cand.parent}; kind={cand.kind} family={cand.family} "
        f"sig={cand.struct_sig}\n"
        f"- Ops: {cand.summary}\n"
        f"- Result: PRUNED ({failure_class}) {note}\n"
        f"- Decision: pruned before/at evaluation stage\n")


def _record_result(reg, cand, metrics):
    meta = reg["candidates"][cand.id]
    g = metrics["reports"].get("grade", {})
    verdict = metrics.get("verdict")
    meta["cycles"] = {c: m.get("cycles") for c, m in g.get("cases", {}).items()}
    meta["score"] = g.get("score")
    meta["area_mm2"] = g.get("area_mm2")
    meta["power_w"] = max(
        (m.get("peak_window_power_w") or 0) for m in g.get("cases", {}).values()
    ) if g.get("cases") else None
    if verdict == "evaluated":
        cand.status = "evaluated"
        cand.failure_class = "none"
    elif verdict == "incorrect":
        cand.status = "incorrect"
        cand.failure_class = "incorrect"
    elif verdict == "screen_pruned":
        cand.status = "pruned"
        cand.failure_class = "screen_pruned"
    else:
        cand.status = "evaluated"
        cand.failure_class = "ineligible"
    meta["status"] = cand.status
    meta["failure_class"] = cand.failure_class
    cand.save()


def _log_eval(reg, cand, metrics):
    """Append one iteration-log entry for a candidate that consumed simulator."""
    g = metrics["reports"].get("grade", {})
    meta = reg["candidates"][cand.id]
    lines = [
        f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} — {cand.id}: {cand.hypothesis}",
        f"- Parent: {cand.parent}; kind={cand.kind} family={cand.family} "
        f"bias={cand.bias} sig={cand.struct_sig} fp={cand.fingerprint}",
        f"- Ops: {cand.summary}",
    ]
    if metrics.get("verdict") == "screen_pruned":
        lines.append(
            f"- Result: pruned by estimate screen "
            f"(est_score={metrics.get('est_score'):.1f})")
        lines.append("- Decision: skipped grade (dominated by incumbent)")
    elif not g.get("cases"):
        lines.append("- Result: evaluation produced no cases")
    else:
        fc = {c: m.get("functional_passed") for c, m in g["cases"].items()}
        cyc = {c: m.get("cycles") for c, m in g["cases"].items()}
        lines.append(
            f"- Verify: functional={fc}; "
            f"M1_P1={cyc.get('M1_P1')} M2_D1={cyc.get('M2_D1')} cycles; "
            f"area={meta.get('area_mm2')}mm2 peakW={meta.get('power_w')}; "
            f"score={meta.get('score')} eligible={g.get('eligible')}")
        for case, m in g["cases"].items():
            busy = m.get("busy_cycles", {})
            top = sorted(((k, v) for k, v in busy.items()
                          if isinstance(v, int) and v), key=lambda kv: -kv[1])[:3]
            lines.append(
                f"  - {case}: hbm_r={m.get('hbm_read_bytes', 0)/1e6:.1f}MB "
                f"top busy: {', '.join(f'{k}={v}' for k, v in top)}")
        lines.append(f"- Decision: status={cand.status} "
                     f"failure_class={cand.failure_class}")
    _append_log("\n" + "\n".join(lines) + "\n")


# ------------------------------------------------------------------ cascade

def _cascade(reg, cand, prop, phase, seeds, screen_floor, budget,
             dry_run=False):
    """Returns True if the candidate consumed a grade run."""
    meta = reg["candidates"][cand.id]
    # 1. DSL/static validation (already applied in apply; re-verify ops legal)
    try:
        dsl.check_phase(prop["ops"], phase)
        recipe, hw = dsl.apply_ops(
            *propose._load_parent_inputs(prop["parent"]), prop["ops"],
            phase=phase)
    except dsl.OperatorLocked as e:
        _record_failure(reg, cand, "locked_operator", str(e))
        return False
    except dsl.DslError as e:
        _record_failure(reg, cand, "invalid_dsl", str(e))
        return False
    if dsl.fingerprint(recipe, hw) != prop["fingerprint"]:
        _record_failure(reg, cand, "invalid_dsl", "fingerprint mismatch")
        return False
    # 2. duplicate pruning (a sibling may have claimed the fp this round)
    fp = prop["fingerprint"]
    holder = reg.get("fingerprints", {}).get(fp)
    if holder and holder != cand.id:
        _record_failure(reg, cand, "duplicate", f"same point as {holder}")
        return False
    # 3. constraint/resource pruning (recipe against hw lane capacity + area)
    errs = (dsl.validate_recipe(prop["recipe"], prop["hardware"])
            + dsl.validate_hardware(prop["hardware"]))
    if errs:
        _record_failure(reg, cand, "constraint_violation", "; ".join(errs))
        return False
    # 3b. known no-op pruning: the current program is single-workgroup, so
    # sm_count (>1 in use) and multicast cannot change its timing.
    NOOP_SINGLE_WG = {"sm_count", "multicast"}
    hw_diffs = [op["path"] for op in prop["ops"] if op["op"] == "hw_set"]
    if hw_diffs and all(p in NOOP_SINGLE_WG for p in hw_diffs):
        _record_failure(reg, cand, "known_noop",
                        "single-WG program: " + ",".join(hw_diffs))
        return False
    if dry_run:
        _record_failure(reg, cand, "dry_run", "mechanism test")
        return False
    if budget["grades"] <= 0:
        _record_failure(reg, cand, "budget_exhausted")
        return False
    # 4-7. generate -> static screen -> [estimate] -> grade
    cand.status = "evaluating"
    cand.save()
    meta["status"] = "evaluating"
    cand.stage_to_worktree()
    try:
        metrics = evaluate_search(cand, seeds, log=_log,
                                  screen_floor=screen_floor)
    except Exception as e:  # generate/parse failure
        _record_failure(reg, cand, "generate_error", str(e)[:300])
        return False
    _record_result(reg, cand, metrics)
    budget["grades"] -= 1
    parent_meta = reg["candidates"].get(prop["parent"], {})
    pscore = parent_meta.get("score")
    improved = (meta.get("score") is not None and pscore is not None
                and meta["score"] > pscore)
    meta["improved"] = bool(improved)
    # dominance pruning (post-eval): correct but strictly worse than parent —
    # keep the node in the DAG, mark it dominated so it won't seed expansion.
    if (meta.get("score") is not None and pscore is not None
            and meta["score"] < pscore):
        meta["dominated_by_parent"] = True
        if meta["status"] == "evaluated":
            meta["status"] = "dominated"
            cand.status = "dominated"
            cand.save()
    _log_eval(reg, cand, metrics)
    archives.update(reg, cand.id)
    return True


# ---------------------------------------------------------------- state.md

def write_search_state(reg: dict) -> None:
    s = reg.get("search", {})
    best = archives.best_id(reg)
    bm = reg["candidates"].get(best, {}) if best else {}
    scored = sorted(
        ((c, m) for c, m in reg["candidates"].items()
         if m.get("score") is not None),
        key=lambda kv: -kv[1]["score"])
    failed = [m for m in reg["candidates"].values()
              if m.get("failure_class") not in (None, "none")]
    fc_hist = {}
    for m in failed:
        fc_hist[m["failure_class"]] = fc_hist.get(m["failure_class"], 0) + 1
    lines = [
        "# search_state (auto-generated; do not edit)",
        "",
        f"- best: {best} score={bm.get('score')} "
        f"(P1={ (bm.get('cycles') or {}).get('M1_P1')}, "
        f"D1={(bm.get('cycles') or {}).get('M2_D1')})",
        f"- phase={s.get('phase')} round={s.get('round')} "
        f"plateau_rounds={s.get('plateau_rounds')} "
        f"explore_ratio={s.get('explore_ratio')} "
        f"need_new_operator={s.get('need_new_operator')}",
        f"- bottleneck: {s.get('bottleneck', 'unknown')}",
        "",
        "## elite (score desc)",
    ]
    for c, m in scored[:5]:
        lines.append(
            f"- {c} score={m['score']:.2f} parent={m.get('parent')} "
            f"sig={m.get('struct_sig')} hyp={m.get('hypothesis', '')[:60]}")
    lines += [
        "",
        "## failure classes",
        *[f"- {k}: {v}" for k, v in sorted(fc_hist.items())],
        "",
        "## pareto front",
        *[f"- {c} {reg['candidates'][c].get('cycles')}"
          for c in reg.get("archives", {}).get("pareto", [])],
        "",
        "## diversity cells",
        *[f"- {k}: {v}" for k, v in
          sorted(reg.get("archives", {}).get("diversity", {}).items())],
    ]
    STATE_MD.write_text("\n".join(lines) + "\n")


def _bottleneck(reg, cid) -> str:
    try:
        metrics = json.loads(
            (EXPERIMENTS / cid / "metrics.json").read_text())
        cases = metrics["reports"]["grade"]["cases"]
        busy = cases["M1_P1"].get("busy_cycles", {})
        top = max(busy.items(), key=lambda kv: kv[1] or 0)
        return f"P1 {top[0]}={top[1]}"
    except Exception:
        return "unknown"


# -------------------------------------------------------------------- main

def _backfill(reg: dict) -> None:
    """Register fingerprints/archives metadata for pre-DSL candidates once."""
    fps = reg.setdefault("fingerprints", {})
    for cid, meta in reg["candidates"].items():
        if "baseline_cycles" in meta and "cycles" not in meta:
            meta["cycles"] = meta.pop("baseline_cycles")
    for cid, recipe in dsl.LEGACY_RECIPES.items():
        meta = reg["candidates"].get(cid)
        if not meta or meta.get("fingerprint"):
            continue
        hw = json.loads(
            (EXPERIMENTS / cid / "inputs" / "hardware.json").read_text())
        fp = dsl.fingerprint(recipe, hw)
        meta["fingerprint"] = fp
        meta.setdefault("struct_sig",
                        "mnk" if recipe["gemm"]["order"] == "mnk" else "nkm")
        meta.setdefault("family", "memory")
        meta.setdefault("bias", "balanced")
        meta.setdefault("kind", "manual")
        meta.setdefault("failure_class", "none")
        fps.setdefault(fp, cid)
    archives.update(reg, "backfill")


def run_search(rounds: int, per_round: int, seeds: list[int],
               dry_run: bool = False, max_grades: int = 999,
               screen: bool = False, phase: int | None = None) -> dict:
    reg = state.load()
    _backfill(reg)
    state.save(reg)
    s = reg.setdefault("search", {
        "phase": 1, "round": 0, "plateau_rounds": 0,
        "explore_ratio": 0.34, "need_new_operator": False,
    })
    if phase is not None:
        s["phase"] = phase
    s_backup = json.loads(json.dumps(s)) if dry_run else None
    phase = s.get("phase", 1)
    budget = {"grades": max_grades}
    rng = random.Random(1000 + s.get("round", 0))
    history = []

    for r in range(rounds):
        round_no = s["round"] + 1
        s["round"] = round_no
        best_before = archives.best_id(reg)
        best_score_before = (reg["candidates"].get(best_before, {})
                             .get("score") or 0)
        props = propose.propose_round(reg, round_no, per_round,
                                      s["explore_ratio"], phase, rng)
        _log(f"\n=== round {round_no}: {len(props)} proposals "
             f"(explore_ratio={s['explore_ratio']:.2f}, "
             f"grades_left={budget['grades']}) ===")
        for prop in props:
            cid = (f"drun-{reg.get('dry_counter', 0) + 1:03d}" if dry_run
                   else state.next_id(reg))
            if dry_run:
                reg["dry_counter"] = reg.get("dry_counter", 0) + 1
            cand = _materialize(reg, prop, round_no, cid, dry=dry_run)
            state.save(reg)
            consumed = _cascade(reg, cand, prop, phase, seeds,
                                screen_floor=(0.6 * best_score_before
                                              if screen else None),
                                budget=budget, dry_run=dry_run)
            state.save(reg)
            if budget["grades"] <= 0 and not dry_run:
                _log("evaluator budget exhausted; stopping round early")
                break

        best_after = archives.best_id(reg)
        best_score_after = (reg["candidates"].get(best_after, {})
                            .get("score") or best_score_before)
        gain = best_score_after / max(best_score_before, 1e-9)
        improved = best_score_after > best_score_before * PLATEAU_EPS
        if (best_after and best_after != best_before
                and best_score_after > best_score_before):
            # move head to the new incumbent; mark it kept
            bm = reg["candidates"][best_after]
            if bm.get("status") in ("evaluated",):
                bm["status"] = "kept"
                try:
                    cobj = Candidate.load(best_after)
                    cobj.status = "kept"
                    cobj.save()
                except Exception:
                    pass
            reg["head"] = best_after
        s["plateau_rounds"] = 0 if improved else s.get("plateau_rounds", 0) + 1
        if s["plateau_rounds"] >= PLATEAU_EXPLORE_AT:
            s["explore_ratio"] = min(0.8, 0.34 + 0.1 * (
                s["plateau_rounds"] - PLATEAU_EXPLORE_AT + 1))
        if s["plateau_rounds"] >= PLATEAU_NEW_OP_AT:
            s["need_new_operator"] = True
        s["best_score"] = best_score_after
        s["best_id"] = best_after
        if best_after:
            s["bottleneck"] = _bottleneck(reg, best_after)
        history.append({
            "round": round_no, "proposals": len(props),
            "best": best_after, "best_score": best_score_after,
            "improved": improved, "gain": round(gain, 4),
        })
        _append_log(
            f"\n### round {round_no} summary: best={best_after} "
            f"score={best_score_after:.3f} gain={gain:.4f} "
            f"plateau={s['plateau_rounds']} explore={s['explore_ratio']:.2f}\n")
        write_search_state(reg)
        state.save(reg)
        _log(f"round {round_no}: best={best_after} "
             f"score={best_score_after:.3f} gain={gain:.4f} "
             f"plateau={s['plateau_rounds']}")
        if budget["grades"] <= 0 and not dry_run:
            break
    if s_backup is not None:
        reg["search"] = s_backup
        state.save(reg)
    return {"history": history, "search": s,
            "candidates": len(reg["candidates"])}
