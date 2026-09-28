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
    # Compiler changes follow the parent branch unless explicitly overridden.
    if prop.get("compiler_path"):
        shutil.copy2(prop["compiler_path"], cand.inputs / "compiler.py")
    else:
        shutil.copy2(parent_dir / "compiler.py", cand.inputs / "compiler.py")
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
    """Resource evidence for both cases; busy times overlap, not a critical path."""
    try:
        metrics = json.loads((EXPERIMENTS / cid / "metrics.json").read_text())
        cases = metrics["reports"]["grade"]["cases"]
        summaries = []
        for case, values in cases.items():
            busy = values.get("busy_cycles", {})
            resources = [(k, v) for k, v in busy.items()
                         if isinstance(v, (int, float)) and not k.endswith(("issue_wait", ".issue"))]
            top = sorted(resources, key=lambda kv: -kv[1])[:3]
            summaries.append(case + " " + ", ".join(f"{k}={v}" for k, v in top))
        return "; ".join(summaries) + " (overlapping service counts)"
    except (OSError, KeyError, TypeError):
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
               screen: bool = False, phase: int | None = None, *,
               jobs: int = 2, validation: str = "grade",
               timeout_seconds: float = 600, memory_mib: int = 3072,
               wall_time_seconds: float = 7200) -> dict:
    from .coordinator import Budget, evaluate_many, recover
    from .cache import input_fingerprint
    reg = state.load()
    if not dry_run:
        recover(reg)
    _backfill(reg)
    s = reg.setdefault("search", {"phase": 2, "round": 0, "plateau_rounds": 0,
                                 "explore_ratio": .34, "need_new_operator": False})
    phase = phase if phase is not None else s.get("phase", 2)
    s["phase"] = phase
    budget = Budget(max_grades, wall_time_seconds)
    history = []
    rng = random.Random(1000 + s.get("round", 0))
    for _ in range(rounds):
        if not dry_run and (not budget.available() or s.get("need_new_operator")):
            break
        round_no = s["round"] + 1
        before = reg["candidates"][reg["head"]].get("score") or 0
        diagnostics = {}
        props = propose.propose_round(reg, round_no, per_round,
                                      s.get("explore_ratio", .34), phase, rng,
                                      diagnostics=diagnostics)
        if dry_run:
            history.append({"round": round_no, "proposals": props, "diagnostics": diagnostics})
            s["round"] = round_no
            continue
        if not props:
            history.append({"round": round_no, "diagnostics": diagnostics, "reason": "space_exhausted"})
            break
        candidates = []
        for prop in props:
            cid = state.next_id(reg)
            cand = _materialize(reg, prop, round_no, cid)
            reg["candidates"][cid]["input_fingerprint"] = input_fingerprint(cand)
            candidates.append(cand)
        state.save(reg)
        _log(f"round {round_no}: {len(candidates)} proposals; grade budget {budget.used}/{budget.maximum}")
        evaluate_many(reg, candidates, seeds, budget, jobs=jobs, validation=validation,
                      timeout_seconds=timeout_seconds, memory_mib=memory_mib)
        best = reg["head"]
        after = reg["candidates"][best].get("score") or before
        s["round"] = round_no
        s["plateau_rounds"] = 0 if after > before * PLATEAU_EPS else s.get("plateau_rounds", 0) + 1
        s["need_new_operator"] = s["plateau_rounds"] >= PLATEAU_NEW_OP_AT
        s.update(best_id=best, best_score=after, bottleneck=_bottleneck(reg, best))
        row = {"round": round_no, "proposals": len(props), "best": best,
               "best_score": after, "diagnostics": diagnostics, "budget": budget.summary()}
        history.append(row)
        _append_log(f"\n### round {round_no} summary: {json.dumps(row)}\n")
        write_search_state(reg)
        state.save(reg)
    return {"history": history, "search": s, "budget": budget.summary(),
            "candidates": len(reg["candidates"])}
