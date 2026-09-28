"""Minimal experiment controller.

The coding agent proposes a hypothesis and edits inputs; this controller owns
the deterministic execution chain:

    candidate -> isolated generate -> screen -> grade (or explicit check+grade)
             -> parse -> metrics.json -> append iteration-log.md

Commands:
    init                                create experiments dir + registry
    new ID --parent P --hypothesis H [--summary S] [--from-worktree]
    run ID [--seed N ...]               isolated generate -> screen -> grade
    keep ID [--note ...]                accept candidate, move head
    revert ID [--note ...]              reject candidate, restore parent inputs
    status                              table of all candidates
    checkout ID                         stage a candidate's inputs to worktree

Nothing here is a search policy: choosing the next candidate is the agent's job.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import state
from .candidate import (
    INPUT_FILES,
    PROGRAM_FILES,
    REPO,
    WORKTREE_PATHS,
    Candidate,
    sha256,
)
from .evaluator import evaluate

LOG = REPO / "project" / "iteration-log.md"
BASELINE_CYCLES = {"M1_P1": 30996995, "M2_D1": 1302032}


def _append_log(text: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text)


def cmd_init(_args) -> None:
    reg = state.load()
    state.save(reg)
    print(f"registry at {state.REGISTRY}")


def cmd_new(args) -> None:
    reg = state.load()
    cid = args.id or state.next_id(reg)
    from .candidate import EXPERIMENTS
    if (EXPERIMENTS / cid).exists():
        raise SystemExit(f"candidate {cid} already exists")
    if args.parent and args.parent not in reg["candidates"]:
        raise SystemExit(f"unknown parent {args.parent}")
    cand = Candidate(id=cid, parent=args.parent,
                     hypothesis=args.hypothesis, summary=args.summary or "")
    cand.save()
    # Start from a complete parent/worktree snapshot, then apply explicit overrides.
    if args.from_worktree or not args.parent:
        cand.snapshot_from_worktree()
    else:
        parent = Candidate.load(args.parent)
        for name in ("compiler.py", "hardware.json", "recipe.json"):
            src = parent.inputs / name
            if src.exists():
                shutil.copy2(src, cand.inputs / name)
    for name, path in (("compiler.py", args.compiler), ("hardware.json", args.hardware)):
        if path:
            shutil.copy2(Path(path), cand.inputs / name)
    for name in INPUT_FILES:
        if not (cand.inputs / name).exists():
            raise SystemExit(f"candidate {cid} missing input {name}")
    if args.checkout:
        cand.stage_to_worktree()
    reg["candidates"][cid] = {
        "parent": args.parent, "status": "created",
        "hypothesis": args.hypothesis,
    }
    state.save(reg)
    print(f"created {cid} (parent={args.parent})")
    if args.checkout:
        print(f"staged {cid} inputs to worktree")


def cmd_run(args) -> None:
    from .coordinator import Budget, evaluate_many, recover
    from .engine import write_search_state
    reg = state.load()
    recover(reg)
    cand = Candidate.load(args.id)
    budget = Budget(args.max_grades, args.wall_time_seconds)
    result = evaluate_many(reg, [cand], args.seed or [7], budget,
                           jobs=args.jobs, validation=args.validation,
                           timeout_seconds=args.timeout_seconds,
                           memory_mib=args.memory_mib)
    write_search_state(reg)
    print(json.dumps({"id": cand.id, "budget": budget.summary(),
                      "verdict": result[0].get("verdict")}, indent=2))


def _verdict(args, verb: str) -> None:
    reg = state.load()
    cand = Candidate.load(args.id)
    if verb == "keep":
        from .archives import is_admissible
        if not is_admissible(reg["candidates"][args.id]):
            raise ValueError("Cannot keep an unverified or ineligible candidate")
    status = "kept" if verb == "keep" else "reverted"
    cand.status = status
    cand.verdict_note = args.note
    cand.save()
    meta = reg["candidates"][args.id]
    meta["status"] = status
    if status == "kept":
        reg["head"] = args.id
        _stage_candidate(cand)
    elif reg["head"] == args.id:
        reg["head"] = cand.parent
        if cand.parent:
            _stage_candidate(Candidate.load(cand.parent))
            print(f"restored parent {cand.parent} inputs to worktree")
    from . import archives, engine
    archives.update(reg, cand.id)
    best = archives.best_id(reg)
    reg.setdefault("search", {}).update(best_id=best, best_score=reg["candidates"][best].get("score") if best else None)
    state.save(reg)
    engine.write_search_state(reg)
    _append_log(
        f"\n### {cand.id} verdict: {status.upper()}"
        + (f" — {args.note}" if args.note else "")
        + "\n"
    )
    print(f"{args.id} -> {status}" + (f" (head={reg['head']})" if status == "kept" else ""))


def _stage_candidate(cand):
    cand.stage_to_worktree()
    if all((cand.programs / name).exists() for name in PROGRAM_FILES):
        for name in PROGRAM_FILES:
            (REPO / "programs" / name).write_bytes((cand.programs / name).read_bytes())
    else:
        import subprocess
        from .evaluator import python_exe
        subprocess.run([python_exe(), "-m", "project.compiler"], cwd=REPO, check=True, timeout=600)


def cmd_checkout(args) -> None:
    _stage_candidate(Candidate.load(args.id))
    print(f"staged {args.id} inputs and generated programs to worktree")


def cmd_status(_args) -> None:
    print(state.status_line(state.load()))


def cmd_graph(_args) -> None:
    """Print the candidate DAG as parent->children edges."""
    reg = state.load()
    children = {}
    for cid, m in reg["candidates"].items():
        children.setdefault(m.get("parent"), []).append(cid)
    def walk(cid, depth):
        m = reg["candidates"].get(cid, {})
        print("  " * depth + f"{cid} score={m.get('score')} "
              f"status={m.get('status')} sig={m.get('struct_sig') or '-'}")
        for c in sorted(children.get(cid, [])):
            walk(c, depth + 1)
    for root in sorted(children.get(None, [])):
        walk(root, 0)


def cmd_state_md(_args) -> None:
    from . import engine
    engine.write_search_state(state.load())
    print("wrote project/search/search_state.md")


def cmd_search_plan(args) -> None:
    from . import engine
    out = engine.run_search(rounds=args.rounds, per_round=args.per_round,
                            seeds=args.seed or [7], dry_run=True,
                            max_grades=0, phase=args.phase)
    print(json.dumps(out, indent=2))


def cmd_search_run(args) -> None:
    from . import engine
    out = engine.run_search(rounds=args.rounds, per_round=args.per_round,
                            seeds=args.seed or [7], dry_run=False,
                            max_grades=args.max_grades,
                            screen=False, phase=args.phase,
                            jobs=args.jobs, validation=args.validation,
                            timeout_seconds=args.timeout_seconds,
                            memory_mib=args.memory_mib,
                            wall_time_seconds=args.wall_time_seconds)
    print(json.dumps(out, indent=2))


def execution_options(parser, default_budget):
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-evals", "--max-grades", dest="max_grades", type=int,
                        default=default_budget, help="actual challenge launches, including failures")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--wall-time-seconds", type=float, default=7200)
    parser.add_argument("--validation", choices=("grade", "full"), default="grade")


def main(argv=None) -> int:
    from .runtime import install_signal_handlers
    install_signal_handlers()
    p = argparse.ArgumentParser(prog="project.search.controller")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    n = sub.add_parser("new")
    n.add_argument("id", nargs="?")
    n.add_argument("--parent")
    n.add_argument("--hypothesis", required=True)
    n.add_argument("--summary")
    n.add_argument("--compiler", help="path to a compiler.py snapshot")
    n.add_argument("--hardware", help="path to a hardware.json snapshot")
    n.add_argument("--from-worktree", action="store_true",
                   help="snapshot inputs from live project/compiler.py + hardware.json")
    n.add_argument("--checkout", action="store_true",
                   help="also stage the new candidate's inputs into the worktree")
    n.set_defaults(fn=cmd_new)
    r = sub.add_parser("run")
    r.add_argument("id")
    r.add_argument("--seed", type=int, action="append")
    execution_options(r, 2)
    r.set_defaults(fn=cmd_run)
    for name in ("keep", "revert"):
        c = sub.add_parser(name)
        c.add_argument("id")
        c.add_argument("--note")
        c.set_defaults(fn=lambda a, s=name: _verdict(a, s))
    c = sub.add_parser("checkout")
    c.add_argument("id")
    c.set_defaults(fn=cmd_checkout)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("graph").set_defaults(fn=cmd_graph)
    sub.add_parser("state-md").set_defaults(fn=cmd_state_md)
    sp = sub.add_parser("search-plan",
                        help="dry-run: exercise graph/archive/prune/log without evaluator")
    sp.add_argument("--rounds", type=int, default=2)
    sp.add_argument("--per-round", type=int, default=6)
    sp.add_argument("--phase", type=int, default=None)
    sp.add_argument("--seed", type=int, action="append")
    sp.set_defaults(fn=cmd_search_plan)
    sr = sub.add_parser("search-run",
                        help="real autonomous search: rounds x proposals")
    sr.add_argument("--rounds", type=int, default=5)
    sr.add_argument("--per-round", type=int, default=6)
    execution_options(sr, 30)
    sr.add_argument("--phase", type=int, default=None,
                    help="operator unlock level: 1=recipe, 2=+hw_set, 3=+workgroups")
    sr.add_argument("--seed", type=int, action="append")
    sr.set_defaults(fn=cmd_search_run)
    args = p.parse_args(argv)
    if hasattr(args, "max_grades") and (args.max_grades < 1 or args.timeout_seconds <= 0
                                       or args.memory_mib <= 0 or args.wall_time_seconds <= 0):
        p.error("evaluation budget and resource limits must be positive")
    if args.cmd in ("status", "graph"):
        args.fn(args)
    else:
        from .coordinator import writer_lock
        with writer_lock():
            args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
