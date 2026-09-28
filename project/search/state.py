"""Registry of all candidates: ids, parents, statuses, current head.

The registry only tracks bookkeeping; the evidence lives in each candidate
directory and in the append-only project/iteration-log.md.
"""

import json
from pathlib import Path

from .candidate import EXPERIMENTS

REGISTRY = EXPERIMENTS / "registry.json"


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"head": None, "counter": 0, "candidates": {}}


def save(reg: dict) -> None:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


def next_id(reg: dict) -> str:
    reg["counter"] += 1
    while f"cand-{reg['counter']:03d}" in reg["candidates"]:
        reg["counter"] += 1
    return f"cand-{reg['counter']:03d}"


def status_line(reg: dict) -> str:
    rows = []
    for cid, meta in sorted(reg["candidates"].items()):
        score = meta.get("score")
        tag = meta.get("struct_sig") or meta.get("kind") or "-"
        fail = meta.get("failure_class")
        rows.append(
            f"  {cid}  parent={meta.get('parent') or '-':<9} "
            f"r{meta.get('round') or 0} {meta.get('kind') or '-':<8} "
            f"status={meta.get('status'):<10} "
            f"score={score if score is not None else '-':<8} "
            f"fail={fail or '-':<16} {tag:<20} "
            f"{meta.get('hypothesis', '')[:50]}"
        )
    s = reg.get("search", {})
    head = f"head={reg['head']} search={s}"
    return head + "\n" + "\n".join(rows)
