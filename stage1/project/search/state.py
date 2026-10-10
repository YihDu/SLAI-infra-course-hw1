"""Registry of all candidates: ids, parents, statuses, current head.

The registry only tracks bookkeeping; the evidence lives in each candidate
directory and in the append-only project/iteration-log.md.
"""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
from pathlib import Path

from .candidate import EXPERIMENTS

REGISTRY = EXPERIMENTS / "registry.json"


def load(path: Path | None = None) -> dict:
    target = path or REGISTRY
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    return {"head": None, "counter": 0, "candidates": {}}


def save(reg: dict, path: Path | None = None) -> None:
    """Compare-and-swap under a short lock; reject stale registry snapshots.

    The coordinator lock owns long operations. This separate lock also protects
    direct callers from silently overwriting a newer snapshot.
    """
    target = path or REGISTRY
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(target.read_text()) if target.exists() else {}
        expected = reg.get("revision", 0)
        if current.get("revision", 0) != expected:
            raise RuntimeError("stale registry snapshot; reload under coordinator writer_lock")
        payload = dict(reg, revision=expected + 1)
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        tmp_fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.tmp-")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
            reg["revision"] = payload["revision"]
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


def next_id(reg: dict, experiments_dir: Path | None = None) -> str:
    exp_dir = experiments_dir or EXPERIMENTS
    reg.setdefault("counter", 0)
    while True:
        reg["counter"] += 1
        cid = f"cand-{reg['counter']:03d}"
        if cid not in reg.get("candidates", {}) and not (exp_dir / cid).exists():
            return cid


def status_line(reg: dict) -> str:
    rows = []
    for cid, meta in sorted(reg.get("candidates", {}).items()):
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
    head = f"head={reg.get('head')} search={s}"
    return head + "\n" + "\n".join(rows)
