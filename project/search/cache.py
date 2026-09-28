"""Execution artifact cache and content-addressed execution keys.

Prevents redundant expensive simulator runs by hashing:
  1. Canonical hardware configuration
  2. Emitted assembly programs (M1_P1.asm, M2_D1.asm)
  3. Frozen evaluator tree and baseline manifest
  4. Pinned Python / NumPy runtime environment
  5. Validation mode and seed set

Only complete, eligible, functionally correct evaluations may be stored or reused.
Cache hits link the authoritative source candidate rather than fabricating fake raw reports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .candidate import EXPERIMENTS, REPO

DEFAULT_CACHE_FILE = EXPERIMENTS / "execution_cache.json"
BASELINE_MANIFEST_PATH = REPO / "baseline_manifest.json"
CODESIGN_DIR = REPO / "codesign"

PINNED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.5.3",
}

_CACHED_EVALUATOR_TREE_HASH: str | None = None
_CACHED_BASELINE_MANIFEST_HASH: str | None = None


@lru_cache(maxsize=1)
def runtime_identity():
    from .evaluator import python_exe
    output = subprocess.check_output([python_exe(), '-c',
        'import sys,numpy,json; print(json.dumps({"python":".".join(map(str,sys.version_info[:3])),"numpy":numpy.__version__}))'], text=True)
    return json.loads(output)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def get_evaluator_tree_hash(codesign_dir: Path | None = None) -> str:
    """Compute deterministic SHA-256 over all files in the codesign tree."""
    global _CACHED_EVALUATOR_TREE_HASH
    target_dir = codesign_dir or CODESIGN_DIR
    if codesign_dir is None and _CACHED_EVALUATOR_TREE_HASH is not None:
        return _CACHED_EVALUATOR_TREE_HASH

    if not target_dir.exists():
        digest = sha256_bytes(b"empty_codesign_tree")
    else:
        entries = []
        for file_path in sorted(target_dir.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts and file_path.suffix != ".pyc":
                rel = file_path.relative_to(target_dir).as_posix()
                entries.append(f"{rel}:{sha256_file(file_path)}")
        digest = sha256_bytes("\n".join(entries).encode("utf-8"))

    if codesign_dir is None:
        _CACHED_EVALUATOR_TREE_HASH = digest
    return digest


def get_baseline_manifest_hash(manifest_path: Path | None = None) -> str:
    """Compute deterministic SHA-256 of the baseline manifest."""
    global _CACHED_BASELINE_MANIFEST_HASH
    target = manifest_path or BASELINE_MANIFEST_PATH
    if manifest_path is None and _CACHED_BASELINE_MANIFEST_HASH is not None:
        return _CACHED_BASELINE_MANIFEST_HASH

    if not target.exists():
        digest = sha256_bytes(b"missing_baseline_manifest")
    else:
        digest = sha256_file(target)

    if manifest_path is None:
        _CACHED_BASELINE_MANIFEST_HASH = digest
    return digest


def input_fingerprint(cand: Any) -> str:
    """Produce deterministic hash of candidate inputs: compiler, recipe, hardware, deps.

    Captures:
      - compiler.py (bytes)
      - hardware.json (canonical JSON)
      - recipe.json (canonical JSON if present)
      - any other staged files in cand.inputs
      - declared candidate dependencies if any
    """
    inputs_dir = getattr(cand, "inputs", None)
    hashes: dict[str, str] = {}

    if isinstance(inputs_dir, Path) and inputs_dir.exists():
        for item in sorted(inputs_dir.iterdir()):
            if item.is_file():
                if item.suffix == ".json":
                    try:
                        parsed = json.loads(item.read_text(encoding="utf-8"))
                        hashes[item.name] = sha256_bytes(canonical_json_bytes(parsed))
                    except Exception:
                        hashes[item.name] = sha256_file(item)
                else:
                    hashes[item.name] = sha256_file(item)

    # Include declared dependencies if candidate defines them
    deps = getattr(cand, "declared_dependencies", None) or getattr(cand, "dependencies", None)
    if deps is not None:
        hashes["_declared_dependencies"] = sha256_bytes(canonical_json_bytes(deps))

    items = [f"{name}:{h}" for name, h in sorted(hashes.items())]
    return sha256_bytes(";".join(items).encode("utf-8"))


def execution_key(
    cand: Any,
    seeds: list[int] | None = None,
    validation: str = "grade",
    codesign_dir: Path | None = None,
    baseline_path: Path | None = None,
) -> str:
    """Compute content-addressed execution key for simulator output equivalence.

    Includes:
      - canonical hardware JSON
      - generated assembly hashes (M1_P1.asm, M2_D1.asm)
      - evaluator tree and baseline manifest hashes
      - pinned Python / NumPy runtime environment versions
      - validation mode ("grade", "check", "estimate")
      - seeds list
    """
    for required in (cand.inputs / "hardware.json", cand.programs / "M1_P1.asm", cand.programs / "M2_D1.asm"):
        if not required.is_file():
            raise FileNotFoundError(required)
    # 1. Canonical hardware hash
    hw_hash = "missing"
    inputs_dir = getattr(cand, "inputs", None)
    hw_path = (inputs_dir / "hardware.json") if isinstance(inputs_dir, Path) else None
    if hw_path and hw_path.exists():
        try:
            hw_obj = json.loads(hw_path.read_text(encoding="utf-8"))
            hw_hash = sha256_bytes(canonical_json_bytes(hw_obj))
        except Exception:
            hw_hash = sha256_file(hw_path)
    elif (REPO / "hardware.json").exists():
        try:
            hw_obj = json.loads((REPO / "hardware.json").read_text(encoding="utf-8"))
            hw_hash = sha256_bytes(canonical_json_bytes(hw_obj))
        except Exception:
            hw_hash = sha256_file(REPO / "hardware.json")

    # 2. Generated assembly hashes
    asm_hashes: dict[str, str] = {}
    programs_dir = getattr(cand, "programs", None)
    for name in ("M1_P1.asm", "M2_D1.asm"):
        p = (programs_dir / name) if isinstance(programs_dir, Path) else None
        if p and p.exists():
            asm_hashes[name] = sha256_file(p)
        elif (REPO / "programs" / name).exists():
            asm_hashes[name] = sha256_file(REPO / "programs" / name)
        else:
            asm_hashes[name] = "missing"

    # 3. Evaluator tree and baseline manifest
    eval_tree_hash = get_evaluator_tree_hash(codesign_dir)
    manifest_hash = get_baseline_manifest_hash(baseline_path)

    # 4. Canonical seeds
    canon_seeds = sorted(int(s) for s in (seeds or []))

    # Combine into canonical key dictionary
    key_dict = {
        "asm_hashes": asm_hashes,
        "baseline_manifest_hash": manifest_hash,
        "evaluator_tree_hash": eval_tree_hash,
        "hardware_hash": hw_hash,
        "mode": validation,
        "runtime": runtime_identity(),
        "entrypoint": sha256_file(REPO / "challenge.py"),
        "seeds": canon_seeds,
    }
    return sha256_bytes(canonical_json_bytes(key_dict))


def _load_cache(cache_path: Path | None = None) -> dict:
    target = cache_path or DEFAULT_CACHE_FILE
    if target.exists():
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(data: dict, cache_path: Path | None = None) -> None:
    target = cache_path or DEFAULT_CACHE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.tmp-",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def is_cacheable(metrics: dict) -> bool:
    """Return True only if evaluation is complete, eligible, and functionally correct."""
    if not isinstance(metrics, dict):
        return False
    reports = metrics.get("reports", {})
    grade = reports.get("grade", {}) if isinstance(reports, dict) else {}
    if not isinstance(grade, dict):
        return False

    # Check score presence
    score = grade.get("score") if grade.get("score") is not None else metrics.get("score")
    if score is None:
        return False

    # Must be eligible
    eligible = grade.get("eligible") if "eligible" in grade else metrics.get("eligible")
    if eligible is not True:
        return False

    # Must pass functional on all cases
    cases = grade.get("cases", {})
    if not isinstance(cases, dict) or set(cases) != {"M1_P1", "M2_D1"}:
        return False
    for case, c_data in cases.items():
        if not isinstance(c_data, dict) or c_data.get("functional_passed") is not True:
            return False

    verdict = metrics.get("verdict")
    if verdict not in (None, "evaluated"):
        return False

    failure_class = metrics.get("failure_class")
    if failure_class not in (None, "none"):
        return False

    return True


def lookup(key: str, cache_path: Path | None = None) -> dict | None:
    """Look up execution key; returns metrics copy or None.

    Only complete evaluated eligible true functional true reports are reused.
    Enriches metrics with source candidate linkage rather than fabricating raw reports.
    """
    cache = _load_cache(cache_path)
    entry = cache.get(key)
    if not entry or not isinstance(entry, dict):
        return None

    entry_metrics = entry.get("metrics")
    if not entry_metrics or not is_cacheable(entry_metrics):
        return None

    reused = copy.deepcopy(entry_metrics)
    reused["cached"] = True
    reused["cache_hit"] = True
    reused["cache_key"] = key
    reused["source_candidate"] = entry.get("candidate_id")
    reused["cached_from"] = entry.get("candidate_id")
    return reused


def store(key: str, cand: Any, metrics: dict, cache_path: Path | None = None) -> bool:
    """Store evaluated result into cache under key.

    Only complete evaluated eligible true functional true reports are stored.
    Returns True if stored, False if rejected by qualification gates.
    """
    if not is_cacheable(metrics):
        return False

    cache = _load_cache(cache_path)
    cid = getattr(cand, "id", None) or str(cand)
    score = (
        metrics.get("reports", {}).get("grade", {}).get("score")
        if isinstance(metrics.get("reports"), dict)
        else metrics.get("score")
    )

    cache[key] = {
        "candidate_id": cid,
        "eligible": True,
        "functional_passed": True,
        "metrics": copy.deepcopy(metrics),
        "score": score,
        "status": "evaluated",
        "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_cache(cache, cache_path)
    return True


def link_source_candidate(cand: Any, source_candidate_id: str) -> None:
    """Link candidate to source candidate artifacts without fabricating raw reports."""
    reports_dir = getattr(cand, "reports", None)
    if isinstance(reports_dir, Path) and reports_dir.exists():
        ref_file = reports_dir / "cached_from.json"
        ref_data = {
            "cached_from": source_candidate_id,
            "linked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Reused evaluation artifacts; authoritative raw reports live on source candidate.",
        }
        ref_file.write_text(json.dumps(ref_data, indent=2, sort_keys=True) + "\n")



def index_history(seeds=(7,), validation='grade'):
    """Import only raw reports whose frozen provenance matches current artifacts."""
    from .candidate import Candidate
    from .evaluator import summarize
    from codesign.challenge.hardware import Hardware
    from codesign.challenge.runner import provenance, seed_digest
    indexed, skipped = [], []
    for report_path in sorted(EXPERIMENTS.glob('cand-*/reports/grade.json')):
        cid = report_path.parent.parent.name
        try:
            cand = Candidate.load(cid)
            raw = json.loads(report_path.read_text())
            if validation != 'grade':
                raise ValueError('historical full validation not imported')
            if raw.get('baseline_manifest_sha256') != sha256_file(BASELINE_MANIFEST_PATH):
                raise ValueError('baseline mismatch')
            if raw.get('seed_sha256') != [seed_digest(s) for s in seeds]:
                raise ValueError('seed mismatch')
            if any(raw.get('runtime_environment', {}).get(k) != v for k, v in runtime_identity().items()):
                raise ValueError('runtime mismatch')
            hardware = Hardware.from_dict(json.loads((cand.inputs / 'hardware.json').read_text()))
            programs = {name: (cand.programs / (name+'.asm')).read_text() for name in ('M1_P1','M2_D1')}
            if raw.get('provenance') != provenance(hardware, programs):
                raise ValueError('artifact or frozen evaluator provenance mismatch')
            metrics = {'candidate': cid, 'reports': {'grade': summarize(raw)}, 'verdict': 'evaluated',
                       'historical_import': True}
            key = execution_key(cand, list(seeds), validation)
            if store(key, cand, metrics): indexed.append(cid)
            else: skipped.append({'candidate': cid, 'reason': 'not eligible and correct'})
        except (ValueError, KeyError, OSError, TypeError) as exc:
            skipped.append({'candidate': cid, 'reason': str(exc)})
    return {'indexed': indexed, 'skipped': skipped}
