"""Elite / Pareto / diversity archives over evaluated candidates.

Stored inside registry.json under "archives"; only correct candidates enter.
"""

ELITE_K = 8


def is_admissible(meta: dict) -> bool:
    """Return True if candidate metadata qualifies for archive inclusion.

    Only admits explicit eligible successful functional metrics metadata;
    preserves legacy valid entries via failure_class none/score where no new fields exist.
    """
    if not isinstance(meta, dict):
        return False
    score = meta.get("score")
    if score is None:
        return False
    failure_class = meta.get("failure_class")
    if failure_class not in (None, "none"):
        return False
    status = meta.get("status")
    if status in ("incorrect", "pruned", "ineligible", "not_verified", "created", "evaluating"):
        return False

    has_eligible = "eligible" in meta
    has_functional = "functional_passed" in meta or "functional" in meta

    if has_eligible or has_functional:
        if meta.get("eligible") is not True:
            return False
        functional = meta.get("functional")
        if isinstance(functional, dict):
            return (set(functional) == {"M1_P1", "M2_D1"}
                    and all(v is True for v in functional.values())
                    and meta.get("functional_passed", True) is True)
        return meta.get("functional_passed") is True

    # Legacy valid entries without explicit eligible / functional fields
    return True


def _correct(meta: dict) -> bool:
    """Backward compatibility alias for is_admissible."""
    return is_admissible(meta)


def update(reg: dict, cid: str) -> None:
    """Recompute archives after candidate cid got metrics."""
    reg.setdefault("archives", {"elite": [], "pareto": [], "diversity": {}})
    arch = reg["archives"]
    cands = reg.get("candidates", {})
    scored = {c: m for c, m in cands.items() if is_admissible(m)}

    arch["elite"] = [
        c for c, _ in sorted(
            scored.items(), key=lambda kv: -kv[1]["score"])[:ELITE_K]
    ]

    # Pareto front on (p1_cycles, d1_cycles) — both lower is better.
    def cycles(m):
        c = m.get("cycles") or {}
        return (c.get("M1_P1", 1 << 62), c.get("M2_D1", 1 << 62))

    pts = sorted(scored.items(), key=lambda kv: cycles(kv[1]))
    front = []
    best_d1 = 1 << 62
    for c, m in pts:
        p1, d1 = cycles(m)
        if d1 < best_d1:
            front.append(c)
            best_d1 = d1
    arch["pareto"] = front

    # Diversity: best scorer per (family, bias) cell and per structural sig.
    div = {}
    for c, m in sorted(scored.items(), key=lambda kv: -kv[1]["score"]):
        cell = f"{m.get('family','?')}|{m.get('bias','?')}"
        sig = m.get("struct_sig") or cell
        div.setdefault(cell, c)
        div.setdefault(f"sig:{sig}", c)
    arch["diversity"] = div


def best_id(reg: dict) -> str | None:
    elite = reg.get("archives", {}).get("elite") or []
    if elite:
        return elite[0]
    head = reg.get("head")
    if head and head in reg.get("candidates", {}):
        if is_admissible(reg["candidates"][head]):
            return head
    return None
