"""Elite / Pareto / diversity archives over evaluated candidates.

Stored inside registry.json under "archives"; only correct candidates enter.
"""

ELITE_K = 8


def _correct(meta: dict) -> bool:
    return meta.get("score") is not None and meta.get("failure_class") in (
        None, "none")


def update(reg: dict, cid: str) -> None:
    """Recompute archives after candidate `cid` got metrics."""
    reg.setdefault("archives", {"elite": [], "pareto": [], "diversity": {}})
    arch = reg["archives"]
    cands = reg["candidates"]
    scored = {c: m for c, m in cands.items() if _correct(m)}

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
        if cycles(m)[1] < best_d1:
            front.append(c)
            best_d1 = cycles(m)[1]
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
    return elite[0] if elite else reg.get("head")
