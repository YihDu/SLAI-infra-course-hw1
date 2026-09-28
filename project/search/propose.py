"""Proposal generator: DSL operators in, candidate plans out.

Separation of concerns:
  * structure = categorical op set (order, residency, fusion, prefetch) — the
    agent's structural hypothesis space, encoded as named templates;
  * parameters = numeric tile choices inside a fixed structure — explored by a
    deterministic grid, never requiring compiler edits.

Priority = best-first over archived parents with a novelty bonus for untried
structural signatures; stagnation cooling is temporary, never permanent.
"""

import itertools
import json
import random
from pathlib import Path

from . import dsl
from .candidate import EXPERIMENTS, Candidate

STRUCTURES = {
    # name -> (family, categorical ops). Numeric tiles filled by the grid.
    "mnk":          ("memory",    [{"op": "gemm_order", "order": "mnk"}]),
    "nkm":          ("memory",    [{"op": "gemm_order", "order": "nkm"}]),
    "nkm_pf":       ("memory",    [{"op": "gemm_order", "order": "nkm"},
                                   {"op": "gemm_prefetch", "enabled": 1}]),
    "scores_rf":    ("attention", [{"op": "scores_residency", "space": "rf"}]),
    "fuse_gelu":    ("fusion",    [{"op": "fuse_bias_gelu", "enabled": 1}]),
    "nkm_rf":       ("attention", [{"op": "gemm_order", "order": "nkm"},
                                   {"op": "scores_residency", "space": "rf"}]),
    "nkm_fuse":     ("fusion",    [{"op": "gemm_order", "order": "nkm"},
                                   {"op": "fuse_bias_gelu", "enabled": 1}]),
    "nkm_pf_rf":    ("memory",    [{"op": "gemm_order", "order": "nkm"},
                                   {"op": "gemm_prefetch", "enabled": 1},
                                   {"op": "scores_residency", "space": "rf"}]),
    "stage_p1_prompt_m16": ("compute", [{"op": "stage_gemm_tiles",
                                         "stage": "p1_prompt", "m_tile": 16}]),
    "stage_d1_n32": ("compute", [{"op": "stage_gemm_tiles",
                                  "stage": "d1", "n_tile": 32,
                                  "k_chunk": 32}]),
}


def register_structure(name: str, family: str, ops: list[dict]) -> None:
    STRUCTURES[name] = (family, ops)


def legal_tiles(order: str) -> list[tuple[int, int, int]]:
    """Grid of (m_tile, n_tile, k_chunk) passing static recipe validation."""
    out = []
    for m, n, k in itertools.product(
            dsl.M_TILE_CHOICES, dsl.N_TILE_CHOICES, dsl.K_CHUNK_CHOICES):
        r = json.loads(json.dumps(dsl.DEFAULT_RECIPE))
        r["gemm"].update(order=order, m_tile=m, n_tile=n, k_chunk=k)
        if not dsl.validate_recipe(r):
            out.append((m, n, k))
    return out


def _load_parent_inputs(cid: str) -> tuple[dict, dict]:
    pdir = EXPERIMENTS / cid / "inputs"
    rfile = pdir / "recipe.json"
    if rfile.exists():
        recipe = json.loads(rfile.read_text())
    else:
        recipe = json.loads(json.dumps(
            dsl.LEGACY_RECIPES.get(cid, dsl.DEFAULT_RECIPE)))
    hw = json.loads((pdir / "hardware.json").read_text())
    return recipe, hw


def struct_sig(ops: list[dict]) -> str:
    """Categorical identity of an op set (numeric params excluded)."""
    parts = []
    for op in ops:
        if op["op"] == "gemm_order":
            parts.append(op["order"])
        elif op["op"] == "gemm_tiles":
            continue
        elif op["op"] == "stage_gemm_tiles":
            parts.append(f"stage:{op['stage']}")
        elif op["op"] == "gemm_prefetch":
            parts.append("pf" if op.get("enabled", 1) else "nopf")
        elif op["op"] == "scores_residency":
            parts.append(f"s:{op['space']}")
        elif op["op"] == "fuse_bias_gelu":
            parts.append("fuse" if op.get("enabled", 1) else "nofuse")
        elif op["op"] == "hw_set":
            parts.append(f"hw:{op.get('path')}={op.get('value')}")
        else:
            parts.append(op["op"])
    return "+".join(sorted(parts)) or "base"


def _tried_fingerprints(reg: dict) -> set:
    return set(reg.get("fingerprints", {}))


def _meta(reg, cid):
    return reg["candidates"].get(cid, {})


def _family_fail_streak(reg: dict, sig: str) -> int:
    """Consecutive evaluated children of this sig that failed/not-improved."""
    streak = 0
    for cid, m in sorted(reg["candidates"].items()):
        if m.get("struct_sig") != sig:
            continue
        if m.get("failure_class") in (None, "none") and m.get("improved"):
            streak = 0
        else:
            streak += 1
    return streak


def _mk_ops(order, tiles, extras):
    ops = [{"op": "gemm_order", "order": order}]
    if tiles:
        ops.append({"op": "gemm_tiles",
                    "m_tile": tiles[0], "n_tile": tiles[1], "k_chunk": tiles[2]})
    ops += extras
    return ops


def propose_round(reg: dict, round_no: int, count: int,
                  explore_ratio: float, phase: int,
                  rng: random.Random) -> list[dict]:
    """Emit `count` proposals: ~3 exploit best, ~2 expand archives, ~1 novelty."""
    from .archives import best_id
    best = best_id(reg)
    tried = _tried_fingerprints(reg)
    proposals = []

    def emit(parent, ops, family, bias, kind, hypothesis):
        recipe, hw = _load_parent_inputs(parent)
        try:
            recipe, hw = dsl.apply_ops(recipe, hw, ops, phase=phase)
        except dsl.DslError:
            return False
        if dsl.validate_recipe(recipe, hw) or dsl.validate_hardware(hw):
            return False
        fp = dsl.fingerprint(recipe, hw)
        if fp in tried:
            return False
        tried.add(fp)
        proposals.append({
            "parent": parent, "ops": ops, "family": family, "bias": bias,
            "kind": kind, "hypothesis": hypothesis,
            "recipe": recipe, "hardware": hw, "fingerprint": fp,
            "struct_sig": struct_sig(ops),
        })
        return True

    n_exploit = max(1, count - int(round(count * explore_ratio)) - 1)
    n_expand = count - n_exploit - 1

    # --- exploit: single-param neighbourhood of the incumbent ---
    base_recipe, base_hw = _load_parent_inputs(best)
    g = base_recipe["gemm"]
    base_tiles = (g["m_tile"], g["n_tile"], g["k_chunk"])
    order = g["order"]
    extras = []
    if g.get("prefetch"):
        extras.append({"op": "gemm_prefetch", "enabled": 1})
    if base_recipe["attention"]["scores"] == "rf":
        extras.append({"op": "scores_residency", "space": "rf"})
    if base_recipe["fusion"]["bias_gelu"]:
        extras.append({"op": "fuse_bias_gelu", "enabled": 1})
    neighbours = []
    for axis, idx in (("m_tile", 0), ("n_tile", 1), ("k_chunk", 2)):
        choices = {"m_tile": dsl.M_TILE_CHOICES, "n_tile": dsl.N_TILE_CHOICES,
                   "k_chunk": dsl.K_CHUNK_CHOICES}[axis]
        if base_tiles[idx] not in choices:
            continue
        cur = choices.index(base_tiles[idx])
        for step in (-1, 1, -2, 2):
            j = cur + step
            if 0 <= j < len(choices):
                t = list(base_tiles)
                t[idx] = choices[j]
                neighbours.append(tuple(t))
    # phase>=2: also one-field hardware steps around the incumbent
    hw_nbrs = []
    if phase >= 2:
        for field, menu in dsl.HW_MENU.items():
            cur_v = base_hw.get(field)
            if cur_v not in menu:
                continue
            ci = menu.index(cur_v)
            for step in (-1, 1):
                j = ci + step
                if 0 <= j < len(menu):
                    hw_nbrs.append((field, menu[j]))
    rng.shuffle(neighbours)
    rng.shuffle(hw_nbrs)
    nbr_iter = iter(neighbours)
    hw_iter = iter(hw_nbrs)

    def exploit_next() -> bool:
        for t in nbr_iter:
            if emit(best, _mk_ops(order, t, extras),
                    _meta(reg, best).get("family", "memory"),
                    _meta(reg, best).get("bias", "balanced"), "exploit",
                    f"exploit {best}: tiles {base_tiles}->{t}"):
                return True
        return False

    def exploit_hw_next() -> bool:
        for field, val in hw_iter:
            if emit(best, [{"op": "hw_set", "path": field, "value": val}],
                    "hardware", "balanced", "exploit",
                    f"exploit {best}: hw {field} {base_hw.get(field)}->{val}"):
                return True
        return False

    for _ in range(n_exploit):
        # alternate recipe-tile and hardware neighbours for efficiency
        if not (exploit_next() or exploit_hw_next()):
            break
        if len(proposals) < n_exploit and phase >= 2 and rng.random() < 0.5:
            exploit_hw_next()

    # --- expand: mutate other promising (elite/pareto/diversity) parents ---
    # With phase>=2, half the expand slots probe single-field hardware moves
    # that need no compiler change (RF ports, DMA depth/engines, cache...).
    hw_toggles = [
        ({"op": "hw_set", "path": "rf_ports", "value": "8R4W"}, "hardware",
         "widen RF ports to relieve MMA read-port bound"),
        ({"op": "hw_set", "path": "dma_depth", "value": 2}, "memory",
         "dma_depth=2 to make prefetch actually overlap"),
        ({"op": "hw_set", "path": "dma_engines", "value": 2}, "memory",
         "second DMA engine"),
        ({"op": "hw_set", "path": "reduction_units", "value": 1}, "compute",
         "reduction unit for LN/softmax reduces"),
        ({"op": "hw_set", "path": "cache_mib", "value": 2}, "memory",
         "2MiB cache behind HBM"),
        ({"op": "hw_set", "path": "hbm_channels", "value": 8}, "memory",
         "double HBM channels"),
        ({"op": "hw_set", "path": "tc_array", "value": "8x8"}, "compute",
         "wider TC array"),
        ({"op": "hw_set", "path": "tc_k_parallel", "value": 2}, "compute",
         "k-parallel TC"),
        ({"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 128},
         "memory", "wider SM NoC link"),
    ]
    arch = reg.get("archives", {})
    parents = []
    pool = (arch.get("pareto", []) + arch.get("elite", [])
            + list(arch.get("diversity", {}).values())
            # fall back to every scored candidate while archives are young
            + [c for c, m in reg["candidates"].items()
               if m.get("score") is not None
               and m.get("status") not in ("dominated", "pruned", "incorrect")])
    for cid in pool:
        if cid != best and cid in reg["candidates"] and cid not in parents:
            parents.append(cid)
    added = 0
    for parent in parents:
        if added >= n_expand:
            break
        pr, phw = _load_parent_inputs(parent)
        pg = pr["gemm"]
        # recipe structural toggles + (phase>=2) single-field hw probes
        toggles = [
            ({"op": "scores_residency", "space": "rf"}, "attention",
             f"expand {parent}: keep attention scores in RF"),
            ({"op": "fuse_bias_gelu", "enabled": 1}, "fusion",
             f"expand {parent}: fuse bias+gelu epilogue"),
            ({"op": "gemm_prefetch", "enabled": 1}, "memory",
             f"expand {parent}: double-buffer B tiles"),
        ]
        if phase >= 2:
            toggles += [
                (dict(op), fam, f"expand {parent}: {hyp} "
                 f"[{op['path']}={op['value']}]")
                for op, fam, hyp in hw_toggles
                if phw.get(op["path"]) != op["value"]
            ]
        rng.shuffle(toggles)
        for op, fam, hyp in toggles:
            already = (
                (op["op"] == "scores_residency" and pr["attention"]["scores"] == "rf")
                or (op["op"] == "fuse_bias_gelu" and pr["fusion"]["bias_gelu"])
                or (op["op"] == "gemm_prefetch" and pg.get("prefetch"))
                or (op["op"] == "hw_set" and phw.get(op["path"]) == op["value"]))
            if already:
                continue
            if emit(parent, [op], fam, _meta(reg, parent).get("bias", "balanced"),
                    "expand", hyp):
                added += 1
                break

    # --- novelty: prefer untried structural signatures, then random legal ---
    sigs_tried = {m.get("struct_sig") for c, m in reg["candidates"].items()
                  if not c.startswith("drun-")}
    novelty_done = False
    candidates_structs = [
        (s, fam, ops) for s, (fam, ops) in STRUCTURES.items()
        if struct_sig(ops) not in sigs_tried
    ]
    rng.shuffle(candidates_structs)
    for sname, fam, ops in candidates_structs:
        if emit(best, ops, fam, "balanced", "novelty",
                f"novelty: untried structure {sname} on {best}"):
            novelty_done = True
            break
    while not novelty_done:
        order = rng.choice(dsl.GEMM_ORDERS)
        tiles = rng.choice(legal_tiles(order))
        ops = _mk_ops(order, tiles, [
            {"op": "gemm_prefetch", "enabled": rng.randint(0, 1)},
            {"op": "scores_residency",
             "space": rng.choice(dsl.SCORES_SPACES)},
            {"op": "fuse_bias_gelu", "enabled": rng.randint(0, 1)},
        ])
        if phase >= 2 and rng.random() < 0.5:
            field = rng.choice(list(dsl.HW_MENU))
            ops.append({"op": "hw_set", "path": field,
                        "value": rng.choice(list(dsl.HW_MENU[field]))})
        if emit(best, ops, "memory", "balanced", "novelty",
                f"novelty: random {order} tiles={tiles}"):
            novelty_done = True

    # --- top-up: keep filling toward `count` with remaining neighbours ---
    while len(proposals) < count and exploit_next():
        pass

    return proposals[:count]
