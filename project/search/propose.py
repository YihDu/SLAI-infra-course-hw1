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
import hashlib
from pathlib import Path

from . import dsl
from .candidate import EXPERIMENTS, REPO, Candidate

structure_signature = dsl.structure_signature

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
    "p1_w1_shared": ("memory", [{"op": "w1_shared_panel", "enabled": 1}]),
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


def struct_sig(ops_or_recipe, hw: dict | None = None) -> str:
    """Categorical identity of effective (recipe, hw) or fallback op set."""
    if isinstance(ops_or_recipe, dict):
        return dsl.structure_signature(ops_or_recipe, hw)
    parts = []
    for op in ops_or_recipe:
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


def stage_tile_neighborhood(recipe: dict, hw: dict | None = None) -> list[dict]:
    """Generate valid 1-step and 2-step stage_gemm_tiles ops around current effective stage tiles."""
    g = recipe.get("gemm", {})
    stages = ("d1", "p1_prompt", "p1_step")
    nbr_ops = []
    seen = set()
    eval_hw = hw or dsl.BASELINE_HW

    for stage in stages:
        cur_stage = recipe.get("gemm_stages", {}).get(stage, {})
        cur_m = cur_stage.get("m_tile", g.get("m_tile", 8))
        cur_n = cur_stage.get("n_tile", g.get("n_tile", 16))
        cur_k = cur_stage.get("k_chunk", g.get("k_chunk", 48))

        # Single-axis variations
        axes = [("n_tile", cur_n, dsl.N_TILE_CHOICES),
                ("k_chunk", cur_k, dsl.K_CHUNK_CHOICES)]
        if stage == "p1_prompt":
            axes.append(("m_tile", cur_m, dsl.M_TILE_CHOICES))

        for axis, cur_val, choices in axes:
            if cur_val not in choices:
                continue
            idx = choices.index(cur_val)
            for step in (-1, 1, -2, 2):
                j = idx + step
                if 0 <= j < len(choices):
                    op = {"op": "stage_gemm_tiles", "stage": stage, axis: choices[j]}
                    try:
                        test_recipe, _ = dsl.apply_ops(recipe, eval_hw, [op])
                        if not dsl.validate_recipe(test_recipe, eval_hw):
                            key = (stage, axis, choices[j])
                            if key not in seen:
                                seen.add(key)
                                nbr_ops.append(op)
                    except dsl.DslError:
                        pass

        # 2D (n_tile, k_chunk) variations
        for n_step in (-1, 1, -2, 2):
            if cur_n in dsl.N_TILE_CHOICES:
                ni = dsl.N_TILE_CHOICES.index(cur_n) + n_step
                if 0 <= ni < len(dsl.N_TILE_CHOICES):
                    new_n = dsl.N_TILE_CHOICES[ni]
                    for k_step in (-1, 1, -2, 2):
                        if cur_k in dsl.K_CHUNK_CHOICES:
                            ki = dsl.K_CHUNK_CHOICES.index(cur_k) + k_step
                            if 0 <= ki < len(dsl.K_CHUNK_CHOICES):
                                new_k = dsl.K_CHUNK_CHOICES[ki]
                                op = {"op": "stage_gemm_tiles", "stage": stage,
                                      "n_tile": new_n, "k_chunk": new_k}
                                try:
                                    test_recipe, _ = dsl.apply_ops(recipe, eval_hw, [op])
                                    if not dsl.validate_recipe(test_recipe, eval_hw):
                                        key = (stage, "nk", new_n, new_k)
                                        if key not in seen:
                                            seen.add(key)
                                            nbr_ops.append(op)
                                except dsl.DslError:
                                    pass

    return nbr_ops


def _compiler_identity(cid):
    path = EXPERIMENTS / cid / "inputs/compiler.py"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _tried_fingerprints(reg: dict) -> set:
    tried = set()
    for cid in reg.get("candidates", {}):
        if cid.startswith("drun-"):
            continue
        try:
            recipe, hw = _load_parent_inputs(cid)
            tried.add(dsl.fingerprint(recipe, hw, compiler=_compiler_identity(cid)))
        except (OSError, ValueError):
            continue
    return tried


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
                  rng: random.Random,
                  diagnostics: dict | None = None) -> list[dict]:
    """Emit proposals strictly meeting quotas without truncation stealing novelty."""
    from .archives import best_id
    best = best_id(reg)
    if not best:
        best = next(iter(reg.get("candidates", {})), "cand-000")
    tried = _tried_fingerprints(reg)

    if diagnostics is not None:
        diagnostics.setdefault("rejections", [])
        diagnostics.setdefault("rejection_reasons", {})
        diagnostics.setdefault("attempts", 0)

    def record_rejection(reason: str, kind: str, ops: list[dict], parent: str):
        if diagnostics is not None:
            diagnostics["rejections"].append(reason)
            # Use base reason key for summary counts
            base_reason = reason.split(":")[0]
            diagnostics["rejection_reasons"][base_reason] = (
                diagnostics["rejection_reasons"].get(base_reason, 0) + 1
            )

    attempts = 0
    exploit_proposals = []
    expand_proposals = []
    novelty_proposals = []

    def emit(parent: str, ops: list[dict], family: str, bias: str, kind: str,
             hypothesis: str, target_list: list[dict], target_quota: int) -> bool:
        nonlocal attempts
        if attempts >= 128:
            return False
        attempts += 1
        if diagnostics is not None:
            diagnostics["attempts"] = attempts
        if len(target_list) >= target_quota:
            record_rejection("quota_full", kind, ops, parent)
            return False
        try:
            recipe, hw = _load_parent_inputs(parent)
        except Exception as e:
            record_rejection(f"parent_load_error: {e}", kind, ops, parent)
            return False
        try:
            recipe, hw = dsl.apply_ops(recipe, hw, ops, phase=phase)
        except dsl.DslError as e:
            record_rejection(f"dsl_error: {e}", kind, ops, parent)
            return False
        r_errs = dsl.validate_recipe(recipe, hw)
        if r_errs:
            record_rejection(f"recipe_invalid: {'; '.join(r_errs)}", kind, ops, parent)
            return False
        hw_errs = dsl.validate_hardware(hw)
        if hw_errs:
            record_rejection(f"hardware_invalid: {'; '.join(hw_errs)}", kind, ops, parent)
            return False
        compiler_path = None
        if recipe.get("gemm_kernels", {}).get("p1_prompt_w1"):
            compiler_path = str(REPO / "project" / "compiler.py")
        compiler_id = (hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()
                       if compiler_path else _compiler_identity(parent))
        fp = dsl.fingerprint(recipe, hw, compiler=compiler_id)
        if fp in tried:
            record_rejection(f"duplicate_fingerprint: {fp}", kind, ops, parent)
            return False
        tried.add(fp)
        sig = dsl.structure_signature(recipe, hw)
        target_list.append({
            "parent": parent, "ops": ops, "family": family, "bias": bias,
            "kind": kind, "hypothesis": hypothesis,
            "recipe": recipe, "hardware": hw, "fingerprint": fp,
            "struct_sig": sig, "compiler_path": compiler_path,
        })
        return True

    # Quota allocation
    if count == 6:
        target_exploit = 2
        target_expand = 2
        target_novelty = 2
    else:
        target_novelty = max(1, int(round(count * explore_ratio)))
        rem = count - target_novelty
        target_expand = max(1, rem // 2) if rem > 1 else max(0, rem)
        target_exploit = max(0, count - target_novelty - target_expand)

    # --- 1. EXPLOIT: single-param neighbourhood of incumbent ---
    base_recipe, base_hw = _load_parent_inputs(best)
    g = base_recipe["gemm"]
    base_tiles = (g["m_tile"], g["n_tile"], g["k_chunk"])
    order = g["order"]
    extras = []
    if g.get("prefetch"):
        extras.append({"op": "gemm_prefetch", "enabled": 1})
    if base_recipe.get("attention", {}).get("scores") == "rf":
        extras.append({"op": "scores_residency", "space": "rf"})
    if base_recipe.get("fusion", {}).get("bias_gelu"):
        extras.append({"op": "fuse_bias_gelu", "enabled": 1})

    gemm_nbrs = []
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
                gemm_nbrs.append((_mk_ops(order, tuple(t), extras),
                                 _meta(reg, best).get("family", "memory"),
                                 f"tiles {base_tiles}->{tuple(t)}"))

    stage_nbrs = [
        ([op], "compute", f"stage tile {op['stage']}")
        for op in stage_tile_neighborhood(base_recipe)
    ]

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
                    hw_nbrs.append(([{"op": "hw_set", "path": field, "value": menu[j]}],
                                    "hardware", f"hw {field} {cur_v}->{menu[j]}"))

    rng.shuffle(gemm_nbrs)
    rng.shuffle(stage_nbrs)
    rng.shuffle(hw_nbrs)

    all_exploit = []
    max_exploit_len = max(len(gemm_nbrs), len(stage_nbrs), len(hw_nbrs), 1)
    for i in range(max_exploit_len):
        if i < len(gemm_nbrs):
            all_exploit.append(gemm_nbrs[i])
        if i < len(stage_nbrs):
            all_exploit.append(stage_nbrs[i])
        if i < len(hw_nbrs):
            all_exploit.append(hw_nbrs[i])

    exploit_attempts = 0
    for ops, fam, desc in all_exploit:
        if len(exploit_proposals) >= target_exploit or exploit_attempts >= 128:
            break
        exploit_attempts += 1
        emit(best, ops, fam, _meta(reg, best).get("bias", "balanced"),
             "exploit", f"exploit {best}: {desc}",
             exploit_proposals, target_exploit)

    # --- 2. EXPAND: mutate archive parents ---
    hw_toggles = [
        ({"op": "hw_set", "path": "rf_ports", "value": "8R4W"}, "hardware",
         "widen RF ports to relieve MMA read-port bound"),
        ({"op": "hw_set", "path": "dma_depth", "value": 2}, "memory",
         "dma_depth=2 to make prefetch overlap"),
        ({"op": "hw_set", "path": "dma_engines", "value": 2}, "memory",
         "second DMA engine"),
        ({"op": "hw_set", "path": "reduction_units", "value": 1}, "compute",
         "reduction unit for LN/softmax reduces"),
        ({"op": "hw_set", "path": "cache_mib", "value": 2}, "memory",
         "2MiB cache behind HBM"),
        ({"op": "hw_set", "path": "hbm_channels", "value": 8}, "memory",
         "double HBM channels"),
        ({"op": "hw_set", "path": "tc_array", "value": "8x8"}, "compute",
         "wider TC array 8x8"),
        ({"op": "hw_set", "path": "tc_array", "value": "8x16"}, "compute",
         "wider TC array 8x16"),
        ({"op": "hw_set", "path": "tc_k_parallel", "value": 2}, "compute",
         "k-parallel TC (2)"),
        ({"op": "hw_set", "path": "tc_k_parallel", "value": 4}, "compute",
         "k-parallel TC (4)"),
        ({"op": "hw_set", "path": "sm_noc_bytes_per_cycle", "value": 128},
         "memory", "wider SM NoC link"),
        ({"op": "hw_set", "path": "sfu_lanes", "value": 16}, "compute",
         "wider SFU lanes"),
        ({"op": "hw_set", "path": "shared_banks", "value": 8}, "memory",
         "more shared memory banks"),
        ({"op": "hw_set", "path": "shared_ports", "value": "2R1W"}, "memory",
         "faster shared memory ports"),
        ({"op": "hw_set", "path": "multicast", "value": True}, "memory",
         "enable multicast"),
    ]

    arch = reg.get("archives", {})
    parents = []
    pool = (arch.get("pareto", []) + arch.get("elite", [])
            + list(arch.get("diversity", {}).values())
            + [c for c, m in reg.get("candidates", {}).items()
               if m.get("score") is not None
               and m.get("status") not in ("dominated", "pruned", "incorrect")])
    for cid in pool:
        if cid != best and cid in reg.get("candidates", {}) and cid not in parents:
            parents.append(cid)
    if not parents:
        parents = [best]

    expand_attempts = 0
    for parent in parents:
        if len(expand_proposals) >= target_expand or expand_attempts >= 128:
            break
        pr, phw = _load_parent_inputs(parent)
        pg = pr.get("gemm", {})
        toggles = [
            ([{"op": "scores_residency", "space": "rf"}], "attention",
             f"expand {parent}: keep attention scores in RF"),
            ([{"op": "fuse_bias_gelu", "enabled": 1}], "fusion",
             f"expand {parent}: fuse bias+gelu epilogue"),
            ([{"op": "gemm_prefetch", "enabled": 1}], "memory",
             f"expand {parent}: double-buffer B tiles"),
        ]
        # Include stage tile toggles in expand
        for st_op in stage_tile_neighborhood(pr)[:4]:
            toggles.append(([st_op], "compute", f"expand {parent}: stage tile {st_op['stage']}"))

        if phase >= 2:
            toggles += [
                ([dict(op)], fam, f"expand {parent}: {hyp} [{op['path']}={op['value']}]")
                for op, fam, hyp in hw_toggles
                if phw.get(op["path"]) != op["value"]
            ]
        rng.shuffle(toggles)
        for op_list, fam, hyp in toggles:
            if len(expand_proposals) >= target_expand or expand_attempts >= 128:
                break
            # Skip if parent already configured with this setting
            already = False
            for op in op_list:
                if op["op"] == "scores_residency" and pr.get("attention", {}).get("scores") == op["space"]:
                    already = True
                elif op["op"] == "fuse_bias_gelu" and pr.get("fusion", {}).get("bias_gelu") == op["enabled"]:
                    already = True
                elif op["op"] == "gemm_prefetch" and pg.get("prefetch") == op["enabled"]:
                    already = True
                elif op["op"] == "hw_set" and phw.get(op["path"]) == op["value"]:
                    already = True
            if already:
                record_rejection("already_configured", "expand", op_list, parent)
                continue

            expand_attempts += 1
            emit(parent, op_list, fam, _meta(reg, parent).get("bias", "balanced"),
                 "expand", hyp, expand_proposals, target_expand)

    # --- 3. NOVELTY: prefer untried structural signatures, then random legal ---
    sigs_tried = {m.get("struct_sig") for c, m in reg.get("candidates", {}).items()
                  if not c.startswith("drun-")}
    for p in exploit_proposals + expand_proposals:
        sigs_tried.add(p["struct_sig"])

    candidates_structs = [
        (s, fam, ops) for s, (fam, ops) in STRUCTURES.items()
        if struct_sig(ops) not in sigs_tried
    ]
    rng.shuffle(candidates_structs)
    novelty_attempts = 0
    for sname, fam, ops in candidates_structs:
        if len(novelty_proposals) >= target_novelty or novelty_attempts >= 128:
            break
        novelty_attempts += 1
        emit(best, ops, fam, "balanced", "novelty",
             f"novelty: untried structure {sname} on {best}",
             novelty_proposals, target_novelty)

    while len(novelty_proposals) < target_novelty and novelty_attempts < 128:
        novelty_attempts += 1
        order = rng.choice(dsl.GEMM_ORDERS)
        tiles = rng.choice(legal_tiles(order))
        extra_ops = [
            {"op": "gemm_prefetch", "enabled": rng.randint(0, 1)},
            {"op": "scores_residency", "space": rng.choice(dsl.SCORES_SPACES)},
            {"op": "fuse_bias_gelu", "enabled": rng.randint(0, 1)},
        ]
        if rng.random() < 0.5:
            stg = rng.choice(["p1_prompt", "d1", "p1_step"])
            if stg == "p1_prompt":
                extra_ops.append({"op": "stage_gemm_tiles", "stage": stg,
                                  "m_tile": rng.choice([8, 16, 32])})
            else:
                extra_ops.append({"op": "stage_gemm_tiles", "stage": stg,
                                  "n_tile": rng.choice([16, 32, 64]),
                                  "k_chunk": rng.choice([16, 32, 64])})
        ops = _mk_ops(order, tiles, extra_ops)
        if phase >= 2 and rng.random() < 0.5:
            field = rng.choice(list(dsl.HW_MENU))
            val = rng.choice(list(dsl.HW_MENU[field]))
            ops.append({"op": "hw_set", "path": field, "value": val})
        emit(best, ops, "memory", "balanced", "novelty",
             f"novelty: random {order} tiles={tiles}",
             novelty_proposals, target_novelty)

    if diagnostics is not None:
        diagnostics["quota_requested"] = {"exploit": target_exploit, "expand": target_expand, "novelty": target_novelty}
        diagnostics["quota_filled"] = {"exploit": len(exploit_proposals), "expand": len(expand_proposals), "novelty": len(novelty_proposals)}
    # Proposals assembled strictly respecting quotas; NO truncation stealing novelty
    proposals = exploit_proposals + expand_proposals + novelty_proposals
    return proposals
