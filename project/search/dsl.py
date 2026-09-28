"""Search DSL: structured operators over (recipe, hardware) design points.

Phase-1 candidates mutate only the recipe (`project/recipe.json`); the compiler
snapshot supplies the operator implementations. Structural ops that are not yet
implemented in the compiler can be represented here but are gated by phase.

A proposal is:
    {"parent": str, "hypothesis": str, "family": FAMILY, "bias": BIAS,
     "kind": "exploit"|"expand"|"novelty",
     "ops": [{"op": NAME, ...params}]}
"""

import copy
import json

FAMILIES = ("memory", "parallelism", "compute", "fusion", "attention", "hardware")
BIASES = ("p1", "d1", "balanced")

# Phase gate: 1 = recipe-only dataflow; 2 = + single-field hardware;
# 3 = + workgroup partition/placement (needs compiler support).
PHASE_UNLOCKED_OPS = {
    1: {"gemm_order", "gemm_tiles", "gemm_prefetch", "scores_residency",
        "fuse_bias_gelu"},
    2: {"gemm_order", "gemm_tiles", "gemm_prefetch", "scores_residency",
        "fuse_bias_gelu", "hw_set", "stage_gemm_tiles"},
    3: {"gemm_order", "gemm_tiles", "gemm_prefetch", "scores_residency",
        "fuse_bias_gelu", "hw_set", "workgroups", "stage_gemm_tiles"},
}

# Legal hardware menu values (mirrors codesign/challenge/hardware.py).
HW_MENU = {
    "sm_count": (4, 8, 12, 16, 20, 24, 28, 32),
    "tc_count": (0, 1, 2, 3, 4, 6, 8),
    "vector_lanes": (8, 16, 32, 64),
    "reduction_units": (0, 1, 2),
    "rf_ports": ("2R1W", "4R2W", "8R4W"),
    "shared_kib": (0, 32, 64, 96, 128, 192, 256),
    "dma_depth": (0, 1, 2),
    "dma_engines": (1, 2, 4),
    "sm_noc_bytes_per_cycle": (32, 64, 128),
    "noc_bytes_per_cycle": (64, 128, 256, 512),
    "hbm_channels": (1, 2, 4, 8),
    "cache_mib": (0, 1, 2, 4, 8, 16),
}
AREA_LIMIT_MM2 = 24.0

DEFAULT_RECIPE = {
    "gemm": {"order": "nkm", "m_tile": 8, "n_tile": 16, "k_chunk": 48,
             "prefetch": 0},
    "attention": {"scores": "hbm"},
    "fusion": {"bias_gelu": 0},
}

# Legacy candidates evaluated before the DSL existed; their effective design
# points are known exactly (asm verified byte-identical to these recipes).
LEGACY_RECIPES = {
    "cand-000": {"gemm": {"order": "mnk", "m_tile": 8, "n_tile": 16,
                          "k_chunk": 48, "prefetch": 0},
                 "attention": {"scores": "hbm"}, "fusion": {"bias_gelu": 0}},
    "cand-001": {"gemm": {"order": "nkm", "m_tile": 8, "n_tile": 16,
                          "k_chunk": 48, "prefetch": 0},
                 "attention": {"scores": "hbm"}, "fusion": {"bias_gelu": 0}},
}

# RF lane capacity (FP32 words per lane) at the baseline vector_lanes=16.
LANE_WORDS = 1024
# Worst-case m_extent per case: P1 prompt GEMMs run m=64, D1 runs m=1.
MAX_M = 64
GEMM_ORDERS = ("mnk", "nkm")
SCORES_SPACES = ("hbm", "rf")
# n_tile must divide every n_extent used by P1 (768,256,1024) and D1 (384,128,512).
N_TILE_CHOICES = (8, 16, 32, 64, 128)
M_TILE_CHOICES = (1, 2, 4, 8, 16, 32, 64)
K_CHUNK_CHOICES = (8, 16, 24, 32, 48, 64, 96, 128)


class DslError(ValueError):
    pass


class OperatorLocked(DslError):
    pass


def _op_gemm_order(recipe, _hw, op):
    order = op.get("order")
    if order not in GEMM_ORDERS:
        raise DslError(f"gemm_order: bad order {order!r}")
    recipe["gemm"]["order"] = order


def _op_gemm_tiles(recipe, _hw, op):
    for key, choices in (("m_tile", M_TILE_CHOICES),
                         ("n_tile", N_TILE_CHOICES),
                         ("k_chunk", K_CHUNK_CHOICES)):
        if key in op:
            v = op[key]
            if not isinstance(v, int) or v not in choices:
                raise DslError(f"gemm_tiles: {key}={v!r} not in {choices}")
            recipe["gemm"][key] = v


def _op_stage_gemm_tiles(recipe, _hw, op):
    stage = op.get("stage")
    if stage not in ("p1_prompt", "p1_step", "d1"):
        raise DslError(f"stage_gemm_tiles: bad stage {stage!r}")
    override = recipe.setdefault("gemm_stages", {}).setdefault(stage, {})
    for key, choices in (("m_tile", M_TILE_CHOICES),
                         ("n_tile", N_TILE_CHOICES),
                         ("k_chunk", K_CHUNK_CHOICES)):
        if key in op:
            value = op[key]
            if type(value) is not int or value not in choices:
                raise DslError(f"stage_gemm_tiles: {key}={value!r} not in {choices}")
            override[key] = value
    if not override:
        raise DslError("stage_gemm_tiles: no tile field")


def _op_gemm_prefetch(recipe, _hw, op):
    recipe["gemm"]["prefetch"] = int(bool(op.get("enabled", 1)))


def _op_scores_residency(recipe, _hw, op):
    space = op.get("space")
    if space not in SCORES_SPACES:
        raise DslError(f"scores_residency: bad space {space!r}")
    recipe["attention"]["scores"] = space


def _op_fuse_bias_gelu(recipe, _hw, op):
    recipe["fusion"]["bias_gelu"] = int(bool(op.get("enabled", 1)))


def _op_hw_set(_recipe, hw, op):
    # Phase-2 operator: single-field hardware mutation (flat fields only).
    path = op.get("path")
    value = op.get("value")
    if path not in HW_MENU:
        raise DslError(f"hw_set: unknown/unsupported field {path!r}")
    if value not in HW_MENU[path]:
        raise DslError(f"hw_set: {path}={value!r} not in {HW_MENU[path]}")
    hw[path] = value


def _op_workgroups(_recipe, _hw, _op):
    # Phase-2 placeholder: partition/placement lives in compiler structure.
    raise DslError("workgroups operator not implemented in compiler yet")


OPERATORS = {
    "gemm_order": _op_gemm_order,
    "gemm_tiles": _op_gemm_tiles,
    "stage_gemm_tiles": _op_stage_gemm_tiles,
    "gemm_prefetch": _op_gemm_prefetch,
    "scores_residency": _op_scores_residency,
    "fuse_bias_gelu": _op_fuse_bias_gelu,
    "hw_set": _op_hw_set,
    "workgroups": _op_workgroups,
}


def register_operator(name: str, fn) -> None:
    """Agents may register new structural operators here."""
    OPERATORS[name] = fn


def check_phase(ops: list[dict], phase: int) -> None:
    unlocked = PHASE_UNLOCKED_OPS[phase]
    for op in ops:
        if op["op"] not in OPERATORS:
            raise DslError(f"unknown operator {op['op']!r}")
        if op["op"] not in unlocked:
            raise OperatorLocked(f"operator {op['op']!r} locked at phase {phase}")


def apply_ops(base_recipe: dict, base_hw: dict, ops: list[dict],
              phase: int = 1) -> tuple[dict, dict]:
    check_phase(ops, phase)
    recipe = copy.deepcopy(base_recipe)
    hw = copy.deepcopy(base_hw)
    for op in ops:
        OPERATORS[op["op"]](recipe, hw, op)
    # Canonicalize: fields the compiler ignores must not create distinct
    # fingerprints (e.g. prefetch is only implemented in the nkm path).
    if recipe["gemm"]["order"] != "nkm":
        recipe["gemm"]["prefetch"] = 0
    return recipe, hw


def lane_words(hw: dict | None = None) -> int:
    """RF words per lane = 64KiB private slot / vector_lanes."""
    vl = (hw or {}).get("vector_lanes", 16)
    return (64 * 1024 // 4) // vl


def validate_recipe(recipe: dict, hw: dict | None = None) -> list[str]:
    """Static resource/divisibility pruning. Returns reasons; empty = legal."""
    errs = []
    g = recipe["gemm"]
    order, m_t, n_t, k_c = (g["order"], g["m_tile"], g["n_tile"], g["k_chunk"])
    lw = lane_words(hw)
    if order == "nkm":
        # All m accumulators for one n-tile are simultaneously live in lane 2.
        if MAX_M * n_t > lw:
            errs.append(f"nkm: m_extent*n_tile={MAX_M*n_t}>{lw}")
    elif order == "mnk":
        if m_t * n_t > lw:
            errs.append(f"mnk: m_tile*n_tile={m_t*n_t}>{lw}")
    else:
        errs.append(f"unknown order {order!r}")
    if k_c * n_t > lw:
        errs.append(f"B tile k_chunk*n_tile={k_c*n_t}>{lw}")
    if m_t * k_c > lw:
        errs.append(f"A tile m_tile*k_chunk={m_t*k_c}>{lw}")
    # scores live in lane 4; worst case row length = 136 (D1 past+8).
    if recipe["attention"]["scores"] == "rf" and 136 > lw:
        errs.append(f"scores_rf: 136>{lw} lane words")
    # m_tile<=2 explodes expanded instruction count (cand-033 pathology).
    if MAX_M // max(m_t, 1) > 32:
        errs.append(f"m_tile={m_t}: >32 m-iterations, instruction explosion")
    if recipe["attention"]["scores"] not in SCORES_SPACES:
        errs.append("bad scores space")
    for stage, override in recipe.get("gemm_stages", {}).items():
        if stage not in ("p1_prompt", "p1_step", "d1"):
            errs.append(f"bad GEMM stage {stage!r}")
            continue
        if set(override) - {"m_tile", "n_tile", "k_chunk"}:
            errs.append(f"unsupported GEMM stage fields: {set(override) - {'m_tile', 'n_tile', 'k_chunk'}}")
            continue
        effective = {**g, **override}
        st_m, st_n, st_k = (effective[k] for k in ("m_tile", "n_tile", "k_chunk"))
        max_rows = 64 if stage == "p1_prompt" else 1
        if effective["order"] == "nkm" and max_rows * st_n > lw:
            errs.append(f"{stage}: accumulators exceed RF lane")
        if st_m * st_k > lw or st_k * st_n > lw:
            errs.append(f"{stage}: operand tile exceeds RF lane")
        if max_rows // max(st_m, 1) > 32:
            errs.append(f"{stage}: too many m iterations")
    return errs


def validate_hardware(hw: dict) -> list[str]:
    """Menu + area pruning using the evaluator's own cost model."""
    errs = []
    try:
        from codesign.challenge.hardware import Hardware
        h = Hardware.from_dict(hw)
        h.validate()
    except Exception as e:
        return [f"hardware invalid: {e}"]
    area = h.area_mm2()
    if area > AREA_LIMIT_MM2:
        errs.append(f"area {area:.2f}mm2 > {AREA_LIMIT_MM2}")
    return errs


def fingerprint(recipe: dict, hw: dict) -> str:
    blob = json.dumps({"r": recipe, "h": hw}, sort_keys=True)
    import hashlib
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def recipe_diff(a: dict, b: dict) -> dict:
    """Flat diff of two recipes: {dotted.path: [old, new]}."""
    out = {}

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{p}{k}.")
        elif x != y:
            out[p[:-1]] = [x, y]

    walk(a, b, "")
    return out
