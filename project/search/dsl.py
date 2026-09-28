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
        "fuse_bias_gelu", "stage_gemm_tiles"},
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
    "sfu_lanes": tuple(range(1, 65)),
    "reduction_units": (0, 1, 2),
    "rf_ports": ("2R1W", "4R2W", "8R4W"),
    "shared_kib": (0, 32, 64, 96, 128, 192, 256),
    "shared_banks": (1, 2, 4, 8),
    "shared_ports": ("1R1W", "2R1W"),
    "dma_depth": (0, 1, 2),
    "dma_engines": (1, 2, 4),
    "sm_noc_bytes_per_cycle": (32, 64, 128),
    "multicast": (False, True),
    "noc_bytes_per_cycle": (64, 128, 256, 512),
    "hbm_channels": (1, 2, 4, 8),
    "cache_mib": (0, 1, 2, 4, 8, 16),
    "tc_array": ("4x8", "8x8", "8x16"),
    "tc_k_parallel": (1, 2, 4),
}
AREA_LIMIT_MM2 = 24.0

BASELINE_HW = {
    "version": "challenge-hardware-v0.4",
    "sm_count": 8,
    "tc_count": 2,
    "tc_array": "4x8",
    "tc_k_parallel": 1,
    "vector_lanes": 16,
    "sfu_lanes": 8,
    "reduction_units": 0,
    "rf_ports": "4R2W",
    "shared_kib": 128,
    "shared_banks": 4,
    "shared_ports": "1R1W",
    "dma_depth": 1,
    "dma_engines": 1,
    "sm_noc_bytes_per_cycle": 64,
    "multicast": False,
    "noc_bytes_per_cycle": 256,
    "hbm_channels": 4,
    "cache_mib": 0,
}

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
    keys_found = 0
    for key, choices in (("m_tile", M_TILE_CHOICES),
                         ("n_tile", N_TILE_CHOICES),
                         ("k_chunk", K_CHUNK_CHOICES)):
        if key in op:
            value = op[key]
            if type(value) is not int or value not in choices:
                raise DslError(f"stage_gemm_tiles: {key}={value!r} not in {choices}")
            override[key] = value
            keys_found += 1
    if keys_found == 0:
        raise DslError("stage_gemm_tiles: no valid tile field provided")


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
    # Phase-2 operator: single-field hardware mutation.
    path = op.get("path")
    value = op.get("value")
    if path not in HW_MENU:
        raise DslError(f"hw_set: unknown/unsupported field {path!r}")
    if value not in HW_MENU[path]:
        raise DslError(f"hw_set: {path}={value!r} not in {HW_MENU[path]}")

    # Dependent constraints and impossible state rejection
    if path in ("tc_array", "tc_k_parallel"):
        if hw.get("tc_count", 0) == 0:
            raise DslError(f"hw_set: cannot set {path} when tc_count is 0")
    elif path in ("shared_banks", "shared_ports"):
        if hw.get("shared_kib", 0) == 0:
            raise DslError(f"hw_set: cannot set {path} when shared_kib is 0")
    elif path == "sfu_lanes":
        vl = hw.get("vector_lanes", 16)
        if not (1 <= value <= vl):
            raise DslError(f"hw_set: sfu_lanes={value} must be between 1 and vector_lanes={vl}")
    elif path == "vector_lanes":
        hw[path] = value
        if hw.get("sfu_lanes", 1) > value:
            hw["sfu_lanes"] = value
        return
    elif path == "tc_count":
        hw[path] = value
        if value == 0:
            hw.pop("tc_array", None)
            hw.pop("tc_k_parallel", None)
        else:
            if hw.get("tc_array") is None:
                hw["tc_array"] = "4x8"
            if hw.get("tc_k_parallel") is None:
                hw["tc_k_parallel"] = 1
        return
    elif path == "shared_kib":
        hw[path] = value
        if value == 0:
            hw.pop("shared_banks", None)
            hw.pop("shared_ports", None)
        else:
            if hw.get("shared_banks") is None:
                hw["shared_banks"] = 4
            if hw.get("shared_ports") is None:
                hw["shared_ports"] = "1R1W"
        return

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


def canonicalize_hardware(hw: dict) -> dict:
    """Enforce canonical dependencies on hardware dictionary."""
    h = copy.deepcopy(hw)
    tc_cnt = h.get("tc_count", 0)
    if tc_cnt == 0:
        h.pop("tc_array", None)
        h.pop("tc_k_parallel", None)
    else:
        if h.get("tc_array") is None:
            h["tc_array"] = "4x8"
        if h.get("tc_k_parallel") is None:
            h["tc_k_parallel"] = 1

    sh_kib = h.get("shared_kib", 0)
    if sh_kib == 0:
        h.pop("shared_banks", None)
        h.pop("shared_ports", None)
    else:
        if h.get("shared_banks") is None:
            h["shared_banks"] = 4
        if h.get("shared_ports") is None:
            h["shared_ports"] = "1R1W"

    if "vector_lanes" in h and "sfu_lanes" in h:
        vl = h["vector_lanes"]
        if h["sfu_lanes"] > vl:
            h["sfu_lanes"] = vl
        elif h["sfu_lanes"] < 1:
            h["sfu_lanes"] = 1
    return h


def apply_ops(base_recipe: dict, base_hw: dict, ops: list[dict],
              phase: int = 1) -> tuple[dict, dict]:
    check_phase(ops, phase)
    recipe = copy.deepcopy(base_recipe)
    hw = copy.deepcopy(base_hw)
    for op in ops:
        OPERATORS[op["op"]](recipe, hw, op)
    # Canonicalize: fields the compiler ignores must not create distinct
    # fingerprints (e.g. prefetch is only implemented in the nkm path).
    if recipe.get("gemm", {}).get("order") != "nkm":
        if "gemm" in recipe and "prefetch" in recipe["gemm"]:
            recipe["gemm"]["prefetch"] = 0
    hw = canonicalize_hardware(hw)
    return recipe, hw


def lane_words(hw: dict | None = None) -> int:
    """RF words per lane = 64KiB private slot / vector_lanes."""
    vl = (hw or {}).get("vector_lanes", 16)
    return (64 * 1024 // 4) // vl


def validate_recipe(recipe: dict, hw: dict | None = None) -> list[str]:
    """Static resource/divisibility pruning. Returns reasons; empty = legal."""
    errs = []
    if "gemm" not in recipe:
        return ["missing gemm in recipe"]
    g = recipe["gemm"]
    order = g.get("order")
    m_t = g.get("m_tile", 0)
    n_t = g.get("n_tile", 0)
    k_c = g.get("k_chunk", 0)
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
    if recipe.get("attention", {}).get("scores") == "rf" and 136 > lw:
        errs.append(f"scores_rf: 136>{lw} lane words")
    # m_tile<=2 explodes expanded instruction count (cand-033 pathology).
    if MAX_M // max(m_t, 1) > 32:
        errs.append(f"m_tile={m_t}: >32 m-iterations, instruction explosion")
    if recipe.get("attention", {}).get("scores") not in SCORES_SPACES:
        errs.append("bad scores space")
    for stage, override in recipe.get("gemm_stages", {}).items():
        if stage not in ("p1_prompt", "p1_step", "d1"):
            errs.append(f"bad GEMM stage {stage!r}")
            continue
        if not isinstance(override, dict):
            errs.append(f"stage {stage} override must be dict")
            continue
        if set(override) - {"m_tile", "n_tile", "k_chunk"}:
            errs.append(f"unsupported GEMM stage fields: {set(override) - {'m_tile', 'n_tile', 'k_chunk'}}")
            continue
        for k, v in override.items():
            choices = {"m_tile": M_TILE_CHOICES, "n_tile": N_TILE_CHOICES, "k_chunk": K_CHUNK_CHOICES}[k]
            if type(v) is not int or v not in choices:
                errs.append(f"{stage}: {k}={v!r} not in {choices}")
        effective = {**g, **override}
        st_m, st_n, st_k = (effective["m_tile"], effective["n_tile"], effective["k_chunk"])
        max_rows = 64 if stage == "p1_prompt" else 1
        if effective.get("order") == "nkm" and max_rows * st_n > lw:
            errs.append(f"{stage}: accumulators exceed RF lane")
        elif effective.get("order") == "mnk" and st_m * st_n > lw:
            errs.append(f"{stage}: accumulators exceed RF lane")
        if st_m * st_k > lw:
            errs.append(f"{stage}: operand tile A exceeds RF lane")
        if st_k * st_n > lw:
            errs.append(f"{stage}: operand tile B exceeds RF lane")
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


def fingerprint(recipe: dict, hw: dict, compiler: str | None = None,
                compiler_id: str | None = None) -> str:
    cid = compiler if compiler is not None else compiler_id
    data = {"r": recipe, "h": hw}
    if cid is not None:
        data["c"] = str(cid)
    blob = json.dumps(data, sort_keys=True)
    import hashlib
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def structure_signature(recipe: dict, hw: dict | None = None) -> str:
    parts = []
    g = recipe.get("gemm", {})
    order = g.get("order", "mnk")
    parts.append(order)
    parts.append("pf" if g.get("prefetch") else "nopf")
    scores = recipe.get("attention", {}).get("scores", "hbm")
    parts.append(f"s:{scores}")
    bias_gelu = recipe.get("fusion", {}).get("bias_gelu", 0)
    parts.append("fuse" if bias_gelu else "nofuse")
    for st in sorted(recipe.get("gemm_stages", {})):
        parts.append(f"stage:{st}")

    if hw:
        tc_cnt = hw.get("tc_count", BASELINE_HW["tc_count"])
        if tc_cnt != BASELINE_HW["tc_count"]:
            parts.append(f"hw:tc_count={tc_cnt}")
        if tc_cnt > 0:
            if hw.get("tc_array") and hw.get("tc_array") != BASELINE_HW["tc_array"]:
                parts.append(f"hw:tc_array={hw['tc_array']}")
            if hw.get("tc_k_parallel") and hw.get("tc_k_parallel") != BASELINE_HW["tc_k_parallel"]:
                parts.append(f"hw:tc_k_parallel={hw['tc_k_parallel']}")
        sh_kib = hw.get("shared_kib", BASELINE_HW["shared_kib"])
        if sh_kib != BASELINE_HW["shared_kib"]:
            parts.append(f"hw:shared_kib={sh_kib}")
        if sh_kib > 0:
            if hw.get("shared_banks") and hw.get("shared_banks") != BASELINE_HW["shared_banks"]:
                parts.append(f"hw:shared_banks={hw['shared_banks']}")
            if hw.get("shared_ports") and hw.get("shared_ports") != BASELINE_HW["shared_ports"]:
                parts.append(f"hw:shared_ports={hw['shared_ports']}")
        for k in ("sm_count", "vector_lanes", "sfu_lanes", "reduction_units",
                  "rf_ports", "dma_depth", "dma_engines", "sm_noc_bytes_per_cycle",
                  "multicast", "noc_bytes_per_cycle", "hbm_channels", "cache_mib"):
            if k in hw and hw[k] != BASELINE_HW.get(k):
                parts.append(f"hw:{k}={hw[k]}")

    return "+".join(sorted(parts)) or "base"


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
