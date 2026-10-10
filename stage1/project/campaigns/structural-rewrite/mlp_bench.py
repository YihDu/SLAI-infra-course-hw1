"""Task 1 microbenchmark: dual-batch persistent MLP panel vs current sharded MLP.

Isolates the full MLP segment for BOTH batch members and all 3 layers:

    residual_layernorm (res = x + att, ln2 = LN(res))
    -> W1 -> bias -> GELU -> W2 -> residual -> out

Variants:
  current : batch_shard replica — per-member 8-WG team, W1 col-sharded
            (paired_n kernel, bias+GELU epilogue) -> gelu HBM, W2 col-sharded
            -> ff2 HBM, add_rows(res, ff2, out, b2).
  panel   : persistent_mlp_panel — all 16 WGs own a 64-neuron panel across
            both members (rows=128).  W1/W2 weight panels staged once in SH,
            GELU stays in RF, W2 emits 16 partial slabs merged by a row-sharded
            reduction that also folds res + b2.

Timing via codesign.challenge.pipeline.estimate_pipeline (same engine as the
frozen evaluator).  Functional check via MicroMachine on a seeded fixture
(scratch NaN-poisoned to catch read-before-write).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import codesign.challenge.hbm_race as _hbm_race
import codesign.challenge.pipeline_events as _pe
from codesign.challenge.abi import build_layout
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.micro import MicroMachine
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.service import mma_service
from codesign.challenge.workload import MODELS

from project.compiler import (
    Builder, Tensor, add, mul, rf, sh, hbm, imm, load_recipe,
)

MODEL = MODELS["M1"]
D, F, LAYERS, ROWS = MODEL.width, MODEL.ffn, MODEL.layers, 128  # dual batch
PANEL = F // 16          # 64 neurons per WG
TILE = 32
SHARDED = ("current", "wave", "wave2")


def gelu_ref(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def ln_ref(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def mlp_persistent_panel(b, ln2, w1, b1, w2, part, name, panel, owner):
    """One owner WG: W1[:,panel]->bias->GELU (RF) ->W2[panel,:] partial (128x256).

    Lane map (16 x 1024 FP32 words):
      0-7   gelu panel: lane rb*2+nh = acc during W1, A operand during W2
      8-11  W1 A tile (per rb) / W2 accumulators (per rb)
      12-13 B operand buffers (nh parity / kh parity)
      14    epilogue temp tile
      15    ones[0:32], bias[64:96]
    """
    # Stage this layer's weight panels into SH once; each serves 128 rows.
    b.ld(hbm(add(w1.offset, panel), D * PANEL, [D, PANEL], [F, 1]),
         sh(0, D * PANEL))
    b.ld(hbm(add(w2.offset, mul(panel, D)), PANEL * D), sh(16384, PANEL * D))
    b.vec("add", [imm(1), imm(0)], rf(15, TILE))          # ones
    # ---- W1 -> bias -> GELU: acc lane = rb*2+nh -------------------------
    for lane in range(8):
        b.vec("add", [imm(0), imm(0)], rf(lane, TILE * TILE))
    with b.loop(name + "k", 0, D, TILE) as inner:
        with b.loop(name + "bnh", 0, 2) as nh:
            b.ld(sh(add(mul(inner, PANEL), mul(nh, TILE)), TILE * TILE,
                    [TILE, TILE], [PANEL, 1]),
                 rf(add(12, nh), TILE * TILE))
        with b.loop(name + "rb", 0, ROWS, TILE) as row:
            rb = {"ceildiv": [row, TILE]}
            b.ld(hbm(add(ln2.offset, mul(row, D), inner), TILE * TILE,
                     [TILE, TILE], [D, 1]),
                 rf(add(8, rb), TILE * TILE))
            for nh in range(2):
                b.emit("MMA.ACC",
                       a=rf(add(8, rb), TILE * TILE),
                       b=rf(add(12, nh), TILE * TILE),
                       acc=rf(add(mul(rb, 2), nh), TILE * TILE),
                       m=TILE, n=TILE, k=TILE, event=None)
    for rb in range(4):
        for nh in range(2):
            acc = rf(rb * 2 + nh, TILE * TILE)
            b.ld(hbm(add(b1.offset, panel, mul(nh, TILE)), TILE),
                 rf(15, TILE, 64))
            b.emit("MMA.ACC", a=rf(15, TILE), b=rf(15, TILE, 64), acc=acc,
                   m=TILE, n=TILE, k=1, event=None)
            t = rf(14, TILE * TILE)
            b.vec("mul", [acc, acc], t)
            b.vec("mul", [t, acc], t)
            b.vec("fma", [t, imm(0.044715), acc], t)
            b.vec("mul", [t, imm(0.7978845608028654)], t)
            b.sfu("tanh", t, t)
            b.vec("add", [t, imm(1)], t)
            b.vec("mul", [acc, imm(0.5)], acc)
            b.vec("mul", [acc, t], acc)
    # ---- W2 partial: acc[rb] (lane 8+rb) over output col chunks ---------
    base = part.offset + owner * ROWS * D
    with b.loop(name + "c", 0, D, TILE) as col:
        for rb in range(4):
            b.vec("add", [imm(0), imm(0)], rf(8 + rb, TILE * TILE))
        for kh in range(2):
            b.ld(sh(add(16384 + kh * TILE * D, col), TILE * TILE,
                    [TILE, TILE], [D, 1]), rf(12 + kh, TILE * TILE))
        for rb in range(4):
            for kh in range(2):
                b.emit("MMA.ACC",
                       a=rf(rb * 2 + kh, TILE * TILE),
                       b=rf(12 + kh, TILE * TILE),
                       acc=rf(8 + rb, TILE * TILE),
                       m=TILE, n=TILE, k=TILE, event=None)
        for rb in range(4):
            b.st(rf(8 + rb, TILE * TILE),
                 hbm(add(base, mul(rb * TILE, D), col), TILE * TILE,
                     [TILE, TILE], [D, 1]))


def mlp_persistent_panel_db(b, ln2, w1, b1, w2, part, name, panel, owner):
    """v2: same dataflow as v1, statically unrolled with double buffering.

    Lanes: 0-7 gelu acc (rb*2+nh); 8/9 W1 A parity; 10-13 B[parity*2+half];
    8-11 reused as W2 accumulators; 14 epilogue temp; 15 ones/bias.
    """
    T = TILE
    b.ld(hbm(add(w1.offset, panel), D * PANEL, [D, PANEL], [F, 1]),
         sh(0, D * PANEL))
    b.ld(hbm(add(w2.offset, mul(panel, D)), PANEL * D), sh(16384, PANEL * D))
    b.vec("add", [imm(1), imm(0)], rf(15, T))
    for lane in range(8):
        b.vec("add", [imm(0), imm(0)], rf(lane, T * T))

    def load_b1(k):
        for nh in range(2):
            b.ld(sh(k * PANEL + nh * T, T * T, [T, T], [PANEL, 1]),
                 rf(10 + (k // T % 2) * 2 + nh, T * T))

    def load_a(i):
        k, rb = (i // 4) * T, i % 4
        b.ld(hbm(ln2.offset + rb * T * D + k, T * T, [T, T], [D, 1]),
             rf(8 + i % 2, T * T))

    steps = (D // T) * 4
    load_b1(0)
    load_a(0)
    for i in range(steps):
        k, rb = (i // 4) * T, i % 4
        if rb == 0 and k + T < D:
            load_b1(k + T)
        for nh in range(2):
            b.emit("MMA.ACC", a=rf(8 + i % 2, T * T),
                   b=rf(10 + (k // T % 2) * 2 + nh, T * T),
                   acc=rf(rb * 2 + nh, T * T), m=T, n=T, k=T, event=None)
        if i + 1 < steps:
            load_a(i + 1)
    for nh in range(2):
        b.ld(hbm(b1.offset + panel + nh * T, T), rf(15, T, 64 + nh * T))
    for rb in range(4):
        for nh in range(2):
            acc = rf(rb * 2 + nh, T * T)
            b.emit("MMA.ACC", a=rf(15, T), b=rf(15, T, 64 + nh * T), acc=acc,
                   m=T, n=T, k=1, event=None)
            t = rf(14, T * T)
            b.vec("mul", [acc, acc], t)
            b.vec("mul", [t, acc], t)
            b.vec("fma", [t, imm(0.044715), acc], t)
            b.vec("mul", [t, imm(0.7978845608028654)], t)
            b.sfu("tanh", t, t)
            b.vec("add", [t, imm(1)], t)
            b.vec("mul", [acc, imm(0.5)], acc)
            b.vec("mul", [acc, t], acc)
    base = part.offset + owner * ROWS * D

    def load_b2(c):
        for kh in range(2):
            b.ld(sh(16384 + kh * T * D + c, T * T, [T, T], [D, 1]),
                 rf(10 + (c // T % 2) * 2 + kh, T * T))

    load_b2(0)
    for c in range(0, D, T):
        for rb in range(4):
            b.vec("add", [imm(0), imm(0)], rf(8 + rb, T * T))
        if c + T < D:
            load_b2(c + T)
        for rb in range(4):
            for kh in range(2):
                b.emit("MMA.ACC", a=rf(rb * 2 + kh, T * T),
                       b=rf(10 + (c // T % 2) * 2 + kh, T * T),
                       acc=rf(8 + rb, T * T), m=T, n=T, k=T, event=None)
            b.st(rf(8 + rb, T * T),
                 hbm(base + rb * T * D + c, T * T, [T, T], [D, 1]))


def merge_partials(b, part, res, b2, out, name, row_start, rows):
    """out[r,:] = res[r,:] + b2 + sum_g partial[g][r,:]; ascending g order."""
    b.ld(hbm(add(b2.offset, 0), D), rf(15, D))
    with b.loop(name + "r", row_start, row_start + rows) as row:
        ro = mul(row, D)
        b.ld(hbm(add(res.offset, ro), D), rf(0, D))
        b.ld(hbm(add(part.offset, ro), D), rf(1, D))
        b.vec("add", [rf(0, D), rf(1, D)], rf(0, D))
        for gidx in range(1, 16):
            b.ld(hbm(add(part.offset, mul(gidx, ROWS * D), ro), D), rf(1, D))
            b.vec("add", [rf(0, D), rf(1, D)], rf(0, D))
        b.vec("add", [rf(0, D), rf(15, D)], rf(0, D))
        b.st(rf(0, D), hbm(add(out.offset, ro), D))


# ---------------------------------------------------------------- programs
def build(variant, recipe):
    layout = build_layout(MODEL, "P1")
    shared = 0 if variant in SHARDED else 128 * 1024
    b = Builder(layout, recipe, groups=16, sm_map=list(range(16)),
                shared_bytes=shared)
    wgs = b.groups

    # Bench input tensors (seeded in the fixture); intermediates after them.
    x0 = x = b.alloc("x", ROWS, D)
    att = b.alloc("att", ROWS, D)
    first_scratch = b.cursor

    def join(ws):
        if len(ws) > 1:
            b.emit("BARRIER", wgs=ws, events=[])

    def place(i, stage):
        b.wg = wgs[i]
        b.label = f"{stage}s{i}_"

    for layer in range(LAYERS):
        w = {k: b.symbol(f"layer{layer}/{k}") for k in
             ("ln2_g", "ln2_b", "w1", "b1", "w2", "b2")}
        # Fresh per-layer buffers (write-once, like the real P1 allocator);
        # layer l consumes layer l-1's output.
        res = b.alloc(f"l{layer}res", ROWS, D)
        ln2 = b.alloc(f"l{layer}ln2", ROWS, D)
        gelu = b.alloc(f"l{layer}gelu", ROWS, F)
        ff2 = b.alloc(f"l{layer}ff2", ROWS, D)
        part = b.alloc(f"l{layer}partials", 16, ROWS, D)
        out = b.alloc(f"l{layer}out", ROWS, D)
        b.set_gemm_stage("p1_prompt")
        # ---- LN2 production (row-sharded) -------------------------------
        shards = 16
        for i in range(shards):
            member = i // 8
            row0 = member * 64 + (i % 8) * 8
            place(i, "resln2")
            b.residual_layernorm_cached(
                Tensor(x.offset + row0 * D, (8, D)),
                Tensor(att.offset + row0 * D, (8, D)),
                Tensor(res.offset + row0 * D, (8, D)),
                w["ln2_g"], w["ln2_b"],
                Tensor(ln2.offset + row0 * D, (8, D)),
                8, D, f"l{layer}g{i}resln2")
            if variant in SHARDED and i % 8 == 7:
                join(wgs[member * 8:(member + 1) * 8])
        if variant not in SHARDED:
            join(wgs)
        # ---- MLP ----------------------------------------------------------
        if variant in SHARDED:
            for member in range(2):
                team = wgs[member * 8:(member + 1) * 8]
                ms = member * 64
                ln2_m = Tensor(ln2.offset + ms * D, (64, D))
                gelu_m = Tensor(gelu.offset + ms * F, (64, F))
                ff2_m = Tensor(ff2.offset + ms * D, (64, D))
                if variant == "wave2":
                    # W1: 4 column passes; each pass is one team wavefront.
                    for t in range(4):
                        b.gemm_wavefront(ln2_m, w["w1"], gelu_m, 64, D, F,
                                         f"l{layer}b{member}w1t{t}", team,
                                         [F * i // 8 + 32 * t for i in range(8)],
                                         epilogue_bias=w["b1"])
                else:
                    for i in range(8):
                        place(member * 8 + i, "w1")
                        b.gemm(ln2_m, w["w1"], gelu_m, 64, D, F, f"l{layer}b{member}g{i}w1",
                               F * i // 8, F * (i + 1) // 8, epilogue_bias=w["b1"])
                join(team)
                if variant in ("wave", "wave2"):
                    b.gemm_wavefront(gelu_m, w["w2"], ff2_m, 64, F, D,
                                     f"l{layer}b{member}w2", team,
                                     [D * i // 8 for i in range(8)])
                else:
                    for i in range(8):
                        place(member * 8 + i, "w2")
                        b.gemm(gelu_m, w["w2"], ff2_m, 64, F, D, f"l{layer}b{member}g{i}w2",
                               D * i // 8, D * (i + 1) // 8)
                join(team)
                for i in range(8):
                    start = ms + 8 * i
                    place(member * 8 + i, "out")
                    b.add_rows(Tensor(res.offset + start * D, (8, D)),
                               Tensor(ff2.offset + start * D, (8, D)),
                               Tensor(out.offset + start * D, (8, D)),
                               8, D, f"l{layer}b{member}g{i}out", w["b2"])
                join(team)
        else:
            for i in range(16):
                place(i, "mlp")
                kernel = (mlp_persistent_panel if variant == "panel"
                          else mlp_persistent_panel_db)
                kernel(
                    b, ln2, w["w1"], w["b1"], w["w2"], part,
                    f"l{layer}g{i}mlp", panel=i * PANEL, owner=i)
            join(wgs)
            for i in range(16):
                place(i, "merge")
                merge_partials(b, part, res, w["b2"], out,
                               f"l{layer}g{i}merge", row_start=8 * i, rows=8)
            join(wgs)
        x = out
    for wg in wgs:
        b.emit("WG.END", wg=wg)
    return "\n".join(b.lines) + "\n", (x0, att, first_scratch, out)


# ---------------------------------------------------------------- driver
def counts(text, hw):
    mma = ld = st = vec = sfu = red = bar = 0
    rf_r = rf_w = ld_bytes = st_bytes = 0
    for ins in iter_parse(text):
        op, a = ins.op, ins.args
        if op == "MMA.ACC":
            mma += 1
            s = mma_service(hw, a["m"], a["n"], a["k"])
            rf_r += s.rf_read_bytes
            rf_w += s.rf_write_bytes
        elif op in ("LD", "ST"):
            n = a["src"]["count"] * 4
            if op == "LD":
                ld += 1
                ld_bytes += n
            else:
                st += 1
                st_bytes += n
        elif op == "VEC":
            vec += 1
            rf_r += 4 * sum(s["count"] for s in a["src"] if "space" in s)
            rf_w += 4 * a["dst"]["count"]
        elif op == "SFU":
            sfu += 1
        elif op == "REDUCE":
            red += 1
        elif op == "BARRIER":
            bar += 1
    return dict(mma=mma, ld=ld, st=st, vec=vec, sfu=sfu, red=red,
                barriers=bar, rf_read_mb=rf_r / 1e6, rf_write_mb=rf_w / 1e6,
                ld_mb=ld_bytes / 1e6, st_mb=st_bytes / 1e6)


def functional(text, tensors, hw):
    x, att, first_scratch, out = tensors
    hbm_words = out.offset + ROWS * D + 16
    h = np.zeros(hbm_words, np.float32)
    rng = np.random.default_rng(3)

    def put(off, arr):
        h[off:off + arr.size] = arr.ravel().astype(np.float32)

    put(x.offset, rng.normal(0, 0.5, (ROWS, D)))
    put(att.offset, rng.normal(0, 0.5, (ROWS, D)))
    weights = {}
    layout = build_layout(MODEL, "P1")
    for layer in range(LAYERS):
        ws = {}
        for k, shp in (("ln2_g", D), ("ln2_b", D), ("w1", (D, F)),
                       ("b1", F), ("w2", (F, D)), ("b2", D)):
            sym = layout.symbols[f"layer{layer}/{k}"]
            scale = 0.06 if k in ("w1", "w2") else 0.4
            arr = rng.normal(0, scale, shp)
            if k == "ln2_g":
                arr = 1 + arr * 0.2
            put(sym.address // 4, arr)
            ws[k] = arr
        weights[layer] = ws
    # Intermediates are invalid until written: MicroMachine raises on any
    # read-before-write inside this range.
    machine = MicroMachine(hw, h, invalid_hbm=[(first_scratch, out.offset + ROWS * D)])
    machine.execute(iter_parse(text), hbm_races_validated=True)
    got = machine.hbm[out.offset:out.offset + ROWS * D].reshape(ROWS, D)
    xr = h[x.offset:x.offset + ROWS * D].reshape(ROWS, D).astype(np.float64)
    at = h[att.offset:att.offset + ROWS * D].reshape(ROWS, D).astype(np.float64)
    for layer in range(LAYERS):
        ws = {k: v.astype(np.float64) for k, v in weights[layer].items()}
        r = xr + at
        ln = ln_ref(r, ws["ln2_g"], ws["ln2_b"])
        f1 = gelu_ref(ln @ ws["w1"] + ws["b1"])
        xr = r + f1 @ ws["w2"] + ws["b2"]
    delta = np.abs(got.astype(np.float64) - xr)
    # Ratio to the frozen tolerance 1e-3 + 1e-3*|ref|; pass iff <= 1.
    rel = float(np.nanmax(delta / (1e-3 + 1e-3 * np.abs(xr))))
    return bool(np.isfinite(got).all()), float(np.nanmax(delta)), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--skip-functional", action="store_true")
    ap.add_argument("--hw", default=None, help="JSON overrides, e.g. '{\"tc_array\":\"8x8\"}'")
    args = ap.parse_args()
    # Dev loop only: O(events*WGs) HBM race validation dominates and is
    # re-checked by the real frozen grade for any kept candidate.
    _hbm_race.validate_hbm_races = lambda *a, **k: None
    # Span-level overlap is a superset of the position-level check: it only
    # declares MORE hazards (pessimistic timing), never fewer.  Real position
    # materialisation makes estimate O(events * pending * tile_positions).
    def _fast_overlap(left, right):
        if "imm" in left or "imm" in right or left["space"] != right["space"]:
            return False
        if left["space"] != "HBM" and left["wg"] != right["wg"]:
            return False
        if left["space"] == "RF" and left["lane"] != right["lane"]:
            return False
        l_lo, l_hi = _pe._operand_span(left)
        r_lo, r_hi = _pe._operand_span(right)
        return l_lo < r_hi and r_lo < l_hi
    _pe._overlap = _fast_overlap
    # Dense hazard arrays: the sparse memmap path thrashes page cache on
    # strided tile views.  Bench programs use ~6M words -> ~100 MB dense.
    from codesign.challenge import pipeline as _pl
    _pl._MemoryTimes.sized = classmethod(
        lambda cls, words, sparse=False: _pl._MemoryTimes(
            np.zeros(words, np.int64), np.zeros(words, np.int64)))
    hw_dict = json.loads((ROOT / "hardware.json").read_text())
    hw_dict.update(json.loads(args.hw) if args.hw else {})
    hw = Hardware.from_dict(hw_dict)
    tag = "" if not args.hw else "_" + "_".join(f"{k}{v}" for k, v in json.loads(args.hw).items())
    recipe = load_recipe()
    variants = ("current", "panel2") if args.variant == "all" else (args.variant,)
    for variant in variants:
        text, tensors = build(variant, recipe)
        (ROOT / f"project/campaigns/structural-rewrite/mlp_{variant}{tag}.asm"
         ).write_text(text)
        trace = {}
        words = tensors[-1].offset + ROWS * D + 16
        res = estimate_pipeline(hw, iter_parse(text), words, None, trace)
        c = counts(text, hw)
        line = (f"[{variant}{tag}] cycles={res.cycles}  events={len(trace)}  "
                f"mma={c['mma']} rf_r={c['rf_read_mb']:.1f}MB "
                f"rf_w={c['rf_write_mb']:.1f}MB ld={c['ld']}({c['ld_mb']:.1f}MB) "
                f"st={c['st']}({c['st_mb']:.1f}MB) vec={c['vec']} "
                f"sfu={c['sfu']} barriers={c['barriers']} "
                f"hbm_r={res.hbm_read_bytes/1e6:.2f}MB hbm_w={res.hbm_write_bytes/1e6:.2f}MB "
                f"cache={res.cache_hits}/{res.cache_hits+res.cache_misses} "
                f"peakW={res.peak_window_power_w:.2f}")
        print(line, flush=True)
        stats = res.resource_stats if hasattr(res, "resource_stats") else None
        if stats:
            (ROOT / f"project/campaigns/structural-rewrite/mlp_{variant}{tag}_stats.json"
             ).write_text(json.dumps({"cycles": res.cycles, "counts": c,
                                      "stats": stats}, indent=1, default=str))
        if not args.skip_functional:
            try:
                finite, maxerr, rel = functional(text, tensors, hw)
                print(f"[{variant}] functional finite={finite} maxabs={maxerr:.2e} "
                      f"rel={rel:.3f} pass={finite and rel <= 1.0}", flush=True)
            except ValueError as exc:
                print(f"[{variant}] functional ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
