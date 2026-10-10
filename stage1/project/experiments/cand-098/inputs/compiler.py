"""Generate literal-ISA baselines for the two active trial cases.

The Python here is a compiler only.  The emitted program uses no operator-level
instructions for LayerNorm, attention, softmax, or GELU.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from codesign.challenge.abi import Layout, build_layout
from codesign.challenge.workload import MODELS, SCENARIOS

# Search recipe: phase-1 software knobs.  The search engine writes
# project/recipe.json per candidate; values here reproduce the B-stationary
# baseline (cand-001) when the file is absent.
DEFAULT_RECIPE = {
    "gemm": {
        "order": "nkm",      # "mnk" (original baseline) | "nkm" (B-stationary)
        "m_tile": 8,
        "n_tile": 16,
        "k_chunk": 48,
        "prefetch": 0,       # double-buffered B tile loads (nkm only)
    },
    "attention": {"scores": "hbm"},    # "hbm" | "rf" (scores+probs stay in RF)
    "fusion": {"bias_gelu": 0},        # fuse b1-add + GELU into one pass
}


def load_recipe(path=None):
    recipe = json.loads(json.dumps(DEFAULT_RECIPE))
    p = Path(path) if path else Path(__file__).resolve().parent / "recipe.json"
    if p.exists():
        for section, values in json.loads(p.read_text()).items():
            if isinstance(values, dict) and isinstance(recipe.get(section), dict):
                recipe[section].update(values)
            else:
                recipe[section] = values
    return recipe


def var(name):
    return {"var": name}


def add(*items):
    return {"add": list(items)}


def mul(*items):
    return {"mul": list(items)}


def minimum(*items):
    return {"min": list(items)}


def mod(a, b):
    return {"mod": [a, b]}


def sub(a, b):
    return add(a, mul(-1, b))


def hbm(offset, count, shape=None, strides=None):
    result = {"space": "HBM", "offset": offset, "count": count, "wg": None, "lane": 0}
    if shape is not None:
        result.update(shape=shape, strides=strides)
    return result


def rf(lane, count, offset=0):
    return {"space": "RF", "offset": offset, "count": count, "wg": "g", "lane": lane}


def sh(offset, count, shape=None, strides=None):
    result = {"space": "SH", "offset": offset, "count": count, "wg": "g", "lane": 0}
    if shape is not None:
        result.update(shape=shape, strides=strides)
    return result


def imm(value):
    return {"imm": value}


@dataclass(frozen=True)
class Tensor:
    offset: int
    shape: tuple[int, ...]


class Builder:
    def __init__(self, layout: Layout, recipe: dict | None = None, shared_bytes: int = 0):
        recipe = recipe or load_recipe()
        self.recipe = recipe
        self.layout = layout
        self.set_gemm_stage(None)
        self.scores_res = recipe["attention"]["scores"]
        self.fuse_bias_gelu = int(recipe["fusion"]["bias_gelu"])
        self.lines = []
        self.loops = []
        self.serial = 0
        self.cursor = layout.symbols["scratch"].address // 4
        self.tensors = {}
        self.emit("WG.BEGIN", wg="g", sm=0, shared_bytes=shared_bytes)

    def set_gemm_stage(self, stage):
        self.stage = stage
        g = dict(self.recipe["gemm"])
        if stage is not None:
            g.update(self.recipe.get("gemm_stages", {}).get(stage, {}))
        if int(g["n_tile"]) <= 0 or int(g["k_chunk"]) <= 0 or int(g["m_tile"]) <= 0:
            raise ValueError("GEMM tiles must be positive")
        self.gemm_n_tile = int(g["n_tile"])
        self.m_tile = int(g["m_tile"])
        self.k_chunk = int(g["k_chunk"])
        self.order = g["order"]
        self.prefetch = int(g["prefetch"])

    def symbol(self, name):
        item = self.layout.symbols[name]
        return Tensor(item.address // 4, item.shape)

    def alloc(self, name, *shape):
        size = 1
        for extent in shape:
            size *= extent
        result = Tensor(self.cursor, tuple(shape))
        self.cursor += size
        self.tensors[name] = result
        return result

    def emit(self, op, **args):
        if "event" in args and args["event"] is None:
            self.serial += 1
            suffix = "".join(f"_{{{name}}}" for name in self.loops)
            args["event"] = f"e{self.serial}{suffix}"
        self.lines.append(f"{op} {json.dumps(args, separators=(',', ':'))}")

    @contextmanager
    def loop(self, name, start, stop, step=1):
        self.emit("FOR", var=name, start=start, stop=stop, step=step)
        self.loops.append(name)
        yield var(name)
        self.loops.pop()
        self.emit("END.FOR")

    def ld(self, source, target):
        self.emit("LD", src=source, dst=target, event=None)

    def st(self, source, target):
        self.emit("ST", src=source, dst=target, event=None)

    def vec(self, kind, sources, target):
        self.emit("VEC", kind=kind, src=sources, dst=target, event=None)

    def reduce(self, kind, source, target):
        self.emit("REDUCE", kind=kind, src=source, dst=target, event=None)

    def sfu(self, kind, source, target):
        self.emit("SFU", kind=kind, src=source, dst=target, event=None)

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        out: Tensor,
        m_extent: int,
        k_extent: int,
        n_extent: int,
        name: str,
    ):
        if b.shape != (k_extent, n_extent):
            raise ValueError("GEMM weight shape")
        kernel = self.recipe.get("gemm_kernels", {}).get("p1_prompt_w1")
        if self.stage == "p1_prompt" and name.endswith("w1") and kernel:
            if kernel.get("kind") != "shared_weight_panel":
                raise ValueError("unsupported P1 W1 kernel")
            self.gemm_shared_weight_panel(a, b, out, m_extent, k_extent, n_extent, name, kernel)
            return
        tile_n = self.gemm_n_tile
        m_tile = self.m_tile
        k_chunk = self.k_chunk
        if n_extent % tile_n:
            raise ValueError("GEMM N extent does not fit tile")
        if self.order == "mnk":
            # Original baseline order: B tile re-loaded for every m-tile.
            with self.loop(f"{name}m", 0, m_extent, m_tile) as row:
                rows = minimum(m_tile, sub(m_extent, row))
                with self.loop(f"{name}n", 0, n_extent, tile_n) as col:
                    cells = mul(rows, tile_n)
                    self.vec("add", [imm(0), imm(0)], rf(2, cells))
                    with self.loop(f"{name}k", 0, k_extent, k_chunk) as inner:
                        depth = minimum(k_chunk, sub(k_extent, inner))
                        self.ld(
                            hbm(
                                add(a.offset, mul(row, k_extent), inner),
                                mul(rows, depth),
                                [rows, depth],
                                [k_extent, 1],
                            ),
                            rf(0, mul(rows, depth)),
                        )
                        self.ld(
                            hbm(
                                add(b.offset, mul(inner, n_extent), col),
                                mul(depth, tile_n),
                                [depth, tile_n],
                                [n_extent, 1],
                            ),
                            rf(1, mul(depth, tile_n)),
                        )
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, mul(rows, depth)),
                            b=rf(1, mul(depth, tile_n)),
                            acc=rf(2, cells),
                            m=rows,
                            n=tile_n,
                            k=depth,
                            event=None,
                        )
                    self.st(
                        rf(2, cells),
                        hbm(
                            add(out.offset, mul(row, n_extent), col),
                            cells,
                            [rows, tile_n],
                            [n_extent, 1],
                        ),
                    )
            return
        if self.order not in ("nkm", "nkm_lanes"):
            raise ValueError(f"Unsupported GEMM order {self.order}")
        # B-stationary order: each weight tile is loaded once per (n, k) chunk
        # and reused across all m-tiles.  All m-tile accumulators for one
        # n-tile stay live in lane 2, which requires m_extent * tile_n <= the
        # lane capacity (1024 FP32 words at 16 vector lanes).
        if self.order == "nkm" and m_extent * tile_n > 1024:
            raise ValueError("GEMM accumulators exceed RF lane capacity")
        def accumulator(row, rows):
            if self.order == "nkm_lanes":
                # Reserve 0/1 for operands and 3 for B prefetch. Each M tile
                # gets a separate partition, so N is not limited by all M rows.
                return rf(add(4, {"ceildiv": [row, m_tile]}), mul(rows, tile_n))
            return rf(2, mul(rows, tile_n), mul(row, tile_n))

        chunks = -(-k_extent // k_chunk)
        with self.loop(f"{name}n", 0, n_extent, tile_n) as col:
            with self.loop(f"{name}z", 0, m_extent, m_tile) as row:
                rows = minimum(m_tile, sub(m_extent, row))
                self.vec(
                    "add", [imm(0), imm(0)], accumulator(row, rows)
                )
            if self.prefetch:
                # Double-buffer B in lanes 1/3: inside iteration i, issue the
                # async load for chunk i+1 before the m-loop MMAs consume the
                # current buffer.  The last iteration re-loads the final tile
                # (clamped offset) — harmless.
                depth0 = minimum(k_chunk, k_extent)
                self.ld(
                    hbm(
                        add(b.offset, col),
                        mul(depth0, tile_n),
                        [depth0, tile_n],
                        [n_extent, 1],
                    ),
                    rf(1, mul(depth0, tile_n)),
                )
            with self.loop(f"{name}k", 0, k_extent, k_chunk) as inner:
                depth = minimum(k_chunk, sub(k_extent, inner))
                if self.prefetch:
                    ki = {"ceildiv": [inner, k_chunk]}
                    lane_cur = add(1, mul(2, mod(ki, 2)))
                    lane_nxt = add(1, mul(2, mod(add(ki, 1), 2)))
                    inner_next = minimum(
                        add(inner, k_chunk), mul(chunks - 1, k_chunk)
                    )
                    depth_next = minimum(k_chunk, sub(k_extent, inner_next))
                    self.ld(
                        hbm(
                            add(b.offset, mul(inner_next, n_extent), col),
                            mul(depth_next, tile_n),
                            [depth_next, tile_n],
                            [n_extent, 1],
                        ),
                        rf(lane_nxt, mul(depth_next, tile_n)),
                    )
                else:
                    lane_cur = 1
                    self.ld(
                        hbm(
                            add(b.offset, mul(inner, n_extent), col),
                            mul(depth, tile_n),
                            [depth, tile_n],
                            [n_extent, 1],
                        ),
                        rf(1, mul(depth, tile_n)),
                    )
                with self.loop(f"{name}m", 0, m_extent, m_tile) as row:
                    rows = minimum(m_tile, sub(m_extent, row))
                    self.ld(
                        hbm(
                            add(a.offset, mul(row, k_extent), inner),
                            mul(rows, depth),
                            [rows, depth],
                            [k_extent, 1],
                        ),
                        rf(0, mul(rows, depth)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=rf(0, mul(rows, depth)),
                        b=rf(lane_cur, mul(depth, tile_n)),
                        acc=accumulator(row, rows),
                        m=rows,
                        n=tile_n,
                        k=depth,
                        event=None,
                    )
            with self.loop(f"{name}s", 0, m_extent, m_tile) as row:
                rows = minimum(m_tile, sub(m_extent, row))
                cells = mul(rows, tile_n)
                self.st(
                    accumulator(row, rows),
                    hbm(
                        add(out.offset, mul(row, n_extent), col),
                        cells,
                        [rows, tile_n],
                        [n_extent, 1],
                    ),
                )

    def gemm_shared_weight_panel(self, a, b, out, m_extent, k_extent, n_extent, name, kernel):
        """Stage a W1 weight panel in SH, then reuse A across its 16-column subtiles."""
        m_panel = int(kernel["m_panel"])
        n_panel = int(kernel["n_panel"])
        sub_n = 16
        k_chunk = 48
        if (m_panel, n_panel, k_extent, n_extent) != (16, 64, 256, 1024):
            raise ValueError("unsupported shared W1 panel shape")
        with self.loop(f"{name}pn", 0, n_extent, n_panel) as panel_col:
            self.ld(
                hbm(add(b.offset, panel_col), k_extent * n_panel,
                    [k_extent, n_panel], [n_extent, 1]),
                sh(0, k_extent * n_panel),
            )
            with self.loop(f"{name}pm", 0, m_extent, m_panel) as row:
                rows = minimum(m_panel, sub(m_extent, row))
                with self.loop(f"{name}zi", 0, n_panel, sub_n) as subcol:
                    self.vec("add", [imm(0), imm(0)],
                             rf(2, mul(rows, sub_n), mul(subcol, m_panel)))
                with self.loop(f"{name}pk", 0, k_extent, k_chunk) as inner:
                    depth = minimum(k_chunk, sub(k_extent, inner))
                    self.ld(
                        hbm(add(a.offset, mul(row, k_extent), inner), mul(rows, depth),
                            [rows, depth], [k_extent, 1]),
                        rf(0, mul(rows, depth)),
                    )
                    with self.loop(f"{name}pc", 0, n_panel, sub_n) as subcol:
                        self.ld(
                            sh(add(mul(inner, n_panel), subcol), mul(depth, sub_n),
                               [depth, sub_n], [n_panel, 1]),
                            rf(1, mul(depth, sub_n)),
                        )
                        self.emit(
                            "MMA.ACC", a=rf(0, mul(rows, depth)),
                            b=rf(1, mul(depth, sub_n)),
                            acc=rf(2, mul(rows, sub_n), mul(subcol, m_panel)),
                            m=rows, n=sub_n, k=depth, event=None,
                        )
                with self.loop(f"{name}ps", 0, n_panel, sub_n) as subcol:
                    self.st(
                        rf(2, mul(rows, sub_n), mul(subcol, m_panel)),
                        hbm(add(out.offset, mul(row, n_extent), panel_col, subcol),
                            mul(rows, sub_n), [rows, sub_n], [n_extent, 1]),
                    )

    def layernorm(
        self, source: Tensor, gamma: Tensor, beta: Tensor, out: Tensor, rows: int, d: int, name: str
    ):
        with self.loop(name, 0, rows) as row:
            base = add(source.offset, mul(row, d))
            target = add(out.offset, mul(row, d))
            self.ld(hbm(base, d), rf(0, d))
            self.reduce("sum", rf(0, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("sub", [rf(0, d), rf(1, 1)], rf(2, d))
            self.vec("mul", [rf(2, d), rf(2, d)], rf(3, d))
            self.reduce("sum", rf(3, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("add", [rf(1, 1), imm(1e-5)], rf(1, 1))
            self.sfu("rsqrt", rf(1, 1), rf(1, 1))
            self.vec("mul", [rf(2, d), rf(1, 1)], rf(2, d))
            self.ld(hbm(gamma.offset, d), rf(3, d))
            self.vec("mul", [rf(2, d), rf(3, d)], rf(2, d))
            self.ld(hbm(beta.offset, d), rf(3, d))
            self.vec("add", [rf(2, d), rf(3, d)], rf(2, d))
            self.st(rf(2, d), hbm(target, d))

    def add_rows(
        self,
        left: Tensor,
        right: Tensor,
        out: Tensor,
        rows: int,
        width: int,
        name: str,
        bias: Tensor | None = None,
    ):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                self.ld(hbm(add(left.offset, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(right.offset, mul(row, width), col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                if bias is not None:
                    self.ld(hbm(add(bias.offset, col), n), rf(3, n))
                    self.vec("add", [rf(2, n), rf(3, n)], rf(2, n))
                self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def add_bias(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                self.ld(hbm(add(source.offset, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(bias.offset, col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def bias_gelu(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str):
        """Fused `add_bias` + `gelu`: one read of `source`, no `biased` scratch."""
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                addr = add(source.offset, mul(row, width), col)
                self.ld(hbm(addr, n), rf(0, n))
                self.ld(hbm(add(bias.offset, col), n), rf(2, n))
                self.vec("add", [rf(0, n), rf(2, n)], rf(0, n))
                self.vec("mul", [rf(0, n), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), rf(0, n)], rf(1, n))
                self.vec("fma", [rf(1, n), imm(0.044715), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), imm(sqrt(2 / 3.141592653589793))], rf(1, n))
                self.sfu("tanh", rf(1, n), rf(1, n))
                self.vec("add", [rf(1, n), imm(1)], rf(1, n))
                self.vec("mul", [rf(0, n), imm(0.5)], rf(0, n))
                self.vec("mul", [rf(0, n), rf(1, n)], rf(0, n))
                self.st(rf(0, n), hbm(add(out.offset, mul(row, width), col), n))

    def gelu(self, source: Tensor, out: Tensor, rows: int, width: int, name: str):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 384) as col:
                n = minimum(384, sub(width, col))
                addr = add(source.offset, mul(row, width), col)
                self.ld(hbm(addr, n), rf(0, n))
                self.vec("mul", [rf(0, n), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), rf(0, n)], rf(1, n))
                self.vec("fma", [rf(1, n), imm(0.044715), rf(0, n)], rf(1, n))
                self.vec("mul", [rf(1, n), imm(sqrt(2 / 3.141592653589793))], rf(1, n))
                self.sfu("tanh", rf(1, n), rf(1, n))
                self.vec("add", [rf(1, n), imm(1)], rf(1, n))
                self.vec("mul", [rf(0, n), imm(0.5)], rf(0, n))
                self.vec("mul", [rf(0, n), rf(1, n)], rf(0, n))
                self.st(rf(0, n), hbm(add(out.offset, mul(row, width), col), n))

    def attention(
        self,
        qkv: Tensor,
        context: Tensor,
        k_out: Tensor,
        v_out: Tensor,
        rows: int,
        past: int,
        d: int,
        h: int,
        hd: int,
        name: str,
    ):
        # Export every layer's K/V in the public head-major ABI.
        with self.loop(f"{name}h", 0, h) as head:
            with self.loop(f"{name}r", 0, rows) as row:
                for kind, target, component in (("k", k_out, 1), ("v", v_out, 2)):
                    source_addr = add(qkv.offset, mul(row, 3 * d), component * d, mul(head, hd))
                    target_addr = add(
                        target.offset, mul(head, target.shape[2] * hd), mul(add(past, row), hd)
                    )
                    self.ld(hbm(source_addr, hd), rf(0, hd))
                    self.st(rf(0, hd), hbm(target_addr, hd))
        total = past + rows
        rf_scores = self.scores_res == "rf"
        if not rf_scores:
            scores = self.alloc(f"{name}_scores", rows, total)
            probs = self.alloc(f"{name}_probs", rows, total)
        with self.loop(f"{name}q", 0, rows) as row:
            limit = add(past, row, 1)
            with self.loop(f"{name}head", 0, h) as head:
                q_addr = add(qkv.offset, mul(row, 3 * d), mul(head, hd))
                self.ld(hbm(q_addr, hd), rf(0, hd))
                with self.loop(f"{name}key", 0, limit, 16) as key:
                    n = minimum(16, sub(limit, key))
                    cells = mul(hd, n)
                    self.ld(
                        hbm(
                            add(k_out.offset, mul(head, k_out.shape[2] * hd), mul(key, hd)),
                            cells,
                            [hd, n],
                            [1, hd],
                        ),
                        rf(1, cells),
                    )
                    if rf_scores:
                        # Scores accumulate directly in lane 4 at [key, key+n).
                        self.vec("add", [imm(0), imm(0)], rf(4, n, key))
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, hd),
                            b=rf(1, cells),
                            acc=rf(4, n, key),
                            m=1,
                            n=n,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [rf(4, n, key), imm(1 / sqrt(hd))], rf(4, n, key))
                    else:
                        self.vec("add", [imm(0), imm(0)], rf(2, n))
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, hd),
                            b=rf(1, cells),
                            acc=rf(2, n),
                            m=1,
                            n=n,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [rf(2, n), imm(1 / sqrt(hd))], rf(2, n))
                        self.st(rf(2, n), hbm(add(scores.offset, mul(row, total), key), n))
                if rf_scores:
                    svec = rf(4, limit)
                else:
                    svec = rf(0, limit)
                    self.ld(hbm(add(scores.offset, mul(row, total)), limit), svec)
                self.reduce("max", svec, rf(1, 1))
                self.vec("sub", [svec, rf(1, 1)], svec)
                self.sfu("exp", svec, svec)
                self.reduce("sum", svec, rf(1, 1))
                self.vec("div", [svec, rf(1, 1)], svec)
                if not rf_scores:
                    self.st(svec, hbm(add(probs.offset, mul(row, total)), limit))
                self.vec("add", [imm(0), imm(0)], rf(2, hd))
                with self.loop(f"{name}value", 0, limit, 16) as key:
                    depth = minimum(16, sub(limit, key))
                    if rf_scores:
                        probs_view = rf(4, depth, key)
                    else:
                        probs_view = rf(0, depth)
                        self.ld(
                            hbm(add(probs.offset, mul(row, total), key), depth), probs_view
                        )
                    self.ld(
                        hbm(
                            add(v_out.offset, mul(head, v_out.shape[2] * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=probs_view,
                        b=rf(1, mul(depth, hd)),
                        acc=rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
                self.st(rf(2, hd), hbm(add(context.offset, mul(row, d), mul(head, hd)), hd))

    def attention_decode(
        self,
        qkv: Tensor,
        context: Tensor,
        history_k: Tensor,
        history_v: Tensor,
        new_k: Tensor,
        new_v: Tensor,
        step: int,
        past: int,
        d: int,
        h: int,
        hd: int,
        name: str,
    ):
        """One decode step: score historical and newly generated KV separately."""
        generated = step + 1
        total = past + generated
        rf_scores = self.scores_res == "rf"
        if not rf_scores:
            scores = self.alloc(f"{name}_scores", total)
            probs = self.alloc(f"{name}_probs", total)
        for kind, target, component in (("k", new_k, 1), ("v", new_v, 2)):
            with self.loop(f"{name}{kind}h", 0, h) as head:
                source = add(qkv.offset, component * d, mul(head, hd))
                destination = add(target.offset, mul(head, target.shape[2] * hd), step * hd)
                self.ld(hbm(source, hd), rf(0, hd))
                self.st(rf(0, hd), hbm(destination, hd))
        with self.loop(f"{name}head", 0, h) as head:
            self.ld(hbm(add(qkv.offset, mul(head, hd)), hd), rf(0, hd))
            for source, count, score_base, stride in (
                (history_k, past, 0, past),
                (new_k, generated, past, new_k.shape[2]),
            ):
                with self.loop(f"{name}key{score_base}", 0, count, 16) as key:
                    n = minimum(16, sub(count, key))
                    cells = mul(hd, n)
                    self.ld(
                        hbm(
                            add(source.offset, mul(head, stride * hd), mul(key, hd)),
                            cells,
                            [hd, n],
                            [1, hd],
                        ),
                        rf(1, cells),
                    )
                    if rf_scores:
                        acc = rf(4, n, add(score_base, key))
                        self.vec("add", [imm(0), imm(0)], acc)
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, hd),
                            b=rf(1, cells),
                            acc=acc,
                            m=1,
                            n=n,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [acc, imm(1 / sqrt(hd))], acc)
                    else:
                        self.vec("add", [imm(0), imm(0)], rf(2, n))
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, hd),
                            b=rf(1, cells),
                            acc=rf(2, n),
                            m=1,
                            n=n,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [rf(2, n), imm(1 / sqrt(hd))], rf(2, n))
                        self.st(rf(2, n), hbm(add(scores.offset, score_base, key), n))
            if rf_scores:
                svec = rf(4, total)
            else:
                svec = rf(0, total)
                self.ld(hbm(scores.offset, total), svec)
            self.reduce("max", svec, rf(1, 1))
            self.vec("sub", [svec, rf(1, 1)], svec)
            self.sfu("exp", svec, svec)
            self.reduce("sum", svec, rf(1, 1))
            self.vec("div", [svec, rf(1, 1)], svec)
            if not rf_scores:
                self.st(svec, hbm(probs.offset, total))
            self.vec("add", [imm(0), imm(0)], rf(2, hd))
            for source, count, prob_base, stride in (
                (history_v, past, 0, past),
                (new_v, generated, past, new_v.shape[2]),
            ):
                with self.loop(f"{name}value{prob_base}", 0, count, 16) as key:
                    depth = minimum(16, sub(count, key))
                    if rf_scores:
                        probs_view = rf(4, depth, add(prob_base, key))
                    else:
                        probs_view = rf(0, depth)
                        self.ld(
                            hbm(add(probs.offset, prob_base, key), depth), probs_view
                        )
                    self.ld(
                        hbm(
                            add(source.offset, mul(head, stride * hd), mul(key, hd)),
                            mul(depth, hd),
                        ),
                        rf(1, mul(depth, hd)),
                    )
                    self.emit(
                        "MMA.ACC",
                        a=probs_view,
                        b=rf(1, mul(depth, hd)),
                        acc=rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
            self.st(rf(2, hd), hbm(add(context.offset, mul(head, hd)), hd))

    def finish(self):
        self.emit("WG.END", wg="g")
        return "\n".join(self.lines) + "\n"


def generate_m1_p1(recipe: dict | None = None) -> tuple[str, Layout, int]:
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    layout = build_layout(model, "P1")
    recipe = recipe or load_recipe()
    kernel = recipe.get("gemm_kernels", {}).get("p1_prompt_w1")
    shared_bytes = 256 * int(kernel["n_panel"]) * 4 if kernel else 0
    b = Builder(layout, recipe, shared_bytes=shared_bytes)
    prompt = b.symbol("input/prompt")
    step = b.symbol("input/step0")
    output = b.symbol("output/hidden")
    # All prompts finish before STEP.COMMIT releases either batch member's first new token.
    for phase, rows in (("p", prompt_rows), ("s", 1)):
        b.set_gemm_stage("p1_prompt" if phase == "p" else "p1_step")
        for member in range(batch):
            initial = (
                Tensor(prompt.offset + member * prompt_rows * d, (rows, d))
                if phase == "p"
                else Tensor(step.offset + member * d, (rows, d))
            )
            x = initial
            for layer in range(model.layers):
                prefix = f"layer{layer}/"
                name = f"{phase}b{member}l{layer}"
                weights = {
                    key: b.symbol(prefix + key)
                    for key in (
                        "ln1_g",
                        "ln1_b",
                        "wqkv",
                        "wo",
                        "ln2_g",
                        "ln2_b",
                        "w1",
                        "b1",
                        "w2",
                        "b2",
                    )
                }
                ln1 = b.alloc(f"{name}_ln1", rows, d)
                qkv = b.alloc(f"{name}_qkv", rows, 3 * d)
                ctx = b.alloc(f"{name}_ctx", rows, d)
                att = b.alloc(f"{name}_att", rows, d)
                res = b.alloc(f"{name}_res", rows, d)
                ln2 = b.alloc(f"{name}_ln2", rows, d)
                ff1 = b.alloc(f"{name}_ff1", rows, f)
                biased = b.alloc(f"{name}_biased", rows, f)
                activated = b.alloc(f"{name}_gelu", rows, f)
                ff2 = b.alloc(f"{name}_ff2", rows, d)
                y = b.alloc(f"{name}_out", rows, d)
                b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, rows, d, name + "ln1")
                b.gemm(ln1, weights["wqkv"], qkv, rows, d, 3 * d, name + "qkv")
                k_out, v_out = (b.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
                kv_offset = member * h * (prompt_rows + 1) * hd
                k_out = Tensor(k_out.offset + kv_offset, k_out.shape)
                v_out = Tensor(v_out.offset + kv_offset, v_out.shape)
                b.attention(
                    qkv,
                    ctx,
                    k_out,
                    v_out,
                    rows,
                    0 if phase == "p" else prompt_rows,
                    d,
                    h,
                    hd,
                    name + "a",
                )
                b.gemm(ctx, weights["wo"], att, rows, d, d, name + "wo")
                b.add_rows(x, att, res, rows, d, name + "res")
                b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, rows, d, name + "ln2")
                b.gemm(ln2, weights["w1"], ff1, rows, d, f, name + "w1")
                if b.fuse_bias_gelu:
                    b.bias_gelu(ff1, weights["b1"], activated, rows, f, name + "biasgelu")
                else:
                    b.add_bias(ff1, weights["b1"], biased, rows, f, name + "bias1")
                    b.gelu(biased, activated, rows, f, name + "gelu")
                b.gemm(activated, weights["w2"], ff2, rows, f, d, name + "w2")
                b.add_rows(res, ff2, y, rows, d, name + "out", weights["b2"])
                x = y
            with b.loop(f"{phase}b{member}output", 0, rows) as row:
                b.ld(hbm(add(x.offset, mul(row, d)), d), rf(0, d))
                b.st(
                    rf(0, d),
                    hbm(
                        add(
                            output.offset,
                            member * (prompt_rows + 1) * d,
                            (0 if phase == "p" else prompt_rows) * d,
                            mul(row, d),
                        ),
                        d,
                    ),
                )
        if phase == "p":
            b.emit("STEP.COMMIT", step=0)
    program = b.finish()
    return program, layout, b.cursor


def generate_m1_d1(gemm_n_tile: int | None = None, recipe: dict | None = None) -> tuple[str, Layout, int]:
    """Compile the D1 model's history and causally released decode steps."""
    model = MODELS["M2"]
    batch, past, steps = SCENARIOS["D1"]
    if batch != 1:
        raise ValueError("D1 baseline requires batch one")
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    layout = build_layout(model, "D1")
    if recipe is None:
        recipe = load_recipe()
    if gemm_n_tile is not None:
        recipe["gemm"]["n_tile"] = gemm_n_tile
    b = Builder(layout, recipe)
    b.set_gemm_stage("d1")
    output = b.symbol("output/hidden")
    for step in range(steps):
        x = b.symbol(f"input/step{step}")
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            name = f"d{step}l{layer}"
            weights = {
                key: b.symbol(prefix + key)
                for key in (
                    "ln1_g",
                    "ln1_b",
                    "wqkv",
                    "wo",
                    "ln2_g",
                    "ln2_b",
                    "w1",
                    "b1",
                    "w2",
                    "b2",
                )
            }
            ln1 = b.alloc(f"{name}_ln1", 1, d)
            qkv = b.alloc(f"{name}_qkv", 1, 3 * d)
            ctx = b.alloc(f"{name}_ctx", 1, d)
            att = b.alloc(f"{name}_att", 1, d)
            res = b.alloc(f"{name}_res", 1, d)
            ln2 = b.alloc(f"{name}_ln2", 1, d)
            ff1 = b.alloc(f"{name}_ff1", 1, f)
            biased = b.alloc(f"{name}_biased", 1, f)
            activated = b.alloc(f"{name}_gelu", 1, f)
            ff2 = b.alloc(f"{name}_ff2", 1, d)
            y = b.alloc(f"{name}_out", 1, d)
            b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, 1, d, name + "ln1")
            b.gemm(ln1, weights["wqkv"], qkv, 1, d, 3 * d, name + "qkv")
            b.attention_decode(
                qkv,
                ctx,
                b.symbol(prefix + "history_k"),
                b.symbol(prefix + "history_v"),
                b.symbol(prefix + "new_k"),
                b.symbol(prefix + "new_v"),
                step,
                past,
                d,
                h,
                hd,
                name + "a",
            )
            b.gemm(ctx, weights["wo"], att, 1, d, d, name + "wo")
            b.add_rows(x, att, res, 1, d, name + "res")
            b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, 1, d, name + "ln2")
            b.gemm(ln2, weights["w1"], ff1, 1, d, f, name + "w1")
            if b.fuse_bias_gelu:
                b.bias_gelu(ff1, weights["b1"], activated, 1, f, name + "biasgelu")
            else:
                b.add_bias(ff1, weights["b1"], biased, 1, f, name + "bias1")
                b.gelu(biased, activated, 1, f, name + "gelu")
            b.gemm(activated, weights["w2"], ff2, 1, f, d, name + "w2")
            b.add_rows(res, ff2, y, 1, d, name + "out", weights["b2"])
            x = y
        b.ld(hbm(x.offset, d), rf(0, d))
        b.st(rf(0, d), hbm(output.offset + step * d, d))
        b.emit("STEP.COMMIT", step=step)
    return b.finish(), layout, b.cursor


if __name__ == "__main__":
    from pathlib import Path

    output = Path("programs")
    output.mkdir(exist_ok=True)
    for filename, generate in (("M1_P1.asm", generate_m1_p1), ("M2_D1.asm", generate_m1_d1)):
        program, _, _ = generate()
        (output / filename).write_text(program, encoding="utf-8")
        print(f"Generated {output / filename}: {len(program.splitlines())} lines")
