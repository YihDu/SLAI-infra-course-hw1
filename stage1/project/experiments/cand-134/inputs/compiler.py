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
    def __init__(self, layout: Layout, recipe: dict | None = None, shared_bytes: int = 0, groups: int = 1):
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
        self.groups = ["g"] if groups == 1 else [f"g{i}" for i in range(groups)]
        self.wg = self.groups[0]
        self.label = ""
        for sm, wg in enumerate(self.groups):
            self.emit("WG.BEGIN", wg=wg, sm=sm, shared_bytes=shared_bytes)

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
            args["event"] = f"{self.label}e{self.serial}{suffix}"
        def bind(value):
            if isinstance(value, dict):
                return {k: (self.wg if k == "wg" and v == "g" else bind(v))
                        for k, v in value.items()}
            if isinstance(value, list):
                return [bind(v) for v in value]
            return value
        args = bind(args)
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
        col_start: int = 0,
        col_stop: int | None = None,
    ):
        col_stop = n_extent if col_stop is None else col_stop
        if b.shape != (k_extent, n_extent):
            raise ValueError("GEMM weight shape")
        pipelines = self.recipe.get("gemm_stages", {}).get("p1_prompt", {}).get(
            "operand_pipeline", []
        )
        pipeline_stage = next(
            (candidate for candidate in pipelines if name.endswith(candidate)), None
        )
        if self.stage == "p1_prompt" and pipeline_stage:
            self.gemm_operand_pipeline(
                a, b, out, m_extent, k_extent, n_extent, name, col_start, col_stop
            )
            return
        kernel = self.recipe.get("gemm_kernels", {}).get("p1_prompt_w1")
        if self.stage == "p1_prompt" and name.endswith("w1") and kernel:
            if kernel.get("kind") != "shared_weight_panel":
                raise ValueError("unsupported P1 W1 kernel")
            self.gemm_shared_weight_panel(a, b, out, m_extent, k_extent, n_extent, name, kernel)
            return
        tile_n = self.gemm_n_tile
        m_tile = self.m_tile
        k_chunk = self.k_chunk
        if not 0 <= col_start < col_stop <= n_extent or col_start % tile_n or col_stop % tile_n:
            raise ValueError("GEMM N extent does not fit tile")
        if self.order == "mnk":
            # Original baseline order: B tile re-loaded for every m-tile.
            with self.loop(f"{name}m", 0, m_extent, m_tile) as row:
                rows = minimum(m_tile, sub(m_extent, row))
                with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
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
        with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
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

    def gemm_operand_pipeline(
        self, a, b, out, m_extent, k_extent, n_extent, name,
        col_start=0, col_stop=None,
    ):
        """Double-buffer both operands for 64-row P1 projection GEMMs.

        The mature N/K/M kernel reuses RF0 for both row blocks, forcing each
        following A load to wait for the preceding MMA read.  Here RF0/RF2 and
        RF6/RF8 alternate between K chunks while RF1/RF3 alternate B. RF4/RF5
        remain the two accumulators.  MMA shapes and ascending K order match the
        parent program exactly.
        """
        col_stop = n_extent if col_stop is None else col_stop
        tile_n, m_tile, k_chunk = self.gemm_n_tile, self.m_tile, self.k_chunk
        if (m_extent, m_tile, tile_n, k_chunk) != (64, 32, 32, 32):
            raise ValueError("P1 operand pipeline requires M64/Mtile32/N32/K32")
        if not 0 <= col_start < col_stop <= n_extent or col_start % tile_n or col_stop % tile_n:
            raise ValueError("P1 operand pipeline N extent does not fit tile")

        def accumulator(row):
            return rf(add(4, {"ceildiv": [row, m_tile]}), m_tile * tile_n)

        def a_lane(row, parity):
            return add(mul(6, parity), mul(2, {"ceildiv": [row, m_tile]}))

        with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
            with self.loop(f"{name}z", 0, m_extent, m_tile) as row:
                self.vec("add", [imm(0), imm(0)], accumulator(row))

            # Seed K=0 operands. All extents divide exactly for the supported
            # prompt projections, so the descriptor sizes stay constant.
            self.ld(
                hbm(add(b.offset, col), k_chunk * tile_n,
                    [k_chunk, tile_n], [n_extent, 1]),
                rf(1, k_chunk * tile_n),
            )
            with self.loop(f"{name}seedm", 0, m_extent, m_tile) as row:
                self.ld(
                    hbm(add(a.offset, mul(row, k_extent)), m_tile * k_chunk,
                        [m_tile, k_chunk], [k_extent, 1]),
                    rf(a_lane(row, 0), m_tile * k_chunk),
                )

            with self.loop(f"{name}k", 0, k_extent, k_chunk) as inner:
                parity = mod({"ceildiv": [inner, k_chunk]}, 2)
                next_parity = sub(1, parity)
                with self.loop(f"{name}nextb", add(inner, k_chunk), k_extent, k_extent) as nxt:
                    self.ld(
                        hbm(add(b.offset, mul(nxt, n_extent), col), k_chunk * tile_n,
                            [k_chunk, tile_n], [n_extent, 1]),
                        rf(add(1, mul(2, next_parity)), k_chunk * tile_n),
                    )
                with self.loop(f"{name}m", 0, m_extent, m_tile) as row:
                    self.emit(
                        "MMA.ACC",
                        a=rf(a_lane(row, parity), m_tile * k_chunk),
                        b=rf(add(1, mul(2, parity)), k_chunk * tile_n),
                        acc=accumulator(row), m=m_tile, n=tile_n, k=k_chunk,
                        event=None,
                    )
                    with self.loop(f"{name}nexta", add(inner, k_chunk), k_extent, k_extent) as nxt:
                        self.ld(
                            hbm(add(a.offset, mul(row, k_extent), nxt), m_tile * k_chunk,
                                [m_tile, k_chunk], [k_extent, 1]),
                            rf(a_lane(row, next_parity), m_tile * k_chunk),
                        )
            with self.loop(f"{name}s", 0, m_extent, m_tile) as row:
                self.st(
                    accumulator(row),
                    hbm(add(out.offset, mul(row, n_extent), col), m_tile * tile_n,
                        [m_tile, tile_n], [n_extent, 1]),
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

    def layernorm_cached_params(
        self, source: Tensor, gamma: Tensor, beta: Tensor, out: Tensor,
        rows: int, d: int, name: str,
    ):
        """Keep prompt LayerNorm affine parameters in RF across all rows."""
        self.ld(hbm(gamma.offset, d), rf(4, d))
        self.ld(hbm(beta.offset, d), rf(5, d))
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
            self.vec("mul", [rf(2, d), rf(4, d)], rf(2, d))
            self.vec("add", [rf(2, d), rf(5, d)], rf(2, d))
            self.st(rf(2, d), hbm(target, d))

    def residual_layernorm_cached(
        self, left: Tensor, right: Tensor, residual: Tensor,
        gamma: Tensor, beta: Tensor, out: Tensor,
        rows: int, d: int, name: str, out_shared=False,
    ):
        """Write the residual once while carrying its RF row directly into LN."""
        self.ld(hbm(gamma.offset, d), rf(4, d))
        self.ld(hbm(beta.offset, d), rf(5, d))
        with self.loop(name, 0, rows) as row:
            base = add(residual.offset, mul(row, d))
            self.ld(hbm(add(left.offset, mul(row, d)), d), rf(0, d))
            self.ld(hbm(add(right.offset, mul(row, d)), d), rf(1, d))
            self.vec("add", [rf(0, d), rf(1, d)], rf(2, d))
            self.st(rf(2, d), hbm(base, d))
            self.reduce("sum", rf(2, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("sub", [rf(2, d), rf(1, 1)], rf(2, d))
            self.vec("mul", [rf(2, d), rf(2, d)], rf(3, d))
            self.reduce("sum", rf(3, d), rf(1, 1))
            self.vec("mul", [rf(1, 1), imm(1 / d)], rf(1, 1))
            self.vec("add", [rf(1, 1), imm(1e-5)], rf(1, 1))
            self.sfu("rsqrt", rf(1, 1), rf(1, 1))
            self.vec("mul", [rf(2, d), rf(1, 1)], rf(2, d))
            self.vec("mul", [rf(2, d), rf(4, d)], rf(2, d))
            self.vec("add", [rf(2, d), rf(5, d)], rf(2, d))
            target = add(out.offset, mul(row, d))
            self.st(rf(2, d), (sh if out_shared else hbm)(target, d))

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

    def bias_gelu(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str, col_start=0, col_stop=None):
        """Fused `add_bias` + `gelu`: one read of `source`, no `biased` scratch."""
        col_stop = width if col_stop is None else col_stop
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", col_start, col_stop, 384) as col:
                n = minimum(384, sub(col_stop, col))
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

    def mlp_rf_stream(
        self, source, w1, b1, w2, out, rows, d, f, name,
        residual=None, output_bias=None, source_shared=False,
    ):
        """Keep one 32-row MLP block on chip across W1 -> GELU -> W2.

        RF2 is the current 32-neuron activation, RF3 its GELU temporary;
        RF4..11 hold all 256 W2 output channels until the final store.
        RF0/1 stream operands, RF12 holds ones for a bias outer product.
        Reduction chunks remain in ascending order, as in the parent GEMMs.
        """
        if (rows, d, f) != (64, 256, 1024):
            raise ValueError("RF streamed MLP requires M1 prompt dimensions")
        mode = self.recipe['fusion']['p1_mlp']
        pipeline = mode in (
            'rf_pipeline', 'rf_pipeline_residual',
            'rf_pipeline_residual_sh_input',
        )
        fused_residual = mode in (
            'rf_pipeline_residual', 'rf_pipeline_residual_sh_input',
        )
        if fused_residual != (residual is not None and output_bias is not None):
            raise ValueError("RF residual epilogue requires residual and output bias")
        tile = 32
        def source_view(offset, count, shape, strides):
            return (sh if source_shared else hbm)(offset, count, shape, strides)
        with self.loop(name + "m", 0, rows, tile) as row:
            self.vec("add", [imm(1), imm(0)], rf(12, tile))
            with self.loop(name + "z", 0, d, tile) as col:
                self.vec("add", [imm(0), imm(0)], rf(add(4, {"ceildiv": [col, tile]}), tile*tile))
            with self.loop(name + "neuron", 0, f, tile) as neuron:
                self.vec("add", [imm(0), imm(0)], rf(2, tile*tile))
                if pipeline:
                    self.ld(source_view(add(source.offset, mul(row, d)), tile*tile,
                                        [tile,tile], [d,1]), rf(0,tile*tile))
                    self.ld(hbm(add(w1.offset, neuron), tile*tile,
                                [tile,tile], [f,1]), rf(1,tile*tile))
                    with self.loop(name + 'k', 0, d, tile) as inner:
                        parity=mod({'ceildiv':[inner,tile]},2)
                        lane_a=mul(13,parity)
                        lane_b=add(1,mul(13,parity))
                        self.emit('MMA.ACC',a=rf(lane_a,tile*tile),b=rf(lane_b,tile*tile),
                                  acc=rf(2,tile*tile),m=tile,n=tile,k=tile,event=None)
                        # Enqueue next operands after issuing this MMA; alternate
                        # RF partitions avoid overwrite dependencies on its inputs.
                        with self.loop(name + 'prefetchk', add(inner,tile), d, d) as nxt:
                            nxt_parity=sub(1,parity)
                            self.ld(source_view(add(source.offset,mul(row,d),nxt),tile*tile,
                                                [tile,tile],[d,1]),rf(mul(13,nxt_parity),tile*tile))
                            self.ld(hbm(add(w1.offset,mul(nxt,f),neuron),tile*tile,
                                        [tile,tile],[f,1]),rf(add(1,mul(13,nxt_parity)),tile*tile))
                else:
                    with self.loop(name + "k", 0, d, tile) as inner:
                        self.ld(source_view(add(source.offset, mul(row, d), inner), tile*tile,
                                            [tile, tile], [d, 1]), rf(0, tile*tile))
                        self.ld(hbm(add(w1.offset, mul(inner, f), neuron), tile*tile,
                                    [tile, tile], [f, 1]), rf(1, tile*tile))
                        self.emit("MMA.ACC", a=rf(0, tile*tile), b=rf(1, tile*tile),
                                  acc=rf(2, tile*tile), m=tile, n=tile, k=tile, event=None)
                self.ld(hbm(add(b1.offset, neuron), tile), rf(0, tile))
                self.emit("MMA.ACC", a=rf(12, tile), b=rf(0, tile),
                          acc=rf(2, tile*tile), m=tile, n=tile, k=1, event=None)
                self.vec("mul", [rf(2, tile*tile), rf(2, tile*tile)], rf(3, tile*tile))
                self.vec("mul", [rf(3, tile*tile), rf(2, tile*tile)], rf(3, tile*tile))
                self.vec("fma", [rf(3, tile*tile), imm(0.044715), rf(2, tile*tile)], rf(3, tile*tile))
                self.vec("mul", [rf(3, tile*tile), imm(sqrt(2 / 3.141592653589793))], rf(3, tile*tile))
                self.sfu("tanh", rf(3, tile*tile), rf(3, tile*tile))
                self.vec("add", [rf(3, tile*tile), imm(1)], rf(3, tile*tile))
                self.vec("mul", [rf(2, tile*tile), imm(0.5)], rf(2, tile*tile))
                self.vec("mul", [rf(2, tile*tile), rf(3, tile*tile)], rf(2, tile*tile))
                if pipeline:
                    self.ld(hbm(add(w2.offset,mul(neuron,d)),tile*tile,
                                [tile,tile],[d,1]),rf(1,tile*tile))
                    with self.loop(name + 'o',0,d,tile) as col:
                        parity=mod({'ceildiv':[col,tile]},2)
                        self.emit('MMA.ACC',a=rf(2,tile*tile),b=rf(add(1,mul(13,parity)),tile*tile),
                                  acc=rf(add(4,{'ceildiv':[col,tile]}),tile*tile),
                                  m=tile,n=tile,k=tile,event=None)
                        with self.loop(name + 'prefetcho',add(col,tile),d,d) as nxt:
                            self.ld(hbm(add(w2.offset,mul(neuron,d),nxt),tile*tile,
                                        [tile,tile],[d,1]),rf(add(1,mul(13,sub(1,parity))),tile*tile))
                else:
                    with self.loop(name + "o", 0, d, tile) as col:
                        self.ld(hbm(add(w2.offset, mul(neuron, d), col), tile*tile,
                                    [tile, tile], [d, 1]), rf(1, tile*tile))
                        self.emit("MMA.ACC", a=rf(2, tile*tile), b=rf(1, tile*tile),
                                  acc=rf(add(4, {"ceildiv": [col, tile]}), tile*tile),
                                  m=tile, n=tile, k=tile, event=None)
            with self.loop(name + "store", 0, d, tile) as col:
                accumulator = rf(add(4, {"ceildiv": [col, tile]}), tile*tile)
                if fused_residual:
                    self.ld(
                        hbm(add(residual.offset, mul(row, d), col), tile*tile,
                            [tile, tile], [d, 1]),
                        rf(0, tile*tile),
                    )
                    self.ld(
                        hbm(add(output_bias.offset, col), tile),
                        rf(1, tile),
                    )
                    self.vec("add", [accumulator, rf(0, tile*tile)], accumulator)
                    self.emit(
                        "MMA.ACC", a=rf(12, tile), b=rf(1, tile),
                        acc=accumulator, m=tile, n=tile, k=1, event=None,
                    )
                self.st(rf(add(4, {"ceildiv": [col, tile]}), tile*tile),
                        hbm(add(out.offset, mul(row, d), col), tile*tile,
                            [tile, tile], [d, 1]))
                # Bound the final store burst to one DMA completion at a time.
                # The frozen scheduler cannot preview the last store of the
                # original eight-transfer burst consistently (cand-108).
                self.emit("WAIT", wg=self.wg,
                          events=[json.loads(self.lines[-1].split(" ", 1)[1])["event"]])

    def mlp_sh_resident(self, source, w1, b1, w2, out, rows, d, f, name):
        """Keep one 32x1024 activation block in SH and reuse LN2 from RF.

        W1 keeps the complete 32x256 input block in RF0..7, eliminating its
        reload for every neuron panel. GELU writes the 128 KiB activation to
        shared memory. W2 then streams it back panel by panel while RF4..11
        retain all 256 output channels. Arithmetic shapes and reduction order
        match the RF-streamed parent.
        """
        if (rows, d, f) != (64, 256, 1024):
            raise ValueError("SH resident MLP requires M1 prompt dimensions")
        tile = 32
        with self.loop(name + "m", 0, rows, tile) as row:
            self.vec("add", [imm(1), imm(0)], rf(12, tile))
            # LN2 input is invariant across all W1 neuron panels.
            with self.loop(name + "inputk", 0, d, tile) as inner:
                self.ld(
                    hbm(add(source.offset, mul(row, d), inner), tile*tile,
                        [tile, tile], [d, 1]),
                    rf({"ceildiv": [inner, tile]}, tile*tile),
                )
            with self.loop(name + "neuron", 0, f, tile) as neuron:
                self.vec("add", [imm(0), imm(0)], rf(8, tile*tile))
                with self.loop(name + "k", 0, d, tile) as inner:
                    self.ld(
                        hbm(add(w1.offset, mul(inner, f), neuron), tile*tile,
                            [tile, tile], [f, 1]),
                        rf(10, tile*tile),
                    )
                    self.emit(
                        "MMA.ACC", a=rf({"ceildiv": [inner, tile]}, tile*tile),
                        b=rf(10, tile*tile), acc=rf(8, tile*tile),
                        m=tile, n=tile, k=tile, event=None,
                    )
                self.ld(hbm(add(b1.offset, neuron), tile), rf(10, tile))
                self.emit("MMA.ACC", a=rf(12, tile), b=rf(10, tile),
                          acc=rf(8, tile*tile), m=tile, n=tile, k=1, event=None)
                self.vec("mul", [rf(8, tile*tile), rf(8, tile*tile)], rf(9, tile*tile))
                self.vec("mul", [rf(9, tile*tile), rf(8, tile*tile)], rf(9, tile*tile))
                self.vec("fma", [rf(9, tile*tile), imm(0.044715), rf(8, tile*tile)],
                         rf(9, tile*tile))
                self.vec("mul", [rf(9, tile*tile), imm(sqrt(2 / 3.141592653589793))],
                         rf(9, tile*tile))
                self.sfu("tanh", rf(9, tile*tile), rf(9, tile*tile))
                self.vec("add", [rf(9, tile*tile), imm(1)], rf(9, tile*tile))
                self.vec("mul", [rf(8, tile*tile), imm(0.5)], rf(8, tile*tile))
                self.vec("mul", [rf(8, tile*tile), rf(9, tile*tile)], rf(8, tile*tile))
                self.st(
                    rf(8, tile*tile),
                    sh(neuron, tile*tile, [tile, tile], [f, 1]),
                )

            with self.loop(name + "z", 0, d, tile) as col:
                self.vec("add", [imm(0), imm(0)],
                         rf(add(4, {"ceildiv": [col, tile]}), tile*tile))
            with self.loop(name + "w2k", 0, f, tile) as neuron:
                self.ld(
                    sh(neuron, tile*tile, [tile, tile], [f, 1]),
                    rf(0, tile*tile),
                )
                with self.loop(name + "o", 0, d, tile) as col:
                    self.ld(
                        hbm(add(w2.offset, mul(neuron, d), col), tile*tile,
                            [tile, tile], [d, 1]),
                        rf(1, tile*tile),
                    )
                    self.emit(
                        "MMA.ACC", a=rf(0, tile*tile), b=rf(1, tile*tile),
                        acc=rf(add(4, {"ceildiv": [col, tile]}), tile*tile),
                        m=tile, n=tile, k=tile, event=None,
                    )
            with self.loop(name + "store", 0, d, tile) as col:
                self.st(
                    rf(add(4, {"ceildiv": [col, tile]}), tile*tile),
                    hbm(add(out.offset, mul(row, d), col), tile*tile,
                        [tile, tile], [d, 1]),
                )
                self.emit("WAIT", wg=self.wg,
                          events=[json.loads(self.lines[-1].split(" ", 1)[1])["event"]])

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
        head_start=0,
        head_stop=None,
    ):
        if (self.recipe.get("attention", {}).get("algorithm") in
                ("blocked_prompt", "blocked_prompt_pipeline")
                and rows == 64 and past == 0):
            return self.attention_blocked(qkv, context, k_out, v_out, rows, past,
                                          d, h, hd, name, head_start, head_stop)
        head_stop = h if head_stop is None else head_stop
        # Export every layer's K/V in the public head-major ABI.
        with self.loop(f"{name}h", head_start, head_stop) as head:
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
            with self.loop(f"{name}head", head_start, head_stop) as head:
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

    def attention_blocked(self, qkv, context, k_out, v_out, rows, past,
                          d, h, hd, name, head_start=0, head_stop=None):
        """Prompt-only query-block attention; causal softmax remains row-local.

        RF4 holds 16x64 scores/probabilities, RF0 the 16x32 queries,
        RF1 a 32x32 K/V panel, RF2 the 16x32 context accumulator,
        RF3 the scalar reduction scratch. Strided RF views avoid packing events.
        All prompt KV is valid before QK; future probabilities are explicitly zero.
        """
        if (rows, past, hd) != (64, 0, 32) or self.scores_res != "rf":
            raise ValueError("Blocked prompt attention requires 64 rows, head width 32 and RF scores")
        head_stop = h if head_stop is None else head_stop
        qm, kn = 16, 32
        pipeline_panels = (
            self.recipe.get("attention", {}).get("algorithm") == "blocked_prompt_pipeline"
        )

        def matrix(lane, m, n, stride, offset=0):
            view = rf(lane, mul(m, n), offset)
            view.update(shape=[m, n], strides=[stride, 1])
            return view

        # Coalesced ABI export: gather 32 token rows instead of one head vector.
        with self.loop(f"{name}h", head_start, head_stop) as head:
            with self.loop(f"{name}r", 0, rows, kn) as row:
                for target, component in ((k_out, 1), (v_out, 2)):
                    source = add(qkv.offset, mul(row, 3*d), component*d, mul(head, hd))
                    dest = add(target.offset, mul(head, target.shape[2]*hd), mul(row, hd))
                    self.ld(hbm(source, kn*hd, [kn, hd], [3*d, 1]), rf(0, kn*hd))
                    self.st(rf(0, kn*hd), hbm(dest, kn*hd))
        with self.loop(f"{name}head", head_start, head_stop) as head:
            with self.loop(f"{name}qblock", 0, rows, qm) as first:
                self.ld(hbm(add(qkv.offset, mul(first, 3*d), mul(head, hd)),
                            qm*hd, [qm, hd], [3*d, 1]), rf(0, qm*hd))
                self.vec("add", [imm(0), imm(0)], rf(4, qm*rows))
                if pipeline_panels:
                    self.ld(
                        hbm(add(k_out.offset, mul(head, k_out.shape[2]*hd)),
                            hd*kn, [hd, kn], [1, hd]),
                        rf(1, hd*kn),
                    )
                with self.loop(f"{name}keys", 0, rows, kn) as key:
                    if pipeline_panels:
                        parity = {"ceildiv": [key, kn]}
                        self.emit("MMA.ACC", a=rf(0, qm*hd),
                                  b=rf(add(1, mul(4, parity)), hd*kn),
                                  acc=matrix(4, qm, kn, rows, key),
                                  m=qm, n=kn, k=hd, event=None)
                        with self.loop(f"{name}nextkey", add(key, kn), rows, rows) as nxt:
                            self.ld(
                                hbm(add(k_out.offset, mul(head, k_out.shape[2]*hd),
                                        mul(nxt, hd)), hd*kn, [hd, kn], [1, hd]),
                                rf(add(5, mul(-4, parity)), hd*kn),
                            )
                    else:
                        self.ld(hbm(add(k_out.offset, mul(head, k_out.shape[2]*hd), mul(key, hd)),
                                    hd*kn, [hd, kn], [1, hd]), rf(1, hd*kn))
                        self.emit("MMA.ACC", a=rf(0, qm*hd), b=rf(1, hd*kn),
                                  acc=matrix(4, qm, kn, rows, key), m=qm, n=kn, k=hd,
                                  event=None)
                self.vec("mul", [rf(4, qm*rows), imm(1/sqrt(hd))], rf(4, qm*rows))
                with self.loop(f"{name}softrow", 0, qm) as local:
                    limit = add(first, local, 1)
                    offset = mul(local, rows)
                    valid = rf(4, limit, offset)
                    self.reduce("max", valid, rf(3, 1))
                    self.vec("sub", [valid, rf(3, 1)], valid)
                    self.sfu("exp", valid, valid)
                    self.reduce("sum", valid, rf(3, 1))
                    self.vec("div", [valid, rf(3, 1)], valid)
                    # A zero-trip static loop avoids an invalid zero-count view.
                    with self.loop(f"{name}tail", limit, rows, rows) as tail:
                        self.vec("add", [imm(0), imm(0)],
                                 rf(4, sub(rows, tail), add(offset, tail)))
                self.vec("add", [imm(0), imm(0)], rf(2, qm*hd))
                if pipeline_panels:
                    self.ld(
                        hbm(add(v_out.offset, mul(head, v_out.shape[2]*hd)), kn*hd),
                        rf(1, kn*hd),
                    )
                with self.loop(f"{name}values", 0, rows, kn) as key:
                    if pipeline_panels:
                        parity = {"ceildiv": [key, kn]}
                        self.emit("MMA.ACC", a=matrix(4, qm, kn, rows, key),
                                  b=rf(add(1, mul(4, parity)), kn*hd),
                                  acc=rf(2, qm*hd), m=qm, n=hd, k=kn, event=None)
                        with self.loop(f"{name}nextvalue", add(key, kn), rows, rows) as nxt:
                            self.ld(
                                hbm(add(v_out.offset, mul(head, v_out.shape[2]*hd),
                                        mul(nxt, hd)), kn*hd),
                                rf(add(5, mul(-4, parity)), kn*hd),
                            )
                    else:
                        self.ld(hbm(add(v_out.offset, mul(head, v_out.shape[2]*hd), mul(key, hd)),
                                    kn*hd), rf(1, kn*hd))
                        self.emit("MMA.ACC", a=matrix(4, qm, kn, rows, key), b=rf(1, kn*hd),
                                  acc=rf(2, qm*hd), m=qm, n=hd, k=kn, event=None)
                self.st(rf(2, qm*hd), hbm(add(context.offset, mul(first, d), mul(head, hd)),
                                        qm*hd, [qm, hd], [d, 1]))

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
        head_start=0,
        head_stop=None,
    ):
        """One decode step: score historical and newly generated KV separately."""
        head_stop = h if head_stop is None else head_stop
        generated = step + 1
        total = past + generated
        rf_scores = self.scores_res == "rf"
        if not rf_scores:
            scores = self.alloc(f"{name}_scores", total)
            probs = self.alloc(f"{name}_probs", total)
        for kind, target, component in (("k", new_k, 1), ("v", new_v, 2)):
            with self.loop(f"{name}{kind}h", head_start, head_stop) as head:
                source = add(qkv.offset, component * d, mul(head, hd))
                destination = add(target.offset, mul(head, target.shape[2] * hd), step * hd)
                self.ld(hbm(source, hd), rf(0, hd))
                self.st(rf(0, hd), hbm(destination, hd))
        with self.loop(f"{name}head", head_start, head_stop) as head:
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
        for wg in self.groups:
            self.emit("WG.END", wg=wg)
        return "\n".join(self.lines) + "\n"


class IncrementalP1Builder(Builder):
    """Keep the mature P1 path; distribute selected prompt GEMMs by output column.

    Four persistent WG tapes, two coarse joins per selected GEMM, no per-tile
    cross-WG events/WAITs. All other stages remain on the primary WG.
    """
    def __init__(self, layout, recipe, shared_bytes=0):
        config = recipe["p1_schedule"]
        shards = int(config.get("shards", 4))
        stages = set(config["gemms"])
        if shards not in (2, 4) or not stages or not stages <= {"qkv", "w1", "w2"}:
            raise ValueError("Unsupported incremental P1 partition")
        if recipe.get("gemm_kernels"):
            raise ValueError("Incremental P1 uses the mature ordinary GEMM primitive")
        self.parallel_gemms = stages
        super().__init__(layout, recipe, shared_bytes=shared_bytes, groups=shards)

    def gemm(self, a, b, out, m_extent, k_extent, n_extent, name, col_start=0, col_stop=None):
        stage = next((key for key in self.parallel_gemms if name.endswith(key)), None)
        if self.stage != "p1_prompt" or stage is None:
            return super().gemm(a, b, out, m_extent, k_extent, n_extent, name, col_start, col_stop)
        if col_start or col_stop is not None:
            raise ValueError("Nested partition unsupported")
        self.emit("BARRIER", wgs=self.groups, events=[])
        stem = name[:-len(stage)]
        for index, wg in enumerate(self.groups):
            self.wg = wg
            self.label = f"{stem}_{stage}_sm{index}_"
            super().gemm(a, b, out, m_extent, k_extent, n_extent,
                         f"{stem}g{index}{stage}", n_extent*index//len(self.groups),
                         n_extent*(index+1)//len(self.groups))
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])


class ParallelD1Builder(Builder):
    """Persistent column/head owners across all causally committed decode steps."""
    def __init__(self, layout, recipe):
        shards = int(recipe["d1_schedule"].get("shards", 4))
        if shards not in (2, 4):
            raise ValueError("D1 supports 2 or 4 column/head shards")
        super().__init__(layout, recipe, groups=shards)

    def gemm(self, a, b, out, m_extent, k_extent, n_extent, name, col_start=0, col_stop=None):
        if col_start or col_stop is not None:
            raise ValueError("Nested D1 partition unsupported")
        self.emit("BARRIER", wgs=self.groups, events=[])
        old_tile, old_chunk = self.gemm_n_tile, self.k_chunk
        # Preserve 64-column panels: use 2 owners for width128, 3 for
        # width384, and 4 for width512 instead of doubling all small MMAs.
        panels = n_extent // old_tile
        owners = max(n for n in range(1, len(self.groups)+1) if panels % n == 0)
        stage = next(key for key in ("qkv", "wo", "w1", "w2") if name.endswith(key))
        plan = self.recipe["d1_schedule"].get("projections", {}).get(stage)
        if plan:
            owners = int(plan["owners"])
            self.gemm_n_tile, self.k_chunk = int(plan["n_tile"]), int(plan["k_chunk"])
            if (not 1 <= owners <= len(self.groups)
                    or self.gemm_n_tile <= 0 or self.k_chunk <= 0
                    or n_extent % (owners*self.gemm_n_tile)
                    or self.gemm_n_tile*self.k_chunk > 1024):
                raise ValueError("Invalid D1 projection partition/RF capacity")
        for index, wg in enumerate(self.groups[:owners]):
            self.wg = wg
            self.label = f"{name}_sm{index}_"
            super().gemm(a, b, out, m_extent, k_extent, n_extent, name + f"g{index}",
                         n_extent*index//owners, n_extent*(index+1)//owners)
        self.gemm_n_tile, self.k_chunk = old_tile, old_chunk
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])

    def attention_decode(self, qkv, context, history_k, history_v, new_k, new_v,
                         step, past, d, h, hd, name):
        self.emit("BARRIER", wgs=self.groups, events=[])
        for index, wg in enumerate(self.groups):
            self.wg = wg
            self.label = f"{name}_sm{index}_"
            super().attention_decode(qkv, context, history_k, history_v, new_k, new_v,
                                     step, past, d, h, hd, name + f"g{index}",
                                     h*index//len(self.groups), h*(index+1)//len(self.groups))
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])


def generate_m1_p1(recipe: dict | None = None) -> tuple[str, Layout, int]:
    recipe = recipe or load_recipe()
    if recipe.get("p1_schedule", {}).get("kind") == "column_head":
        return generate_m1_p1_multism(recipe)
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    layout = build_layout(model, "P1")
    recipe = recipe or load_recipe()
    kernel = recipe.get("gemm_kernels", {}).get("p1_prompt_w1")
    mlp_mode = recipe.get("fusion", {}).get("p1_mlp")
    if mlp_mode == "sh_resident":
        shared_bytes = 128 * 1024
    elif mlp_mode == "rf_pipeline_residual_sh_input":
        shared_bytes = 64 * 1024
    else:
        shared_bytes = 256 * int(kernel["n_panel"]) * 4 if kernel else 0
    builder = IncrementalP1Builder if recipe.get("p1_schedule", {}).get("kind") == "incremental_gemm" else Builder
    b = builder(layout, recipe, shared_bytes=shared_bytes)
    prompt = b.symbol("input/prompt")
    step = b.symbol("input/step0")
    output = b.symbol("output/hidden")
    def emit_layer(phase, rows, member, layer, x):
        prefix = f"layer{layer}/"
        name = f"{phase}b{member}l{layer}"
        weights = {key: b.symbol(prefix + key) for key in (
            "ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b",
            "w1", "b1", "w2", "b2",
        )}
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
        if phase == "p" and recipe.get("fusion", {}).get("p1_norm_residual"):
            b.layernorm_cached_params(
                x, weights["ln1_g"], weights["ln1_b"], ln1,
                rows, d, name + "ln1",
            )
        else:
            b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, rows, d, name + "ln1")
        b.gemm(ln1, weights["wqkv"], qkv, rows, d, 3 * d, name + "qkv")
        k_out, v_out = (b.symbol(prefix + f"new_{kind}") for kind in ("k", "v"))
        kv_offset = member * h * (prompt_rows + 1) * hd
        k_out = Tensor(k_out.offset + kv_offset, k_out.shape)
        v_out = Tensor(v_out.offset + kv_offset, v_out.shape)
        b.attention(
            qkv, ctx, k_out, v_out, rows,
            0 if phase == "p" else prompt_rows,
            d, h, hd, name + "a",
        )
        b.gemm(ctx, weights["wo"], att, rows, d, d, name + "wo")
        if phase == "p" and recipe.get("fusion", {}).get("p1_norm_residual"):
            ln2_target = (
                Tensor(0, (rows, d))
                if mlp_mode == "rf_pipeline_residual_sh_input" else ln2
            )
            b.residual_layernorm_cached(
                x, att, res, weights["ln2_g"], weights["ln2_b"], ln2_target,
                rows, d, name + "resln2",
                out_shared=mlp_mode == "rf_pipeline_residual_sh_input",
            )
        else:
            b.add_rows(x, att, res, rows, d, name + "res")
            b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, rows, d, name + "ln2")
        if phase == "p" and mlp_mode in (
            "rf_stream", "rf_pipeline", "rf_pipeline_residual",
            "rf_pipeline_residual_sh_input", "sh_resident",
        ):
            if mlp_mode == "sh_resident":
                b.mlp_sh_resident(
                    ln2, weights["w1"], weights["b1"], weights["w2"],
                    ff2, rows, d, f, name + "mlp",
                )
            elif mlp_mode in ("rf_pipeline_residual", "rf_pipeline_residual_sh_input"):
                source = Tensor(0, (rows, d)) if mlp_mode.endswith("sh_input") else ln2
                b.mlp_rf_stream(
                    source, weights["w1"], weights["b1"], weights["w2"],
                    y, rows, d, f, name + "mlp",
                    residual=res, output_bias=weights["b2"],
                    source_shared=mlp_mode.endswith("sh_input"),
                )
            else:
                b.mlp_rf_stream(
                    ln2, weights["w1"], weights["b1"], weights["w2"],
                    ff2, rows, d, f, name + "mlp",
                )
        else:
            b.gemm(ln2, weights["w1"], ff1, rows, d, f, name + "w1")
            if b.fuse_bias_gelu:
                b.bias_gelu(ff1, weights["b1"], activated, rows, f, name + "biasgelu")
            else:
                b.add_bias(ff1, weights["b1"], biased, rows, f, name + "bias1")
                b.gelu(biased, activated, rows, f, name + "gelu")
            b.gemm(activated, weights["w2"], ff2, rows, f, d, name + "w2")
        if not (phase == "p" and mlp_mode in (
            "rf_pipeline_residual", "rf_pipeline_residual_sh_input",
        )):
            b.add_rows(res, ff2, y, rows, d, name + "out", weights["b2"])
        return y

    # All prompts finish before STEP.COMMIT releases either batch member's first token.
    for phase, rows in (("p", prompt_rows), ("s", 1)):
        b.set_gemm_stage("p1_prompt" if phase == "p" else "p1_step")
        xs = [
            Tensor(prompt.offset + member * prompt_rows * d, (rows, d))
            if phase == "p" else Tensor(step.offset + member * d, (rows, d))
            for member in range(batch)
        ]
        def emit_output(member, x):
            with b.loop(f"{phase}b{member}output", 0, rows) as row:
                b.ld(hbm(add(x.offset, mul(row, d)), d), rf(0, d))
                target = add(
                    output.offset, member * (prompt_rows + 1) * d,
                    (0 if phase == "p" else prompt_rows) * d, mul(row, d),
                )
                b.st(rf(0, d), hbm(target, d))
        if recipe.get("p1_order") == "layer_major":
            for layer in range(model.layers):
                for member in range(batch):
                    xs[member] = emit_layer(phase, rows, member, layer, xs[member])
            for member, x in enumerate(xs):
                emit_output(member, x)
        else:
            for member in range(batch):
                for layer in range(model.layers):
                    xs[member] = emit_layer(phase, rows, member, layer, xs[member])
                emit_output(member, xs[member])
        if phase == "p":
            b.emit("STEP.COMMIT", step=0)
    program = b.finish()
    return program, layout, b.cursor


def generate_m1_p1_multism(recipe):
    """Column GEMMs + head attention + row vector stages, joined at dependencies."""
    shards = int(recipe["p1_schedule"].get("shards", 4))
    if shards not in (2, 4):
        raise ValueError("P1 column/head schedule supports 2 or 4 SMs")
    if recipe.get("gemm_kernels") or not recipe["fusion"]["bias_gelu"]:
        raise ValueError("multi-SM skeleton requires ordinary GEMM and fused bias/GELU")
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    layout = build_layout(model, "P1")
    serial_control = recipe["p1_schedule"].get("placement", "parallel") == "serial"
    b = Builder(layout, recipe, groups=1 if serial_control else shards)
    prompt, step, output = (b.symbol(key) for key in
                            ("input/prompt", "input/step0", "output/hidden"))

    def join():
        b.emit("BARRIER", wgs=b.groups, events=[])

    def group(index, label):
        b.wg = b.groups[0 if serial_control else index]
        b.label = label + f"_sm{index}_"

    def sliced(tensor, start, count, width=d):
        return Tensor(tensor.offset + start * width, (count, width))

    def row_parts():
        for i in range(shards):
            start, stop = rows * i // shards, rows * (i + 1) // shards
            # Single-token stages stay on SM0.
            if rows == 1:
                start, stop = (0, 1) if i == 0 else (1, 1)
            if stop > start:
                yield i, start, stop - start

    def gemm_stage(a, weight, out, k, n, stage):
        for i in range(shards):
            group(i, name + "_" + stage)
            b.gemm(a, weight, out, rows, k, n, name + f"g{i}" + stage,
                   n * i // shards, n * (i + 1) // shards)

    for phase, rows in (("p", prompt_rows), ("s", 1)):
        b.set_gemm_stage("p1_prompt" if phase == "p" else "p1_step")
        for member in range(batch):
            x = Tensor(prompt.offset + member * prompt_rows * d, (rows, d)) if phase == "p" else Tensor(step.offset + member * d, (rows, d))
            for layer in range(model.layers):
                name = f"{phase}b{member}l{layer}"
                prefix = f"layer{layer}/"
                w = {key: b.symbol(prefix + key) for key in
                     ("ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b", "w1", "b1", "w2", "b2")}
                t = {key: b.alloc(name + "_" + key, rows, width) for key, width in
                     (("ln1", d), ("qkv", 3*d), ("ctx", d), ("att", d), ("res", d),
                      ("ln2", d), ("ff1", f), ("gelu", f), ("ff2", d), ("out", d))}
                for i, start, count in row_parts():
                    group(i, name + "_ln1")
                    b.layernorm(sliced(x, start, count), w["ln1_g"], w["ln1_b"],
                                sliced(t["ln1"], start, count), count, d, name + f"g{i}ln1")
                join()
                gemm_stage(t["ln1"], w["wqkv"], t["qkv"], d, 3*d, "qkv")
                join()
                kv = [b.symbol(prefix + f"new_{kind}") for kind in ("k", "v")]
                kv = [Tensor(v.offset + member*h*(prompt_rows+1)*hd, v.shape) for v in kv]
                for i in range(shards):
                    group(i, name + "_attention")
                    b.attention(t["qkv"], t["ctx"], *kv, rows,
                                0 if phase == "p" else prompt_rows, d, h, hd,
                                name + f"g{i}a", h*i//shards, h*(i+1)//shards)
                join()
                gemm_stage(t["ctx"], w["wo"], t["att"], d, d, "wo")
                join()
                for i, start, count in row_parts():
                    group(i, name + "_residual1")
                    b.add_rows(sliced(x, start, count), sliced(t["att"], start, count),
                               sliced(t["res"], start, count), count, d, name + f"g{i}res")
                    group(i, name + "_ln2")
                    b.layernorm(sliced(t["res"], start, count), w["ln2_g"], w["ln2_b"],
                                sliced(t["ln2"], start, count), count, d, name + f"g{i}ln2")
                join()
                gemm_stage(t["ln2"], w["w1"], t["ff1"], d, f, "w1")
                for i in range(shards):
                    group(i, name + "_gelu")
                    b.bias_gelu(t["ff1"], w["b1"], t["gelu"], rows, f, name + f"g{i}biasgelu",
                                f*i//shards, f*(i+1)//shards)
                join()
                gemm_stage(t["gelu"], w["w2"], t["ff2"], f, d, "w2")
                join()
                for i, start, count in row_parts():
                    group(i, name + "_residual2")
                    b.add_rows(sliced(t["res"], start, count), sliced(t["ff2"], start, count),
                               sliced(t["out"], start, count), count, d, name + f"g{i}out", w["b2"])
                x = t["out"]
            for i, start, count in row_parts():
                group(i, f"{phase}b{member}_output")
                with b.loop(f"{phase}b{member}g{i}output", start, start+count) as row:
                    b.ld(hbm(add(x.offset, mul(row, d)), d), rf(0, d))
                    b.st(rf(0, d), hbm(add(output.offset, member*(prompt_rows+1)*d,
                         (0 if phase == "p" else prompt_rows)*d, mul(row, d)), d))
        if phase == "p":
            b.emit("STEP.COMMIT", step=0)
    return b.finish(), layout, b.cursor


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
    builder = ParallelD1Builder if recipe.get("d1_schedule", {}).get("kind") == "column_head" else Builder
    b = builder(layout, recipe)
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
