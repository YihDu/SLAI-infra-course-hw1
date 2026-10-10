"""Generate literal-ISA baselines for the two active trial cases.

The Python here is a compiler only.  The emitted program uses no operator-level
instructions for LayerNorm, attention, softmax, or GELU.
"""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from math import prod, sqrt
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


def d1_rf_tile(base, per_lane, index, cells):
    """RF view of packed resident tile `index`: lane base+index//per_lane."""
    if per_lane == 1:
        return rf(add(base, index), cells)
    if per_lane not in (2,4,8,16):
        raise ValueError("RF-resident packing must divide the lane")
    # floor(i/p) == ceildiv(i-(p-1), p) for i >= 0.
    return rf(add(base, {"ceildiv": [sub(index, per_lane-1), per_lane]}),
              cells, mul(mod(index, per_lane), cells))


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
    def __init__(self, layout: Layout, recipe: dict | None = None, shared_bytes: int = 0,
                 groups: int = 1, sm_map: list[int] | None = None):
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
        for index, wg in enumerate(self.groups):
            sm = sm_map[index] if sm_map is not None else index
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
        epilogue_bias: Tensor | None = None,
    ):
        col_stop = n_extent if col_stop is None else col_stop
        if b.shape != (k_extent, n_extent):
            raise ValueError("GEMM weight shape")
        if epilogue_bias is not None and self.order != "nkm_lanes" and self.order != "nkm":
            raise ValueError("bias+GELU epilogue requires an N/K/M GEMM order")
        pipelines = self.recipe.get("gemm_stages", {}).get("p1_prompt", {}).get(
            "operand_pipeline", []
        )
        pipeline_stage = next(
            (candidate for candidate in pipelines if name.endswith(candidate)), None
        )
        paired_stages = self.recipe.get("gemm_stages", {}).get("p1_prompt", {}).get(
            "paired_n", []
        )
        if self.stage == "p1_prompt" and any(name.endswith(s) for s in paired_stages):
            self.gemm_prompt_paired_n(a, b, out, m_extent, k_extent, n_extent,
                                      name, col_start, col_stop, epilogue_bias)
            return
        resident_stages=self.recipe.get("gemm_stages",{}).get("p1_prompt",{}).get("a_resident",[])
        if self.stage=="p1_prompt" and any(name.endswith(s) for s in resident_stages):
            self.gemm_prompt_a_resident(a,b,out,m_extent,k_extent,n_extent,name,
                                        col_start,col_stop,epilogue_bias)
            return
        if self.stage == "p1_prompt" and pipeline_stage:
            self.gemm_operand_pipeline(
                a, b, out, m_extent, k_extent, n_extent, name, col_start, col_stop,
                epilogue_bias,
            )
            return
        if (self.stage == "p1_step" and
                self.recipe.get("gemm_stages", {}).get("p1_step", {}).get("a_resident")):
            self.gemm_step_a_resident(a,b,out,m_extent,k_extent,n_extent,name,
                                      col_start,col_stop,epilogue_bias)
            return
        if (self.stage == "d1"
                and self.recipe.get("d1_schedule", {}).get("operand_pipeline")):
            self.gemm_d1_operand_pipeline(
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
        if epilogue_bias is not None:
            self.vec("add", [imm(1), imm(0)], rf(12, m_tile))
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
            if epilogue_bias is not None:
                self.ld(hbm(add(epilogue_bias.offset, col), tile_n), rf(13, tile_n))
            with self.loop(f"{name}s", 0, m_extent, m_tile) as row:
                rows = minimum(m_tile, sub(m_extent, row))
                cells = mul(rows, tile_n)
                if epilogue_bias is not None:
                    self.bias_gelu_epilogue(accumulator(row, rows), rows, cells, tile_n)
                self.st(
                    accumulator(row, rows),
                    hbm(
                        add(out.offset, mul(row, n_extent), col),
                        cells,
                        [rows, tile_n],
                        [n_extent, 1],
                    ),
                )

    def gemm_prompt_paired_n(self, a, b, out, rows, k_extent, n_extent,
                             name, col_start, col_stop, epilogue_bias=None):
        """Share each 32x32 A tile across two adjacent W1 output panels.

        Four accumulators retain both row blocks and column panels. B panels
        are prefetched to alternating RF pairs; GELU runs on each live result.
        """
        if (rows, k_extent, n_extent) != (64, 256, 1024):
            raise ValueError("Paired N kernel requires prompt W1 dimensions")
        cfg = self.recipe.get("gemm_stages", {}).get("p1_prompt", {})
        chunk = int(cfg.get("paired_k_chunk", self.k_chunk))
        if self.m_tile != 32 or chunk not in (16, 32) or self.gemm_n_tile != 32:
            raise ValueError("Paired N kernel requires M32 K16/32 N32")
        if col_start % 64 or col_stop % 64 or not 0 <= col_start < col_stop <= n_extent:
            raise ValueError("Paired N ownership requires 64 aligned columns")
        if epilogue_bias is None:
            raise ValueError("Paired N W1 requires fused bias and GELU")
        a_preload = bool(cfg.get("paired_a_preload", 0))
        b_prefetch = bool(cfg.get("paired_b_prefetch", 1))
        cells = chunk * 32

        def load_next_weights(inner, parity, col):
            with self.loop(name + "nextb", add(inner, chunk), k_extent,
                           k_extent) as nxt:
                self.ld(hbm(add(b.offset, mul(nxt, n_extent), col),
                            cells, [chunk, 32], [n_extent, 1]),
                        rf(add(1, mul(5, sub(1, parity))), cells))
                self.ld(hbm(add(b.offset, mul(nxt, n_extent), col, 32),
                            cells, [chunk, 32], [n_extent, 1]),
                        rf(add(3, mul(4, sub(1, parity))), cells))

        self.vec("add", [imm(1), imm(0)], rf(12, 32))
        with self.loop(name + "pair", col_start, col_stop, 64) as col:
            for lane in (4, 5, 8, 9):
                self.vec("add", [imm(0), imm(0)], rf(lane, 1024))
            self.ld(hbm(add(b.offset, col), cells, [chunk, 32], [n_extent, 1]),
                    rf(1, cells))
            self.ld(hbm(add(b.offset, col, 32), cells, [chunk, 32], [n_extent, 1]),
                    rf(3, cells))
            with self.loop(name + "k", 0, k_extent, chunk) as inner:
                parity = mod({"ceildiv": [inner, chunk]}, 2)
                left = add(1, mul(5, parity))
                right = add(3, mul(4, parity))
                if b_prefetch:
                    load_next_weights(inner, parity, col)
                a_sh = (self.stage == "p1_prompt" and any(
                    name.endswith(sfx) for sfx in getattr(self, "act_sh", ())))
                def a_src(off):
                    if a_sh:
                        return sh(off, 32 * chunk, [32, chunk], [k_extent, 1])
                    return hbm(add(a.offset, off), 32 * chunk,
                               [32, chunk], [k_extent, 1])
                if a_preload:
                    for row_index in (0, 1):
                        self.ld(a_src(add(row_index * 32 * k_extent, inner)),
                                rf(2 * row_index, 32 * chunk))
                for row_index in (0, 1):
                    a_lane = 2 * row_index
                    acc_left = 4 + 4 * row_index
                    if not a_preload:
                        self.ld(a_src(add(row_index * 32 * k_extent, inner)),
                                rf(a_lane, 32 * chunk))
                    self.emit("MMA.ACC", a=rf(a_lane, 32 * chunk), b=rf(left, cells),
                              acc=rf(acc_left, 1024), m=32, n=32, k=chunk,
                              event=None)
                    self.emit("MMA.ACC", a=rf(a_lane, 32 * chunk), b=rf(right, cells),
                              acc=rf(acc_left + 1, 1024), m=32, n=32, k=chunk,
                              event=None)
                if not b_prefetch:
                    load_next_weights(inner, parity, col)
            for panel in (0, 1):
                self.ld(hbm(add(epilogue_bias.offset, col, panel * 32), 32),
                        rf(13, 32))
                for row_index in (0, 1):
                    acc = rf(4 + 4 * row_index + panel, 1024)
                    self.bias_gelu_epilogue(acc, 32, 1024, 32)
                    self.st(acc, hbm(add(out.offset, row_index * 32 * n_extent,
                                         col, panel * 32),
                                     1024, [32, 32], [n_extent, 1]))

    def gemm_prompt_a_resident(self,a,b,out,rows,k_extent,n_extent,name,
                               col_start,col_stop,epilogue_bias=None):
        """Reuse a 32x256 RF input block across a WG's output-column panels."""
        if rows!=64 or k_extent!=256 or self.gemm_n_tile!=32 or self.k_chunk!=32:
            raise ValueError("Prompt RF A residency requires M64 K256 Ntile32 Ktile32")
        # wstage: serve staged B tiles from SH when this WG's slice was
        # prefetched during the previous stage.
        wsh = getattr(self, "p1_sh", {}).get((self.wg, b.offset, col_start))
        if wsh is not None:
            sh_base, ncols = wsh
            def bsrc(inner, col):
                return sh(add(sh_base, mul(inner, ncols), sub(col, col_start)),
                          1024, [32, 32], [ncols, 1])
        else:
            def bsrc(inner, col):
                return hbm(add(b.offset, mul(inner, n_extent), col), 1024,
                           [32, 32], [n_extent, 1])
        with self.loop(name+"m",0,rows,32) as row:
            with self.loop(name+"aload",0,256,32) as inner:
                self.ld(hbm(add(a.offset,mul(row,k_extent),inner),1024,[32,32],[k_extent,1]),
                        rf({"ceildiv":[inner,32]},1024))
            if epilogue_bias is not None:
                self.vec("add",[imm(1),imm(0)],rf(12,32))
            with self.loop(name+"n",col_start,col_stop,32) as col:
                self.vec("add",[imm(0),imm(0)],rf(8,1024))
                self.ld(bsrc(0, col),rf(9,1024))
                with self.loop(name+"k",0,256,32) as inner:
                    idx={"ceildiv":[inner,32]}
                    parity=mod(idx,2)
                    with self.loop(name+"nextb",add(inner,32),256,256) as nxt:
                        self.ld(bsrc(nxt, col),
                                rf(add(9,sub(1,parity)),1024))
                    self.emit("MMA.ACC",a=rf(idx,1024),b=rf(add(9,parity),1024),
                              acc=rf(8,1024),m=32,n=32,k=32,event=None)
                if epilogue_bias is not None:
                    self.ld(hbm(add(epilogue_bias.offset,col),32),rf(13,32))
                    self.bias_gelu_epilogue(rf(8,1024),32,1024,32)
                self.st(rf(8,1024),hbm(add(out.offset,mul(row,n_extent),col),1024,
                                     [32,32],[n_extent,1]))

    def gemm_step_a_resident(self,a,b,out,rows,k_extent,n_extent,name,
                             col_start,col_stop,epilogue_bias=None):
        """Keep the two generated-position inputs in RF across all K/N tiles.

        A is split into <=1024-word row-major panels in RF0/1. RF2/3 buffer B;
        RF4 accumulates both rows. Shaped RF views select each K chunk without
        another HBM transfer. This preserves ascending K accumulation order.
        """
        n,k=self.gemm_n_tile,self.k_chunk
        span=min(k_extent,1024//rows)
        if rows not in (1,2) or k_extent%span or span%k or k*n>1024 or k_extent//span>2:
            raise ValueError("Unsupported resident step shape")
        chunks = k_extent // k
        nbuf = min(max(int(self.recipe.get("gemm_stages", {}).get(
            "p1_step", {}).get("b_bufs", 2)), 2), chunks)
        acc_lane = 10 if nbuf > 2 else 4
        # commit_overlap_prefetch: if this WG's weight column slice was staged
        # into its private SH before STEP.COMMIT, B tiles copy SH->RF instead.
        # The prefetch packs the [k_extent, ncols] slice contiguous, so a tile
        # (inner, col) lives at sh_base + inner*ncols + (col - col_start).
        sh_slice = getattr(self, "step_sh", {}).get((self.wg, b.offset, col_start))
        def b_src(inner):
            if sh_slice is None:
                return hbm(add(b.offset, mul(inner, n_extent), col),
                           k*n, [k,n], [n_extent,1])
            sh_base, ncols = sh_slice
            return sh(add(sh_base, mul(inner, ncols), sub(col, col_start)),
                      k*n, [k,n], [ncols,1])
        for panel in range(k_extent//span):
            self.ld(hbm(a.offset+panel*span,rows*span,[rows,span],[k_extent,1]),
                    rf(panel,rows*span))
        if epilogue_bias is not None:
            self.vec("add",[imm(1),imm(0)],rf(12,rows))
        with self.loop(name+"n",col_start,col_stop,n) as col:
            self.vec("add",[imm(0),imm(0)],rf(acc_lane,rows*n))
            self.ld(b_src(0),rf(2,k*n))
            if nbuf > 2:
                # Software pipeline: keep nbuf-1 B loads in flight (bounded by
                # dma_engines*dma_depth slots), each freed lane refilled a
                # chunk before its MMA consumes it.
                with self.loop(name+"seed", k, mul(nbuf - 1, k), k) as sd:
                    self.ld(b_src(sd), rf(add(2, mod({"ceildiv": [sd, k]}, nbuf)), k * n))
            with self.loop(name+"k",0,k_extent,k) as inner:
                chunk=mod({"ceildiv":[inner,k]},nbuf)
                with self.loop(name+"nextb",minimum(add(inner,mul(nbuf-1,k)),k_extent),k_extent,k_extent) as nxt:
                    self.ld(b_src(nxt),rf(add(2,mod({"ceildiv":[nxt,k]},nbuf)),k*n))
                panel={"ceildiv":[sub(inner,span-1),span]}
                a_view=rf(panel,rows*k,mod(inner,span))
                a_view.update(shape=[rows,k],strides=[span,1])
                self.emit("MMA.ACC",a=a_view,b=rf(add(2,chunk),k*n),acc=rf(acc_lane,rows*n),
                          m=rows,n=n,k=k,event=None)
            if epilogue_bias is not None:
                self.ld(hbm(add(epilogue_bias.offset,col),n),rf(13,n))
                self.bias_gelu_epilogue(rf(acc_lane,rows*n),rows,rows*n,n)
            self.st(rf(acc_lane,rows*n),hbm(add(out.offset,col),rows*n,[rows,n],[n_extent,1]))

    def bias_gelu_epilogue(self, acc, rows, cells, tile_n, lanes=(12, 13, 14)):
        """acc += 1 x bias (lanes: ones, bias, temp), then tanh-GELU in place.

        Same operation order as `bias_gelu`, applied to the live accumulator so
        the pre-activation never round-trips HBM.
        """
        ones, bias, temp = lanes
        self.emit("MMA.ACC", a=rf(ones, rows), b=rf(bias, tile_n), acc=acc,
                  m=rows, n=tile_n, k=1, event=None)
        t = rf(temp, cells)
        self.vec("mul", [acc, acc], t)
        self.vec("mul", [t, acc], t)
        self.vec("fma", [t, imm(0.044715), acc], t)
        self.vec("mul", [t, imm(sqrt(2 / 3.141592653589793))], t)
        self.sfu("tanh", t, t)
        self.vec("add", [t, imm(1)], t)
        self.vec("mul", [acc, imm(0.5)], acc)
        self.vec("mul", [acc, t], acc)

    def gemm_operand_pipeline(
        self, a, b, out, m_extent, k_extent, n_extent, name,
        col_start=0, col_stop=None, epilogue_bias=None,
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
        merged = m_extent == 128
        if (m_extent not in (64, 128) or m_tile != 32 or tile_n not in (16, 32)
                or k_chunk != 32):
            raise ValueError("P1 operand pipeline requires M64/128 Mtile32 N16/32 K32")
        if not 0 <= col_start < col_stop <= n_extent or col_start % tile_n or col_stop % tile_n:
            raise ValueError("P1 operand pipeline N extent does not fit tile")
        a_sh = self.stage == "p1_prompt" and any(
            name.endswith(sfx) for sfx in getattr(self, "act_sh", ()))
        a_memory = (
            sh if (name.endswith("qkv")
                   and self.recipe.get("fusion", {}).get("p1_ln1_sh"))
            else hbm
        )
        if a_sh:
            base = a.offset
            a_memory = lambda off, cnt, shp, st: sh(
                sub(off, base), cnt, shp, st)

        def accumulator(row):
            return rf(add(8 if merged else 4, {"ceildiv": [row, m_tile]}), m_tile * tile_n)

        def a_lane(row, parity):
            if merged:
                return add(mul(4, parity), {"ceildiv": [row, m_tile]})
            return add(mul(6, parity), mul(2, {"ceildiv": [row, m_tile]}))

        # b_buffers=3 rotates B through RF1/7/13 so the next-chunk B load never
        # overwrites the panel the two just-issued MMAs are still reading.
        b_buffers = int(self.recipe.get("gemm_stages", {}).get("p1_prompt", {})
                        .get("b_buffers", 2))
        if b_buffers not in (2, 3):
            raise ValueError("P1 operand pipeline supports 2 or 3 B buffers")
        b_stride = 2 if b_buffers == 2 else 6
        epi_lanes = (12, 13, 14) if b_buffers == 2 else (10, 11, 15)
        if merged:
            epi_lanes = (0, 1, 2)

        def b_lane(chunk):
            if merged:
                return add(12, mod(chunk, b_buffers))
            return add(1, mul(b_stride, mod(chunk, b_buffers)))

        # wstage: when this WG's B slice was prefetched into SH during the
        # previous stage, read B tiles from the staged [k, nc] block instead
        # of issuing HBM reads at consumption time.
        wsh = getattr(self, "p1_sh", {}).get((self.wg, b.offset, col_start))

        def b_source(row):
            if wsh is not None:
                sh_base, ncols = wsh
                return sh(add(sh_base, mul(row, ncols), sub(col, col_start)),
                          k_chunk * tile_n, [k_chunk, tile_n], [ncols, 1])
            return hbm(add(b.offset, mul(row, n_extent), col), k_chunk * tile_n,
                       [k_chunk, tile_n], [n_extent, 1])

        if epilogue_bias is not None and not merged:
            self.vec("add", [imm(1), imm(0)], rf(epi_lanes[0], m_tile))
        with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
            with self.loop(f"{name}z", 0, m_extent, m_tile) as row:
                self.vec("add", [imm(0), imm(0)], accumulator(row))

            # Seed K=0 operands. All extents divide exactly for the supported
            # prompt projections, so the descriptor sizes stay constant.
            self.ld(
                b_source(0) if wsh is not None else
                hbm(add(b.offset, col), k_chunk * tile_n,
                    [k_chunk, tile_n], [n_extent, 1]),
                rf(12 if merged else 1, k_chunk * tile_n),
            )
            with self.loop(f"{name}seedm", 0, m_extent, m_tile) as row:
                self.ld(
                    a_memory(add(a.offset, mul(row, k_extent)), m_tile * k_chunk,
                             [m_tile, k_chunk], [k_extent, 1]),
                    rf(a_lane(row, 0), m_tile * k_chunk),
                )

            with self.loop(f"{name}k", 0, k_extent, k_chunk) as inner:
                chunk = {"ceildiv": [inner, k_chunk]}
                parity = mod(chunk, 2)
                next_parity = sub(1, parity)
                with self.loop(f"{name}nextb", add(inner, k_chunk), k_extent, k_extent) as nxt:
                    self.ld(
                        b_source(nxt) if wsh is not None else
                        hbm(add(b.offset, mul(nxt, n_extent), col), k_chunk * tile_n,
                            [k_chunk, tile_n], [n_extent, 1]),
                        rf(b_lane(add(chunk, 1)), k_chunk * tile_n),
                    )
                with self.loop(f"{name}m", 0, m_extent, m_tile) as row:
                    self.emit(
                        "MMA.ACC",
                        a=rf(a_lane(row, parity), m_tile * k_chunk),
                        b=rf(b_lane(chunk), k_chunk * tile_n),
                        acc=accumulator(row), m=m_tile, n=tile_n, k=k_chunk,
                        event=None,
                    )
                    with self.loop(f"{name}nexta", add(inner, k_chunk), k_extent, k_extent) as nxt:
                        self.ld(
                            a_memory(add(a.offset, mul(row, k_extent), nxt), m_tile * k_chunk,
                                     [m_tile, k_chunk], [k_extent, 1]),
                            rf(a_lane(row, next_parity), m_tile * k_chunk),
                        )
            if epilogue_bias is not None:
                if merged:
                    self.vec("add", [imm(1), imm(0)], rf(epi_lanes[0], m_tile))
                self.ld(hbm(add(epilogue_bias.offset, col), tile_n),
                        rf(epi_lanes[1], tile_n))
            with self.loop(f"{name}s", 0, m_extent, m_tile) as row:
                if epilogue_bias is not None:
                    self.bias_gelu_epilogue(accumulator(row), m_tile, m_tile * tile_n,
                                            tile_n, epi_lanes)
                self.st(
                    accumulator(row),
                    hbm(add(out.offset, mul(row, n_extent), col), m_tile * tile_n,
                        [m_tile, tile_n], [n_extent, 1]),
                )

    def gemm_d1_operand_pipeline(
        self, a, b, out, m_extent, k_extent, n_extent, name,
        col_start=0, col_stop=None,
    ):
        """Double-buffer single-row D1 operands without changing K grouping."""
        col_stop = n_extent if col_stop is None else col_stop
        tile_n, k_chunk = self.gemm_n_tile, self.k_chunk
        a_resident = (
            self.recipe.get("d1_schedule", {}).get("operand_pipeline")
            == "a_resident"
        )
        if (m_extent != 1 or b.shape != (k_extent, n_extent)
                or col_start % tile_n or col_stop % tile_n
                or k_extent % k_chunk or k_chunk * tile_n > 1024
                or (a_resident and k_extent > 1024)):
            raise ValueError("unsupported D1 operand pipeline shape")
        # SH-resident weights: panels preloaded once by `d1_preload`; B tile
        # (col, inner) lives contiguously at base + (col_index*chunks + k_index)*tile.
        sh_base = getattr(self, "d1_sh_panels", {}).get((self.wg, b.offset, col_start))
        chunks = k_extent // k_chunk

        def store_projection(acc, col):
            epilogue = getattr(self, "d1_gelu_epilogue", None)
            residual = getattr(self, "d1_residual_epilogue", None)
            if residual is not None:
                # out = (res + ff2) + b2, the add_rows order; res row stays in
                # RF15[896:1024) from this owner's W1 prologue.
                bias, target = residual
                self.ld(hbm(add(bias.offset, col), tile_n), rf(2, tile_n))
                self.vec("add", [rf(15, tile_n, add(896, col)), acc], acc)
                self.vec("add", [acc, rf(2, tile_n)], acc)
                self.st(acc, hbm(add(target.offset, col), tile_n))
                return
            if epilogue is None:
                self.st(acc, hbm(add(out.offset, col), tile_n))
                return
            bias, activated = epilogue
            # RF1/2 are dead operand buffers here; resident weights start at 5.
            # bias None: already folded into this owner's beta W columns.
            cached=getattr(self,"d1_rf_bias",{}).get((self.wg,bias.offset)) if bias else None
            if cached is not None:
                start,offset=cached
                bias_view=rf(14,tile_n,add(offset,sub(col,start)))
            elif bias is not None:
                self.ld(hbm(add(bias.offset, col), tile_n), rf(2, tile_n))
                bias_view=rf(2,tile_n)
            if bias is not None:
                self.vec("add", [acc, bias_view], acc)
            self.vec("mul", [acc, acc], rf(1, tile_n))
            self.vec("mul", [rf(1, tile_n), acc], rf(1, tile_n))
            self.vec("fma", [rf(1, tile_n), imm(0.044715), acc], rf(1, tile_n))
            self.vec("mul", [rf(1, tile_n), imm(sqrt(2 / 3.141592653589793))], rf(1, tile_n))
            self.sfu("tanh", rf(1, tile_n), rf(1, tile_n))
            self.vec("add", [rf(1, tile_n), imm(1)], rf(1, tile_n))
            self.vec("mul", [acc, imm(0.5)], acc)
            self.vec("mul", [acc, rf(1, tile_n)], acc)
            self.st(acc, hbm(add(activated.offset, col), tile_n))

        def b_tile(col, inner):
            if sh_base is None:
                return hbm(add(b.offset, mul(inner, n_extent), col), k_chunk * tile_n,
                           [k_chunk, tile_n], [n_extent, 1])
            index = add(mul({"ceildiv": [sub(col, col_start), tile_n]}, chunks),
                        {"ceildiv": [inner, k_chunk]})
            return sh(add(sh_base, mul(index, k_chunk * tile_n)), k_chunk * tile_n)

        prologue = getattr(self, "d1_ln_prologue", None)
        fold = prologue is not None and prologue[0] == "fold"
        deferred = []

        # Folded LN keeps this owner's beta W columns in RF0[768:) (above
        # every A operand) from step 0 on; the cold step accumulates them
        # beside the projection, sharing each B tile.
        beta_cold = fold and prologue[5]
        beta_lane_offset = 768 + prologue[4] if fold else 0

        def beta_acc(col):
            return rf(0, tile_n, add(beta_lane_offset, sub(col, col_start)))

        chunked_a = bool(self.recipe.get("d1_schedule", {}).get("chunked_a", 0))

        def load_a(count):
            if prologue is None and chunked_a and count > k_chunk:
                # Per-chunk A copies let the first MMA start on chunk 0.
                for offset in range(0, count, k_chunk):
                    self.ld(hbm(a.offset + offset, k_chunk), rf(0, k_chunk, offset))
            elif prologue is None:
                self.ld(hbm(a.offset, count), rf(0, count))
            elif fold:
                deferred.append(self.d1_owner_layernorm_fold(*prologue[1:4]))
                if beta_cold:
                    self.ld(hbm(prologue[6].offset, count), rf(15, count, 640))
                if col_stop - col_start != tile_n:
                    deferred.pop()()
            else:
                self.d1_owner_layernorm(*prologue[1:])

        def beta_mma(inner, b_view, col):
            if beta_cold:
                self.emit("MMA.ACC", a=rf(15, k_chunk, add(640, inner)), b=b_view,
                          acc=beta_acc(col), m=1, n=tile_n, k=k_chunk, event=None)

        def zero_beta(col):
            if beta_cold:
                self.vec("add", [imm(0), imm(0)], beta_acc(col))

        def k_ranges():
            # With a folded LN, issue the first two MMAs before the variance
            # chain so it overlaps TC work instead of gating the projection.
            if deferred and k_extent > 2 * k_chunk:
                return [(f"{name}k", 0, 2 * k_chunk), (f"{name}kv", 2 * k_chunk, k_extent)]
            return [(f"{name}k", 0, k_extent)]

        def between_ranges():
            if deferred:
                deferred.pop()()

        raw_store = store_projection

        def store_projection(acc, col):
            if fold:
                between_ranges()
                folded_bias = prologue[7]
                if beta_cold and folded_bias is not None:
                    self.ld(hbm(add(folded_bias.offset, col), tile_n), rf(2, tile_n))
                    self.vec("add", [beta_acc(col), rf(2, tile_n)], beta_acc(col))
                self.vec("mul", [acc, rf(15, 1, 768)], acc)
                self.vec("add", [acc, beta_acc(col)], acc)
            raw_store(acc, col)

        acc_split = int(self.recipe.get("d1_schedule", {}).get("acc_split", 0))
        load_ahead = bool(self.recipe.get("d1_schedule", {}).get("load_ahead", 0))
        compact = bool(self.recipe.get("d1_schedule", {}).get("compact_rf", 0))
        def split_acc(parity):
            return (rf(3,tile_n,mul(128,parity)) if compact
                    else rf(3+parity if isinstance(parity,int) else add(3,parity),tile_n))
        rf_panel = getattr(self, "d1_rf_panels", {}).get((self.wg, b.offset, col_start))
        if rf_panel is not None:
            if not a_resident:
                raise ValueError("RF-resident D1 weights require a_resident")
            base, per_lane = rf_panel
            load_a(k_extent)
            with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
                self.vec("add", [imm(0), imm(0)], rf(3, tile_n))
                if acc_split:
                    self.vec("add", [imm(0), imm(0)], split_acc(1))
                zero_beta(col)
                for index, (loop_name, lo, hi) in enumerate(k_ranges()):
                    if index:
                        between_ranges()
                    with self.loop(loop_name, lo, hi, k_chunk) as inner:
                        idx = add(mul({"ceildiv": [sub(col, col_start), tile_n]}, chunks),
                                  {"ceildiv": [inner, k_chunk]})
                        b_view = d1_rf_tile(base, per_lane, idx, k_chunk * tile_n)
                        self.emit(
                            "MMA.ACC", a=rf(0, k_chunk, inner), b=b_view,
                            acc=split_acc(mod({"ceildiv": [inner,k_chunk]},2)) if acc_split else rf(3,tile_n),
                            m=1, n=tile_n, k=k_chunk, event=None,
                        )
                        beta_mma(inner, b_view, col)
                if acc_split:
                    self.vec("add", [rf(3, tile_n), split_acc(1)], rf(3, tile_n))
                store_projection(rf(3, tile_n), col)
            return
        cells = k_chunk * tile_n
        if (self.recipe.get("d1_schedule", {}).get("b_prefetch") and a_resident
                and sh_base is not None and col_stop - col_start == tile_n
                and chunks <= 4 and cells <= 768):
            # Single-tile SH owner: issue every B tile before the A prologue
            # into four disjoint RF homes, so no MMA waits on an SH copy.
            homes = [rf(1, cells), rf(2, cells), rf(14, cells), rf(4, cells, 256)]
            col = col_start
            for kk in range(chunks):
                self.ld(b_tile(col, kk * k_chunk), homes[kk])
            load_a(k_extent)
            self.vec("add", [imm(0), imm(0)], rf(3, tile_n))
            if acc_split:
                self.vec("add", [imm(0), imm(0)], split_acc(1))
            zero_beta(col)
            for kk in range(chunks):
                if kk == 2:
                    between_ranges()
                inner = kk * k_chunk
                self.emit("MMA.ACC", a=rf(0, k_chunk, inner), b=homes[kk],
                          acc=split_acc(kk % 2) if acc_split else rf(3, tile_n),
                          m=1, n=tile_n, k=k_chunk, event=None)
                beta_mma(inner, homes[kk], col)
            if acc_split:
                self.vec("add", [rf(3, tile_n), split_acc(1)], rf(3, tile_n))
            store_projection(rf(3, tile_n), col)
            return
        if a_resident:
            load_a(k_extent)
        with self.loop(f"{name}n", col_start, col_stop, tile_n) as col:
            acc_lane = 3 if a_resident else 4
            self.vec("add", [imm(0), imm(0)], rf(acc_lane, tile_n))
            if a_resident and acc_split:
                self.vec("add", [imm(0), imm(0)], split_acc(1))
            zero_beta(col)
            if not a_resident:
                self.ld(hbm(a.offset, k_chunk), rf(0, k_chunk))
            self.ld(
                b_tile(col, 0) if sh_base is not None else
                hbm(add(b.offset, col), k_chunk * tile_n,
                    [k_chunk, tile_n], [n_extent, 1]),
                rf(1, k_chunk * tile_n),
            )
            for index, (loop_name, lo, hi) in enumerate(k_ranges()):
                if index:
                    between_ranges()
                with self.loop(loop_name, lo, hi, k_chunk) as inner:
                    parity = mod({"ceildiv": [inner, k_chunk]}, 2)
                    a_view = (
                        rf(0, k_chunk, inner)
                        if a_resident else rf(mul(2, parity), k_chunk)
                    )
                    b_lane = (
                        add(1, parity)
                        if a_resident else add(1, mul(2, parity))
                    )
                    acc_op = (
                        split_acc(parity)
                        if a_resident and acc_split else rf(acc_lane, tile_n)
                    )
                    def compute_tile():
                        self.emit(
                            "MMA.ACC",
                            a=a_view, b=rf(b_lane, k_chunk * tile_n),
                            acc=acc_op, m=1, n=tile_n, k=k_chunk,
                            event=None,
                        )
                        beta_mma(inner, rf(b_lane, k_chunk * tile_n), col)

                    # Issue the independent next-buffer LD before the current
                    # MMA can block this tape on its operand/accumulator RAW.
                    # Local RF hazards still protect the previous buffer reader.
                    if not load_ahead:
                        compute_tile()
                    with self.loop(f"{name}prefetch", add(inner, k_chunk),
                                   k_extent, k_extent) as nxt:
                        next_parity = sub(1, parity)
                        if not a_resident:
                            self.ld(
                                hbm(add(a.offset, nxt), k_chunk),
                                rf(mul(2, next_parity), k_chunk),
                            )
                        self.ld(
                            b_tile(col, nxt) if sh_base is not None else
                            hbm(add(b.offset, mul(nxt, n_extent), col),
                                k_chunk * tile_n, [k_chunk, tile_n], [n_extent, 1]),
                            rf((add(1, next_parity) if a_resident
                                else add(1, mul(2, next_parity))), k_chunk * tile_n),
                        )
                    if load_ahead:
                        compute_tile()
            if a_resident and acc_split:
                self.vec("add", [rf(3, tile_n), split_acc(1)], rf(3, tile_n))
            store_projection(rf(acc_lane, tile_n), col)

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
        rows: int, d: int, name: str, out_shared=False, in_shared=None,
    ):
        """Keep prompt LayerNorm affine parameters in RF across all rows."""
        self.ld(hbm(gamma.offset, d), rf(4, d))
        self.ld(hbm(beta.offset, d), rf(5, d))
        with self.loop(name, 0, rows) as row:
            base = add(source.offset, mul(row, d))
            target = add(out.offset, mul(row, d))
            if in_shared is None:
                self.ld(hbm(base, d), rf(0, d))
            else:
                self.ld(sh(add(in_shared, mul(row, d)), d), rf(0, d))
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
            self.st(rf(2, d), (sh if out_shared else hbm)(target, d))

    def residual_layernorm_cached(
        self, left: Tensor, right: Tensor, residual: Tensor,
        gamma: Tensor, beta: Tensor, out: Tensor,
        rows: int, d: int, name: str, out_shared=False,
        in_shared=None, res_shared=None,
    ):
        """Write the residual once while carrying its RF row directly into LN."""
        self.ld(hbm(gamma.offset, d), rf(4, d))
        self.ld(hbm(beta.offset, d), rf(5, d))
        with self.loop(name, 0, rows) as row:
            base = add(residual.offset, mul(row, d))
            if in_shared is None:
                self.ld(hbm(add(left.offset, mul(row, d)), d), rf(0, d))
            else:
                self.ld(sh(add(in_shared, mul(row, d)), d), rf(0, d))
            self.ld(hbm(add(right.offset, mul(row, d)), d), rf(1, d))
            self.vec("add", [rf(0, d), rf(1, d)], rf(2, d))
            if res_shared is None:
                self.st(rf(2, d), hbm(base, d))
            else:
                self.st(rf(2, d), sh(add(res_shared, mul(row, d)), d))
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
        left_shared=None,
        out_shared=None,
    ):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 1024) as col:
                n = minimum(1024, sub(width, col))
                if left_shared is None:
                    self.ld(hbm(add(left.offset, mul(row, width), col), n), rf(0, n))
                else:
                    self.ld(sh(add(left_shared, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(right.offset, mul(row, width), col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                if bias is not None:
                    self.ld(hbm(add(bias.offset, col), n), rf(3, n))
                    self.vec("add", [rf(2, n), rf(3, n)], rf(2, n))
                if out_shared is None:
                    self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))
                else:
                    self.st(rf(2, n), sh(add(out_shared, mul(row, width), col), n))

    def add_bias(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str):
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", 0, width, 1024) as col:
                n = minimum(1024, sub(width, col))
                self.ld(hbm(add(source.offset, mul(row, width), col), n), rf(0, n))
                self.ld(hbm(add(bias.offset, col), n), rf(1, n))
                self.vec("add", [rf(0, n), rf(1, n)], rf(2, n))
                self.st(rf(2, n), hbm(add(out.offset, mul(row, width), col), n))

    def bias_gelu(self, source: Tensor, bias: Tensor, out: Tensor, rows: int, width: int, name: str, col_start=0, col_stop=None):
        """Fused `add_bias` + `gelu`: one read of `source`, no `biased` scratch."""
        col_stop = width if col_stop is None else col_stop
        with self.loop(name, 0, rows) as row:
            with self.loop(f"{name}c", col_start, col_stop, 1024) as col:
                n = minimum(1024, sub(col_stop, col))
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
        neuron_start=0, neuron_stop=None, weights_shared=False,
    ):
        """Keep one 32-row MLP block on chip across W1 -> GELU -> W2.

        RF2 is the current 32-neuron activation, RF3 its GELU temporary;
        RF4..11 hold all 256 W2 output channels until the final store.
        RF0/1 stream operands, RF12 holds ones for a bias outer product.
        Reduction chunks remain in ascending order, as in the parent GEMMs.
        """
        if rows not in (64, 128) or (d, f) != (256, 1024):
            raise ValueError("RF streamed MLP requires M1 prompt dimensions")
        neuron_stop = f if neuron_stop is None else neuron_stop
        panel = neuron_stop - neuron_start
        if not 0 <= neuron_start < neuron_stop <= f or panel % 32:
            raise ValueError("MLP neuron shard must be aligned to 32")
        if weights_shared:
            if 2 * d * panel * 4 > 128 * 1024:
                raise ValueError("MLP weight shard exceeds 128 KiB SH")
            self.ld(hbm(w1.offset + neuron_start, d*panel, [d,panel], [f,1]),
                    sh(0,d*panel))
            self.ld(hbm(w2.offset + neuron_start*d, panel*d),sh(d*panel,panel*d))

        def weight1(inner, neuron):
            if weights_shared:
                return sh(add(mul(inner,panel),sub(neuron,neuron_start)),1024,
                          [32,32],[panel,1])
            return hbm(add(w1.offset,mul(inner,f),neuron),1024,[32,32],[f,1])

        def weight2(neuron, col):
            if weights_shared:
                return sh(add(d*panel,mul(sub(neuron,neuron_start),d),col),1024,
                          [32,32],[d,1])
            return hbm(add(w2.offset,mul(neuron,d),col),1024,[32,32],[d,1])
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
            with self.loop(name + "neuron", neuron_start, neuron_stop, tile) as neuron:
                self.vec("add", [imm(0), imm(0)], rf(2, tile*tile))
                if pipeline:
                    self.ld(source_view(add(source.offset, mul(row, d)), tile*tile,
                                        [tile,tile], [d,1]), rf(0,tile*tile))
                    self.ld(weight1(0,neuron), rf(1,tile*tile))
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
                            self.ld(weight1(nxt,neuron),rf(add(1,mul(13,nxt_parity)),tile*tile))
                else:
                    with self.loop(name + "k", 0, d, tile) as inner:
                        self.ld(source_view(add(source.offset, mul(row, d), inner), tile*tile,
                                            [tile, tile], [d, 1]), rf(0, tile*tile))
                        self.ld(weight1(inner,neuron), rf(1, tile*tile))
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
                    self.ld(weight2(neuron,0),rf(1,tile*tile))
                    with self.loop(name + 'o',0,d,tile) as col:
                        parity=mod({'ceildiv':[col,tile]},2)
                        self.emit('MMA.ACC',a=rf(2,tile*tile),b=rf(add(1,mul(13,parity)),tile*tile),
                                  acc=rf(add(4,{'ceildiv':[col,tile]}),tile*tile),
                                  m=tile,n=tile,k=tile,event=None)
                        with self.loop(name + 'prefetcho',add(col,tile),d,d) as nxt:
                            self.ld(weight2(neuron,nxt),rf(add(1,mul(13,sub(1,parity))),tile*tile))
                else:
                    with self.loop(name + "o", 0, d, tile) as col:
                        self.ld(weight2(neuron,col), rf(1, tile*tile))
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

    def mlp_rf_neuron_panel(self, source, w1, b1, w2, out, start, name):
        """Keep the entire 128x64 local GELU panel in RF0..7.

        Reuse each W1/W2 weight tile over all four 32-row blocks. W2 keeps
        one 128x32 output column tile in RF8..11. RF12/13 buffer weights and
        RF14/15 buffer W1 row inputs; GELU borrows dead RF8..10.
        """
        rows,d,f,tile = 128,256,1024,32
        def activation(part,row):
            return rf(add(mul(part,4),{"ceildiv":[row,32]}),1024)
        with self.loop(name+"neuron",0,2) as part:
            neuron=add(start,mul(part,32))
            with self.loop(name+"zero",0,rows,32) as row:
                self.vec("add",[imm(0),imm(0)],activation(part,row))
            self.ld(hbm(add(w1.offset,neuron),1024,[32,32],[f,1]),rf(12,1024))
            with self.loop(name+"k",0,d,32) as inner:
                parity=mod({"ceildiv":[inner,32]},2)
                with self.loop(name+"nextb",add(inner,32),d,d) as nxt:
                    self.ld(hbm(add(w1.offset,mul(nxt,f),neuron),1024,[32,32],[f,1]),
                            rf(add(12,sub(1,parity)),1024))
                self.ld(hbm(add(source.offset,inner),1024,[32,32],[d,1]),rf(14,1024))
                with self.loop(name+"row",0,rows,32) as row:
                    rp=mod({"ceildiv":[row,32]},2)
                    with self.loop(name+"nexta",add(row,32),rows,rows) as nxt:
                        self.ld(hbm(add(source.offset,mul(nxt,d),inner),1024,[32,32],[d,1]),
                                rf(add(14,sub(1,rp)),1024))
                    self.emit("MMA.ACC",a=rf(add(14,rp),1024),b=rf(add(12,parity),1024),
                              acc=activation(part,row),m=32,n=32,k=32,event=None)
            self.vec("add",[imm(1),imm(0)],rf(8,32))
            self.ld(hbm(add(b1.offset,neuron),32),rf(9,32))
            with self.loop(name+"gelu",0,rows,32) as row:
                self.bias_gelu_epilogue(activation(part,row),32,1024,32,(8,9,10))
        with self.loop(name+"col",0,d,32) as col:
            with self.loop(name+"ozero",0,rows,32) as row:
                self.vec("add",[imm(0),imm(0)],rf(add(8,{"ceildiv":[row,32]}),1024))
            self.ld(hbm(add(w2.offset,start*d,col),1024,[32,32],[d,1]),rf(12,1024))
            self.ld(hbm(add(w2.offset,(start+32)*d,col),1024,[32,32],[d,1]),rf(13,1024))
            with self.loop(name+"opart",0,2) as part:
                with self.loop(name+"orow",0,rows,32) as row:
                    self.emit("MMA.ACC",a=activation(part,row),b=rf(add(12,part),1024),
                              acc=rf(add(8,{"ceildiv":[row,32]}),1024),m=32,n=32,k=32,event=None)
            with self.loop(name+"store",0,rows,32) as row:
                self.st(rf(add(8,{"ceildiv":[row,32]}),1024),
                        hbm(add(out.offset,mul(row,d),col),1024,[32,32],[d,1]))

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
            with self.loop(f"{name}c", 0, width, 1024) as col:
                n = minimum(1024, sub(width, col))
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
        panel_pipeline = (
            self.recipe.get("d1_schedule", {}).get("attention_pipeline")
            and rf_scores
        )
        tile = int(self.recipe.get("d1_schedule", {}).get("attention_tile", 16))
        if tile not in (16, 32):
            raise ValueError("D1 attention tile must be 16 or 32")
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
                if panel_pipeline and count >= 2*tile:
                    self.ld(
                        hbm(add(source.offset, mul(head, stride * hd)), hd*tile,
                            [hd, tile], [1, hd]),
                        rf(1, hd*tile),
                    )
                with self.loop(f"{name}key{score_base}", 0, count, tile) as key:
                    n = minimum(tile, sub(count, key))
                    cells = mul(hd, n)
                    if panel_pipeline and count >= 2*tile:
                        parity = mod({"ceildiv": [key, tile]}, 2)
                        key_lane = add(1, parity)
                    else:
                        key_lane = 1
                        self.ld(
                            hbm(
                                add(source.offset, mul(head, stride * hd), mul(key, hd)),
                                cells, [hd, n], [1, hd],
                            ),
                            rf(1, cells),
                        )
                    if rf_scores:
                        acc = rf(4, n, add(score_base, key))
                        self.vec("add", [imm(0), imm(0)], acc)
                        self.emit(
                            "MMA.ACC",
                            a=rf(0, hd),
                            b=rf(key_lane, cells),
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
                            b=rf(key_lane, cells),
                            acc=rf(2, n),
                            m=1,
                            n=n,
                            k=hd,
                            event=None,
                        )
                        self.vec("mul", [rf(2, n), imm(1 / sqrt(hd))], rf(2, n))
                        self.st(rf(2, n), hbm(add(scores.offset, score_base, key), n))
                    if panel_pipeline and count >= 2*tile:
                        with self.loop(f"{name}nextkey{score_base}", add(key, tile),
                                       count, count) as nxt:
                            self.ld(
                                hbm(add(source.offset, mul(head, stride*hd), mul(nxt, hd)),
                                    hd*tile, [hd, tile], [1, hd]),
                                rf(add(1, sub(1, parity)), hd*tile),
                            )
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
                if panel_pipeline and count >= 2*tile:
                    self.ld(
                        hbm(add(source.offset, mul(head, stride * hd)), tile*hd),
                        rf(1, tile*hd),
                    )
                with self.loop(f"{name}value{prob_base}", 0, count, tile) as key:
                    depth = minimum(tile, sub(count, key))
                    if rf_scores:
                        probs_view = rf(4, depth, add(prob_base, key))
                    else:
                        probs_view = rf(0, depth)
                        self.ld(
                            hbm(add(probs.offset, prob_base, key), depth), probs_view
                        )
                    if panel_pipeline and count >= 2*tile:
                        parity = mod({"ceildiv": [key, tile]}, 2)
                        value_lane = add(1, mul(2, parity))
                    else:
                        value_lane = 1
                        self.ld(
                            hbm(add(source.offset, mul(head, stride*hd), mul(key, hd)),
                                mul(depth, hd)),
                            rf(1, mul(depth, hd)),
                        )
                    self.emit(
                        "MMA.ACC",
                        a=probs_view,
                        b=rf(value_lane, mul(depth, hd)),
                        acc=rf(2, hd),
                        m=1,
                        n=hd,
                        k=depth,
                        event=None,
                    )
                    if panel_pipeline and count >= 2*tile:
                        with self.loop(f"{name}nextvalue{prob_base}", add(key, tile),
                                       count, count) as nxt:
                            self.ld(
                                hbm(add(source.offset, mul(head, stride*hd), mul(nxt, hd)),
                                    tile*hd),
                                rf(add(1, mul(2, sub(1, parity))), tile*hd),
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
        if shards not in (2, 4, 8, 16):
            raise ValueError("D1 supports 2, 4, 8 or 16 column/head shards")
        self.d1_weights = recipe["d1_schedule"].get("weights", "hbm")
        if self.d1_weights not in ("hbm", "sh"):
            raise ValueError("D1 weights must be hbm or sh")
        shared = int(recipe["d1_schedule"].get("shared_bytes", 0)) if self.d1_weights == "sh" else 0
        layer_groups = bool(recipe["d1_schedule"].get("layer_groups", 0))
        if (layer_groups
                and recipe["d1_schedule"].get("preload") not in ("first_use", "upfront")):
            raise ValueError("D1 layer groups require first-use weight preload")
        total = shards * (2 if layer_groups else 1)
        super().__init__(layout, recipe, shared_bytes=shared, groups=total,
                         sm_map=[i % shards for i in range(total)])
        self.d1_all_groups = self.groups[:]
        self.d1_layer_groups = ([self.groups[:shards], self.groups[shards:]]
                                if layer_groups else None)
        self.d1_sh_panels = {}
        self.d1_rf_panels = {}
        self.d1_rf_cursor = {}
        self.d1_sh_cursor = {wg: 0 for wg in self.groups}
        self.d1_sh_capacity = shared // 4
        self.d1_kv_sh = {}
        self.d1_kv_done = set()
        self.d1_param_map = {}
        self.d1_rf_param_map = {}
        self.d1_rf_bias = {}
        self.d1_rf_param_layers = set()

    def ld(self, source, target):
        if (self.d1_rf_param_map and source.get("space")=="HBM"
                and target.get("space")=="RF" and "shape" not in source):
            off=source["offset"]
            terms=off["add"] if isinstance(off,dict) and list(off)==["add"] else [off]
            ints=[v for v in terms if isinstance(v,int)]
            if len(ints)==1:
                item=self.d1_rf_param_map.get((self.wg,ints[0]))
                if item is not None:
                    rest=[v for v in terms if not isinstance(v,int)]
                    offset=add(item,*rest) if rest else item
                    return self.vec("add",[rf(14,source["count"],offset),imm(0)],target)
        if (self.d1_param_map and source.get("space") == "HBM"
                and "shape" not in source):
            off = source["offset"]
            terms = off["add"] if isinstance(off, dict) and list(off) == ["add"] else [off]
            ints = [t for t in terms if isinstance(t, int)]
            if len(ints) == 1:
                for (wg, p_off), (p_sh, p_size) in self.d1_param_map.items():
                    if wg == self.wg and p_off <= ints[0] < p_off + p_size:
                        rest = [t for t in terms if not isinstance(t, int)]
                        sh_off = p_sh + (ints[0] - p_off)
                        new_off = sh_off if not rest else add(sh_off, *rest)
                        return super().ld(sh(new_off, source["count"]), target)
        return super().ld(source, target)

    def d1_owner_layernorm(self, x, right, gamma, beta, d=128):
        """Each projection owner rebuilds the LN row locally into RF0[0:d).

        Replaces the single-WG add_rows/layernorm HBM handoff: with ``right``
        the residual x+right is formed first (kept in RF15[896:896+d) for the
        W2 residual epilogue). Op order matches add_rows + layernorm exactly.
        RF15 is scratch (attention's partial lane, dead at GEMM prologues).
        """
        src = rf(15, d, 896)
        self.ld(hbm(gamma.offset, d), rf(15, d, 512))
        self.ld(hbm(beta.offset, d), rf(15, d, 640))
        if right is None:
            self.ld(hbm(x.offset, d), src)
        else:
            self.ld(hbm(x.offset, d), rf(15, d, 0))
            self.ld(hbm(right.offset, d), rf(15, d, 128))
            self.vec("add", [rf(15, d, 0), rf(15, d, 128)], src)
        scalar = rf(15, 1, 768)
        self.reduce("sum", src, scalar)
        self.vec("mul", [scalar, imm(1 / d)], scalar)
        self.vec("sub", [src, scalar], rf(0, d))
        self.vec("mul", [rf(0, d), rf(0, d)], rf(15, d, 256))
        self.reduce("sum", rf(15, d, 256), scalar)
        self.vec("mul", [scalar, imm(1 / d)], scalar)
        self.vec("add", [scalar, imm(1e-5)], scalar)
        self.sfu("rsqrt", scalar, scalar)
        self.vec("mul", [rf(0, d), scalar], rf(0, d))
        self.vec("mul", [rf(0, d), rf(15, d, 512)], rf(0, d))
        self.vec("add", [rf(0, d), rf(15, d, 640)], rf(0, d))

    def d1_owner_layernorm_fold(self, x, right, gamma, d=128):
        """Centred LN fold: RF0 = (x - mu) * gamma; returns the deferred
        variance chain that leaves s = rsqrt(var + eps) in RF15[768].

        LN(x)W = s * [((x - mu) * gamma) W] + beta W. Centring before the
        matmul keeps high-mean rows free of cancellation; s and beta W are
        applied in the projection epilogue.
        """
        src = rf(15, d, 896)
        centred = rf(15, d, 256)
        scalar = rf(15, 1, 768)
        self.ld(hbm(gamma.offset, d), rf(15, d, 512))
        if right is None:
            self.ld(hbm(x.offset, d), src)
        else:
            self.ld(hbm(x.offset, d), rf(15, d, 0))
            self.ld(hbm(right.offset, d), rf(15, d, 128))
            self.vec("add", [rf(15, d, 0), rf(15, d, 128)], src)
        self.reduce("sum", src, scalar)
        self.vec("mul", [scalar, imm(1 / d)], scalar)
        self.vec("sub", [src, scalar], centred)
        self.vec("mul", [centred, rf(15, d, 512)], rf(0, d))

        def variance():
            self.vec("mul", [centred, centred], rf(15, d, 128))
            self.reduce("sum", rf(15, d, 128), scalar)
            self.vec("mul", [scalar, imm(1 / d)], scalar)
            self.vec("add", [scalar, imm(1e-5)], scalar)
            self.sfu("rsqrt", scalar, scalar)
        return variance

    def d1_preload_rf_params(self, weights, layer):
        if layer in self.d1_rf_param_layers:
            return
        cfg=self.recipe["d1_schedule"]
        if not self.d1_layer_groups or cfg.get("compact_rf"):
            raise ValueError("RF parameters require per-layer groups and RF14 free")
        self.d1_rf_param_layers.add(layer)
        for i,wg in enumerate(self.groups):
            self.wg=wg
            self.label=f"l{layer}_rfparams{i}_"
            if i==0:
                for j,key in enumerate(("ln1_g","ln1_b","ln2_g","ln2_b","b2")):
                    t=weights[key]
                    super().ld(hbm(t.offset,128),rf(14,128,j*128))
                    self.d1_rf_param_map[(wg,t.offset)]=j*128
            owners,tile,_=self.d1_plan(f"l{layer}w1",512)
            if owners!=len(self.groups) or 512//owners!=tile:
                raise ValueError("RF bias preload requires one W1 tile per owner")
            start=i*tile
            super().ld(hbm(weights["b1"].offset+start,tile),rf(14,tile,640))
            self.d1_rf_bias[(wg,weights["b1"].offset)]=(start,640)
        self.wg=self.groups[0]
        self.label=""

    def d1_param_preload(self, t):
        """Replicate a small read-only parameter into every WG's SH once."""
        size = prod(t.shape)
        for wg in self.groups:
            base = self.d1_sh_cursor[wg]
            if base + size > self.d1_sh_capacity:
                raise ValueError("D1 params exceed shared quota")
            self.d1_sh_cursor[wg] = base + size
            self.wg = wg
            self.label = "parampre_"
            self.ld(hbm(t.offset, size), sh(base, size))
            self.d1_param_map[(wg, t.offset)] = (base, size)
        self.wg = self.groups[0]
        self.label = ""

    def d1_plan(self, name, n_extent):
        panels = n_extent // self.gemm_n_tile
        owners = max(n for n in range(1, len(self.groups)+1) if panels % n == 0)
        tile, chunk = self.gemm_n_tile, self.k_chunk
        stage = next(key for key in ("qkv", "wo", "w1", "w2") if name.endswith(key))
        plan = self.recipe["d1_schedule"].get("projections", {}).get(stage)
        if plan:
            owners, tile, chunk = int(plan["owners"]), int(plan["n_tile"]), int(plan["k_chunk"])
            if (not 1 <= owners <= len(self.groups) or tile <= 0 or chunk <= 0
                    or n_extent % (owners*tile) or tile*chunk > 1024):
                raise ValueError("Invalid D1 projection partition/RF capacity")
        return owners, tile, chunk

    def d1_rf_resident(self, name):
        m = re.search(r"l(\d+)(qkv|wo|w1|w2)$", name)
        wanted = self.recipe["d1_schedule"].get("rf_resident", [])
        return bool(m) and f"l{m.group(1)}{m.group(2)}" in wanted

    def d1_preload_rf(self, b, name, k_extent, n_extent):
        """Load each owner's panel HBM->RF lanes 5.. once; tiles packed per lane."""
        owners, tile, chunk = self.d1_plan(name, n_extent)
        cells = tile * chunk
        lane_words = 1024
        if lane_words % cells:
            raise ValueError("RF-resident D1 tile must divide an RF lane")
        per_lane = lane_words // cells
        for index, wg in enumerate(self.groups[:owners]):
            col_start = n_extent*index//owners
            col_stop = n_extent*(index+1)//owners
            tiles = (col_stop - col_start) // tile * (k_extent // chunk)
            lanes = -(-tiles // per_lane)
            base = self.d1_rf_cursor.setdefault(wg, 4 if self.recipe["d1_schedule"].get("compact_rf") else 5)
            limit=14 if self.recipe["d1_schedule"].get("rf_params") else 16
            if base + lanes > limit:
                raise ValueError("D1 RF-resident weights exceed free RF lanes")
            self.d1_rf_cursor[wg] = base + lanes
            self.d1_rf_panels[(wg, b.offset, col_start)] = (base, per_lane)
            self.wg = wg
            self.label = f"{name}_prerf{index}_"
            chunks = k_extent // chunk
            staged = bool(self.recipe["d1_schedule"].get("rf_preload_sh",0))
            if staged:
                width = col_stop-col_start
                sh_base = self.d1_sh_cursor[wg]
                size = k_extent*width
                if sh_base+size > self.d1_sh_capacity:
                    raise ValueError("D1 RF preload staging exceeds SH quota")
                self.d1_sh_cursor[wg] += size
                self.ld(hbm(b.offset+col_start,size,[k_extent,width],[n_extent,1]),
                        sh(sh_base,size))
            with self.loop(f"{name}prerf{index}n", col_start, col_stop, tile) as col:
                with self.loop(f"{name}prerf{index}k", 0, k_extent, chunk) as inner:
                    idx = add(mul({"ceildiv": [sub(col, col_start), tile]}, chunks),
                              {"ceildiv": [inner, chunk]})
                    source = (sh(add(sh_base,mul(inner,width),sub(col,col_start)),cells,
                                 [chunk,tile],[width,1]) if staged else
                              hbm(add(b.offset,mul(inner,n_extent),col),cells,
                                  [chunk,tile],[n_extent,1]))
                    self.ld(source,
                            d1_rf_tile(base, per_lane, idx, cells))
        self.wg = self.groups[0]
        self.label = ""

    def d1_preload(self, b, name, k_extent, n_extent):
        """Copy each owner's weight panel HBM->SH once, in (col, k) tile order."""
        if self.d1_rf_resident(name):
            return self.d1_preload_rf(b, name, k_extent, n_extent)
        owners, tile, chunk = self.d1_plan(name, n_extent)
        for index, wg in enumerate(self.groups[:owners]):
            col_start = n_extent*index//owners
            col_stop = n_extent*(index+1)//owners
            base = self.d1_sh_cursor[wg]
            size = k_extent * (col_stop - col_start)
            if base + size > self.d1_sh_capacity:
                raise ValueError("D1 SH-resident weights exceed shared quota")
            self.d1_sh_cursor[wg] = base + size
            self.d1_sh_panels[(wg, b.offset, col_start)] = base
            self.wg = wg
            self.label = f"{name}_pre{index}_"
            chunks = k_extent // chunk
            with self.loop(f"{name}pre{index}n", col_start, col_stop, tile) as col:
                with self.loop(f"{name}pre{index}k", 0, k_extent, chunk) as inner:
                    index_expr = add(mul({"ceildiv": [sub(col, col_start), tile]}, chunks),
                                     {"ceildiv": [inner, chunk]})
                    self.ld(hbm(add(b.offset, mul(inner, n_extent), col), chunk*tile,
                                [chunk, tile], [n_extent, 1]),
                            sh(add(base, mul(index_expr, chunk*tile)), chunk*tile))
        self.wg = self.groups[0]
        self.label = ""

    def gemm(self, a, b, out, m_extent, k_extent, n_extent, name, col_start=0, col_stop=None):
        if col_start or col_stop is not None:
            raise ValueError("Nested D1 partition unsupported")
        self.emit("BARRIER", wgs=self.groups, events=[])
        old_tile, old_chunk = self.gemm_n_tile, self.k_chunk
        # Preserve 64-column panels: use 2 owners for width128, 3 for
        # width384, and 4 for width512 instead of doubling all small MMAs.
        owners, self.gemm_n_tile, self.k_chunk = self.d1_plan(name, n_extent)
        if (self.d1_weights == "sh"
                and self.recipe["d1_schedule"].get("preload") == "first_use"
                and (self.groups[0], b.offset, 0) not in self.d1_sh_panels
                and (self.groups[0], b.offset, 0) not in self.d1_rf_panels):
            # Step-0 first touch after the leading barrier: the panel copy
            # pipelines with this GEMM instead of being drained up front.
            self.d1_preload(b, name, k_extent, n_extent)
        for index, wg in enumerate(self.groups[:owners]):
            self.wg = wg
            self.label = f"{name}_sm{index}_"
            super().gemm(a, b, out, m_extent, k_extent, n_extent, name + f"g{index}",
                         n_extent*index//owners, n_extent*(index+1)//owners)
        self.gemm_n_tile, self.k_chunk = old_tile, old_chunk
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])

    def bias_gelu(self, source, bias, out, rows, width, name, col_start=0, col_stop=None):
        if not self.recipe.get("d1_schedule", {}).get("parallel_gelu"):
            return super().bias_gelu(
                source, bias, out, rows, width, name, col_start, col_stop
            )
        if col_start or col_stop is not None or rows != 1 or width % len(self.groups):
            raise ValueError("D1 parallel GELU requires a full divisible single row")
        # W1's trailing barrier already releases every activation shard. W2's
        # leading barrier joins these GELU shards, so no extra barrier is needed.
        for index, wg in enumerate(self.groups):
            self.wg = wg
            self.label = f"{name}_sm{index}_"
            super().bias_gelu(
                source, bias, out, rows, width, name + f"g{index}",
                width * index // len(self.groups),
                width * (index + 1) // len(self.groups),
            )
        self.wg = self.groups[0]
        self.label = ""

    def attention_decode_split(self, qkv, context, history_k, history_v, new_k, new_v,
                               step, past, d, h, hd, name):
        """Split-K decode attention: P = shards/heads WGs per head.

        Part p scores history keys [p*past/P, (p+1)*past/P) (part 0 also the
        generated keys), keeps local max m_p, sum l_p and unnormalised context
        c_p, and publishes (c_p, m_p, l_p) to HBM scratch.  After a barrier part
        0 merges: w_p = exp(m_p - max m), ctx = sum w_p c_p / sum w_p l_p.
        RF15 holds scalars/partials (free of the D1 GEMM and resident lanes).
        """
        parts = len(self.groups) // h
        if parts * h != len(self.groups) or past % parts or parts > 4:
            raise ValueError("Split decode attention needs 2..4 parts per head")
        generated = step + 1
        span = past // parts
        width = hd + 2
        partials = self.alloc(f"{name}_partials", h * parts * width)
        compact = bool(self.recipe["d1_schedule"].get("compact_rf",0))
        def score_view(count,offset=0):
            return rf(3,count,add(256,offset)) if compact else rf(4,count,offset)
        def meta_view(count,offset=0):
            return rf(3,count,add(512,offset)) if compact else rf(15,count,offset)

        kv_mode = self.recipe["d1_schedule"].get("kv_sh", 0)
        kv_resident = self.d1_weights == "sh" and kv_mode
        kv_comps = {"k", "v"} if kv_mode == 1 or kv_mode is True else {str(kv_mode)}
        if kv_resident and (history_k.offset, "k") not in self.d1_kv_done:
            # Each part WG keeps its K/V history slice in SH once; per-step
            # reads then use local SH->RF instead of pending HBM transfers.
            for head in range(h):
                for part in range(parts):
                    wg = self.groups[head * parts + part]
                    base = self.d1_sh_cursor[wg]
                    need = len(kv_comps) * span * hd
                    if base + need > self.d1_sh_capacity:
                        raise ValueError("D1 KV history exceeds shared quota")
                    self.d1_sh_cursor[wg] = base + need
                    self.wg = wg
                    self.label = f"{name}_kvpre{head}_{part}_"
                    for comp, src in (("k", history_k), ("v", history_v)):
                        if comp not in kv_comps:
                            continue
                        self.d1_kv_sh[(wg, comp, src.offset)] = base
                        self.ld(hbm(src.offset + head * past * hd + part * span * hd,
                                    span * hd),
                                sh(base, span * hd))
                        base += span * hd
            self.d1_kv_done.add((history_k.offset, "k"))
            self.wg = self.groups[0]
            self.label = ""

        def strided(lane, count, offset, stride):
            view = rf(lane, count, offset)
            view.update(shape=[count], strides=[stride])
            return view

        self.emit("BARRIER", wgs=self.groups, events=[])
        for head in range(h):
            for part in range(parts):
                self.wg = self.groups[head * parts + part]
                self.label = f"{name}_sp{head}_{part}_"
                tag = f"{name}h{head}p{part}"
                ranges = [(history_k, history_v, part * span, span, past)]
                if part == 0:
                    for target, component in ((new_k, 1), (new_v, 2)):
                        self.ld(hbm(qkv.offset + component * d + head * hd, hd), rf(0, hd))
                        self.st(rf(0, hd), hbm(
                            target.offset + head * target.shape[2] * hd + step * hd, hd))
                    ranges.append((new_k, new_v, 0, generated, new_k.shape[2]))
                self.ld(hbm(qkv.offset + head * hd, hd), rf(0, hd))
                if self.recipe["d1_schedule"].get("attn_prefetch"):
                    self.d1_attention_part_prefetch(ranges, head, hd, score_view, meta_view,
                                                    partials.offset + (head * parts + part) * width)
                    continue
                base = 0
                for key_src, _, start, count, stride in ranges:
                    k_sh = self.d1_kv_sh.get((self.wg, "k", key_src.offset))
                    with self.loop(f"{tag}k{base}", 0, count, 32) as key:
                        n = minimum(32, sub(count, key))
                        if k_sh is not None:
                            self.ld(sh(add(k_sh, mul(key, hd)), mul(hd, n),
                                       [hd, n], [1, hd]),
                                    rf(1, mul(hd, n)))
                        else:
                            self.ld(hbm(add(key_src.offset + head * stride * hd + start * hd,
                                            mul(key, hd)), mul(hd, n), [hd, n], [1, hd]),
                                    rf(1, mul(hd, n)))
                        acc = score_view(n, add(base,key))
                        self.vec("add", [imm(0), imm(0)], acc)
                        self.emit("MMA.ACC", a=rf(0, hd), b=rf(1, mul(hd, n)), acc=acc,
                                  m=1, n=n, k=hd, event=None)
                        self.vec("mul", [acc, imm(1 / sqrt(hd))], acc)
                    base += count
                svec = score_view(base)
                self.reduce("max", svec, meta_view(1, 0))
                self.vec("sub", [svec, meta_view(1, 0)], svec)
                self.sfu("exp", svec, svec)
                self.reduce("sum", svec, meta_view(1, 1))
                self.vec("add", [imm(0), imm(0)], rf(2, hd))
                base = 0
                for _, value_src, start, count, stride in ranges:
                    v_sh = self.d1_kv_sh.get((self.wg, "v", value_src.offset))
                    with self.loop(f"{tag}v{base}", 0, count, 32) as key:
                        depth = minimum(32, sub(count, key))
                        if v_sh is not None:
                            self.ld(sh(add(v_sh, mul(key, hd)), mul(depth, hd)),
                                    rf(1, mul(depth, hd)))
                        else:
                            self.ld(hbm(add(value_src.offset + head * stride * hd + start * hd,
                                            mul(key, hd)), mul(depth, hd)),
                                    rf(1, mul(depth, hd)))
                        self.emit("MMA.ACC", a=score_view(depth, add(base,key)),
                                  b=rf(1, mul(depth, hd)), acc=rf(2, hd),
                                  m=1, n=hd, k=depth, event=None)
                    base += count
                slot = partials.offset + (head * parts + part) * width
                self.st(rf(2, hd), hbm(slot, hd))
                self.st(meta_view(2, 0), hbm(slot + hd, 2))
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])
        for head in range(h):
            self.wg = self.groups[head * parts]
            self.label = f"{name}_merge{head}_"
            self.ld(hbm(partials.offset + head * parts * width, parts * width),
                    meta_view(parts * width, 16))
            m_view = meta_view(parts,16+hd)
            m_view.update(shape=[parts],strides=[width])
            l_view = meta_view(parts,16+hd+1)
            l_view.update(shape=[parts],strides=[width])
            self.reduce("max", m_view, meta_view(1, 0))
            self.vec("sub", [m_view, meta_view(1, 0)], meta_view(parts, 8))
            self.sfu("exp", meta_view(parts, 8), meta_view(parts, 8))
            self.vec("mul", [meta_view(parts, 8), l_view], meta_view(parts, 12))
            self.reduce("sum", meta_view(parts, 12), meta_view(1, 1))
            self.vec("add", [imm(0), imm(0)], rf(2, hd))
            contexts = meta_view(parts * hd, 16)
            contexts.update(shape=[parts, hd], strides=[width, 1])
            self.emit("MMA.ACC", a=meta_view(parts, 8), b=contexts, acc=rf(2, hd),
                      m=1, n=hd, k=parts, event=None)
            self.vec("div", [rf(2, hd), meta_view(1, 1)], rf(2, hd))
            self.st(rf(2, hd), hbm(context.offset + head * hd, hd))
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])

    def d1_attention_part_prefetch(self, ranges, head, hd, score_view, meta_view, slot):
        """One split-attention part with every K/V tile issued up front.

        Each range fits one 32-key tile, so K and V get disjoint RF homes
        (history K: RF1, V: RF14; generated K/V: RF2/RF3 tails) and the V
        copy overlaps scoring instead of waiting on the K tile's reader.
        Arithmetic and reduction order match the sequential path.
        """
        homes = [(rf(1, 32 * hd), rf(14, 32 * hd)), (rf(2, 8 * hd, 512), rf(3, 8 * hd, 512))]
        if len(ranges) > len(homes) or any(c > (32 if i == 0 else 8)
                                           for i, (_, _, _, c, _) in enumerate(ranges)):
            raise ValueError("attention prefetch needs one tile per range")
        tiles = []
        for (key_src, value_src, start, count, stride), (k_home, v_home) in zip(ranges, homes):
            k_view = dict(k_home, count=hd * count)
            v_view = dict(v_home, count=hd * count)
            for comp, src, view, shape in (("k", key_src, k_view, ([hd, count], [1, hd])),
                                            ("v", value_src, v_view, None)):
                sh_base = self.d1_kv_sh.get((self.wg, comp, src.offset))
                if sh_base is not None:
                    source = (sh(sh_base, hd * count, *shape) if shape
                              else sh(sh_base, hd * count))
                else:
                    offset = src.offset + head * stride * hd + start * hd
                    source = (hbm(offset, hd * count, *shape) if shape
                              else hbm(offset, hd * count))
                self.ld(source, view)
            tiles.append((count, k_view, v_view))
        base = 0
        for count, k_view, _ in tiles:
            acc = score_view(count, base)
            self.vec("add", [imm(0), imm(0)], acc)
            self.emit("MMA.ACC", a=rf(0, hd), b=k_view, acc=acc, m=1, n=count, k=hd, event=None)
            self.vec("mul", [acc, imm(1 / sqrt(hd))], acc)
            base += count
        svec = score_view(base)
        self.reduce("max", svec, meta_view(1, 0))
        self.vec("sub", [svec, meta_view(1, 0)], svec)
        self.sfu("exp", svec, svec)
        self.reduce("sum", svec, meta_view(1, 1))
        self.vec("add", [imm(0), imm(0)], rf(2, hd))
        base = 0
        for count, _, v_view in tiles:
            self.emit("MMA.ACC", a=score_view(count, base), b=v_view, acc=rf(2, hd),
                      m=1, n=hd, k=count, event=None)
            base += count
        self.st(rf(2, hd), hbm(slot, hd))
        self.st(meta_view(2, 0), hbm(slot + hd, 2))

    def attention_decode(self, qkv, context, history_k, history_v, new_k, new_v,
                         step, past, d, h, hd, name):
        if (self.recipe["d1_schedule"].get("attention_split")
                and len(self.groups) > h):
            return self.attention_decode_split(qkv, context, history_k, history_v,
                                               new_k, new_v, step, past, d, h, hd, name)
        self.emit("BARRIER", wgs=self.groups, events=[])
        owners = min(h, len(self.groups))
        for index, wg in enumerate(self.groups[:owners]):
            self.wg = wg
            self.label = f"{name}_sm{index}_"
            super().attention_decode(qkv, context, history_k, history_v, new_k, new_v,
                                     step, past, d, h, hd, name + f"g{index}",
                                     h*index//owners, h*(index+1)//owners)
        self.wg = self.groups[0]
        self.label = ""
        self.emit("BARRIER", wgs=self.groups, events=[])


def emit_merged_step(b, model, prompt_rows, step, output, fuse_gelu):
    """P1 new-position phase with both batch rows fused into m=2 GEMMs.

    All WGs split every projection by output columns, so each streams only
    1/W of the weights (the per-team step phase streamed 1/team each).  WG i
    owns head i's Q/K/V columns and runs that head's attention for both
    members, so QKV -> attention needs no join.  Row-wise LN/residual stages
    put row r on WG r.
    """
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch = 2
    groups = b.groups
    count = len(groups)
    tile = b.gemm_n_tile

    def join():
        b.emit("BARRIER", wgs=groups, events=[])

    def place(i, stage):
        b.wg = groups[i]
        b.label = f"sm_{stage}s{i}_"

    def column_owners(n):
        owners = min(count, n // tile)
        while n % (owners * tile):
            owners -= 1
        return owners

    def gemm_split(a, weight, out, k, n, stage, bias=None):
        owners = column_owners(n)
        for i in range(owners):
            place(i, stage)
            b.gemm(a, weight, out, batch, k, n, f"{name}g{i}{stage}",
                   n * i // owners, n * (i + 1) // owners, epilogue_bias=bias)

    heads_owners = min(h, count)
    if (h % heads_owners) or (hd * h // heads_owners) % tile:
        raise ValueError("merged step needs head-aligned QKV column tiles")
    x = Tensor(step.offset, (batch, d))
    for layer in range(model.layers):
        name = f"sml{layer}"
        prefix = f"layer{layer}/"
        w = {key: b.symbol(prefix + key) for key in (
            "ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b", "w1", "b1", "w2", "b2")}
        t = {key: b.alloc(name + "_" + key, batch, width) for key, width in (
            ("ln1", d), ("qkv", 3 * d), ("ctx", d), ("att", d), ("res", d),
            ("ln2", d), ("ff1", f), ("gelu", f), ("ff2", d), ("out", d))}
        kvs = [b.symbol(prefix + f"new_{kind}") for kind in ("k", "v")]
        for r in range(batch):
            place(r, "ln1")
            b.layernorm_cached_params(
                Tensor(x.offset + r * d, (1, d)), w["ln1_g"], w["ln1_b"],
                Tensor(t["ln1"].offset + r * d, (1, d)), 1, d, f"{name}r{r}ln1")
        join()
        for i in range(heads_owners):
            hs, he = h * i // heads_owners, h * (i + 1) // heads_owners
            place(i, "qkv")
            for cbase, cname in ((0, "q"), (d, "k"), (2 * d, "v")):
                b.gemm(t["ln1"], w["wqkv"], t["qkv"], batch, d, 3 * d,
                       f"{name}g{i}{cname}_qkv", cbase + hs * hd, cbase + he * hd)
            place(i, "a")
            for member in range(batch):
                kv = [Tensor(v.offset + member * h * (prompt_rows + 1) * hd, v.shape)
                      for v in kvs]
                b.attention(
                    Tensor(t["qkv"].offset + member * 3 * d, (1, 3 * d)),
                    Tensor(t["ctx"].offset + member * d, (1, d)),
                    *kv, 1, prompt_rows, d, h, hd, f"{name}g{i}m{member}a", hs, he)
        join()
        gemm_split(t["ctx"], w["wo"], t["att"], d, d, "wo")
        join()
        for r in range(batch):
            place(r, "resln2")
            b.residual_layernorm_cached(
                Tensor(x.offset + r * d, (1, d)), Tensor(t["att"].offset + r * d, (1, d)),
                Tensor(t["res"].offset + r * d, (1, d)), w["ln2_g"], w["ln2_b"],
                Tensor(t["ln2"].offset + r * d, (1, d)), 1, d, f"{name}r{r}resln2")
        join()
        if fuse_gelu:
            gemm_split(t["ln2"], w["w1"], t["gelu"], d, f, "w1", bias=w["b1"])
        else:
            gemm_split(t["ln2"], w["w1"], t["ff1"], d, f, "w1")
            join()
            for r in range(batch):
                place(r, "gelu")
                b.bias_gelu(Tensor(t["ff1"].offset + r * f, (1, f)), w["b1"],
                            Tensor(t["gelu"].offset + r * f, (1, f)), 1, f, f"{name}r{r}gelu")
        join()
        gemm_split(t["gelu"], w["w2"], t["ff2"], f, d, "w2")
        join()
        for r in range(batch):
            place(r, "out")
            b.add_rows(Tensor(t["res"].offset + r * d, (1, d)),
                       Tensor(t["ff2"].offset + r * d, (1, d)),
                       Tensor(t["out"].offset + r * d, (1, d)), 1, d, f"{name}r{r}out", w["b2"])
        join()
        x = t["out"]
    for r in range(batch):
        place(r, "output")
        b.ld(hbm(x.offset + r * d, d), rf(0, d))
        b.st(rf(0, d), hbm(output.offset + r * (prompt_rows + 1) * d + prompt_rows * d, d))
    b.wg = groups[0]
    b.label = ""


def generate_m1_p1_batch_shard(recipe):
    """batch x team hybrid: each member owned by a team_size-WG column/head shard.

    Teams stay independent until the global STEP.COMMIT, so cross-WG history
    unions pair only inside each member's team.  LN/residual/output stages split
    by rows, QKV splits into Q/K/V column slices aligned to each shard's head
    ownership (so attention needs no join after QKV), WO/W1/W2 split by output
    columns, and the fused RF MLP can either stay on the team primary or run as
    ordinary column-sharded GEMMs.
    """
    team_size = int(recipe["p1_schedule"].get("team", 2))
    sm_count = int(recipe["p1_schedule"].get("sm_count", 4))
    batch, prompt_rows, _ = SCENARIOS["P1"]
    if team_size not in (2, 4, 8, 16):
        raise ValueError("batch_shard supports team=2, 4, 8 or 16 per member")
    if batch * team_size > 4 * sm_count:
        raise ValueError("batch_shard exceeds resident workgroup capacity")
    mlp_mode = recipe["p1_schedule"].get("mlp", "fused_primary")
    fuse_gelu = bool(recipe["p1_schedule"].get("fuse_gelu", 0))
    if mlp_mode not in ("fused_primary", "sharded"):
        raise ValueError("batch_shard mlp must be fused_primary or sharded")
    if recipe.get("gemm_kernels"):
        raise ValueError("batch_shard uses the ordinary GEMM primitive")
    shard_recipe = json.loads(json.dumps(recipe))
    # SH is private per SM: cross-WG operands always travel through HBM.
    shard_recipe.setdefault("fusion", {}).update(
        p1_ln1_sh=0, p1_mlp="rf_pipeline_residual"
    )
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    layout = build_layout(model, "P1")
    step_sh_specs = recipe["p1_schedule"].get("step_sh")
    act_sh = set(recipe["p1_schedule"].get("act_sh", []))
    priv_sh = bool(recipe["p1_schedule"].get("priv_sh", 0))
    shared_bytes = (int(recipe["p1_schedule"].get("shared_bytes", 0))
                    if step_sh_specs else 0)
    if act_sh and not step_sh_specs:
        shared_bytes = max(shared_bytes, 128 * 1024)
    if priv_sh:
        # Private row-slab residency: each WG keeps its row shard of `res`/
        # `out` in SH.  They alias one 2048-word buffer — every stage reads
        # row r from SH before writing row r back to the same slot, so a
        # single buffer is safe.  step_sh staging starts above it.
        shared_bytes = max(shared_bytes, 8 * 1024)
    if recipe["p1_schedule"].get("wstage"):
        # Single staged weight slice: largest is a 32,768-word w1/w2 slab.
        shared_bytes = max(shared_bytes,
                           ((2048 if priv_sh else 0) + 32768) * 4)
    PRIV_RES = PRIV_OUT = 0
    PRIV_BASE = 2048 if priv_sh else 0
    # Round-robin placement keeps matching member shards co-resident on one SM.
    b = Builder(
        layout, shard_recipe, groups=batch * team_size,
        sm_map=[index % sm_count for index in range(batch * team_size)],
        shared_bytes=shared_bytes,
    )
    b.step_sh = {}
    b.act_sh = act_sh
    prompt, step, output = (
        b.symbol(key) for key in ("input/prompt", "input/step0", "output/hidden")
    )

    global_join = bool(recipe["p1_schedule"].get("global_join", 0))

    def join(wgs):
        if len(wgs) > 1:
            b.emit("BARRIER", wgs=wgs, events=[])

    def gjoin(wgs):
        # Keep both prompt teams stage-aligned so the second team's weight
        # reads land while the first team's fetches are still cache-resident.
        join(b.groups if global_join else wgs)

    stag = bool(recipe["p1_schedule"].get("team_stagger", 0))

    def stag_anchor(tag, wt):
        # Member-0 WGs close each weight stage with a named marker load; its
        # completion indicates the stage's weight fills are cache-resident.
        if member == 0 and stag:
            for i in range(team_size):
                b.wg = wgs[i]
                b.emit("LD", src=hbm(wt.offset, 64), dst=rf(15, 64),
                       event=f"stg{tag}g{i}")

    def stag_wait(tag):
        # Member-1 WGs hold their weight stream until member 0's fills are
        # resident, so every rep-2 weight read resolves as a cache hit.
        if member == 1 and stag:
            for i in range(team_size):
                b.wg = wgs[i]
                b.emit("WAIT", wg=wgs[i], events=[f"stg{tag}g{i}"])

    def row_part(rows, i):
        if rows < team_size:
            return (0, rows) if i == 0 else (rows, 0)
        start = rows * i // team_size
        return start, rows * (i + 1) // team_size - start

    step_sh_at = recipe["p1_schedule"].get("step_sh_at")
    owners8 = min(h, len(b.groups))

    def emit_step_prefetch(gi, wg):
        # Stage the step-phase weight slices this WG will consume into its own
        # SH.  Ownership map mirrors emit_merged_step (heads_owners for
        # qkv/wo/w2, all WGs for w1).  Emitted pre-commit so the transfers
        # overlap the prompt tail.
        cap = shared_bytes // 4
        base = PRIV_BASE + (32768 if wstage else 0)
        stream = []
        for layer in range(model.layers):
            ws = {key: b.symbol(f"layer{layer}/{key}")
                  for key in ("wqkv", "wo", "w1", "w2")}
            if gi < owners8:
                for cb in (0, d, 2 * d):
                    stream.append((f"l{layer}qkv", ws["wqkv"], d,
                                   cb + gi * hd, hd))
                stream.append((f"l{layer}wo", ws["wo"], d,
                               d * gi // owners8, d // owners8))
            stream.append((f"l{layer}w1", ws["w1"], d,
                           f * gi // len(b.groups),
                           f // len(b.groups)))
            if gi < owners8:
                stream.append((f"l{layer}w2", ws["w2"], f,
                               d * gi // owners8, d // owners8))
        b.wg = wg
        b.label = "steppre_"
        used = base
        for spec in step_sh_specs:
            for tag, wt, k, c0, nc in stream:
                key = (wg, wt.offset, c0)
                if spec not in (tag, tag[2:]) or key in b.step_sh:
                    continue
                if used + k * nc > cap:
                    continue
                b.ld(
                    hbm(wt.offset + c0, k * nc, [k, nc], [wt.shape[1], 1]),
                    sh(used, k * nc),
                )
                b.step_sh[key] = (used, nc)
                used += k * nc
        b.label = ""

    # wstage: prompt-phase weight residency.  Each WG prefetches its
    # next-stage weight slice into one staging region in its private SH
    # while the current stage's MMAs drain; the GEMM B reads then come
    # from SH instead of the HBM queue.  Region is WSTAGE..WSTAGE+32k.
    wstage = recipe["p1_schedule"].get("wstage")
    WSTAGE = PRIV_BASE
    b.p1_sh = {}
    own_wo = min(team_size, d // 32)

    def emit_wstage(wg, wt, k, c0, nc):
        key = (wg, wt.offset, c0)
        if key in b.p1_sh:
            return
        b.wg = wg
        keep = b.label
        b.label = "wstage_"
        b.ld(
            hbm(wt.offset + c0, k * nc, [k, nc], [wt.shape[1], 1]),
            sh(WSTAGE, k * nc),
        )
        b.p1_sh[key] = (WSTAGE, nc)
        b.label = keep

    step_merge = bool(recipe["p1_schedule"].get("step_merge", 0))
    for phase, rows in (("p", prompt_rows), ("s", 1)):
        b.set_gemm_stage("p1_prompt" if phase == "p" else "p1_step")
        past = 0 if phase == "p" else prompt_rows
        if phase == "s" and step_merge:
            emit_merged_step(b, model, prompt_rows, step, output, fuse_gelu)
            continue
        for member in range(batch):
            wgs = b.groups[member * team_size : (member + 1) * team_size]

            def place(i, stage):
                b.wg = wgs[i]
                b.label = f"{phase}b{member}_{stage}s{i}_"

            x = (
                Tensor(prompt.offset + member * prompt_rows * d, (rows, d))
                if phase == "p"
                else Tensor(step.offset + member * d, (rows, d))
            )
            for layer in range(model.layers):
                name = f"{phase}b{member}l{layer}"
                prefix = f"layer{layer}/"
                w = {
                    key: b.symbol(prefix + key)
                    for key in (
                        "ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b",
                        "w1", "b1", "w2", "b2",
                    )
                }
                t = {
                    key: b.alloc(name + "_" + key, rows, width)
                    for key, width in (
                        ("ln1", d), ("qkv", 3 * d), ("ctx", d), ("att", d),
                        ("res", d), ("ln2", d), ("ff1", f), ("gelu", f),
                        ("ff2", d), ("out", d),
                    )
                }
                kv = [b.symbol(prefix + f"new_{kind}") for kind in ("k", "v")]
                kv = [
                    Tensor(v.offset + member * h * (prompt_rows + 1) * hd, v.shape)
                    for v in kv
                ]
                for i in range(team_size):
                    start, count = row_part(rows, i)
                    if not count:
                        continue
                    place(i, "ln1")
                    b.layernorm_cached_params(
                        Tensor(x.offset + start * d, (count, d)),
                        w["ln1_g"], w["ln1_b"],
                        Tensor(t["ln1"].offset + start * d, (count, d)),
                        count, d, f"{name}g{i}ln1",
                        in_shared=(PRIV_OUT if priv_sh and layer else None),
                    )
                    if phase == "p" and wstage and layer == 0:
                        hs, he = h * i // team_size, h * (i + 1) // team_size
                        if he > hs:
                            for cb in (0, d, 2 * d):
                                emit_wstage(wgs[i], w["wqkv"], d,
                                            cb + hs * hd, (he - hs) * hd)
                gjoin(wgs)
                if act_sh and "qkv" in act_sh:
                    for i in range(team_size):
                        b.wg = wgs[i]
                        b.ld(hbm(t["ln1"].offset, rows * d), sh(0, rows * d))
                stag_wait(f"l{layer}qkv")
                for i in range(team_size):
                    hs, he = h * i // team_size, h * (i + 1) // team_size
                    if he == hs:
                        continue
                    place(i, "qkv")
                    for cbase, cname in ((0, "q"), (d, "k"), (2 * d, "v")):
                        b.gemm(
                            t["ln1"], w["wqkv"], t["qkv"], rows, d, 3 * d,
                            f"{name}g{i}{cname}_qkv",
                            cbase + hs * hd, cbase + he * hd,
                        )
                    place(i, "a")
                    b.attention(
                        t["qkv"], t["ctx"], *kv, rows, past, d, h, hd,
                        f"{name}g{i}a", hs, he,
                    )
                    if phase == "p" and wstage and i < own_wo:
                        emit_wstage(wgs[i], w["wo"], d,
                                    d * i // own_wo, d // own_wo)
                stag_anchor(f"l{layer}qkv", w["wqkv"])
                gjoin(wgs)
                stag_wait(f"l{layer}wo")
                own_wo = min(team_size, d // 32)
                for i in range(own_wo):
                    place(i, "wo")
                    b.gemm(
                        t["ctx"], w["wo"], t["att"], rows, d, d,
                        f"{name}g{i}wo", d * i // own_wo, d * (i + 1) // own_wo,
                    )
                gjoin(wgs)
                for i in range(team_size):
                    start, count = row_part(rows, i)
                    if not count:
                        continue
                    place(i, "resln2")
                    b.residual_layernorm_cached(
                        Tensor(x.offset + start * d, (count, d)),
                        Tensor(t["att"].offset + start * d, (count, d)),
                        Tensor(t["res"].offset + start * d, (count, d)),
                        w["ln2_g"], w["ln2_b"],
                        Tensor(t["ln2"].offset + start * d, (count, d)),
                        count, d, f"{name}g{i}resln2",
                        in_shared=(PRIV_OUT if priv_sh and layer else None),
                        res_shared=(PRIV_RES if priv_sh else None),
                    )
                    if phase == "p" and wstage:
                        emit_wstage(wgs[i], w["w1"], d,
                                    f * i // team_size, f // team_size)
                stag_anchor(f"l{layer}wo", w["wo"])
                gjoin(wgs)
                if act_sh and "w1" in act_sh:
                    for i in range(team_size):
                        b.wg = wgs[i]
                        b.ld(hbm(t["ln2"].offset, rows * d), sh(0, rows * d))
                stag_wait(f"l{layer}w1")
                if phase == "p" and mlp_mode == "fused_primary":
                    place(0, "mlp")
                    b.mlp_rf_stream(
                        t["ln2"], w["w1"], w["b1"], w["w2"], t["out"],
                        rows, d, f, name + "mlp",
                        residual=t["res"], output_bias=w["b2"],
                    )
                else:
                    for i in range(team_size):
                        place(i, "w1")
                        if fuse_gelu:
                            b.gemm(
                                t["ln2"], w["w1"], t["gelu"], rows, d, f,
                                f"{name}g{i}w1", f * i // team_size, f * (i + 1) // team_size,
                                epilogue_bias=w["b1"],
                            )
                        else:
                            b.gemm(
                                t["ln2"], w["w1"], t["ff1"], rows, d, f,
                                f"{name}g{i}w1", f * i // team_size, f * (i + 1) // team_size,
                            )
                            place(i, "gelu")
                            b.bias_gelu(
                                t["ff1"], w["b1"], t["gelu"], rows, f,
                                f"{name}g{i}biasgelu",
                                f * i // team_size, f * (i + 1) // team_size,
                            )
                        if phase == "p" and wstage and i < own_wo:
                            emit_wstage(wgs[i], w["w2"], f,
                                        d * i // own_wo, d // own_wo)
                    stag_anchor(f"l{layer}w1", w["w1"])
                    gjoin(wgs)
                    stag_wait(f"l{layer}w2")
                    for i in range(own_wo):
                        place(i, "w2")
                        b.gemm(
                            t["gelu"], w["w2"], t["ff2"], rows, f, d,
                            f"{name}g{i}w2", d * i // own_wo, d * (i + 1) // own_wo,
                        )
                    stag_anchor(f"l{layer}w2", w["w2"])
                    gjoin(wgs)
                    for i in range(team_size):
                        start, count = row_part(rows, i)
                        if not count:
                            continue
                        place(i, "out")
                        b.add_rows(
                            Tensor(t["res"].offset + start * d, (count, d)),
                            Tensor(t["ff2"].offset + start * d, (count, d)),
                            Tensor(t["out"].offset + start * d, (count, d)),
                            count, d, f"{name}g{i}out", w["b2"],
                            left_shared=(PRIV_RES if priv_sh else None),
                            out_shared=(PRIV_OUT if priv_sh else None),
                        )
                        if (phase == "p" and wstage
                                and layer + 1 < model.layers):
                            wn = b.symbol(f"layer{layer + 1}/wqkv")
                            hs, he = (h * i // team_size,
                                      h * (i + 1) // team_size)
                            if he > hs:
                                for cb in (0, d, 2 * d):
                                    emit_wstage(wgs[i], wn, d,
                                                cb + hs * hd, (he - hs) * hd)
                gjoin(wgs)
                x = t["out"]
                # step_sh_at=L emits the prefetch into each WG's tape right
                # after layer L, spreading the transfer energy over a wider
                # window instead of concentrating it pre-commit.
                if phase == "p" and step_sh_specs and step_sh_at == layer:
                    for i, wg in enumerate(wgs):
                        emit_step_prefetch(member * team_size + i, wg)
            for i in range(team_size):
                start, count = row_part(rows, i)
                if not count:
                    continue
                place(i, "output")
                with b.loop(f"{phase}b{member}g{i}output", start, start + count) as row:
                    if priv_sh:
                        b.ld(sh(add(PRIV_OUT, mul(sub(row, start), d)), d), rf(0, d))
                    else:
                        b.ld(hbm(add(x.offset, mul(row, d)), d), rf(0, d))
                    b.st(
                        rf(0, d),
                        hbm(
                            add(
                                output.offset,
                                member * (prompt_rows + 1) * d,
                                past * d,
                                mul(row, d),
                            ),
                            d,
                        ),
                    )
        if phase == "p":
            if step_sh_specs and step_sh_at is None:
                # commit_overlap_prefetch: each persistent WG stages the
                # step-phase weight slices it will consume into its own SH
                # while the prompt tail still executes.  The ownership map
                # mirrors emit_merged_step (heads_owners for qkv/wo/w2,
                # all WGs for w1).
                for gi, wg in enumerate(b.groups):
                    emit_step_prefetch(gi, wg)
            if recipe["p1_schedule"].get("step_warm"):
                # Pre-commit cache warming: stream each WG's own step weight
                # slices through scratch RF lanes in consumption order, so the
                # fills land before the post-commit consumers query them.
                for gi, wg in enumerate(b.groups):
                    stream = []
                    for layer in range(model.layers):
                        ws = {key: b.symbol(f"layer{layer}/{key}")
                              for key in ("wqkv", "wo", "w1", "w2")}
                        if gi < owners8:
                            for cb in (0, d, 2 * d):
                                stream.append((ws["wqkv"], d, cb + gi * hd, hd))
                            stream.append((ws["wo"], d,
                                           d * gi // owners8, d // owners8))
                        stream.append((ws["w1"], d,
                                       f * gi // len(b.groups),
                                       f // len(b.groups)))
                        if gi < owners8:
                            stream.append((ws["w2"], f,
                                           d * gi // owners8, d // owners8))
                    b.wg = wg
                    b.label = "stepwarm_"
                    slot = 0
                    for wt, k, c0, nc in stream:
                        rows_per = max(1, 1024 // nc)
                        for row in range(0, k, rows_per):
                            take = min(rows_per, k - row)
                            b.ld(
                                hbm(wt.offset + c0 + row * wt.shape[1],
                                    take * nc, [take, nc], [wt.shape[1], 1]),
                                rf(slot, take * nc),
                            )
                            slot = (slot + 1) % 16
                b.label = ""
            b.emit("STEP.COMMIT", step=0)
    return b.finish(), layout, b.cursor


def generate_m1_p1_merged_batch(recipe):
    """One column team projects both prompts; attention remains batch-local."""
    recipe = json.loads(json.dumps(recipe))
    recipe.setdefault("fusion", {})["p1_ln1_sh"] = 0
    model = MODELS["M1"]
    d, f, h, hd = model.width, model.ffn, model.heads, model.head_width
    batch, prompt_rows, _ = SCENARIOS["P1"]
    rows = batch * prompt_rows
    count = int(recipe["p1_schedule"].get("sm_count", 16))
    if count != 16:
        raise ValueError("merged batch first implementation requires 16 SMs")
    layout = build_layout(model, "P1")
    resident_mlp = bool(recipe["p1_schedule"].get("neuron_mlp", 0))
    if resident_mlp:
        recipe["fusion"]["p1_mlp"] = "rf_pipeline"
    b = Builder(layout, recipe, groups=count, shared_bytes=128*1024 if resident_mlp else 0)
    prompt, step, output = [b.symbol(k) for k in
                            ("input/prompt", "input/step0", "output/hidden")]
    b.set_gemm_stage("p1_prompt")
    x = Tensor(prompt.offset, (rows, d))

    def join():
        b.emit("BARRIER", wgs=b.groups, events=[])

    def place(i, stage):
        b.wg = b.groups[i]
        b.label = f"mb_{stage}s{i}_"

    def projection(a, weight, out, k, n, stage, bias=None):
        owners = min(count, n // b.gemm_n_tile)
        while n % (owners * b.gemm_n_tile):
            owners -= 1
        for i in range(owners):
            place(i, stage)
            b.gemm(a, weight, out, rows, k, n, f"{name}g{i}{stage}",
                   n*i//owners, n*(i+1)//owners, epilogue_bias=bias)
        join()

    for layer in range(model.layers):
        name = f"mbl{layer}"
        prefix = f"layer{layer}/"
        w = {key: b.symbol(prefix + key) for key in (
            "ln1_g", "ln1_b", "wqkv", "wo", "ln2_g", "ln2_b",
            "w1", "b1", "w2", "b2")}
        t = {key: b.alloc(name + "_" + key, rows, width) for key, width in (
            ("ln1", d), ("qkv", 3*d), ("ctx", d), ("att", d), ("res", d),
            ("ln2", d), ("gelu", f), ("ff2", d), ("out", d))}
        for i in range(count):
            place(i, "ln1")
            start, nr = rows*i//count, rows//count
            b.layernorm_cached_params(Tensor(x.offset+start*d, (nr,d)),
                w["ln1_g"], w["ln1_b"], Tensor(t["ln1"].offset+start*d, (nr,d)),
                nr, d, f"{name}g{i}ln1")
        join()
        projection(t["ln1"], w["wqkv"], t["qkv"], d, 3*d, "qkv")
        for member in range(batch):
            kv = [b.symbol(prefix+f"new_{kind}") for kind in ("k", "v")]
            kv = [Tensor(v.offset+member*h*(prompt_rows+1)*hd, v.shape) for v in kv]
            for head in range(h):
                i = member*h+head
                place(i, "a")
                b.attention(Tensor(t["qkv"].offset+member*prompt_rows*3*d, (prompt_rows,3*d)),
                    Tensor(t["ctx"].offset+member*prompt_rows*d, (prompt_rows,d)),
                    *kv, prompt_rows, 0, d, h, hd, f"{name}g{i}a", head, head+1)
        join()
        projection(t["ctx"], w["wo"], t["att"], d, d, "wo")
        for i in range(count):
            place(i, "resln2")
            start, nr = rows*i//count, rows//count
            b.residual_layernorm_cached(Tensor(x.offset+start*d, (nr,d)),
                Tensor(t["att"].offset+start*d, (nr,d)),
                Tensor(t["res"].offset+start*d, (nr,d)), w["ln2_g"], w["ln2_b"],
                Tensor(t["ln2"].offset+start*d, (nr,d)), nr,d,f"{name}g{i}resln2")
        join()
        if resident_mlp:
            partials = b.alloc(name+"_partials",count,rows,d)
            for i in range(count):
                place(i,"mlp")
                partial = Tensor(partials.offset+i*rows*d,(rows,d))
                if recipe["p1_schedule"]["neuron_mlp"] == 2:
                    b.mlp_rf_neuron_panel(t["ln2"],w["w1"],w["b1"],w["w2"],
                                          partial,f*i//count,f"{name}g{i}mlp")
                else:
                    b.mlp_rf_stream(t["ln2"],w["w1"],w["b1"],w["w2"],
                        partial,rows,d,f,f"{name}g{i}mlp",
                        neuron_start=f*i//count,neuron_stop=f*(i+1)//count,weights_shared=True)
            join()
            for i in range(count):
                place(i,"mlp_merge")
                start,nr=rows*i//count,rows//count
                with b.loop(f"{name}g{i}merge",start*d,(start+nr)*d,1024) as off:
                    b.vec("add",[imm(0),imm(0)],rf(0,1024))
                    with b.loop(f"{name}g{i}part",0,count) as part:
                        b.ld(hbm(add(partials.offset,mul(part,rows*d),off),1024),rf(1,1024))
                        b.vec("add",[rf(0,1024),rf(1,1024)],rf(0,1024))
                    b.st(rf(0,1024),hbm(add(t["ff2"].offset,off),1024))
            join()
        else:
            projection(t["ln2"], w["w1"], t["gelu"], d, f, "w1", w["b1"])
            projection(t["gelu"], w["w2"], t["ff2"], f, d, "w2")
        for i in range(count):
            place(i, "out")
            start, nr = rows*i//count, rows//count
            b.add_rows(Tensor(t["res"].offset+start*d, (nr,d)),
                Tensor(t["ff2"].offset+start*d, (nr,d)),
                Tensor(t["out"].offset+start*d, (nr,d)), nr,d,f"{name}g{i}out",w["b2"])
        join()
        x = t["out"]
    for i in range(count):
        place(i, "output")
        start, nr = rows*i//count, rows//count
        with b.loop(f"mbg{i}output", start, start+nr) as row:
            # Every owner's row range lies entirely inside one batch member.
            member = start // prompt_rows
            b.ld(hbm(add(x.offset,mul(row,d)),d),rf(0,d))
            b.st(rf(0,d),hbm(add(output.offset+member*d,mul(row,d)),d))
    b.emit("STEP.COMMIT",step=0)
    b.set_gemm_stage("p1_step")
    emit_merged_step(b,model,prompt_rows,step,output,True)
    return b.finish(),layout,b.cursor


def generate_m1_p1(recipe: dict | None = None) -> tuple[str, Layout, int]:
    recipe = recipe or load_recipe()
    schedule_kind = recipe.get("p1_schedule", {}).get("kind")
    if schedule_kind == "merged_batch":
        return generate_m1_p1_merged_batch(recipe)
    if schedule_kind == "column_head":
        return generate_m1_p1_multism(recipe)
    if schedule_kind == "batch_shard":
        return generate_m1_p1_batch_shard(recipe)
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
    if schedule_kind == "incremental_gemm":
        b = IncrementalP1Builder(layout, recipe, shared_bytes=shared_bytes)
    elif schedule_kind == "batch_parallel":
        b = Builder(layout, recipe, shared_bytes=shared_bytes, groups=batch)
    elif schedule_kind is None:
        b = Builder(layout, recipe, shared_bytes=shared_bytes)
    else:
        raise ValueError(f"unsupported P1 schedule: {schedule_kind}")
    prompt = b.symbol("input/prompt")
    step = b.symbol("input/step0")
    output = b.symbol("output/hidden")
    # All prompts finish before STEP.COMMIT releases either batch member's first new token.
    for phase, rows in (("p", prompt_rows), ("s", 1)):
        b.set_gemm_stage("p1_prompt" if phase == "p" else "p1_step")
        for member in range(batch):
            if schedule_kind == "batch_parallel":
                b.wg = b.groups[member]
                b.label = f"{phase}b{member}_"
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
                if phase == "p" and recipe.get("fusion", {}).get("p1_norm_residual"):
                    ln1_target = (
                        Tensor(0, (rows, d))
                        if recipe.get("fusion", {}).get("p1_ln1_sh") else ln1
                    )
                    b.layernorm_cached_params(
                        x, weights["ln1_g"], weights["ln1_b"], ln1_target,
                        rows, d, name + "ln1",
                        out_shared=bool(recipe.get("fusion", {}).get("p1_ln1_sh")),
                    )
                else:
                    b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, rows, d, name + "ln1")
                qkv_source = (
                    Tensor(0, (rows, d))
                    if phase == "p" and recipe.get("fusion", {}).get("p1_ln1_sh")
                    else ln1
                )
                b.gemm(qkv_source, weights["wqkv"], qkv, rows, d, 3 * d, name + "qkv")
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
                if phase == "p" and recipe.get("fusion", {}).get("p1_norm_residual"):
                    ln2_target = (
                        Tensor(0, (rows, d))
                        if mlp_mode == "rf_pipeline_residual_sh_input"
                        else ln2
                    )
                    b.residual_layernorm_cached(
                        x, att, res, weights["ln2_g"], weights["ln2_b"], ln2_target,
                        rows, d, name + "resln2",
                        out_shared=mlp_mode == "rf_pipeline_residual_sh_input",
                    )
                else:
                    b.add_rows(x, att, res, rows, d, name + "res")
                    b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, rows, d, name + "ln2")
                if phase == "p" and recipe.get("fusion", {}).get("p1_mlp") in (
                    "rf_stream", "rf_pipeline", "rf_pipeline_residual",
                    "rf_pipeline_residual_sh_input", "sh_resident"
                ):
                    if recipe["fusion"]["p1_mlp"] == "sh_resident":
                        b.mlp_sh_resident(ln2, weights["w1"], weights["b1"], weights["w2"],
                                          ff2, rows, d, f, name + "mlp")
                    elif recipe["fusion"]["p1_mlp"] in (
                        "rf_pipeline_residual", "rf_pipeline_residual_sh_input"
                    ):
                        source = (
                            Tensor(0, (rows, d))
                            if mlp_mode == "rf_pipeline_residual_sh_input"
                            else ln2
                        )
                        b.mlp_rf_stream(
                            source, weights["w1"], weights["b1"], weights["w2"],
                            y, rows, d, f, name + "mlp",
                            residual=res, output_bias=weights["b2"],
                            source_shared=mlp_mode == "rf_pipeline_residual_sh_input",
                        )
                    else:
                        b.mlp_rf_stream(ln2, weights["w1"], weights["b1"], weights["w2"],
                                        ff2, rows, d, f, name + "mlp")
                else:
                    b.gemm(ln2, weights["w1"], ff1, rows, d, f, name + "w1")
                    if b.fuse_bias_gelu:
                        b.bias_gelu(ff1, weights["b1"], activated, rows, f, name + "biasgelu")
                    else:
                        b.add_bias(ff1, weights["b1"], biased, rows, f, name + "bias1")
                        b.gelu(biased, activated, rows, f, name + "gelu")
                    b.gemm(activated, weights["w2"], ff2, rows, f, d, name + "w2")
                if not (phase == "p" and recipe.get("fusion", {}).get("p1_mlp")
                        in ("rf_pipeline_residual", "rf_pipeline_residual_sh_input")):
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
    if (getattr(b, "d1_weights", "hbm") == "sh"
            and recipe["d1_schedule"].get("preload", "upfront") == "upfront"):
        for layer in range(model.layers):
            if getattr(b, "d1_layer_groups", None):
                # Preload into the owning layer-team's tapes so both layers'
                # weight streams overlap during the cold start.
                b.groups = b.d1_layer_groups[layer]
            prefix = f"layer{layer}/"
            for key, k_extent, n_extent in (
                ("wqkv", d, 3 * d), ("wo", d, d), ("w1", d, f), ("w2", f, d),
            ):
                stage = key[1:] if key == "wqkv" else key
                b.d1_preload(b.symbol(prefix + key), f"pre{layer}{stage}", k_extent, n_extent)
        if getattr(b, "d1_layer_groups", None):
            b.groups = b.d1_all_groups
    if (getattr(b, "d1_weights", "hbm") == "sh"
            and recipe["d1_schedule"].get("params_sh", 0)):
        for layer in range(model.layers):
            prefix = f"layer{layer}/"
            for key in ("ln1_g", "ln1_b", "ln2_g", "ln2_b", "b1", "b2"):
                b.d1_param_preload(b.symbol(prefix + key))
    for step in range(steps):
        x = b.symbol(f"input/step{step}")
        for layer in range(model.layers):
            if getattr(b, "d1_layer_groups", None):
                # Each model layer owns a separate persistent RF slot on every
                # SM. Explicit handoff orders layer outputs across both teams.
                b.emit("BARRIER", wgs=b.d1_all_groups, events=[])
                b.groups = b.d1_layer_groups[layer]
                b.wg = b.groups[0]
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
            if recipe.get("d1_schedule",{}).get("rf_params"):
                b.d1_preload_rf_params(weights,layer)
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
            fused_ln = bool(recipe.get("d1_schedule", {}).get("fused_ln", 0))
            last = layer == model.layers - 1
            fold = recipe.get("d1_schedule", {}).get("fused_ln") == "fold"
            if fold:
                # beta W (RF0[768:) for QKV, RF0[832:) for W1) is accumulated
                # in assembly during step 0 and reused by every later step.
                b.d1_ln_prologue = ("fold", x, None, weights["ln1_g"], 0,
                                    step == 0, weights["ln1_b"], None)
            elif fused_ln:
                # Token-local handoff: every projection owner rebuilds LN1 in
                # RF from x instead of a single WG materialising ln1 in HBM.
                b.d1_ln_prologue = ("ln", x, None, weights["ln1_g"], weights["ln1_b"])
            else:
                b.layernorm(x, weights["ln1_g"], weights["ln1_b"], ln1, 1, d, name + "ln1")
            b.gemm(ln1, weights["wqkv"], qkv, 1, d, 3 * d, name + "qkv")
            b.d1_ln_prologue = None
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
            if fold:
                fold_b1 = bool(recipe["d1_schedule"].get("fold_bias", 0))
                b.d1_ln_prologue = ("fold", x, att, weights["ln2_g"], 64,
                                    step == 0, weights["ln2_b"],
                                    weights["b1"] if fold_b1 else None)
            elif fused_ln:
                b.d1_ln_prologue = ("ln", x, att, weights["ln2_g"], weights["ln2_b"])
            else:
                b.add_rows(x, att, res, 1, d, name + "res")
                b.layernorm(res, weights["ln2_g"], weights["ln2_b"], ln2, 1, d, name + "ln2")
            fuse_d1_epilogue = bool(recipe.get("d1_schedule", {}).get("fuse_gelu", 0))
            if fuse_d1_epilogue:
                if not isinstance(b, ParallelD1Builder) or recipe["d1_schedule"].get("operand_pipeline") != "a_resident":
                    raise ValueError("D1 GELU epilogue requires parallel a_resident GEMM")
                folded = fold and recipe["d1_schedule"].get("fold_bias", 0)
                b.d1_gelu_epilogue = (None if folded else weights["b1"], activated)
            b.gemm(ln2, weights["w1"], ff1, 1, d, f, name + "w1")
            b.d1_ln_prologue = None
            if fuse_d1_epilogue:
                b.d1_gelu_epilogue = None
            elif b.fuse_bias_gelu:
                b.bias_gelu(ff1, weights["b1"], activated, 1, f, name + "biasgelu")
            else:
                b.add_bias(ff1, weights["b1"], biased, 1, f, name + "bias1")
                b.gelu(biased, activated, 1, f, name + "gelu")
            if fused_ln:
                # W2 owners finish the residual in RF; the last layer writes
                # its slice straight into the published output row.
                target = Tensor(output.offset + step * d, (1, d)) if last else y
                b.d1_residual_epilogue = (weights["b2"], target)
                b.gemm(activated, weights["w2"], ff2, 1, f, d, name + "w2")
                b.d1_residual_epilogue = None
            else:
                b.gemm(activated, weights["w2"], ff2, 1, f, d, name + "w2")
                b.add_rows(res, ff2, y, 1, d, name + "out", weights["b2"])
            x = y
        if not fused_ln:
            b.ld(hbm(x.offset, d), rf(0, d))
            b.st(rf(0, d), hbm(output.offset + step * d, d))
        b.emit("STEP.COMMIT", step=step)
    if getattr(b, "d1_layer_groups", None):
        b.groups = b.d1_all_groups
    return b.finish(), layout, b.cursor


if __name__ == "__main__":
    from pathlib import Path

    output = Path("programs")
    output.mkdir(exist_ok=True)
    for filename, generate in (("M1_P1.asm", generate_m1_p1), ("M2_D1.asm", generate_m1_d1)):
        program, _, _ = generate()
        (output / filename).write_text(program, encoding="utf-8")
        print(f"Generated {output / filename}: {len(program.splitlines())} lines")
