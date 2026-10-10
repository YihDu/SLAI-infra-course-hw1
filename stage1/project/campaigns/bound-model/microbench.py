"""Microbenchmarks of the frozen timing model (dev analysis, not a score).

Runs tiny hand-built programs through codesign.challenge.pipeline.estimate_pipeline
to measure the constants that decide whether a 10x redesign is reachable:
cross-SM handoff latency, MMA chain vs interleaved throughput, LD bandwidth per
source space, and GEMV cost.
"""
import json
import sys
from pathlib import Path

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline

ROOT = Path(__file__).resolve().parents[3]
BASE = 1 << 20  # scratch HBM element offset, far from nothing (no ABI here)


def hw(**kw):
    data = json.loads((ROOT / "hardware.json").read_text())
    data.update(kw)
    for key in [k for k, v in data.items() if v is None]:
        data.pop(key)
    return Hardware.from_dict(data)


class P:
    def __init__(self):
        self.lines, self.n = [], 0

    def op(self, code, **args):
        if code not in ("WG.BEGIN", "WG.END", "BARRIER", "WAIT", "STEP.COMMIT"):
            self.n += 1
            args.setdefault("event", f"e{self.n}")
        self.lines.append(f"{code} {json.dumps(args, separators=(',', ':'))}")
        return args.get("event")

    def text(self):
        return "\n".join(self.lines) + "\n"


def rf(offset, count, lane=0, wg="g"):
    return {"space": "RF", "offset": offset, "count": count, "wg": wg, "lane": lane}


def sh(offset, count, wg="g"):
    return {"space": "SH", "offset": offset, "count": count, "wg": wg, "lane": 0}


def hbm(offset, count):
    return {"space": "HBM", "offset": offset, "count": count, "wg": None, "lane": 0}


def run(p, hardware, words=BASE * 4):
    trace = {}
    res = estimate_pipeline(hardware, iter_parse(p.text()), words, None, trace)
    return res.cycles, trace


def span(trace, event):
    t = trace[event]
    return t["finish"] - t["issue"]


def bench_handoff(hardware, count):
    """g0 writes `count` floats to HBM, barrier, g1 (other SM) loads them."""
    p = P()
    p.op("WG.BEGIN", wg="g0", sm=0, shared_bytes=0)
    p.op("WG.BEGIN", wg="g1", sm=1, shared_bytes=0)
    p.op("VEC", kind="add", src=[{"imm": 1.0}, {"imm": 0.0}], dst=rf(0, count, wg="g0"))
    st = p.op("ST", src=rf(0, count, wg="g0"), dst=hbm(BASE, count))
    p.op("BARRIER", wgs=["g0", "g1"], events=[st])
    ld = p.op("LD", src=hbm(BASE, count), dst=rf(0, count, wg="g1"))
    p.op("WG.END", wg="g0")
    p.op("WG.END", wg="g1")
    cycles, trace = run(p, hardware)
    return {"count": count, "st": span(trace, st), "ld": span(trace, ld),
            "st_finish_to_ld_finish": trace[ld]["finish"] - trace[st]["finish"],
            "total": cycles}


def bench_mma(hardware, chains, per_chain, m=32, n=32, k=32):
    """`chains` independent accumulators, `per_chain` dependent MMAs each, interleaved."""
    p = P()
    p.op("WG.BEGIN", wg="g", sm=0, shared_bytes=0)
    a, b = rf(0, m * k, lane=0), rf(0, k * n, lane=1)
    for c in range(chains):
        p.op("VEC", kind="add", src=[{"imm": 0.0}, {"imm": 0.0}], dst=rf(0, m * n, lane=2 + c))
    p.op("VEC", kind="add", src=[{"imm": 0.0}, {"imm": 0.0}], dst=a)
    p.op("VEC", kind="add", src=[{"imm": 0.0}, {"imm": 0.0}], dst=b)
    first = None
    for _ in range(per_chain):
        for c in range(chains):
            e = p.op("MMA.ACC", a=a, b=b, acc=rf(0, m * n, lane=2 + c), m=m, n=n, k=k)
            first = first or e
    last = e
    p.op("WG.END", wg="g")
    cycles, trace = run(p, hardware)
    total = trace[last]["finish"] - trace[first]["issue"]
    macs = chains * per_chain * m * n * k
    return {"chains": chains, "mma": chains * per_chain, "cycles": total,
            "cycles_per_mma": total / (chains * per_chain), "mac_per_cycle": macs / total}


def bench_ld(hardware, space, count, repeats=8, lanes=2):
    """Back-to-back independent loads into RF from HBM or SH."""
    p = P()
    shared = count * 4 if space == "SH" else 0
    p.op("WG.BEGIN", wg="g", sm=0, shared_bytes=shared)
    if space == "SH":
        p.op("VEC", kind="add", src=[{"imm": 1.0}, {"imm": 0.0}], dst=rf(0, count, lane=3))
        seed = p.op("ST", src=rf(0, count, lane=3), dst=sh(0, count))
        p.op("WAIT", wg="g", events=[seed])
    evs = []
    for r in range(repeats):
        src = hbm(BASE + r * count, count) if space == "HBM" else sh(0, count)
        evs.append(p.op("LD", src=src, dst=rf(0, count, lane=r % lanes)))
    p.op("WG.END", wg="g")
    cycles, trace = run(p, hardware)
    total = trace[evs[-1]]["finish"] - trace[evs[0]]["issue"]
    return {"space": space, "bytes": 4 * count * repeats, "cycles": total,
            "bytes_per_cycle": 4 * count * repeats / total,
            "single_ld_latency": span(trace, evs[0])}


def main():
    configs = {
        "current(2tc kp4)": hw(),
        "1tc kp1": hw(tc_count=1, tc_k_parallel=1),
        "1tc kp2": hw(tc_count=1, tc_k_parallel=2),
    }
    out = {}
    h = configs["current(2tc kp4)"]
    out["handoff"] = [bench_handoff(h, c) for c in (1, 32, 256, 1024)]
    out["mma"] = {
        name: [bench_mma(cfg, c, 8) for c in (1, 2, 4)] for name, cfg in configs.items()
    }
    out["mma_gemv"] = {
        name: [bench_mma(cfg, c, 8, m=1, n=32, k=32) for c in (1, 4)]
        for name, cfg in configs.items()
    }
    out["ld"] = [bench_ld(h, s, 1024) for s in ("HBM", "SH")]
    out["ld_nocache"] = [bench_ld(hw(cache_mib=0), "HBM", 1024)]
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
