"""Static P1 traffic census (no timing): HBM load/store cache lines per stage
and operand class, MMA count and RF bytes. Seconds instead of a 30-min
estimate; used to budget dataflow/schedule hypotheses.
Usage: python p1_traffic.py [PROGRAM] [--hw JSON]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from codesign.challenge.abi import build_layout
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.service import mma_service
from codesign.challenge.workload import MODELS


def line_count(o):
    off = o["offset"]
    shape = o.get("shape", [o["count"]])
    st = o.get("strides", [1])
    if len(shape) == 1:
        if st[0] == 1:
            return (off + shape[0] - 1) * 4 // 64 - off * 4 // 64 + 1
        pos = (off + i * st[0] for i in range(shape[0]))
    else:
        if st[1] == 1:
            return sum((off + r * st[0] + shape[1] - 1) * 4 // 64 - (off + r * st[0]) * 4 // 64 + 1
                       for r in range(shape[0]))
        pos = (off + r * st[0] + c * st[1] for r in range(shape[0]) for c in range(shape[1]))
    return len({p * 4 // 64 for p in pos})


def census(text, hw):
    layout = build_layout(MODELS["M1"], "P1")
    regions = sorted((s.address // 4, s.address // 4 + s.nbytes // 4, n)
                     for n, s in layout.symbols.items())

    def cls(off):
        for lo, hi, n in regions:
            if lo <= off < hi:
                return "act" if n == "scratch" else ("w:" + n.split("/")[-1])
        return "act"

    ld = collections.Counter()
    st = collections.Counter()
    mma = collections.Counter()
    rf_read = collections.Counter()
    committed = False
    for ins in iter_parse(text):
        if ins.op == "STEP.COMMIT":
            committed = True
            continue
        a = ins.args
        if not isinstance(a, dict) or "event" not in a:
            continue
        m = re.match(r"(?:[ps]b\d_)?([a-z0-9]+?)s\d+_", a["event"])
        stage = ("step:" if committed else "") + (m.group(1) if m else a["event"].split("_")[0][:10])
        if ins.op == "LD" and a["src"]["space"] == "HBM":
            ld[(stage, cls(a["src"]["offset"]))] += line_count(a["src"])
        elif ins.op == "ST" and a["dst"]["space"] == "HBM":
            st[stage] += line_count(a["dst"])
        elif ins.op == "MMA.ACC":
            s = mma_service(hw, a["m"], a["n"], a["k"])
            mma[stage] += 1
            rf_read[stage] += s.rf_read_bytes
    return ld, st, mma, rf_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?", default=str(ROOT / "programs/M1_P1.asm"))
    ap.add_argument("--hw", default=None)
    args = ap.parse_args()
    hw_dict = json.loads((ROOT / "hardware.json").read_text())
    hw_dict.update(json.loads(args.hw) if args.hw else {})
    ld, st, mma, rf_read = census(Path(args.program).read_text(), Hardware.from_dict(hw_dict))
    total = sum(ld.values())
    print(f"HBM LD lines {total}  ST lines {sum(st.values())}  "
          f"=> cache-bank slot floor (queries only) {total / 4:.0f} cycles")
    by_stage = collections.defaultdict(dict)
    for (stage, c), v in ld.items():
        by_stage[stage][c] = v
    for stage, d in sorted(by_stage.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {stage:14s} LD {sum(d.values()):7d}  " +
              " ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) +
              f"  | MMA {mma.get(stage, 0)} rfR {rf_read.get(stage, 0) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
