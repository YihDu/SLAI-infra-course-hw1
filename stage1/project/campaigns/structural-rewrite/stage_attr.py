"""Per-stage wall-clock attribution for a generated program via estimate_pipeline."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline

hw = Hardware.from_dict(json.loads((ROOT / "hardware.json").read_text()))
text = (ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / "programs/M1_P1.asm").read_text()
trace = {}
res = estimate_pipeline(hw, iter_parse(text), 1 << 26, None, trace)
print("total cycles:", res.cycles)

stage_of = {}
pat = re.compile(r"^(?:p|s)b\d_([a-z0-9]+?)s\d+_e|^sm_([a-z0-9]+?)s\d+_e")
# also catch fused_primary/other labels
for name, t in trace.items():
    m = pat.match(name)
    stage = (m.group(1) or m.group(2)) if m else "other"
    stage_of.setdefault(stage, []).append((t["issue"], t["finish"]))

print(f"{'stage':12s} {'span':>8s} {'events':>7s}  first-issue  last-finish")
tot = 0
for stage, ts in sorted(stage_of.items(), key=lambda kv: min(i for i, _ in kv[1])):
    span = max(f for _, f in ts) - min(i for i, _ in ts)
    tot += span
    print(f"{stage:12s} {span:8d} {len(ts):7d}  {min(i for i,_ in ts):11d} {max(f for _,f in ts):11d}")
print("sum of spans:", tot)
