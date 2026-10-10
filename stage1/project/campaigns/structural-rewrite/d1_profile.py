"""Fast D1 timing profile (dev only; no race validation, dense hazard arrays).

Prints total cycles, per-(step,layer) wall span, and per-stage span/sum of
event durations grouped by the stage token in event names.
Usage: python d1_profile.py [programs/M2_D1.asm] [--hw '{json overrides}']
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import codesign.challenge.hbm_race as _hbm_race
import codesign.challenge.pipeline as _pl
from codesign.challenge.abi import build_layout
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.workload import MODELS

_hbm_race.validate_hbm_races = lambda *a, **k: None
_pl._MemoryTimes.sized = classmethod(
    lambda cls, words, sparse=False: _pl._MemoryTimes(
        np.zeros(words, np.int64), np.zeros(words, np.int64)))

STAGES = ("ln1", "qkv", "wo", "res", "ln2", "w1", "gelu", "w2", "out", "a", "pre")


def profile(text, hw):
    layout = build_layout(MODELS["M2"], "D1")
    words = required_hbm_words(text, layout)
    trace = {}
    res = estimate_pipeline(hw, iter_parse(text), words, None, trace)
    spans = defaultdict(lambda: [1 << 62, 0, 0])
    for name, ev in trace.items():
        m = re.match(r"d(\d)l(\d)([a-z]+\d?)_", name)
        key = (int(m.group(1)), int(m.group(2)), m.group(3)) if m else ("other",)
        s = spans[key]
        s[0] = min(s[0], ev["issue"])
        s[1] = max(s[1], ev["finish"])
        s[2] += 1
    return res, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?", default=str(ROOT / "programs/M2_D1.asm"))
    ap.add_argument("--hw", default=None)
    args = ap.parse_args()
    hw_dict = json.loads((ROOT / "hardware.json").read_text())
    hw_dict.update(json.loads(args.hw) if args.hw else {})
    res, spans = profile(Path(args.program).read_text(), Hardware.from_dict(hw_dict))
    print("cycles", res.cycles, "hbm_r", res.hbm_read_bytes, "peakW",
          round(res.peak_window_power_w, 2))
    by_sl = defaultdict(lambda: [1 << 62, 0])
    stage_tot = defaultdict(int)
    for key, (a, b, n) in spans.items():
        if len(key) == 3:
            sl = by_sl[key[:2]]
            sl[0], sl[1] = min(sl[0], a), max(sl[1], b)
            stage_tot[key[2]] += b - a
    for sl in sorted(by_sl):
        a, b = by_sl[sl]
        stg = sorted(((k[2], spans[k][1] - spans[k][0]) for k in spans
                      if len(k) == 3 and k[:2] == sl), key=lambda t: spans[(*sl, t[0])][0])
        print(f"step{sl[0]} l{sl[1]} [{a},{b}) span={b - a}  " +
              " ".join(f"{s}:{w}" for s, w in stg))
    print("stage span totals:", dict(sorted(stage_tot.items(), key=lambda t: -t[1])))


if __name__ == "__main__":
    main()
