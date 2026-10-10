"""Cheap P1 timing screen for structural variants.

Generates M1_P1 under recipe overrides and times it with the dev-patched
estimator (dense hazard arrays, span-conservative overlap, race validation
off).  Screening only — kept candidates still go through frozen check/grade.

Usage: p1_screen.py <name> '<recipe overrides JSON>' [--hw '<overrides>']
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import codesign.challenge.hbm_race as _hbm_race
import codesign.challenge.pipeline as _pl
import codesign.challenge.pipeline_events as _pe
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words

from project.compiler import generate_m1_p1, load_recipe

OUT = ROOT / "project/campaigns/structural-rewrite"


def merge(recipe, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(recipe.get(k), dict):
            merge(recipe[k], v)
        else:
            recipe[k] = v
    return recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("overrides", nargs="?", default="{}")
    ap.add_argument("--hw", default=None)
    args = ap.parse_args()

    _hbm_race.validate_hbm_races = lambda *a, **k: None

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
    _pl._MemoryTimes.sized = classmethod(
        lambda cls, words, sparse=False: _pl._MemoryTimes(
            np.zeros(words, np.int64), np.zeros(words, np.int64)))

    recipe = merge(load_recipe(), json.loads(args.overrides))
    t0 = time.time()
    text, layout, cursor = generate_m1_p1(recipe)
    asm = OUT / f"p1_{args.name}.asm"
    asm.write_text(text)
    print(f"generated {asm.name}: {len(text.splitlines())} lines "
          f"in {time.time()-t0:.1f}s", flush=True)

    hw_dict = json.loads((ROOT / "hardware.json").read_text())
    if args.hw:
        hw_dict.update(json.loads(args.hw))
    hw = Hardware.from_dict(hw_dict)
    words = required_hbm_words(text, layout)
    trace = {}
    t0 = time.time()
    res = estimate_pipeline(hw, iter_parse(text), words, None, trace)
    print(f"[{args.name}] cycles={res.cycles} hbm_r={res.hbm_read_bytes/1e6:.2f}MB "
          f"hbm_w={res.hbm_write_bytes/1e6:.2f}MB "
          f"cache={res.cache_hits}/{res.cache_hits+res.cache_misses} "
          f"peakW={res.peak_window_power_w:.2f} ({time.time()-t0:.0f}s)",
          flush=True)
    (OUT / f"p1_{args.name}.json").write_text(json.dumps({
        "cycles": res.cycles,
        "hbm_read_bytes": res.hbm_read_bytes,
        "hbm_write_bytes": res.hbm_write_bytes,
        "cache_hits": res.cache_hits,
        "cache_misses": res.cache_misses,
        "peak_power_w": res.peak_window_power_w,
        "stats": getattr(res, "resource_stats", None),
    }, indent=1, default=str))


if __name__ == "__main__":
    main()
