"""Find the exact peak 1000-cycle power window and attribute its energy.

Dev-only analysis: wraps _EnergyRuns.add to capture (kind, start, finish, pj)
runs, then reports the hottest window's resource breakdown.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from codesign.challenge.hardware import Hardware, cost_model
from codesign.challenge.isa import iter_parse
import codesign.challenge.pipeline as _pl
import codesign.challenge.pipeline_events as _pe
from codesign.challenge.runner import required_hbm_words

RUNS = []
_orig_add = _pl._EnergyRuns.add


def _spy_add(self, key, start, finish, energy):
    RUNS.append((key[0], start, finish, energy))
    return _orig_add(self, key, start, finish, energy)


def main():
    hw_path, asm_path = sys.argv[1], sys.argv[2]
    hw = Hardware.from_dict(json.loads(Path(hw_path).read_text()))
    text = Path(asm_path).read_text()
    sys.path.insert(0, str(ROOT / "project"))
    from project.compiler import generate_m1_p1, load_recipe
    recipe = load_recipe()
    if len(sys.argv) > 3:
        def _merge(r, o):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(r.get(k), dict):
                    _merge(r[k], v)
                else:
                    r[k] = v
        _merge(recipe, json.loads(sys.argv[3]))
    _, layout, _ = generate_m1_p1(recipe)
    words = required_hbm_words(text, layout)
    _pl._EnergyRuns.add = _spy_add
    _pe._EnergyRuns = _pl._EnergyRuns  # same class object already
    t0 = time.time()
    res = _pl.estimate_pipeline(hw, iter_parse(text), words, None, None)
    print(f"estimate {res.cycles} cyc peakW {res.peak_window_power_w:.2f} "
          f"({time.time()-t0:.0f}s) runs={len(RUNS)}", flush=True)

    window = cost_model()["power_window_cycles"]
    dens = np.zeros(res.cycles + window + 1, np.float64)
    for kind, start, finish, pj in RUNS:
        span = max(1, finish - start)
        dens[start] += pj / span
        dens[finish] -= pj / span
    prof = np.cumsum(dens)
    csum = np.concatenate([[0.0], np.cumsum(prof)])
    starts = np.arange(0, res.cycles)
    energies = csum[starts + window] - csum[starts]
    peak_i = int(np.argmax(energies))
    lo, hi = peak_i, peak_i + window
    print(f"peak window [{lo},{hi})", flush=True)

    by_kind = defaultdict(float)
    for kind, start, finish, pj in RUNS:
        ov = min(finish, hi) - max(start, lo)
        if ov > 0:
            by_kind[kind] += pj * ov / max(1, finish - start)
    for kind, pj in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {kind:18s} {pj/1e6:8.2f}uJ  {pj/1e6/(window*5e-4):6.2f}W")
    (ROOT / "project/campaigns/structural-rewrite/peak_window.json").write_text(
        json.dumps({"window": [lo, hi], "cycles": res.cycles,
                    "peakW": res.peak_window_power_w,
                    "by_kind": {k: v / 1e6 for k, v in by_kind.items()}},
                   indent=1))


if __name__ == "__main__":
    main()
