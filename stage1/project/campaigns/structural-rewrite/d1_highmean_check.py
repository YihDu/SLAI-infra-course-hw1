"""Numerics check for LN algebraic folds: high-mean, low-variance D1 inputs.

Patches the fixture/reference input generator (dev only) so every hidden
input is scale*x + mean, then runs dev_functional's oracle comparison.
Usage: python d1_highmean_check.py PROGRAM [--hw JSON] [--mean 20 --scale 0.05]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import codesign.challenge.reference as ref
import codesign.challenge.runner as runner
from codesign.challenge.hardware import Hardware
from project.search.dev_functional import check_functional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--hw", default=None)
    ap.add_argument("--mean", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--seeds", default="7,19")
    args = ap.parse_args()
    original = ref.make_inputs

    def shifted(*a, **k):
        weights, inputs = original(*a, **k)
        return weights, (inputs * args.scale + args.mean).astype(inputs.dtype)

    ref.make_inputs = runner.make_inputs = shifted
    hw_dict = json.loads((ROOT / "hardware.json").read_text())
    hw_dict.update(json.loads(args.hw) if args.hw else {})
    hw = Hardware.from_dict(hw_dict)
    program = Path(args.program).read_text()
    for seed in map(int, args.seeds.split(",")):
        r = check_functional("M2_D1", seed, hw, program)
        worst = max((v["max_absolute_error"] or float("inf")) for v in r["checks"].values())
        print(f"seed={seed} mean={args.mean} scale={args.scale} passed={r['passed']} "
              f"max_abs_err={worst:.3e}")


if __name__ == "__main__":
    main()
