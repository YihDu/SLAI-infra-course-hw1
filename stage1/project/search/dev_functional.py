"""Dev-time functional check that skips the frozen HBM race validator.

`check_case` pays O(events^2) memory for `validate_hbm_races` before touching
tensors, which makes multi-WG iteration expensive.  This module runs the same
fixture execution and oracle comparison with the race check bypassed
(`hbm_races_validated=True`).  It is a development aid only: a program that
passes here can still carry HBM races, and every kept candidate still needs the
real frozen grade for eligibility.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from codesign.challenge.abi import build_layout  # noqa: F401  (kept for parity with runner imports)
from codesign.challenge.workload import MODELS
from codesign.challenge.hardware import Hardware
from codesign.challenge.reference import run_scenario
from codesign.challenge.runner import make_fixture, required_hbm_words
from codesign.challenge.isa import iter_parse


def check_functional(case: str, seed: int, hardware: Hardware, program: str) -> dict:
    """Same oracle comparison as runner.check_case, minus race validation."""
    fixture = make_fixture(case, seed, hardware, program)
    result = fixture.machine.execute(iter_parse(program), hbm_races_validated=True)
    model = MODELS[case.split("_")[0]]
    expected_hidden, expected_kv = run_scenario(model, case.split("_")[1], seed)
    expected = {"output/hidden": expected_hidden}
    for layer, (key, value) in enumerate(expected_kv):
        expected[f"layer{layer}/new_k"] = key
        expected[f"layer{layer}/new_v"] = value
    checks = {}
    for name, oracle in expected.items():
        symbol = fixture.layout.symbols[name]
        start = symbol.address // 4
        actual = result[start : start + oracle.size].reshape(oracle.shape)
        delta = np.abs(actual.astype(np.float64) - oracle.astype(np.float64))
        finite = bool(np.isfinite(delta).all())
        checks[name] = {
            "max_absolute_error": float(delta.max()) if finite else None,
            "allclose_1e-3_1e-3": finite
            and bool(np.allclose(actual, oracle, rtol=1e-3, atol=1e-3)),
            "all_elements_written": all(
                index in fixture.machine.written_hbm
                for index in range(start, start + oracle.size)
            ),
        }
    return {
        "case": case,
        "passed": all(
            v["allclose_1e-3_1e-3"] and v["all_elements_written"]
            for v in checks.values()
        ),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("M1_P1", "M2_D1"), required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    hardware = Hardware.from_dict(json.loads(args.hardware.read_text()))
    out = check_functional(
        args.case, args.seed, hardware, args.program.read_text()
    )
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
