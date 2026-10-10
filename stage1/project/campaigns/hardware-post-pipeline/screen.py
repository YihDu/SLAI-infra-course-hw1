"""Single-field timing screen for the verified post-pipeline programs.

Estimate is ranking evidence only. The selected hardware must pass a complete
grade with the same programs before promotion.
"""
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "project/experiments/cand-122"
BASELINE = {"M1_P1": 30996995, "M2_D1": 1302032}

VARIANTS = {
    "current": {},
    "noc512": {"noc_bytes_per_cycle": 512},
    "dma4": {"dma_engines": 4},
    "sfu16": {"sfu_lanes": 16},
    "reduce2": {"reduction_units": 2},
    "hbm8": {"hbm_channels": 8},
    "cache4": {"cache_mib": 4},
    "cache8": {"cache_mib": 8},
    "cache16": {"cache_mib": 16},
    "multicast": {"multicast": True},
    "cache16_sfu16": {"cache_mib": 16, "sfu_lanes": 16},
    "cache16_sfu16_hbm8": {
        "cache_mib": 16, "sfu_lanes": 16, "hbm_channels": 8,
    },
    "cache16_sfu16_hbm8_reduce2": {
        "cache_mib": 16, "sfu_lanes": 16, "hbm_channels": 8,
        "reduction_units": 2,
    },
    "sfu16_hbm8_reduce2": {
        "sfu_lanes": 16, "hbm_channels": 8, "reduction_units": 2,
    },
}


def main():
    hardware = json.loads((PARENT / "inputs/hardware.json").read_text())
    def evaluate(item):
        name, changes = item
        variant = dict(hardware)
        variant.update(changes)
        hardware_path = HERE / f"hardware-{name}.json"
        report_path = HERE / f"estimate-{name}.json"
        hardware_path.write_text(json.dumps(variant, indent=2, sort_keys=True) + "\n")
        if not report_path.exists():
            subprocess.run([
                str(ROOT / ".venv/bin/python"), str(ROOT / "challenge.py"), "estimate",
                "--hardware", str(hardware_path),
                "--program-p1", str(PARENT / "programs/M1_P1.asm"),
                "--program-d1", str(PARENT / "programs/M2_D1.asm"),
                "--report", str(report_path),
            ], cwd=ROOT, check=True, timeout=600,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        report = json.loads(report_path.read_text())
        cases = {case: report["cases"][case]["timing"] for case in BASELINE}
        score = 1000 * math.sqrt(math.prod(BASELINE[c] / cases[c]["cycles"] for c in BASELINE))
        row = {
            "name": name, "changes": changes, "score_proxy": score,
            "cycles": {case: cases[case]["cycles"] for case in BASELINE},
            "area_mm2": cases["M1_P1"]["area_mm2"],
            "peak_power_w": {case: cases[case]["peak_window_power_w"] for case in BASELINE},
        }
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate, VARIANTS.items()))
    rows.sort(key=lambda row: row["score_proxy"], reverse=True)
    (HERE / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
