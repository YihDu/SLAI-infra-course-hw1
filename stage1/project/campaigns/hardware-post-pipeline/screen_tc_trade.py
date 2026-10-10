"""Screen the legal cache/TC area exchange against cand-133."""
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / "tc-trade"
PARENT = ROOT / "project/experiments/cand-133"
BASELINE = {"M1_P1": 30996995, "M2_D1": 1302032}
VARIANTS = {
    "tc3_cache4_sh64": {"tc_count": 3, "cache_mib": 4, "shared_kib": 64},
    "tc3_cache2_sh64": {"tc_count": 3, "cache_mib": 2, "shared_kib": 64},
}


def evaluate(item):
    name, changes = item
    hardware = json.loads((PARENT / "inputs/hardware.json").read_text())
    hardware.update(changes)
    hardware_path = HERE / f"hardware-{name}.json"
    report_path = HERE / f"estimate-{name}.json"
    hardware_path.write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "challenge.py"), "estimate",
        "--hardware", str(hardware_path),
        "--program-p1", str(PARENT / "programs/M1_P1.asm"),
        "--program-d1", str(PARENT / "programs/M2_D1.asm"),
        "--report", str(report_path),
    ], cwd=ROOT, check=True, timeout=600,
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = json.loads(report_path.read_text())
    timing = {case: report["cases"][case]["timing"] for case in BASELINE}
    row = {
        "name": name, "changes": changes,
        "cycles": {case: timing[case]["cycles"] for case in BASELINE},
        "score_proxy": 1000 * math.sqrt(math.prod(
            BASELINE[case] / timing[case]["cycles"] for case in BASELINE
        )),
        "area_mm2": timing["M1_P1"]["area_mm2"],
        "peak_power_w": {
            case: timing[case]["peak_window_power_w"] for case in BASELINE
        },
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate, VARIANTS.items()))
    rows.sort(key=lambda row: row["score_proxy"], reverse=True)
    (HERE / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
