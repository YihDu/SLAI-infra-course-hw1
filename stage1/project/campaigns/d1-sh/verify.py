"""Frozen D1 correctness/race and timing verification for a screened variant."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import check_case, estimate_case

directory = Path(sys.argv[1])
program = (directory / "M2_D1.asm").read_text()
hardware = Hardware.from_dict(json.loads((directory / "hardware.json").read_text()))
report = {"scope": "D1 only; not a joint grade"}
try:
    for seed in (7, 19):
        report[f"check_{seed}"] = check_case("M2_D1", seed, hardware, program)
        (directory / "frozen-verification.json").write_text(json.dumps(report, indent=2))
        assert report[f"check_{seed}"]["passed"], f"seed {seed} failed"
        print(f"frozen check seed {seed} passed", flush=True)
    report["estimate"] = estimate_case("M2_D1", hardware, program)
    (directory / "frozen-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report["estimate"][key] for key in
                      ("cycles", "area_mm2", "peak_window_power_w")}), flush=True)
finally:
    with Path("project/iteration-log.md").open("a") as log:
        log.write(f"\n## {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — Frozen D1 verification {directory.name}\n"
                  f"- Change: none; verify screened variant using unmodified check_case and estimate_case.\n"
                  f"- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/verify.py {directory}` -> `{directory}/frozen-verification.json`.\n"
                  f"- Result: check seeds={[s for s in (7,19) if report.get(f'check_{s}', {}).get('passed')]}; timing completed={'estimate' in report}.\n"
                  "- Decision: D1 evidence only; joint grade required before promotion.\n")
