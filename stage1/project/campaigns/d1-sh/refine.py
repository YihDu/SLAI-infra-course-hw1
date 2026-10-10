"""Reproducible single-idea D1 screen; reports are dev evidence, not grades."""
import argparse
import importlib.util
import copy
from datetime import datetime, timezone
import json
import shlex
from pathlib import Path
import sys

from codesign.challenge.hardware import Hardware
from project import compiler
from project.search.dev_functional import check_functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bound-model"))
import critpath
from variants import merge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--case", choices=("M1_P1", "M2_D1"), default="M2_D1")
    p.add_argument("--out-root", type=Path, default=Path("project/campaigns/d1-sh"))
    p.add_argument("--base", type=Path, default=Path("project/campaigns/d1-sh/d16_kv_accs/recipe.json"))
    p.add_argument("--hardware", type=Path, default=Path("project/campaigns/d1-sh/hardware-16sm-4ch.json"))
    p.add_argument("--patch", default="{}")
    p.add_argument("--hardware-patch", default="{}")
    p.add_argument("--compiler", type=Path, help="Pinned generator for queued campaigns")
    args = p.parse_args()
    generator = compiler
    if args.compiler:
        spec = importlib.util.spec_from_file_location("campaign_compiler", args.compiler)
        generator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = generator
        spec.loader.exec_module(generator)
    out = args.out_root / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    recipe = merge(copy.deepcopy(json.loads(args.base.read_text())), json.loads(args.patch))
    hardware_data = json.loads(args.hardware.read_text())
    hardware_data.update(json.loads(args.hardware_patch))
    hardware = Hardware.from_dict(hardware_data)
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (out / "hardware.json").write_text(json.dumps(hardware_data, indent=2) + "\n")
    (out / "compiler.py").write_text(Path(generator.__file__).read_text())
    result = {}
    try:
        if hardware.area_mm2() > 24:
            raise ValueError("area exceeds 24 mm2")
        generate = generator.generate_m1_p1 if args.case == "M1_P1" else generator.generate_m1_d1
        program, _, _ = generate(recipe=recipe)
        (out / f"{args.case}.asm").write_text(program)
        for seed in (7, 19):
            check = check_functional(args.case, seed, hardware, program)
            (out / f"dev-check-{seed}.json").write_text(json.dumps(check, indent=2))
            if not check["passed"]:
                raise ValueError(f"functional comparison failed, seed {seed}")
        print(f"{args.case} functional seeds 7/19 passed; timing", flush=True)
        obs, timing, elapsed = critpath.run(hardware, program, args.case)
        result = critpath.summarize(obs, timing, elapsed)
        result["validation"] = "dev functional seeds 7/19; HBM race check pending"
        (out / "critpath.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("cycles", "peak_window_power_w", "area_mm2")}), flush=True)
    except Exception as exc:
        result = {"error": repr(exc)}
        (out / "error.json").write_text(json.dumps(result, indent=2))
        raise
    finally:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with Path("project/iteration-log.md").open("a") as log:
            log.write(f"\n## {now} — {args.case} screen {args.name}\n"
                      f"- Change: base={args.base}; patch={args.patch}; hardware={args.hardware}; hardware_patch={args.hardware_patch}\n"
                      f"- Verify: `PYTHONPATH=. .venv/bin/python project/campaigns/d1-sh/refine.py {shlex.join(sys.argv[1:])}`; reports `{out}`. Development checks only; full frozen race validation still required.\n"
                      f"- Result: {args.case}={result.get('cycles', 'failed')}; area={result.get('area_mm2')}; peak W={result.get('peak_window_power_w')}; error={result.get('error')}.\n"
                      "- Decision: screen only, no incumbent promotion; compare before full frozen validation.\n")


if __name__ == "__main__":
    main()
