"""Generate P1 recipe variants from the worktree recipe and write them to a dir.

Usage: python variants.py OUTDIR NAME=JSON_PATCH ...
Each JSON patch is deep-merged into project/recipe.json; lists replace.
Writes OUTDIR/NAME/{recipe.json,M1_P1.asm} and prints event counts.
"""
import copy
import json
import sys
from pathlib import Path

from codesign.challenge.isa import iter_parse
from project import compiler

ROOT = Path(__file__).resolve().parents[3]


def merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def main():
    out = Path(sys.argv[1])
    base = json.loads((ROOT / "project/recipe.json").read_text())
    for spec in sys.argv[2:]:
        name, patch = spec.split("=", 1)
        recipe = merge(copy.deepcopy(base), json.loads(patch))
        text, _, _ = compiler.generate_m1_p1(recipe)
        text = "\n".join(text) if isinstance(text, list) else text
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
        (d / "M1_P1.asm").write_text(text)
        print(name, "events", sum(1 for _ in iter_parse(text)))


if __name__ == "__main__":
    main()
