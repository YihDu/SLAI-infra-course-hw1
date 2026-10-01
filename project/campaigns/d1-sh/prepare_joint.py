"""Combine a screened D1 recipe with cand-149's P1 without staging worktree."""
import hashlib
import json
from pathlib import Path
import sys

from project import compiler

variant = Path(sys.argv[1])
parent = Path("project/experiments/cand-149")
out = variant / "joint"
out.mkdir(exist_ok=False)
recipe = json.loads((parent / "inputs/recipe.json").read_text())
recipe["d1_schedule"] = json.loads((variant / "recipe.json").read_text())["d1_schedule"]
programs = out / "programs"
programs.mkdir()
p1, _, _ = compiler.generate_m1_p1(recipe)
d1, _, _ = compiler.generate_m1_d1(recipe=recipe)
assert p1 == (parent / "programs/M1_P1.asm").read_text(), "P1 changed unexpectedly"
assert d1 == (variant / "M2_D1.asm").read_text(), "D1 changed unexpectedly"
(out / "compiler.py").write_text(Path(compiler.__file__).read_text())
(out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
(out / "hardware.json").write_text((variant / "hardware.json").read_text())
(programs / "M1_P1.asm").write_text(p1)
(programs / "M2_D1.asm").write_text(d1)
manifest = {"parent": "cand-149", "status": "prepared, joint grade pending",
            "P1_identical_to_parent": True, "D1_identical_to_screen": True,
            "sha256": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in out.rglob("*") if p.is_file()}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
