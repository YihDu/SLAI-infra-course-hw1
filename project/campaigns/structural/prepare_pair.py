"""Freeze a screened P1 and D1 into one shared-hardware joint candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

p1_dir = Path(sys.argv[1]).resolve()
d1_dir = Path(sys.argv[2]).resolve()
out = p1_dir / "joint"
out.mkdir(exist_ok=False)
(out / "programs").mkdir()

spec = importlib.util.spec_from_file_location("p1_pair_compiler", p1_dir / "compiler.py")
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)
recipe = json.loads((p1_dir / "recipe.json").read_text())
recipe["d1_schedule"] = json.loads((d1_dir / "recipe.json").read_text())["d1_schedule"]
p1 = compiler.generate_m1_p1(recipe)[0]
d1 = compiler.generate_m1_d1(recipe=recipe)[0]
assert p1 == (p1_dir / "M1_P1.asm").read_text(), "P1 regeneration mismatch"
assert d1 == (d1_dir / "M2_D1.asm").read_text(), "D1 regeneration mismatch"
hardware = json.loads((p1_dir / "hardware.json").read_text())
assert hardware == json.loads((d1_dir / "hardware.json").read_text()), "shared hardware mismatch"
(out / "compiler.py").write_bytes((p1_dir / "compiler.py").read_bytes())
(out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
(out / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
(out / "programs/M1_P1.asm").write_text(p1)
(out / "programs/M2_D1.asm").write_text(d1)
manifest = {"P1_source":str(p1_dir), "D1_source":str(d1_dir),
            "exact_programs":True,
            "sha256":{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in out.rglob("*") if p.is_file()}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(out)
