"""Generate D1 recipe variants (deep-merged into project/recipe.json)."""
import copy, json, sys
from pathlib import Path
from codesign.challenge.isa import iter_parse
from project import compiler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bound-model"))
from variants import merge  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1])
base = json.loads((ROOT / "project/recipe.json").read_text())
for spec in sys.argv[2:]:
    name, patch = spec.split("=", 1)
    recipe = merge(copy.deepcopy(base), json.loads(patch))
    text, _, _ = compiler.generate_m1_d1(recipe=recipe)
    text = "\n".join(text) if isinstance(text, list) else text
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    (d / "M2_D1.asm").write_text(text)
    print(name, "events", sum(1 for _ in iter_parse(text)))
