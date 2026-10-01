"""Register an isolated P1/D1 combination, preserving the worktree compiler."""
import argparse
import json
from pathlib import Path

from project import compiler
from project.search import state
from project.search.candidate import Candidate, EXPERIMENTS
from project.search.coordinator import writer_lock

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--p1",type=Path,required=True,help="P1 directory with recipe.json")
p.add_argument("--d1",type=Path,required=True,help="D1 directory with recipe.json")
p.add_argument("--hardware",type=Path,required=True)
p.add_argument("--parent",default="cand-149")
p.add_argument("--hypothesis",required=True)
p.add_argument("--expect-p1",type=Path,required=True)
p.add_argument("--expect-d1",type=Path,required=True)
args=p.parse_args()
recipe=json.loads((args.p1/"recipe.json").read_text())
recipe["d1_schedule"]=json.loads((args.d1/"recipe.json").read_text())["d1_schedule"]
p1=compiler.generate_m1_p1(recipe)[0]
d1=compiler.generate_m1_d1(recipe=recipe)[0]
assert p1==args.expect_p1.read_text(),"P1 snapshot mismatch"
assert d1==args.expect_d1.read_text(),"D1 snapshot mismatch"
with writer_lock():
    reg=state.load()
    cid=state.next_id(reg)
    assert not (EXPERIMENTS/cid).exists()
    cand=Candidate(cid,args.parent,args.hypothesis,
                   "Isolated structural search; sources in proposal.json",kind="manual")
    cand.save()
    (cand.inputs/"compiler.py").write_text(Path(compiler.__file__).read_text())
    (cand.inputs/"recipe.json").write_text(json.dumps(recipe,indent=2)+"\n")
    (cand.inputs/"hardware.json").write_text(args.hardware.read_text())
    (cand.programs/"M1_P1.asm").write_text(p1)
    (cand.programs/"M2_D1.asm").write_text(d1)
    (cand.dir/"proposal.json").write_text(json.dumps({
        "sources":{k:str(v) for k,v in vars(args).items()},
        "program_equality_verified":True},indent=2)+"\n")
    reg["candidates"][cid]={"parent":args.parent,"status":"created",
                            "hypothesis":args.hypothesis}
    state.save(reg)
print(cid)
