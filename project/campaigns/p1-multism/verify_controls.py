"""Check that the serial full-grade control computes exactly the parallel schedule."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
exp=root/'project/experiments'

def canonical(program):
    def bind(v):
        if isinstance(v,dict):return {k:('g' if k=='wg' else bind(x)) for k,x in v.items()}
        if isinstance(v,list):return [bind(x) for x in v]
        return v
    result=[]
    for line in program.splitlines():
        if not line.strip():continue
        op,args=line.split(' ',1)
        if op in ('WG.BEGIN','WG.END'):continue
        a=bind(json.loads(args))
        if op=='BARRIER':a['wgs']=['g']
        result.append((op,a))
    return result

parallel=(exp/'cand-099/programs/M1_P1.asm').read_text()
serial=(exp/'cand-100/programs/M1_P1.asm').read_text()
assert canonical(parallel)==canonical(serial)
ref=(exp/'cand-098/programs/M2_D1.asm').read_bytes()
assert all((exp/c/'programs/M2_D1.asm').read_bytes()==ref for c in ('cand-099','cand-100'))
hw=(exp/'cand-098/inputs/hardware.json').read_text()
assert all(json.loads((exp/c/'inputs/hardware.json').read_text())==json.loads(hw) for c in ('cand-099','cand-100'))
from project.compiler import generate_m1_p1,generate_m1_d1
recipe=json.loads((exp/'cand-098/inputs/recipe.json').read_text())
for name,fn in [('M1_P1',generate_m1_p1),('M2_D1',generate_m1_d1)]:
    assert fn(recipe=recipe)[0]==(exp/f'cand-098/programs/{name}.asm').read_text()
report=dict(serial_parallel_same_arithmetic_and_addresses=True,d1_byte_identical=True,
            hardware_unchanged=True,default_programs_byte_identical_to_cand098=True,
            limitation='Serialization equivalence does not certify cross-SM races, scheduling, timing, or power.')
(root/'project/campaigns/p1-multism/control-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
