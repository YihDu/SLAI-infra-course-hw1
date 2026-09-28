"""Requested ordered structural calibration series; no tile/hardware sweep."""
import json
from pathlib import Path
from project.search import state,dsl,engine
root=Path(__file__).resolve().parents[3]
reg=state.load();base=root/'project/experiments/cand-098/inputs'
hw=json.loads((base/'hardware.json').read_text());compiler=root/'project/compiler.py'
ids=[]
for stages in [['w1'],['w2'],['w1','w2'],['qkv'],['qkv','w1','w2']]:
 r=json.loads((base/'recipe.json').read_text())
 r['p1_schedule']=dict(kind='incremental_gemm',gemms=stages,shards=4)
 cid=state.next_id(reg)
 prop=dict(parent='cand-098',hypothesis='Incremental full P1: '+','.join(stages)+' prompt GEMMs on 4 persistent SMs; mature Attention/LN/residual, D1 and hardware unchanged',
    ops=[dict(op='p1_schedule',**r['p1_schedule'])],family='p1_incremental',bias='p1',kind='manual',
    struct_sig='p1-incremental-'+','.join(stages),fingerprint=dsl.fingerprint(r,hw,compiler.read_text()),
    recipe=r,hardware=hw,compiler_path=str(compiler),
    validation_cost_calibration=True,
    calibration_reason='User requested one complete grade attempt for each of five incremental integration structures; bounded at 3 GiB, not a successful cost-screen result.')
 engine._materialize(reg,prop,reg.get('search',{}).get('round',0),cid);ids.append(cid)
state.save(reg)
(root/'project/campaigns/validation-efficient/series.json').write_text(json.dumps(ids,indent=2)+'\n')
print(ids)
