"""Count real generated HBM operand bytes by ABI/scratch tensor; not cache misses."""
import bisect
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from codesign.challenge.isa import iter_parse
from codesign.challenge.hbm_race import _bounds
root=Path(__file__).resolve().parents[3]
candidate=root/'project/experiments/cand-098'
spec=importlib.util.spec_from_file_location('traffic_compiler',candidate/'inputs/compiler.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
original=c.Builder.alloc
allocations=[]
def alloc(self,name,*shape):
 t=original(self,name,*shape);allocations.append((t.offset,t.offset+math.prod(t.shape),name,False));return t
c.Builder.alloc=alloc
result={}
for case,generate in [('M1_P1',c.generate_m1_p1),('M2_D1',c.generate_m1_d1)]:
 allocations.clear()
 recipe=json.loads((candidate/'inputs/recipe.json').read_text())
 text,layout,_=generate(recipe=recipe)
 assert text==(candidate/f'programs/{case}.asm').read_text()
 regions=sorted(allocations+[(s.address//4,(s.address+s.nbytes)//4,s.name,s.readonly) for s in layout.symbols.values() if s.name!='scratch'])
 starts=[r[0] for r in regions];counts=defaultdict(lambda:dict(read_bytes=0,write_bytes=0,loads=0,stores=0))
 ops=defaultdict(int)
 for ins in iter_parse(text):
  ops[ins.op]+=1
  if ins.op not in ('LD','ST'):continue
  for field in ('src','dst'):
   a=ins.args[field]
   if a['space']!='HBM':continue
   lo,hi=_bounds(a);region=regions[bisect.bisect_right(starts,lo)-1]
   assert lo>=region[0] and hi<=region[1],(case,ins,region)
   name=region[2];direction='read' if field=='src' else 'write'
   entry=counts[name];entry[direction+'_bytes']+=a['count']*4
   entry['loads' if field=='src' else 'stores']+=1
 aggregates=defaultdict(lambda:dict(read_bytes=0,write_bytes=0,loads=0,stores=0))
 for name,entry in counts.items():
  kind=name.rsplit('/',1)[-1] if '/' in name else name.split('_',1)[-1]
  for k,v in entry.items():aggregates[kind][k]+=v
 weights={k:v for k,v in counts.items() if k.startswith('layer') and k.rsplit('/',1)[-1] in ('wqkv','wo','w1','w2','b1','b2','ln1_g','ln1_b','ln2_g','ln2_b')}
 unique=sum(s.nbytes for s in layout.symbols.values() if s.name in weights)
 result[case]=dict(ops=ops,aggregates=aggregates,weights_read_bytes=sum(v['read_bytes'] for v in weights.values()),
                   unique_weight_bytes=unique,weights=weights,tensors=counts)
(root/'project/campaigns/validation-efficient/traffic.json').write_text(json.dumps(result,indent=2)+'\n')
for case,r in result.items():print(case,'weights',r['unique_weight_bytes'],'requested',r['weights_read_bytes'],'WG',r['ops']['WG.BEGIN'],'commits',r['ops']['STEP.COMMIT'],r['aggregates'])
