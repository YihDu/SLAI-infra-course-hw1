"""Inspect emitted ownership/operations; these are static counts, not utilization."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from codesign.challenge.isa import iter_parse
root=Path(__file__).resolve().parents[3]
p=root/'project/experiments/cand-099/programs/M1_P1.asm'
counts=defaultdict(Counter);sync=Counter();groups={}
for ins in iter_parse(p.read_text()):
 a=ins.args
 if ins.op=='WG.BEGIN':groups[a['wg']]=a['sm']
 if ins.op in ('BARRIER','WAIT','STEP.COMMIT'):sync[ins.op]+=1
 if 'event' not in a:continue
 sm=groups[a['event'].split('_')[2].replace('sm','g')]
 c=counts[sm];c['instructions']+=1;c[ins.op]+=1
 if ins.op=='MMA.ACC':c['mma_flops']+=2*a['m']*a['n']*a['k']
 if ins.op in ('LD','ST'):
  for direction in ('src','dst'):
   if a[direction]['space']=='HBM':c['hbm_requested_read_bytes' if direction=='src' else 'hbm_requested_write_bytes']+=a[direction]['count']*4
out=dict(candidate='cand-099',sms=counts,synchronization=sync,note='Static ownership/request counts only; no measured utilization or cache-filtered HBM traffic.')
(root/'project/campaigns/p1-multism/static-schedule.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
