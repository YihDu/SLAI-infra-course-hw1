"""Bounded multiple-choice MILP over executable D1 projection partitions.

Analytic stage latency proxy, NOT frozen cycle prediction. Event cap is empirical,
not a proof of race-checker memory. No evaluator or scoring behavior is changed.
Run with isolated highspy==1.15.1 environment and PYTHONPATH=repo.
"""
import itertools
import json
from math import ceil
from pathlib import Path
import highspy
from codesign.challenge.hardware import Hardware
from codesign.challenge.service import mma_service, vector_cycles

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
hw=Hardware.from_dict(json.loads((ROOT/'project/experiments/cand-112/inputs/hardware.json').read_text()))
stages={'qkv':(128,384),'wo':(128,128),'w1':(128,512),'w2':(512,128)}
parent=json.loads((ROOT/'project/experiments/cand-112/metrics.json').read_text())
base_events=parent['static']['M2_D1']['expanded_instructions']-8-8-160
# Four WG begins/ends, eight commits, 160 coarse barriers.
assert base_events==15056
base_gemm_events=sum(16*(n//64)*(2+3*(k//16)) for k,n in stages.values())
fixed_events=base_events-base_gemm_events
options={}
for stage,(k,n) in stages.items():
 options[stage]=[]
 for tile in (16,32,64,128):
  for chunk in (8,16,32,64):
   if tile*chunk>1024 or k%chunk:continue
   for owners in (1,2,3,4):
    if n%(tile*owners):continue
    demand=mma_service(hw,1,tile,chunk)
    # Sum of per-instruction service on one tape; shared NoC floor prevents
    # claiming unlimited speedup from per-SM DMA bandwidth alone.
    ld_a=4+ceil(chunk*4/128)+ceil(chunk*4/64)
    ld_b=4+ceil(chunk*tile*4/128)+ceil(chunk*tile*4/64)
    mma=1+ceil(demand.rf_read_bytes/128)+demand.compute_cycles+ceil(demand.rf_write_bytes/64)
    st=4+ceil(tile*4/128)*2
    per_panel=vector_cycles(tile,hw.vector_lanes)+ceil(tile*4/64)+(k//chunk)*(ld_a+ld_b+mma)+st
    cost=max((n//tile//owners)*per_panel,ceil(4*(k*n+(n//tile)*k+n)/hw.noc_bytes_per_cycle))
    events=16*(n//tile)*(2+3*(k//chunk))
    options[stage].append(dict(owners=owners,n_tile=tile,k_chunk=chunk,cost_proxy=cost,events=events,rf_words=tile*chunk))
solver=highspy.Highs();solver.setOptionValue('output_flag',False);solver.setOptionValue('time_limit',10.0);solver.setOptionValue('threads',1)
x={}
for stage,opts in options.items():
 for i,o in enumerate(opts):x[stage,i]=solver.addVariable(lb=0,ub=1,type=highspy.HighsVarType.kInteger,name=f'{stage}_{i}')
 solver.addConstr(sum(x[stage,i] for i in range(len(opts)))==1)
event_cap=15500
solver.addConstr(sum(o['events']*x[s,i] for s,opts in options.items() for i,o in enumerate(opts))<=event_cap-fixed_events)
solver.minimize(sum(16*o['cost_proxy']*x[s,i] for s,opts in options.items() for i,o in enumerate(opts)))
assert solver.getModelStatus()==highspy.HighsModelStatus.kOptimal
selected={s:next(o for i,o in enumerate(opts) if solver.val(x[s,i])>.5) for s,opts in options.items()}
# Independent exhaustive oracle for this deliberately small model.
feasible=(combo for combo in itertools.product(*options.values()) if fixed_events+sum(o['events'] for o in combo)<=event_cap)
oracle=min(sum(16*o['cost_proxy'] for o in combo) for combo in feasible)
assert abs(oracle-solver.getObjectiveValue())<1e-6
result=dict(solver='HiGHS '+solver.version(),status='optimal_for_proxy_model',objective=solver.getObjectiveValue(),enumeration_objective=oracle,selected=selected,event_cap=event_cap,estimated_events=fixed_events+sum(o['events'] for o in selected.values()),fixed_events=fixed_events,options_per_stage={s:len(v) for s,v in options.items()},caveats=['Objective is a service proxy, not full cycles.','Event budget is empirical; exact validation-cost model and bounded frozen grade remain required.','Placement uses SM0..owners-1; this first experiment does not solve general placement/fusion/residency.'])
solver.writeModel(str(OUT/'model.lp'))
(OUT/'solution.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'options.json').write_text(json.dumps(options,indent=2)+'\n')
print(json.dumps(result,indent=2))
