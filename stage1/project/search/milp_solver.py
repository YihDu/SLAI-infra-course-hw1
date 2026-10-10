"""Pure bounded D1 MILP proposal process; run in isolated highspy environment."""
import argparse
import json
from math import ceil
from pathlib import Path
from codesign.challenge.hardware import Hardware
from codesign.challenge.service import mma_service, vector_cycles

STAGES = {'qkv': (128,384), 'wo': (128,128), 'w1': (128,512), 'w2': (512,128)}


def options_for(hw):
    result = {}
    capacity = 16384 // hw.vector_lanes
    for stage, (k,n) in STAGES.items():
        opts = []
        for tile in (16,32,64,128):
            for chunk in (8,16,32,64):
                if tile*chunk > min(capacity,1024) or k % chunk:
                    continue
                for owners in range(1,min(4,hw.sm_count)+1):
                    if n % (tile*owners):
                        continue
                    demand=mma_service(hw,1,tile,chunk)
                    # This implementation is calibrated only for current RF/NoC hardware.
                    ld_a=4+ceil(chunk*4/128)+ceil(chunk*4/64)
                    ld_b=4+ceil(chunk*tile*4/128)+ceil(chunk*tile*4/64)
                    mma=1+ceil(demand.rf_read_bytes/128)+demand.compute_cycles+ceil(demand.rf_write_bytes/64)
                    st=4+ceil(tile*4/128)*2
                    panel=vector_cycles(tile,hw.vector_lanes)+ceil(tile*4/64)+(k//chunk)*(ld_a+ld_b+mma)+st
                    cost=max((n//tile//owners)*panel,ceil(4*(k*n+(n//tile)*k+n)/hw.noc_bytes_per_cycle))
                    opts.append(dict(owners=owners,n_tile=tile,k_chunk=chunk,cost_proxy=cost,
                                     events=16*(n//tile)*(2+3*(k//chunk))))
        result[stage]=opts
    return result


def solve(request):
    import highspy
    hw=Hardware.from_dict(request['hardware'])
    if hw.rf_ports!='8R4W' or hw.sm_noc_bytes_per_cycle!=128 or hw.vector_lanes!=16:
        raise ValueError('Uncalibrated hardware for current D1 service model')
    options=options_for(hw)
    for stage,opts in options.items():
        for o in opts:
            key=stage+'/'+':'.join(str(o[k]) for k in ('owners','n_tile','k_chunk'))
            o['uncalibrated_cost_proxy']=o['cost_proxy']
            o['cost_proxy']=max(1.0,o['cost_proxy']+request.get('corrections',{}).get(key,0.0))
    h=highspy.Highs()
    h.setOptionValue('output_flag',False);h.setOptionValue('threads',1)
    h.setOptionValue('time_limit',request.get('seconds',10))
    h.setOptionValue('random_seed',0)
    x={}
    for s,opts in options.items():
        for i in range(len(opts)):
            x[s,i]=h.addVariable(lb=0,ub=1,type=highspy.HighsVarType.kInteger,name=f'{s}_{i}')
        h.addConstr(sum(x[s,i] for i in range(len(opts)))==1)
    fixed=request['fixed_events'];cap=request['event_cap']
    h.addConstr(sum(o['events']*x[s,i] for s,opts in options.items() for i,o in enumerate(opts))<=cap-fixed)
    for plan in request.get('excluded',[]):
        matches=[]
        for s,opts in options.items():
            for i,o in enumerate(opts):
                if all(o[k]==plan.get(s,{}).get(k) for k in ('owners','n_tile','k_chunk')):
                    matches.append(x[s,i]);break
        if len(matches)==len(STAGES):
            h.addConstr(sum(matches)<=len(STAGES)-1)
    objective=sum(16*o['cost_proxy']*x[s,i] for s,opts in options.items() for i,o in enumerate(opts))
    h.minimize(objective)
    status=h.getModelStatus()
    if status==highspy.HighsModelStatus.kInfeasible:
        return {'status':'exhausted','solver':h.version()}
    if status!=highspy.HighsModelStatus.kOptimal:
        return {'status':'solver_incomplete','detail':str(status),'solver':h.version()}
    selected={s:next(o for i,o in enumerate(opts) if h.val(x[s,i])>.5) for s,opts in options.items()}
    return dict(status='optimal_for_proxy_model',solver=h.version(),selected=selected,
                objective=h.getObjectiveValue(),events=fixed+sum(o['events'] for o in selected.values()),
                event_cap=cap,model_version=2,seconds=h.getRunTime(),mip_gap=h.getInfo().mip_gap)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('request');parser.add_argument('output')
    args=parser.parse_args()
    result=solve(json.loads(Path(args.request).read_text()))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
