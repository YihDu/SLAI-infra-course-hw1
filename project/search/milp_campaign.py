"""Sequential MILP -> preflight -> frozen grade -> measured feedback campaign.

The coordinator owns all writes. Workers only receive files and return proposals.
Stopping at an inner plateau explicitly requests outer work; it is not a claim of
universal performance optimality. Resume preserves exclusions and stop evidence.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from . import state, engine, dsl
from .candidate import Candidate, REPO, EXPERIMENTS
from .coordinator import Budget, writer_lock, evaluate_many, recover
from .runtime import run as run_bounded, install_signal_handlers

FIELDS=('owners','n_tile','k_chunk')


def projection_plan(recipe):
    schedule=recipe.get('d1_schedule',{})
    if schedule.get('kind')!='column_head':
        return None
    if schedule.get('projections'):
        return {s:{k:int(o[k]) for k in FIELDS} for s,o in schedule['projections'].items()}
    if schedule.get('partition')!='panel_owners':
        return None  # The old all-N32 generator is a different implementation.
    return {s:dict(owners=owners,n_tile=64,k_chunk=16)
            for s,owners in [('qkv',3),('wo',2),('w1',4),('w2',2)]}


def update_feedback(campaign, metrics, before, solution):
    grade=metrics.get('reports',{}).get('grade',{})
    measured=grade.get('eligible') is True and grade.get('score') is not None
    row={'candidate':metrics['candidate'],'verdict':metrics.get('verdict'),
         'failure':metrics.get('failure_class'),'proxy':solution['objective'],
         'predicted_events':solution['events'],
         'peak_memory_mib':metrics.get('runs',{}).get('grade',{}).get('peak_memory_mib')}
    if measured:
        row.update(score=grade['score'],cycles={s:c['cycles'] for s,c in grade['cases'].items()},
                   incumbent_gain=grade['score']/before-1)
        campaign['flat_valid']=0 if row['incumbent_gain']>=.10 else campaign['flat_valid']+1
        if campaign['flat_valid']>=3:
            campaign['stop_reason']='outer_structure_required'
    elif row['failure']=='memory_limit':
        # Empirical trust-region restriction for this skeleton only. Never use
        # a failed measurement as a performance sample or a universal cut.
        campaign['event_cap']=min(campaign['event_cap'],solution['events']-1)
    campaign['observations'].append(row)
    return row


def learn_projection_delta(campaign, parent_recipe, recipe, parent_cycles, cycles, hardware, candidate):
    """Conditional correction from an actual one-projection controlled comparison.

    Do not infer stage costs when several projections changed. A correction is
    local to this shared hardware/skeleton; interactions remain unmodelled.
    """
    from .milp_solver import options_for
    from codesign.challenge.hardware import Hardware
    left,right=projection_plan(parent_recipe),projection_plan(recipe)
    if not left or not right:return None
    changed=[stage for stage in left if left[stage]!=right.get(stage)]
    if len(changed)!=1:return None
    stage=changed[0];options=options_for(Hardware.from_dict(hardware))[stage]
    def key(o):return stage+'/'+':'.join(str(o[k]) for k in FIELDS)
    def cost(o):return next((x['cost_proxy'] for x in options if all(x[k]==o[k] for k in FIELDS)),None)
    a,b=cost(left[stage]),cost(right[stage])
    if a is None or b is None:return None
    corrections=campaign.setdefault('corrections',{})
    correction=corrections.get(key(left[stage]),0.0)+(cycles-parent_cycles)/16-(b-a)
    corrections[key(right[stage])]=correction
    evidence=dict(candidate=candidate,stage=stage,actual_case_delta=cycles-parent_cycles,
                  proxy_case_delta=16*(b-a),option=key(right[stage]),correction_per_invocation=correction)
    campaign.setdefault('calibration_evidence',[]).append(evidence)
    return evidence


def run(args):
    reg=state.load();recover(reg)
    folder=Path(args.resume).resolve() if args.resume else REPO/'project/campaigns'/time.strftime('auto-milp-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True,exist_ok=bool(args.resume))
    checkpoint=folder/'campaign.json'
    if args.resume:
        campaign=json.loads(checkpoint.read_text())
    else:
        campaign=dict(parent=reg['head'],event_cap=15500,fixed_events=5264,
                      flat_valid=0,observations=[],excluded=[],attempts=[],stop_reason=None)
    if args.resume and campaign['stop_reason'] in ('budget_exhausted','proposal_limit'):
        campaign['stop_reason']=None  # Explicit invocation grants a new bounded budget.
    if campaign['stop_reason']:
        print(json.dumps(campaign,indent=2));return
    # Reconcile any completed/recovered attempt whose result was not yet
    # checkpointed when the coordinator was interrupted.
    observed={row['candidate'] for row in campaign['observations']}
    for index,cid in enumerate(campaign['attempts']):
        if cid in observed:continue
        metrics_path=EXPERIMENTS/cid/'metrics.json'
        solution_path=folder/f'{index:03d}-solution.json'
        if metrics_path.exists() and solution_path.exists():
            metrics=json.loads(metrics_path.read_text())
            metrics['candidate']=cid
            parent=reg['candidates'][cid]['parent']
            update_feedback(campaign,metrics,reg['candidates'][parent]['score'],
                            json.loads(solution_path.read_text()))
    parent_dir=EXPERIMENTS/campaign['parent']/'inputs'
    hardware=json.loads((parent_dir/'hardware.json').read_text())
    compiler=parent_dir/'compiler.py'
    # Exclude historical executable plans even when the P1 component differed.
    for cid in reg['candidates']:
        p=EXPERIMENTS/cid/'inputs'
        if not (p/'recipe.json').exists() or not (p/'hardware.json').exists():continue
        if json.loads((p/'hardware.json').read_text())!=hardware:continue
        plan=projection_plan(json.loads((p/'recipe.json').read_text()))
        if plan and plan not in campaign['excluded']:campaign['excluded'].append(plan)
        meta=reg['candidates'][cid];parent=meta.get('parent')
        learned={e['candidate'] for e in campaign.get('calibration_evidence',[])}
        if cid not in learned and meta.get('eligible') is True and parent:
            pm=reg['candidates'].get(parent,{})
            pp=EXPERIMENTS/parent/'inputs'
            if (meta.get('cycles',{}).get('M2_D1') and pm.get('cycles',{}).get('M2_D1')
                    and (pp/'recipe.json').exists()
                    and json.loads((pp/'hardware.json').read_text())==hardware):
                learn_projection_delta(campaign,json.loads((pp/'recipe.json').read_text()),
                    json.loads((p/'recipe.json').read_text()),pm['cycles']['M2_D1'],
                    meta['cycles']['M2_D1'],hardware,cid)
    budget=Budget(args.max_grades,args.wall_seconds,audit_path=folder/f'launches-{time.time_ns()}.jsonl')
    def save():
        campaign['best']=reg['head'];campaign['last_budget']=budget.summary()
        temporary=checkpoint.with_suffix('.tmp');temporary.write_text(json.dumps(campaign,indent=2)+'\n');temporary.replace(checkpoint)
        engine.write_search_state(reg)
    save()
    for _ in range(args.max_grades*3):
        if not budget.available() or campaign['stop_reason']:break
        index=len(campaign['attempts'])
        request=folder/f'{index:03d}-request.json';answer=folder/f'{index:03d}-solution.json'
        request.write_text(json.dumps(dict(hardware=hardware,fixed_events=campaign['fixed_events'],
                           event_cap=campaign['event_cap'],excluded=campaign['excluded'],
                           corrections=campaign.get('corrections',{}),seconds=10),indent=2)+'\n')
        result=run_bounded([args.solver_python,'-m','project.search.milp_solver',str(request),str(answer)],
                           REPO,folder/f'{index:03d}-solver',timeout_seconds=30,memory_mib=512)
        if result['exit_code']!=0:
            campaign['stop_reason']='solver_process_failed';save();break
        solution=json.loads(answer.read_text())
        if solution['status']!='optimal_for_proxy_model':
            campaign['stop_reason']=solution['status'];save();break
        plan={s:{k:o[k] for k in FIELDS} for s,o in solution['selected'].items()}
        recipe=json.loads((EXPERIMENTS/reg['head']/'inputs/recipe.json').read_text())
        recipe['d1_schedule']['projections']=plan
        cid=state.next_id(reg);before=reg['candidates'][reg['head']]['score']
        proposal=dict(parent=reg['head'],recipe=recipe,hardware=hardware,compiler_path=str(compiler),
                      fingerprint=dsl.fingerprint(recipe,hardware,compiler.read_text()),
                      hypothesis='Automatic MILP D1 projection proposal with measured feedback and event-budget trust region.',
                      ops=[],family='persistent_decode',bias='d1',kind='milp',
                      struct_sig='d1-milp-projections',solver_evidence=str(answer.relative_to(REPO)))
        candidate=engine._materialize(reg,proposal,reg['search']['round'],cid)
        state.save(reg)
        campaign['attempts'].append(cid);campaign['excluded'].append(plan);save()
        metrics=evaluate_many(reg,[candidate],[7],budget,jobs=1,timeout_seconds=600,
                              memory_mib=args.memory_mib)[0]
        row=update_feedback(campaign,metrics,before,solution)
        if row.get('cycles'):
            parent_recipe=json.loads((EXPERIMENTS/candidate.parent/'inputs/recipe.json').read_text())
            learn_projection_delta(campaign,parent_recipe,recipe,
                reg['candidates'][candidate.parent]['cycles']['M2_D1'],row['cycles']['M2_D1'],hardware,cid)
        save()
        print(json.dumps(row),flush=True)
    if not campaign['stop_reason']:
        campaign['stop_reason']='budget_exhausted' if not budget.available() else 'proposal_limit'
    # Only verified best artifacts may become the default submission.
    from .controller import _stage_candidate
    _stage_candidate(Candidate.load(reg['head']))
    save()
    print(json.dumps({'folder':str(folder),'best':reg['head'],'stop_reason':campaign['stop_reason'],
                      'budget':budget.summary()},indent=2),flush=True)


if __name__=='__main__':
    install_signal_handlers()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-grades',type=int,default=6)
    p.add_argument('--wall-seconds',type=float,default=3600)
    p.add_argument('--memory-mib',type=float,default=6144)
    p.add_argument('--solver-python',default='/tmp/transformer-milp-env/bin/python')
    p.add_argument('--resume')
    args=p.parse_args()
    if args.max_grades<1 or args.wall_seconds<=0 or args.memory_mib<=0:p.error('positive budgets required')
    with writer_lock():run(args)
