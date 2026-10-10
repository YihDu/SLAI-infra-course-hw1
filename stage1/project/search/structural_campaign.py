"""Expand RF dataflow, measure controls, then compose software/hardware winners."""
import argparse
import json
import time
from pathlib import Path

from . import state, engine, dsl, propose
from .candidate import REPO
from .campaign import make_candidate, restore_best
from .coordinator import Budget, evaluate_many, recover, writer_lock


def run(max_evals=8, jobs=2):
    folder = REPO / 'project/campaigns' / time.strftime('structural-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    compiler = REPO / 'project/compiler.py'
    compiler_source = compiler.read_bytes()
    reg = state.load()
    recover(reg)
    budget = Budget(max_evals, 3600, audit_path=folder / 'launches.jsonl')
    summary = {'baseline': 'cand-083', 'candidates': [], 'pruned': []}

    def create(parent, ops, hypothesis):
        r, h = propose._load_parent_inputs(parent)
        r, _ = dsl.apply_ops(r, h, ops, phase=2)
        cost = dsl.evaluation_cost_reasons(r)
        if cost:
            summary["pruned"].append({"parent":parent,"ops":ops,"reasons":cost})
            return None
        c, errors = make_candidate(reg, parent, ops, hypothesis, compiler)
        if errors:
            summary['pruned'].append({'parent': parent, 'ops': ops, 'reasons': errors})
        elif c:
            summary['candidates'].append(c.id)
        return c

    def evaluate(candidates):
        candidates = [c for c in candidates if c]
        if candidates and budget.available():
            evaluate_many(reg, candidates, [7], budget, jobs=jobs,
                          timeout_seconds=600, memory_mib=3072)
        summary['budget'] = budget.summary()
        summary['best'] = reg['head']
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    try:
        pilots = []
        for m, n, k in [(16,32,32),(16,64,16),(8,128,8),(8,64,16)]:
            pilots.append(create('cand-083', [{'op':'stage_gemm_tiles','stage':'p1_prompt',
                'order':'nkm_lanes','m_tile':m,'n_tile':n,'k_chunk':k}],
                f'RF accumulator partitions: P1 prompt M={m}, N={n}, K={k}; preserve B reuse'))
        evaluate(pilots)
        software_best = reg['head']
        bridge = create('cand-083', [{'op':'hw_set','path':'sm_count','value':4}],
                        'Area bridge: remove four idle SMs before widening TC')
        evaluate([bridge])
        if bridge and reg['candidates'][bridge.id].get('score'):
            wide = create(bridge.id, [{'op':'hw_set','path':'tc_array','value':'8x16'}],
                          'Area bridge successor: wider TC with four SMs')
            evaluate([wide])
            if wide and reg['candidates'][wide.id].get('score') and software_best != 'cand-083':
                compose = create(software_best, [{'op':'hw_set','path':'sm_count','value':4},
                    {'op':'hw_set','path':'tc_array','value':'8x16'}],
                    'Compose best RF dataflow with measured area bridge')
                evaluate([compose])
        if budget.available():
            parent = reg['head']
            c = create(parent, [{'op':'stage_gemm_tiles','stage':'d1','n_tile':64,'k_chunk':16}],
                       'Expand decode N tile after prompt/hardware controls')
            evaluate([c])
    finally:
        restore_best(reg)
        compiler.write_bytes(compiler_source)
        summary['budget'] = budget.summary()
        summary['best'] = reg['head']
        summary['results'] = {cid: reg['candidates'][cid] for cid in summary['candidates']}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        engine.write_search_state(reg)
        print(json.dumps({'folder':str(folder),'best':reg['head'],'budget':budget.summary()},indent=2),flush=True)


def refine(max_evals=3, jobs=2):
    """Refine measured survivors, including a joint RF-capacity/hardware expansion."""
    reg = state.load()
    recover(reg)
    parent = reg['head']
    folder = REPO / 'project/campaigns' / time.strftime('structural-refine-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    compiler = REPO / 'project/compiler.py'
    source = compiler.read_bytes()
    budget = Budget(min(max_evals, 3), 1800, audit_path=folder / 'launches.jsonl')
    candidates, pruned = [], []
    plans = [
        ([{'op':'stage_gemm_tiles','stage':'p1_prompt','order':'nkm_lanes',
           'm_tile':32,'n_tile':32,'k_chunk':32}], 'Larger M tile reduces repeated B reads'),
        ([{'op':'stage_gemm_tiles','stage':'p1_prompt','order':'nkm_lanes',
           'm_tile':64,'n_tile':16,'k_chunk':16}], 'Full-M tile trades N reuse for B reuse'),
        ([{'op':'hw_set','path':'vector_lanes','value':8},
          {'op':'stage_gemm_tiles','stage':'p1_prompt','order':'nkm_lanes',
           'm_tile':16,'n_tile':128,'k_chunk':16}],
         'Joint RF expansion: 8 vector partitions allow N128 with K16, avoiding scalar checker'),
    ]
    try:
        for ops, hypothesis in plans[:budget.maximum]:
            c, errors = make_candidate(reg, parent, ops, hypothesis, compiler)
            if c: candidates.append(c)
            else: pruned.append({'ops':ops,'reasons':errors})
        evaluate_many(reg, candidates, [7], budget, jobs=jobs, timeout_seconds=600, memory_mib=3072)
    finally:
        restore_best(reg)
        compiler.write_bytes(source)
        summary = {'parent':parent,'best':reg['head'],'budget':budget.summary(),'pruned':pruned,
                   'results':{c.id:reg['candidates'][c.id] for c in candidates}}
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'folder':str(folder),'best':reg['head'],'budget':budget.summary()},indent=2),flush=True)


if __name__ == '__main__':
    from .runtime import install_signal_handlers
    install_signal_handlers()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-evals', type=int, default=8)
    p.add_argument('--refine', action='store_true')
    p.add_argument('--jobs', type=int, choices=(1,2), default=2)
    args = p.parse_args()
    if not 1 <= args.max_evals <= 8:
        p.error('max-evals must be 1..8')
    with writer_lock():
        (refine if args.refine else run)(args.max_evals,args.jobs)
