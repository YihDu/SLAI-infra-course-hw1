"""Outer structures own inner tuning budgets; hardware/tile changes cannot reset them."""
import hashlib
import json
from functools import lru_cache
from .candidate import EXPERIMENTS


def outer_structure(recipe):
    schedule=recipe.get('p1_schedule',{})
    return {
        'parallel_skeleton': {k:schedule[k] for k in ('kind','gemms','phases') if k in schedule},
        'fusion_partition': recipe.get('fusion',{}),
        'residency_plan': {
            'scores':recipe.get('attention',{}).get('scores','hbm'),
            'gemm_order':recipe.get('gemm',{}).get('order','mnk'),
            'stage_orders':{k:v['order'] for k,v in recipe.get('gemm_stages',{}).items() if 'order' in v},
            'kernels':{k:v.get('kind') for k,v in recipe.get('gemm_kernels',{}).items()},
            'explicit':recipe.get('residency',{}),
        },
        'attention_algorithm':recipe.get('attention',{}).get('algorithm','causal_materialized_or_rf'),
        'persistent_scheduling':recipe.get('persistent',{}),
    }


def outer_signature(recipe):
    return hashlib.sha256(json.dumps(outer_structure(recipe),sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=4096)
def _saved_signature(cid):
    p=EXPERIMENTS/cid/'inputs/recipe.json'
    return outer_signature(json.loads(p.read_text())) if p.exists() else None


def blocked_structures(reg):
    policy=reg.get('search',{}).get('hierarchy',{})
    if not policy.get('enabled'):return {}
    minimum=float(policy.get('minimum_gain',.10));patience=int(policy.get('patience',3))
    start=int(policy.get('start_after_id',0));streaks={};blocked={}
    candidates=reg.get('candidates',{})
    for cid,m in candidates.items():
        if not cid.startswith('cand-') or int(cid.split('-')[1])<=start:continue
        # Failed/unmeasured candidates must never be treated as small performance gains.
        if m.get('eligible') is not True or not m.get('score'):continue
        parent=candidates.get(m.get('parent'),{});parent_score=parent.get('score')
        if not parent_score:continue
        sig=m.get('outer_sig') or _saved_signature(cid)
        if sig is None:continue
        gain=m['score']/parent_score-1
        streaks[sig]=0 if gain>=minimum else streaks.get(sig,0)+1
        if streaks[sig]>=patience:
            blocked[sig]={'streak':streaks[sig],'minimum_gain':minimum,'last_candidate':cid,
                          'action':'change_outer_structure'}
        else:blocked.pop(sig,None)
    return blocked
