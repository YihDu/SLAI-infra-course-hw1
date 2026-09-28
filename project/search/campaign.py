"""Bounded repair-validation campaign: reproduction, hardware controls, adaptation."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

from . import dsl, engine, propose, state
from .candidate import Candidate, EXPERIMENTS, REPO, WORKTREE_PATHS
from .coordinator import Budget, evaluate_many, recover, writer_lock

HARDWARE_CONTROLS = [('tc_k_parallel', 2), ('tc_k_parallel', 4),
                     ('tc_array', '8x8'), ('tc_array', '8x16'),
                     ('tc_count', 4), ('dma_engines', 2), ('cache_mib', 4)]


def make_candidate(reg, parent, ops, hypothesis, compiler_path=None):
    recipe, hw = propose._load_parent_inputs(parent)
    recipe, hw = dsl.apply_ops(recipe, hw, ops, phase=2)
    errors = dsl.validate_recipe(recipe, hw) + dsl.validate_hardware(hw)
    if errors:
        return None, errors
    prop = {'parent': parent, 'ops': ops, 'hypothesis': hypothesis,
            'family': 'hardware' if ops and ops[0]['op'] == 'hw_set' else 'compute',
            'bias': 'balanced', 'kind': 'manual', 'recipe': recipe, 'hardware': hw,
            'fingerprint': dsl.fingerprint(recipe, hw),
            'struct_sig': propose.structure_signature(recipe, hw),
            'compiler_path': str(compiler_path or EXPERIMENTS / parent / 'inputs/compiler.py')}
    cand = engine._materialize(reg, prop, reg['search'].get('round', 12) + 1, state.next_id(reg))
    state.save(reg)
    return cand, []


def restore_best(reg):
    cand = Candidate.load(reg['head'])
    cand.stage_to_worktree()
    for name in ('M1_P1.asm', 'M2_D1.asm'):
        shutil.copy2(cand.programs / name, REPO / 'programs' / name)
    reg['search']['bottleneck'] = engine._bottleneck(reg, cand.id)
    state.save(reg)
    engine.write_search_state(reg)


def run(max_evals=12, jobs=2, timeout_seconds=600, memory_mib=3072, wall_seconds=7200):
    if not 1 <= max_evals <= 12:
        raise ValueError('repair campaign budget must be 1..12')
    folder = REPO / 'project' / 'campaigns' / time.strftime('repair-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    budget = Budget(max_evals, wall_seconds, audit_path=folder / 'launches.jsonl')
    reg = state.load(); recover(reg)
    summary = {'baseline': 'cand-073', 'candidates': [], 'skipped': [], 'budget': {}}
    def save():
        summary['budget'] = budget.summary(); summary['best'] = reg['head']
        summary['best_score'] = reg['candidates'][reg['head']].get('score')
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def evaluate(cands, force=False, parallel=jobs):
        summary['candidates'].extend(c.id for c in cands); save()
        values = evaluate_many(reg, cands, [7], budget, jobs=parallel, force=force,
                               timeout_seconds=timeout_seconds, memory_mib=memory_mib)
        save(); return values
    try:
        cand, errors = make_candidate(reg, 'cand-073', [], 'Reproduce cand-073 with isolated single-grade runner')
        result = evaluate([cand], force=True, parallel=1)[0]
        expected = json.loads((EXPERIMENTS / 'cand-073/metrics.json').read_text())['reports']['grade']
        actual = result.get('reports', {}).get('grade', {})
        if (result.get('verdict') != 'evaluated' or actual.get('eligible') is not True
                or actual.get('score') != expected['score']
                or any(actual.get('cases', {}).get(k, {}).get('cycles') != v['cycles']
                       for k, v in expected['cases'].items())
                or any((cand.programs / n).read_bytes() != (EXPERIMENTS / 'cand-073/programs' / n).read_bytes()
                       for n in ('M1_P1.asm', 'M2_D1.asm'))):
            summary['stopped'] = 'baseline_reproduction_failed'; return summary
        if not budget.available():
            return summary
        controls = []
        for field, value in HARDWARE_CONTROLS:
            c, errors = make_candidate(reg, 'cand-073', [{'op': 'hw_set', 'path': field, 'value': value}],
                                       f'Hardware sensitivity: {field}={value} against cand-073')
            if c: controls.append(c)
            else: summary['skipped'].append({'field': field, 'value': value, 'reasons': errors})
        if budget.available(): evaluate(controls)
        # Compose at most two independently beneficial hardware changes, one field at a time.
        useful = []
        for c in controls:
            m = reg['candidates'][c.id]
            if m.get('score') and m['score'] > expected['score']:
                useful.append(c)
        useful.sort(key=lambda c: (-reg['candidates'][c.id]['score'], c.id))
        composed = 0
        for c in useful:
            if composed >= 2 or not budget.available(): break
            op = c.ops[0]; _, hw = propose._load_parent_inputs(reg['head'])
            if hw[op['path']] == op['value']: continue
            child, errors = make_candidate(reg, reg['head'], [op], f'Compose beneficial control {c.id}')
            if child: evaluate([child]); composed += 1
        # One effective stage-neighbor on the workload with smaller baseline speedup.
        if budget.available():
            parent = reg['head']; recipe, hw = propose._load_parent_inputs(parent)
            cycles = reg['candidates'][parent]['cycles']
            stage = 'p1_prompt' if 30996995 / cycles['M1_P1'] < 1302032 / cycles['M2_D1'] else 'd1'
            effective = {**recipe['gemm'], **recipe.get('gemm_stages', {}).get(stage, {})}
            for key, choices in [('k_chunk', dsl.K_CHUNK_CHOICES), ('n_tile', dsl.N_TILE_CHOICES), ('m_tile', dsl.M_TILE_CHOICES)]:
                trial = None
                for value in sorted(choices, key=lambda v: (abs(v - effective[key]), v)):
                    if value == effective[key]: continue
                    op = {'op': 'stage_gemm_tiles', 'stage': stage, key: value}
                    rr, hh = dsl.apply_ops(recipe, hw, [op], phase=2)
                    if dsl.fingerprint(rr, hh) in reg.get('fingerprints', {}): continue
                    trial, errors = make_candidate(reg, parent, [op], f'Stage sensitivity: {stage} {key}={value}')
                    if trial: break
                if trial: evaluate([trial]); break
        # One archive expansion in a mechanism not selected by the first seven controls.
        if budget.available():
            pool = [c for c in reg.get('archives', {}).get('pareto', []) if c != reg['head']]
            pool += [reg['head']]
            selected = None
            for parent in pool:
                recipe, hw = propose._load_parent_inputs(parent)
                for field, value in [('reduction_units', 1), ('hbm_channels', 8), ('sfu_lanes', 16)]:
                    if hw.get(field) == value: continue
                    ops = [{'op': 'hw_set', 'path': field, 'value': value}]
                    try: rr, hh = dsl.apply_ops(recipe, hw, ops, phase=2)
                    except dsl.DslError: continue
                    if dsl.fingerprint(rr, hh) in reg.get('fingerprints', {}): continue
                    selected, errors = make_candidate(reg, parent, ops, f'Archive mechanism expansion: {field}={value}')
                    if selected: break
                if selected: break
            if selected: evaluate([selected])
        return summary
    finally:
        restore_best(reg); save()
        print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-evals', type=int, default=12)
    p.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    p.add_argument('--timeout-seconds', type=float, default=600)
    p.add_argument('--memory-mib', type=int, default=3072)
    p.add_argument('--wall-seconds', type=float, default=7200)
    a = p.parse_args()
    with writer_lock():
        run(a.max_evals, a.jobs, a.timeout_seconds, a.memory_mib, a.wall_seconds)

if __name__ == '__main__': main()
