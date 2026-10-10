"""Guard against ineffective OOM pruning and hardware/tile plateau resets."""
import copy
import json
import sys
import unittest
from project.search import dsl
from project.search.hierarchy import outer_signature, blocked_structures
from project.search.validation_cost import analyze
from codesign.challenge.hbm_race import validate_hbm_races


def toy_program(n=24, groups=2):
    lines=[f'WG.BEGIN {{"wg":"g{i}","sm":{i},"shared_bytes":0}}' for i in range(groups)]
    for phase in range(2):
        for i in range(groups):
            rf=dict(space='RF',wg=f'g{i}',lane=0,offset=0,count=1)
            for k in range(n):
                src=[{'imm':0},{'imm':1}] if phase==0 and k==0 else [rf,{'imm':1}]
                lines.append('VEC '+json.dumps(dict(kind='add',src=src,dst=rf,event=f'e{phase}_{i}_{k}')))
        lines.append('BARRIER '+json.dumps(dict(wgs=[f'g{i}' for i in range(groups)],events=[])))
    lines += [f'WG.END {{"wg":"g{i}"}}' for i in range(groups)]
    return '\n'.join(lines)+'\n'


class ValidationCostTests(unittest.TestCase):
    def test_bound_against_actual_frozen_checker(self):
        source=toy_program();observed={}
        def observe(frame,event,arg):
            if frame.f_code.co_name=='validate_hbm_races' and event=='return':
                knowledge=frame.f_locals['event_knowledge']
                observed['cardinality']=sum(len(k.events) for k in knowledge.values())
                observed['bytes']=sum(sys.getsizeof(k.events) for k in knowledge.values())
            return observe
        previous=sys.gettrace()
        try:
            sys.settrace(observe);validate_hbm_races(source)
        finally:sys.settrace(previous)
        forecast=analyze(source)
        self.assertLessEqual(forecast['sum_dependency_depth_lower_bound'],observed['cardinality'])
        self.assertLessEqual(forecast['retained_event_sets_logical_bytes_lower_bound'],observed['bytes'])
        self.assertEqual(forecast['events'],96)
        self.assertEqual(forecast['barrier_directed_pair_relations'],4)

    def test_single_wg_uses_real_fast_path(self):
        self.assertEqual(analyze(toy_program(groups=1))['retained_event_sets_logical_bytes_lower_bound'],0)

    def test_longer_dependency_chains_grow_superlinearly(self):
        a=analyze(toy_program(16));b=analyze(toy_program(32))
        self.assertGreater(b['retained_event_sets_logical_bytes_lower_bound'],3*a['retained_event_sets_logical_bytes_lower_bound'])


class HierarchyTests(unittest.TestCase):
    def test_tiles_and_placement_cannot_reset_structure_budget(self):
        r=copy.deepcopy(dsl.DEFAULT_RECIPE)
        r['p1_schedule']=dict(kind='incremental_gemm',gemms=['w1'],shards=4,sm_placement=[0,1,2,3])
        sig=outer_signature(r)
        r['gemm']['n_tile']=64;r['p1_schedule']['shards']=2;r['p1_schedule']['sm_placement']=[2,3]
        self.assertEqual(outer_signature(r),sig)
        r['p1_schedule']['gemms']=['w1','w2']
        self.assertNotEqual(outer_signature(r),sig)

    def test_three_small_measured_gains_block_but_memory_failure_does_not(self):
        reg={'search':{'hierarchy':{'enabled':True,'start_after_id':100}},'candidates':{
            'cand-098':{'score':1000},
            'cand-101':{'score':1050,'eligible':True,'parent':'cand-098','outer_sig':'a'},
            'cand-102':{'score':1060,'eligible':True,'parent':'cand-098','outer_sig':'a'},
            'cand-103':{'score':None,'eligible':False,'failure_class':'memory_limit','outer_sig':'a'},
        }}
        self.assertEqual(blocked_structures(reg),{})
        reg['candidates']['cand-104']={'score':1090,'eligible':True,'parent':'cand-098','outer_sig':'a'}
        self.assertIn('a',blocked_structures(reg))
        reg['candidates']['cand-105']={'score':1200,'eligible':True,'parent':'cand-098','outer_sig':'a'}
        self.assertNotIn('a',blocked_structures(reg))

if __name__=='__main__':unittest.main()
