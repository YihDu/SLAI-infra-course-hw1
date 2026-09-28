import json
import tempfile
import unittest
from pathlib import Path
from project.search import state
from project.search.milp_campaign import update_feedback, projection_plan, learn_projection_delta


class MilpLoopTests(unittest.TestCase):
    def test_stale_registry_cannot_erase_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'registry.json';initial={'candidates':{}}
            state.save(initial,p)
            stale=state.load(p);fresh=state.load(p)
            fresh['candidates']['new']={'status':'created'};state.save(fresh,p)
            with self.assertRaisesRegex(RuntimeError,'stale registry'):
                state.save(stale,p)
            self.assertIn('new',state.load(p)['candidates'])
            state.save(fresh,p)  # Same live owner may continue saving.

    def test_memory_failure_does_not_count_as_low_gain(self):
        c={'event_cap':15500,'flat_valid':1,'observations':[]}
        update_feedback(c,{'candidate':'x','failure_class':'memory_limit'},100,{'objective':1,'events':15376})
        self.assertEqual(c['flat_valid'],1);self.assertEqual(c['event_cap'],15375)
        self.assertNotIn('score',c['observations'][0])

    def test_three_measured_flat_candidates_request_outer_work(self):
        c={'event_cap':15500,'flat_valid':0,'observations':[]}
        for _ in range(3):
            update_feedback(c,{'candidate':'x','reports':{'grade':{'eligible':True,'score':101,'cases':{}}}},100,{'objective':1,'events':15000})
        self.assertEqual(c['stop_reason'],'outer_structure_required')

    def test_only_controlled_projection_delta_calibrates(self):
        root=Path(__file__).resolve().parents[3]/'project/experiments'
        left=json.loads((root/'cand-114/inputs/recipe.json').read_text())
        right=json.loads((root/'cand-115/inputs/recipe.json').read_text())
        hardware=json.loads((root/'cand-114/inputs/hardware.json').read_text())
        c={}
        evidence=learn_projection_delta(c,left,right,212659,210849,hardware,'test')
        self.assertEqual(evidence['stage'],'w1')
        self.assertAlmostEqual(c['corrections']['w1/4:128:8'],-203.125)
        right['d1_schedule']['projections']['wo']['n_tile']=16
        self.assertIsNone(learn_projection_delta({},left,right,212659,210849,hardware,'two_changes'))

    def test_historical_decode_plan_is_excluded_correctly(self):
        self.assertIsNone(projection_plan({'d1_schedule':{'kind':'column_head'}}))
        p=projection_plan({'d1_schedule':{'kind':'column_head','partition':'panel_owners'}})
        self.assertEqual(p['wo']['owners'],2);self.assertEqual(p['qkv']['owners'],3)

if __name__=='__main__':unittest.main()
