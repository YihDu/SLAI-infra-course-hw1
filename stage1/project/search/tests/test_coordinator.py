import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from project.search import cache, coordinator, evaluator, records, state, archives

class CoordinatorTests(unittest.TestCase):
    def test_isolated_parallel_duplicate_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp)
            cands=[]
            for i in range(4):
                p=root/str(i);p.mkdir();(p/'reports').mkdir()
                cands.append(SimpleNamespace(id=str(i),dir=p,reports=p/'reports',status='created',save=lambda:None))
            reg={'candidates':{c.id:{} for c in cands}}
            saved={}; recorded=[]; active=0; maximum=0; lock=threading.Lock()
            def fake_eval(cand,seeds,**kw):
                nonlocal active,maximum
                kw['before_launch']('grade')
                with lock:
                    active+=1; maximum=max(active,maximum)
                time.sleep(.05)
                with lock: active-=1
                return {'candidate':cand.id,'verdict':'evaluated','evaluation_attempts':1,'reports':{}}
            stack.enter_context(patch.object(state,'save'))
            stack.enter_context(patch.object(records,'record_result',side_effect=lambda r,c,m:recorded.append((c.id,m))))
            stack.enter_context(patch.object(evaluator,'prepare',return_value={}))
            stack.enter_context(patch.object(evaluator,'evaluate',side_effect=fake_eval))
            stack.enter_context(patch.object(cache,'index_history'))
            stack.enter_context(patch.object(cache,'execution_key',side_effect=lambda c,*a:'same' if c.id in ('0','2') else c.id))
            stack.enter_context(patch.object(cache,'lookup',side_effect=lambda k:saved.get(k)))
            stack.enter_context(patch.object(cache,'store',side_effect=lambda k,c,m:saved.update({k:m})))
            budget=coordinator.Budget(2)
            result=coordinator.evaluate_many(reg,cands,[7],budget,jobs=2)
            self.assertEqual(maximum,2)
            self.assertEqual(budget.used,2)
            self.assertTrue(result[2]['cache_hit'])
            self.assertEqual(result[2]['cache_source'],'0')
            self.assertEqual(result[3]['failure_class'],'budget_exhausted')
            self.assertEqual([cid for cid,_ in recorded],['0','1','2','3'])
    def test_slow_first_does_not_block_refill_or_change_publication_order(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            candidates = []
            for i in range(4):
                folder = root / str(i)
                folder.mkdir(); (folder / 'reports').mkdir()
                candidates.append(SimpleNamespace(id=str(i), dir=folder,
                    reports=folder/'reports', status='created', save=lambda:None))
            reg = {'candidates':{c.id:{} for c in candidates}}
            replacement_started = threading.Event()
            observed = []
            published = []
            launches = []

            def evaluate(cand, seeds, **kw):
                kw['before_launch']('grade')
                launches.append(cand.id)
                if cand.id == '0':
                    observed.append(replacement_started.wait(3))
                if cand.id == '2':
                    replacement_started.set()
                return {'candidate':cand.id, 'verdict':'evaluated',
                        'evaluation_attempts':1, 'reports':{}}

            stack.enter_context(patch.object(state, 'save'))
            stack.enter_context(patch.object(records, 'record_result',
                side_effect=lambda r,c,m:published.append(c.id)))
            stack.enter_context(patch.object(evaluator, 'prepare', return_value={}))
            stack.enter_context(patch.object(evaluator, 'evaluate', side_effect=evaluate))
            stack.enter_context(patch.object(cache, 'index_history'))
            stack.enter_context(patch.object(cache, 'execution_key',
                side_effect=lambda c,*a:'shared' if c.id in ('0','3') else c.id))
            stack.enter_context(patch.object(cache, 'lookup', return_value=None))
            stack.enter_context(patch.object(cache, 'store'))
            budget = coordinator.Budget(3)
            results = coordinator.evaluate_many(reg, candidates, [7], budget, jobs=2)
            self.assertEqual(observed, [True], 'Replacement must start before slow first exits')
            self.assertEqual(published, ['0','1','2','3'])
            self.assertEqual(set(launches), {'0','1','2'})
            self.assertEqual(budget.used, 3)
            self.assertTrue(results[3]['cache_hit'])
            self.assertEqual(results[3]['cache_source'], '0')

    def test_incomplete_success_cannot_enter_archives(self):
        for flags in ({'eligible':True}, {'eligible':True,'functional':{'M1_P1':True}}):
            self.assertFalse(archives.is_admissible({'score':9999,'status':'evaluated',**flags}))

if __name__=='__main__':unittest.main()
