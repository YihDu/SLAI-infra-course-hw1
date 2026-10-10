import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from project.search import runtime, evaluator
from project.search.coordinator import Budget

class RuntimeTests(unittest.TestCase):
    def test_timeout_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            result=runtime.run([sys.executable,'-c','import time; time.sleep(30)'],p,p/'timeout',.15)
            self.assertEqual(result['failure_class'],'timeout')
            with patch.object(runtime,'footprint',return_value=5*1024**2):
                result=runtime.run([sys.executable,'-c','import time; time.sleep(30)'],p,p/'memory',5,1)
            self.assertEqual(result['failure_class'],'memory_limit')
    def test_budget_parallel_failures_count(self):
        b=Budget(3)
        def claim(i):
            try: b.claim('grade'); return True
            except RuntimeError: return False
        with ThreadPoolExecutor(8) as pool:
            self.assertEqual(sum(pool.map(claim,range(20))),3)
        self.assertEqual(b.used,3)
    def test_isolated_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for part in ('inputs','programs','reports'): (p/part).mkdir()
            (p/'inputs/compiler.py').write_text("from pathlib import Path\nfor n in ('M1_P1','M2_D1'): Path('programs/'+n+'.asm').write_text('STEP.COMMIT {\\\"step\\\":0}\\n')")
            cand=SimpleNamespace(id='fake',dir=p,inputs=p/'inputs',programs=p/'programs',reports=p/'reports')
            out=evaluator.prepare(cand,log=lambda s:None)
            self.assertNotIn('failure_class',out)
            self.assertEqual(out['static']['M1_P1']['expanded_instructions'],1)
    def test_single_grade_and_full(self):
        for validation,modes in [('grade',['grade']),('full',['check','grade'])]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); (p/'reports').mkdir()
                cand=SimpleNamespace(id='fake',dir=p,inputs=p,programs=p,reports=p/'reports')
                called=[]
                def fake(cmd,cwd,prefix,*args):
                    mode=cmd[2]; called.append(mode)
                    report={'mode':mode,'cases':{k:{'functional_passed':True} for k in ('M1_P1','M2_D1')}}
                    if mode=='grade': report.update(experimental_score=1000,eligible=True)
                    Path(cmd[cmd.index('--report')+1]).write_text(json.dumps(report))
                    return {'exit_code':0,'failure_class':None}
                with patch.object(runtime,'run',side_effect=fake):
                    result=evaluator.evaluate(cand,[7],log=lambda s:None,validation=validation,prepared=True)
                self.assertEqual(called,modes)
                self.assertEqual(result['evaluation_attempts'],len(modes))
                self.assertEqual(result['verdict'],'evaluated')
