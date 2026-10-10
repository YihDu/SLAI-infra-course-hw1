"""Exercise actual generation/parser/cost subprocesses; grade must stay unlaunched."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from project.search import evaluator, runtime


def program(groups=2, repeat=3072):
    lines=[]
    for g in range(groups):
        lines.append('WG.BEGIN '+json.dumps(dict(wg=f'g{g}',sm=g,shared_bytes=0)))
    for phase in range(2):
        for g in range(groups):
            rf=dict(space='RF',wg=f'g{g}',lane=0,offset=0,count=1)
            lines.append('VEC '+json.dumps(dict(kind='add',src=[{'imm':0},{'imm':0}],dst=rf,event=f'init{phase}_{g}')))
            lines.append('FOR '+json.dumps(dict(var='i',start=0,stop=repeat,step=1)))
            lines.append('VEC '+json.dumps(dict(kind='add',src=[rf,{'imm':1}],dst=rf,event=f'e{phase}_{g}_'+'{i}')))
            lines.append('END.FOR {}')
        lines.append('BARRIER '+json.dumps(dict(wgs=[f'g{g}' for g in range(groups)],events=[])))
    for g in range(groups):lines.append('WG.END '+json.dumps(dict(wg=f'g{g}')))
    return '\n'.join(lines)+'\n'


class ValidationPreflightTests(unittest.TestCase):
    def test_actual_preflight_rejects_before_any_grade_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=root/'inputs';reports=root/'reports';programs=root/'programs'
            for folder in (inputs,reports,programs):folder.mkdir()
            (inputs/'compiler.py').write_text(
                'from pathlib import Path\n'
                f"Path('programs/M1_P1.asm').write_text({program()!r})\n"
                f"Path('programs/M2_D1.asm').write_text({program(1,1)!r})\n")
            cand=SimpleNamespace(id='preflight-test',dir=root,inputs=inputs,reports=reports,programs=programs)
            launch=Mock()
            with patch.object(runtime,'run',wraps=runtime.run) as observed:
                result=evaluator.evaluate(cand,[7],log=lambda _:None,memory_mib=256,
                                          timeout_seconds=60,before_launch=launch)
            self.assertEqual(result['failure_class'],'validation_cost_pruned',result)
            self.assertEqual(result['evaluation_attempts'],0)
            launch.assert_not_called()
            self.assertEqual(observed.call_count,3)  # generate, parse/static, cost model
            self.assertFalse((reports/'grade.json').exists())
            # Reusing prepared inputs also cannot silently drop/ignore a rejected cost report.
            with patch.object(runtime,'run') as forbidden:
                result=evaluator.evaluate(cand,[7],prepared=True,before_launch=launch)
            forbidden.assert_not_called();launch.assert_not_called()
            self.assertEqual(result['verdict'],'pruned')
            self.assertIn('validation_cost',result)
            self.assertIn('validation_cost_progress',result)
            self.assertEqual(json.loads((root/'metrics.json').read_text())['evaluation_attempts'],0)

    def test_explicit_calibration_is_preserved_in_prepared_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reports=root/'reports';reports.mkdir()
            (reports/'validation-cost.json').write_text(json.dumps({'M1_P1':{'decision':'prune_validation_cost'}}))
            (root/'proposal.json').write_text(json.dumps({'validation_cost_calibration':True,'calibration_reason':'bounded diagnostic'}))
            cand=SimpleNamespace(id='calibration-test',dir=root,reports=reports,inputs=root,programs=root)
            def exhausted(_):raise RuntimeError('budget_exhausted_for_test')
            with patch.object(runtime,'run') as forbidden:
                result=evaluator.evaluate(cand,[7],prepared=True,before_launch=exhausted)
            forbidden.assert_not_called()
            self.assertEqual(result['validation_cost_override'],'bounded diagnostic')
            self.assertEqual(result['failure_class'],'budget_exhausted')
            self.assertEqual(result['evaluation_attempts'],0)

if __name__=='__main__':unittest.main()
