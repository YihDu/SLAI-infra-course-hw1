"""Run in the optional HiGHS environment, never the frozen evaluator env."""
import itertools
import json
import unittest
from pathlib import Path
import importlib.util
from project.search.milp_solver import solve, options_for, STAGES
HAS_HIGHS = importlib.util.find_spec('highspy') is not None
from codesign.challenge.hardware import Hardware

ROOT=Path(__file__).resolve().parents[3]


@unittest.skipUnless(HAS_HIGHS, "optional HiGHS environment required")
class SolverTests(unittest.TestCase):
    def setUp(self):
        self.hardware=json.loads((ROOT/'project/experiments/cand-114/inputs/hardware.json').read_text())
        recipe=json.loads((ROOT/'project/experiments/cand-114/inputs/recipe.json').read_text())
        self.known=recipe['d1_schedule']['projections']

    def test_excluded_solution_and_exact_finite_optimum(self):
        request=dict(hardware=self.hardware,fixed_events=5264,event_cap=15500,excluded=[self.known])
        result=solve(request)
        selected={s:{k:o[k] for k in ('owners','n_tile','k_chunk')} for s,o in result['selected'].items()}
        self.assertNotEqual(selected,self.known)
        options=options_for(Hardware.from_dict(self.hardware))
        best=float('inf')
        for combo in itertools.product(*options.values()):
            if 5264+sum(o['events'] for o in combo)>15500:continue
            if all(all(o[k]==self.known[s][k] for k in ('owners','n_tile','k_chunk')) for s,o in zip(STAGES,combo)):continue
            best=min(best,sum(16*o['cost_proxy'] for o in combo))
        self.assertEqual(result['objective'],best)
        self.assertLessEqual(result['events'],15500)

    def test_impossible_budget_does_not_emit_candidate(self):
        self.assertEqual(solve(dict(hardware=self.hardware,fixed_events=5264,event_cap=1))['status'],'exhausted')

if __name__=='__main__':unittest.main()
