import json
import unittest
from pathlib import Path

from codesign.challenge.isa import iter_parse
from project import compiler


ROOT = Path(__file__).resolve().parents[3]


class P1OperandPipelineTests(unittest.TestCase):
    def setUp(self):
        self.parent = ROOT / "project/experiments/cand-118"
        self.recipe = json.loads((self.parent / "inputs/recipe.json").read_text())

    def test_opt_in_does_not_change_incumbent(self):
        p1, _, _ = compiler.generate_m1_p1(recipe=self.recipe)
        d1, _, _ = compiler.generate_m1_d1(recipe=self.recipe)
        self.assertEqual(p1, (self.parent / "programs/M1_P1.asm").read_text())
        self.assertEqual(d1, (self.parent / "programs/M2_D1.asm").read_text())

    def test_qkv_pipeline_is_legal_and_d1_unchanged(self):
        self.recipe["gemm_stages"]["p1_prompt"]["operand_pipeline"] = ["qkv"]
        p1, _, _ = compiler.generate_m1_p1(recipe=self.recipe)
        d1, _, _ = compiler.generate_m1_d1(recipe=self.recipe)
        instructions = tuple(iter_parse(p1))
        self.assertTrue(any(i.op == "MMA.ACC" for i in instructions))
        self.assertIn('"var":"pb0l0qkvseedm"', p1)
        self.assertIn('"var":"pb0l0qkvnexta"', p1)
        self.assertEqual(d1, (self.parent / "programs/M2_D1.asm").read_text())

    def test_attention_panel_pipeline_is_legal_and_d1_unchanged(self):
        self.recipe["attention"]["algorithm"] = "blocked_prompt_pipeline"
        p1, _, _ = compiler.generate_m1_p1(recipe=self.recipe)
        d1, _, _ = compiler.generate_m1_d1(recipe=self.recipe)
        tuple(iter_parse(p1))
        self.assertIn('nextkey', p1)
        self.assertIn('nextvalue', p1)
        self.assertEqual(d1, (self.parent / "programs/M2_D1.asm").read_text())

    def test_shared_resident_mlp_uses_exact_capacity_and_keeps_d1(self):
        self.recipe["fusion"]["p1_mlp"] = "sh_resident"
        p1, _, _ = compiler.generate_m1_p1(recipe=self.recipe)
        d1, _, _ = compiler.generate_m1_d1(recipe=self.recipe)
        instructions = tuple(iter_parse(p1))
        begin = next(i for i in instructions if i.op == "WG.BEGIN")
        self.assertEqual(begin.args["shared_bytes"], 128 * 1024)
        self.assertTrue(any(i.op == "ST" and i.args["dst"]["space"] == "SH"
                            for i in instructions))
        self.assertEqual(d1, (self.parent / "programs/M2_D1.asm").read_text())

    def test_unsupported_shape_fails_before_emission(self):
        model = compiler.MODELS["M1"]
        layout = compiler.build_layout(model, "P1")
        recipe = json.loads(json.dumps(self.recipe))
        recipe["gemm_stages"]["p1_prompt"]["operand_pipeline"] = ["qkv"]
        builder = compiler.Builder(layout, recipe)
        builder.set_gemm_stage("p1_prompt")
        with self.assertRaisesRegex(ValueError, "requires M64"):
            builder.gemm(compiler.Tensor(0, (1, 256)), compiler.Tensor(0, (256, 768)),
                         compiler.Tensor(0, (1, 768)), 1, 256, 768, "testqkv")


if __name__ == "__main__":
    unittest.main()
