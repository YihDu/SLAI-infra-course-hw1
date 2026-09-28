"""Tests for search policy, DSL, and proposal generation."""

import copy
import random
import unittest

from project.search import dsl, propose
from project.search.candidate import Candidate


class TestHardwareMenuAndDependencies(unittest.TestCase):
    def setUp(self):
        self.base_hw = copy.deepcopy(dsl.BASELINE_HW)
        self.base_recipe = copy.deepcopy(dsl.DEFAULT_RECIPE)

    def test_full_hardware_menu_presence(self):
        """HW_MENU must include tc_array, tc_k_parallel, sfu_lanes, shared_banks/ports, multicast."""
        self.assertIn("tc_array", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["tc_array"], ("4x8", "8x8", "8x16"))
        self.assertIn("tc_k_parallel", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["tc_k_parallel"], (1, 2, 4))
        self.assertIn("sfu_lanes", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["sfu_lanes"], tuple(range(1, 65)))
        self.assertIn("shared_banks", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["shared_banks"], (1, 2, 4, 8))
        self.assertIn("shared_ports", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["shared_ports"], ("1R1W", "2R1W"))
        self.assertIn("multicast", dsl.HW_MENU)
        self.assertEqual(dsl.HW_MENU["multicast"], (False, True))

    def test_apply_ops_hardware_mutation_phase2(self):
        """Mutating hardware fields via apply_ops in phase 2 must succeed and validate."""
        ops = [
            {"op": "hw_set", "path": "tc_array", "value": "8x8"},
            {"op": "hw_set", "path": "tc_k_parallel", "value": 2},
            {"op": "hw_set", "path": "sfu_lanes", "value": 16},
            {"op": "hw_set", "path": "shared_banks", "value": 8},
            {"op": "hw_set", "path": "shared_ports", "value": "2R1W"},
            {"op": "hw_set", "path": "multicast", "value": True},
        ]
        r, hw = dsl.apply_ops(self.base_recipe, self.base_hw, ops, phase=2)
        self.assertEqual(hw["tc_array"], "8x8")
        self.assertEqual(hw["tc_k_parallel"], 2)
        self.assertEqual(hw["sfu_lanes"], 16)
        self.assertEqual(hw["shared_banks"], 8)
        self.assertEqual(hw["shared_ports"], "2R1W")
        self.assertTrue(hw["multicast"])
        self.assertEqual(dsl.validate_hardware(hw), [])

    def test_canonical_dependency_tc_zero(self):
        """Setting tc_count to 0 removes tc_array and tc_k_parallel."""
        ops = [{"op": "hw_set", "path": "tc_count", "value": 0}]
        r, hw = dsl.apply_ops(self.base_recipe, self.base_hw, ops, phase=2)
        self.assertEqual(hw["tc_count"], 0)
        self.assertNotIn("tc_array", hw)
        self.assertNotIn("tc_k_parallel", hw)
        self.assertEqual(dsl.validate_hardware(hw), [])

    def test_canonical_dependency_shared_zero(self):
        """Setting shared_kib to 0 removes shared_banks and shared_ports."""
        ops = [{"op": "hw_set", "path": "shared_kib", "value": 0}]
        r, hw = dsl.apply_ops(self.base_recipe, self.base_hw, ops, phase=2)
        self.assertEqual(hw["shared_kib"], 0)
        self.assertNotIn("shared_banks", hw)
        self.assertNotIn("shared_ports", hw)
        self.assertEqual(dsl.validate_hardware(hw), [])

    def test_canonical_dependency_sfu_clamped_on_vector_reduction(self):
        """Reducing vector_lanes below sfu_lanes clamps sfu_lanes canonically."""
        hw = copy.deepcopy(self.base_hw)
        hw["vector_lanes"] = 16
        hw["sfu_lanes"] = 16
        ops = [{"op": "hw_set", "path": "vector_lanes", "value": 8}]
        _, res_hw = dsl.apply_ops(self.base_recipe, hw, ops, phase=2)
        self.assertEqual(res_hw["vector_lanes"], 8)
        self.assertEqual(res_hw["sfu_lanes"], 8)
        self.assertEqual(dsl.validate_hardware(res_hw), [])

    def test_reject_impossible_tc_subfield_when_tc_zero(self):
        """Setting tc_array or tc_k_parallel when tc_count == 0 must reject explicitly."""
        hw = copy.deepcopy(self.base_hw)
        hw["tc_count"] = 0
        hw.pop("tc_array", None)
        hw.pop("tc_k_parallel", None)
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, hw,
                          [{"op": "hw_set", "path": "tc_array", "value": "8x8"}],
                          phase=2)
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, hw,
                          [{"op": "hw_set", "path": "tc_k_parallel", "value": 2}],
                          phase=2)

    def test_reject_impossible_shared_subfield_when_shared_zero(self):
        """Setting shared_banks or shared_ports when shared_kib == 0 must reject explicitly."""
        hw = copy.deepcopy(self.base_hw)
        hw["shared_kib"] = 0
        hw.pop("shared_banks", None)
        hw.pop("shared_ports", None)
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, hw,
                          [{"op": "hw_set", "path": "shared_banks", "value": 4}],
                          phase=2)
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, hw,
                          [{"op": "hw_set", "path": "shared_ports", "value": "1R1W"}],
                          phase=2)

    def test_reject_impossible_sfu_lanes_exceeding_vector_lanes(self):
        """Setting sfu_lanes > vector_lanes must reject explicitly."""
        hw = copy.deepcopy(self.base_hw)
        hw["vector_lanes"] = 8
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, hw,
                          [{"op": "hw_set", "path": "sfu_lanes", "value": 16}],
                          phase=2)

    def test_reject_unknown_hardware_field_and_value(self):
        """Unknown field or invalid value must raise DslError explicitly."""
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, self.base_hw,
                          [{"op": "hw_set", "path": "quantum_cores", "value": 4}],
                          phase=2)
        with self.assertRaises(dsl.DslError):
            dsl.apply_ops(self.base_recipe, self.base_hw,
                          [{"op": "hw_set", "path": "sm_count", "value": 99}],
                          phase=2)

    def test_validate_hardware_rejects_impossible_hardware(self):
        """validate_hardware directly flags invalid/impossible configurations."""
        bad_hw = copy.deepcopy(self.base_hw)
        bad_hw["tc_count"] = 0
        bad_hw["tc_array"] = "4x8"
        errs = dsl.validate_hardware(bad_hw)
        self.assertTrue(any("TC=0" in e for e in errs))

        bad_hw2 = copy.deepcopy(self.base_hw)
        bad_hw2["shared_kib"] = 0
        bad_hw2["shared_banks"] = 4
        errs2 = dsl.validate_hardware(bad_hw2)
        self.assertTrue(any("shared=0" in e for e in errs2))

        bad_hw3 = copy.deepcopy(self.base_hw)
        bad_hw3["vector_lanes"] = 8
        bad_hw3["sfu_lanes"] = 16
        errs3 = dsl.validate_hardware(bad_hw3)
        self.assertTrue(any("sfu_lanes" in e for e in errs3))

        bad_hw4 = copy.deepcopy(self.base_hw)
        bad_hw4["unknown_subsystem"] = 123
        errs4 = dsl.validate_hardware(bad_hw4)
        self.assertTrue(len(errs4) > 0)


class TestFingerprintAndCompilerIdentity(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(dsl.DEFAULT_RECIPE)
        self.hw = copy.deepcopy(dsl.BASELINE_HW)

    def test_legacy_fingerprint_compatibility(self):
        """fingerprint(recipe, hw) without compiler identity produces unchanged legacy digest."""
        import json, hashlib
        expected = hashlib.sha256(
            json.dumps({"r": self.recipe, "h": self.hw}, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.assertEqual(dsl.fingerprint(self.recipe, self.hw), expected)

    def test_optional_compiler_identity(self):
        """Optional compiler identity is permitted and produces distinct fingerprints."""
        fp_none = dsl.fingerprint(self.recipe, self.hw)
        fp_c1 = dsl.fingerprint(self.recipe, self.hw, compiler="comp_v1")
        fp_c2 = dsl.fingerprint(self.recipe, self.hw, compiler="comp_v2")
        fp_cid = dsl.fingerprint(self.recipe, self.hw, compiler_id="comp_v1")

        self.assertNotEqual(fp_none, fp_c1)
        self.assertNotEqual(fp_c1, fp_c2)
        self.assertEqual(fp_c1, fp_cid)


class TestStructureSignature(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(dsl.DEFAULT_RECIPE)
        self.hw = copy.deepcopy(dsl.BASELINE_HW)

    def test_structure_signature_exposed(self):
        """structure_signature must be exposed in dsl and propose modules."""
        self.assertTrue(callable(dsl.structure_signature))
        self.assertTrue(callable(propose.structure_signature))
        self.assertEqual(
            dsl.structure_signature(self.recipe, self.hw),
            propose.structure_signature(self.recipe, self.hw)
        )

    def test_structure_signature_effective_recipe_and_hardware(self):
        """Signature represents effective recipe state (order, pf, s, fuse, stages) and hw diffs."""
        sig1 = dsl.structure_signature(self.recipe, self.hw)
        self.assertEqual(sig1, "nkm+nofuse+nopf+s:hbm")

        r2 = copy.deepcopy(self.recipe)
        r2["fusion"]["bias_gelu"] = 1
        r2["gemm"]["prefetch"] = 1
        r2["attention"]["scores"] = "rf"
        r2["gemm_stages"] = {"d1": {"n_tile": 32}}
        hw2 = copy.deepcopy(self.hw)
        hw2["rf_ports"] = "8R4W"
        hw2["tc_array"] = "8x8"
        sig2 = dsl.structure_signature(r2, hw2)

        self.assertIn("fuse", sig2)
        self.assertIn("pf", sig2)
        self.assertIn("s:rf", sig2)
        self.assertIn("stage:d1", sig2)
        self.assertIn("hw:rf_ports=8R4W", sig2)
        self.assertIn("hw:tc_array=8x8", sig2)

    def test_struct_sig_backward_compatibility(self):
        """struct_sig handles both dict (recipe, hw) and legacy op lists."""
        sig_dict = propose.struct_sig(self.recipe, self.hw)
        self.assertEqual(sig_dict, "nkm+nofuse+nopf+s:hbm")

        ops = [{"op": "gemm_order", "order": "mnk"}, {"op": "fuse_bias_gelu", "enabled": 1}]
        sig_ops = propose.struct_sig(ops)
        self.assertEqual(sig_ops, "fuse+mnk")


class TestStageGemmTilesNeighborhoodsAndValidation(unittest.TestCase):
    def setUp(self):
        self.recipe = copy.deepcopy(dsl.DEFAULT_RECIPE)
        self.hw = copy.deepcopy(dsl.BASELINE_HW)

    def test_stage_gemm_tiles_phase1_unlocked(self):
        """stage_gemm_tiles is unlocked in Phase 1."""
        self.assertIn("stage_gemm_tiles", dsl.PHASE_UNLOCKED_OPS[1])
        op = [{"op": "stage_gemm_tiles", "stage": "p1_prompt", "m_tile": 16}]
        r, _ = dsl.apply_ops(self.recipe, self.hw, op, phase=1)
        self.assertEqual(r["gemm_stages"]["p1_prompt"]["m_tile"], 16)

    def test_stage_tile_neighborhood_generation(self):
        """stage_tile_neighborhood produces legal 1-step and 2-step stage tile ops."""
        nbrs = propose.stage_tile_neighborhood(self.recipe)
        self.assertTrue(len(nbrs) > 0)
        stages = {op["stage"] for op in nbrs}
        self.assertTrue(stages.issubset({"d1", "p1_prompt", "p1_step"}))
        for op in nbrs:
            self.assertEqual(op["op"], "stage_gemm_tiles")
            r, _ = dsl.apply_ops(self.recipe, self.hw, [op], phase=1)
            errs = dsl.validate_recipe(r, self.hw)
            self.assertEqual(errs, [], f"Op {op} produced invalid recipe: {errs}")

    def test_validate_recipe_rejects_bad_stage(self):
        """validate_recipe rejects invalid stage names, non-dict overrides, or illegal tiles."""
        r = copy.deepcopy(self.recipe)
        r["gemm_stages"] = {"invalid_stage": {"m_tile": 16}}
        errs = dsl.validate_recipe(r, self.hw)
        self.assertTrue(any("invalid_stage" in e for e in errs))

        r2 = copy.deepcopy(self.recipe)
        r2["gemm_stages"] = {"d1": {"m_tile": 999}}
        errs2 = dsl.validate_recipe(r2, self.hw)
        self.assertTrue(any("999" in e for e in errs2))

        # Test accumulator overflow in stage
        r3 = copy.deepcopy(self.recipe)
        r3["gemm_stages"] = {"p1_prompt": {"n_tile": 128}}
        # max_rows * n_tile = 64 * 128 = 8192 > 1024
        errs3 = dsl.validate_recipe(r3, self.hw)
        self.assertTrue(any("RF lane" in e for e in errs3))


class TestProposalQuotaAndPolicy(unittest.TestCase):
    def test_exact_quota_count_6(self):
        """count=6 must yield exactly 2 exploit, 2 expand, 2 novelty without truncation."""
        reg = {
            "candidates": {
                "cand-000": {
                    "score": 1000.0, "status": "evaluated", "struct_sig": "mnk",
                    "cycles": {"M1_P1": 30996995, "M2_D1": 1302032},
                    "family": "memory", "bias": "balanced",
                },
                "cand-001": {
                    "score": 1173.1, "status": "evaluated", "struct_sig": "nkm",
                    "cycles": {"M1_P1": 25000000, "M2_D1": 1200000},
                    "family": "memory", "bias": "balanced",
                }
            },
            "archives": {
                "elite": ["cand-001", "cand-000"],
                "pareto": ["cand-001"],
                "diversity": {"memory_balanced": "cand-001"},
            },
            "fingerprints": {},
        }
        rng = random.Random(42)
        diagnostics = {}
        props = propose.propose_round(reg, round_no=1, count=6,
                                      explore_ratio=0.34, phase=2,
                                      rng=rng, diagnostics=diagnostics)

        self.assertEqual(len(props), 6)
        kinds = [p["kind"] for p in props]
        self.assertEqual(kinds.count("exploit"), 2)
        self.assertEqual(kinds.count("expand"), 2)
        self.assertEqual(kinds.count("novelty"), 2)
        self.assertEqual(kinds, ["exploit", "exploit", "expand", "expand", "novelty", "novelty"])

        # Check diagnostics
        self.assertIn("attempts", diagnostics)
        self.assertIn("rejections", diagnostics)
        self.assertIn("rejection_reasons", diagnostics)

    def test_finite_attempts_bound(self):
        """Attempts must terminate within <= 128 finite bounds."""
        reg = {
            "candidates": {
                "cand-000": {
                    "score": 1000.0, "status": "evaluated", "struct_sig": "mnk",
                    "cycles": {"M1_P1": 30996995, "M2_D1": 1302032},
                    "family": "memory", "bias": "balanced",
                }
            },
            "archives": {"elite": ["cand-000"]},
            "fingerprints": {},
        }
        rng = random.Random(99)
        diag = {}
        props = propose.propose_round(reg, round_no=1, count=6,
                                      explore_ratio=0.34, phase=1,
                                      rng=rng, diagnostics=diag)
        self.assertLessEqual(diag["attempts"], 128)

    def test_diagnostics_records_rejection_reasons(self):
        """Rejection reasons (e.g. duplicate_fingerprint or recipe_invalid) are recorded."""
        # Find a neighbor fingerprint of cand-000 to pre-seed as tried
        r, hw = propose._load_parent_inputs("cand-000")
        ops = propose.stage_tile_neighborhood(r, hw)
        rec, _ = dsl.apply_ops(r, hw, ops[:1])
        fp = dsl.fingerprint(rec, hw)

        reg = {
            "candidates": {
                "cand-000": {
                    "score": 1000.0, "status": "evaluated", "struct_sig": "mnk",
                    "cycles": {"M1_P1": 30996995, "M2_D1": 1302032},
                    "family": "memory", "bias": "balanced",
                }
            },
            "archives": {"elite": ["cand-000"]},
            "fingerprints": {fp: "cand-old"},
        }
        from unittest.mock import patch
        first = propose.propose_round(reg, 1, 6, .34, 1, random.Random(42))
        diag = {}
        with patch.object(propose, '_tried_fingerprints', return_value={p['fingerprint'] for p in first}):
            propose.propose_round(reg, 1, 6, .34, 1, random.Random(42), diagnostics=diag)
        self.assertTrue(len(diag["rejections"]) > 0)
        self.assertTrue(len(diag["rejection_reasons"]) > 0)
        self.assertIn("duplicate_fingerprint", diag["rejection_reasons"])

    def test_existing_call_signature_preserved(self):
        """Calling propose_round with 6 arguments works without error."""
        reg = {
            "candidates": {
                "cand-000": {
                    "score": 1000.0, "status": "evaluated", "struct_sig": "mnk",
                    "cycles": {"M1_P1": 30996995, "M2_D1": 1302032},
                    "family": "memory", "bias": "balanced",
                }
            },
            "archives": {"elite": ["cand-000"]},
            "fingerprints": {},
        }
        rng = random.Random(7)
        props = propose.propose_round(reg, 1, 6, 0.34, 1, rng)
        self.assertEqual(len(props), 6)


if __name__ == "__main__":
    unittest.main()
