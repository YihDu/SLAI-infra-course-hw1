"""Unit tests for search records, state atomic save, archives admission, and execution cache.

Uses temporary directories throughout: no historical files modified and no actual
evaluator runs invoked.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from project.search import archives, cache, records, state


class TestState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.reg_path = self.dir_path / "registry.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_atomic_save_and_load(self):
        reg = {"head": "cand-001", "counter": 1, "candidates": {"cand-001": {"score": 1200.0}}}
        state.save(reg, path=self.reg_path)
        self.assertTrue(self.reg_path.exists())
        loaded = state.load(path=self.reg_path)
        self.assertEqual(loaded["head"], "cand-001")
        self.assertEqual(loaded["candidates"]["cand-001"]["score"], 1200.0)

        # Ensure no leftover temporary files in directory
        tmp_files = list(self.dir_path.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0)

    def test_next_id_checks_registry_and_existing_dirs(self):
        reg = {"counter": 0, "candidates": {"cand-001": {}}}
        # Create cand-002 on disk without putting it into registry
        cand2_dir = self.dir_path / "cand-002"
        cand2_dir.mkdir(parents=True, exist_ok=True)

        # next_id should skip cand-001 (in reg) and cand-002 (on disk) and return cand-003
        nid = state.next_id(reg, experiments_dir=self.dir_path)
        self.assertEqual(nid, "cand-003")
        self.assertEqual(reg["counter"], 3)


class TestArchives(unittest.TestCase):
    def test_is_admissible_new_fields(self):
        # Explicit eligible=True and functional_passed=True -> admitted
        meta_valid = {
            "score": 1500.0,
            "eligible": True,
            "functional_passed": True,
            "failure_class": "none",
            "status": "evaluated",
        }
        self.assertTrue(archives.is_admissible(meta_valid))

        # Explicit eligible=False -> rejected
        meta_ineligible = dict(meta_valid, eligible=False)
        self.assertFalse(archives.is_admissible(meta_ineligible))

        # Explicit functional_passed=False -> rejected
        meta_func_fail = dict(meta_valid, functional_passed=False)
        self.assertFalse(archives.is_admissible(meta_func_fail))

        # Partial case functional failure -> rejected
        meta_case_fail = dict(meta_valid, functional={"M1_P1": True, "M2_D1": False})
        self.assertFalse(archives.is_admissible(meta_case_fail))

        # Failure class not none -> rejected
        meta_fail_class = dict(meta_valid, failure_class="generate_error")
        self.assertFalse(archives.is_admissible(meta_fail_class))

    def test_preserve_legacy_valid_entries(self):
        # Legacy entries lack eligible and functional_passed fields
        legacy_meta = {
            "score": 1173.0,
            "failure_class": "none",
            "status": "kept",
            "cycles": {"M1_P1": 22373717, "M2_D1": 1310832},
        }
        self.assertTrue(archives.is_admissible(legacy_meta))

        # Legacy failed entry with resource limit -> rejected
        legacy_failed = dict(legacy_meta, failure_class="evaluation_resource_limit")
        self.assertFalse(archives.is_admissible(legacy_failed))

        # Legacy entry without score -> rejected
        legacy_no_score = dict(legacy_meta, score=None)
        self.assertFalse(archives.is_admissible(legacy_no_score))

    def test_archive_update_and_pareto(self):
        reg = {
            "head": "cand-000",
            "candidates": {
                "cand-000": {"score": 1000.0, "cycles": {"M1_P1": 30000, "M2_D1": 1000}, "failure_class": "none"},
                "cand-001": {"score": 1200.0, "cycles": {"M1_P1": 20000, "M2_D1": 1100}, "failure_class": "none"},
                "cand-002": {"score": 1300.0, "cycles": {"M1_P1": 18000, "M2_D1": 900}, "failure_class": "none"},
                "cand-003": {"score": 1400.0, "eligible": False, "failure_class": "ineligible"},  # ineligible
            },
        }
        archives.update(reg, "cand-002")
        arch = reg["archives"]
        self.assertIn("cand-002", arch["elite"])
        self.assertNotIn("cand-003", arch["elite"])
        # Pareto front should contain cand-002 (18000, 900)
        self.assertIn("cand-002", arch["pareto"])
        self.assertEqual(archives.best_id(reg), "cand-002")


class TestRecords(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.log_path = self.dir_path / "iteration-log.md"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_incumbent_promotion_and_pareto_retention(self):
        reg = {
            "head": "cand-001",
            "candidates": {
                "cand-001": {
                    "score": 1200.0,
                    "cycles": {"M1_P1": 20000, "M2_D1": 1000},
                    "failure_class": "none",
                    "status": "kept",
                }
            },
        }

        # Case 1: Candidate cand-002 improves score, but M2_D1 cycles are SLOWER (1050 > 1000)
        # Should NOT promote to head; retains Pareto
        cand2 = SimpleNamespace(id="cand-002", parent="cand-001", hypothesis="tradeoff", summary="", status="created")
        metrics2 = {
            "verdict": "evaluated",
            "reports": {
                "grade": {
                    "score": 1300.0,
                    "eligible": True,
                    "cases": {
                        "M1_P1": {"cycles": 15000, "functional_passed": True},
                        "M2_D1": {"cycles": 1050, "functional_passed": True},
                    },
                }
            },
        }
        records.record_result(reg, cand2, metrics2, log_path=self.log_path)
        self.assertEqual(reg["head"], "cand-001")  # head not promoted
        self.assertEqual(reg["candidates"]["cand-002"]["status"], "evaluated")
        self.assertIn("cand-002", reg["archives"]["pareto"])
        self.assertEqual(reg["search"]["best_id"], "cand-002")  # best score updated in search

        # Case 2: Candidate cand-003 improves score AND BOTH cycles <= prior head
        # (14000 <= 20000 and 950 <= 1000) -> Promotes to head!
        cand3 = SimpleNamespace(id="cand-003", parent="cand-001", hypothesis="win-win", summary="", status="created")
        metrics3 = {
            "verdict": "evaluated",
            "reports": {
                "grade": {
                    "score": 1400.0,
                    "eligible": True,
                    "cases": {
                        "M1_P1": {"cycles": 14000, "functional_passed": True},
                        "M2_D1": {"cycles": 950, "functional_passed": True},
                    },
                }
            },
        }
        records.record_result(reg, cand3, metrics3, log_path=self.log_path)
        self.assertEqual(reg["head"], "cand-003")
        self.assertEqual(reg["candidates"]["cand-003"]["status"], "kept")
        self.assertEqual(cand3.status, "kept")
        self.assertEqual(reg["search"]["best_id"], "cand-003")

    def test_iteration_log_and_missing_errors_graceful(self):
        reg = {"head": None, "candidates": {}}
        cand = SimpleNamespace(id="cand-fail", parent="cand-000", hypothesis="buggy", summary="test", status="created")
        # Empty/missing error details in metrics
        metrics_fail = {
            "verdict": "incorrect",
            "elapsed": 1.25,
            "reports": {},
        }
        # Must not raise an exception even with missing fields
        records.record_result(reg, cand, metrics_fail, log_path=self.log_path)
        self.assertEqual(reg["candidates"]["cand-fail"]["status"], "incorrect")
        self.assertEqual(reg["candidates"]["cand-fail"]["failure_class"], "incorrect")

        # Verify log file was written
        self.assertTrue(self.log_path.exists())
        log_content = self.log_path.read_text(encoding="utf-8")
        self.assertIn("cand-fail: buggy", log_content)
        self.assertIn("Verify: FAILED", log_content)
        self.assertIn("elapsed=1.25s", log_content)


class TestCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.cache_file = self.dir_path / "cache.json"

        # Set up mock candidate directory structure
        self.cand_dir = self.dir_path / "cand-100"
        self.cand_dir.mkdir(parents=True)
        self.inputs_dir = self.cand_dir / "inputs"
        self.inputs_dir.mkdir()
        self.programs_dir = self.cand_dir / "programs"
        self.programs_dir.mkdir()
        self.reports_dir = self.cand_dir / "reports"
        self.reports_dir.mkdir()

        # Dummy input files
        (self.inputs_dir / "compiler.py").write_text("# compiler mock\n")
        (self.inputs_dir / "hardware.json").write_text(json.dumps({"sm_count": 8, "cache_mib": 2}))
        (self.inputs_dir / "recipe.json").write_text(json.dumps({"gemm": {"order": "nkm"}}))

        # Dummy assembly files
        (self.programs_dir / "M1_P1.asm").write_text("WG.BEGIN 0\nWG.END 0\n")
        (self.programs_dir / "M2_D1.asm").write_text("WG.BEGIN 0\nWG.END 0\n")

        self.cand = SimpleNamespace(
            id="cand-100",
            dir=self.cand_dir,
            inputs=self.inputs_dir,
            programs=self.programs_dir,
            reports=self.reports_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_input_fingerprint(self):
        fp1 = cache.input_fingerprint(self.cand)
        self.assertIsInstance(fp1, str)
        self.assertEqual(len(fp1), 64)

        # Modifying compiler.py changes fingerprint
        (self.inputs_dir / "compiler.py").write_text("# modified\n")
        fp2 = cache.input_fingerprint(self.cand)
        self.assertNotEqual(fp1, fp2)

    def test_execution_key_components(self):
        # Create mock codesign and baseline manifest
        codesign_dir = self.dir_path / "mock_codesign"
        codesign_dir.mkdir()
        (codesign_dir / "evaluator.py").write_text("# evaluator\n")

        baseline_file = self.dir_path / "mock_baseline.json"
        baseline_file.write_text(json.dumps({"version": 1}))

        key1 = cache.execution_key(
            self.cand,
            seeds=[7],
            validation="grade",
            codesign_dir=codesign_dir,
            baseline_path=baseline_file,
        )
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 64)

        # Different seed changes key
        key2 = cache.execution_key(
            self.cand,
            seeds=[7, 42],
            validation="grade",
            codesign_dir=codesign_dir,
            baseline_path=baseline_file,
        )
        self.assertNotEqual(key1, key2)

        # Different ASM changes key
        (self.programs_dir / "M1_P1.asm").write_text("WG.BEGIN 1\nWG.END 1\n")
        key3 = cache.execution_key(
            self.cand,
            seeds=[7],
            validation="grade",
            codesign_dir=codesign_dir,
            baseline_path=baseline_file,
        )
        self.assertNotEqual(key1, key3)

    def test_cache_store_lookup_and_linkage(self):
        key = "dummy_key_12345"
        valid_metrics = {
            "verdict": "evaluated",
            "score": 2100.0,
            "eligible": True,
            "reports": {
                "grade": {
                    "score": 2100.0,
                    "eligible": True,
                    "cases": {
                        "M1_P1": {"cycles": 12000000, "functional_passed": True},
                        "M2_D1": {"cycles": 700000, "functional_passed": True},
                    },
                }
            },
        }

        # Storing valid metrics succeeds
        stored = cache.store(key, self.cand, valid_metrics, cache_path=self.cache_file)
        self.assertTrue(stored)

        # Lookup returns enriched metrics pointing to source candidate
        hit = cache.lookup(key, cache_path=self.cache_file)
        self.assertIsNotNone(hit)
        self.assertTrue(hit["cached"])
        self.assertEqual(hit["source_candidate"], "cand-100")
        self.assertEqual(hit["score"], 2100.0)

        # Link source candidate in reports directory
        cache.link_source_candidate(self.cand, "cand-073")
        ref_file = self.reports_dir / "cached_from.json"
        self.assertTrue(ref_file.exists())
        ref_data = json.loads(ref_file.read_text())
        self.assertEqual(ref_data["cached_from"], "cand-073")

    def test_cache_rejects_ineligible_or_failed(self):
        key = "fail_key"
        failing_metrics = {
            "verdict": "incorrect",
            "score": 1000.0,
            "eligible": True,
            "reports": {
                "grade": {
                    "score": 1000.0,
                    "eligible": True,
                    "cases": {
                        "M1_P1": {"cycles": 12000, "functional_passed": False},
                    },
                }
            },
        }
        # Must refuse to store failing metrics
        stored = cache.store(key, self.cand, failing_metrics, cache_path=self.cache_file)
        self.assertFalse(stored)
        self.assertIsNone(cache.lookup(key, cache_path=self.cache_file))



if __name__ == "__main__":
    unittest.main()
