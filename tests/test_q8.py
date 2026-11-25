"""Q8: the frozen 2A tolerance, and the three holes it has to close.

The ruling of 2026-11-25 froze `relative_l2 <= 1e-05` with
`source=preregistered` and `post_hoc_adjustment=forbidden`. What is asserted
here is not the *value* -- that is a decision, not a property -- but that the
value cannot be swapped, relabelled, or bypassed:

  * 1e-9 cannot be called pre-registered          -> TestPolicy
  * a measured run cannot cite a test tolerance   -> TestPolicy
  * rel_l2 tiny + NaN present                     -> FAIL, not PASS
  * reference missing / wrong digest              -> BLOCKED, not "large error"
  * a record cannot shrink its coverage requirement -> TestEnvelope
  * CPU evidence still cannot fold to PASS        -> TestCeiling

Two things are deliberately *not* asserted: that C beats B', and that a local
boundary anomaly fails the gate. Q8 froze one global threshold; per-position
values are reported, and inventing a local gate afterwards would be the
post-hoc adjustment the ruling forbids.
"""

import copy
import json
import os
import random
import re
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import accept as A
from src.v2.cpu_ref import adapter_2a as A2
from src.v2.cpu_ref import envelope as E
from src.v2.cpu_ref import q8
from src.v2.cpu_ref import sp_attention as SP

D = 8
L = 96
WS = 4
TOLERANCE = 1e-9

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def world(seed=5, n=L, d=D):
    rnd = random.Random(seed)
    return ([[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)],
            [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)],
            [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)])


def run(all_gather=False):
    q, k, v = world()
    ref = SP.dense_attention(q, k, v)
    sp = SP.sp_forward(q, k, v, WS, all_gather=all_gather)
    return sp, ref


def coverage_of(out, ref, positions=None):
    pos = positions if positions is not None else q8.boundary_positions(L, WS)
    return {"positions": list(pos),
            "sp_rows": {p: out[p] for p in pos},
            "ref_rows": {p: ref[p] for p in pos}}


def identity():
    return {"path": "reference_10pos.json",
            "sha256": "a" * 64, "expected_sha256": "a" * 64,
            "source": "job1_manifest"}


def as_if_measured(sp_result):
    """A schema double, NOT hardware evidence.

    It exists to prove the fold can reach PASS once the evidence level says the
    numbers came off a device. No test claims this happened, and the record it
    produces has to name a non-synthetic model, so `as_if_measured` alone cannot
    buy a PASS.
    """
    out = copy.deepcopy(sp_result)
    out["evidence_level"] = E.MEASURED
    for a in out["accounting"]:
        a["backend"] = "cuda"
    return out


def work_block(n=L):
    """A named workload so a measured double can clear `check_workload`."""
    return E.workload(
        model={"synthetic": False, "name": "schema-double", "revision": "none",
               "note": "test double; not a checkpoint"},
        seq_len=n,
        position_ids={"start": 0, "end": n, "logical_len": n,
                      "rope_phase_verified": False})


# ---------------------------------------------------------------------------
# the policy block
# ---------------------------------------------------------------------------

class TestPolicy(unittest.TestCase):
    def test_frozen_values(self):
        p = q8.policy()
        self.assertEqual(p["metric"], "relative_l2")
        self.assertEqual(p["threshold"], 1e-5)
        self.assertEqual(p["direction"], "<=")
        self.assertEqual(p["source"], "preregistered")
        self.assertEqual(p["post_hoc_adjustment"], "forbidden")
        self.assertEqual(p["frozen_on"], "2026-11-25")
        self.assertEqual(q8.validate_policy(p), [])

    def test_a_preregistered_number_must_be_the_frozen_one(self):
        p = q8.policy(q8.SOURCE_PREREG, q8.CPU_TEST_TOL)
        problems = q8.validate_policy(p)
        self.assertTrue(problems)
        self.assertIn("CPU reference test constant", " ".join(problems))

    def test_a_test_value_may_not_gate_hardware(self):
        p = q8.policy(q8.SOURCE_TEST, TOLERANCE)
        self.assertEqual(q8.validate_policy(p), [])
        self.assertTrue(q8.validate_policy(p, evidence_level=E.MEASURED))

    def test_a_test_value_cannot_carry_a_freeze_date(self):
        p = q8.policy(q8.SOURCE_TEST, TOLERANCE)
        p["frozen_on"] = q8.FROZEN_ON
        self.assertIn("freeze date", " ".join(q8.validate_policy(p)))

    def test_tampering_with_any_declared_field_is_caught(self):
        for field, bad in (("metric", "cosine"), ("direction", ">="),
                           ("post_hoc_adjustment", "allowed"), ("q8", "Q9")):
            p = q8.policy()
            p[field] = bad
            self.assertTrue(q8.validate_policy(p), field)

    def test_a_negative_or_non_finite_threshold_is_rejected(self):
        for thr in (-1e-5, float("nan"), float("inf"), "1e-5", True):
            p = q8.policy(q8.SOURCE_TEST, 1e-9)
            p["threshold"] = thr
            self.assertTrue(q8.validate_policy(p), repr(thr))

    def test_measured_constant_matches_the_envelope_vocabulary(self):
        self.assertEqual(q8.MEASURED_EVIDENCE, E.MEASURED)


class TestBoundaryPositions(unittest.TestCase):
    def test_primary_workload_reproduces_the_frozen_tuple(self):
        self.assertEqual(q8.boundary_positions(), q8.BOUNDARY_POSITIONS)

    def test_matches_the_execution_document(self):
        """Guards against a transcription error in the constant above.

        The frozen tuple lives in the legacy stage-2 constants, which this round
        may not modify. Reading it back is the only way to prove the copy above
        is the same ten numbers and not ten plausible ones.
        """
        path = os.path.join(_REPO, "src", "legacy", "stage2", "constants.py")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"SAMPLE_POSITIONS[^=]*=\s*\(([^)]*)\)", text)
        self.assertIsNotNone(m, "SAMPLE_POSITIONS not found in the frozen constants")
        frozen = tuple(int(x) for x in re.findall(r"\d+", m.group(1)))
        self.assertEqual(q8.boundary_positions(), frozen)

    def test_short_sequence_obeys_the_same_rule(self):
        self.assertEqual(q8.boundary_positions(96, 4),
                         (23, 24, 25, 47, 48, 49, 71, 72, 73, 95))
        self.assertEqual(q8.boundary_positions(16, 4),
                         (3, 4, 5, 7, 8, 9, 11, 12, 13, 15))

    def test_two_ranks(self):
        self.assertEqual(q8.boundary_positions(16, 2), (7, 8, 9, 15))

    def test_no_duplicates_and_sorted(self):
        for L_ in (5, 8, 96, 130944):
            for n in (1, 2, 3, 4):
                pos = q8.boundary_positions(L_, n)
                self.assertEqual(list(pos), sorted(set(pos)))
                self.assertTrue(all(0 <= p < L_ for p in pos))
                self.assertEqual(pos[-1], L_ - 1)

    def test_bad_arguments(self):
        for args in ((0, 4), (96, 0), (-1, 4)):
            with self.assertRaises(ValueError):
                q8.boundary_positions(*args)


class TestAuxiliaryDiagnostic(unittest.TestCase):
    def test_identical_logits_agree_perfectly(self):
        rows = [[0.1, 2.0, 0.3, 4.0, 0.5, 6.0] for _ in range(3)]
        got = q8.topk_agreement(rows, rows, k=3)
        self.assertEqual(got["top1"], 1.0)
        self.assertEqual(got["topk_overlap"], 1.0)

    def test_disjoint_top1(self):
        a = [[9.0, 8.0, 1.0, 1.0, 1.0, 1.0]]
        b = [[1.0, 1.0, 1.0, 1.0, 8.0, 9.0]]
        got = q8.topk_agreement(a, b, k=2)
        self.assertEqual(got["top1"], 0.0)
        self.assertEqual(got["topk_overlap"], 0.0)

    def test_row_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            q8.topk_agreement([[1.0]], [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# 2A-04
# ---------------------------------------------------------------------------

class TestNumericGate(unittest.TestCase):
    def test_matching_tensors_pass(self):
        sp, ref = run()
        c = A.check_numeric(sp["output"], ref, TOLERANCE,
                            tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.PASS)
        self.assertLessEqual(c["detail"]["max_abs"], 1e-12)
        self.assertAlmostEqual(c["detail"]["cosine"], 1.0, places=12)
        self.assertEqual(c["detail"]["tolerance_source"], q8.SOURCE_TEST)

    def test_nan_with_tiny_rel_l2_is_fail(self):
        sp, ref = run()
        out = [list(r) for r in sp["output"]]
        out[0][0] = float("nan")
        c = A.check_numeric(out, ref, TOLERANCE, tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.FAIL)
        self.assertEqual(c["detail"]["reason"], "non_finite_in_sp_output")
        self.assertIsNone(c["detail"]["rel_l2"])

    def test_inf_is_fail(self):
        sp, ref = run()
        out = [list(r) for r in sp["output"]]
        out[3][1] = float("inf")
        c = A.check_numeric(out, ref, TOLERANCE, tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.FAIL)

    def test_a_broken_reference_blocks_instead_of_failing(self):
        sp, ref = run()
        bad_ref = [list(r) for r in ref]
        bad_ref[2][0] = float("nan")
        c = A.check_numeric(sp["output"], bad_ref, TOLERANCE,
                            tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["reason"], "reference_not_finite")

    def test_a_dropped_rank_is_a_shape_failure_not_a_truncated_comparison(self):
        sp, ref = run()
        short = sp["output"][:-1]
        c = A.check_numeric(short, ref, TOLERANCE, tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.FAIL)
        self.assertEqual(c["detail"]["reason"], "shape_mismatch")

    def test_no_tolerance_is_refused(self):
        sp, ref = run()
        with self.assertRaises(ValueError):
            A.check_numeric(sp["output"], ref, None)

    def test_a_test_tolerance_cannot_be_called_preregistered(self):
        sp, ref = run()
        with self.assertRaises(ValueError) as cm:
            A.check_numeric(sp["output"], ref, TOLERANCE,
                            tolerance_source=q8.SOURCE_PREREG)
        self.assertIn("preregistered", str(cm.exception))

    def test_evaluate_refuses_a_test_tolerance_on_a_measured_run(self):
        sp, ref = run()
        with self.assertRaises(ValueError):
            A.evaluate(as_if_measured(sp), ref, TOLERANCE,
                       tolerance_source=q8.SOURCE_TEST,
                       coverage=coverage_of(sp["output"], ref),
                       ref_identity=identity())

    def test_logit_agreement_is_recorded_as_non_gating(self):
        sp, ref = run()
        logits = ([[1.0, 2.0, 3.0]] * 4, [[1.0, 2.0, 3.0]] * 4)
        c = A.check_numeric(sp["output"], ref, TOLERANCE,
                            tolerance_source=q8.SOURCE_TEST, logits=logits)
        self.assertEqual(c["detail"]["logit_agreement"]["top1"], 1.0)
        self.assertFalse(c["detail"]["logit_agreement"]["gating"])

    def test_above_threshold_names_the_reason(self):
        sp, ref = run()
        out = [[x + 0.5 for x in r] for r in sp["output"]]
        c = A.check_numeric(out, ref, 1e-12, tolerance_source=q8.SOURCE_TEST)
        self.assertEqual(c["status"], A.FAIL)
        self.assertEqual(c["detail"]["reason"], "rel_l2_above_threshold")


# ---------------------------------------------------------------------------
# 2A-04b
# ---------------------------------------------------------------------------

class TestBoundaryCoverage(unittest.TestCase):
    def test_complete_coverage_passes(self):
        sp, ref = run()
        c = A.check_boundary_coverage(coverage_of(sp["output"], ref), L, WS,
                                      TOLERANCE)
        self.assertEqual(c["status"], A.PASS)
        self.assertEqual(c["detail"]["missing"], [])
        self.assertEqual(c["detail"]["above_threshold"], [])

    def test_no_sample_blocks(self):
        c = A.check_boundary_coverage(None, L, WS, TOLERANCE)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["required"], list(q8.boundary_positions(L, WS)))

    def test_a_missing_position_blocks(self):
        sp, ref = run()
        cov = coverage_of(sp["output"], ref)
        dropped = cov["positions"].pop()
        del cov["sp_rows"][dropped]
        c = A.check_boundary_coverage(cov, L, WS, TOLERANCE)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["missing"], [dropped])

    def test_a_missing_reference_blocks(self):
        sp, ref = run()
        cov = coverage_of(sp["output"], ref)
        dropped = cov["positions"][2]
        del cov["ref_rows"][dropped]
        c = A.check_boundary_coverage(cov, L, WS, TOLERANCE)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["missing_reference"], [dropped])

    def test_nan_at_a_boundary_is_fail(self):
        sp, ref = run()
        cov = coverage_of(sp["output"], ref)
        p = cov["positions"][0]
        cov["sp_rows"][p] = list(cov["sp_rows"][p])
        cov["sp_rows"][p][0] = float("nan")
        c = A.check_boundary_coverage(cov, L, WS, TOLERANCE)
        self.assertEqual(c["status"], A.FAIL)
        self.assertEqual(c["detail"]["reason"], "non_finite_or_width_mismatch_at_boundary")

    def test_sampled_rows_must_be_the_compared_rows(self):
        sp, ref = run()
        cov = coverage_of(sp["output"], ref)
        p = cov["positions"][0]
        cov["sp_rows"][p] = [x + 1.0 for x in cov["sp_rows"][p]]
        c = A.check_boundary_coverage(cov, L, WS, TOLERANCE,
                                      sp_output=sp["output"])
        self.assertEqual(c["status"], A.FAIL)
        self.assertEqual(c["detail"]["reason"],
                         "coverage_rows_differ_from_compared_output")

    def test_a_local_excess_is_reported_but_not_gated(self):
        """Documents the ruling, so nobody later mistakes silence for approval."""
        sp, ref = run()
        cov = coverage_of(sp["output"], ref)
        p = cov["positions"][1]
        cov["sp_rows"][p] = [x * 1.5 for x in cov["sp_rows"][p]]
        c = A.check_boundary_coverage(cov, L, WS, TOLERANCE)
        self.assertEqual(c["status"], A.PASS)
        self.assertEqual(c["detail"]["above_threshold"], [p])
        self.assertIn("per-position rel_l2 magnitude", c["detail"]["not_gating"])
        entry = [e for e in c["detail"]["per_position"] if e["position"] == p][0]
        self.assertGreater(entry["rel_l2"], TOLERANCE)


# ---------------------------------------------------------------------------
# 2A-04c and 2A-05
# ---------------------------------------------------------------------------

class TestReferenceIdentity(unittest.TestCase):
    def test_matching_digests_pass(self):
        c = A.check_reference_identity(identity(), E.MEASURED)
        self.assertEqual(c["status"], A.PASS)

    def test_mismatch_blocks_and_says_invalid(self):
        bad = identity()
        bad["sha256"] = "b" * 64
        c = A.check_reference_identity(bad, E.MEASURED)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["reason"], "reference_sha_mismatch")

    def test_a_measured_run_needs_both_digests(self):
        c = A.check_reference_identity({"path": "x", "sha256": "a" * 64}, E.MEASURED)
        self.assertEqual(c["status"], A.BLOCKED)
        self.assertEqual(c["detail"]["reason"], "reference_identity_unverifiable")

    def test_absent_identity_blocks(self):
        c = A.check_reference_identity(None, E.STRUCTURAL)
        self.assertEqual(c["status"], A.BLOCKED)


class TestBackend(unittest.TestCase):
    def acct(self, *backends):
        return [{"rank": i, "backend": b} for i, b in enumerate(backends)]

    def test_cpu_reference_is_not_applicable(self):
        c = A.check_backend(self.acct("cpu_ref_f64", "cpu_ref_f64"), E.STRUCTURAL)
        self.assertEqual(c["status"], A.NA)

    def test_mixed_backends_fail(self):
        c = A.check_backend(self.acct("cuda", "cpu_ref_f64"), E.MEASURED)
        self.assertEqual(c["status"], A.FAIL)

    def test_uniform_device_backend_passes_only_with_measured_evidence(self):
        self.assertEqual(A.check_backend(self.acct("cuda", "cuda"),
                                         E.STRUCTURAL)["status"], A.NA)
        self.assertEqual(A.check_backend(self.acct("cuda", "cuda"),
                                         E.MEASURED)["status"], A.PASS)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    def test_the_cpu_record_still_cannot_pass(self):
        sp, ref = run()
        res = A.evaluate(sp, ref, TOLERANCE, tolerance_source=q8.SOURCE_TEST,
                         coverage=coverage_of(sp["output"], ref),
                         ref_identity=identity())
        self.assertEqual(res["overall"], A.BLOCKED)
        ids = {c["id"]: c["status"] for c in res["checks"]}
        self.assertEqual(ids["2A-04"], A.PASS)
        self.assertEqual(ids["2A-04b"], A.PASS)
        self.assertEqual(ids["2A-04c"], A.PASS)
        self.assertEqual(ids["2A-05"], A.NA)

    def test_a_measured_record_folds_to_pass(self):
        sp, ref = run()
        res = A.evaluate(as_if_measured(sp), ref, q8.THRESHOLD,
                         tolerance_source=q8.SOURCE_PREREG,
                         coverage=coverage_of(sp["output"], ref),
                         ref_identity=identity())
        self.assertEqual(res["overall"], A.PASS)
        self.assertEqual(res["evidence_level"], E.MEASURED)

    def test_a_measured_record_missing_coverage_is_blocked_not_passed(self):
        sp, ref = run()
        res = A.evaluate(as_if_measured(sp), ref, q8.THRESHOLD,
                         tolerance_source=q8.SOURCE_PREREG,
                         ref_identity=identity())
        self.assertEqual(res["overall"], A.BLOCKED)

    def test_a_measured_record_with_measured_traffic_folds_to_pass(self):
        sp, ref = run()
        env = A2.from_2a(
            as_if_measured(sp), ref, q8.THRESHOLD, experiment_id="q8-measured",
            tolerance_source=q8.SOURCE_PREREG,
            coverage=coverage_of(sp["output"], ref),
            ref_identity=identity(), work=work_block(),
            communication={"rounds": 3, "bytes_per_round": 12288,
                           "backend": "nccl"})
        self.assertEqual(env["verdict"]["overall"], E.PASS)
        self.assertEqual(E.validate(json.loads(json.dumps(env)), strict=False), [])

    def test_a_measured_record_without_traffic_stays_blocked(self):
        sp, ref = run()
        env = A2.from_2a(as_if_measured(sp), ref, q8.THRESHOLD,
                         experiment_id="q8-measured-no-comm",
                         tolerance_source=q8.SOURCE_PREREG,
                         coverage=coverage_of(sp["output"], ref),
                         ref_identity=identity(), work=work_block())
        self.assertEqual(env["verdict"]["overall"], E.BLOCKED)
        self.assertIn("sp.communication", env["verdict"]["blocked_requires_gpu"])

    def test_a_cuda_run_cannot_be_filed_as_structural(self):
        sp, ref = run()
        sp = copy.deepcopy(sp)
        for a in sp["accounting"]:
            a["backend"] = "cuda"
        with self.assertRaises(ValueError):
            A.evaluate(sp, ref, TOLERANCE, tolerance_source=q8.SOURCE_TEST)

    def test_the_q8_rows_survive_json(self):
        sp, ref = run()
        env = A2.from_2a(sp, ref, TOLERANCE, experiment_id="q8-json",
                         coverage=coverage_of(sp["output"], ref),
                         ref_identity=identity())
        back = E.from_json(E.to_json(env))
        self.assertEqual(E.validate(back, strict=False), [])
        pol = back["sp_structure"]["numerical"]["tolerance_policy"]
        self.assertEqual(pol["threshold"], TOLERANCE)
        self.assertEqual(pol["source"], q8.SOURCE_TEST)


class TestEnvelope(unittest.TestCase):
    def env(self, **kw):
        sp, ref = run()
        return A2.from_2a(sp, ref, TOLERANCE, experiment_id="q8-env",
                          coverage=coverage_of(sp["output"], ref),
                          ref_identity=identity(), **kw)

    def problems(self, env):
        return E.validate(json.loads(json.dumps(env)), strict=False)

    def test_pre_registered_policy_is_accepted(self):
        sp, ref = run()
        env = A2.from_2a(sp, ref, q8.THRESHOLD, experiment_id="q8-prereg",
                         tolerance_source=q8.SOURCE_PREREG,
                         coverage=coverage_of(sp["output"], ref),
                         ref_identity=identity())
        self.assertEqual(self.problems(env), [])
        num = env["sp_structure"]["numerical"]
        self.assertEqual(num["tolerance_policy"]["source"], q8.SOURCE_PREREG)

    def test_a_record_cannot_shrink_its_coverage_requirement(self):
        env = self.env()
        env["sp_structure"]["numerical"]["boundary_coverage"]["required"] = [0, 1]
        self.assertIn("coverage requirement", " ".join(self.problems(env)))

    def test_a_record_cannot_swap_the_threshold(self):
        env = self.env()
        env["sp_structure"]["numerical"]["tolerance_policy"] = q8.policy(
            q8.SOURCE_PREREG, q8.CPU_TEST_TOL)
        self.assertIn("CPU reference test constant",
                      " ".join(self.problems(env)))

    def test_flat_and_nested_source_must_agree(self):
        env = self.env()
        env["sp_structure"]["numerical"]["tolerance_source"] = q8.SOURCE_PREREG
        self.assertIn("disagrees", " ".join(self.problems(env)))

    def test_threshold_and_policy_must_agree(self):
        env = self.env()
        env["sp_structure"]["numerical"]["tolerance"] = 1e-3
        self.assertIn("disagrees", " ".join(self.problems(env)))

    def test_missing_reference_identity_is_rejected(self):
        env = self.env()
        del env["sp_structure"]["numerical"]["reference_identity"]
        self.assertIn("reference_identity", " ".join(self.problems(env)))

    def test_missing_boundary_block_is_rejected(self):
        env = self.env()
        del env["sp_structure"]["numerical"]["boundary_coverage"]
        self.assertIn("boundary_coverage", " ".join(self.problems(env)))

    def test_missing_numerical_block_is_rejected(self):
        env = self.env()
        del env["sp_structure"]["numerical"]
        self.assertIn("2A-04", " ".join(self.problems(env)))

    def test_a_synthetic_workload_cannot_be_filed_as_measured(self):
        sp, ref = run()
        with self.assertRaises(E.EnvelopeError) as cm:
            A2.from_2a(
                as_if_measured(sp), ref, q8.THRESHOLD, experiment_id="q8-synth",
                tolerance_source=q8.SOURCE_PREREG,
                coverage=coverage_of(sp["output"], ref),
                ref_identity=identity(),
                communication={"rounds": 2, "bytes_per_round": 100,
                               "backend": "nccl"})
        joined = str(cm.exception)
        self.assertIn("synthetic", joined)
        self.assertIn("claiming a PASS", joined)


class TestCommunicationRecord(unittest.TestCase):
    def test_cpu_default_declares_structure_only(self):
        b = A2.communication_block()
        self.assertEqual(b["evidence"], "declared_structural")
        self.assertIsNone(b["rounds"])

    def test_a_measured_record_needs_rounds_and_bytes(self):
        for bad in ({"rounds": 0, "bytes_per_round": 1},
                    {"backend": "nccl"},
                    {"rounds": 2, "bytes_per_round": None}):
            with self.assertRaises(ValueError):
                A2.communication_block(bad)

    def test_unknown_keys_are_refused(self):
        with self.assertRaises(ValueError):
            A2.communication_block({"rounds": 1, "bytes_per_round": 1,
                                    "measured_by": "vibes"})


if __name__ == "__main__":
    unittest.main()
