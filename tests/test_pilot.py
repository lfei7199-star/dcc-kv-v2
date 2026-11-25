"""Tests for the pilot record contract (`src/v2/cpu_ref/pilot.py`).

Two kinds of test live here, and they must not be confused:

  * structural -- each gate fires when it should, and does not fire when the
    record is honest. Synthetic records prove a gate works; they prove nothing
    about sigma, rho or DEFF.
  * arithmetic -- the bootstrap estimand, the pp conversion and the sensitivity
    bands. These use hand-computable inputs so the expected value is derivable,
    not "whatever the code printed the first time".
"""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import pilot as P  # noqa: E402

FAST_B = 300          # bootstrap iterations for tests; the design value is 10 000


# ---------------------------------------------------------------------------
# synthetic record builder
# ---------------------------------------------------------------------------

def _layout():
    """182 inputs -> 200 rows: 164 singletons then 18 pairs (2 golds each)."""
    out = []
    for c in range(182):
        sha = hashlib.sha256(("doc-%03d" % c).encode("utf-8")).hexdigest()
        m = 2 if c >= 164 else 1
        for g in range(m):
            out.append({"cluster_index": c, "input_sha256": sha,
                        "cluster_id": sha[:16], "gold_index": g,
                        "cluster_size": m})
    return out


LAYOUT = _layout()


def _clamp(x):
    return max(0.0, min(1.0, x))


def arm_f1(arm, row_id):
    """Deterministic per-arm score with a strong cluster component.

    The cluster term dominates the row jitter on purpose: that is what makes
    the 18 pairs correlated, which is the structure under test.
    """
    c = LAYOUT[row_id]["cluster_index"]
    shared = ((c * 37) % 23) / 22.0
    jitter = 0.0013 * (row_id % 7)
    if arm == "B":
        return _clamp(0.62 + 0.10 * shared + jitter)
    if arm == "B_prime":
        return _clamp(0.58 + 0.10 * shared + jitter + 0.004 * ((row_id * 5) % 3))
    if arm == "C":
        return _clamp(0.60 + 0.10 * shared + jitter + 0.003 * ((row_id * 7) % 4))
    if arm == "A_dense":
        return _clamp(0.64 + 0.10 * shared + jitter)
    raise ValueError(arm)


def make_cell(arm, budget_t, seed, f1=None):
    f1 = f1 or arm_f1
    rows = []
    for row_id, meta in enumerate(LAYOUT):
        rows.append({
            "schema_version": P.SCHEMA_VERSION,
            "row_id": row_id,
            "cluster_id": meta["cluster_id"],
            "cluster_size": meta["cluster_size"],
            "input_sha256": meta["input_sha256"],
            "gold_index": meta["gold_index"],
            "arm": arm,
            "budget_t": budget_t,
            "seed": seed,
            "f1": f1(arm, row_id),
            "pred_sha256": hashlib.sha256(
                ("%s-%d" % (arm, meta["cluster_index"])).encode()).hexdigest(),
            "pred_len": 12,
            "truncated": False,
            "effective_len": 65536,
        })
    return rows


def baseline(seeds=(7,)):
    """Three cells: B@1637, B'@1637, C@819 (one seed) -> both estimands."""
    rows = []
    rows += make_cell("B", 1637, None)
    rows += make_cell("B_prime", 1637, None)
    for s in seeds:
        rows += make_cell("C", 819, s)
    return rows, meta()


def meta():
    return {
        "schema_version": P.SCHEMA_VERSION,
        "cluster_serialization": P.CLUSTER_SERIALIZATION,
        "dataset": "zai-org/LongBench@5e628be4",
        "dataset_sha256": "896a04a0" + "0" * 56,
        "n_rows": 200,
        "n_clusters": 182,
        "evaluator": {"name": "f1_token", "version": "1", "raw_scale": "proportion"},
        "construction": {"randomness_source": "calibration_query_subsample"},
        "n_ref": 13096,
        "query_sha256": "a" * 64,
        "world_size": 4,
        "L": 65536,
        "purpose": "variance_estimation",
        "verdict_authority": False,
        "tokenizer": {"name": "qwen2.5", "version": "1", "revision": "main"},
        "estimand_pairs": [
            {"estimand": "Delta_shared",
             "candidate": {"arm": "B_prime", "budget_t": 1637},
             "reference": {"arm": "B", "budget_t": 1637}},
            {"estimand": "Delta_DCC",
             "candidate": {"arm": "C", "budget_t": 819},
             "reference": {"arm": "B_prime", "budget_t": 1637}},
        ],
    }


def run(rows, m=None, delta=None, b=FAST_B):
    return P.validate_pilot(rows, m if m is not None else meta(),
                            delta_pp=delta, bootstrap_B=b)


def fails(report):
    return {g["code"] for g in report["gates"] if g["status"] == "FAIL"}


def finds(report, gid):
    return [g for g in report["gates"] if g["id"] == gid]


# ---------------------------------------------------------------------------
# identity: serialization must be unambiguous before hashing
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):
    def test_field_boundary_ambiguity_is_removed(self):
        a = P.serialize_input("ab", "c")
        b = P.serialize_input("a", "bc")
        self.assertNotEqual(a, b)
        self.assertNotEqual(P.input_sha256("ab", "c"),
                            P.input_sha256("a", "bc"))

    def test_nfkc_normalization(self):
        self.assertEqual(P.input_sha256("ＡＢＣ", "q"), P.input_sha256("ABC", "q"))

    def test_newline_normalization(self):
        self.assertEqual(P.input_sha256("a\r\nb", "q"), P.input_sha256("a\nb", "q"))
        self.assertEqual(P.input_sha256("a\rb", "q"), P.input_sha256("a\nb", "q"))

    def test_internal_newlines_are_kept(self):
        self.assertNotEqual(P.input_sha256("a\nb", "q"), P.input_sha256("ab", "q"))

    def test_version_prefix_is_bound(self):
        self.assertTrue(P.serialize_input("x", "y").startswith(
            (P.CLUSTER_SERIALIZATION + "\x00").encode("ascii")))

    def test_short_id(self):
        sha = P.input_sha256("ctx", "q")
        self.assertEqual(len(sha), 64)
        self.assertEqual(P.cluster_id_from_sha(sha), sha[:16])
        with self.assertRaises(ValueError):
            P.cluster_id_from_sha("tooshort")

    def test_collision_detection_reports(self):
        shas = ["a" * 16 + "0" * 48, "a" * 16 + "1" * 48, "b" * 64]
        col = P.short_id_collisions(shas)
        self.assertEqual(len(col), 1)
        self.assertEqual(col[0][0], "a" * 16)

    def test_collision_detection_clean(self):
        self.assertEqual(P.short_id_collisions(["a" * 64, "b" * 64]), [])


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

class TestBootstrapEstimand(unittest.TestCase):
    def test_constant_data_has_zero_se(self):
        boot = P.cluster_bootstrap({"a": [2.0], "b": [2.0], "c": [2.0, 2.0]},
                                   iters=200, seed=1)
        self.assertEqual(boot["se"], 0.0)
        self.assertAlmostEqual(boot["point"], 2.0)

    # sizes 1/1/2 so the row mean and the cluster mean genuinely differ
    UNEQUAL = {"a": [0.0], "b": [0.0], "c": [10.0, 10.0]}   # row mean 5.0

    def test_point_estimate_is_the_row_mean(self):
        boot = P.cluster_bootstrap(self.UNEQUAL, iters=100, seed=1)
        self.assertAlmostEqual(boot["point"], 20.0 / 4.0, places=12)

    def test_translation_invariance(self):
        """Shifting every d_i must not move the SE.

        This is the test that kills the naive `sum m_c mean_c / N` variant: its
        extra `Var(sum k_c m_c) * mean^2` term makes the SE depend on the level
        of the estimate, which no variance estimator may do.
        """
        base = P.cluster_bootstrap(self.UNEQUAL, iters=600, seed=42)
        shifted = P.cluster_bootstrap(
            {k: [v + 5.0 for v in vs] for k, vs in self.UNEQUAL.items()},
            iters=600, seed=42)
        self.assertAlmostEqual(base["se"], shifted["se"], places=12)
        self.assertAlmostEqual(shifted["point"], base["point"] + 5.0, places=12)

    def test_cluster_equal_weight_is_a_different_parameter(self):
        """Kept in the suite precisely because it must never reach a record.

        Equal-weight averaging targets (0+0+10)/3 = 3.33 while the row-weighted
        estimand is 20/4 = 5.00 -- a 50% shift on the same data.
        """
        eq = P.cluster_bootstrap(self.UNEQUAL, iters=2000, seed=7,
                                 weight_by_size=False)
        eq_mean = sum(eq["replicates"]) / len(eq["replicates"])
        self.assertAlmostEqual(eq_mean, 10.0 / 3.0, delta=0.15)
        self.assertGreater(abs(eq_mean - 5.0), 1.0)

    def test_bootstrap_mean_tracks_point_at_the_real_cluster_count(self):
        """Bootstrap mean tracks the point estimate within 1% at n=182.

        This is a 1%-tolerance check, NOT a negligibility claim: the Hajek
        ratio bias is a methodological approximation that is expected to be
        small relative to the variance scale here, but no numerical bound is
        asserted and the H2 decision never relies on it.
        """
        data = {}
        for i in range(164):
            data["s%03d" % i] = [0.020 + 0.0001 * (i % 5)]
        for i in range(18):
            data["p%03d" % i] = [0.020 + 0.0001 * (i % 5), 0.021 + 0.0001 * (i % 3)]
        boot = P.cluster_bootstrap(data, iters=3000, seed=20261125)
        boot_mean = sum(boot["replicates"]) / len(boot["replicates"])
        self.assertAlmostEqual(boot_mean / boot["point"], 1.0, delta=0.01)

    def test_equal_sizes_make_both_variants_identical(self):
        data = {"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}
        w = P.cluster_bootstrap(data, iters=500, seed=3)
        e = P.cluster_bootstrap(data, iters=500, seed=3, weight_by_size=False)
        self.assertEqual(w["replicates"], e["replicates"])

    def test_bootstrap_is_deterministic(self):
        data = {"a": [0.1], "b": [0.9], "c": [0.4, 0.5]}
        a = P.cluster_bootstrap(data, iters=300, seed=99)
        b = P.cluster_bootstrap(data, iters=300, seed=99)
        self.assertEqual(a["replicates"], b["replicates"])

    def test_different_seed_differs(self):
        data = {"a": [0.1], "b": [0.9], "c": [0.4, 0.5]}
        a = P.cluster_bootstrap(data, iters=50, seed=1)
        b = P.cluster_bootstrap(data, iters=50, seed=2)
        self.assertNotEqual(a["replicates"], b["replicates"])

    def test_se_interval_brackets_se(self):
        data = {"a": [0.0], "b": [1.0], "c": [0.4, 0.6], "d": [0.2, 0.3],
                "e": [0.9], "f": [0.1]}
        point = P.cluster_bootstrap(data, iters=800, seed=5)["se"]
        ci = P.bootstrap_se_ci(data, outer=25, inner=60, seed=5)
        self.assertIsNotNone(ci)
        self.assertLessEqual(ci["ci_low"], ci["ci_high"])
        self.assertGreater(ci["ci_high"], 0.0)
        self.assertLess(ci["ci_low"], point * 3 + 1e-9)


class TestCorrelation(unittest.TestCase):
    def test_pearson_exact_line(self):
        self.assertAlmostEqual(P.pearson([1, 2, 3], [2, 4, 6]), 1.0, places=12)
        self.assertAlmostEqual(P.pearson([1, 2, 3], [6, 4, 2]), -1.0, places=12)

    def test_pearson_zero_variance_is_none_not_zero(self):
        """Returning 0.0 here would make deff_model look like a real reading."""
        self.assertIsNone(P.pearson([1, 1, 1], [1, 2, 3]))

    def test_fisher_ci(self):
        ci = P.fisher_ci(0.5, 18)
        se = 1.0 / math.sqrt(15)
        self.assertAlmostEqual(ci["ci_low"], math.tanh(math.atanh(0.5) - 1.96 * se),
                               places=12)
        self.assertAlmostEqual(ci["ci_high"], math.tanh(math.atanh(0.5) + 1.96 * se),
                               places=12)

    def test_fisher_needs_pairs(self):
        self.assertIsNone(P.fisher_ci(0.5, 3))

    def test_rho_intervals_have_all_required_keys(self):
        pairs = [(0.1 * i, 0.1 * i + 0.01 * (i % 3)) for i in range(18)]
        rho = P.estimate_rho(pairs, iters=400, seed=11)
        for k in P.RHO_REQUIRED:
            self.assertIn(k, rho)
        self.assertEqual(rho["n_pairs"], 18)
        self.assertEqual(rho["bootstrap_B"], 400)
        self.assertLessEqual(rho["ci_low"], rho["ci_high"])

    def test_rho_undefined_when_degenerate(self):
        pairs = [(0.5, 0.5)] * 18
        rho = P.estimate_rho(pairs, iters=100, seed=1)
        self.assertIsNone(rho["point"])
        self.assertEqual(rho["method"], "degenerate_zero_variance")


class TestSensitivityBands(unittest.TestCase):
    def test_thresholds_reproduce_the_design_numbers(self):
        g_row = P.sigma_gate_pp(-3.875, 1.5, 1.0)
        g_model = P.sigma_gate_pp(-3.875, 1.5, P.DEFF_MODEL_MAX)
        g_uni = P.sigma_gate_pp(-3.875, 1.5, P.DEFF_UNIVERSAL)
        self.assertAlmostEqual(g_row, 17.1365, places=4)
        self.assertAlmostEqual(g_model, 15.7754, places=4)
        self.assertAlmostEqual(g_uni, 12.1173, places=4)

    def test_state_unconditional(self):
        self.assertEqual(P.sensitivity_state(11.0, -3.875, 1.5)[0], "unconditional")

    def test_state_model_dependent(self):
        self.assertEqual(P.sensitivity_state(16.0, -3.875, 1.5)[0], "model_dependent")

    def test_state_beyond(self):
        self.assertEqual(P.sensitivity_state(20.0, -3.875, 1.5)[0], "beyond")

    def test_state_unknown_without_delta(self):
        self.assertEqual(P.sensitivity_state(5.0, -3.875, None)[0], "unknown_delta")

    def test_margin_inside_point_estimate(self):
        self.assertEqual(P.sensitivity_state(5.0, -0.5, 1.5)[0], "beyond")

    def test_positive_mean_cannot_be_rejected(self):
        """A mean above -delta makes NI_REJECTED structurally unreachable."""
        state, detail = P.sensitivity_state(0.47, 2.05, 1.5)
        self.assertEqual(state, "beyond")
        self.assertIn("unreachable", detail)


class TestVerdict(unittest.TestCase):
    def test_not_testable_without_delta(self):
        self.assertEqual(P.fold_verdict(-3.875, -6.0, -1.7, None), "NOT_TESTABLE")

    def test_ni_pass(self):
        self.assertEqual(P.fold_verdict(-0.2, -1.0, 0.6, 1.5), "NI_PASS")

    def test_ni_rejected_needs_whole_interval_outside(self):
        self.assertEqual(P.fold_verdict(-3.875, -6.0, -1.75, 1.5), "NI_REJECTED")

    def test_crossing_interval_is_inconclusive(self):
        self.assertEqual(P.fold_verdict(-3.875, -6.0, -1.0, 1.5), "INCONCLUSIVE")

    def test_point_outside_alone_is_not_rejection(self):
        """The exact error corrected in design/22 §4.2."""
        self.assertNotEqual(P.fold_verdict(-3.875, -6.7, -1.05, 1.5), "NI_REJECTED")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

class TestGatesClean(unittest.TestCase):
    def test_baseline_passes(self):
        rows, m = baseline()
        rep = run(rows, m)
        self.assertFalse(rep["blocking"], fails(rep))
        self.assertIsNotNone(rep["derived"])
        self.assertEqual(rep["verdict_authority"], False)

    def test_both_estimands_present(self):
        rows, m = baseline()
        rep = run(rows, m)
        ests = rep["derived"]["estimands"]
        self.assertEqual(sorted(ests), ["Delta_DCC", "Delta_shared"])
        self.assertEqual(len(ests["Delta_shared"]["per_seed"]), 1)
        self.assertIsNone(ests["Delta_shared"]["per_seed"][0]["seed"])

    def test_dcc_pairs_seeded_candidate_against_unseeded_reference(self):
        rows, m = baseline(seeds=(7,))
        rep = run(rows, m)
        dcc = rep["derived"]["estimands"]["Delta_DCC"]["per_seed"][0]
        self.assertEqual(dcc["seed"], 7)
        self.assertEqual(dcc["n_rows"], 200)

    def test_multiple_seeds_give_sigma_s(self):
        rows = make_cell("B", 1637, None) + make_cell("B_prime", 1637, None)
        for s in (1, 2, 3):
            rows += make_cell("C", 819, s)
        rep = run(rows, meta(), b=120)
        agg = rep["derived"]["estimands"]["Delta_DCC"]["aggregate"]
        self.assertEqual(agg["n_seeds"], 3)
        self.assertIsNotNone(agg["sigma_s_pp"])
        self.assertIn("sigma_D_median_pp", agg)
        self.assertNotIn("sigma_D", agg)


class TestGatesFail(unittest.TestCase):
    def test_g6_summary_only(self):
        rep = run([])
        self.assertTrue(rep["blocking"])
        self.assertIn("INVALID_SUMMARY_ONLY", fails(rep))
        self.assertIsNone(rep["derived"])

    def test_g1_missing_field(self):
        rows, m = baseline()
        del rows[0]["pred_len"]
        rep = run(rows, m)
        self.assertIn("INVALID_ROW_SCHEMA", fails(rep))

    def test_g1_unknown_arm(self):
        rows, m = baseline()
        rows[0]["arm"] = "D_typo"
        self.assertIn("BLOCKED_MISSING_ROWS", fails(run(rows, m)))

    def test_g2_wrong_histogram(self):
        rows, m = baseline()
        # turn a singleton into a second row of an existing pair cluster
        rows[0]["cluster_id"] = rows[164]["cluster_id"]
        rows[0]["cluster_size"] = 2
        self.assertIn("BLOCKED_CLUSTER_MISMATCH", fails(run(rows, m)))

    def test_g2_meta_disagrees(self):
        rows, m = baseline()
        m["n_clusters"] = 181
        self.assertIn("BLOCKED_CLUSTER_MISMATCH", fails(run(rows, m)))

    def test_g3_pair_input_differs(self):
        rows, m = baseline()
        rows[165]["input_sha256"] = "f" * 64
        self.assertIn("BLOCKED_INPUT_MISMATCH", fails(run(rows, m)))

    def test_g3_gold_not_distinct(self):
        rows, m = baseline()
        rows[165]["gold_index"] = rows[164]["gold_index"]
        self.assertIn("BLOCKED_INPUT_MISMATCH", fails(run(rows, m)))

    def test_g3_declared_size_wrong(self):
        rows, m = baseline()
        rows[164]["cluster_size"] = 1
        self.assertIn("BLOCKED_INPUT_MISMATCH", fails(run(rows, m)))

    def test_g4_pred_divergence_warns_but_does_not_block(self):
        rows, m = baseline()
        rows[165]["pred_sha256"] = "0" * 64
        rep = run(rows, m)
        warn = [g for g in rep["gates"] if g["code"] == "WARN_PRED_DIVERGENCE"]
        self.assertEqual(len(warn), 1)
        self.assertFalse(warn[0]["blocking"])
        self.assertFalse(rep["blocking"])

    def test_g7_f1_out_of_range(self):
        rows, m = baseline()
        rows[3]["f1"] = 1.5
        self.assertIn("FAIL_SCALE", fails(run(rows, m)))

    def test_g7_f1_percent_scale(self):
        rows, m = baseline()
        rows[3]["f1"] = 62.0
        self.assertIn("FAIL_SCALE", fails(run(rows, m)))

    def test_g7_declared_percent(self):
        rows, m = baseline()
        m["evaluator"]["raw_scale"] = "percent"
        self.assertIn("FAIL_SCALE", fails(run(rows, m)))

    def test_g8_rademacher_rejected(self):
        rows, m = baseline()
        m["construction"]["randomness_source"] = "rademacher_projection"
        rep = run(rows, m)
        self.assertIn("REJECT_LEGACY_PROVENANCE", fails(rep))

    def test_g9_cluster_id_tampered(self):
        rows, m = baseline()
        rows[5]["cluster_id"] = "0123456789abcdef"
        self.assertIn("FAIL_CLUSTER_HASH", fails(run(rows, m)))

    def test_g9_collision_is_checked_not_argued(self):
        shas = ["a" * 16 + "0" * 48, "a" * 16 + "1" * 48]
        self.assertTrue(P.short_id_collisions(shas))

    def test_g10_unknown_serialization(self):
        rows, m = baseline()
        m["cluster_serialization"] = "pilot-input/v9"
        self.assertIn("REJECT_UNKNOWN_SERIALIZATION", fails(run(rows, m)))

    def test_g11_duplicate_row(self):
        rows, m = baseline()
        rows.append(dict(rows[0]))
        self.assertIn("FAIL_ROW_INDEX", fails(run(rows, m)))

    def test_g11_short_cell(self):
        rows, m = baseline()
        rows.pop(7)
        self.assertIn("FAIL_ROW_INDEX", fails(run(rows, m)))

    def test_g13_bare_sigma_d_rejected(self):
        rows, m = baseline()
        rows[0]["sigma_D"] = 21.0
        self.assertIn("FAIL_UNCONDITIONAL_SIGMA", fails(run(rows, m)))

    def test_g14_unknown_schema(self):
        rows, m = baseline()
        m["schema_version"] = "pilot-0"
        self.assertIn("REJECT_UNKNOWN_SCHEMA", fails(run(rows, m)))

    def test_g14_missing_meta_keys(self):
        rows, m = baseline()
        del m["tokenizer"]
        self.assertIn("INVALID_META_SCHEMA", fails(run(rows, m)))

    def test_g14_row_schema_field_required(self):
        rows, m = baseline()
        del rows[0]["schema_version"]
        self.assertIn("INVALID_ROW_SCHEMA", fails(run(rows, m)))

    def test_g15_d_without_f1(self):
        rows, m = baseline()
        rows[0]["d"] = -0.04
        del rows[0]["f1"]
        rep = run(rows, m)
        self.assertIn("INVALID_PAIRED_SOURCE", fails(rep))

    def test_g15_stored_d_disagrees_with_f1(self):
        rows, m = baseline()
        rows[0]["d"] = -0.99
        self.assertIn("INVALID_PAIRED_SOURCE", fails(run(rows, m)))

    def test_g15_stored_d_agreeing_is_accepted(self):
        rows, m = baseline()
        # rows[0] is B (reference), rows[200] is B' (candidate) for row_id 0
        rows[0]["d"] = rows[200]["f1"] - rows[0]["f1"]
        rep = run(rows, m)
        self.assertFalse(rep["blocking"], fails(rep))

    def test_g15_missing_pair_partner(self):
        rows, m = baseline()
        del rows[17]
        rep = run(rows, m)
        self.assertTrue(rep["blocking"])

    def test_g15_unknown_estimand_direction(self):
        rows, m = baseline()
        m["estimand_pairs"][0]["candidate"]["arm"] = "C"
        self.assertIn("INVALID_PAIRED_SOURCE", fails(run(rows, m)))

    def test_g16_authority_claim_rejected(self):
        rows, m = baseline()
        m["verdict_authority"] = True
        self.assertIn("REJECT_AUTHORITY_CLAIM", fails(run(rows, m)))

    def test_g16_wrong_purpose(self):
        rows, m = baseline()
        m["purpose"] = "final_evaluation"
        self.assertIn("REJECT_AUTHORITY_CLAIM", fails(run(rows, m)))

    def test_g17_pilot_supplied_delta(self):
        rows, m = baseline()
        m["delta_pp"] = 1.5
        rep = run(rows, m)
        self.assertIn("REJECT_PILOT_SUPPLIED_DELTA", fails(rep))

    def test_g17_nested_delta_field(self):
        rows, m = baseline()
        m["criterion"] = {"ni_margin": 1.5}
        self.assertIn("REJECT_PILOT_SUPPLIED_DELTA", fails(run(rows, m)))


class TestDerivedGates(unittest.TestCase):
    def test_g5_incomplete_rho(self):
        rec = {"rho_D": {"method": "pearson"}, "rho_d": {}, "seed": 1, "d_i": [0.1]}
        derived = {"estimands": {"Delta_shared": {"per_seed": [rec]}}}
        codes = {g["code"] for g in P.check_derived_consistency(derived)
                 if g["status"] == "FAIL"}
        self.assertIn("INCOMPLETE_RHO", codes)

    def test_g5_complete_rho_passes(self):
        rho = {k: 0 for k in P.RHO_REQUIRED}
        rec = {"rho_D": rho, "rho_d": rho, "seed": 1, "d_i": [0.1]}
        derived = {"estimands": {"Delta_shared": {"per_seed": [rec]}}}
        codes = {g["code"] for g in P.check_derived_consistency(derived)
                 if g["status"] == "FAIL"}
        self.assertNotIn("INCOMPLETE_RHO", codes)

    def test_g12_shared_d_column_detected(self):
        rho = {k: 0 for k in P.RHO_REQUIRED}
        rec = {"rho_D": rho, "rho_d": rho, "seed": None, "d_i": [0.1, 0.2]}
        derived = {"estimands": {"Delta_shared": {"per_seed": [rec]},
                                 "Delta_DCC": {"per_seed": [dict(rec)]}}}
        codes = {g["code"] for g in P.check_derived_consistency(derived)
                 if g["status"] == "FAIL"}
        self.assertIn("FAIL_ESTIMAND_CONFLATION", codes)

    def test_g12_real_run_keeps_columns_separate(self):
        rows, m = baseline()
        rep = run(rows, m)
        ests = rep["derived"]["estimands"]
        shared = ests["Delta_shared"]["per_seed"][0]["d_i"]
        dcc = ests["Delta_DCC"]["per_seed"][0]["d_i"]
        self.assertNotEqual(shared, dcc)


# ---------------------------------------------------------------------------
# derived statistics on a full record
# ---------------------------------------------------------------------------

class TestDerivedRecord(unittest.TestCase):
    def setUp(self):
        rows, m = baseline()
        self.rep = run(rows, m, delta=1.5)
        self.rec = self.rep["derived"]["estimands"]["Delta_shared"]["per_seed"][0]

    def test_pp_conversion(self):
        self.assertAlmostEqual(self.rec["mean_d_pp"],
                               self.rec["mean_d"] * 100.0, places=10)
        self.assertAlmostEqual(self.rec["sd_d_row_pp"],
                               self.rec["sd_d_row"] * 100.0, places=10)

    def test_mean_is_the_known_offset(self):
        """B' is 0.04 below B by construction, so mean_d must be about -4 pp."""
        self.assertAlmostEqual(self.rec["mean_d_pp"], -4.0, delta=0.4)

    def test_se_hierarchy(self):
        self.assertIsNotNone(self.rec["se_iid_pp"])
        self.assertIsNotNone(self.rec["se_cluster_w_pp"])
        self.assertGreater(self.rec["se_cluster_w_pp"], 0.0)

    def test_ci_is_mean_plus_minus_1_96_se(self):
        half = 1.96 * self.rec["se_cluster_w_pp"]
        self.assertAlmostEqual(self.rec["ci_low"], self.rec["mean_d_pp"] - half, places=10)
        self.assertAlmostEqual(self.rec["ci_high"], self.rec["mean_d_pp"] + half, places=10)

    def test_deff_empirical_definition(self):
        if self.rec["deff_empirical"] is None:
            self.skipTest("se_iid is zero on this record")
        ratio = self.rec["se_cluster_w_pp"] / self.rec["se_iid_pp"]
        self.assertAlmostEqual(self.rec["deff_empirical"], ratio ** 2, places=10)

    def test_deff_empirical_within_universal_bound(self):
        if self.rec["deff_empirical"] is None:
            self.skipTest("se_iid is zero on this record")
        self.assertGreaterEqual(self.rec["deff_empirical"], 0.5)
        self.assertLess(self.rec["deff_empirical"], P.DEFF_IMPLAUSIBLE)

    def test_deff_model_uses_rho_d_only(self):
        if self.rec["rho_d"]["point"] is None:
            self.assertIsNone(self.rec["deff_model"])
        else:
            self.assertAlmostEqual(self.rec["deff_model"],
                                   1.0 + 0.18 * self.rec["rho_d"]["point"], places=12)

    def test_rho_uses_18_pairs(self):
        self.assertEqual(self.rec["rho_d"]["n_pairs"], 18)
        self.assertEqual(self.rec["rho_D"]["n_pairs"], 18)

    def test_d_i_is_recomputable_from_raw_scores(self):
        for d, fc, fr in zip(self.rec["d_i"], self.rec["f1_cand"], self.rec["f1_ref"]):
            self.assertAlmostEqual(d, fc - fr, places=12)

    def test_verdict_uses_cluster_ci_not_model(self):
        expected = P.fold_verdict(self.rec["mean_d_pp"], self.rec["ci_low"],
                                  self.rec["ci_high"], 1.5)
        self.assertEqual(self.rec["verdict"], expected)

    def test_no_delta_means_not_testable(self):
        rows, m = baseline()
        rec = run(rows, m)["derived"]["estimands"]["Delta_shared"]["per_seed"][0]
        self.assertEqual(rec["verdict"], "NOT_TESTABLE")

    def test_authority_never_true(self):
        self.assertFalse(self.rec["verdict_authority"])
        self.assertFalse(self.rep["verdict_authority"])
        self.assertIn("NI_REJECTED", self.rep["may_not_be_cited_as"])


class TestModelMisfitIsOnlyADiagnostic(unittest.TestCase):
    def _run(self, tol):
        old = P.MODEL_MISFIT_TOL
        P.MODEL_MISFIT_TOL = tol
        try:
            rows, m = baseline()
            rep = run(rows, m)
            return rep["derived"]["estimands"]["Delta_shared"]["per_seed"][0]
        finally:
            P.MODEL_MISFIT_TOL = old

    def test_flag_flips_but_statistics_do_not_move(self):
        strict = self._run(-1.0)      # everything is a misfit
        loose = self._run(1e9)        # nothing is
        self.assertIn("model_misfit", strict["flags"])
        self.assertNotIn("model_misfit", loose["flags"])
        for key in ("se_cluster_w_pp", "ci_low", "ci_high", "mean_d_pp",
                    "deff_empirical", "deff_model"):
            self.assertEqual(strict[key], loose[key], key)

    def test_model_never_replaces_measured_se(self):
        rec = self._run(1e9)
        if rec["deff_model"] is None or rec["deff_empirical"] is None:
            self.skipTest("model reading unavailable on this record")
        model_se = rec["se_iid_pp"] * math.sqrt(rec["deff_model"])
        self.assertNotAlmostEqual(model_se, rec["se_cluster_w_pp"], places=6)


    def test_rho_d_and_rho_D_are_not_interchangeable(self):
        """Same input + greedy gives rho_D ~ 1, but rho_d is a difference and
        can have another sign entirely. Using rho_D for DEFF would be wrong.

        synthetic construction; not a narrativeqa reading. The pair is kept
        permanently as a validator counterexample (rho_D > 0 while rho_d < 0),
        and must never be reused as a real-data prior or delta input.
        """
        rows, m = baseline()
        rec = run(rows, m)["derived"]["estimands"]["Delta_shared"]["per_seed"][0]
        self.assertGreater(rec["rho_D"]["point"], 0.9)
        self.assertNotAlmostEqual(rec["rho_d"]["point"], rec["rho_D"]["point"], places=2)
        if rec["rho_d"]["point"] < 0:
            self.assertIn("deff_empirical_below_one", rec["flags"])


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_report(self):
        rows, m = baseline()
        a = P.dump_report(run(rows, m))
        b = P.dump_report(run(rows, m))
        self.assertEqual(a, b)

    def test_row_order_does_not_matter(self):
        rows, m = baseline()
        a = P.dump_report(run(rows, m))
        b = P.dump_report(run(list(reversed(rows)), m))
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_tool():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "validate_pilot.py")
    spec = importlib.util.spec_from_file_location("validate_pilot_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pilot-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, rows, m):
        with open(os.path.join(self.tmp, "pilot_meta.json"), "w", encoding="utf-8") as f:
            json.dump(m, f)
        with open(os.path.join(self.tmp, "pilot_rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_valid_dir_exits_zero(self):
        rows, m = baseline()
        self._write(rows, m)
        tool = _load_tool()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = tool.main(["-", self.tmp, "--bootstrap-b", "120"])
        self.assertEqual(code, 0)
        self.assertIn("verdict_authority=false", buf.getvalue())

    def test_summary_only_exits_one(self):
        with open(os.path.join(self.tmp, "pilot_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta(), f)
        with open(os.path.join(self.tmp, "pilot_rows.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": {"mean_d": 0.04, "sd_d": 0.1}}) + "\n")
        tool = _load_tool()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = tool.main(["-", self.tmp])
        self.assertEqual(code, 1)
        self.assertIn("INVALID_SUMMARY_ONLY", buf.getvalue())

    def test_derived_file_written_only_when_unblocked(self):
        rows, m = baseline()
        self._write(rows, m)
        tool = _load_tool()
        out = os.path.join(self.tmp, "derived.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            tool.main(["-", self.tmp, "--bootstrap-b", "120", "--out", out])
        self.assertTrue(os.path.exists(out))
        with open(out, encoding="utf-8") as f:
            derived = json.load(f)
        self.assertFalse(derived["verdict_authority"])
        self.assertIn("Delta_shared", derived["estimands"])


class TestImportIsolation(unittest.TestCase):
    def test_pilot_does_not_pull_in_the_solver(self):
        import subprocess
        code = (
            "import sys, os;"
            "sys.path.insert(0, os.path.abspath('.'));"
            "import src.v2.cpu_ref.pilot as P;"
            "bad=[m for m in ('src.v2.cpu_ref.compact','src.v2.cpu_ref.record',"
            "'src.v2.cpu_ref.envelope','src.v2.cpu_ref.accept') if m in sys.modules];"
            "print(','.join(bad))"
        )
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
