"""Tests for the 2C diagnostic conventions (2026-11-25 ruling, items 1 and 2).

The point of this file is not "the numbers are good" -- it is "each number means
exactly one thing". A record full of fields is worthless if two scripts disagree
on what `dead` counts, so every convention here is pinned by recomputing it
independently and comparing, and the frozen tolerances are asserted to be frozen.

Covered:
  - bind tolerance: fixed at 1e-6 beta units, tested on exact boundary cases
  - dead keys: contribution-based, denominator defined, distinct from bind_lo
  - beta design vs value design: two problems, two rank/cond pairs
  - rank deficiency: raises, and does NOT fall back to ridge/pinv
  - ill-conditioned but full rank: solves, and reports the large condition number
  - mass residual: pooled over all calibration queries, raw + denominator kept
  - record schema: missing/contradictory fields are rejected
"""

import math
import random
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import compact as C
from src.v2.cpu_ref import record as R

T = 40
BUDGET = 8
N_REF = 2 * BUDGET
D = 8


def make_world(seed=11, t=T, d=D):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    v = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    return k, v


def make_queries(n, seed=101, d=D):
    rnd = random.Random(seed)
    return [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)]


def good_record(seed=11):
    k, v = make_world(seed)
    q = make_queries(N_REF, seed=seed + 5)
    rep = C.build(q, k, v, BUDGET)
    return rec_for(rep, q), rep, k, v, q


def rec_for(rep, q, heldout=None):
    return R.edge_record(rep, source="src0", destination=0,
                         calib_queries=q, heldout_queries=heldout)


# ---------------------------------------------------------------------------


class TestFrozenConventions(unittest.TestCase):
    """A tolerance that can be chosen per run is not a tolerance."""

    def test_tolerances_are_the_frozen_values(self):
        self.assertEqual(C.BIND_TOL, 1e-6)
        self.assertEqual(C.DEAD_REL, 1e-6)
        self.assertEqual(C.MASS_RESIDUAL_AGG, "pooled_over_all_calibration_queries")

    def test_record_echoes_the_frozen_values(self):
        rec, *_ = good_record()
        self.assertEqual(rec["definitions"]["bind_tol"], C.BIND_TOL)
        self.assertEqual(rec["definitions"]["dead_rel"], C.DEAD_REL)
        self.assertEqual(rec["diagnostics"]["beta_bounds"]["bind_tol"], C.BIND_TOL)
        self.assertEqual(rec["diagnostics"]["beta_dead"]["dead_rel"], C.DEAD_REL)

    def test_tampered_tolerance_is_rejected(self):
        """If a record was produced under a different convention it must fail."""
        rec, *_ = good_record()
        rec["definitions"]["bind_tol"] = 1e-3
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("bind_tol", str(ctx.exception))

    def test_tampered_aggregation_is_rejected(self):
        rec, *_ = good_record()
        rec["definitions"]["mass_residual_agg"] = "mean_of_per_query_ratios"
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("aggregation", str(ctx.exception))


class TestBindTolerance(unittest.TestCase):
    """The boundary rule, on exact boundary cases rather than lucky fixtures."""

    def test_exact_boundary_cases(self):
        hi, lo, tol = 3.0, -3.0, 1e-6
        beta = [
            hi,                  # exactly on the upper bound
            hi - tol / 2,        # inside the tolerance band -> counts
            hi - 2 * tol,        # outside the band -> does not count
            lo,                  # exactly on the lower bound
            lo + tol / 2,        # inside -> counts
            lo + 2 * tol,        # outside -> does not count
            0.0,
        ]
        on_lo, on_hi = C.count_on_bound(beta, lo=lo, hi=hi, tol=tol)
        self.assertEqual(on_hi, 2)
        self.assertEqual(on_lo, 2)

    def test_saturation_is_counted_from_beta_not_from_w(self):
        """A saturated key sits at the bound in beta units; w = e^3 there."""
        k, v = make_world(seed=13)
        q = make_queries(N_REF, seed=17)
        idx = C.select_keys(q, k, BUDGET)
        k_sel = [k[j] for j in idx]
        # squeeze the box so saturation is guaranteed, then check the count
        beta, stats = C.beta_box_ls(q, k_sel, k, lo=-0.01, hi=0.01)
        on_lo, on_hi = C.count_on_bound(beta, lo=-0.01, hi=0.01)
        self.assertEqual(stats["bind_hi_count"], on_hi)
        self.assertEqual(stats["bind_lo_count"], on_lo)
        self.assertEqual(on_lo + on_hi, len(beta), "a 0.02-wide box must pin every key")
        self.assertAlmostEqual(stats["bind_hi_frac"] + stats["bind_lo_frac"], 1.0)

    def test_bound_counts_never_exceed_t(self):
        rec, *_ = good_record()
        bb = rec["diagnostics"]["beta_bounds"]
        self.assertLessEqual(bb["bind_hi_count"] + bb["bind_lo_count"], BUDGET)


class TestDeadKeys(unittest.TestCase):
    """dead is a mass-share quantity, not a synonym for lower-bound saturation."""

    def test_count_dead_on_synthetic_contributions(self):
        # total = 3.000000001, threshold = 1e-6 * total ~= 3.000000001e-6
        contrib = [1.0, 1e-9, 2.0]
        self.assertEqual(C.count_dead(contrib, 1e-6), 1)

    def test_count_dead_boundary_is_strict(self):
        """The rule is c_j < rel * total, strictly, so equality stays alive.

        Checked with an exactly representable ratio instead of a near-1e-6 one:
        contrib 1.5 total with rel 1/3 puts the second key exactly on the
        threshold, which is the only way to test the comparison operator rather
        than the floating-point rounding around it.
        """
        contrib = [1.0, 0.5]
        self.assertEqual(C.count_dead(contrib, 1.0 / 3.0), 0)
        self.assertEqual(C.count_dead(contrib, 0.34), 1)

    def test_dead_independent_recomputation(self):
        k, v = make_world(seed=19)
        q = make_queries(N_REF, seed=23)
        idx = C.select_keys(q, k, BUDGET)
        k_sel = [k[j] for j in idx]
        beta, stats = C.beta_box_ls(q, k_sel, k)

        A, _ = C._mass_system(q, k_sel, k)
        w = [math.exp(b) for b in beta]
        contrib, total = C.key_contributions(A, w)
        expect = sum(1 for c in contrib if c < C.DEAD_REL * total)

        self.assertEqual(stats["dead_count"], expect)
        self.assertAlmostEqual(stats["dead_denominator"], total, places=9)

    def test_dead_and_bind_lo_are_separate_quantities(self):
        """With the box on, w >= e^-3 > 0, so dead cannot mean w == 0."""
        _, stats = C.beta_box_ls(*_sel_fixture(seed=29))
        self.assertIn("dead_count", stats)
        self.assertIn("bind_lo_count", stats)
        # dead is decided by contributions, bind_lo by beta values: recompute each
        beta = stats["beta_min"]
        self.assertGreaterEqual(beta, C.BETA_LO - 1e-9)
        # the recorded rule names the denominator, so a reader can reproduce it
        self.assertIn("sum_j c_j", stats["dead_scope"])
        self.assertIsNotNone(stats["dead_denominator"])

    def test_dead_fraction_matches_count(self):
        _, stats = C.beta_box_ls(*_sel_fixture(seed=31))
        self.assertAlmostEqual(stats["dead_frac"], stats["dead_count"] / BUDGET)


def _sel_fixture(seed=29):
    k, v = make_world(seed=seed)
    q = make_queries(N_REF, seed=seed + 3)
    idx = C.select_keys(q, k, BUDGET)
    return q, [k[j] for j in idx], k


class TestTwoDesigns(unittest.TestCase):
    """beta fit and Value OLS are different least-squares problems."""

    def test_the_two_blocks_are_labelled_and_distinct(self):
        rec, *_ = good_record()
        bd = rec["diagnostics"]["beta_design"]
        vd = rec["diagnostics"]["value_design"]
        self.assertEqual(bd["problem"], "beta_box_ls")
        self.assertEqual(vd["problem"], "value_ols")
        for blk in (bd, vd):
            self.assertEqual(blk["p"], BUDGET)
            self.assertEqual(blk["n"], N_REF)
            self.assertIsNotNone(blk["rank"])
            self.assertIsNotNone(blk["cond"])

    def test_conditions_are_not_forced_equal(self):
        """Two different matrices rarely share a condition number."""
        rec, *_ = good_record(seed=23)
        bd = rec["diagnostics"]["beta_design"]
        vd = rec["diagnostics"]["value_design"]
        self.assertNotEqual(bd["cond"], vd["cond"],
                            "beta and value design share one cond; they are being "
                            "conflated")

    def test_condition_numbers_are_finite_when_full_rank(self):
        rec, *_ = good_record(seed=37)
        for name in ("beta_design", "value_design"):
            blk = rec["diagnostics"][name]
            self.assertFalse(blk["rank_deficient"])
            self.assertEqual(blk["rank"], blk["p"])
            self.assertTrue(math.isfinite(blk["cond"]))
            self.assertGreater(blk["cond"], 0.0)


class TestRankAndCondition(unittest.TestCase):
    """Deficiency raises; it is never regularized away (ruling item 2)."""

    def test_duplicate_key_makes_value_design_rank_deficient(self):
        k, v = make_world(seed=41)
        q = make_queries(N_REF, seed=43)
        y = C.attention_output(q, k, v)
        # two identical retained keys -> two identical columns in X
        k_dup = [k[0], k[0], k[1], k[2], k[3], k[4], k[5], k[6]]
        beta = [0.0] * len(k_dup)
        with self.assertRaises(C.RankDeficientError) as ctx:
            C.value_ols(q, k_dup, v, y, beta)
        stats = ctx.exception.stats
        self.assertTrue(stats.get("rank_deficient"))
        self.assertLess(stats["rank"], len(k_dup))

    def test_no_silent_fallback_on_failure(self):
        """The failure must not turn into a ridge solution."""
        k, v = make_world(seed=47)
        q = make_queries(N_REF, seed=53)
        y = C.attention_output(q, k, v)
        k_dup = [k[0], k[0], k[1], k[2], k[3], k[4], k[5], k[6]]
        try:
            C.value_ols(q, k_dup, v, y, [0.0] * len(k_dup))
        except C.RankDeficientError:
            pass
        else:
            self.fail("rank-deficient Value OLS returned instead of raising")

        policy = R.solver_policy()
        self.assertEqual(policy["fallback"], "")
        self.assertEqual(policy["on_failure"], "raise_and_record")
        self.assertTrue(policy["solve_separate_from_diagnose"])

    def test_ill_conditioned_but_full_rank_still_solves(self):
        """Near-dependent columns: solvable, but cond must say so."""
        rnd = random.Random(59)
        n, p = 24, 4
        x = []
        for _ in range(n):
            a, b, c = (rnd.gauss(0, 1) for _ in range(3))
            # fourth column is almost a linear combination of the first two
            x.append([a, b, c, a + b + 1e-10 * rnd.gauss(0, 1)])
        y = [[rnd.gauss(0, 1)] for _ in range(n)]
        stats = C.singular_stats(x)
        self.assertEqual(stats["rank"], p, "rank must stay full at 1e-10 perturbation")
        self.assertGreater(stats["cond"], 1e6, "a near-dependency must show up in cond")

        sol, info = C.qr_lstsq(x, y)
        self.assertEqual(len(sol), p)
        for row in sol:
            self.assertTrue(math.isfinite(row[0]))

    def test_condition_number_is_reported_not_clipped(self):
        rnd = random.Random(61)
        x = [[1.0, 1.0], [1.0, 1.0 + 1e-9], [1.0, 1.0 - 1e-9]]
        stats = C.singular_stats(x)
        self.assertGreater(stats["cond"], 1e6)
        self.assertLessEqual(stats["rank"], 2)


class TestMassResidual(unittest.TestCase):
    """Pooled over all calibration queries, raw and denominator both kept."""

    def test_ratio_equals_raw_over_denominator(self):
        _, stats = C.beta_box_ls(*_sel_fixture(seed=67))
        self.assertAlmostEqual(stats["mass_residual"],
                               stats["mass_residual_raw"] / stats["mass_residual_denom"],
                               places=12)

    def test_matches_independent_pooled_computation(self):
        q, k_sel, k = _sel_fixture(seed=71)
        beta, stats = C.beta_box_ls(q, k_sel, k)

        A, m = C._mass_system(q, k_sel, k)
        w = [math.exp(b) for b in beta]
        resid = [sum(A[i][j] * w[j] for j in range(len(w))) - m[i]
                 for i in range(len(A))]
        num = math.sqrt(sum(r * r for r in resid))
        den = math.sqrt(sum(v * v for v in m))
        self.assertAlmostEqual(stats["mass_residual"], num / den, places=12)
        self.assertAlmostEqual(stats["mass_residual_denom"], den, places=9)

    def test_aggregation_is_declared_in_the_record(self):
        rec, *_ = good_record()
        mr = rec["diagnostics"]["mass_residual"]
        self.assertEqual(mr["aggregation"], "pooled_over_all_calibration_queries")
        self.assertIsNotNone(mr["raw"])
        self.assertIsNotNone(mr["denominator"])
        self.assertAlmostEqual(mr["normalized"], mr["raw"] / mr["denominator"], places=12)


class TestRecordSchema(unittest.TestCase):
    """A record that cannot be compared is not a record."""

    def test_good_record_validates(self):
        rec, *_ = good_record()
        self.assertEqual(R.validate(rec), [])

    def test_missing_field_raises(self):
        rec, *_ = good_record()
        del rec["payload"]
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("payload", str(ctx.exception))

    def test_budget_disagreement_raises(self):
        rec, *_ = good_record()
        rec["budget_tokens"] = BUDGET + 1
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("budget", str(ctx.exception))

    def test_duplicate_indices_raise(self):
        rec, *_ = good_record()
        rec["selected_indices"][1] = rec["selected_indices"][0]
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("duplicate", str(ctx.exception))

    def test_fallback_flag_raises(self):
        rec, *_ = good_record()
        rec["solver"]["fallback_used"] = True
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("fallback", str(ctx.exception))

    def test_position_span_beyond_logical_len_raises(self):
        rec, *_ = good_record()
        rec["position_range"] = [0, rec["logical_len"] + 10]
        with self.assertRaises(R.RecordError) as ctx:
            R.validate(rec)
        self.assertIn("position", str(ctx.exception))

    def test_record_is_json_serialisable(self):
        rec, *_ = good_record()
        text = R.to_json(rec)
        self.assertIn('"schema"', text)
        self.assertNotIn("NaN", text)


class TestPayloadAccounting(unittest.TestCase):
    """Payload is per edge; there is no cross-edge total by construction."""

    def test_breakdown_adds_up(self):
        rec, rep, *_ = good_record()
        p = rec["payload"]
        self.assertEqual(p["counted_for"], "this edge only")
        total = (p["keys_bytes"] + p["values_bytes"] + p["beta_bytes"]
                 + p["indices_bytes"] + p["meta_bytes"])
        self.assertEqual(p["bytes"], total)
        self.assertEqual(p["t"], BUDGET)

    def test_dtype_bytes_changes_the_size(self):
        k, v = make_world(seed=73)
        q = make_queries(N_REF, seed=79)
        rep = C.build(q, k, v, BUDGET)
        f64 = R.payload_breakdown(rep, dtype_bytes=8)
        bf16 = R.payload_breakdown(rep, dtype_bytes=2)
        self.assertLess(bf16["bytes"], f64["bytes"])
        self.assertEqual(f64["indices_bytes"], bf16["indices_bytes"],
                         "index width must not silently follow the float dtype")


if __name__ == "__main__":
    unittest.main()
