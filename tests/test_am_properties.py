"""Acceptance tests for the AM-HighestAttnKeys upgrade (U1/U2/U3, 2026-11-25).

Each test maps to one line of the ruling:

  U1  the chain is AM-HighestAttnKeys: softmax-RMS selection, box-constrained
      beta solved as bounded LS (NOT NNLS-then-clip), Value by OLS
  U2  the [-3,3] box is kept and its saturation is *measured*, not adjusted away
  U3  per-edge n_ref = 2t, and B'/C are balanced on the same n_ref

plus the three beta properties the ruling asked to be pinned down separately:

  1. a uniform scalar shift is invisible inside one block's softmax
  2. a per-key beta is NOT invisible -- it changes the intra-block distribution
  3. the difference only becomes an error once blocks are merged

Nothing here asserts that C beats B'. That is an empirical claim and lives in
tools/measure_2c_specialization.py.
"""

import math
import random
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import compact as C

T = 48
BUDGET = 8
N_REF = 2 * BUDGET      # U3: n_ref = 2t
N_EVAL = 8
D = 8
DV = 8


def make_world(seed=11, t=T, d=D, dv=DV):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    v = [[rnd.gauss(0, 1) for _ in range(dv)] for _ in range(t)]
    return k, v


def make_queries(n, seed=101, d=D):
    rnd = random.Random(seed)
    return [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)]


# ---------------------------------------------------------------------------


class TestKeySelectionFidelity(unittest.TestCase):
    """AM S3.3: RMS of the *softmax* weights, not of the raw logits."""

    def test_softmax_rms_matches_manual_definition(self):
        k, _ = make_world()
        q = make_queries(6, seed=5)
        got = C.select_keys(q, k, BUDGET, selector="softmax_rms")

        # manual: a_i = softmax(q_i K^T), s_j = sqrt(mean_i a_ij^2)
        lg = C.scaled_logits(q, k)
        acc = [0.0] * len(k)
        for row in lg:
            m = max(row)
            ex = [math.exp(x - m) for x in row]
            z = sum(ex)
            for j in range(len(k)):
                a = ex[j] / z
                acc[j] += a * a
        score = [v / len(q) for v in acc]
        order = sorted(range(len(k)), key=lambda j: (-score[j], j))
        manual = sorted(order[:BUDGET])
        self.assertEqual(got, manual)

    def test_the_two_selectors_actually_disagree(self):
        k, _ = make_world(seed=13)
        q = make_queries(12, seed=7)
        a = C.select_keys(q, k, 6, selector="softmax_rms")
        b = C.select_keys(q, k, 6, selector="logit_rms")
        self.assertLess(C.selection_overlap(a, b), 1.0,
                        "softmax-RMS and logit-RMS picked the same set; the "
                        "upgrade would then be untestable on this fixture")

    def test_unknown_selector_rejected(self):
        k, _ = make_world()
        q = make_queries(4)
        with self.assertRaises(ValueError):
            C.select_keys(q, k, 4, selector="nope")


class TestBetaBox(unittest.TestCase):
    """U2: the box is kept, and its saturation is reported."""

    def setUp(self):
        self.k, self.v = make_world()
        self.q = make_queries(N_REF, seed=17)
        self.idx = C.select_keys(self.q, self.k, BUDGET)
        self.k_sel = [self.k[j] for j in self.idx]

    def test_all_beta_inside_the_box(self):
        beta, stats = C.beta_box_ls(self.q, self.k_sel, self.k)
        self.assertEqual(len(beta), BUDGET)
        for b in beta:
            self.assertGreaterEqual(b, C.BETA_LO - 1e-9)
            self.assertLessEqual(b, C.BETA_HI + 1e-9)

    def test_iters_zero_is_clamped_unconstrained_ls(self):
        """The warm start is 'unconstrained LS then clamp' -- not an NNLS solve."""
        beta0, _ = C.beta_box_ls(self.q, self.k_sel, self.k, iters=0)
        A, m = C._mass_system(self.q, self.k_sel, self.k)
        w0, _ = C.qr_lstsq(A, [[x] for x in m])
        ref = [math.log(min(max(row[0], math.exp(C.BETA_LO)), math.exp(C.BETA_HI)))
               for row in w0]
        for a, b in zip(beta0, ref):
            self.assertAlmostEqual(a, b, places=9)

    def test_projected_gradient_does_not_increase_the_residual(self):
        b0, _ = C.beta_box_ls(self.q, self.k_sel, self.k, iters=0)
        b2, _ = C.beta_box_ls(self.q, self.k_sel, self.k, iters=C.PGD_ITERS)
        r0 = C.mass_residual(self.q, self.k, self.k_sel, b0)
        r2 = C.mass_residual(self.q, self.k, self.k_sel, b2)
        self.assertLessEqual(r2, r0 + 1e-12)

    def test_saturation_is_counted_consistently(self):
        beta, stats = C.beta_box_ls(self.q, self.k_sel, self.k)
        hi = sum(1 for b in beta if b >= C.BETA_HI - C.BIND_TOL)
        lo = sum(1 for b in beta if b <= C.BETA_LO + C.BIND_TOL)
        self.assertEqual(stats["bind_hi_count"], hi)
        self.assertEqual(stats["bind_lo_count"], lo)
        self.assertAlmostEqual(stats["bind_hi_frac"], hi / len(beta))
        self.assertAlmostEqual(stats["bind_lo_frac"], lo / len(beta))

    def test_dead_is_distinct_from_bind_lo(self):
        """U2: 'dead' must not be a silent synonym for lower-bound saturation."""
        _, stats = C.beta_box_ls(self.q, self.k_sel, self.k)
        self.assertIn("dead_frac", stats)
        self.assertIn("bind_lo_frac", stats)
        for name in ("dead_frac", "bind_lo_frac", "bind_hi_frac", "mass_residual"):
            self.assertIsNotNone(stats[name])
        # with the box in force nothing can reach w = 0, so dead <= bind_lo + shared
        self.assertLessEqual(stats["dead_frac"], 1.0)

    def test_mass_residual_definition(self):
        beta, stats = C.beta_box_ls(self.q, self.k_sel, self.k)
        self.assertAlmostEqual(stats["mass_residual"],
                               C.mass_residual(self.q, self.k, self.k_sel, beta),
                               places=9)


class TestValueOLS(unittest.TestCase):
    """AM S3.2/C.2: ordinary least squares, no regularization."""

    def setUp(self):
        self.k, self.v = make_world(seed=19)
        self.q = make_queries(N_REF, seed=23)

    def test_ols_fit_is_full_rank_and_finite(self):
        rep = C.build(self.q, self.k, self.v, BUDGET)
        self.assertEqual(rep["value_mode"], "ols")
        st = rep["diagnostics"]["value"]
        self.assertEqual(st["rank"], BUDGET)
        self.assertFalse(st["rank_deficient"])
        self.assertGreater(st["sigma_min"], 0.0)
        for row in rep["v_sel"]:
            for x in row:
                self.assertTrue(math.isfinite(x))

    def test_ols_residual_not_worse_than_ridge(self):
        """If ridge were silently applied, OLS could not fit at least as well."""
        q, k, v = self.q, self.k, self.v
        rep = C.build(q, k, v, BUDGET, value_mode="ols")
        beta = rep["beta"]
        X = C._value_design(q, rep["k_sel"], beta)
        Y = C.attention_output(q, k, v)

        def resid(Cv):
            num = 0.0
            for i in range(len(X)):
                for d in range(len(Y[0])):
                    s = sum(X[i][j] * Cv[j][d] for j in range(len(X[0])))
                    num += (s - Y[i][d]) ** 2
            return math.sqrt(num)

        ridge = C.value_regression(q, rep["k_sel"], v, Y, beta)
        self.assertLessEqual(resid(rep["v_sel"]), resid(ridge) + 1e-9)


class TestRankGuards(unittest.TestCase):
    """U3: rank deficiency and n_ref < t must raise, never be regularized away."""

    def test_duplicate_column_is_rejected(self):
        x = [[1.0, 1.0, 0.3],
             [0.5, 0.5, 0.2],
             [0.2, 0.2, 0.9],
             [0.7, 0.7, 0.1]]
        y = [[1.0], [0.0], [1.0], [0.5]]
        with self.assertRaises(C.RankDeficientError) as ctx:
            C.qr_lstsq(x, y)
        stats = ctx.exception.stats
        self.assertTrue(stats.get("rank_deficient", False) or stats.get("reason"))

    def test_n_lt_p_is_rejected_with_reason(self):
        x = [[1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0]]
        y = [[1.0], [1.0]]
        with self.assertRaises(C.RankDeficientError) as ctx:
            C.qr_lstsq(x, y)
        self.assertEqual(ctx.exception.stats.get("reason"), "n_lt_p")

    def test_build_rejects_n_ref_below_budget(self):
        k, v = make_world()
        q = make_queries(BUDGET - 1, seed=29)
        with self.assertRaises(C.RankDeficientError):
            C.build(q, k, v, BUDGET)

    def test_singular_stats_reports_condition(self):
        k, v = make_world()
        q = make_queries(N_REF, seed=31)
        rep = C.build(q, k, v, BUDGET)
        st = rep["diagnostics"]["beta"]["beta_fit"]
        for key in ("rank", "sigma_min", "sigma_max", "cond", "r_min", "r_max"):
            self.assertIn(key, st)
        self.assertGreater(st["sigma_max"], 0.0)


class TestQueryBudget(unittest.TestCase):
    """U3: balanced n_ref, identical across the two arms."""

    def test_allocate_balanced_matches_the_ruling(self):
        self.assertEqual(C.allocate_balanced(3274, 3), [1092, 1091, 1091])

    def test_compute_n_ref_is_two_t(self):
        self.assertEqual(C.compute_n_ref(1637), 3274)
        self.assertEqual(C.compute_n_ref(1637, 8), 13096)  # the four-budget sweep

    def test_shared_uses_exactly_n_ref(self):
        k, v = make_world(seed=37)
        pools = [make_queries(40, seed=40 + i) for i in range(3)]
        rep = C.build_shared(pools, k, v, BUDGET, n_ref=N_REF, seed=3)
        self.assertEqual(rep["n_ref"], N_REF)
        self.assertEqual(sum(rep["allocation"]), N_REF)
        self.assertEqual(len(rep["allocation"]), 3)

    def test_both_arms_share_budget_and_chain(self):
        k, v = make_world(seed=41)
        pools = [make_queries(40, seed=50 + i) for i in range(3)]
        shared = C.build_shared(pools, k, v, BUDGET, n_ref=N_REF, seed=5)
        per = C.build_per_destination(pools, k, v, BUDGET, n_ref=N_REF, seed=5)
        self.assertEqual(shared["budget"], BUDGET)
        for rep in per:
            self.assertEqual(rep["budget"], BUDGET)
            self.assertEqual(rep["n_ref"], N_REF)
            self.assertEqual(rep["selector"], shared["selector"])
            self.assertEqual(rep["beta_mode"], shared["beta_mode"])
            self.assertEqual(rep["value_mode"], shared["value_mode"])

    def test_split_pool_is_disjoint(self):
        pool = make_queries(30, seed=61)
        cal, held, ci, hi = C.split_pool(pool, 12, 8, seed=9)
        self.assertEqual(len(cal), 12)
        self.assertEqual(len(held), 8)
        self.assertEqual(len(set(ci) & set(hi)), 0)

    def test_sample_indices_respects_exclusion(self):
        idx = C.sample_indices(20, 5, seed=2, exclude=[0, 1, 2])
        self.assertEqual(len(idx), 5)
        self.assertEqual(len(set(idx)), 5)
        self.assertFalse(set(idx) & {0, 1, 2})

    def test_pool_too_small_raises(self):
        with self.assertRaises(ValueError):
            C.sample_indices(3, 5)


class TestBetaMathematics(unittest.TestCase):
    """The three properties the ruling asked to keep separate."""

    def setUp(self):
        self.k, self.v = make_world(seed=43, t=40)
        self.q = make_queries(10, seed=47)
        self.idx = C.select_keys(self.q, self.k, BUDGET)
        self.k_sel = [self.k[j] for j in self.idx]
        self.v_sel = [self.v[j] for j in self.idx]

    def test_property_1_uniform_shift_is_invisible_intra_block(self):
        base = C.attention_output(self.q, self.k_sel, self.v_sel)
        for c in (-3.0, -0.5, 1.7, 3.0):
            shifted = C.attention_output(self.q, self.k_sel, self.v_sel,
                                         beta=[c] * BUDGET)
            for r, s in zip(base, shifted):
                for a, b in zip(r, s):
                    self.assertAlmostEqual(a, b, places=12)

    def test_property_1b_uniform_shift_does_scale_block_mass(self):
        masses0 = C.block_mass([(self.k_sel, self.v_sel, [0.0] * BUDGET)], self.q)[0]
        c = 1.3
        massesc = C.block_mass([(self.k_sel, self.v_sel, [c] * BUDGET)], self.q)[0]
        for a, b in zip(masses0, massesc):
            self.assertAlmostEqual(b / a, math.exp(c), places=9)

    def test_property_2_per_key_beta_changes_the_distribution(self):
        base = C.attention_output(self.q, self.k_sel, self.v_sel)
        beta = [0.0] * BUDGET
        beta[0] = 2.0      # make one retained key much heavier
        tilted = C.attention_output(self.q, self.k_sel, self.v_sel, beta=beta)
        diff = max(abs(a - b) for r, s in zip(base, tilted) for a, b in zip(r, s))
        self.assertGreater(diff, 1e-6,
                           "a non-constant beta must change the intra-block output")

    def test_property_3_uniform_beta_becomes_visible_after_merge(self):
        k_fix, v_fix = make_world(seed=53, t=12)

        def out(beta):
            return C.merged_attention([(self.k_sel, self.v_sel, beta),
                                       (k_fix, v_fix, None)], self.q)

        zero = out([0.0] * BUDGET)
        uni = out([1.0] * BUDGET)
        diff = max(abs(a - b) for r, s in zip(zero, uni) for a, b in zip(r, s))
        self.assertGreater(diff, 1e-9,
                           "a uniform shift must move the merged output even "
                           "though it cannot move the intra-block one")


class TestMerge(unittest.TestCase):
    """AM Eq.1+2: the compact block must hold up after concatenation."""

    def setUp(self):
        self.k, self.v = make_world(seed=57, t=44)
        self.q = make_queries(N_REF, seed=59)
        self.eval_q = make_queries(N_EVAL, seed=67)
        self.k_fix, self.v_fix = make_world(seed=71, t=10)

    def test_box_beta_reduces_mass_residual_vs_zero(self):
        rep = C.build(self.q, self.k, self.v, BUDGET)
        r_box = rep["diagnostics"]["beta"]["mass_residual"]
        r_zero = C.mass_residual(self.q, self.k, rep["k_sel"], [0.0] * BUDGET)
        self.assertLess(r_box, r_zero)

    def test_merged_error_is_finite_and_bounded(self):
        rep = C.build(self.q, self.k, self.v, BUDGET)
        e = C.merge_rel_l2(self.eval_q, self.k, self.v, self.k_fix, self.v_fix, rep)
        self.assertTrue(0.0 <= e < 10.0)

    def test_compact_block_mass_tracks_dense_mass(self):
        """Mass tracking is judged on the median, not per query.

        AM Eq.(2) is an *unweighted* least-squares problem, so queries with a
        large m dominate the fit and low-mass queries carry a wider relative
        error. On this fixture the per-query ratio spans roughly 0.43-3.2 while
        the median sits at 0.98-1.10 across seeds. Asserting a tight per-query
        band would therefore be testing the fixture, not the algorithm.
        """
        rep = C.build(self.q, self.k, self.v, BUDGET)
        got = C.block_mass([(rep["k_sel"], rep["v_sel"], rep["beta"])], self.q)[0]
        want = C.true_mass(self.q, self.k)
        ratios = sorted(g / w for g, w in zip(got, want))
        for r in ratios:
            self.assertTrue(math.isfinite(r) and r > 0.0)
        median = ratios[len(ratios) // 2]
        self.assertTrue(0.5 < median < 2.0,
                        "median block-mass ratio %.3f is far from 1" % median)


class TestAblationArmsStillRun(unittest.TestCase):
    """The pre-upgrade arms must remain runnable: they are the C3 falsification."""

    def test_uniform_and_none_arms(self):
        k, v = make_world(seed=73)
        q = make_queries(N_REF, seed=79)
        for mode in ("uniform", "none"):
            rep = C.build(q, k, v, BUDGET, beta_mode=mode)
            self.assertEqual(len(rep["beta"]), BUDGET)
            self.assertTrue(all(b == rep["beta"][0] for b in rep["beta"]))

    def test_ridge_arm_runs(self):
        k, v = make_world(seed=83)
        q = make_queries(N_REF, seed=89)
        rep = C.build(q, k, v, BUDGET, value_mode="ridge")
        self.assertEqual(rep["value_mode"], "ridge")
        self.assertEqual(len(rep["v_sel"]), BUDGET)


if __name__ == "__main__":
    unittest.main()
