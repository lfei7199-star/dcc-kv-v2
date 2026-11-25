"""2C / 2D structural tests: shared-mode (B') vs destination-conditioned (C).

Deliberately NOT asserted here: "per-destination reconstruction error is lower
than shared". That is an empirical claim which depends on budget tightness and
seed -- a first probe showed 7 of 9 (seed, budget) cells agreeing, with one seed
reversing. Burying it in a unit test would invite parameter tuning until it goes
green. It lives in tools/measure_2c_specialization.py instead, which records the
full table to results/cpu/2c_specialization.json.

What IS asserted: the structural properties the frozen doc requires.
  - 2D-02: identical reference queries -> numerically identical per-edge result
  - budget identity between B' and C arms
  - beta conserves total attention mass
  - destination conditioning actually changes the selected key sets

Fixture note (2026-11-25 upgrade): BUDGET/N_FIT were 12/4 before the AM chain
landed, i.e. n_ref < t. OLS is underdetermined in that regime, so those sizes
violated a precondition of the algorithm rather than testing it. They are now
8/16, which is n_ref >= 2t per the U3 ruling. The assertions themselves are
unchanged.
"""

import math
import random
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import compact as C

BUDGET = 8
N_FIT = 16      # = 2 * BUDGET, the U3 lower bound for per-edge reference queries
N_EVAL = 8


def make_world(L=64, dk=8, dv=8, seed=23):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(dk)] for _ in range(L)]
    v = [[rnd.gauss(0, 1) for _ in range(dv)] for _ in range(L)]
    return k, v


def subspace_queries(dims, n, seed, dk=8):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        row = [0.0] * dk
        for d in dims:
            row[d] = rnd.gauss(0, 1)
        out.append(row)
    return out


def three_destinations(seed=31):
    fit = [subspace_queries((2 * d, 2 * d + 1), N_FIT, seed + d) for d in range(3)]
    evl = [subspace_queries((2 * d, 2 * d + 1), N_EVAL, seed + 100 + d) for d in range(3)]
    return fit, evl


class TestCollapse(unittest.TestCase):
    """2D-02: Q1=Q2=Q3 -> the three per-edge results must be numerically equal."""

    def test_per_destination_reps_agree(self):
        k, v = make_world()
        q = subspace_queries((0, 1), N_FIT, seed=5)
        per = C.build_per_destination([q, q, q], k, v, BUDGET)
        self.assertEqual(per[0]["indices"], per[1]["indices"])
        self.assertEqual(per[0]["indices"], per[2]["indices"])
        self.assertEqual(per[0]["beta"], per[1]["beta"])
        self.assertEqual(per[0]["beta"], per[2]["beta"])
        self.assertEqual(per[0]["v_sel"], per[1]["v_sel"])
        self.assertEqual(per[0]["v_sel"], per[2]["v_sel"])

    def test_single_destination_equals_single_shared(self):
        k, v = make_world()
        q = subspace_queries((2, 3), N_FIT, seed=6)
        one = C.build_per_destination([q], k, v, BUDGET)[0]
        sh = C.build_shared([q], k, v, BUDGET)
        self.assertEqual(one["indices"], sh["indices"])
        self.assertEqual(one["v_sel"], sh["v_sel"])


class TestBudget(unittest.TestCase):
    def test_both_arms_use_the_same_budget(self):
        k, v = make_world()
        fit, _ = three_destinations()
        shared = C.build_shared(fit, k, v, BUDGET)
        per = C.build_per_destination(fit, k, v, BUDGET)
        self.assertEqual(len(shared["indices"]), BUDGET)
        for rep in per:
            self.assertEqual(rep["budget"], BUDGET)
            self.assertEqual(len(rep["indices"]), BUDGET)
            self.assertEqual(len(rep["v_sel"]), BUDGET)


class TestBetaMass(unittest.TestCase):
    def test_uniform_beta_conserves_mass(self):
        k, v = make_world()
        q = subspace_queries((0, 1), 1, seed=9)
        idx = C.select_keys(q, k, BUDGET)
        k_sel = [k[j] for j in idx]
        beta = C.beta_uniform(q, k_sel, k)
        true_m = C.true_mass(q, k)[0]
        sel = [sum(q[0][d] * k_sel[j][d] for d in range(8)) * (1.0 / math.sqrt(8)) + beta
               for j in range(len(k_sel))]
        self.assertAlmostEqual(C._logsumexp(sel), math.log(true_m), places=9)


class TestSelectionDiffers(unittest.TestCase):
    def test_conditioning_changes_key_sets(self):
        k, v = make_world()
        fit, _ = three_destinations()
        per = C.build_per_destination(fit, k, v, BUDGET)
        sets = [set(rep["indices"]) for rep in per]
        self.assertFalse(sets[0] == sets[1] == sets[2],
                         "destination conditioning produced identical key sets")

    def test_errors_are_finite(self):
        k, v = make_world()
        fit, evl = three_destinations()
        shared = C.build_shared(fit, k, v, BUDGET)
        per = C.build_per_destination(fit, k, v, BUDGET)
        for d in range(3):
            for comp in (shared, per[d]):
                e = C.error_on(evl[d], k, v, comp)
                self.assertTrue(0.0 <= e < 10.0, "error out of range: %r" % (e,))


if __name__ == "__main__":
    unittest.main()
