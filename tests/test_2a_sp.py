"""2A: SP merge correctness, order independence, and negative-arm rejection.

All tests are CPU reference tests. `test_never_passes_on_cpu` pins the rule that
this skeleton may not claim 2A PASS without CUDA evidence.
"""

import itertools
import math
import random
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import accept
from src.v2.cpu_ref.sp_attention import (
    dense_attention,
    partition_bounds,
    sp_forward,
)
from src.v2.cpu_ref.state import block_state, merge_all

TOL = 1e-10


def make_inputs(L=64, dk=8, dv=8, seed=7):
    rnd = random.Random(seed)
    q = [[rnd.gauss(0, 1) for _ in range(dk)] for _ in range(L)]
    k = [[rnd.gauss(0, 1) for _ in range(dk)] for _ in range(L)]
    v = [[rnd.gauss(0, 1) for _ in range(dv)] for _ in range(L)]
    return q, k, v


class TestPartition(unittest.TestCase):
    def test_frozen_layout(self):
        b = partition_bounds(130944, 4)
        self.assertEqual(b, [(0, 32736), (32736, 65472), (65472, 98208), (98208, 130944)])
        for s, e in b:
            self.assertEqual(e - s, 32736)

    def test_covers_everything(self):
        for L in (10, 64, 65, 130944):
            for n in (1, 2, 3, 4):
                b = partition_bounds(L, n)
                self.assertEqual(b[0][0], 0)
                self.assertEqual(b[-1][1], L)
                for i in range(1, len(b)):
                    self.assertEqual(b[i][0], b[i - 1][1])


class TestSPMatchesDense(unittest.TestCase):
    def test_world_sizes(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        for n in (1, 2, 4):
            got = sp_forward(q, k, v, n)["output"]
            self.assertLessEqual(accept.rel_l2(got, ref), TOL, "world_size=%d" % n)

    def test_non_divisible_length(self):
        q, k, v = make_inputs(L=37)
        ref = dense_attention(q, k, v)
        got = sp_forward(q, k, v, 4)["output"]
        self.assertLessEqual(accept.rel_l2(got, ref), TOL)


class TestOrderIndependence(unittest.TestCase):
    def test_all_permutations_of_four_blocks(self):
        q, k, v = make_inputs(L=64, seed=11)
        bounds = partition_bounds(64, 4)
        s_r, e_r = bounds[3]
        q_rows = q[s_r:e_r]
        scale = 1.0 / math.sqrt(len(q[0]))

        states = []
        for (bs, be) in bounds:
            kb, vb = k[bs:be], v[bs:be]
            sc = [[sum(qv[d] * kb[j][d] for d in range(len(qv))) * scale for j in range(be - bs)] for qv in q_rows]
            is_own = (bs == s_r and be == e_r)
            vis = [[(bs + j) <= (s_r + i) for j in range(be - bs)] if is_own else [True] * (be - bs)
                   for i in range(len(q_rows))]
            states.append(block_state(sc, vb, vis))

        base = merge_all(states).output()
        n_perm = 0
        for perm in itertools.permutations(range(4)):
            got = merge_all([states[i] for i in perm]).output()
            self.assertLessEqual(accept.rel_l2(got, base), TOL, "perm=%r" % (perm,))
            n_perm += 1
        self.assertEqual(n_perm, 24)


class TestNegativeArm(unittest.TestCase):
    def test_all_gather_arm_is_rejected(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        bad = sp_forward(q, k, v, 4, all_gather=True)
        res = accept.evaluate(bad, ref, tolerance=TOL)
        self.assertEqual(res["overall"], accept.FAIL)
        ids = {c["id"]: c["status"] for c in res["checks"]}
        self.assertEqual(ids["2A-01"], accept.FAIL)
        self.assertEqual(ids["2A-02"], accept.FAIL)

    def test_positive_arm_structurally_ok(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        good = sp_forward(q, k, v, 4)
        res = accept.evaluate(good, ref, tolerance=TOL)
        ids = {c["id"]: c["status"] for c in res["checks"]}
        self.assertEqual(ids["2A-01"], accept.PASS)
        self.assertEqual(ids["2A-02"], accept.PASS)
        self.assertEqual(ids["2A-03"], accept.PASS)
        self.assertEqual(ids["2A-04"], accept.PASS)

    def test_remote_block_counts(self):
        q, k, v = make_inputs()
        acct = sp_forward(q, k, v, 4)["accounting"]
        self.assertEqual([a["remote_block_count"] for a in acct], [0, 1, 2, 3])

    def test_numeric_tolerance_must_be_preregistered(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        with self.assertRaises(ValueError):
            accept.check_numeric(ref, ref, tolerance=None)


class TestNoFalsePass(unittest.TestCase):
    def test_never_passes_on_cpu(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        res = accept.evaluate(sp_forward(q, k, v, 4), ref, tolerance=TOL)
        self.assertEqual(res["overall"], accept.BLOCKED)
        self.assertNotEqual(res["overall"], accept.PASS)

    def test_bad_tolerance_fails(self):
        q, k, v = make_inputs()
        ref = dense_attention(q, k, v)
        res = accept.evaluate(sp_forward(q, k, v, 4), ref, tolerance=0.0)
        ids = {c["id"]: c["status"] for c in res["checks"]}
        self.assertEqual(ids["2A-04"], accept.FAIL)


if __name__ == "__main__":
    unittest.main()
