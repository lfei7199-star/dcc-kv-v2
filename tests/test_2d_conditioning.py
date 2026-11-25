"""2D acceptance: one source, three destinations, construction semantics only.

The six checks come from the 2026-11-25 ruling. What they are allowed to claim
is deliberately narrow -- `PASS_CPU_SEMANTICS` on the construction path, and
nothing about distribution, NCCL, or a real model.

Two things this file refuses to do:

  * assert that C beats B'. Destination conditioning is a property of the code
    and can be checked; whether it *helps* is a measurement.
  * treat "logical_len survived serialisation" as evidence about RoPE phases.
    The phase check is BLOCKED until a real checkpoint run exists.

The fix is shared with test_am_properties: identical queries on every edge must
collapse, distinct queries must not -- and both are checks on identity, not on
quality.
"""

import json
import random
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import compact as C
from src.v2.cpu_ref import record as R
from src.v2.cpu_ref import two_d as T

D = 8
L = 40
BUDGET = 6
N_REF = 2 * BUDGET


def make_world(seed=11, t=L, d=D):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    v = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    return k, v


def subspace_queries(dims, n, seed, d=D):
    """Queries living in a 2-D subspace, so three destinations are separable."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        row = [0.0] * d
        for dd in dims:
            row[dd] = rnd.gauss(0, 1)
        out.append(row)
    return out


def topology(seed=11, budget=BUDGET, pool_extra=12, heldout_n=5):
    k, v = make_world(seed)
    src = T.make_source(k, v)
    n_ref = 2 * budget
    pools = [subspace_queries((2 * d, 2 * d + 1), n_ref + pool_extra, seed + 20 + d)
             for d in range(3)]
    held = [subspace_queries((2 * d, 2 * d + 1), heldout_n, seed + 90 + d)
            for d in range(3)]
    return src, pools, held


# ---------------------------------------------------------------------------


class Test2D01Collapse(unittest.TestCase):
    """Identical query sets on all three destinations -> one representation."""

    def test_identical_pools_collapse(self):
        src, _, _ = topology()
        q = subspace_queries((0, 1), N_REF + 10, 555)
        reps, recs = T.build_topology(src, [q, q, q], BUDGET, seed=3)
        verdict = T.check_collapse(reps)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])
        digests = {R.payload_digest(r) for r in reps}
        self.assertEqual(len(digests), 1)

    def test_collapse_tolerance_is_recorded(self):
        src, _, _ = topology()
        q = subspace_queries((0, 1), N_REF + 10, 555)
        reps, _ = T.build_topology(src, [q, q, q], BUDGET, seed=3)
        verdict = T.check_collapse(reps)
        for row in verdict["detail"]["per_edge"]:
            self.assertEqual(row["tolerance"], T.COLLAPSE_TOL)

    def test_collapse_detects_a_disturbance(self):
        """The check must be able to fail, otherwise it proves nothing."""
        src, _, _ = topology()
        q = subspace_queries((0, 1), N_REF + 10, 555)
        reps, _ = T.build_topology(src, [q, q, q], BUDGET, seed=3)
        reps[1]["beta"][0] += 1e-3
        self.assertEqual(T.check_collapse(reps)["status"], T.FAIL)


class Test2D02Independence(unittest.TestCase):
    """Three destinations, three own query sets, three independent builds."""

    def test_distinct_pools_give_distinct_payloads(self):
        src, pools, held = topology()
        reps, recs = T.build_topology(src, pools, BUDGET, heldouts=held, seed=3)
        verdict = T.check_independence(reps)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])
        self.assertEqual(verdict["detail"]["distinct_payloads"], 3)

    def test_same_budget_and_nref_across_edges(self):
        """Only the query set may differ, otherwise attribution is broken."""
        src, pools, held = topology()
        reps, recs = T.build_topology(src, pools, BUDGET, heldouts=held, seed=3)
        for rep, rec in zip(reps, recs):
            self.assertEqual(rep["budget"], BUDGET)
            self.assertEqual(rec["n_ref"], N_REF)
            self.assertEqual(rep["selector"], reps[0]["selector"])
            self.assertEqual(rep["beta_mode"], reps[0]["beta_mode"])
            self.assertEqual(rep["value_mode"], reps[0]["value_mode"])

    def test_identical_pools_do_not_trigger_independence(self):
        src, _, _ = topology()
        q = subspace_queries((0, 1), N_REF + 10, 777)
        reps, _ = T.build_topology(src, [q, q, q], BUDGET, seed=3)
        self.assertEqual(T.check_independence(reps)["status"], T.FAIL)


class Test2D03HeldoutIsolation(unittest.TestCase):
    """Held-out queries must never enter selection, beta or the Value fit."""

    def setUp(self):
        self.src, self.pools, self.held = topology(seed=13)

    def test_calibration_and_heldout_are_disjoint(self):
        cal, held, ci, hi = C.split_pool(self.pools[0], N_REF, 5, seed=3)
        self.assertEqual(len(cal), N_REF)
        self.assertEqual(len(held), 5)
        self.assertEqual(set(ci) & set(hi), set())
        self.assertEqual(T.check_heldout_disjoint(ci, hi)["status"], T.PASS)

    def test_disjointness_check_can_fail(self):
        self.assertEqual(T.check_heldout_disjoint([0, 1, 2], [2, 3])["status"], T.FAIL)

    def test_swapping_heldout_does_not_move_the_payload(self):
        src, pools, held = self.src, self.pools, self.held
        alt = [subspace_queries((2 * d, 2 * d + 1), 5, 700 + d) for d in range(3)]

        _, rec_a = T.build_edge(src, 0, pools[0], BUDGET, heldout_queries=held[0], seed=3)
        _, rec_b = T.build_edge(src, 0, pools[0], BUDGET, heldout_queries=alt[0], seed=3)

        verdict = T.check_heldout_invariance(rec_a, rec_b)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])
        self.assertTrue(verdict["detail"]["heldout_pools_differ"],
                        "the swap was a no-op; the check would be vacuous")

    def test_heldout_hash_is_recorded_and_distinct_from_calibration(self):
        src, pools, held = self.src, self.pools, self.held
        _, rec = T.build_edge(src, 0, pools[0], BUDGET, heldout_queries=held[0], seed=3)
        self.assertIsNotNone(rec["query_hash_heldout"])
        self.assertNotEqual(rec["query_hash_heldout"], rec["query_hash_calibration"])
        self.assertEqual(rec["n_heldout"], len(held[0]))

    def test_split_pool_never_leaks(self):
        cal, held, ci, hi = C.split_pool(self.pools[1], 10, 6, seed=11)
        self.assertEqual(set(ci) & set(hi), set())
        self.assertEqual(len(ci), 10)
        self.assertEqual(len(hi), 6)


class Test2D04Budget(unittest.TestCase):
    """Each edge's budget is checked on its own; the edges are never pooled."""

    def test_asymmetric_budgets_are_kept_per_edge(self):
        src, _, _ = topology(budget=10, pool_extra=20)
        pools = [subspace_queries((2 * d, 2 * d + 1), 3 * 10 + 10, 300 + d) for d in range(3)]
        budgets = [6, 8, 10]
        reps, recs = T.build_topology(src, pools, budgets, seed=3)

        verdict = T.check_budget(reps, recs, budgets)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])
        self.assertEqual([r["budget"] for r in reps], budgets)
        self.assertEqual([c["budget_tokens"] for c in recs], budgets)

    def test_budget_and_nref_move_together_by_default(self):
        """n_ref = 2t means a bigger budget also draws a different query sample.

        Consequence: the selected key sets are NOT nested across budgets. This is
        the concrete reason the four-budget sweep fixes n_ref = 13096 instead of
        letting it scale with t -- otherwise budget and sample size are confounded
        and the budget effect cannot be isolated.
        """
        src, _, _ = topology(budget=10, pool_extra=20)
        pool = subspace_queries((0, 1), 60, 401)
        reps6, _ = T.build_topology(src, [pool] * 3, 6, seed=3)
        reps10, _ = T.build_topology(src, [pool] * 3, 10, seed=3)
        self.assertEqual(len(reps6[0]["indices"]), 6)
        self.assertEqual(len(reps10[0]["indices"]), 10)
        self.assertFalse(set(reps6[0]["indices"]) <= set(reps10[0]["indices"]),
                         "unexpected nesting: n_ref changed with the budget, so the "
                         "score ordering must have changed as well")

    def test_nesting_holds_once_n_ref_is_fixed(self):
        """Fix the query count and the budget becomes the only variable."""
        src, _, _ = topology(budget=10, pool_extra=20)
        pool = subspace_queries((0, 1), 60, 401)
        sets = {}
        for b in (6, 8, 10):
            rep, _ = T.build_edge(src, 0, pool, b, n_ref=40, seed=3)
            sets[b] = rep["indices"]
        self.assertTrue(set(sets[6]) <= set(sets[8]) <= set(sets[10]))
        self.assertLess(len(sets[6]), len(sets[8]))

    def test_sum_of_budgets_is_not_any_edge_budget(self):
        src, pools, _ = topology()
        budgets = [BUDGET, BUDGET + 1, BUDGET + 2]
        reps, recs = T.build_topology(src, pools, budgets, seed=3)
        verdict = T.check_budget(reps, recs, budgets)
        self.assertEqual(verdict["detail"]["sum_of_budgets"], sum(budgets))
        for rep, b in zip(reps, budgets):
            self.assertEqual(rep["budget"], b)
            self.assertNotEqual(rep["budget"], sum(budgets))

    def test_payload_is_labelled_per_edge(self):
        src, pools, _ = topology()
        _, recs = T.build_topology(src, pools, BUDGET, seed=3)
        for rec in recs:
            self.assertEqual(rec["payload"]["counted_for"], "this edge only")
            self.assertNotIn("total_bytes", rec["payload"])
            self.assertEqual(rec["payload"]["t"], rec["budget_tokens"])

    def test_check_budget_catches_a_tampered_record(self):
        src, pools, _ = topology()
        reps, recs = T.build_topology(src, pools, BUDGET, seed=3)
        recs[1]["budget_tokens"] = BUDGET + 1
        self.assertEqual(T.check_budget(reps, recs, [BUDGET] * 3)["status"], T.FAIL)

    def test_pool_smaller_than_nref_is_rejected(self):
        src, _, _ = topology()
        with self.assertRaises(ValueError):
            T.build_edge(src, 0, subspace_queries((0, 1), 3, 9), BUDGET, seed=3)


class Test2D05Record(unittest.TestCase):
    """Every edge emits a complete, self-describing record."""

    def test_all_edges_validate(self):
        src, pools, held = topology(seed=17)
        _, recs = T.build_topology(src, pools, BUDGET, heldouts=held, seed=3)
        verdict = T.check_record_complete(recs)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])

    def test_record_carries_the_required_payload(self):
        src, pools, held = topology(seed=19)
        _, rec = T.build_edge(src, 0, pools[0], BUDGET, heldout_queries=held[0], seed=3)
        for key in ("query_hash_calibration", "selected_indices", "beta", "values",
                    "payload", "payload_digest", "diagnostics", "solver"):
            self.assertIn(key, rec)
        self.assertEqual(len(rec["beta"]), BUDGET)
        self.assertEqual(len(rec["values"]), BUDGET)
        self.assertEqual(len(rec["selected_indices"]), BUDGET)

    def test_records_of_different_edges_differ(self):
        src, pools, held = topology(seed=23)
        _, recs = T.build_topology(src, pools, BUDGET, heldouts=held, seed=3)
        self.assertEqual(len({r["payload_digest"] for r in recs}), 3)
        self.assertEqual(len({r["query_hash_calibration"] for r in recs}), 3)

    def test_completeness_check_can_fail(self):
        src, pools, _ = topology(seed=29)
        _, recs = T.build_topology(src, pools, BUDGET, seed=3)
        del recs[2]["payload"]
        self.assertEqual(T.check_record_complete(recs)["status"], T.FAIL)


class Test2D06PositionMetadata(unittest.TestCase):
    """logical_len and the global span survive a wire round trip."""

    def test_roundtrip_preserves_position_fields(self):
        src, pools, _ = topology(seed=31)
        reps, _ = T.build_topology(src, pools, BUDGET, seed=3)
        wire = json.loads(json.dumps(T.to_wire(reps[0])))
        back = T.from_wire(wire)
        verdict = T.check_position_roundtrip(wire, back)
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"])
        self.assertEqual(back["logical_len"], L)
        self.assertEqual(back["position_range"], [0, L])

    def test_roundtrip_preserves_beta_and_values(self):
        src, pools, _ = topology(seed=37)
        reps, _ = T.build_topology(src, pools, BUDGET, seed=3)
        wire = json.loads(json.dumps(T.to_wire(reps[0])))
        back = T.from_wire(wire)
        self.assertEqual(back["beta"], reps[0]["beta"])
        self.assertEqual(back["indices"], reps[0]["indices"])
        for a, b in zip(back["v_sel"], reps[0]["v_sel"]):
            self.assertEqual(a, b)

    def test_dropping_the_field_is_detected(self):
        src, pools, _ = topology(seed=41)
        reps, _ = T.build_topology(src, pools, BUDGET, seed=3)
        wire = json.loads(json.dumps(T.to_wire(reps[0])))
        back = T.from_wire(wire)
        back["logical_len"] = None
        self.assertEqual(T.check_position_roundtrip(wire, back)["status"], T.FAIL)

    def test_source_position_range_is_carried_into_the_compact_object(self):
        k, v = make_world(seed=43)
        src = T.make_source(k, v, logical_len=1000, pos_start=200)
        rep, rec = T.build_edge(src, 0, subspace_queries((0, 1), N_REF + 5, 51),
                                BUDGET, seed=3)
        self.assertEqual(rep["position_range"], [200, 200 + L])
        self.assertEqual(rec["logical_len"], 1000)
        self.assertEqual(rec["position_range"], [200, 200 + L])
        R.validate(rec)  # span <= logical_len holds

    def test_rope_phase_is_blocked_not_passed(self):
        """Metadata travelling is not the same as the phase being right."""
        verdict = T.check_rope_phase()
        self.assertEqual(verdict["status"], T.BLOCKED)
        self.assertNotEqual(verdict["status"], T.PASS)


class TestVerdictFolding(unittest.TestCase):
    """The overall verdict can never be upgraded to a bare PASS."""

    def test_cpu_run_is_scoped(self):
        src, pools, held = topology(seed=47)
        reps, recs = T.build_topology(src, pools, BUDGET, heldouts=held, seed=3)
        wire = json.loads(json.dumps(T.to_wire(reps[0])))
        checks = [
            T.check_collapse([reps[0], reps[0]]),
            T.check_independence(reps),
            T.check_budget(reps, recs, [BUDGET] * 3),
            T.check_record_complete(recs),
            T.check_position_roundtrip(wire, T.from_wire(wire)),
            T.check_rope_phase(),
        ]
        out = T.evaluate(checks)
        self.assertEqual(out["overall"], T.PASS)
        self.assertNotEqual(out["overall"], "PASS")
        self.assertIn("2D-06b", out["blocked_requires_gpu"])
        self.assertIn("CPU", out["scope"])

    def test_any_failure_dominates(self):
        good = {"id": "x", "status": T.PASS, "detail": {}}
        bad = {"id": "y", "status": T.FAIL, "detail": {}}
        out = T.evaluate([good, bad])
        self.assertEqual(out["overall"], T.FAIL)


if __name__ == "__main__":
    unittest.main()
