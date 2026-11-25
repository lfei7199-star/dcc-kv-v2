"""The shared record: one envelope, staged diagnostics, and the five things it
must refuse.

The ruling of 2026-11-25 named five negative cases. Each one has a class here:

  * 2A mis-filling beta diagnostics            -> TestStage2ANegative
  * 2D missing beta or Value on any edge       -> TestStage2DNegative
  * three edges pooled into a per-edge payload -> TestDoubleCounting
  * schema_version disagreeing with the record -> TestSchemaVersion
  * CPU-only record returning hardware PASS    -> TestCpuCeiling

Two more properties are asserted because the whole design rests on them:
NOT_APPLICABLE and MISSING are different states (TestStatusVocabulary), and the
2A acceptor does not import the compression solver (TestImportIsolation).

Nothing here asserts that C beats B'. Destination conditioning is an
implementation property; whether it helps is a measurement, not a gate.
"""

import copy
import json
import os
import random
import subprocess
import sys
import unittest

import _bootstrap  # noqa: F401

from src.v2.cpu_ref import accept as A
from src.v2.cpu_ref import adapter_2a as A2
from src.v2.cpu_ref import adapter_compact as AC
from src.v2.cpu_ref import envelope as E
from src.v2.cpu_ref import sp_attention as SP
from src.v2.cpu_ref import two_d as T

D = 8
L = 48
WS = 4
BUDGETS = [6, 8, 10]
N_REF = [2 * b for b in BUDGETS]
TOL = 1e-9

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_CACHE = {}


def _world(seed=3, n=L, d=D):
    rnd = random.Random(seed)
    return ([[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)],
            [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)])


def _pool(dims, n, seed, d=D):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        row = [0.0] * d
        for dd in dims:
            row[dd] = rnd.gauss(0, 1)
        out.append(row)
    return out


def sp_run(all_gather=False, world=WS):
    q, k = _world()
    _, v = _world(4)
    ref = SP.dense_attention(q, k, v)
    sp = SP.sp_forward(q, k, v, world, all_gather=all_gather)
    return sp, ref


def env_2a(**kw):
    """Fresh 2A envelope (verdicts are recomputed, so never mutated in place)."""
    sp, ref = sp_run()
    return A2.from_2a(sp, ref, TOL, experiment_id="2a-test", **kw)


def topology_2d():
    """One source, three destinations with their own pools and budgets."""
    if "2d" in _CACHE:
        return _CACHE["2d"]
    k, v = _world(seed=11)
    src = T.make_source(k, v, logical_len=1000, pos_start=64)
    pools = [_pool((2 * d, 2 * d + 1), 2 * max(BUDGETS) + 8, 60 + d)
             for d in range(3)]
    reps, recs = T.build_topology(src, pools, BUDGETS, seed=5)
    env = AC.from_2d(recs, experiment_id="2d-test", source="source",
                     destinations=[0, 1, 2])
    _CACHE["2d"] = (src, pools, reps, recs, env)
    return _CACHE["2d"]


def env_2d():
    return topology_2d()[4]


def env_2c():
    if "2c" not in _CACHE:
        _, _, _, recs, _ = topology_2d()
        _CACHE["2c"] = AC.from_2c(recs[0], experiment_id="2c-test")
    return _CACHE["2c"]


def thaw(env):
    """Deep copy through JSON: the same path a stored record takes on re-read."""
    return json.loads(json.dumps(env))


def problems(env):
    return E.validate(env, strict=False)


class _Base(unittest.TestCase):
    def assertProblem(self, env, needle):
        ps = problems(env)
        self.assertTrue(any(needle in p for p in ps),
                        "expected %r among problems:\n  %s"
                        % (needle, "\n  ".join(ps) or "<none>"))

    def assertClean(self, env):
        ps = problems(env)
        self.assertEqual(ps, [], "expected a clean record, got:\n  %s" % ps)


# ---------------------------------------------------------------------------
# schema version
# ---------------------------------------------------------------------------

class TestSchemaVersion(_Base):
    def test_the_three_records_share_one_version(self):
        self.assertEqual(env_2a()["schema_version"], E.SCHEMA_VERSION)
        self.assertEqual(env_2c()["schema_version"], E.SCHEMA_VERSION)
        self.assertEqual(env_2d()["schema_version"], E.SCHEMA_VERSION)

    def test_roundtrip_keeps_every_field(self):
        back = E.from_json(E.to_json(env_2d()))
        self.assertEqual(back, thaw(env_2d()))

    def test_foreign_version_is_refused_on_read(self):
        env = thaw(env_2d())
        env["schema_version"] = "dcckv-experiment-record/0"
        with self.assertRaises(E.SchemaMismatchError):
            E.from_json(json.dumps(env))

    def test_foreign_version_is_refused_on_validate(self):
        env = thaw(env_2a())
        env["schema_version"] = "dcckv-edge-record/1"
        with self.assertRaises(E.SchemaMismatchError):
            E.validate(env)

    def test_version_is_not_coerced(self):
        """A float or a prefix is a different record, not a formatting quirk."""
        env = thaw(env_2c())
        env["schema_version"] = 1
        with self.assertRaises(E.SchemaMismatchError):
            E.from_json(json.dumps(env))


# ---------------------------------------------------------------------------
# NOT_APPLICABLE vs MISSING
# ---------------------------------------------------------------------------

class TestStatusVocabulary(_Base):
    def test_the_three_statuses_are_distinct(self):
        self.assertEqual(len({E.APPLICABLE, E.NOT_APPLICABLE, E.MISSING}), 3)

    def test_an_absent_block_reads_as_missing(self):
        self.assertEqual(E.classify_block(None), E.MISSING)
        self.assertEqual(E.classify_block({}), E.MISSING)

    def test_2a_compact_is_not_applicable(self):
        env = env_2a()
        self.assertEqual(env["compact_diagnostics"]["status"], E.NOT_APPLICABLE)
        self.assertIn("never selects keys", env["compact_diagnostics"]["reason"])
        self.assertClean(env)

    def test_2a_declaring_compact_missing_is_rejected(self):
        """2A's absence is 'does not exist', not 'was not produced'."""
        env = thaw(env_2a())
        env["compact_diagnostics"] = {"status": E.MISSING, "reason": "lost it"}
        self.assertProblem(env, "must be NOT_APPLICABLE")

    def test_2d_declaring_compact_missing_is_rejected(self):
        """And on 2D the same word means a defect, so it must not be tolerated."""
        env = thaw(env_2d())
        env["compact_diagnostics"] = {"status": E.MISSING, "reason": "lost it"}
        self.assertProblem(env, "incomplete evidence")

    def test_2d_sp_missing_is_a_gap_not_a_defect(self):
        env = env_2d()
        self.assertEqual(env["sp_structure"]["status"], E.MISSING)
        self.assertClean(env)
        self.assertEqual(env["verdict"]["overall"], E.BLOCKED)
        self.assertEqual(env["verdict"]["partial"]["compact"], E.PASS_CPU_SEMANTICS)


# ---------------------------------------------------------------------------
# negative case 1 -- 2A carrying beta diagnostics
# ---------------------------------------------------------------------------

class TestStage2ANegative(_Base):
    def test_2a_with_beta_numbers_is_rejected(self):
        env = thaw(env_2a())
        env["compact_diagnostics"]["beta_binding"] = {
            "status": E.APPLICABLE, "lo": -3.0, "hi": 3.0, "bind_tol": 1e-6,
            "bind_hi_frac": 0.42, "bind_lo_frac": 0.0}
        self.assertProblem(env, "never fit these quantities")

    def test_2a_with_mass_residual_is_rejected(self):
        env = thaw(env_2a())
        env["compact_diagnostics"]["mass_residual"] = {
            "status": E.APPLICABLE, "normalized": 0.31,
            "aggregation": "pooled_over_all_calibration_queries"}
        self.assertProblem(env, "never fit these quantities")

    def test_2a_flipped_to_applicable_is_rejected(self):
        env = thaw(env_2a())
        env["compact_diagnostics"]["status"] = E.APPLICABLE
        self.assertProblem(env, "must be NOT_APPLICABLE")

    def test_2a_with_beta_tensors_is_rejected(self):
        env = thaw(env_2a())
        env["compact_diagnostics"]["beta"] = [0.1, 0.2]
        self.assertProblem(env, "never fit these quantities")

    def test_2a_cannot_claim_a_compression_result(self):
        env = thaw(env_2a())
        env["scope"]["claims"] = list(env["scope"]["claims"]) + [
            "shared_query_construction"]
        self.assertProblem(env, "not in the 2A vocabulary")

    def test_2a_cannot_claim_nccl(self):
        env = thaw(env_2a())
        env["scope"]["claims"] = list(env["scope"]["claims"]) + ["nccl_transport"]
        self.assertProblem(env, "needs measured hardware evidence")

    def test_2a_without_an_sp_record_is_rejected(self):
        env = thaw(env_2a())
        env["sp_structure"] = {"status": E.MISSING, "reason": "no run"}
        self.assertProblem(env, "must produce its SP record")

    def test_2a_with_destinations_is_rejected(self):
        env = thaw(env_2a())
        env["topology"]["destinations"] = [0, 1, 2]
        self.assertProblem(env, "2A has no destination set")

    def test_2a_numeric_tolerance_source_is_declared(self):
        env = env_2a()
        num = env["sp_structure"]["numerical"]
        self.assertEqual(num["tolerance_source"], "test_value")
        self.assertIsNotNone(num["rel_l2"])

    def test_2a_has_no_compact_check_rows(self):
        env = env_2a()
        groups = {c["group"] for c in env["verdict"]["checks"]}
        self.assertNotIn("compact", groups)

    def test_all_gather_arm_fails_the_same_checker_and_the_record_says_so(self):
        sp, ref = sp_run(all_gather=True)
        res = A.evaluate(sp, ref, TOL)
        self.assertEqual(res["overall"], A.FAIL)
        env = A2.from_2a(sp, ref, TOL, experiment_id="2a-neg",
                         accept_result=res)
        self.assertEqual(env["verdict"]["overall"], E.FAIL)
        self.assertIn("sp.no_full_copy", env["verdict"]["failed"])


# ---------------------------------------------------------------------------
# negative case 2 -- 2D missing a block on one edge
# ---------------------------------------------------------------------------

class TestStage2DNegative(_Base):
    def test_missing_beta_binding_on_one_edge_is_rejected(self):
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][1]["beta_binding"]
        self.assertProblem(env, "entries[1] lacks")

    def test_missing_value_design_on_one_edge_is_rejected(self):
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][0]["value_design"]
        self.assertProblem(env, "value_design")

    def test_missing_mass_residual_on_one_edge_is_rejected(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][2]["mass_residual"] = {
            "status": E.MISSING, "reason": "not computed"}
        self.assertProblem(env, "mass_residual")

    def test_missing_payload_is_rejected(self):
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][0]["payload"]
        self.assertProblem(env, "payload")

    def test_missing_key_selection_is_rejected(self):
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][0]["key_selection"]
        self.assertProblem(env, "key_selection")

    def test_entry_count_must_match_destinations(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"].pop()
        self.assertProblem(env, "each Source->Destination edge needs its own block")

    def test_unknown_destination_is_rejected(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][0]["edge"]["destination"] = 7
        self.assertProblem(env, "not in topology.destinations")

    def test_duplicate_destination_is_rejected(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][1]["edge"]["destination"] = 0
        self.assertProblem(env, "same destination")

    def test_beta_and_value_must_name_different_problems(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][0]["value_design"]["problem"] = \
            env["compact_diagnostics"]["entries"][0]["beta_design"]["problem"]
        self.assertProblem(env, "they are different systems")

    def test_a_rank_without_its_method_is_rejected(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][0]["beta_design"]["method"] = None
        self.assertProblem(env, "not comparable")

    def test_empty_box_is_rejected(self):
        env = thaw(env_2d())
        bb = env["compact_diagnostics"]["entries"][0]["beta_binding"]
        bb["lo"], bb["hi"] = 3.0, -3.0
        self.assertProblem(env, "box is empty")

    def test_destination_semantics_must_be_applicable(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["destination_semantics"] = {
            "status": E.NOT_APPLICABLE, "reason": "skipped"}
        self.assertProblem(env, "destination_semantics must be")

    def test_acceptance_fails_when_an_edge_lacks_beta(self):
        """Rejected by the validator is not enough: 2D-07 must go red too."""
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][1]["beta_binding"]
        verdict = T.check_envelope(env)
        self.assertEqual(verdict["status"], T.FAIL)
        self.assertTrue(verdict["detail"]["problems"])

    def test_acceptance_fails_when_an_edge_lacks_value(self):
        env = thaw(env_2d())
        del env["compact_diagnostics"]["entries"][2]["value_design"]
        self.assertEqual(T.check_envelope(env)["status"], T.FAIL)


# ---------------------------------------------------------------------------
# negative case 3 -- pooled payload
# ---------------------------------------------------------------------------

class TestDoubleCounting(_Base):
    def _expected(self, env):
        return [E.compact_payload_bytes(p["t"], p["dk"], p["dv"], p["dtype_bytes"])
                for p in (e["payload"] for e in env["compact_diagnostics"]["entries"])]

    def test_a_valid_record_has_no_payload_problem(self):
        env = env_2d()
        self.assertEqual(E.check_no_double_count(env["compact_diagnostics"]["entries"]),
                         [])

    def test_the_three_edge_total_repeated_per_edge_is_caught(self):
        """The merge mistake: sum the payloads, then write the sum into each edge."""
        env = thaw(env_2d())
        total = sum(self._expected(env))
        for p in (e["payload"] for e in env["compact_diagnostics"]["entries"]):
            p["bytes"] = total
        ps = E.check_no_double_count(env["compact_diagnostics"]["entries"])
        self.assertTrue(any("sum of all 3 edges" in p for p in ps), ps)
        self.assertTrue(any("own tensors give" in p for p in ps), ps)
        self.assertProblem(env, "counted once per edge")

    def test_a_single_inflated_payload_is_caught(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][2]["payload"]["bytes"] *= 2
        self.assertProblem(env, "own tensors give")

    def test_counted_for_must_stay_per_edge(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][0]["payload"]["counted_for"] = \
            "all edges of this source"
        self.assertProblem(env, "must be 'this edge only'")

    def test_a_pooling_key_inside_a_payload_is_forbidden(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"][0]["payload"]["total_bytes"] = 999
        self.assertProblem(env, "pooling key")

    def test_a_compact_level_total_is_forbidden(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["total_payload"] = {"bytes": 2616}
        self.assertProblem(env, "no cross-edge total")

    def test_acceptance_fails_on_a_pooled_payload(self):
        env = thaw(env_2d())
        total = sum(self._expected(env))
        for p in (e["payload"] for e in env["compact_diagnostics"]["entries"]):
            p["bytes"] = total
        self.assertEqual(T.check_envelope(env)["status"], T.FAIL)

    def test_unit_level_pooling_detection(self):
        entries = [
            {"payload": {"t": 4, "dk": 8, "dv": 8, "dtype_bytes": 8,
                         "counted_for": "this edge only", "bytes": 4 * 8 * 8 * 2 + 4 * 8
                         + 4 * 4 + 32}},
            {"payload": {"t": 6, "dk": 8, "dv": 8, "dtype_bytes": 8,
                         "counted_for": "this edge only", "bytes": 6 * 8 * 8 * 2 + 6 * 8
                         + 6 * 4 + 32}},
        ]
        self.assertEqual(E.check_no_double_count(entries), [])
        pooled = sum(e["payload"]["bytes"] for e in entries)
        for e in entries:
            e["payload"]["bytes"] = pooled
        self.assertTrue(E.check_no_double_count(entries))

    def test_budgets_stay_per_edge_in_the_record(self):
        env = env_2d()
        got = [e["budget_tokens"] for e in env["compact_diagnostics"]["entries"]]
        self.assertEqual(got, BUDGETS)
        self.assertNotIn(sum(BUDGETS), got)


# ---------------------------------------------------------------------------
# negative case 5 -- CPU cannot reach PASS
# ---------------------------------------------------------------------------

class TestCpuCeiling(_Base):
    def test_fold_caps_structural_evidence(self):
        ok = {"id": "x", "group": "compact", "status": E.PASS}
        out = E.fold_verdict([ok], evidence_level=E.STRUCTURAL)
        self.assertEqual(out["overall"], E.PASS_CPU_SEMANTICS)
        self.assertEqual(out["ceiling"], E.PASS_CPU_SEMANTICS)

    def test_measured_evidence_may_reach_pass(self):
        ok = {"id": "x", "group": "compact", "status": E.PASS}
        self.assertEqual(E.fold_verdict([ok], evidence_level=E.MEASURED)["overall"],
                         E.PASS)

    def test_a_hand_edited_pass_is_refused(self):
        env = thaw(env_2d())
        env["verdict"]["overall"] = E.PASS
        self.assertProblem(env, "the checks fold to")

    def test_a_hand_edited_pass_also_breaks_the_ceiling(self):
        env = thaw(env_2d())
        env["verdict"]["overall"] = E.PASS
        env["verdict"]["ceiling"] = E.PASS
        env["verdict"]["blocked_requires_gpu"] = []
        self.assertProblem(env, "ceiling is")

    def test_dropping_the_blocked_rows_is_refused(self):
        env = thaw(env_2d())
        env["verdict"]["checks"] = [c for c in env["verdict"]["checks"]
                                    if c["status"] != E.BLOCKED]
        self.assertProblem(env, "cannot fold to a pass")

    def test_2a_never_returns_pass(self):
        self.assertNotEqual(env_2a()["verdict"]["overall"], E.PASS)

    def test_2d_never_returns_pass(self):
        self.assertNotEqual(env_2d()["verdict"]["overall"], E.PASS)

    def test_measured_nccl_needs_measured_evidence(self):
        env = thaw(env_2a())
        env["sp_structure"]["communication"]["evidence"] = "measured_nccl"
        self.assertProblem(env, "not measured_hardware")

    def test_accept_refuses_a_pass_envelope(self):
        env = thaw(env_2a())
        env["verdict"]["overall"] = E.PASS
        sp, ref = sp_run()
        res = A.evaluate(sp, ref, TOL)
        with self.assertRaises(ValueError):
            A.attach_envelope(res, env)

    def test_accept_refuses_a_foreign_stage(self):
        sp, ref = sp_run()
        res = A.evaluate(sp, ref, TOL)
        with self.assertRaises(ValueError):
            A.attach_envelope(res, env_2d())


# ---------------------------------------------------------------------------
# the 2A acceptor must not depend on the solver
# ---------------------------------------------------------------------------

class TestImportIsolation(unittest.TestCase):
    def _modules_after(self, statement):
        code = ("import sys; sys.path.insert(0, %r); %s; "
                "print(sorted(m for m in sys.modules if 'cpu_ref' in m))"
                % (_REPO, statement))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()

    def test_2a_adapter_does_not_load_the_solver(self):
        loaded = self._modules_after("import src.v2.cpu_ref.adapter_2a")
        self.assertIn("adapter_2a", loaded)
        self.assertNotIn("compact", loaded)
        self.assertNotIn("record", loaded)

    def test_accept_does_not_load_the_solver(self):
        loaded = self._modules_after("import src.v2.cpu_ref.accept")
        self.assertNotIn("compact", loaded)
        self.assertNotIn("record", loaded)

    def test_the_compact_adapter_does_load_it(self):
        """The isolation above must be a property of the module, not of Python."""
        loaded = self._modules_after("import src.v2.cpu_ref.adapter_compact")
        self.assertIn("compact", loaded)


# ---------------------------------------------------------------------------
# the 2A record really describes the 2A run
# ---------------------------------------------------------------------------

class Test2AEnvelope(_Base):
    def test_every_matrix_row_has_a_status(self):
        env = env_2a()
        sp = env["sp_structure"]
        for sub in ("rank_topology", "owner_kv", "remote_staging", "no_full_copy",
                    "communication"):
            self.assertEqual(sp[sub]["status"], E.APPLICABLE, sub)

    def test_residency_and_staging_are_separate_blocks(self):
        env = env_2a()
        own = env["sp_structure"]["owner_kv"]
        stg = env["sp_structure"]["remote_staging"]
        self.assertIn("dense KV resident", own["counted_for"])
        self.assertIn("transiently", stg["counted_for"])
        self.assertEqual([r["remote_block_count"] for r in stg["per_rank"]],
                         [0, 1, 2, 3])
        ratio = own["per_rank"][0]["ratio_of_full"]
        self.assertAlmostEqual(ratio, 1.0 / WS, places=6)

    def test_gqa_is_declared_inapplicable_for_the_reference(self):
        env = env_2a()
        self.assertEqual(env["sp_structure"]["gqa_repeat"]["status"],
                         E.NOT_APPLICABLE)

    def test_synthetic_workload_is_declared(self):
        env = env_2a()
        self.assertIs(env["workload"]["model"]["synthetic"], True)
        self.assertFalse(env["workload"]["position_ids"]["rope_phase_verified"])

    def test_a_real_model_needs_a_revision(self):
        env = thaw(env_2a())
        env["workload"]["model"] = {"synthetic": False, "name": "Qwen2.5-7B-Instruct",
                                    "revision": None}
        self.assertProblem(env, "HF 'main' is not a revision")

    def test_provenance_pairs_vcs_and_commit(self):
        env = thaw(env_2a())
        env["provenance"]["commit"] = "0" * 40
        self.assertProblem(env, "an absent repository has no commit")
        env2 = thaw(env_2a())
        env2["provenance"]["vcs"] = "git"
        self.assertProblem(env2, "not a 40-hex hash")

    def test_provenance_names_the_artefacts(self):
        env = env_2a()
        paths = [f["path"] for f in env["provenance"]["files"]]
        self.assertIn("accept.py", paths)
        self.assertIn("sp_attention.py", paths)
        for f in env["provenance"]["files"]:
            self.assertIn(f["path"], paths)
            self.assertEqual(len(f["sha256"]), 64)

    def test_2a_record_attaches_to_the_acceptance_result(self):
        sp, ref = sp_run()
        res = A.evaluate(sp, ref, TOL)
        env = A2.from_2a(sp, ref, TOL, experiment_id="2a-attach",
                         accept_result=res)
        out = A.attach_envelope(res, env)
        self.assertEqual(out["record_summary"]["stage"], "2A")
        self.assertEqual(out["record"]["verdict"]["overall"], res["overall"])


# ---------------------------------------------------------------------------
# the 2D record: per edge, cited but not borrowed
# ---------------------------------------------------------------------------

class Test2DEnvelope(_Base):
    def test_valid_record_passes_the_interface_check(self):
        verdict = T.check_envelope(env_2d())
        self.assertEqual(verdict["status"], T.PASS, verdict["detail"]["problems"])
        self.assertEqual(verdict["detail"]["entries"], 3)

    def test_each_edge_keeps_its_own_budget_and_pair(self):
        env = env_2d()
        entries = env["compact_diagnostics"]["entries"]
        self.assertEqual([e["budget_tokens"] for e in entries], BUDGETS)
        self.assertEqual([e["key_selection"]["n_ref"] for e in entries], N_REF)
        conds = [e["beta_design"]["cond"] for e in entries]
        self.assertEqual(len(set(conds)), 3)

    def test_destination_semantics_records_three_query_sets(self):
        sem = env_2d()["compact_diagnostics"]["destination_semantics"]
        self.assertTrue(sem["per_destination_queries"])
        self.assertFalse(sem["shared_query_set"])
        self.assertEqual(len(set(sem["query_hashes"])), 3)

    def test_a_shared_query_set_must_say_so(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["destination_semantics"]["shared_query_set"] = True
        self.assertProblem(env, "the recorded hashes differ")

    def test_position_metadata_travels_without_claiming_the_phase(self):
        env = env_2d()
        pos = env["compact_diagnostics"]["entries"][0]["position"]
        self.assertEqual(pos["logical_len"], 1000)
        self.assertEqual(pos["span"], [64, 64 + L])
        self.assertFalse(pos["rope_phase_verified"])

    def test_an_upstream_reference_cannot_fill_a_block(self):
        """The realistic shape: 2D's owner_kv is filled from 2A's measurement."""
        env = thaw(env_2d())
        env["upstream_refs"] = [{"stage": "2A", "experiment_id": "2a-test",
                                 "verdict": E.BLOCKED}]
        env["sp_structure"]["owner_kv"] = {"status": E.APPLICABLE,
                                           "from_upstream": "2a-test",
                                           "per_rank": []}
        self.assertProblem(env, "must not be padded with numbers")

    def test_a_sp_row_cannot_be_borrowed_while_sp_is_applicable(self):
        env = thaw(env_2a())
        env["sp_structure"]["owner_kv"]["upstream_value"] = "2a-run-7"
        self.assertProblem(env, "cannot supply this stage's own measurement")

    def test_citing_2a_does_not_change_the_verdict(self):
        env = env_2d()
        cited = thaw(env)
        cited["upstream_refs"] = [{"stage": "2A", "experiment_id": "2a-test",
                                   "verdict": E.BLOCKED,
                                   "note": "reference only"}]
        self.assertClean(cited)
        self.assertEqual(cited["verdict"]["overall"], env["verdict"]["overall"])

    def test_hardware_evidence_row_is_blocked(self):
        verdict = T.check_hardware_evidence(env_2d())
        self.assertEqual(verdict["status"], T.BLOCKED)
        self.assertEqual(verdict["detail"]["sp_status"], E.MISSING)

    def test_2d_without_a_source_is_rejected(self):
        env = thaw(env_2d())
        env["topology"]["source"] = None
        self.assertProblem(env, "needs an explicit source label")


# ---------------------------------------------------------------------------
# the requirement matrix itself
# ---------------------------------------------------------------------------

class TestRequirementTable(unittest.TestCase):
    def test_sp_rows_are_required_on_2a(self):
        for row in ("rank_topology", "owner_kv_resident", "remote_staging",
                    "no_full_copy"):
            self.assertEqual(E.REQUIREMENT_TABLE[row]["2A"], E.REQUIRED, row)

    def test_communication_is_evidence_classed_everywhere(self):
        for stage in ("2A", "2C", "2D"):
            self.assertEqual(E.REQUIREMENT_TABLE["communication"][stage],
                             E.REQUIRED_EVIDENCE, stage)

    def test_beta_binding_is_forbidden_on_2a_and_required_on_2d(self):
        self.assertEqual(E.REQUIREMENT_TABLE["beta_binding"]["2A"], E.FORBIDDEN)
        self.assertEqual(E.REQUIREMENT_TABLE["beta_binding"]["2D"], E.REQUIRED)

    def test_rank_cond_mass_are_forbidden_on_2a(self):
        for row in ("beta_value_rank_cond", "mass_residual"):
            self.assertEqual(E.REQUIREMENT_TABLE[row]["2A"], E.FORBIDDEN, row)
            self.assertEqual(E.REQUIREMENT_TABLE[row]["2D"], E.REQUIRED, row)

    def test_destination_semantics_only_on_2d(self):
        self.assertEqual(E.REQUIREMENT_TABLE["destination_semantics"]["2A"],
                         E.FORBIDDEN)
        self.assertEqual(E.REQUIREMENT_TABLE["destination_semantics"]["2C"],
                         E.FORBIDDEN)
        self.assertEqual(E.REQUIREMENT_TABLE["destination_semantics"]["2D"],
                         E.REQUIRED)

    def test_every_row_is_reachable_from_the_json_shape(self):
        covered = set(E.SP_ROWS) | set(E.COMPACT_ROWS) | {"communication"}
        self.assertEqual(set(E.REQUIREMENT_TABLE), covered)


# ---------------------------------------------------------------------------
# one format, three stages
# ---------------------------------------------------------------------------

class TestSharedShape(unittest.TestCase):
    def test_shell_keys_are_identical(self):
        shells = [set(e.keys()) for e in (env_2a(), env_2c(), env_2d())]
        self.assertEqual(shells[0], shells[1])
        self.assertEqual(shells[1], shells[2])

    def test_provenance_topology_workload_share_their_vocabulary(self):
        a, c, d = env_2a(), env_2c(), env_2d()
        for key in ("vcs", "commit", "files", "config_sha", "environment"):
            for env in (a, c, d):
                self.assertIn(key, env["provenance"])
        for key in ("world_size", "ranks", "source", "destinations", "rank_map"):
            for env in (a, c, d):
                self.assertIn(key, env["topology"])
        for key in ("model", "seq_len", "position_ids", "dtype"):
            for env in (a, c, d):
                self.assertIn(key, env["workload"])

    def test_2c_and_2d_entries_carry_the_same_blocks(self):
        e2c = env_2c()["compact_diagnostics"]["entries"][0]
        e2d = env_2d()["compact_diagnostics"]["entries"][0]
        for block in E.ENTRY_BLOCKS:
            self.assertIn(block, e2c)
            self.assertIn(block, e2d)
        self.assertIsNone(e2c["edge"])
        self.assertEqual(e2d["edge"], {"source": "source", "destination": 0})

    def test_2c_has_no_destination_rows(self):
        verdict = env_2c()["verdict"]
        self.assertEqual(verdict["partial"]["destination"], E.NOT_APPLICABLE)

    def test_summary_is_a_reading_aid_not_a_verdict(self):
        s = E.summary(env_2d())
        self.assertEqual(s["stage"], "2D")
        self.assertEqual(s["compact_status"], E.APPLICABLE)
        self.assertEqual(s["sp_status"], E.MISSING)

    def test_mutating_a_copy_does_not_touch_the_cached_record(self):
        env = thaw(env_2d())
        env["compact_diagnostics"]["entries"].pop()
        self.assertEqual(len(env_2d()["compact_diagnostics"]["entries"]), 3)
        self.assertIsNot(copy.deepcopy(env), env)


if __name__ == "__main__":
    unittest.main()
