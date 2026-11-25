"""Run the 2D CPU closed loop and write results/cpu/2d_closure.json.

1 source x 3 destinations, asymmetric budgets on purpose: an equal-budget
topology cannot show that the per-edge budget is respected rather than pooled.

Nothing here is a quality result. The output is a construction-semantics verdict
plus the per-edge records, and the overall status is PASS_CPU_SEMANTICS at best.

Two stage records are emitted alongside the checks
--------------------------------------------------
`records["2A"]` and `records["2D"]` are the same kind of object: one envelope
(see envelope.py) with a shared shell and stage-specific diagnostics. The 2A
record carries a real dense-SP run over the same tensors and says
`compact_diagnostics = NOT_APPLICABLE`; the 2D record carries one entry per edge
and says `sp_structure = MISSING` with a reason. Neither borrows the other's
numbers: 2A has no beta to lend, and 2D did not measure a transport.

The two verdicts in this file are different objects and both are printed:
  verdict        construction semantics (2D-01..06), PASS_CPU_SEMANTICS at best
  records[*].verdict   the stage verdict, BLOCKED while SP is unmeasured
Reporting both is the point -- collapsing them would either overclaim the
construction or understate it.
"""

import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import accept as A          # noqa: E402
from src.v2.cpu_ref import adapter_2a as A2     # noqa: E402
from src.v2.cpu_ref import adapter_compact as AC  # noqa: E402
from src.v2.cpu_ref import compact as C          # noqa: E402
from src.v2.cpu_ref import envelope as E         # noqa: E402
from src.v2.cpu_ref import q8                    # noqa: E402
from src.v2.cpu_ref import record as R          # noqa: E402
from src.v2.cpu_ref import sp_attention as SP   # noqa: E402
from src.v2.cpu_ref import two_d as T           # noqa: E402

D = 8
L = 96
BUDGETS = [6, 8, 10]        # deliberately unequal
N_HELDOUT = 6
SEED = 20261125
WORLD_SIZE = 4

# Q8 is frozen at rel-L2 <= 1e-5 with source=preregistered, but this run is a
# float64 CPU reference compared against an in-process dense pass, so it cites a
# test value. The record says so; promoting 1e-9 to "preregistered" is refused
# by q8.validate_policy.
TOLERANCE = 1e-9
TOLERANCE_SOURCE = "test_value"


def world(seed, t=L, d=D):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    v = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    q = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    return q, k, v


def subspace_queries(dims, n, seed, d=D):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        row = [0.0] * d
        for dd in dims:
            row[dd] = rnd.gauss(0, 1)
        out.append(row)
    return out


def main():
    q, k, v = world(SEED)
    src = T.make_source(k, v, logical_len=L, pos_start=0)

    # calibration pools sized for the largest per-edge n_ref, plus headroom
    max_nref = 2 * max(BUDGETS)
    pools = [subspace_queries((2 * d, 2 * d + 1), max_nref + 16, SEED + 20 + d)
             for d in range(3)]
    held = [subspace_queries((2 * d, 2 * d + 1), N_HELDOUT, SEED + 90 + d)
            for d in range(3)]

    reps, recs = T.build_topology(src, pools, BUDGETS, heldouts=held, seed=SEED)

    # --- 2D-01: identical pools must collapse -------------------------------
    shared_pool = subspace_queries((0, 1), max_nref + 16, SEED + 555)
    same_reps, _ = T.build_topology(src, [shared_pool] * 3, BUDGETS[0], seed=SEED)

    # --- 2D-03b: held-out invariance ---------------------------------------
    alt_held = subspace_queries((0, 1), N_HELDOUT, SEED + 777)
    _, rec_a = T.build_edge(src, 0, pools[0], BUDGETS[0], heldout_queries=held[0], seed=SEED)
    _, rec_b = T.build_edge(src, 0, pools[0], BUDGETS[0], heldout_queries=alt_held, seed=SEED)

    # --- 2D-03a: index disjointness on a real split ------------------------
    _, _, ci, hi = C.split_pool(pools[0], 2 * BUDGETS[0], N_HELDOUT, seed=SEED)

    wire = json.loads(json.dumps(T.to_wire(reps[0])))
    back = T.from_wire(wire)

    # --- stage records: one dense-SP run, one per-edge construction ---------
    ref = SP.dense_attention(q, k, v)
    sp = SP.sp_forward(q, k, v, WORLD_SIZE)
    # The frozen 2A positions for THIS length, not the 130944 tuple: the same
    # rule, exercised on a tensor small enough to run. 2A-04b reports them in
    # full; on the real workload it is the ten numbers from the execution doc.
    positions = q8.boundary_positions(L, WORLD_SIZE)
    coverage = {"positions": list(positions),
                "sp_rows": {p: sp["output"][p] for p in positions},
                "ref_rows": {p: ref[p] for p in positions}}
    # An in-process reference has no manifest entry, so nothing to compare a
    # digest against. Recording the digest still identifies what was compared,
    # and 2A-04c says "no manifest digest" instead of pretending to verify one.
    ref_digest = hashlib.sha256(
        json.dumps(ref, sort_keys=True).encode("utf-8")).hexdigest()
    env_2a = A2.from_2a(
        sp, ref, TOLERANCE, experiment_id="2a-cpu-closure-v1",
        tolerance_source=TOLERANCE_SOURCE,
        coverage=coverage,
        ref_identity={"path": None, "sha256": ref_digest,
                      "expected_sha256": None, "source": "computed_in_process"},
        notes=["dense SP reference over the same tensors as 2D",
               "tolerance is a CPU test value; Q8 is frozen at 1e-5 for hardware "
               "runs and this record does not cite it"])
    env_2d = AC.from_2d(recs, experiment_id="2d-cpu-closure-v1",
                        source="source", destinations=[0, 1, 2],
                        upstream_refs=[{"stage": E.STAGE_2A,
                                        "experiment_id": env_2a["experiment_id"],
                                        "verdict": env_2a["verdict"]["overall"],
                                        "note": "reference only: the 2A numbers are "
                                                "not used as 2D measurements"}])

    checks = [
        T.check_collapse(same_reps),
        T.check_independence(reps),
        T.check_heldout_disjoint(ci, hi),
        T.check_heldout_invariance(rec_a, rec_b),
        T.check_budget(reps, recs, BUDGETS),
        T.check_record_complete(recs),
        T.check_position_roundtrip(wire, back),
        T.check_rope_phase(),
        T.check_envelope(env_2d),
        T.check_hardware_evidence(env_2d),
    ]
    verdict = T.evaluate(checks)

    out = {
        "meta": {
            "status": verdict["overall"],
            "scope": verdict["scope"],
            "note": ("CPU construction semantics only. No process, no NCCL, no "
                     "all_to_all_v, no real checkpoint. Not a paper result."),
            "chain": ("AM-HighestAttnKeys: softmax-RMS select, box beta [-3,3] "
                      "iters=2, OLS value"),
            "backend": "cpu_ref_f64",
            "dtype_bytes": R.FLOAT_BYTES,
            "L_source_tokens": L,
            "d_model": D,
            "budgets_per_edge": BUDGETS,
            "n_ref_rule": "2t per edge",
            "n_heldout_per_destination": N_HELDOUT,
            "seed": SEED,
            "world_size_2a": WORLD_SIZE,
            "tolerance_2a": TOLERANCE,
            "tolerance_2a_source": TOLERANCE_SOURCE,
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "record_schema": R.SCHEMA_VERSION,
            "envelope_schema": E.SCHEMA_VERSION,
            "definitions": dict(R.DEFINITIONS),
            "solver_policy": R.solver_policy(),
            "requirement_table": {k: dict(v) for k, v in E.REQUIREMENT_TABLE.items()},
        },
        "verdict": verdict,
        "edges": recs,
        "records": {E.STAGE_2A: env_2a, E.STAGE_2D: env_2d},
        "records_summary": {E.STAGE_2A: E.summary(env_2a),
                            E.STAGE_2D: E.summary(env_2d)},
    }

    dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "cpu", "2d_closure.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("check      status                        detail")
    for c in checks:
        d = c["detail"]
        if c["id"] == "2D-02":
            extra = "distinct_payloads=%d" % d["distinct_payloads"]
        elif c["id"] == "2D-04":
            extra = " ".join("e%d:%dB/%dB" % (r["edge"], r["budget"], r["payload_bytes"])
                             for r in d["per_edge"])
        elif c["id"] == "2D-05":
            extra = "%d/%d edges complete" % (sum(1 for r in d["per_edge"] if r["ok"]),
                                              len(d["per_edge"]))
        elif c["id"] == "2D-06a":
            extra = "logical_len=%s span=%s" % (d["logical_len"], d["position_range"])
        elif c["id"] == "2D-06b":
            extra = "needs real checkpoint"
        elif c["id"] == "2D-07":
            extra = "entries=%d counted_for=%s stage=%s" % (
                d["entries"], d["counted_for"], d["stage_verdict"])
        elif c["id"] == "2D-08":
            extra = "sp=%s blocked=%d rows" % (d["sp_status"], len(d["blocked_rows"]))
        else:
            extra = d.get("note", "")
        print("  %-8s %-29s %s" % (c["id"], c["status"], extra))

    print("\noverall: %s  (scope: %s)" % (verdict["overall"], verdict["scope"]))
    print("blocked : %s" % (verdict["blocked_requires_gpu"] or "none"))
    for stage, env in ((E.STAGE_2A, env_2a), (E.STAGE_2D, env_2d)):
        v = env["verdict"]
        print("record %-3s %-29s partial=%s" % (stage, v["overall"], v["partial"]))
    print("-> %s" % dest)


if __name__ == "__main__":
    main()
