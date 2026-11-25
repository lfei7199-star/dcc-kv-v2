"""2D on CPU: one source, three destinations, destination-conditioned construction.

What this closes and what it does not
------------------------------------
2D is 1 source x 3 destinations. On CPU the claim under test is the *construction
semantics*:

  2D-01  identical query sets on all three destinations collapse to one result
  2D-02  three destinations with their own queries are built independently
  2D-03  held-out queries never enter key selection, beta, or the Value fit
  2D-04  each edge's budget is checked on its own; edges are never pooled
  2D-05  every edge emits a complete record (query hash, indices, beta, Value,
         payload bytes, solve diagnostics)
  2D-06  logical_len and the global position span survive serialisation
  2D-07  the shared record validates: one entry per edge, per-edge payloads,
         no cross-edge total (interface check, not a quality claim)
  2D-08  the transfer side stays BLOCKED: a CPU run measured no NCCL

It does NOT show that the distributed transfer works. There is no process, no
NCCL, no all_to_all_v, no real model. `check_rope_phase` is reported as
BLOCKED_CPU_NO_GPU_EVIDENCE on purpose: keeping the position metadata proves the
metadata travels, not that the RoPE phase seen by later tokens is right. That
comparison needs a real checkpoint and a real forward pass.

Deliberately absent: a test that C beats B'. Destination conditioning is an
implementation property and can be asserted; whether it *helps* is a measurement
(see tools/measure_2c_specialization.py), not a gate.
"""

from . import compact as C
from . import envelope as E
from . import record as R

PASS = E.PASS_CPU_SEMANTICS
FAIL = E.FAIL
BLOCKED = E.BLOCKED

# Two constructions from identical inputs run identical float operations, so
# exact equality is expected. The tolerance exists only so a future refactor that
# changes summation order does not fail the check for the wrong reason; it is
# recorded in every collapse verdict.
COLLAPSE_TOL = 1e-12


# ---------------------------------------------------------------------------
# topology plumbing
# ---------------------------------------------------------------------------

def make_source(k, v, logical_len=None, pos_start=0, pos_end=None):
    """One source block: its keys/values plus the position bookkeeping.

    logical_len is AM S2's T -- the length the compact cache *pretends* to have
    so appended tokens see the same positions. It is metadata here; nothing in
    this module consumes it.
    """
    t = len(k)
    if len(v) != t:
        raise ValueError("k and v lengths differ: %d vs %d" % (t, len(v)))
    return {
        "k": k,
        "v": v,
        "source_len": t,
        "logical_len": t if logical_len is None else int(logical_len),
        "pos_start": int(pos_start),
        "pos_end": int(pos_start + t if pos_end is None else pos_end),
    }


def build_edge(source, destination, calib_queries, budget,
               heldout_queries=None, n_ref=None, seed=0,
               dtype_bytes=R.FLOAT_BYTES, **kwargs):
    """Build one Source -> Destination edge and its record.

    n_ref defaults to 2t (U3). Sampling uses `seed`, which is the *same* for all
    edges by default: the only thing that should differ between edges is the
    pool they draw from, so that any difference in the output is attributable to
    the destination's queries and not to a different subsample.
    """
    if n_ref is None:
        n_ref = C.compute_n_ref(budget)
    if len(calib_queries) < n_ref:
        raise ValueError(
            "calibration pool too small for edge %r: need %d, have %d"
            % (destination, n_ref, len(calib_queries)))
    use, _ = C.sample_pool(calib_queries, n_ref, seed=seed)

    store_tensors = kwargs.pop("store_tensors", True)
    rep = C.build(use, source["k"], source["v"], budget,
                  position_range=(source["pos_start"], source["pos_end"]),
                  logical_len=source["logical_len"],
                  **kwargs)
    rec = R.edge_record(
        rep,
        source=source.get("label", "source"),
        destination=destination,
        calib_queries=use,
        heldout_queries=heldout_queries,
        dtype_bytes=dtype_bytes,
        store_tensors=store_tensors,
    )
    return rep, rec


def build_topology(source, pools, budgets, heldouts=None, seed=0, **kwargs):
    """Build all edges of the 1-source x n-destination topology.

    budgets may be an int (same for every edge) or a per-edge sequence. Per-edge
    budgets are the point of 2D-04: passing one int is a convenience, not a
    claim that the edges share a budget.
    """
    if isinstance(budgets, int):
        budgets = [budgets] * len(pools)
    if len(budgets) != len(pools):
        raise ValueError("budgets and pools differ in length: %d vs %d"
                         % (len(budgets), len(pools)))
    if heldouts is None:
        heldouts = [None] * len(pools)

    reps, recs = [], []
    for di, (pool, budget) in enumerate(zip(pools, budgets)):
        rep, rec = build_edge(source, di, pool, budget,
                              heldout_queries=heldouts[di], seed=seed, **kwargs)
        reps.append(rep)
        recs.append(rec)
    return reps, recs


# ---------------------------------------------------------------------------
# serialisation (2D-06)
# ---------------------------------------------------------------------------

def to_wire(rep):
    """JSON-safe form of a compact representation, metadata included."""
    return {
        "indices": list(rep["indices"]),
        "k_sel": [list(r) for r in rep["k_sel"]],
        "v_sel": [list(r) for r in rep["v_sel"]],
        "beta": list(rep["beta"]),
        "budget": rep["budget"],
        "logical_len": rep["logical_len"],
        "source_len": rep.get("source_len"),
        "position_range": list(rep["position_range"]),
        "selector": rep["selector"],
        "beta_mode": rep["beta_mode"],
        "value_mode": rep["value_mode"],
        "diagnostics": rep["diagnostics"],
    }


def from_wire(obj):
    """Inverse of to_wire. Diagnostics survive as plain data."""
    out = dict(obj)
    out["indices"] = list(obj["indices"])
    out["position_range"] = list(obj["position_range"])
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def _max_abs(a, b):
    m = 0.0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d > m:
            m = d
    return m


def check_collapse(reps, tol=COLLAPSE_TOL):
    """2D-01: identical query sets must collapse to one representation."""
    detail, ok = [], True
    if len(reps) < 2:
        return {"id": "2D-01", "status": FAIL,
                "detail": {"note": "collapse needs at least two edges"}}
    base = reps[0]
    for di, r in enumerate(reps[1:], 1):
        same_idx = r["indices"] == base["indices"]
        db = _max_abs(r["beta"], base["beta"])
        dv = 0.0
        for ra, rb in zip(r["v_sel"], base["v_sel"]):
            d = _max_abs(ra, rb)
            if d > dv:
                dv = d
        good = same_idx and db <= tol and dv <= tol
        ok = ok and good
        detail.append({"edge": di, "indices_identical": same_idx,
                       "max_abs_beta_diff": db, "max_abs_value_diff": dv,
                       "tolerance": tol, "ok": good})
    return {
        "id": "2D-01", "status": PASS if ok else FAIL,
        "detail": {"note": "identical queries on every destination",
                   "payload_digest": R.payload_digest(base),
                   "per_edge": detail},
    }


def check_independence(reps):
    """2D-02: distinct query sets must produce distinct representations.

    Only requires that the edges are not all identical -- requiring all pairs to
    differ would be a claim about the data, not about the code.
    """
    digests = [R.payload_digest(r) for r in reps]
    distinct = len(set(digests))
    idx_sets = {tuple(r["indices"]) for r in reps}
    ok = distinct > 1 and len(idx_sets) > 1
    return {
        "id": "2D-02", "status": PASS if ok else FAIL,
        "detail": {"edges": len(reps), "distinct_payloads": distinct,
                   "distinct_key_sets": len(idx_sets),
                   "payload_digests": digests,
                   "note": "edges built from separate calibration pools"},
    }


def check_heldout_disjoint(calib_indices, held_indices):
    """2D-03a: the two index sets must not intersect."""
    inter = sorted(set(calib_indices) & set(held_indices))
    return {
        "id": "2D-03a", "status": PASS if not inter else FAIL,
        "detail": {"n_calib": len(calib_indices), "n_heldout": len(held_indices),
                   "intersection": inter},
    }


def check_heldout_invariance(rec_a, rec_b):
    """2D-03b: replacing only the held-out pool must not move the payload.

    This is the functional form of "held-out queries never enter the fit": if the
    construction touched them, swapping them would change the digest.
    """
    ha = rec_a["payload_digest"]
    hb = rec_b["payload_digest"]
    different_pools = rec_a["query_hash_heldout"] != rec_b["query_hash_heldout"]
    ok = ha == hb and different_pools
    return {
        "id": "2D-03b", "status": PASS if ok else FAIL,
        "detail": {"payload_digest_a": ha, "payload_digest_b": hb,
                   "heldout_pools_differ": different_pools,
                   "note": "the check is vacuous unless the held-out pools differ"},
    }


def check_budget(reps, recs, expected_budgets):
    """2D-04: per-edge budget identity. Structurally refuses to pool."""
    rows, ok = [], True
    for di, (rep, rec) in enumerate(zip(reps, recs)):
        want = expected_budgets[di]
        good = (rep["budget"] == want
                and rec["budget_tokens"] == want
                and len(rec["selected_indices"]) == want
                and rec["payload"]["t"] == want
                and rec["payload"]["counted_for"] == "this edge only")
        ok = ok and good
        rows.append({"edge": di, "expected": want, "budget": rep["budget"],
                     "record_budget": rec["budget_tokens"],
                     "payload_bytes": rec["payload"]["bytes"], "ok": good})
    return {
        "id": "2D-04", "status": PASS if ok else FAIL,
        "detail": {"per_edge": rows,
                   "sum_of_budgets": sum(expected_budgets),
                   "note": ("the sum is reported for reading only; it is not a "
                            "budget of any single edge and must not be used as one")},
    }


def check_record_complete(recs):
    """2D-05: every edge's record validates against the frozen schema."""
    rows, ok = [], True
    for di, rec in enumerate(recs):
        try:
            R.validate(rec, strict=True)
            rows.append({"edge": di, "ok": True, "problems": []})
        except R.RecordError as exc:
            ok = False
            rows.append({"edge": di, "ok": False, "problems": [str(exc)]})
    return {"id": "2D-05", "status": PASS if ok else FAIL, "detail": {"per_edge": rows}}


def check_position_roundtrip(rep_json, rep_back):
    """2D-06a: logical_len and the global span survive a JSON round trip."""
    fields = ("logical_len", "source_len")
    bad = [f for f in fields if rep_json.get(f) != rep_back.get(f)]
    if list(rep_json["position_range"]) != list(rep_back["position_range"]):
        bad.append("position_range")
    if list(rep_json["indices"]) != list(rep_back["indices"]):
        bad.append("indices")
    ok = not bad
    return {
        "id": "2D-06a", "status": PASS if ok else FAIL,
        "detail": {"logical_len": rep_back.get("logical_len"),
                   "position_range": rep_back.get("position_range"),
                   "lost_fields": bad},
    }


def check_rope_phase():
    """2D-06b: real RoPE phase consistency. Not decidable on CPU."""
    return {
        "id": "2D-06b", "status": BLOCKED,
        "detail": {"note": ("position metadata travelling is necessary but not "
                            "sufficient: the RoPE phase seen by appended tokens "
                            "must be compared against a dense forward pass on a "
                            "real checkpoint. Requires 2A GPU run.")},
    }


# ---------------------------------------------------------------------------
# 2D-07/08: the shared record
# ---------------------------------------------------------------------------

def check_envelope(env):
    """2D-07: the 2D record validates, is per-edge, and stays inside its stage.

    This is the interface check, not a second opinion on the construction: it
    says the record can be read, every edge carries its own diagnostics, and no
    per-edge payload was pooled. Whether the construction is any good is 2D-01..06.
    """
    problems = E.validate(env, strict=False)
    v = env.get("verdict") or {}
    entries = (env.get("compact_diagnostics") or {}).get("entries") or []
    ok = not problems and v.get("overall") != E.FAIL
    return {
        "id": "2D-07", "status": PASS if ok else FAIL,
        "detail": {
            "problems": problems,
            "schema_version": env.get("schema_version"),
            "stage": env.get("stage"),
            "entries": len(entries),
            "counted_for": [e.get("payload", {}).get("counted_for") for e in entries],
            # the construction verdict and the stage verdict are different
            # objects; both are printed so neither can be mistaken for the other
            "stage_verdict": v.get("overall"),
            "partial": v.get("partial"),
        },
    }


def check_hardware_evidence(env):
    """2D-08: what this record did not measure has to stay visible.

    On CPU the answer is always "the SP side", so this row is BLOCKED by
    construction. It exists so that a future run cannot quietly drop the gap:
    the row is read from the record, not assumed.
    """
    v = env.get("verdict") or {}
    blocked = [c["id"] for c in (v.get("checks") or []) if c.get("status") == BLOCKED]
    sp_status = E.classify_block(env.get("sp_structure"))
    return {
        "id": "2D-08",
        "status": BLOCKED if blocked else PASS,
        "detail": {"blocked_rows": blocked,
                   "sp_status": sp_status,
                   "evidence_level": (env.get("scope") or {}).get("evidence_level"),
                   "note": ("no process, no NCCL, no all_to_all_v: the transfer "
                            "side of 2D is not measured here")},
    }


def evaluate(checks, scope="CPU construction semantics only"):
    """Fold the checks into one verdict.

    The overall string is never a bare "PASS": the strongest thing a CPU run may
    claim is PASS_CPU_SEMANTICS, and blocked items stay visible in the list.
    """
    has_fail = any(c["status"] == FAIL for c in checks)
    blocked = [c["id"] for c in checks if c["status"] == BLOCKED]
    overall = FAIL if has_fail else PASS
    return {
        "overall": overall,
        "scope": scope,
        "blocked_requires_gpu": blocked,
        "checks": checks,
    }
