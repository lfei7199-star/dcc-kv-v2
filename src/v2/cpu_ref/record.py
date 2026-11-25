"""Per-edge construction record: one record per Source -> Destination edge.

Why a module and not a dict literal in each script
-------------------------------------------------
The frozen doc requires that a field means the same thing everywhere it appears.
A field list alone does not achieve that: two scripts can both write
`mass_residual` while one pools over queries and the other averages per query,
and the numbers then look comparable when they are not. So the definitions live
here, are echoed into every record, and `validate()` refuses a record whose
echoed tolerance disagrees with the frozen constant.

Three rules this module enforces
--------------------------------
1. **Per edge, never pooled.** The three edges of 2D are three separate least
   squares problems with separate budgets, separate query sets and separate
   rank/condition numbers. There is deliberately no "total budget" field: the
   sum of three per-edge budgets is not a budget of anything.
2. **beta design and value design are different matrices.** `beta_design` is
   A (n_ref x t) with target m; `value_design` is X (n_ref x t) with target Y.
   They are never merged into one rank/cond, because a single number would then
   describe neither problem.
3. **Solve and diagnose are separate.** `solver.method` names how the answer was
   obtained; `diagnostics.*_design.method` names how rank/cond were obtained.
   They are not the same computation, and on CUDA they cannot be: the CUDA
   `gels` driver assumes full rank and does not return a usable rank, so the
   rank must come from a separate diagnostic pass. `solver.fallback` is the
   empty string by contract -- a failed solve is recorded as failed.
"""

import hashlib
import json

from . import compact as C

SCHEMA_VERSION = "dcckv-edge-record/1"

# CPU reference path. The CUDA path will report a different value; a record is
# only comparable with another record carrying the same diagnostic method.
DIAG_METHOD = "householder_qr_f64_cpu"
SOLVER_METHOD = "householder_qr_f64_cpu"

INDEX_BYTES = 4     # int32 selected-key indices
META_BYTES = 32     # logical_len, position_range, budget, dtype tag

FLOAT_BYTES = 8     # CPU reference dtype. BF16 runs record 2 and must say so.

# Echoed verbatim into every record. If any of these changes, records produced
# before and after the change are not comparable and `validate` will say so.
DEFINITIONS = {
    "bind_tol": C.BIND_TOL,
    "bind_rule": "a key is on a bound when |beta_j - bound| <= bind_tol, in beta units",
    "dead_rel": C.DEAD_REL,
    "dead_rule": C.DEAD_SCOPE,
    "mass_residual_agg": C.MASS_RESIDUAL_AGG,
    "mass_residual_rule": ("||A w - m||_2 / ||m||_2 where A and m are stacked over ALL "
                           "n_ref calibration queries and the norm is taken once over the "
                           "stacked vector; it is NOT a mean of per-query ratios"),
    "beta_design": ("A in R^(n_ref x t), A_ij = exp(q_i . C_k_j / sqrt(d)) after a per-row "
                    "max shift; target m_i = sum over ALL source keys of exp(q_i . K / sqrt(d))"),
    "value_design": ("X in R^(n_ref x t), X_ij = softmax_j(q_i . C_k_j / sqrt(d) + beta_j); "
                     "target Y = dense attention output on the same queries"),
    "payload": ("per edge: t*dk*dtype + t*dv*dtype (keys, values) + t*dtype (beta) + "
                "t*index_bytes (indices) + meta_bytes. No cross-edge total is defined."),
}


class RecordError(ValueError):
    """The record is missing a required field, or two fields contradict."""


def solver_policy():
    """The solve/fail contract. Recorded, not implied."""
    return {
        "fallback": "",                      # never "ridge", never "pinv"
        "on_failure": "raise_and_record",
        "solve_separate_from_diagnose": True,
        "note": ("CUDA torch.linalg.lstsq only supports driver='gels', which assumes "
                 "full rank and does not return a usable rank. A GPU run must obtain "
                 "rank/cond from a separate diagnostic pass and record both. A failed "
                 "solve is recorded as failed; switching to ridge/pinv is forbidden."),
    }


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def query_hash(rows):
    """Kept as an alias: the implementation lives in compact so the builder and
    the record cannot drift apart."""
    return C.query_hash(rows)


def payload_digest(rep):
    """Identity of the constructed payload: indices, beta and Value together.

    Two constructions that agree here are the same compact representation, which
    is what 2D-01 (same queries collapse) and 2D-03 (held-out isolation) need to
    compare.
    """
    h = hashlib.sha256()
    h.update(("%d|" % len(rep["indices"])).encode())
    h.update(",".join(str(i) for i in rep["indices"]).encode())
    h.update(b"|")
    h.update(",".join(format(b, ".17g") for b in rep["beta"]).encode())
    h.update(b"|")
    for row in rep["v_sel"]:
        h.update(",".join(format(x, ".17g") for x in row).encode())
        h.update(b";")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# payload accounting
# ---------------------------------------------------------------------------

def payload_breakdown(rep, dtype_bytes=FLOAT_BYTES):
    """Per-edge byte count. Structurally unable to pool across edges."""
    t = len(rep["indices"])
    dk = len(rep["k_sel"][0]) if t else 0
    dv = len(rep["v_sel"][0]) if t else 0
    keys = t * dk * dtype_bytes
    values = t * dv * dtype_bytes
    beta = t * dtype_bytes
    idx = t * INDEX_BYTES
    total = keys + values + beta + idx + META_BYTES
    return {
        "counted_for": "this edge only",
        "t": t, "dk": dk, "dv": dv, "dtype_bytes": dtype_bytes,
        "keys_bytes": keys, "values_bytes": values, "beta_bytes": beta,
        "indices_bytes": idx, "meta_bytes": META_BYTES,
        "bytes": total,
        "bits_per_source_token": (8.0 * total / rep["source_len"]) if rep.get("source_len") else None,
    }


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def _design_block(stats):
    """Normalise one design-matrix diagnosis into a fixed shape."""
    if stats is None:
        return None
    return {
        "problem": stats.get("problem"),
        "method": DIAG_METHOD,
        "n": stats.get("n"), "p": stats.get("p"),
        "rank": stats.get("rank"),
        "cond": stats.get("cond"),
        "sigma_min": stats.get("sigma_min"),
        "sigma_max": stats.get("sigma_max"),
        "tol": stats.get("tol"),
        "rank_deficient": stats.get("rank_deficient"),
    }


def edge_record(rep, source, destination, calib_queries, heldout_queries=None,
                dtype_bytes=FLOAT_BYTES, store_tensors=True):
    """Build the canonical record for one Source -> Destination edge.

    source / destination are labels (a rank id, a node id, an int). They are
    stored as-is so a record can be traced back to a topology.
    """
    diag = rep["diagnostics"]
    b = diag["beta"]
    v = diag["value"]

    beta_bounds = {
        "lo": b.get("lo"), "hi": b.get("hi"),
        "bind_tol": b.get("bind_tol", C.BIND_TOL),
        "bind_hi_count": b.get("bind_hi_count"),
        "bind_hi_frac": b.get("bind_hi_frac"),
        "bind_lo_count": b.get("bind_lo_count"),
        "bind_lo_frac": b.get("bind_lo_frac"),
        "beta_min": b.get("beta_min"),
        "beta_max": b.get("beta_max"),
        "beta_mean": b.get("beta_mean"),
    }

    beta_dead = {
        "dead_rel": b.get("dead_rel", C.DEAD_REL),
        "dead_count": b.get("dead_count"),
        "dead_frac": b.get("dead_frac"),
        "denominator": b.get("dead_denominator"),
        "scope": b.get("dead_scope", C.DEAD_SCOPE),
    }

    mass = {
        "aggregation": b.get("mass_residual_agg", C.MASS_RESIDUAL_AGG),
        "normalized": b.get("mass_residual"),
        "raw": b.get("mass_residual_raw"),
        "denominator": b.get("mass_residual_denom"),
        "n_ref": b.get("n_ref"),
    }

    calib = list(calib_queries) if calib_queries is not None else []
    held = list(heldout_queries) if heldout_queries is not None else []
    calib_hash = (C.query_hash(calib) if calib
                  else rep.get("query_hash"))
    n_ref = len(calib) if calib else rep.get("n_queries")

    rec = {
        "schema": SCHEMA_VERSION,
        "definitions": dict(DEFINITIONS),
        "edge": {"source": source, "destination": destination},
        "budget_tokens": rep["budget"],
        "n_ref": n_ref,
        "selector": rep["selector"],
        "beta_mode": rep["beta_mode"],
        "value_mode": rep["value_mode"],
        "query_hash_calibration": calib_hash,
        "query_hash_heldout": query_hash(held) if held else None,
        "n_heldout": len(held),
        "selected_indices": list(rep["indices"]),
        "logical_len": rep["logical_len"],
        "source_len": rep.get("source_len"),
        "position_range": list(rep.get("position_range") or [0, rep["logical_len"]]),
        "payload": payload_breakdown(rep, dtype_bytes=dtype_bytes),
        "payload_digest": payload_digest(rep),
        "diagnostics": {
            "beta_bounds": beta_bounds,
            "beta_dead": beta_dead,
            "mass_residual": mass,
            "beta_design": _design_block(b.get("beta_design") or b.get("beta_fit")),
            "value_design": _design_block(v),
        },
        "solver": {
            "method": SOLVER_METHOD,
            "status": b.get("solver_status"),
            "value_status": v.get("solver_status"),
            "fallback_used": False,
            "dtype_bytes": dtype_bytes,
            "policy": solver_policy(),
        },
    }

    if store_tensors:
        rec["beta"] = list(rep["beta"])
        rec["values"] = [list(row) for row in rep["v_sel"]]

    return rec


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_REQUIRED = (
    "schema", "definitions", "edge", "budget_tokens", "n_ref", "selector",
    "beta_mode", "value_mode", "query_hash_calibration", "selected_indices",
    "logical_len", "position_range", "payload", "payload_digest", "diagnostics",
    "solver",
)

_REQUIRED_DIAG = ("beta_bounds", "beta_dead", "mass_residual",
                  "beta_design", "value_design")


def validate(rec, strict=True):
    """Return a list of problems; raise RecordError if strict and non-empty.

    This is the guard against the failure the ruling described: a full-looking
    record whose fields cannot be compared across budgets.
    """
    problems = []

    for key in _REQUIRED:
        if key not in rec:
            problems.append("missing field: %s" % key)
    if problems:
        if strict:
            raise RecordError("; ".join(problems))
        return problems

    if rec["schema"] != SCHEMA_VERSION:
        problems.append("schema %r != %r" % (rec["schema"], SCHEMA_VERSION))

    # the echoed conventions must still match the frozen constants
    d = rec["definitions"]
    if abs(d.get("bind_tol", -1) - C.BIND_TOL) > 0:
        problems.append("bind_tol drift: record %r vs frozen %r"
                        % (d.get("bind_tol"), C.BIND_TOL))
    if abs(d.get("dead_rel", -1) - C.DEAD_REL) > 0:
        problems.append("dead_rel drift: record %r vs frozen %r"
                        % (d.get("dead_rel"), C.DEAD_REL))
    if d.get("mass_residual_agg") != C.MASS_RESIDUAL_AGG:
        problems.append("mass_residual aggregation differs from frozen convention")

    t = rec["budget_tokens"]
    if len(rec["selected_indices"]) != t:
        problems.append("selected_indices has %d entries, budget is %d"
                        % (len(rec["selected_indices"]), t))
    if len(set(rec["selected_indices"])) != len(rec["selected_indices"]):
        problems.append("selected_indices contains duplicates")

    diag = rec["diagnostics"]
    for key in _REQUIRED_DIAG:
        if key not in diag or diag[key] is None:
            problems.append("missing diagnostics block: %s" % key)
    if problems:
        if strict:
            raise RecordError("; ".join(problems))
        return problems

    bb = diag["beta_bounds"]
    if bb.get("bind_hi_count", 0) + 0 > t or bb.get("bind_lo_count", 0) + 0 > t:
        problems.append("bound-saturation count exceeds t")
    for frac_key in ("bind_hi_frac", "bind_lo_frac"):
        f = bb.get(frac_key)
        if f is not None and not (0.0 <= f <= 1.0):
            problems.append("%s out of [0,1]: %r" % (frac_key, f))

    bd = diag["beta_dead"]
    if bd.get("dead_count") is not None and bd["dead_count"] > t:
        problems.append("dead_count exceeds t")
    if bd.get("dead_frac") is not None and not (0.0 <= bd["dead_frac"] <= 1.0):
        problems.append("dead_frac out of [0,1]: %r" % bd["dead_frac"])

    mr = diag["mass_residual"]
    if mr.get("normalized") is not None and mr["normalized"] < 0.0:
        problems.append("mass_residual is negative")

    # the two design problems are distinct objects and carry their own numbers
    for name in ("beta_design", "value_design"):
        blk = diag[name]
        if blk.get("p") != t:
            problems.append("%s.p = %r but t = %d" % (name, blk.get("p"), t))
        if blk.get("rank") is None or blk.get("cond") is None:
            problems.append("%s lacks rank/cond" % name)

    if rec["solver"].get("fallback_used"):
        problems.append("solver.fallback_used is true; fallback is forbidden by contract")

    pr = rec["position_range"]
    if len(pr) != 2 or pr[0] > pr[1]:
        problems.append("position_range is not an ordered pair: %r" % (pr,))
    elif pr[1] - pr[0] > rec["logical_len"]:
        problems.append("position span exceeds logical_len")

    if strict and problems:
        raise RecordError("; ".join(problems))
    return problems


def to_json(rec):
    return json.dumps(rec, indent=2, ensure_ascii=False, allow_nan=False)
