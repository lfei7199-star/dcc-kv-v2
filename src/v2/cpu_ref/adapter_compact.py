"""2C / 2D -> envelope adapter: project the per-edge record into the shared shell.

Why a projection and not a second record
----------------------------------------
`record.edge_record` already holds the authoritative construction numbers: it was
written by the pass that ran the solve, and `payload_digest` identifies what it
produced. This adapter copies those fields into the five shared blocks so 2A and
2D read alike. It never re-solves, never re-derives a rank, and never fills a
field the inner record does not have -- a missing block stays MISSING and blocks
acceptance instead of acquiring a plausible number.

The three edges stay three edges
--------------------------------
Each entry keeps its own budget, its own rank/cond pair and its own payload,
labelled `counted_for = "this edge only"`. `check_no_double_count` recomputes
every entry from its own (t, dk, dv, dtype), so the merge-step mistake the ruling
names -- three edges collapsed into one repeated total -- is caught by arithmetic
rather than by review.

SP on CPU is declared MISSING, not NOT_APPLICABLE
-------------------------------------------------
2C and 2D run *on* SP. This CPU run did not run SP at all: no process, no
transport, no ranks. That is a gap in evidence (`MISSING` + reason), not a
quantity that does not exist (`NOT_APPLICABLE`, which is what 2A's absent beta
is). The adapter writes the gap, and the envelope cannot fold to a pass while it
stands. A 2D record may cite the 2A envelope through `upstream_refs`, but the
citation is a pointer and never a measurement -- `validate` rejects any attempt
to fill an SP block from upstream.
"""

import hashlib
import os
import sys

from . import envelope as E
from . import record as R


def indices_digest(indices):
    return hashlib.sha256(",".join(str(i) for i in indices).encode()).hexdigest()


def default_files():
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("compact.py", "record.py", "two_d.py", "adapter_compact.py",
             "envelope.py")
    return [(n, E.file_sha256(os.path.join(here, n))) for n in names
            if os.path.exists(os.path.join(here, n))]


def environment():
    return {
        "backend": "cpu_ref_f64",
        "python": sys.version.split()[0],
        "dtype_bytes": R.FLOAT_BYTES,
        "diagnostic_method": R.DIAG_METHOD,
        "solver_method": R.SOLVER_METHOD,
    }


# ---------------------------------------------------------------------------
# entry: one construction
# ---------------------------------------------------------------------------

def entry_from_record(rec, label, edge=None):
    """Project one edge_record into the five shared blocks.

    Only fields that exist are copied. Nothing is recomputed, so an inner record
    that failed to produce a value design cannot come out of here with one.
    """
    d = rec["diagnostics"]
    bd = d.get("beta_design") or {}
    vd = d.get("value_design") or {}
    bb = d.get("beta_bounds") or {}
    dead = d.get("beta_dead") or {}
    mr = d.get("mass_residual") or {}

    entry = {
        "label": label,
        "edge": dict(edge) if edge else None,
        "budget_tokens": rec["budget_tokens"],
        "inner_schema": rec["schema"],
        "payload_digest": rec["payload_digest"],
        "key_selection": E.applicable(
            selector=rec["selector"],
            beta_mode=rec["beta_mode"],
            value_mode=rec["value_mode"],
            budget_tokens=rec["budget_tokens"],
            n_ref=rec["n_ref"],
            query_hash=rec["query_hash_calibration"],
            indices=indices_digest(rec["selected_indices"]),
            n_indices=len(rec["selected_indices"]),
            heldout={"n": rec["n_heldout"],
                     "query_hash": rec["query_hash_heldout"]},
        ),
        "beta_design": E.applicable(
            problem=bd.get("problem"), method=bd.get("method"),
            n=bd.get("n"), p=bd.get("p"), rank=bd.get("rank"),
            cond=bd.get("cond"), sigma_min=bd.get("sigma_min"),
            sigma_max=bd.get("sigma_max"), tol=bd.get("tol"),
            rank_deficient=bd.get("rank_deficient"),
        ),
        "value_design": E.applicable(
            problem=vd.get("problem"), method=vd.get("method"),
            n=vd.get("n"), p=vd.get("p"), rank=vd.get("rank"),
            cond=vd.get("cond"), sigma_min=vd.get("sigma_min"),
            sigma_max=vd.get("sigma_max"), tol=vd.get("tol"),
            rank_deficient=vd.get("rank_deficient"),
        ),
        "beta_binding": E.applicable(
            lo=bb.get("lo"), hi=bb.get("hi"), bind_tol=bb.get("bind_tol"),
            bind_hi_count=bb.get("bind_hi_count"), bind_hi_frac=bb.get("bind_hi_frac"),
            bind_lo_count=bb.get("bind_lo_count"), bind_lo_frac=bb.get("bind_lo_frac"),
            beta_min=bb.get("beta_min"), beta_max=bb.get("beta_max"),
            beta_mean=bb.get("beta_mean"),
            dead={"dead_rel": dead.get("dead_rel"),
                  "dead_count": dead.get("dead_count"),
                  "dead_frac": dead.get("dead_frac"),
                  "denominator": dead.get("denominator"),
                  "scope": dead.get("scope")},
        ),
        "mass_residual": E.applicable(
            aggregation=mr.get("aggregation"),
            normalized=mr.get("normalized"),
            raw=mr.get("raw"),
            denominator=mr.get("denominator"),
            n_ref=mr.get("n_ref"),
        ),
        "payload": dict(rec["payload"]),
        "position": {
            "logical_len": rec["logical_len"],
            "source_len": rec.get("source_len"),
            "span": list(rec["position_range"]),
            "rope_phase_verified": False,
        },
        "solver": {
            "method": rec["solver"]["method"],
            "beta_status": rec["solver"]["status"],
            "value_status": rec["solver"]["value_status"],
            "fallback_used": rec["solver"]["fallback_used"],
            "diagnostics_separate_from_solve": True,
        },
    }
    return entry


def _compact_checks(entries, prefix):
    """One row per block per entry, so a hole is visible at row level too."""
    rows = []

    def add(cid, group, ok, detail):
        rows.append({"id": cid, "group": group,
                     "status": E.PASS_CPU_SEMANTICS if ok else E.FAIL,
                     "detail": detail})

    for e in entries:
        lb = e["label"]
        add("%s.key_selection.%s" % (prefix, lb), "compact",
            e["key_selection"].get("n_indices") == e["budget_tokens"],
            {"n_indices": e["key_selection"].get("n_indices"),
             "budget_tokens": e["budget_tokens"]})
        add("%s.beta_design.%s" % (prefix, lb), "compact",
            e["beta_design"].get("rank") is not None
            and e["beta_design"].get("cond") is not None,
            {"rank": e["beta_design"].get("rank"),
             "cond": e["beta_design"].get("cond"),
             "rank_deficient": e["beta_design"].get("rank_deficient")})
        add("%s.value_design.%s" % (prefix, lb), "compact",
            e["value_design"].get("rank") is not None
            and e["value_design"].get("cond") is not None,
            {"rank": e["value_design"].get("rank"),
             "cond": e["value_design"].get("cond"),
             "rank_deficient": e["value_design"].get("rank_deficient")})
        add("%s.beta_binding.%s" % (prefix, lb), "compact",
            e["beta_binding"].get("bind_hi_frac") is not None
            and e["beta_binding"].get("bind_lo_frac") is not None,
            {"bind_hi_frac": e["beta_binding"].get("bind_hi_frac"),
             "bind_lo_frac": e["beta_binding"].get("bind_lo_frac"),
             "dead_frac": e["beta_binding"]["dead"].get("dead_frac")})
        add("%s.mass_residual.%s" % (prefix, lb), "compact",
            e["mass_residual"].get("normalized") is not None,
            {"normalized": e["mass_residual"].get("normalized"),
             "aggregation": e["mass_residual"].get("aggregation")})
    return rows


_SP_BLOCKED = [
    {"id": "sp.rank_topology", "group": "sp", "status": E.BLOCKED,
     "detail": {"note": "no SP process in this CPU run; topology is declared, not run"}},
    {"id": "sp.owner_kv_resident", "group": "sp", "status": E.BLOCKED,
     "detail": {"note": "dense KV residency is a 2A measurement"}},
    {"id": "sp.remote_staging", "group": "sp", "status": E.BLOCKED,
     "detail": {"note": "no remote blocks are staged without an SP run"}},
    {"id": "sp.no_full_copy", "group": "sp", "status": E.BLOCKED,
     "detail": {"note": "the negative arm needs a real SP run to be meaningful"}},
    {"id": "sp.communication", "group": "sp", "status": E.BLOCKED,
     "detail": {"note": "no NCCL, no all_to_all_v; nothing to measure"}},
]

_SP_MISSING = ("this stage runs on SP, but this CPU run has no process, no ranks "
               "and no transport; the SP record is a gap in evidence, not an "
               "inapplicable quantity")


def from_2c(rec, *, experiment_id, label="B_prime_shared", prov=None, work=None,
            upstream_refs=None, notes=None):
    """2C: the shared-query arm, one construction, no destination topology."""
    entry = entry_from_record(rec, label)
    checks = _compact_checks([entry], "compact") + [dict(c) for c in _SP_BLOCKED]

    return E.new_envelope(
        E.STAGE_2C, experiment_id,
        scope={
            "evidence_level": E.STRUCTURAL,
            "claims": ["shared_query_construction", "per_budget_identity",
                       "merge_mass_conservation"],
            "does_not_claim": ["nccl_transport", "all_to_all_v_bandwidth",
                               "wall_clock_speedup", "compression_quality_benefit",
                               "real_model_numerics"],
        },
        prov=prov or E.provenance(
            files=default_files(),
            config={"budget": rec["budget_tokens"], "n_ref": rec["n_ref"],
                    "selector": rec["selector"], "beta_mode": rec["beta_mode"],
                    "value_mode": rec["value_mode"]},
            environment=environment(),
            generator="src/v2/cpu_ref/adapter_compact.py"),
        topo=E.topology(world_size=None, ranks=[], source="source",
                        destinations=[], rank_map={},
                        layout="single_construction_no_sp"),
        work=work or E.workload(
            model={"synthetic": True, "name": "synthetic_gaussian",
                   "revision": None, "note": "construction semantics only"},
            seq_len=rec["source_len"] or rec["logical_len"],
            position_ids={"start": rec["position_range"][0],
                          "end": rec["position_range"][1],
                          "logical_len": rec["logical_len"],
                          "rope_phase_verified": False}),
        sp={"status": E.MISSING, "reason": _SP_MISSING},
        compact=E.applicable(per_edge=False, entries=[entry],
                             note="one construction, no destination set"),
        checks=checks,
        upstream_refs=upstream_refs,
        notes=notes,
    )


def from_2d(recs, *, experiment_id, source, destinations, prov=None, work=None,
            upstream_refs=None, notes=None, shared_query_set=False):
    """2D: one source, one entry per destination, each with its own numbers."""
    entries = []
    for i, rec in enumerate(recs):
        dest = destinations[i] if i < len(destinations) else rec["edge"]["destination"]
        entries.append(entry_from_record(rec, label="edge-%d" % i,
                                         edge={"source": source, "destination": dest}))

    hashes = [e["key_selection"]["query_hash"] for e in entries]
    per_dest = len(set(hashes)) > 1
    sem = E.applicable(
        shared_query_set=bool(shared_query_set),
        per_destination_queries=bool(per_dest),
        query_hashes=hashes,
        note=("each destination draws from its own pool" if per_dest else
              "the destinations share one query set"),
    )
    checks = _compact_checks(entries, "compact") + [dict(c) for c in _SP_BLOCKED]
    checks.append({"id": "destination.independence", "group": "destination",
                   "status": E.PASS_CPU_SEMANTICS if per_dest else E.FAIL,
                   "detail": {"distinct_query_hashes": len(set(hashes)),
                              "destinations": len(destinations)}})

    return E.new_envelope(
        E.STAGE_2D, experiment_id,
        scope={
            "evidence_level": E.STRUCTURAL,
            "claims": ["destination_conditioned_construction", "per_edge_budget",
                       "heldout_isolation", "position_metadata_preserved"],
            "does_not_claim": ["nccl_transport", "all_to_all_v_bandwidth",
                               "distance_decay_of_gain", "wall_clock_speedup",
                               "compression_quality_benefit"],
        },
        prov=prov or E.provenance(
            files=default_files(),
            config={"budgets": [r["budget_tokens"] for r in recs],
                    "n_ref": [r["n_ref"] for r in recs],
                    "selector": recs[0]["selector"],
                    "beta_mode": recs[0]["beta_mode"],
                    "value_mode": recs[0]["value_mode"]},
            environment=environment(),
            generator="src/v2/cpu_ref/adapter_compact.py"),
        topo=E.topology(world_size=None, ranks=[], source=source,
                        destinations=list(destinations), rank_map={},
                        layout="one_source_n_destinations_no_transport"),
        work=work or E.workload(
            model={"synthetic": True, "name": "synthetic_gaussian",
                   "revision": None, "note": "construction semantics only"},
            seq_len=recs[0]["source_len"] or recs[0]["logical_len"],
            position_ids={"start": recs[0]["position_range"][0],
                          "end": recs[0]["position_range"][1],
                          "logical_len": recs[0]["logical_len"],
                          "rope_phase_verified": False}),
        sp={"status": E.MISSING, "reason": _SP_MISSING},
        compact=E.applicable(per_edge=True, entries=entries,
                             destination_semantics=sem,
                             note="each Source->Destination edge carries its own "
                                  "budget, rank/cond pair and payload"),
        checks=checks,
        upstream_refs=upstream_refs,
        notes=notes,
    )
