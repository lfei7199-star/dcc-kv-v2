"""2A -> envelope adapter.

The one rule this file exists to hold: 2A acceptance must not import the
compression solver. A bug in `beta_box_ls` must not be able to fail 2A, and a 2A
run must not be able to report a beta number it never computed. So this module
imports `envelope` (dependency-free), `sp_attention` and `accept` -- and nothing
else. `tests/test_envelope.py::TestImportIsolation` asserts that `compact` and
`record` are not even loaded in a fresh interpreter after importing this module.

Translation, not re-decisions
-----------------------------
The checks are 2A-01..05 and they already exist in `accept`. This adapter does
not re-derive them; it reports each one in the shared vocabulary and adds the
one row `accept` deliberately leaves out (communication), which on CPU can only
be *declared structural*. A status of PASS from `accept` becomes
PASS_CPU_SEMANTICS here, because the shared vocabulary has no other honest
reading of a float64 CPU reference run.
"""

import os
import platform
import sys

from . import accept as A
from . import envelope as E
from . import q8
from . import sp_attention as SP

_ACCEPT_TO_ENVELOPE = {
    A.PASS: E.PASS_CPU_SEMANTICS,
    A.FAIL: E.FAIL,
    A.NA: E.BLOCKED,
    A.BLOCKED: E.BLOCKED,
}


def _map(status):
    if status not in _ACCEPT_TO_ENVELOPE:
        raise ValueError("unknown 2A check status %r" % (status,))
    return _ACCEPT_TO_ENVELOPE[status]


def environment(extra=None):
    env = {
        "backend": "cpu_ref_f64",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    env.update(extra or {})
    return env


def default_files():
    """The modules that produced the numbers, hashed at record time."""
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("sp_attention.py", "state.py", "accept.py", "adapter_2a.py",
             "envelope.py", "q8.py")
    return [(n, E.file_sha256(os.path.join(here, n))) for n in names
            if os.path.exists(os.path.join(here, n))]


def _check(cid, group, status, detail, source=None):
    out = {"id": cid, "group": group, "status": status, "detail": detail}
    if source:
        out["source_check"] = source
    return out


def rank_topology_block(sp_result):
    """Where each rank sits and what it owns. Shared by 2A and any SP-bearing stage."""
    accounting = sp_result["accounting"]
    return E.applicable(
        world_size=sp_result["world_size"],
        L=sp_result["L"],
        per_rank=[{"rank": a["rank"], "own_span": list(a["own_span"]),
                   "own_len": a["own_len"],
                   "source_blocks": [list(b) for b in a["source_blocks"]]}
                  for a in accounting],
    )


def owner_kv_block(sp_result, full_bytes):
    """Dense KV actually resident on each rank. Never derived from traffic."""
    rows = []
    for a in sp_result["accounting"]:
        rows.append({
            "rank": a["rank"],
            "own_len": a["own_len"],
            "resident_kv_numel": a["resident_kv_numel"],
            "resident_kv_bytes": a["resident_kv_bytes"],
            "ratio_of_full": a["resident_kv_bytes"] / full_bytes if full_bytes else None,
            "dtype_bytes": a["dtype_bytes"],
        })
    return E.applicable(
        counted_for="dense KV resident on this rank",
        full_kv_bytes=full_bytes,
        per_rank=rows,
        source="layout_derived",
    )


def remote_staging_block(sp_result):
    """Transient remote spans. Kept apart from resident bytes on purpose."""
    rows = []
    for a in sp_result["accounting"]:
        rows.append({
            "rank": a["rank"],
            "remote_block_count": a["remote_block_count"],
            "staged_kv_numel": a["staged_kv_numel"],
            "staged_kv_bytes": a["staged_kv_bytes"],
        })
    return E.applicable(
        counted_for="remote KV staged transiently on this rank",
        per_rank=rows,
        note="staging is transient and is never summed with resident bytes",
    )


def no_full_copy_block(sp_result, full_bytes):
    holders = [a["rank"] for a in sp_result["accounting"]
               if a["resident_kv_bytes"] >= full_bytes - 1]
    return E.applicable(
        full_kv_bytes=full_bytes,
        ranks_holding_full_copy=holders,
        negative_arm="sp_forward(all_gather=True) is built and must be rejected",
    )


def communication_block(record=None, note=None):
    """CPU can only declare structure; a real run may supply measured traffic.

    Hardcoding `declared_structural` made this row BLOCKED forever, which would
    have made the whole 2A verdict unreachable on hardware -- and Q8 with it. A
    measured record is accepted only as `measured_nccl`, and the envelope refuses
    that without `measured_hardware` evidence level.
    """
    if record is None:
        return E.applicable(
            evidence="declared_structural",
            rounds=None,
            bytes_per_round=None,
            backend="none_cpu_in_process",
            note=note or ("no process, no NCCL, no all_to_all_v: block counts and "
                          "byte counts are derived from the partition layout"),
        )
    allowed = {"rounds", "bytes_per_round", "backend", "note"}
    unknown = set(record) - allowed
    if unknown:
        raise ValueError("communication record carries unknown keys %r"
                         % sorted(unknown))
    if not record.get("rounds") or record.get("bytes_per_round") is None:
        raise ValueError("a measured communication record needs rounds and "
                         "bytes_per_round; those are the numbers 2A-06 is about")
    return E.applicable(evidence="measured_nccl", **record)


def gqa_block(n_heads=None, n_kv_heads=None):
    if n_heads is None or n_kv_heads is None:
        return E.not_applicable("the CPU reference is single-head; GQA is not exercised")
    return E.applicable(n_heads=n_heads, n_kv_heads=n_kv_heads,
                        group_size=n_heads // n_kv_heads)


def numerical_block(num_check, cov_check, id_check, note=None):
    """The Q8 block: metric, tolerance policy, boundary coverage, identity.

    `tolerance_policy` is the validated object from `q8`, not a free string, so a
    reader can tell a frozen 1e-5 from a CPU test constant without guessing. The
    per-position numbers live here as diagnostics; the gate is `rel_l2 <=
    policy.threshold` plus full sampling plus a matching reference digest.
    """
    n = num_check["detail"]
    c = cov_check["detail"]
    i = id_check["detail"]
    return E.applicable(
        rel_l2=n["rel_l2"],
        max_abs=n["max_abs"],
        cosine=n["cosine"],
        tolerance=n["tolerance"],
        tolerance_source=n["tolerance_source"],
        tolerance_policy=n["tolerance_policy"],
        non_finite=n["non_finite"],
        shape=n["shape"],
        logit_agreement=n.get("logit_agreement"),
        merge_order_tested="forward",
        boundary_coverage={
            "status": cov_check["status"],
            "required": c["required"],
            "sampled": c["sampled"],
            "missing": c["missing"],
            "missing_reference": c["missing_reference"],
            "per_position": c["per_position"],
            "above_threshold": c["above_threshold"],
            "not_gating": c["not_gating"],
        },
        reference_identity={
            "status": id_check["status"],
            "path": i.get("path"),
            "sha256": i.get("sha256"),
            "expected_sha256": i.get("expected_sha256"),
        },
        note=note,
    )


def from_2a(sp_result, ref_output, tolerance, *, experiment_id, prov=None,
            work=None, scope_extra=None, tolerance_source="test_value",
            accept_result=None, n_heads=None, n_kv_heads=None,
            upstream_refs=None, notes=None, coverage=None, ref_identity=None,
            logits=None, communication=None):
    """Build the 2A envelope. The verdict can never be PASS on a CPU record."""
    if accept_result is None:
        accept_result = A.evaluate(sp_result, ref_output, tolerance,
                                   tolerance_source=tolerance_source,
                                   coverage=coverage,
                                   ref_identity=ref_identity,
                                   logits=logits)

    L = sp_result["L"]
    dk, dv = A._infer_dims(sp_result)
    full = SP.full_kv_bytes(L, dk, dv)

    by_id = {c["id"]: c for c in accept_result["checks"]}
    numeric = by_id["2A-04"]
    cov = by_id.get("2A-04b")
    ident = by_id.get("2A-04c")
    if cov is None or ident is None:
        raise ValueError("accept_result predates the Q8 rows (2A-04b/2A-04c); "
                         "re-run A.evaluate so the record and the verdict agree")

    comm_block = communication_block(communication)
    comm_status = E.PASS if comm_block.get("evidence") == "measured_nccl" \
        else E.BLOCKED

    checks = [
        _check("sp.rank_topology", "sp", _map(by_id["2A-01"]["status"]),
               {"n_ranks": len(sp_result["accounting"]),
                "world_size": sp_result["world_size"]}, source="2A-01"),
        _check("sp.owner_kv_resident", "sp", _map(by_id["2A-01"]["status"]),
               {"full_kv_bytes": full,
                "ratios": [r["resident_ratio"] for r in by_id["2A-01"]["detail"]]},
               source="2A-01"),
        _check("sp.remote_staging", "sp", _map(by_id["2A-03"]["status"]),
               {"remote_block_counts":
                [r["remote_block_count"] for r in by_id["2A-03"]["detail"]]},
               source="2A-03"),
        _check("sp.no_full_copy", "sp", _map(by_id["2A-02"]["status"]),
               by_id["2A-02"]["detail"], source="2A-02"),
        _check("sp.numerical", "sp", _map(numeric["status"]),
               {"rel_l2": numeric["detail"]["rel_l2"],
                "tolerance": numeric["detail"]["tolerance"],
                "tolerance_source": numeric["detail"]["tolerance_source"],
                "rel_l2_gate": True,
                "max_abs_gate": False,
                "cosine_gate": False}, source="2A-04"),
        _check("sp.numerical_coverage", "sp", _map(cov["status"]),
               {"required": cov["detail"]["required"],
                "missing": cov["detail"]["missing"],
                "above_threshold": cov["detail"]["above_threshold"]},
               source="2A-04b"),
        _check("sp.reference_identity", "sp", _map(ident["status"]),
               {"sha256": ident["detail"].get("sha256"),
                "expected_sha256": ident["detail"].get("expected_sha256")},
               source="2A-04c"),
        _check("sp.backend", "sp", _map(by_id["2A-05"]["status"]),
               {"note": by_id["2A-05"]["detail"].get("note"),
                "backend": sp_result["accounting"][0]["backend"]}, source="2A-05"),
        _check("sp.communication", "sp", comm_status,
               {"evidence": comm_block.get("evidence"),
                "rounds": comm_block.get("rounds"),
                "bytes_per_round": comm_block.get("bytes_per_round"),
                "note": comm_block.get("note")}, source=None),
    ]

    level = accept_result.get("evidence_level") or E.STRUCTURAL
    scope = {
        "evidence_level": level,
        "claims": ["dense_sp_sharding", "owner_kv_residency",
                   "remote_staging_layout", "no_full_kv_copy", "merge_numerics_cpu"],
        "does_not_claim": ["nccl_transport", "all_to_all_v_bandwidth",
                           "wall_clock_speedup", "compression_quality",
                           "real_model_numerics"],
    }
    scope.update(scope_extra or {})

    sp = {
        "status": E.APPLICABLE,
        "reason": None,
        "rank_topology": rank_topology_block(sp_result),
        "owner_kv": owner_kv_block(sp_result, full),
        "remote_staging": remote_staging_block(sp_result),
        "no_full_copy": no_full_copy_block(sp_result, full),
        "communication": comm_block,
        "gqa_repeat": gqa_block(n_heads, n_kv_heads),
        "numerical": numerical_block(numeric, cov, ident),
    }

    return E.new_envelope(
        E.STAGE_2A, experiment_id,
        scope=scope,
        prov=prov or E.provenance(
            files=default_files(),
            config={"world_size": sp_result["world_size"], "L": L,
                    "tolerance": tolerance,
                    "tolerance_source": numeric["detail"]["tolerance_source"],
                    "q8": q8.policy(numeric["detail"]["tolerance_source"],
                                    numeric["detail"]["tolerance"])},
            environment=environment(),
                                  generator="src/v2/cpu_ref/adapter_2a.py"),
        topo=E.topology(world_size=sp_result["world_size"],
                        ranks=[a["rank"] for a in sp_result["accounting"]],
                        source=None, destinations=[],
                        rank_map={a["rank"]: list(a["own_span"])
                                  for a in sp_result["accounting"]},
                        layout="contiguous_equal_split"),
        work=work or E.workload(
            model={"synthetic": True, "name": "synthetic_gaussian",
                   "revision": None, "note": "structure only, no checkpoint"},
            seq_len=L,
            position_ids={"start": 0, "end": L, "logical_len": L,
                          "rope_phase_verified": False,
                          "note": "no RoPE is applied in the reference"}),
        sp=sp,
        compact=E.not_applicable(
            "2A shards a dense KV cache; it never selects keys, fits beta or "
            "solves for values, so there is no compression diagnostic to report"),
        checks=checks,
        upstream_refs=upstream_refs,
        notes=notes,
    )
