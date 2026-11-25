"""One record envelope, three stages, no fabricated fields.

Why an envelope instead of "one merged report"
----------------------------------------------
2A and 2D measure different facts. 2A measures dense SP: shard ownership,
residency, remote staging, merge numerics. 2D measures a construction that runs
*on top of* real SP: key selection, beta fit, Value OLS, per-edge payload. If the
two are forced into one table, the 2A row for `beta_bind_hi` has to be filled
with something -- and any value there is invented. The envelope therefore keeps
one shell (provenance / topology / workload / tensor accounting / communication
vocabulary) and makes the compression block *structurally absent* on 2A.

Three statuses, deliberately not two
------------------------------------
  APPLICABLE      the stage executed this block and the numbers are its own
  NOT_APPLICABLE  the stage does not have this quantity at all (2A has no beta)
  MISSING         the stage should have it and this record does not

Collapsing NOT_APPLICABLE and MISSING into "null" is the mistake this module
exists to prevent: a reviewer cannot then tell "2A never fits a beta" from "the
2D record lost a beta block", and the second is a defect while the first is not.

What is enforced mechanically
-----------------------------
* `schema_version` is checked before anything else; a foreign version is refused,
  not coerced. A record whose fields were written against another definition is
  not readable, because the field meanings are what changed.
* stage, scope and field combination must agree -- presence alone is not enough.
  A 2A record carrying beta numbers is rejected even though every field parses.
* the per-edge payload can never be a pooled total: bytes are recomputed from
  that edge's own (t, dk, dv, dtype) and pooling keys are forbidden outright.
* the verdict is recomputed from the checks. A hand-edited "PASS" is refused,
  and no CPU evidence level can reach PASS at all.
* upstream references (2D citing 2A) are references, never measurements: they
  cannot fill a block this stage did not measure.

This module imports nothing from `compact` or `record`. That is a hard rule, not
a style choice: the 2A acceptor must not acquire a dependency on a solver it
never calls, or a bug in the solver becomes a 2A failure.
"""

import hashlib
import json

from . import q8

SCHEMA_VERSION = "dcckv-experiment-record/1"

STAGE_2A = "2A"
STAGE_2C = "2C"
STAGE_2D = "2D"
STAGES = (STAGE_2A, STAGE_2C, STAGE_2D)

# ---- block statuses -------------------------------------------------------
APPLICABLE = "APPLICABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
MISSING = "MISSING"
STATUSES = (APPLICABLE, NOT_APPLICABLE, MISSING)

# ---- verdicts -------------------------------------------------------------
PASS = "PASS"
PASS_CPU_SEMANTICS = "PASS_CPU_SEMANTICS"
BLOCKED = "BLOCKED_CPU_NO_GPU_EVIDENCE"
FAIL = "FAIL"

STRUCTURAL = "structural_cpu"        # numbers derived from a layout, not measured
MEASURED = "measured_hardware"       # numbers produced by a real device run
EVIDENCE_LEVELS = (STRUCTURAL, MEASURED)

# ---- requirement classes --------------------------------------------------
REQUIRED = "required"                    # absent => record defect => reject
REQUIRED_EVIDENCE = "required_evidence"  # absent => evidence gap  => blocked
FORBIDDEN = "forbidden"                  # present with numbers => reject

# The ruling's acceptance matrix, as data. `validate` is driven by this table,
# so a test can assert the table and the behaviour cannot silently diverge.
REQUIREMENT_TABLE = {
    "rank_topology": {
        STAGE_2A: REQUIRED, STAGE_2C: REQUIRED_EVIDENCE, STAGE_2D: REQUIRED_EVIDENCE},
    "owner_kv_resident": {
        STAGE_2A: REQUIRED, STAGE_2C: REQUIRED_EVIDENCE, STAGE_2D: REQUIRED_EVIDENCE},
    "remote_staging": {
        STAGE_2A: REQUIRED, STAGE_2C: REQUIRED_EVIDENCE, STAGE_2D: REQUIRED_EVIDENCE},
    "no_full_copy": {
        STAGE_2A: REQUIRED, STAGE_2C: REQUIRED_EVIDENCE, STAGE_2D: REQUIRED_EVIDENCE},
    "communication": {
        STAGE_2A: REQUIRED_EVIDENCE, STAGE_2C: REQUIRED_EVIDENCE,
        STAGE_2D: REQUIRED_EVIDENCE},
    "beta_binding": {
        STAGE_2A: FORBIDDEN, STAGE_2C: REQUIRED, STAGE_2D: REQUIRED},
    "beta_value_rank_cond": {
        STAGE_2A: FORBIDDEN, STAGE_2C: REQUIRED, STAGE_2D: REQUIRED},
    "mass_residual": {
        STAGE_2A: FORBIDDEN, STAGE_2C: REQUIRED, STAGE_2D: REQUIRED},
    "destination_semantics": {
        STAGE_2A: FORBIDDEN, STAGE_2C: FORBIDDEN, STAGE_2D: REQUIRED},
}

# Rows of the matrix that live inside `sp_structure`, and the sub-block each one
# is read from. Keeps the table and the JSON shape from drifting apart.
SP_ROWS = {
    "rank_topology": "rank_topology",
    "owner_kv_resident": "owner_kv",
    "remote_staging": "remote_staging",
    "no_full_copy": "no_full_copy",
}
COMPACT_ROWS = ("beta_binding", "beta_value_rank_cond", "mass_residual",
                "destination_semantics")

# Per-entry compression blocks every APPLICABLE compression record must carry.
ENTRY_BLOCKS = ("key_selection", "beta_design", "value_design",
                "beta_binding", "mass_residual", "payload")

# Claims a stage may make. Free text is not allowed here: a claim is a token, so
# "close enough to NCCL" cannot be expressed.
CLAIMS = {
    STAGE_2A: {"dense_sp_sharding", "owner_kv_residency", "remote_staging_layout",
               "no_full_kv_copy", "merge_numerics_cpu"},
    STAGE_2C: {"shared_query_construction", "per_budget_identity",
               "merge_mass_conservation"},
    STAGE_2D: {"destination_conditioned_construction", "per_edge_budget",
               "heldout_isolation", "position_metadata_preserved"},
}
# Only a real device run may use these, on any stage.
HARDWARE_CLAIMS = {"nccl_transport", "all_to_all_v_bandwidth", "wall_clock_speedup",
                   "real_model_numerics", "rope_phase_correct"}

# Keys that would turn a per-edge number into a pooled one.
POOLING_KEYS = ("total", "total_bytes", "sum_bytes", "grand_total", "bytes_total",
                "all_edges", "across_edges", "aggregate_bytes")
UPSTREAM_FILLERS = ("from_upstream", "upstream_value", "borrowed_from")

INDEX_BYTES = 4
META_BYTES = 32


class SchemaMismatchError(ValueError):
    """The record was written against a different field definition."""


class EnvelopeError(ValueError):
    """The record is well-formed JSON but its content contradicts itself."""


# ---------------------------------------------------------------------------
# status helpers
# ---------------------------------------------------------------------------

def _status(block):
    if not isinstance(block, dict):
        return None
    return block.get("status")


def classify_block(block):
    """Normalise any block into APPLICABLE / NOT_APPLICABLE / MISSING.

    An absent block is MISSING, never NOT_APPLICABLE: absence is the thing the
    ruling says must not be silently read as "does not apply".
    """
    st = _status(block)
    if st in STATUSES:
        return st
    return MISSING


def applicable(**payload):
    out = {"status": APPLICABLE}
    out.update(payload)
    return out


def not_applicable(reason):
    return {"status": NOT_APPLICABLE, "reason": reason}


def missing(reason):
    return {"status": MISSING, "reason": reason}


# ---------------------------------------------------------------------------
# payload accounting (shared vocabulary for 2A dense KV and 2C/2D compact KV)
# ---------------------------------------------------------------------------

def compact_payload_bytes(t, dk, dv, dtype_bytes, index_bytes=INDEX_BYTES,
                          meta_bytes=META_BYTES):
    """Bytes of one compact object: keys, values, beta, indices, meta."""
    return t * dk * dtype_bytes + t * dv * dtype_bytes + t * dtype_bytes \
        + t * index_bytes + meta_bytes


def check_payload_entry(payload, label="edge"):
    """Problems with one per-edge payload. Recompute, never trust."""
    problems = []
    if not isinstance(payload, dict):
        return ["%s: payload is not an object" % label]
    for k in POOLING_KEYS:
        if k in payload:
            problems.append(
                "%s: payload carries pooling key %r; a per-edge payload may not "
                "describe more than one edge" % (label, k))
    if payload.get("counted_for") != "this edge only":
        problems.append("%s: payload.counted_for is %r, must be 'this edge only'"
                        % (label, payload.get("counted_for")))
    try:
        t = int(payload["t"])
        dk = int(payload["dk"])
        dv = int(payload["dv"])
        dt = int(payload["dtype_bytes"])
        declared = int(payload["bytes"])
    except (KeyError, TypeError, ValueError):
        return problems + ["%s: payload lacks t/dk/dv/dtype_bytes/bytes" % label]
    expect = compact_payload_bytes(t, dk, dv, dt)
    if declared != expect:
        problems.append(
            "%s: payload.bytes = %d but this edge's own tensors give %d "
            "(t=%d, dk=%d, dv=%d, dtype_bytes=%d); a pooled or borrowed total "
            "looks exactly like this" % (label, declared, expect, t, dk, dv, dt))
    return problems


def check_no_double_count(entries):
    """Problems with a list of per-edge payloads taken together.

    The failure this catches is the one a merge step produces: three edges whose
    payload is replaced by the three-edge sum. Recomputation catches it because
    the sum cannot equal any single edge's own count, and the explicit
    "equals the pooled total" test names the cause instead of just rejecting.
    """
    problems = []
    declared, expected = [], []
    for i, e in enumerate(entries):
        p = e.get("payload") if isinstance(e, dict) else None
        problems.extend(check_payload_entry(p, label="entry%d" % i))
        if isinstance(p, dict) and {"t", "dk", "dv", "dtype_bytes"} <= set(p):
            declared.append(p.get("bytes"))
            expected.append(compact_payload_bytes(int(p["t"]), int(p["dk"]),
                                                  int(p["dv"]), int(p["dtype_bytes"])))
    if len(declared) > 1:
        pooled = sum(expected)
        hits = [i for i, b in enumerate(declared) if b == pooled]
        if hits:
            problems.append(
                "entries %r each report %d bytes, which is the sum of all %d "
                "edges; the same payload was counted once per edge"
                % (hits, pooled, len(entries)))
    return problems


# ---------------------------------------------------------------------------
# provenance / topology / workload
# ---------------------------------------------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha(obj):
    """Stable digest of a configuration dict, independent of key order."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def provenance(files=None, config=None, environment=None, generator=None,
               vcs="no_git_repository", commit=None, generated_at=None):
    """Where this record came from.

    `commit` is None when there is no repository. A commit hash is never
    invented to fill the slot -- the validator enforces the pairing.
    """
    return {
        "vcs": vcs,
        "commit": commit,
        "files": [{"path": p, "sha256": s} for p, s in (files or [])],
        "config_sha": config_sha(config) if config is not None else None,
        "environment": dict(environment or {}),
        "generator": generator,
        "generated_at": generated_at,
    }


def workload(model, seq_len, position_ids, dtype="float64"):
    """What was being run. `model.synthetic` must be stated, not implied."""
    return {
        "model": dict(model),
        "seq_len": int(seq_len),
        "position_ids": dict(position_ids),
        "dtype": dtype,
    }


def topology(world_size=None, ranks=None, source=None, destinations=None,
             rank_map=None, layout=None):
    return {
        "world_size": world_size,
        "ranks": list(ranks or []),
        "source": source,
        "destinations": list(destinations or []),
        "rank_map": dict(rank_map or {}),
        "layout": layout,
    }


def sp_structure(status_block, **blocks):
    """Assemble the SP block. Every row of the matrix gets an explicit status."""
    out = {"status": status_block["status"], "reason": status_block.get("reason")}
    out.update(blocks)
    return out


# ---------------------------------------------------------------------------
# envelope assembly
# ---------------------------------------------------------------------------

_REQUIRED_TOP = ("schema_version", "experiment_id", "stage", "scope",
                 "provenance", "topology", "workload", "sp_structure",
                 "compact_diagnostics", "verdict")


def new_envelope(stage, experiment_id, *, scope, prov, topo, work, sp,
                 compact, checks, upstream_refs=None, notes=None):
    """Assemble and validate one record. Raises rather than returning junk."""
    env = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "stage": stage,
        "scope": dict(scope),
        "provenance": prov,
        "topology": topo,
        "workload": work,
        "sp_structure": sp,
        "compact_diagnostics": compact,
        "upstream_refs": list(upstream_refs or []),
        "verdict": fold_verdict(checks, evidence_level=scope.get("evidence_level")),
        "notes": list(notes or []),
    }
    validate(env, strict=True)
    return env


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _check_schema(rec):
    v = rec.get("schema_version")
    if v != SCHEMA_VERSION:
        raise SchemaMismatchError(
            "record schema_version %r is not %r; fields written against another "
            "definition are not readable here" % (v, SCHEMA_VERSION))


def check_provenance(rec):
    p = rec.get("provenance") or {}
    problems = []
    vcs, commit = p.get("vcs"), p.get("commit")
    if vcs is None:
        problems.append("provenance.vcs is absent; state it even when there is none")
    elif vcs == "git":
        if not isinstance(commit, str) or len(commit) != 40 \
                or any(c not in "0123456789abcdef" for c in commit):
            problems.append("provenance says vcs=git but commit %r is not a 40-hex hash"
                            % (commit,))
    else:
        if commit is not None:
            problems.append("provenance.commit = %r with vcs = %r; an absent "
                            "repository has no commit to cite" % (commit, vcs))
    files = p.get("files")
    if not isinstance(files, list) or not files:
        problems.append("provenance.files is empty; no artefact is identified")
    else:
        for f in files:
            if not f.get("path") or not f.get("sha256"):
                problems.append("provenance.files entry lacks path/sha256: %r" % (f,))
    env = p.get("environment") or {}
    if not env.get("backend"):
        problems.append("provenance.environment.backend is absent")
    return problems


def check_workload(rec):
    w = rec.get("workload") or {}
    problems = []
    model = w.get("model") or {}
    if "synthetic" not in model:
        problems.append("workload.model.synthetic must be stated explicitly")
    elif not model["synthetic"]:
        if not model.get("name") or not model.get("revision"):
            problems.append("a real model needs both name and revision; HF 'main' "
                            "is not a revision")
    pos = w.get("position_ids") or {}
    for k in ("start", "end", "logical_len"):
        if pos.get(k) is None:
            problems.append("workload.position_ids.%s is absent" % k)
    if pos.get("start") is not None and pos.get("end") is not None \
            and pos["end"] < pos["start"]:
        problems.append("position_ids span is inverted")
    if "rope_phase_verified" not in pos:
        problems.append("workload.position_ids.rope_phase_verified must be stated; "
                        "carrying the metadata is not verifying the phase")
    return problems


def check_topology(rec):
    stage = rec.get("stage")
    t = rec.get("topology") or {}
    problems = []
    if stage == STAGE_2A:
        ws = t.get("world_size")
        if not isinstance(ws, int) or ws < 2:
            problems.append("2A needs world_size >= 2, got %r" % (ws,))
        ranks = t.get("ranks") or []
        if len(ranks) != (ws or -1):
            problems.append("2A ranks %r do not match world_size %r" % (ranks, ws))
        if t.get("destinations"):
            problems.append("2A has no destination set; destinations %r were "
                            "recorded" % (t.get("destinations"),))
    elif stage == STAGE_2C:
        if t.get("destinations"):
            problems.append("2C is the shared-query arm; it has no destination set")
    elif stage == STAGE_2D:
        dests = t.get("destinations") or []
        if len(dests) < 2:
            problems.append("2D needs at least two destinations, got %r" % (dests,))
        if t.get("source") is None:
            problems.append("2D needs an explicit source label")
        if t.get("source") in dests:
            problems.append("2D source and a destination share the label %r"
                            % (t.get("source"),))
    return problems


def check_scope(rec):
    s = rec.get("scope") or {}
    stage = rec.get("stage")
    level = s.get("evidence_level")
    problems = []
    if level not in EVIDENCE_LEVELS:
        problems.append("scope.evidence_level %r is not one of %r"
                        % (level, EVIDENCE_LEVELS))
    claims = s.get("claims")
    if not isinstance(claims, list) or not claims:
        problems.append("scope.claims is empty; a record must say what it claims")
        claims = []
    allowed = CLAIMS.get(stage, set())
    for c in claims:
        if c in HARDWARE_CLAIMS:
            if level != MEASURED:
                problems.append("claim %r needs measured hardware evidence, level is %r"
                                % (c, level))
        elif c not in allowed:
            problems.append("claim %r is not in the %s vocabulary %r"
                            % (c, stage, sorted(allowed)))
    dn = s.get("does_not_claim")
    if not isinstance(dn, list) or not dn:
        problems.append("scope.does_not_claim is empty; say what the record is not")
    else:
        for c in claims:
            if c in dn:
                problems.append("claim %r is also listed in does_not_claim" % c)
    model = (rec.get("workload") or {}).get("model") or {}
    if model.get("synthetic") is True and level == MEASURED:
        problems.append("workload.model is synthetic but the evidence level is "
                        "measured_hardware; a synthetic run cannot produce "
                        "hardware evidence, so this record is claiming a PASS "
                        "nothing measured")
    if model.get("synthetic") is True:
        for c in claims:
            if c in HARDWARE_CLAIMS:
                problems.append("synthetic workload cannot claim %r" % c)
    return problems


def check_sp_structure(rec):
    stage = rec.get("stage")
    sp = rec.get("sp_structure") or {}
    problems = []
    st = classify_block(sp)

    hard_rows = [r for r, cell in ((r, REQUIREMENT_TABLE[r][stage]) for r in SP_ROWS)
                 if cell == REQUIRED]
    if st != APPLICABLE:
        if not sp.get("reason"):
            problems.append("sp_structure is %s with no reason; a gap must say why"
                            % st)
        if st == NOT_APPLICABLE and hard_rows:
            problems.append("sp_structure is NOT_APPLICABLE but %s requires %r; this "
                            "stage cannot be inapplicable to its own SP layout"
                            % (stage, hard_rows))
        if hard_rows and st == MISSING:
            problems.append("sp_structure is MISSING but %s requires %r; a stage whose "
                            "own subject is SP must produce its SP record"
                            % (stage, hard_rows))
        leftover = [k for k in ("rank_topology", "owner_kv", "remote_staging",
                                "no_full_copy", "communication")
                    if sp.get(k) is not None]
        if leftover and st != APPLICABLE:
            problems.append("sp_structure is %s but still carries %r; a gap must not "
                            "be padded with numbers" % (st, leftover))
        return problems

    for row, sub in SP_ROWS.items():
        blk = sp.get(sub)
        bst = classify_block(blk)
        # Once the block claims "SP ran here", every row of the matrix has to be
        # filled. Whether a row is REQUIRED or only REQUIRED_EVIDENCE decides what
        # happens when the whole block is absent, not what a present block may skip.
        if bst != APPLICABLE:
            problems.append("sp_structure.%s is %s while the SP block claims a run "
                            "(%s)" % (sub, bst, (blk or {}).get("reason")))
        for filler in UPSTREAM_FILLERS:
            if isinstance(blk, dict) and filler in blk:
                problems.append("sp_structure.%s carries %r; an upstream reference "
                                "cannot supply this stage's own measurement"
                                % (sub, filler))

    gq = sp.get("gqa_repeat")
    if isinstance(gq, dict) and classify_block(gq) == APPLICABLE:
        nh, nkv = gq.get("n_heads"), gq.get("n_kv_heads")
        if not isinstance(nh, int) or not isinstance(nkv, int) or nkv <= 0:
            problems.append("sp_structure.gqa_repeat is APPLICABLE but n_heads/"
                            "n_kv_heads are %r/%r" % (nh, nkv))
        elif nh % nkv:
            problems.append("sp_structure.gqa_repeat: %d heads do not group over %d "
                            "kv heads" % (nh, nkv))
        elif gq.get("group_size") not in (None, nh // nkv):
            problems.append("sp_structure.gqa_repeat.group_size = %r, expected %d"
                            % (gq.get("group_size"), nh // nkv))

    # the communication row is required for every stage that ran SP, but a CPU
    # run can only declare structure -- "measured_nccl" needs measured evidence
    comm = sp.get("communication")
    if not isinstance(comm, dict):
        problems.append("sp_structure.communication is absent; the matrix row is "
                        "required for every stage")
    else:
        ev = comm.get("evidence")
        if ev not in ("declared_structural", "measured_nccl"):
            problems.append("sp_structure.communication.evidence %r is not "
                            "'declared_structural' or 'measured_nccl'" % (ev,))
        if ev == "measured_nccl" and \
                (rec.get("scope") or {}).get("evidence_level") != MEASURED:
            problems.append("communication claims a measured NCCL record while the "
                            "evidence level is not measured_hardware")

    # 2A-04 is a gate row, and Q8 defines what it may say. The policy block is
    # validated here so a record cannot report a number and quietly carry a
    # different tolerance, nor shrink its own coverage requirement.
    if stage == STAGE_2A:
        problems.extend(check_q8(rec, sp))
    return problems


def check_q8(rec, sp):
    """The 2A numeric rows: policy, coverage requirement, reference identity."""
    problems = []
    num = sp.get("numerical")
    if not isinstance(num, dict) or classify_block(num) != APPLICABLE:
        problems.append("sp_structure.numerical is %s; 2A-04 is a gate row and "
                        "cannot be absent on a 2A record"
                        % classify_block(num))
        return problems

    level = (rec.get("scope") or {}).get("evidence_level")
    pol = num.get("tolerance_policy")
    problems.extend(q8.validate_policy(
        pol, evidence_level=level,
        where="sp_structure.numerical.tolerance_policy"))

    flat = num.get("tolerance_source")
    if pol is not None and flat is not None and flat != pol.get("source"):
        problems.append("sp_structure.numerical.tolerance_source = %r disagrees "
                        "with tolerance_policy.source = %r"
                        % (flat, pol.get("source")))
    if isinstance(pol, dict) and num.get("tolerance") != pol.get("threshold"):
        problems.append("sp_structure.numerical.tolerance = %r disagrees with "
                        "its policy threshold %r"
                        % (num.get("tolerance"), pol.get("threshold")))

    seq = (rec.get("workload") or {}).get("seq_len")
    ws = (rec.get("topology") or {}).get("world_size")
    bc = num.get("boundary_coverage")
    if not isinstance(bc, dict):
        problems.append("sp_structure.numerical.boundary_coverage is absent; the "
                        "frozen positions must be declared even when unsampled")
        return problems
    if isinstance(seq, int) and isinstance(ws, int) and seq > 0 and ws > 0:
        expected = list(q8.boundary_positions(seq, ws))
        if bc.get("required") != expected:
            problems.append(
                "sp_structure.numerical.boundary_coverage.required = %r, but the "
                "frozen rule for L=%d, world_size=%d gives %r; a record may not "
                "shrink its own coverage requirement"
                % (bc.get("required"), seq, ws, expected))
        sampled = set(bc.get("sampled") or [])
        missing = [p for p in expected if p not in sampled]
        if missing:
            row = [c for c in (rec.get("verdict") or {}).get("checks") or []
                   if c.get("id") == "sp.numerical_coverage"]
            if row and row[0].get("status") == PASS:
                problems.append(
                    "boundary coverage is missing %r but sp.numerical_coverage "
                    "says PASS" % (missing,))

    ident = num.get("reference_identity")
    if not isinstance(ident, dict):
        problems.append("sp_structure.numerical.reference_identity is absent; a "
                        "comparison whose reference cannot be identified is invalid")
    elif level == MEASURED:
        sha, exp = ident.get("sha256"), ident.get("expected_sha256")
        if not sha or not exp:
            problems.append("reference_identity lacks a digest on a measured run; "
                            "the reference must be identified by SHA256")
        elif sha != exp:
            problems.append("reference_identity.sha256 does not match "
                            "expected_sha256; the comparison is invalid, not large")
    return problems


def _entry_blocks(entry):
    """Which required blocks this entry does not actually carry.

    `payload` has no status of its own, so it is judged by content: a dict with a
    byte count and the per-edge label. Everything else must say APPLICABLE --
    a block that says NOT_APPLICABLE on a compression stage is a hole, not an
    exemption.
    """
    absent = [b for b in ENTRY_BLOCKS if b != "payload"
              and classify_block(entry.get(b)) != APPLICABLE]
    p = entry.get("payload")
    if not isinstance(p, dict) or "bytes" not in p:
        absent.append("payload")
    return absent


def check_compact(rec):
    stage = rec.get("stage")
    cd = rec.get("compact_diagnostics") or {}
    st = classify_block(cd)
    problems = []

    forbidden_here = all(REQUIREMENT_TABLE[r][stage] == FORBIDDEN for r in COMPACT_ROWS)
    if forbidden_here:
        if st != NOT_APPLICABLE:
            problems.append(
                "%s does not execute selection/beta/Value OLS, so compact_diagnostics "
                "must be NOT_APPLICABLE; it is %s" % (stage, st))
        for k in ("entries", "total_payload", "beta", "values", "selected_indices",
                  "beta_binding", "mass_residual", "beta_design", "value_design"):
            if cd.get(k) not in (None, [], {}):
                problems.append(
                    "compact_diagnostics.%s is populated on %s; the stage never fit "
                    "these quantities, so the value cannot be a measurement" % (k, stage))
        return problems

    if st != APPLICABLE:
        problems.append("compact_diagnostics is %s but %s executes compression; a "
                        "missing block here is incomplete evidence, not an "
                        "inapplicable one" % (st, stage))
        return problems

    if cd.get("total_payload") is not None:
        problems.append("compact_diagnostics.total_payload exists; payloads are per "
                        "edge and there is no cross-edge total")

    entries = cd.get("entries")
    if not isinstance(entries, list) or not entries:
        return problems + ["compact_diagnostics.entries is empty"]

    for i, e in enumerate(entries):
        label = "entries[%d]" % i
        absent = _entry_blocks(e)
        if absent:
            problems.append("%s lacks %r; every edge carries its own construction "
                            "diagnostics" % (label, absent))
            continue
        problems.extend(_check_entry_numbers(e, label))

    dests = (rec.get("topology") or {}).get("destinations") or []
    if stage == STAGE_2D:
        if len(entries) != len(dests):
            problems.append("2D has %d destinations but %d diagnostic entries; each "
                            "Source->Destination edge needs its own block"
                            % (len(dests), len(entries)))
        seen = []
        for i, e in enumerate(entries):
            edge = e.get("edge") or {}
            if edge.get("source") is None or edge.get("destination") is None:
                problems.append("entries[%d].edge must name source and destination" % i)
                continue
            if edge["destination"] not in dests:
                problems.append("entries[%d] describes destination %r, which is not "
                                "in topology.destinations %r"
                                % (i, edge["destination"], dests))
            seen.append(edge["destination"])
        if len(set(seen)) != len(seen):
            problems.append("two entries describe the same destination: %r" % (seen,))
    else:
        for i, e in enumerate(entries):
            if e.get("edge"):
                problems.append("entries[%d] names an edge on %s, which has no "
                                "destination topology" % (i, stage))

    problems.extend(check_no_double_count(entries))

    if stage == STAGE_2D:
        ds = cd.get("destination_semantics")
        if classify_block(ds) != APPLICABLE:
            problems.append("compact_diagnostics.destination_semantics must be "
                            "APPLICABLE on 2D")
        else:
            hashes = ds.get("query_hashes") or []
            if len(hashes) != len(dests):
                problems.append("destination_semantics.query_hashes has %d entries "
                                "for %d destinations" % (len(hashes), len(dests)))
            if ds.get("shared_query_set") is True and len(set(hashes)) > 1:
                problems.append("destination_semantics says the query set is shared "
                                "but the recorded hashes differ")
            if not isinstance(ds.get("per_destination_queries"), bool):
                problems.append("destination_semantics.per_destination_queries must "
                                "be an explicit boolean")
    return problems


def _check_entry_numbers(e, label):
    problems = []
    t = (e.get("payload") or {}).get("t")

    bd = e["beta_design"]
    vd = e["value_design"]
    for name, blk in (("beta_design", bd), ("value_design", vd)):
        if blk.get("rank") is None or blk.get("cond") is None:
            problems.append("%s.%s lacks rank/cond" % (label, name))
        if t is not None and blk.get("p") != t:
            problems.append("%s.%s.p = %r but the edge holds t = %r"
                            % (label, name, blk.get("p"), t))
        if blk.get("method") is None:
            problems.append("%s.%s.method is absent; a rank without the method that "
                            "produced it is not comparable" % (label, name))
    if bd.get("rank") is not None and vd.get("rank") is not None \
            and bd.get("cond") is not None and vd.get("cond") is not None:
        if bd.get("problem") == vd.get("problem"):
            problems.append("%s: beta_design and value_design name the same LS "
                            "problem %r; they are different systems"
                            % (label, bd.get("problem")))

    bb = e["beta_binding"]
    if bb.get("lo") is None or bb.get("hi") is None:
        problems.append("%s.beta_binding lacks the box bounds" % label)
    elif bb["lo"] >= bb["hi"]:
        problems.append("%s.beta_binding box is empty: [%r, %r]"
                        % (label, bb["lo"], bb["hi"]))
    if bb.get("bind_tol") is None:
        problems.append("%s.beta_binding lacks bind_tol" % label)
    for k in ("bind_hi_frac", "bind_lo_frac"):
        f = bb.get(k)
        if f is None:
            problems.append("%s.beta_binding.%s is absent" % (label, k))
        elif not (0.0 <= f <= 1.0):
            problems.append("%s.beta_binding.%s out of [0,1]: %r" % (label, k, f))

    mr = e["mass_residual"]
    if mr.get("aggregation") is None:
        problems.append("%s.mass_residual.aggregation is absent; per-query and "
                        "pooled values are not comparable" % label)
    if mr.get("normalized") is None:
        problems.append("%s.mass_residual.normalized is absent" % label)
    elif mr["normalized"] < 0.0:
        problems.append("%s.mass_residual.normalized is negative" % label)

    ks = e["key_selection"]
    for k in ("selector", "budget_tokens", "n_ref", "query_hash"):
        if ks.get(k) is None:
            problems.append("%s.key_selection.%s is absent" % (label, k))
    if ks.get("budget_tokens") is not None and t is not None \
            and ks["budget_tokens"] != t:
        problems.append("%s.key_selection.budget_tokens = %r but payload.t = %r"
                        % (label, ks["budget_tokens"], t))
    return problems


def check_verdict(rec):
    """The verdict must be the fold of its own checks, not an assertion."""
    v = rec.get("verdict") or {}
    problems = []
    checks = v.get("checks")
    if not isinstance(checks, list) or not checks:
        return ["verdict.checks is empty; a verdict needs its rows"]
    for c in checks:
        if not c.get("id") or not c.get("group"):
            problems.append("verdict check lacks id/group: %r" % (c,))
        if c.get("status") not in (PASS, PASS_CPU_SEMANTICS, BLOCKED, FAIL):
            problems.append("verdict check %r has unknown status %r"
                            % (c.get("id"), c.get("status")))
    ids = [c.get("id") for c in checks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate check ids: %r" % (ids,))

    level = (rec.get("scope") or {}).get("evidence_level")
    folded = fold_verdict(checks, evidence_level=level)
    if v.get("overall") != folded["overall"]:
        problems.append("verdict.overall = %r but the checks fold to %r"
                        % (v.get("overall"), folded["overall"]))
    if v.get("overall") == PASS and level != MEASURED:
        problems.append("a %r record claims PASS; the ceiling is %r"
                        % (level, PASS_CPU_SEMANTICS))
    if v.get("ceiling") != folded["ceiling"]:
        problems.append("verdict.ceiling = %r, expected %r"
                        % (v.get("ceiling"), folded["ceiling"]))
    if sorted(v.get("blocked_requires_gpu") or []) != sorted(folded["blocked_requires_gpu"]):
        problems.append("verdict.blocked_requires_gpu disagrees with the checks")

    # a group with no rows may not also claim numbers (2A must not grow a
    # compact row just because the serialiser felt like it)
    groups = {c.get("group") for c in checks}
    compact_status = classify_block(rec.get("compact_diagnostics"))
    if compact_status == APPLICABLE and "compact" not in groups:
        problems.append("compact_diagnostics is APPLICABLE but no compact check "
                        "row exists")
    if compact_status != APPLICABLE and "compact" in groups:
        problems.append("compact_diagnostics is %s but compact check rows exist"
                        % compact_status)

    # a stage that did not run SP cannot fold to a pass: the gap has to be a
    # visible BLOCKED row, not a silent absence
    sp_status = classify_block(rec.get("sp_structure"))
    if sp_status != APPLICABLE:
        blocked_sp = [c for c in checks
                      if c.get("group") == "sp" and c.get("status") == BLOCKED]
        if not blocked_sp:
            problems.append("sp_structure is %s but no SP row is BLOCKED; a stage "
                            "that did not run SP cannot fold to a pass" % sp_status)
    return problems


def validate(rec, strict=True):
    """Return a list of problems; raise if strict and any are found."""
    _check_schema(rec)          # always fatal: the fields may mean other things

    problems = []
    for k in _REQUIRED_TOP:
        if k not in rec:
            problems.append("missing top-level field: %s" % k)
    if problems:
        if strict:
            raise EnvelopeError("; ".join(problems))
        return problems

    if rec["stage"] not in STAGES:
        problems.append("stage %r is not one of %r" % (rec["stage"], STAGES))
        if strict:
            raise EnvelopeError("; ".join(problems))
        return problems
    if not rec.get("experiment_id"):
        problems.append("experiment_id is empty; records must be traceable")

    for fn in (check_provenance, check_workload, check_topology, check_scope,
               check_sp_structure, check_compact, check_verdict):
        problems.extend(fn(rec))

    if strict and problems:
        raise EnvelopeError("; ".join(problems))
    return problems


# ---------------------------------------------------------------------------
# verdict folding
# ---------------------------------------------------------------------------

def _fold_group(statuses):
    if not statuses:
        return NOT_APPLICABLE
    if any(s == FAIL for s in statuses):
        return FAIL
    if any(s == BLOCKED for s in statuses):
        return BLOCKED
    if any(s == PASS_CPU_SEMANTICS for s in statuses):
        return PASS_CPU_SEMANTICS
    return PASS


def fold_verdict(checks, evidence_level=None):
    """Single fold used by both construction and re-verification."""
    ceiling = PASS if evidence_level == MEASURED else PASS_CPU_SEMANTICS
    groups = {}
    for g in ("sp", "compact", "destination"):
        groups[g] = _fold_group([c.get("status") for c in checks
                                 if c.get("group") == g])
    failed = [c["id"] for c in checks if c.get("status") == FAIL]
    blocked = [c["id"] for c in checks if c.get("status") == BLOCKED]
    if failed:
        overall = FAIL
    elif blocked:
        overall = BLOCKED
    elif ceiling == PASS:
        overall = PASS
    else:
        overall = PASS_CPU_SEMANTICS
    # a hardware-only stage can never be settled above the ceiling its evidence
    # supports, even if every recorded row happens to read PASS
    if overall == PASS and ceiling != PASS:
        overall = PASS_CPU_SEMANTICS
    return {
        "overall": overall,
        "ceiling": ceiling,
        "evidence_level": evidence_level,
        "partial": groups,
        "blocked_requires_gpu": blocked,
        "failed": failed,
        "checks": list(checks),
    }


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def from_json(text):
    """Parse, refusing a foreign schema before anything else is touched."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise SchemaMismatchError("record is not a JSON object")
    _check_schema(obj)
    return obj


def to_json(rec):
    return json.dumps(rec, indent=2, ensure_ascii=False, allow_nan=False)


def summary(rec):
    """One-line-per-row reading aid, never a substitute for the record."""
    v = rec.get("verdict") or {}
    return {
        "experiment_id": rec.get("experiment_id"),
        "stage": rec.get("stage"),
        "overall": v.get("overall"),
        "partial": v.get("partial"),
        "blocked_requires_gpu": v.get("blocked_requires_gpu"),
        "compact_status": classify_block(rec.get("compact_diagnostics")),
        "sp_status": classify_block(rec.get("sp_structure")),
    }
