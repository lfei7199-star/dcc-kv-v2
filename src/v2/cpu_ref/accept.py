"""2A structural acceptance checker (CPU-side twin of the frozen 2A-01..05).

Design rules taken from the frozen doc:
  - the all-gather negative arm must be rejected by *this same checker*,
    not by a human writing FAIL by hand
  - resident tensor bytes are computed from numel * element_size, never from
    communication payload
  - the numeric gate is Q8, frozen at rel-L2 <= 1e-5 (see `q8.py`): a measured
    run may cite only the pre-registered value, and 1e-9 stays a CPU test
    constant
  - on CPU this checker can never return PASS: 2A-05 (backend consistency)
    needs a real device, so the best possible verdict is BLOCKED

Q8 is three rows, not one
-------------------------
The ruling's gate is a conjunction, so it is checked as one:

  2A-04   rel_l2 <= 1e-5, with NaN/Inf fatal whatever rel_l2 says
  2A-04b  every frozen boundary position sampled and finite
  2A-04c  the reference identity / SHA256 matches

Keeping them apart is what stops "rel_l2 is 3e-6" from being read as "the run is
fine" when one position returned Inf or the reference was a different file.
`max_abs` and `cosine` are recorded and never gate; per-position rel_l2 is
recorded but not gated, because Q8 froze one global threshold and inventing a
local one after seeing these numbers would be the post-hoc adjustment the ruling
forbids.

`attach_envelope` is how a 2A run acquires its shared record. It validates the
envelope and refuses one that claims PASS, names another stage, or disagrees with
the checks below. This module still imports nothing from `compact`: the 2A
acceptor must not be able to fail because of a solver it never calls.
"""

import math

from . import envelope as E
from . import q8
from .sp_attention import full_kv_bytes

PASS = E.PASS
FAIL = E.FAIL
NA = "NOT_APPLICABLE_NO_CUDA"
BLOCKED = E.BLOCKED

SHARD_RATIO_TOL = 0.02


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------

def rel_l2(a, b):
    num = 0.0
    den = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            num += (x - y) ** 2
            den += y ** 2
    return math.sqrt(num) / (math.sqrt(den) + 1e-300)


def max_abs(a, b):
    m = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = abs(x - y)
            if d > m:
                m = d
    return m


def cosine(a, b):
    num = den1 = den2 = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            num += x * y
            den1 += x * x
            den2 += y * y
    return num / (math.sqrt(den1) * math.sqrt(den2) + 1e-300)


def _shape(rows):
    if not rows:
        return [0, 0]
    widths = {len(r) for r in rows}
    return [len(rows), widths.pop() if len(widths) == 1 else -1]


def _non_finite(rows):
    """(count, first location). Ragged or non-numeric rows count as bad."""
    n = 0
    first = None
    for i, row in enumerate(rows):
        for j, x in enumerate(row):
            ok = isinstance(x, (int, float)) and math.isfinite(x)
            if not ok:
                n += 1
                if first is None:
                    first = [i, j]
    return n, first


def _norm_rows(d):
    if not isinstance(d, dict):
        return {}
    return {int(k): v for k, v in d.items()}


def evidence_level_of(sp_result):
    """Structural or measured, and it may not be downgraded.

    A run whose accounting says `cuda*` cannot be filed as `structural_cpu`: that
    would let real hardware evidence duck every ceiling in the vocabulary, Q8
    included.
    """
    level = sp_result.get("evidence_level") or E.STRUCTURAL
    if level not in E.EVIDENCE_LEVELS:
        raise ValueError("evidence_level %r is not one of %r"
                         % (level, E.EVIDENCE_LEVELS))
    backends = {str(a.get("backend")) for a in sp_result["accounting"]}
    non_cpu = {b for b in backends if not b.startswith("cpu")}
    if level == E.STRUCTURAL and non_cpu:
        raise ValueError("a run with backend(s) %r cannot be recorded as %r"
                         % (sorted(non_cpu), E.STRUCTURAL))
    return level


# ---------------------------------------------------------------------------
# 2A-01 .. 2A-03: structure
# ---------------------------------------------------------------------------

def check_shard(accounting, L, dk, dv, expected_world_size):
    """2A-01: each rank's resident KV must be ~1/n of the full KV."""
    full = full_kv_bytes(L, dk, dv)
    expect = 1.0 / expected_world_size
    rows, ok = [], True
    for a in accounting:
        ratio = a["resident_kv_bytes"] / full
        good = abs(ratio - expect) <= SHARD_RATIO_TOL
        ok = ok and good
        rows.append({"rank": a["rank"], "resident_ratio": ratio,
                     "expected": expect, "ok": good})
    return {"id": "2A-01", "status": PASS if ok else FAIL, "detail": rows}


def check_remote_blocks(accounting):
    """2A-03: rank r must receive exactly r remote source blocks (0/1/2/3)."""
    rows, ok = [], True
    for a in accounting:
        expect = a["rank"]
        good = a["remote_block_count"] == expect
        ok = ok and good
        rows.append({"rank": a["rank"],
                     "remote_block_count": a["remote_block_count"],
                     "expected": expect, "ok": good})
    return {"id": "2A-03", "status": PASS if ok else FAIL, "detail": rows}


def check_no_full_copy(accounting, L, dk, dv):
    """2A-02: no rank may hold the full K/V."""
    full = full_kv_bytes(L, dk, dv)
    offenders = [a["rank"] for a in accounting
                 if a["resident_kv_bytes"] >= full - 1]
    return {
        "id": "2A-02",
        "status": PASS if not offenders else FAIL,
        "detail": {"ranks_holding_full_copy": offenders},
    }


# ---------------------------------------------------------------------------
# 2A-04: the Q8 numeric gate
# ---------------------------------------------------------------------------

def check_numeric(sp_output, ref_output, tolerance, *, tolerance_source=None,
                  evidence_level=None, logits=None):
    """2A-04: agreement with the Q8 gate.

    The tolerance is never chosen here. Without one the check raises, and a
    number labelled `preregistered` that is not 1e-5 is refused outright.
    """
    if tolerance is None:
        raise ValueError("2A-04 requires a pre-registered tolerance; refusing to "
                         "pick one after seeing results")

    src = tolerance_source or q8.SOURCE_CALLER
    pol = q8.policy(src, float(tolerance))
    problems = q8.validate_policy(pol, evidence_level=evidence_level)
    if problems:
        raise ValueError("refusing this 2A-04 tolerance: " + "; ".join(problems))

    sp_shape = _shape(sp_output)
    ref_shape = _shape(ref_output)
    detail = {
        "rel_l2": None,
        "max_abs": None,
        "cosine": None,
        "tolerance": float(tolerance),
        "tolerance_source": src,
        "tolerance_policy": pol,
        "shape": {"sp": sp_shape, "ref": ref_shape},
        "non_finite": {"sp": 0, "ref": 0},
        "non_finite_first": {"sp": None, "ref": None},
        "logit_agreement": None,
        "reason": None,
        "gates": ["rel_l2 <= threshold", "no NaN/Inf", "reference identity (2A-04c)"],
    }

    # A dropped or duplicated rank shows up as a shape difference. `zip` would
    # silently truncate and report a beautiful rel_l2 over the rows that happen
    # to line up, so shape is checked before any arithmetic.
    if sp_shape != ref_shape or sp_shape[1] < 0 or ref_shape[1] < 0:
        detail["reason"] = "shape_mismatch"
        return {"id": "2A-04", "status": FAIL, "detail": detail}

    n_sp, first_sp = _non_finite(sp_output)
    n_ref, first_ref = _non_finite(ref_output)
    detail["non_finite"] = {"sp": n_sp, "ref": n_ref}
    detail["non_finite_first"] = {"sp": first_sp, "ref": first_ref}

    if n_ref:
        # Not a large error -- the reference itself is unusable, so there is
        # nothing to be close to or far from.
        detail["reason"] = "reference_not_finite"
        return {"id": "2A-04", "status": BLOCKED, "detail": detail}

    if n_sp:
        # Deliberately before the rel-L2 comparison: a tensor can have a tiny
        # relative error and still contain Inf where the reference is finite.
        detail["reason"] = "non_finite_in_sp_output"
        return {"id": "2A-04", "status": FAIL, "detail": detail}

    rl2 = rel_l2(sp_output, ref_output)
    detail["rel_l2"] = rl2
    detail["max_abs"] = max_abs(sp_output, ref_output)
    detail["cosine"] = cosine(sp_output, ref_output)

    if logits is not None:
        sp_logits, ref_logits = logits
        k = 5
        detail["logit_agreement"] = q8.topk_agreement(sp_logits, ref_logits, k=k)
        detail["logit_agreement"]["gating"] = False

    ok = rl2 <= float(tolerance)
    if not ok:
        detail["reason"] = "rel_l2_above_threshold"
    return {"id": "2A-04", "status": PASS if ok else FAIL, "detail": detail}


# ---------------------------------------------------------------------------
# 2A-04b: the frozen boundary positions
# ---------------------------------------------------------------------------

def check_boundary_coverage(coverage, L, world_size, tolerance=None,
                            sp_output=None):
    """2A-04b: every frozen position must have finite output on both sides.

    The ruling made this part of the gate: a global rel-L2 that passes must not
    be able to hide a boundary that returned nothing, or Inf, or a different
    number of rows.

    Per-position rel_l2 is recorded and named when it exceeds the global
    threshold, but it does not gate -- Q8 froze one global threshold, and a local
    one invented after looking at these numbers would be exactly the post-hoc
    adjustment the ruling forbids. The residual risk is written down in
    design/17.
    """
    required = list(q8.boundary_positions(L, world_size))
    if not coverage:
        return {
            "id": "2A-04b", "status": BLOCKED,
            "detail": {"required": required, "sampled": [], "missing": required,
                       "missing_reference": required,
                       "per_position": [], "above_threshold": [],
                       "not_gating": ["per-position rel_l2 magnitude"],
                       "reason": "no boundary sample supplied; a global rel-L2 "
                                 "cannot rule out a local anomaly at a shard "
                                 "boundary"},
        }

    sp_rows = _norm_rows(coverage.get("sp_rows"))
    ref_rows = _norm_rows(coverage.get("ref_rows"))
    sampled = sorted(sp_rows) if sp_rows else []
    missing = [p for p in required if p not in sp_rows]
    missing_ref = [p for p in required if p not in ref_rows]

    per_position = []
    bad_sp = []
    bad_ref = []
    above = []
    mismatched = []
    for p in required:
        if p not in sp_rows or p not in ref_rows:
            continue
        row_sp, row_ref = sp_rows[p], ref_rows[p]
        # The sampled row has to be the row the global comparison used. A
        # separate copy that happens to look fine is a bookkeeping claim, not
        # coverage.
        if sp_output is not None and p < len(sp_output) and \
                list(row_sp) != list(sp_output[p]):
            mismatched.append(p)
        n_sp = _non_finite([row_sp])[0]
        n_ref = _non_finite([row_ref])[0]
        if n_ref:
            bad_ref.append(p)
        if n_sp:
            bad_sp.append(p)
        entry = {"position": p, "rel_l2": None, "max_abs": None,
                 "finite_sp": n_sp == 0, "finite_ref": n_ref == 0}
        if not n_sp and not n_ref and len(row_sp) == len(row_ref):
            entry["rel_l2"] = rel_l2([row_sp], [row_ref])
            entry["max_abs"] = max_abs([row_sp], [row_ref])
            if tolerance is not None and entry["rel_l2"] > float(tolerance):
                above.append(p)
        elif len(row_sp) != len(row_ref):
            entry["width_mismatch"] = [len(row_sp), len(row_ref)]
            bad_sp.append(p)
        per_position.append(entry)

    detail = {
        "required": required,
        "sampled": sampled,
        "missing": missing,
        "missing_reference": missing_ref,
        "per_position": per_position,
        "above_threshold": above,
        "rows_identical_to_global_output": not mismatched if sp_output is not None else None,
        "gating": ["all required positions sampled", "all finite",
                   "reference present at every position"],
        "not_gating": ["per-position rel_l2 magnitude"],
        "reason": None,
    }

    if mismatched:
        detail["reason"] = "coverage_rows_differ_from_compared_output"
        detail["mismatched_positions"] = mismatched
        return {"id": "2A-04b", "status": FAIL, "detail": detail}
    if bad_sp:
        detail["reason"] = "non_finite_or_width_mismatch_at_boundary"
        return {"id": "2A-04b", "status": FAIL, "detail": detail}
    if missing:
        detail["reason"] = "required_positions_not_sampled"
        return {"id": "2A-04b", "status": BLOCKED, "detail": detail}
    if missing_ref:
        detail["reason"] = "reference_missing_at_required_position"
        return {"id": "2A-04b", "status": BLOCKED, "detail": detail}
    if bad_ref:
        detail["reason"] = "reference_not_finite_at_boundary"
        return {"id": "2A-04b", "status": BLOCKED, "detail": detail}
    return {"id": "2A-04b", "status": PASS, "detail": detail}


# ---------------------------------------------------------------------------
# 2A-04c: reference identity
# ---------------------------------------------------------------------------

def check_reference_identity(ref_identity, evidence_level=None):
    """2A-04c: the reference must be the file the manifest promised.

    A reference that cannot be identified is not a reference. This row is BLOCKED
    rather than FAIL on purpose: a SHA mismatch means the comparison is invalid,
    not that SP disagrees with it.
    """
    if not ref_identity:
        return {
            "id": "2A-04c", "status": BLOCKED,
            "detail": {"reason": "no reference identity supplied; the comparison "
                                 "cannot be reproduced from this record",
                       "sha256": None, "expected_sha256": None},
        }

    sha = ref_identity.get("sha256")
    expected = ref_identity.get("expected_sha256")
    detail = {"path": ref_identity.get("path"), "sha256": sha,
              "expected_sha256": expected, "source": ref_identity.get("source"),
              "reason": None}

    if sha and expected and sha != expected:
        detail["reason"] = "reference_sha_mismatch"
        return {"id": "2A-04c", "status": BLOCKED, "detail": detail}

    if evidence_level == E.MEASURED and not (sha and expected):
        detail["reason"] = "reference_identity_unverifiable"
        return {"id": "2A-04c", "status": BLOCKED, "detail": detail}

    if evidence_level != E.MEASURED and not sha:
        detail["reason"] = "reference_has_no_digest"
        return {"id": "2A-04c", "status": BLOCKED, "detail": detail}

    if not expected:
        detail["note"] = ("a digest is recorded but there is no manifest digest "
                          "to compare it against")
    return {"id": "2A-04c", "status": PASS, "detail": detail}


# ---------------------------------------------------------------------------
# 2A-05: backend
# ---------------------------------------------------------------------------

def check_backend(accounting, evidence_level=None):
    """2A-05: one backend, explicitly fixed, and not a CPU stand-in.

    Until now this row could only return NA, which made 2A unreachable on real
    hardware and Q8 decorative. The testable part is uniformity plus "not the CPU
    reference": whether the string is honest is a job-manifest question, not
    something this checker can decide.
    """
    backends = sorted({str(a.get("backend")) for a in accounting})
    detail = {"backends": backends, "evidence_level": evidence_level,
              "note": None}

    if len(backends) > 1:
        detail["note"] = "ranks disagree on the backend; a mixed run has no single numerics"
        return {"id": "2A-05", "status": FAIL, "detail": detail}

    b = backends[0] if backends else None
    if b is None or b.startswith("cpu"):
        detail["note"] = ("CPU reference cannot satisfy backend consistency; "
                          "a real device run is required")
        return {"id": "2A-05", "status": NA, "detail": detail}

    if evidence_level != E.MEASURED:
        detail["note"] = ("a device backend without measured_hardware evidence "
                          "level; the claim and its evidence disagree")
        return {"id": "2A-05", "status": NA, "detail": detail}

    detail["note"] = ("uniform device backend with measured evidence; device "
                      "identity is audited from the job manifest, not here")
    return {"id": "2A-05", "status": PASS, "detail": detail}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def evaluate(sp_result, ref_output, tolerance, envelope=None, *,
             tolerance_source=None, coverage=None, ref_identity=None,
             logits=None):
    """Run all 2A checks. Overall verdict can never be PASS on a CPU record.

    `envelope` is the shared record built by `adapter_2a.from_2a`. It is
    validated and attached, never used to change a check: the record describes
    this verdict, it does not produce it.

    The Q8 tolerance is validated before any check runs, so a measured run with a
    test-value tolerance fails loudly instead of quietly reporting a number.
    """
    level = evidence_level_of(sp_result)
    src = tolerance_source or q8.SOURCE_CALLER
    if tolerance is not None:
        problems = q8.validate_policy(q8.policy(src, float(tolerance)),
                                      evidence_level=level)
        if problems:
            raise ValueError("2A tolerance rejected: " + "; ".join(problems))

    L = sp_result["L"]
    dk, dv = _infer_dims(sp_result)
    world_size = sp_result["world_size"]

    checks = [
        check_shard(sp_result["accounting"], L, dk, dv, world_size),
        check_no_full_copy(sp_result["accounting"], L, dk, dv),
        check_remote_blocks(sp_result["accounting"]),
        check_numeric(sp_result["output"], ref_output, tolerance,
                      tolerance_source=src, evidence_level=level, logits=logits),
        check_boundary_coverage(coverage, L, world_size, tolerance,
                                sp_result["output"]),
        check_reference_identity(ref_identity, level),
        check_backend(sp_result["accounting"], level),
    ]
    # Anything that is not a PASS blocks. Folding only NA would let a row that
    # reports BLOCKED directly (2A-04b, 2A-04c) fall through to PASS, which is
    # the exact failure mode this vocabulary exists to prevent.
    if any(c["status"] == FAIL for c in checks):
        overall = FAIL
    elif all(c["status"] == PASS for c in checks):
        overall = PASS
    else:
        overall = BLOCKED
    result = {"overall": overall, "checks": checks, "evidence_level": level,
              "tolerance_source": src}
    if envelope is not None:
        result = attach_envelope(result, envelope)
    return result


def attach_envelope(result, envelope, strict=True):
    """Validate a 2A envelope and attach it to the acceptance result.

    Three refusals, each of which has happened to somebody:
      * a record for another stage -- 2A must not accept a 2D record as its own
      * a record claiming PASS -- impossible on CPU, so it means hand editing
      * a record whose verdict differs from the checks it is supposed to describe
    """
    E.validate(envelope, strict=strict)
    if envelope.get("stage") != E.STAGE_2A:
        raise ValueError("envelope stage %r cannot be attached to a 2A result"
                         % (envelope.get("stage"),))
    overall = envelope["verdict"]["overall"]
    if overall == E.PASS:
        raise ValueError("a 2A envelope claims PASS; the CPU ceiling is %s"
                         % E.PASS_CPU_SEMANTICS)
    if overall != result["overall"]:
        raise ValueError("envelope overall %r disagrees with the 2A checks %r"
                         % (overall, result["overall"]))
    out = dict(result)
    out["record"] = envelope
    out["record_summary"] = E.summary(envelope)
    return out


def _infer_dims(sp_result):
    """Recover (dk, dv) from accounting without trusting caller-supplied values."""
    a0 = sp_result["accounting"][0]
    total = a0["resident_kv_numel"] // a0["own_len"]
    dv = len(sp_result["output"][0])
    return total - dv, dv
