"""Q8 -- the 2A numeric tolerance, frozen 2026-11-25 BEFORE any GPU run.

The ruling
----------
    metric                relative_l2
    threshold             1e-05
    direction             <=
    source                preregistered
    post-hoc adjustment   forbidden

and the gate is a conjunction, not the metric alone:

    rel_l2(O_sp, O_ref) = ||O_sp - O_ref||_2 / (||O_ref||_2 + eps) <= 1e-5
    AND every frozen sample position produced finite output
    AND the reference identity / SHA256 matches

`max_abs` and `cosine` are recorded and never gate. Top-1 / top-k logit
agreement is an auxiliary diagnostic. NaN or Inf anywhere is FAIL no matter how
small rel_l2 is. A missing or unreproducible reference is an invalid comparison
(BLOCKED/INVALID), never "a very large error".

Why not the 1e-9 the CPU tests use
----------------------------------
Floating-point addition is not associative. Different rank blocks mean a
different reduction order, and online-softmax/LSE merging is a different
summation from the dense path; BF16 with FP32 upcast differs again. Gating a
real 4-GPU run at 1e-9 would fail on ordinary merge rounding and read as an
implementation bug. 1e-9 stays `CPU_TEST_TOL` for the float64 reference and is
**not** a legal tolerance source for a measured run -- `validate_policy` refuses
to let it be labelled `preregistered`.

Why not wider than 1e-5
-----------------------
1e-5 is a frozen engineering threshold, not something derived from model length
or GPU model. It is loose enough to absorb merge rounding and tight enough that
a real attention/SP defect -- wrong shard, dropped block, broken LSE merge,
wrong mask -- cannot pass: those produce relative errors around 1e-2, three
orders of magnitude away.

Import discipline
-----------------
This module imports nothing but `math`. `envelope` and `accept` may depend on
it; it depends on neither. The Q8 policy therefore cannot be edited by way of a
solver, and it cannot be re-derived by one either.
"""

import math

Q8_ID = "Q8"

# ---- the frozen policy ----------------------------------------------------
METRIC = "relative_l2"
THRESHOLD = 1e-5
DIRECTION = "<="
POST_HOC = "forbidden"
FROZEN_ON = "2026-11-25"
FROZEN_BY = "西瓜先生 2026-11-25 裁定"

# ---- tolerance sources ----------------------------------------------------
# A measured (hardware) run may cite only PREREG. The other two exist so the
# CPU reference can keep testing at 1e-9 without anyone mistaking it for a gate.
SOURCE_PREREG = "preregistered"
SOURCE_TEST = "test_value"          # CPU reference test constant
SOURCE_CALLER = "caller_supplied"   # ad-hoc, structural evidence only
SOURCES = (SOURCE_PREREG, SOURCE_TEST, SOURCE_CALLER)

CPU_TEST_TOL = 1e-9                 # NOT a GPU tolerance source
EPS = 1e-300                        # denominator guard, identical to accept.rel_l2

# Duplicated from envelope.py on purpose: `envelope` imports this module, so the
# reverse import would be a cycle. tests/test_q8.py asserts the two agree.
MEASURED_EVIDENCE = "measured_hardware"

# ---- the frozen sample positions ------------------------------------------
# 2A runs the full 130944 tokens, so the diagnostic positions are the shard
# boundaries (both sides) and the last token. Frozen in the execution doc as
# (32735, 32736, 32737, 65471, 65472, 65473, 98207, 98208, 98209, 130943).
L_PRIMARY = 130944
WORLD_SIZE_PRIMARY = 4
BOUNDARY_RADIUS = 1
BOUNDARY_POSITIONS = (32735, 32736, 32737, 65471, 65472, 65473,
                      98207, 98208, 98209, 130943)


def boundary_positions(L=L_PRIMARY, n_ranks=WORLD_SIZE_PRIMARY,
                       radius=BOUNDARY_RADIUS):
    """Positions each 2A run must sample: every interior boundary +/- radius,
    plus the last token.

    At (130944, 4) this reproduces the frozen tuple exactly; the function exists
    so the same rule can be exercised on short CPU tensors instead of being a
    literal nobody can test.
    """
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive int, got %r" % (L,))
    if not isinstance(n_ranks, int) or n_ranks <= 0:
        raise ValueError("n_ranks must be a positive int, got %r" % (n_ranks,))
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative int, got %r" % (radius,))

    bounds = [L * i // n_ranks for i in range(n_ranks + 1)]
    out = set()
    for i in range(1, n_ranks):
        for d in range(-radius, radius + 1):
            p = bounds[i] + d
            if 0 <= p < L:
                out.add(p)
    out.add(L - 1)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# the policy block carried by every record that ran 2A-04
# ---------------------------------------------------------------------------

def policy(source=SOURCE_PREREG, threshold=THRESHOLD):
    """The Q8 block as it is written into a record.

    `frozen_on` is set only for a pre-registered tolerance. A test value with a
    freeze date would be a claim that a number nobody froze was frozen.
    """
    return {
        "q8": Q8_ID,
        "metric": METRIC,
        "threshold": float(threshold),
        "direction": DIRECTION,
        "source": source,
        "post_hoc_adjustment": POST_HOC,
        "frozen_on": FROZEN_ON if source == SOURCE_PREREG else None,
    }


def validate_policy(block, evidence_level=None, where="tolerance_policy"):
    """Return the problems with a Q8 block. Empty list means it is usable.

    Three things this refuses, each of which is a real way to launder a
    threshold:
      * `source='preregistered'` on any number other than 1e-5
      * a measured run citing anything other than `preregistered`
      * a freeze date attached to a value that was never frozen
    """
    if not isinstance(block, dict):
        return ["%s is absent; 2A-04 cannot be reported without its policy" % where]

    problems = []
    if block.get("q8") != Q8_ID:
        problems.append("%s.q8 = %r, expected %r"
                        % (where, block.get("q8"), Q8_ID))
    if block.get("metric") != METRIC:
        problems.append("%s.metric = %r, the frozen metric is %r"
                        % (where, block.get("metric"), METRIC))
    if block.get("direction") != DIRECTION:
        problems.append("%s.direction = %r, the frozen direction is %r"
                        % (where, block.get("direction"), DIRECTION))
    if block.get("post_hoc_adjustment") != POST_HOC:
        problems.append("%s.post_hoc_adjustment = %r, the ruling froze %r"
                        % (where, block.get("post_hoc_adjustment"), POST_HOC))

    src = block.get("source")
    if src not in SOURCES:
        problems.append("%s.source = %r is not one of %r"
                        % (where, src, SOURCES))

    thr = block.get("threshold")
    if isinstance(thr, bool) or not isinstance(thr, (int, float)):
        problems.append("%s.threshold = %r is not a number" % (where, thr))
    elif not math.isfinite(float(thr)) or float(thr) < 0.0:
        problems.append("%s.threshold = %r is not a finite non-negative number"
                        % (where, thr))
    elif src == SOURCE_PREREG and float(thr) != THRESHOLD:
        problems.append(
            "%s.threshold = %r is labelled 'preregistered'; only %r was frozen. "
            "%r is the CPU reference test constant and may not be promoted to a "
            "GPU gate" % (where, thr, THRESHOLD, CPU_TEST_TOL))

    frozen = block.get("frozen_on")
    if src == SOURCE_PREREG and frozen != FROZEN_ON:
        problems.append("%s.frozen_on = %r, the freeze date is %r"
                        % (where, frozen, FROZEN_ON))
    if src in (SOURCE_TEST, SOURCE_CALLER) and frozen is not None:
        problems.append("%s carries frozen_on = %r but its source is %r; a value "
                        "nobody froze must not carry a freeze date"
                        % (where, frozen, src))

    if evidence_level == MEASURED_EVIDENCE and src != SOURCE_PREREG:
        problems.append(
            "%s.source = %r on a measured_hardware run; only 'preregistered' may "
            "gate hardware evidence (ad-hoc tolerances would be a post-hoc "
            "adjustment)" % (where, src))
    return problems


# ---------------------------------------------------------------------------
# auxiliary diagnostics -- recorded, never gating
# ---------------------------------------------------------------------------

def _argmax(row):
    best_i, best = 0, row[0]
    for i in range(1, len(row)):
        if row[i] > best:
            best_i, best = i, row[i]
    return best_i


def _topk(row, k):
    idx = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
    return idx[:k]


def topk_agreement(sp_rows, ref_rows, k=5):
    """Top-1 and top-k overlap between two logit blocks.

    Auxiliary only. The ruling put top-1 / top-k agreement outside the 2A gate:
    it is a diagnostic for the *decoded* output, while 2A-04 compares the
    attention output. Reporting the two side by side is useful; gating on either
    alone is not.
    """
    if len(sp_rows) != len(ref_rows):
        raise ValueError("cannot compare %d rows against %d"
                         % (len(sp_rows), len(ref_rows)))
    if not sp_rows:
        raise ValueError("no rows to compare")
    if k <= 0:
        raise ValueError("k must be positive")

    top1 = 0
    overlap = 0.0
    for a, b in zip(sp_rows, ref_rows):
        top1 += 1 if _argmax(a) == _argmax(b) else 0
        ka = set(_topk(a, k))
        kb = set(_topk(b, k))
        overlap += len(ka & kb) / float(k)
    n = len(sp_rows)
    return {"k": k, "n": n, "top1": top1 / float(n),
            "topk_overlap": overlap / float(n)}
