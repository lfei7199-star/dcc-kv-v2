"""Pilot record contract -- gates, pairing and the statistics a pilot may compute.

What this module is
-------------------
`design/23` turns the pilot into a *data contract*: the only legal use of a
pilot is estimating variance, so every number a pilot reports has to be
recomputable from the row-level `f1` values. This module is that contract in
code:

    row-level jsonl  ->  gates (G1..G17)  ->  paired differences  ->  derived stats

The derived statistics are exactly the ones `design/22` §3.4.3 allows:

    mean_d          mean of the per-row paired difference
    se_iid          sd(d_i) / sqrt(200)                     [diagnostic]
    se_cluster_w    cluster bootstrap, ROW-WEIGHTED         [the only SE used for CI]
    deff_empirical  (se_cluster_w / se_iid)^2               [diagnostic]
    deff_model      1 + 0.18 * rho_d                        [diagnostic]
    rho_D, rho_d    within-pair correlations                [diagnostic]

What this module refuses to do
------------------------------
* It never chooses, adjusts or infers `delta`. A pilot that supplies its own
  margin is rejected (`REJECT_PILOT_SUPPLIED_DELTA`) -- that is the "move the NI
  margin after seeing the data" failure mode this project already ruled out.
* It never emits `NI_PASS` / `NI_REJECTED` on its own authority. Without an
  externally supplied `delta` every record says `NOT_TESTABLE`; with one, the
  verdict is still marked `verdict_authority=False` and is single-seed only.
* It never lets the convenient equal-variance formula stand in for the measured
  variance. `deff_model` is a diagnostic row; if `deff_empirical` and
  `deff_model` disagree by more than 0.10 the record is flagged `model_misfit`
  and `se_cluster_w` is left untouched. Replacing a measured variance with a
  prettier model variance is the bug, not the fix.

Why the cluster bootstrap is not "182 clusters, equal weight"
-------------------------------------------------------------
The estimand is the 200-row average, `(1/200) sum_i d_i`. 18 clusters hold two
rows, 164 hold one. If each replicate simply averaged the drawn cluster means,
the target would silently become `(1/182) sum_c mean_c` -- the 18 double-row
clusters would be systematically down-weighted, i.e. a *different parameter*.
So each replicate rebuilds the row weights from the drawn clusters:

    T* = sum_{c drawn} m_c * mean_c  /  sum_{c drawn} m_c

the ratio form, which keeps the target on the 200-row mean and is invariant to
shifting every `d_i` by a constant (the naive `/200` version is not, and that
extra term is a defect rather than a refinement).
`weight_by_size=False` exists only so the tests can demonstrate the wrong
estimand; it must never reach a record.

Import discipline: stdlib only (hashlib / math / unicodedata / json), and no
import of `compact`, `record`, `accept` or `envelope`. This is record plumbing,
not acceptance logic, and it must not become a back door into 2A.
"""

import hashlib
import json
import math
import unicodedata

# ---------------------------------------------------------------------------
# frozen identities
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "pilot-1"
KNOWN_SCHEMA_VERSIONS = (SCHEMA_VERSION,)

CLUSTER_SERIALIZATION = "pilot-input/v1"
KNOWN_SERIALIZATIONS = (CLUSTER_SERIALIZATION,)

N_ROWS = 200
N_CLUSTERS = 182
N_PAIRS = 18
CLUSTER_HISTOGRAM = {1: 164, 2: 18}

Z95 = 1.96                 # normal-approximation two-sided 95%, as frozen
DEFF_UNIVERSAL = 2.0       # DEFF <= 2 given cluster size <= 2 and independence
DEFF_MODEL_MAX = 1.18      # equal-variance + common rho_d, rho_d = 1
MODEL_MISFIT_TOL = 0.10
RHO_CI_DISAGREE = 0.15
DEFF_IMPLAUSIBLE = 2.5

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_MIN_VALID = 100

D_PP = 100.0               # f1 in [0,1] is reported in pp only in derived layer

ARMS = ("A_dense", "B", "B_prime", "C")
BUDGETS = (819, 1637, 3274, 6548)

ESTIMAND_CANDIDATE_REFERENCE = {
    "Delta_shared": ("B_prime", "B"),
    "Delta_DCC": ("C", "B_prime"),
}

# Arms whose construction has no seed (R-d: `shared_side_deterministic: true`).
# Declared in metadata; the default matches the frozen R-d structure, and the
# report records which list was actually used so nothing is inferred silently.
DEFAULT_DETERMINISTIC_ARMS = ("A_dense", "B", "B_prime")

# keys that would let a pilot name its own margin
DELTA_KEYS = ("delta", "delta_pp", "delta_shared", "delta_dcc", "delta_Q10",
              "ni_margin", "ni_margin_pp", "margin_pp")

ROW_REQUIRED = (
    "schema_version", "row_id", "cluster_id", "cluster_size", "input_sha256",
    "gold_index", "arm", "budget_t", "seed", "f1", "pred_sha256", "pred_len",
    "truncated", "effective_len",
)

META_REQUIRED = (
    "schema_version", "cluster_serialization", "dataset", "dataset_sha256",
    "n_rows", "n_clusters", "evaluator", "construction", "n_ref",
    "query_sha256", "world_size", "L", "purpose", "verdict_authority",
    "tokenizer", "estimand_pairs",
)

RHO_REQUIRED = ("method", "point", "ci_low", "ci_high", "ci_method",
                "n_pairs", "bootstrap_B")

# gate identifiers (design/23 §5 + G15..G17 added per the 23:00 ruling)
GATE_IDS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10",
            "G11", "G12", "G13", "G14", "G15", "G16", "G17")


# ---------------------------------------------------------------------------
# deterministic PRNG (no dependence on CPython's random module internals)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_GOLD = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def _advance(state):
    """One SplitMix64 step -> (new_state, 64-bit output)."""
    state = (state + _GOLD) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * _M1) & _MASK64
    z = ((z ^ (z >> 27)) * _M2) & _MASK64
    return state, z ^ (z >> 31)


class SplitMix64(object):
    """Tiny deterministic generator: same sequence on every Python version."""

    def __init__(self, seed):
        self.state = seed & _MASK64

    def next_u64(self):
        self.state, out = _advance(self.state)
        return out

    def below(self, n):
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0
        # rejection sampling keeps the draw exactly uniform
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            r = self.next_u64()
            if r < limit:
                return r % n


# ---------------------------------------------------------------------------
# input identity: serialization must be unambiguous before it is hashed
# ---------------------------------------------------------------------------

def _norm_text(s):
    if not isinstance(s, str):
        raise TypeError("input field must be str, got %r" % (type(s).__name__,))
    s = unicodedata.normalize("NFKC", s)
    return s.replace("\r\n", "\n").replace("\r", "\n")


def serialize_input(context, question):
    """Length-prefixed payload; the prefix is what removes boundary ambiguity.

    `("ab", "c")` and `("a", "bc")` concatenate to the same bytes. The 8-byte
    big-endian length prefix makes that impossible; `\\x00` and the version tag
    are readability, not the guarantee.
    """
    ctx = _norm_text(context).encode("utf-8", "strict")
    q = _norm_text(question).encode("utf-8", "strict")
    return (CLUSTER_SERIALIZATION.encode("ascii") + b"\x00"
            + len(ctx).to_bytes(8, "big") + ctx
            + len(q).to_bytes(8, "big") + q)


def input_sha256(context, question):
    return hashlib.sha256(serialize_input(context, question)).hexdigest()


def cluster_id_from_sha(sha):
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("input_sha256 must be 64 hex chars")
    return sha[:16]


def short_id_collisions(shas):
    """Distinct full hashes mapping to one 16-hex id. Must be run, not argued."""
    seen = {}
    collisions = []
    for sha in shas:
        cid = sha[:16]
        prev = seen.get(cid)
        if prev is not None and prev != sha:
            collisions.append((cid, prev, sha))
        seen[cid] = sha
    return collisions


# ---------------------------------------------------------------------------
# small numeric helpers (no statistics module: keep ddof explicit and visible)
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / float(len(xs)) if xs else None


def _sd(xs, ddof=1):
    n = len(xs)
    if n - ddof <= 0:
        return None
    mu = sum(xs) / float(n)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / float(n - ddof))


def _pct(sorted_xs, q):
    """Linear-interpolation percentile on an already sorted list."""
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (len(sorted_xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def pearson(xs, ys):
    """Pearson r, or None when either side has no variance.

    Returning None (never 0.0) matters: with two identical values per pair a
    zero-variance side is *undefined*, not "no correlation", and silently
    reporting 0.0 would make `deff_model` look like a real reading.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / float(n)
    my = sum(ys) / float(n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    r = sxy / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, r))


# ---------------------------------------------------------------------------
# cluster bootstrap that keeps the 200-row estimand
# ---------------------------------------------------------------------------

def _cluster_arrays(d_by_cluster):
    keys = sorted(d_by_cluster)
    sizes, means = [], []
    for k in keys:
        vals = d_by_cluster[k]
        if not vals:
            raise ValueError("empty cluster %r" % (k,))
        sizes.append(len(vals))
        means.append(sum(vals) / float(len(vals)))
    return sizes, means


def _bootstrap_core(sizes, means, iters, seed, weight_by_size=True):
    """Replicate estimates of the row-weighted mean.

    weight_by_size=True builds `sum m_c mean_c / sum m_c` over the drawn
    clusters -- the ratio (Hajek) form. It is chosen for translation
    invariance: shifting every `d_i` by a constant must not move the SE, and
    the ratio form cancels that shift while the naive `sum m_c mean_c / N`
    form does not (its extra term `Var(sum k_c m_c) * mean^2` makes SE depend
    on the level of the estimate -- a defect, not a refinement).

    The price is an O(1/n_clusters) bias in the bootstrap *mean*: with 3
    clusters of sizes 1/1/2 it averages 4.37 against a true 5.00. Hajek ratio
    bias is a methodological approximation; with 182 clusters it is expected to
    be small relative to the variance scale, but no numerical negligibility
    claim is used in the H2 decision. The point estimate itself is exact.

    `weight_by_size=False` averages the drawn cluster means, which targets
    `(1/182) sum_c mean_c` -- a different parameter, kept only so the test suite
    can prove the difference.
    """
    if iters <= 0:
        raise ValueError("iters must be positive")
    n = len(sizes)
    if n == 0:
        raise ValueError("no clusters")

    st = seed & _MASK64
    out = []
    if weight_by_size:
        for _ in range(iters):
            num = 0.0
            den = 0
            for _ in range(n):
                st, z = _advance(st)
                c = z % n
                m = sizes[c]
                num += m * means[c]
                den += m
            out.append(num / den)
    else:
        for _ in range(iters):
            acc = 0.0
            for _ in range(n):
                st, z = _advance(st)
                acc += means[z % n]
            out.append(acc / n)
    return out


def _draw(sizes, means, seed):
    """One cluster resample -> (drawn sizes, drawn means, next state)."""
    n = len(sizes)
    st = seed & _MASK64
    osz, omn = [], []
    for _ in range(n):
        st, z = _advance(st)
        c = z % n
        osz.append(sizes[c])
        omn.append(means[c])
    return osz, omn, st


def cluster_bootstrap(d_by_cluster, iters=BOOTSTRAP_B, seed=BOOTSTRAP_SEED,
                      weight_by_size=True):
    """Bootstrap of the row-weighted mean over clusters.

    Returns replicates, their mean, sd (ddof=1), and percentile bounds. The CI
    here is the percentile interval of the estimator, used only as a robustness
    readout; the reported interval in the record is `mean +- 1.96 * se`.
    """
    sizes, means = _cluster_arrays(d_by_cluster)
    reps = _bootstrap_core(sizes, means, iters, seed, weight_by_size)
    ordered = sorted(reps)
    point = sum(s * v for s, v in zip(sizes, means)) / float(sum(sizes))
    return {
        "replicates": reps,
        "point": point,
        "se": _sd(reps),
        "ci_low": _pct(ordered, 2.5),
        "ci_high": _pct(ordered, 97.5),
        "iters": iters,
        "weight_by_size": bool(weight_by_size),
    }


def bootstrap_se_ci(d_by_cluster, outer=60, inner=100, seed=BOOTSTRAP_SEED):
    """Nested bootstrap: an interval for `se_cluster_w` itself.

    `se` is an estimate too, so it gets an interval. Nested resampling is the
    only honest way to get one; the default sizes are deliberately modest
    because this is a second-order readout, not the decision quantity.
    """
    if outer <= 0 or inner <= 0:
        return None
    sizes, means = _cluster_arrays(d_by_cluster)
    st = seed & _MASK64
    ses = []
    for _ in range(outer):
        osz, omn, st = _draw(sizes, means, st)
        reps = _bootstrap_core(osz, omn, inner, st, True)
        st = (st + 0x1234567) & _MASK64
        s = _sd(reps)
        if s is not None:
            ses.append(s)
    if len(ses) < 5:
        return None
    ordered = sorted(ses)
    return {"ci_low": _pct(ordered, 2.5), "ci_high": _pct(ordered, 97.5),
            "outer": outer, "inner": inner,
            "ci_method": "nested_bootstrap(outer=%d,inner=%d)" % (outer, inner)}


# ---------------------------------------------------------------------------
# rho: two intervals, both restricted to the 18 double-row clusters
# ---------------------------------------------------------------------------

def fisher_ci(r, n_pairs, z=Z95):
    """Fisher-z interval. Only defined for n_pairs > 3."""
    if r is None or n_pairs <= 3:
        return None
    zr = math.atanh(max(-0.999999, min(0.999999, r)))
    se = 1.0 / math.sqrt(n_pairs - 3.0)
    lo = math.tanh(zr - z * se)
    hi = math.tanh(zr + z * se)
    return {"ci_low": lo, "ci_high": hi, "se_z": se,
            "ci_method": "fisher_z", "n_pairs": n_pairs,
            "assumption": "bivariate normal (questionable: F1 is bounded and "
                          "zero-inflated)"}


def rho_bootstrap_ci(pair_values, iters=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Resample the 18 pairs; Pearson each time; percentile interval."""
    if iters <= 0:
        return None
    n = len(pair_values)
    if n < 3:
        return None
    rng = SplitMix64(seed)
    reps = []
    for _ in range(iters):
        xs, ys = [], []
        for _ in range(n):
            a, b = pair_values[rng.below(n)]
            xs.append(a)
            ys.append(b)
        r = pearson(xs, ys)
        if r is not None:
            reps.append(r)
    if len(reps) < BOOTSTRAP_MIN_VALID:
        return {"ci_low": None, "ci_high": None, "valid": len(reps),
                "iters": iters, "ci_method": "cluster_bootstrap_percentile",
                "note": "INSUFFICIENT_BOOTSTRAP"}
    ordered = sorted(reps)
    return {"ci_low": _pct(ordered, 2.5), "ci_high": _pct(ordered, 97.5),
            "valid": len(reps), "iters": iters,
            "ci_method": "cluster_bootstrap_percentile"}


def estimate_rho(pair_values, iters=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """`{method, point, ci_low, ci_high, ci_method, n_pairs, bootstrap_B}` + alt.

    Bootstrap is the primary interval (F1 is bounded and zero-inflated), Fisher-z
    is reported alongside and never averaged with it.
    """
    xs = [a for a, _ in pair_values]
    ys = [b for _, b in pair_values]
    point = pearson(xs, ys)
    boot = rho_bootstrap_ci(pair_values, iters, seed) or {}
    fish = fisher_ci(point, len(pair_values))
    out = {
        "method": "pearson_within_pair" if point is not None
                  else "degenerate_zero_variance",
        "point": point,
        "ci_low": boot.get("ci_low"),
        "ci_high": boot.get("ci_high"),
        "ci_method": boot.get("ci_method", "cluster_bootstrap_percentile"),
        "n_pairs": len(pair_values),
        "bootstrap_B": iters,
        "bootstrap_seed": seed,
        "alt": fish,
    }
    if boot.get("note"):
        out["note"] = boot["note"]
    if fish and out["ci_low"] is not None:
        if (abs(fish["ci_low"] - out["ci_low"]) > RHO_CI_DISAGREE
                or abs(fish["ci_high"] - out["ci_high"]) > RHO_CI_DISAGREE):
            out["interval_disagreement"] = True
    return out


# ---------------------------------------------------------------------------
# sensitivity bands (design/22 §4.5), computed from the data rather than
# hard-coding the legacy 17.1365 / 15.775 / 12.117 triple
# ---------------------------------------------------------------------------

def sigma_gate_pp(mean_pp, delta_pp, deff):
    """Largest `sd(d)` (pp) for which `NI_REJECTED` still holds at this DEFF.

    Solves `|mean| - Z*se < delta` with `se = sd*sqrt(deff)/sqrt(200)`.
    """
    budget = abs(mean_pp) - delta_pp
    if budget <= 0:
        return 0.0
    return budget * math.sqrt(N_ROWS) / (Z95 * math.sqrt(deff))


def sensitivity_state(sd_row_pp, mean_pp, delta_pp):
    """`unconditional` / `model_dependent` / `beyond` (+ a detail band).

    `unconditional`  : holds even at DEFF = 2 -> no variance assumption needed
    `model_dependent`: needs DEFF below the equal-variance model bound (1.18)
    `beyond`         : no DEFF can save it; only INCONCLUSIVE is reachable

    The bands only mean something while the point estimate sits below the
    margin. A mean at or above `-delta` makes `NI_REJECTED` unreachable for a
    structural reason (CI_high >= mean), and saying "unconditional" there would
    describe a rejection that cannot happen -- so that case is `beyond` with an
    explicit reason.
    """
    if delta_pp is None:
        return "unknown_delta", "delta not frozen"
    if sd_row_pp is None:
        return "unknown_delta", "sd unavailable"
    if mean_pp >= -delta_pp:
        return "beyond", ("point estimate %.4f pp is at or above the margin "
                          "-%.4f pp: CI_high >= mean, so NI_REJECTED is "
                          "unreachable at any variance" % (mean_pp, delta_pp))
    budget = abs(mean_pp) - delta_pp
    g_row = sigma_gate_pp(mean_pp, delta_pp, 1.0)
    g_model = sigma_gate_pp(mean_pp, delta_pp, DEFF_MODEL_MAX)
    g_uni = sigma_gate_pp(mean_pp, delta_pp, DEFF_UNIVERSAL)
    detail = "gates pp: universal<%.4f model<%.4f row<%.4f" % (g_uni, g_model, g_row)
    if sd_row_pp >= g_row:
        return "beyond", detail
    if sd_row_pp < g_uni:
        return "unconditional", detail
    if sd_row_pp < g_model:
        # only fails if DEFF > 2 is reachable, i.e. only in the extreme corner
        return "model_dependent", detail + " ; band [universal, model)"
    # needs DEFF below the equal-variance bound to still reject
    return "model_dependent", detail + " ; band [model, row)"


# ---------------------------------------------------------------------------
# pairing: d_i must be reconstructible from two raw f1 values
# ---------------------------------------------------------------------------

def _row_key(arm, budget_t, seed, row_id):
    return (seed, arm, budget_t, row_id)


def seed_axis(rows, spec, det_arms=DEFAULT_DETERMINISTIC_ARMS):
    """The seed values one pair is evaluated at.

    A deterministic candidate (B') has exactly one construction, so the axis is
    `[None]`; a seeded candidate (C) gives one record per seed, which is what
    `sigma_D_by_seed` needs. Taking the union of both arms' seeds would silently
    multiply the deterministic side.
    """
    cand = spec["candidate"]
    if cand["arm"] in det_arms:
        return [None]
    seeds = sorted({r.get("seed") for r in rows
                    if r.get("arm") == cand["arm"]
                    and r.get("budget_t") == cand.get("budget_t")},
                   key=lambda s: (s is None, s))
    return seeds or [None]


def index_rows(rows):
    idx = {}
    dupes = []
    for r in rows:
        k = _row_key(r.get("arm"), r.get("budget_t"), r.get("seed"), r.get("row_id"))
        if k in idx:
            dupes.append(k)
        idx[k] = r
    return idx, dupes


def pair_rows(idx, spec, seed, det_arms=DEFAULT_DETERMINISTIC_ARMS):
    """Build (row_id, f1_cand, f1_ref, cluster_id) for one estimand pair.

    `Delta_DCC` pairs a seeded `C` against an unseeded `B'`; the deterministic
    side is looked up under `seed=None` no matter what seed axis the pair is on.
    Reading the reference under the candidate's seed instead would make every
    pair look missing -- and a "missing pairing" verdict is exactly the kind of
    error that looks like a data problem rather than a lookup bug.

    Returns (pairs, problems). A pair is only legal when both arms carry a raw
    `f1` and the two rows are provably the same input (same input_sha256, same
    gold_index, same cluster_id). Anything else is INVALID_PAIRED_SOURCE --
    a stored `d_i` alone is not evidence.
    """
    cand, ref = spec["candidate"], spec["reference"]
    cand_seed = None if cand["arm"] in det_arms else seed
    ref_seed = None if ref["arm"] in det_arms else seed
    pairs, problems = [], []
    for row_id in range(N_ROWS):
        rc = idx.get(_row_key(cand["arm"], cand.get("budget_t"), cand_seed, row_id))
        rr = idx.get(_row_key(ref["arm"], ref.get("budget_t"), ref_seed, row_id))
        if rc is None or rr is None:
            problems.append({"row_id": row_id,
                             "why": "missing arm row (candidate=%s reference=%s)"
                                    % (rc is not None, rr is not None)})
            continue
        fc, fr = rc.get("f1"), rr.get("f1")
        if fc is None or fr is None:
            problems.append({"row_id": row_id,
                             "why": "raw f1 missing on at least one arm; a stored "
                                    "d_i without both f1 values is not auditable"})
            continue
        if rc.get("input_sha256") != rr.get("input_sha256"):
            problems.append({"row_id": row_id,
                             "why": "input_sha256 differs between arms"})
            continue
        if rc.get("gold_index") != rr.get("gold_index"):
            problems.append({"row_id": row_id,
                             "why": "gold_index differs between arms"})
            continue
        if rc.get("cluster_id") != rr.get("cluster_id"):
            problems.append({"row_id": row_id,
                             "why": "cluster_id differs between arms"})
            continue
        d = fc - fr
        stored = rc.get("d") if rc.get("d") is not None else rr.get("d")
        if stored is not None:
            try:
                if abs(float(stored) - d) > 1e-12:
                    problems.append({"row_id": row_id,
                                     "why": "stored d=%.17g != f1_cand-f1_ref=%.17g"
                                            % (float(stored), d)})
                    continue
            except (TypeError, ValueError):
                problems.append({"row_id": row_id, "why": "stored d is not numeric"})
                continue
        pairs.append({"row_id": row_id, "d": d, "f1_cand": fc, "f1_ref": fr,
                      "cluster_id": rc.get("cluster_id")})
    return pairs, problems


def d_by_cluster(pairs):
    out = {}
    for p in pairs:
        out.setdefault(p["cluster_id"], []).append(p["d"])
    return out


def rho_pairs(pairs, block):
    """Within-cluster value pairs for the 18 double-row clusters.

    block="d"     -> (d_c1, d_c2)   for rho_d
    block="cand"  -> (f1_cand, f1_cand) for rho_D on the candidate arm
    """
    grouped = {}
    for p in pairs:
        grouped.setdefault(p["cluster_id"], []).append(p)
    out = []
    for _, group in sorted(grouped.items()):
        if len(group) != 2:
            continue
        a, b = group
        if block == "d":
            out.append((a["d"], b["d"]))
        elif block == "cand":
            out.append((a["f1_cand"], b["f1_cand"]))
        elif block == "ref":
            out.append((a["f1_ref"], b["f1_ref"]))
        else:
            raise ValueError("unknown block %r" % (block,))
    return out


# ---------------------------------------------------------------------------
# verdict (fold discipline: recompute, never trust the record)
# ---------------------------------------------------------------------------

def fold_verdict(mean_pp, ci_low, ci_high, delta_pp):
    """Three-state + NOT_TESTABLE. Requires the whole interval, not a point."""
    if delta_pp is None or ci_low is None or ci_high is None:
        return "NOT_TESTABLE"
    if ci_low > -delta_pp:
        return "NI_PASS"
    if ci_high < -delta_pp:
        return "NI_REJECTED"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def _gate(gid, code, status, detail, blocking=None):
    if blocking is None:
        blocking = status == "FAIL"
    return {"id": gid, "code": code, "status": status, "detail": detail,
            "blocking": bool(blocking)}


def _walk_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield prefix + str(k), v
            for item in _walk_keys(v, prefix + str(k) + "."):
                yield item
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for item in _walk_keys(v, "%s[%d]." % (prefix, i)):
                yield item


def check_meta(meta):
    """Structural checks that need only the metadata block."""
    gates = []
    if not isinstance(meta, dict):
        return [_gate("G14", "REJECT_UNKNOWN_SCHEMA", "FAIL",
                      "metadata is not an object")]

    missing = [k for k in META_REQUIRED if k not in meta]
    if missing:
        gates.append(_gate("G14", "INVALID_META_SCHEMA", "FAIL",
                           "missing metadata keys: %s" % ", ".join(missing)))
    else:
        gates.append(_gate("G14", "OK_META_KEYS", "PASS", "metadata keys present"))

    sv = meta.get("schema_version")
    if sv is not None:
        if sv in KNOWN_SCHEMA_VERSIONS:
            gates.append(_gate("G14", "OK_SCHEMA", "PASS", "schema_version=%s" % sv))
        else:
            gates.append(_gate("G14", "REJECT_UNKNOWN_SCHEMA", "FAIL",
                               "unknown schema_version %r (known: %s)"
                               % (sv, ", ".join(KNOWN_SCHEMA_VERSIONS))))

    ser = meta.get("cluster_serialization")
    if ser is not None:
        if ser in KNOWN_SERIALIZATIONS:
            gates.append(_gate("G10", "OK_SERIALIZATION", "PASS", ser))
        else:
            gates.append(_gate("G10", "REJECT_UNKNOWN_SERIALIZATION", "FAIL",
                               "unknown cluster_serialization %r" % (ser,)))

    src = None
    construction = meta.get("construction")
    if isinstance(construction, dict):
        src = construction.get("randomness_source")
    if src == "calibration_query_subsample":
        gates.append(_gate("G8", "OK_PROVENANCE", "PASS", src))
    elif src is None:
        gates.append(_gate("G8", "REJECT_LEGACY_PROVENANCE", "FAIL",
                           "construction.randomness_source missing"))
    else:
        gates.append(_gate("G8", "REJECT_LEGACY_PROVENANCE", "FAIL",
                           "randomness_source=%r; only "
                           "'calibration_query_subsample' is legal (a Rademacher "
                           "projection belongs to the legacy implementation)" % (src,)))

    if meta.get("n_rows") not in (None, N_ROWS):
        gates.append(_gate("G2", "BLOCKED_CLUSTER_MISMATCH", "FAIL",
                           "meta.n_rows=%r, expected %d" % (meta.get("n_rows"), N_ROWS)))
    if meta.get("n_clusters") not in (None, N_CLUSTERS):
        gates.append(_gate("G2", "BLOCKED_CLUSTER_MISMATCH", "FAIL",
                           "meta.n_clusters=%r, expected %d"
                           % (meta.get("n_clusters"), N_CLUSTERS)))

    # authority: a pilot may not claim it can decide anything
    authority = meta.get("verdict_authority")
    purpose = meta.get("purpose")
    if authority is not False:
        gates.append(_gate("G16", "REJECT_AUTHORITY_CLAIM", "FAIL",
                           "verdict_authority=%r; a pilot record must carry "
                           "false" % (authority,)))
    elif purpose != "variance_estimation":
        gates.append(_gate("G16", "REJECT_AUTHORITY_CLAIM", "FAIL",
                           "purpose=%r; expected 'variance_estimation'" % (purpose,)))
    else:
        gates.append(_gate("G16", "OK_AUTHORITY", "PASS",
                           "verdict_authority=false, purpose=variance_estimation"))

    # a pilot must not name its own margin
    hits = [k for k, _ in _walk_keys({k: v for k, v in meta.items()
                                      if k not in ("estimand_pairs",)})
            if k.split(".")[-1] in DELTA_KEYS]
    if hits:
        gates.append(_gate("G17", "REJECT_PILOT_SUPPLIED_DELTA", "FAIL",
                           "pilot record carries margin field(s): %s"
                           % ", ".join(hits)))
    else:
        gates.append(_gate("G17", "OK_NO_DELTA", "PASS",
                           "no margin field in the pilot record"))

    # scale declaration
    ev = meta.get("evaluator")
    raw_scale = ev.get("raw_scale") if isinstance(ev, dict) else None
    if raw_scale is None:
        gates.append(_gate("G7", "FAIL_SCALE", "FAIL", "evaluator.raw_scale missing"))
    elif raw_scale != "proportion":
        gates.append(_gate("G7", "FAIL_SCALE", "FAIL",
                           "evaluator.raw_scale=%r; values must be stored in "
                           "[0,1] (divide by 100, do not relabel)" % (raw_scale,)))
    else:
        gates.append(_gate("G7", "OK_SCALE_DECL", "PASS", raw_scale))

    # estimand pairs must be declared, and must match the legal direction
    pairs = meta.get("estimand_pairs")
    if not isinstance(pairs, list) or not pairs:
        gates.append(_gate("G15", "INVALID_PAIRED_SOURCE", "FAIL",
                           "meta.estimand_pairs missing; pairing is declared, "
                           "never inferred"))
    else:
        bad = []
        for spec in pairs:
            name = spec.get("estimand")
            expect = ESTIMAND_CANDIDATE_REFERENCE.get(name)
            if expect is None:
                bad.append("unknown estimand %r" % (name,))
                continue
            cand = spec.get("candidate") or {}
            ref = spec.get("reference") or {}
            if (cand.get("arm"), ref.get("arm")) != expect:
                bad.append("%s direction %s->%s, expected %s->%s"
                           % (name, cand.get("arm"), ref.get("arm"), expect[0], expect[1]))
            if cand.get("arm") not in ARMS or ref.get("arm") not in ARMS:
                bad.append("%s uses an unknown arm" % (name,))
            for side in (cand, ref):
                if side.get("budget_t") is not None and side["budget_t"] not in BUDGETS:
                    bad.append("%s budget_t=%r not in %s"
                               % (name, side.get("budget_t"), BUDGETS))
        if bad:
            gates.append(_gate("G15", "INVALID_PAIRED_SOURCE", "FAIL", "; ".join(bad)))
        else:
            gates.append(_gate("G15", "OK_PAIRS_DECLARED", "PASS",
                               "%d estimand pair(s)" % len(pairs)))
    return gates


def check_rows(rows):
    """G1..G4, G7, G9, G11, G13, G15 on the row-level records."""
    gates = []
    if not rows:
        return [_gate("G6", "INVALID_SUMMARY_ONLY", "FAIL",
                      "no row-level records at all -- aggregates alone cannot "
                      "produce d_i, rho_d or any DEFF")]

    # G1: required fields, arm/budget legality, f1 range
    missing_fields = []
    bad_arm, bad_budget, bad_range, bad_rowid = [], [], [], []
    for r in rows:
        miss = [k for k in ROW_REQUIRED if k not in r]
        if miss:
            missing_fields.append((r.get("row_id"), miss))
            continue
        if r["arm"] not in ARMS:
            bad_arm.append(r["arm"])
        if r["budget_t"] is not None and r["budget_t"] not in BUDGETS:
            bad_budget.append(r["budget_t"])
        try:
            f1 = float(r["f1"])
        except (TypeError, ValueError):
            bad_range.append(r.get("row_id"))
            continue
        if not (0.0 <= f1 <= 1.0) or f1 != f1 or f1 in (float("inf"), float("-inf")):
            bad_range.append(r.get("row_id"))
        if not isinstance(r["row_id"], int) or not (0 <= r["row_id"] < N_ROWS):
            bad_rowid.append(r["row_id"])

    if missing_fields:
        gates.append(_gate("G1", "INVALID_ROW_SCHEMA", "FAIL",
                           "missing fields on %d row(s), first=%r"
                           % (len(missing_fields), missing_fields[0])))
    elif bad_arm:
        gates.append(_gate("G1", "BLOCKED_MISSING_ROWS", "FAIL",
                           "unknown arm(s): %s" % sorted(set(map(str, bad_arm)))))
    elif bad_budget:
        gates.append(_gate("G1", "BLOCKED_MISSING_ROWS", "FAIL",
                           "unknown budget_t: %s" % sorted(set(map(str, bad_budget)))))
    elif bad_rowid:
        gates.append(_gate("G1", "BLOCKED_MISSING_ROWS", "FAIL",
                           "row_id outside 0..%d: %r" % (N_ROWS - 1, bad_rowid[:5])))
    else:
        gates.append(_gate("G1", "OK_FIELDS", "PASS",
                           "%d row(s), fields complete" % len(rows)))

    # G7: no clipping, no mixed scale
    if bad_range:
        gates.append(_gate("G7", "FAIL_SCALE", "FAIL",
                           "f1 outside [0,1] on %d row(s), first=%r; clipping is "
                           "forbidden -- fix the evaluator" % (len(bad_range), bad_range[0])))
    elif not missing_fields:
        gates.append(_gate("G7", "OK_F1_RANGE", "PASS",
                           "f1 in [0,1] on all %d row(s)" % len(rows)))

    # G11: unique key + full 200 rows per cell
    cells = {}
    dup_keys = []
    seen = set()
    for r in rows:
        if "arm" not in r or "row_id" not in r:
            continue
        k = _row_key(r.get("arm"), r.get("budget_t"), r.get("seed"), r.get("row_id"))
        if k in seen:
            dup_keys.append(k)
        seen.add(k)
        cells.setdefault((r.get("seed"), r.get("arm"), r.get("budget_t")), set()).add(
            r.get("row_id"))
    short = [(k, len(v)) for k, v in sorted(cells.items(), key=lambda kv: str(kv[0]))
             if len(v) != N_ROWS]
    if dup_keys:
        gates.append(_gate("G11", "FAIL_ROW_INDEX", "FAIL",
                           "duplicate unique key(s): %r" % (dup_keys[:3],)))
    elif short:
        gates.append(_gate("G11", "FAIL_ROW_INDEX", "FAIL",
                           "cell(s) without 200 distinct row_id: %r" % (short[:3],)))
    else:
        gates.append(_gate("G11", "OK_ROW_INDEX", "PASS",
                           "%d cell(s) x %d row_id" % (len(cells), N_ROWS)))

    # G2/G3/G4 are per *cell* (seed x arm x budget), not pooled over the file.
    # Every arm repeats the same 200 inputs, so a pooled histogram would see
    # each singleton cluster three times and look wrong while being right.
    cell_rows = {}
    for r in rows:
        cell_rows.setdefault((r.get("seed"), r.get("arm"), r.get("budget_t")),
                             []).append(r)

    bad_hist = []
    size_maps = {}
    for cell, group in sorted(cell_rows.items(), key=lambda kv: str(kv[0])):
        sizes = {}
        for r in group:
            cid = r.get("cluster_id")
            if cid is not None:
                sizes[cid] = sizes.get(cid, 0) + 1
        size_maps[cell] = sizes
        hist = {}
        for v in sizes.values():
            hist[v] = hist.get(v, 0) + 1
        if len(sizes) != N_CLUSTERS or hist != CLUSTER_HISTOGRAM:
            bad_hist.append((cell, len(sizes), hist))
    if not cell_rows:
        gates.append(_gate("G2", "BLOCKED_CLUSTER_MISMATCH", "FAIL", "no cells"))
    elif bad_hist:
        gates.append(_gate("G2", "BLOCKED_CLUSTER_MISMATCH", "FAIL",
                           "cell(s) with wrong cluster structure, first=%r "
                           "(expected %d clusters, histogram %s)"
                           % (bad_hist[0], N_CLUSTERS, CLUSTER_HISTOGRAM)))
    else:
        gates.append(_gate("G2", "OK_CLUSTERS", "PASS",
                           "%d cell(s), each %d clusters with histogram %s"
                           % (len(cell_rows), N_CLUSTERS, CLUSTER_HISTOGRAM)))

    # G3: declared cluster_size honest, pair members share the input, golds differ
    declared_bad, input_bad, gold_bad = [], [], []
    for cell, group in sorted(cell_rows.items(), key=lambda kv: str(kv[0])):
        sizes = size_maps[cell]
        by_cluster = {}
        for r in group:
            cid = r.get("cluster_id")
            if cid is None:
                continue
            by_cluster.setdefault(cid, []).append(r)
            if r.get("cluster_size") != sizes.get(cid):
                declared_bad.append((cell, cid, r.get("cluster_size"), sizes.get(cid)))
        for cid, members in by_cluster.items():
            if len(members) == 2:
                if members[0].get("input_sha256") != members[1].get("input_sha256"):
                    input_bad.append((cell, cid))
                if sorted(m.get("gold_index") for m in members) != [0, 1]:
                    gold_bad.append((cell, cid))
    if declared_bad:
        gates.append(_gate("G3", "BLOCKED_INPUT_MISMATCH", "FAIL",
                           "declared cluster_size disagrees with observed, first=%r"
                           % (declared_bad[0],)))
    elif input_bad:
        gates.append(_gate("G3", "BLOCKED_INPUT_MISMATCH", "FAIL",
                           "%d double-row cluster(s) do not share input_sha256: %r"
                           % (len(input_bad), input_bad[:3])))
    elif gold_bad:
        gates.append(_gate("G3", "BLOCKED_INPUT_MISMATCH", "FAIL",
                           "%d double-row cluster(s) lack gold_index {0,1}: %r"
                           % (len(gold_bad), gold_bad[:3])))
    else:
        gates.append(_gate("G3", "OK_PAIR_INPUT", "PASS",
                           "every double-row cluster shares one input and holds "
                           "gold_index {0,1}"))

    # G4: greedy prediction should be identical inside a cluster (warning only)
    pred_bad = []
    for cell, group in sorted(cell_rows.items(), key=lambda kv: str(kv[0])):
        by_cluster = {}
        for r in group:
            by_cluster.setdefault(r.get("cluster_id"), []).append(r)
        for cid, members in by_cluster.items():
            if len({m.get("pred_sha256") for m in members}) > 1:
                pred_bad.append((cell, cid))
    if pred_bad:
        gates.append(_gate("G4", "WARN_PRED_DIVERGENCE", "WARN",
                           "%d (cell,cluster) pair(s) have differing pred_sha256; "
                           "greedy should not diverge -- if it really differs the "
                           "rho premise needs revisiting" % len(pred_bad),
                           blocking=False))
    else:
        gates.append(_gate("G4", "OK_PRED_STABLE", "PASS",
                           "pred_sha256 identical within cluster x arm"))

    # G9: cluster_id must be input_sha256[:16], and 16-hex ids must not collide
    inconsistent = []
    shas = set()
    for r in rows:
        sha = r.get("input_sha256")
        cid = r.get("cluster_id")
        if isinstance(sha, str) and len(sha) == 64:
            shas.add(sha)
            if cid != sha[:16]:
                inconsistent.append((r.get("row_id"), cid, sha[:16]))
        else:
            inconsistent.append((r.get("row_id"), cid, "bad input_sha256"))
    collisions = short_id_collisions(sorted(shas))
    if inconsistent:
        gates.append(_gate("G9", "FAIL_CLUSTER_HASH", "FAIL",
                           "cluster_id != input_sha256[:16] on %d row(s), first=%r"
                           % (len(inconsistent), inconsistent[0])))
    elif collisions:
        gates.append(_gate("G9", "FAIL_CLUSTER_HASH", "FAIL",
                           "%d short-id collision(s): %r" % (len(collisions), collisions[:2])))
    else:
        gates.append(_gate("G9", "OK_CLUSTER_HASH", "PASS",
                           "%d distinct full hashes -> %d distinct short ids"
                           % (len(shas), len({s[:16] for s in shas}))))

    # G13: no bare sigma_D anywhere
    bare = [k for k, _ in _walk_keys(rows) if k.split(".")[-1] == "sigma_D"]
    if bare:
        gates.append(_gate("G13", "FAIL_UNCONDITIONAL_SIGMA", "FAIL",
                           "bare 'sigma_D' appears %d time(s); use sigma_D_by_seed / "
                           "sigma_D_median / sigma_s" % len(bare)))
    else:
        gates.append(_gate("G13", "OK_SIGMA_NAMING", "PASS", "no bare sigma_D"))

    # G15: a stored d_i without both raw f1 values is not auditable
    d_only = [r.get("row_id") for r in rows
              if r.get("d") is not None and r.get("f1") is None]
    if d_only:
        gates.append(_gate("G15", "INVALID_PAIRED_SOURCE", "FAIL",
                           "%d row(s) carry d without raw f1 (first=%r); the "
                           "audit chain is broken" % (len(d_only), d_only[0])))
    return gates


# ---------------------------------------------------------------------------
# derived statistics
# ---------------------------------------------------------------------------

def derive_record(estimand, spec, seed, pairs, delta_pp=None,
                  bootstrap_B=BOOTSTRAP_B, boot_seed=BOOTSTRAP_SEED,
                  se_ci=(0, 0), rho_iters=None):
    """One derived record: 200-row estimand, row-weighted cluster bootstrap."""
    n = len(pairs)
    ds = [p["d"] for p in pairs]
    mean_d = _mean(ds)
    mean_pp = mean_d * D_PP
    sd_row = _sd(ds, ddof=1)
    sd_pp = sd_row * D_PP if sd_row is not None else None
    se_iid = sd_row / math.sqrt(n) if sd_row is not None else None

    boot = cluster_bootstrap(d_by_cluster(pairs), bootstrap_B, boot_seed)
    se_cluster = boot["se"]

    se_pp = se_cluster * D_PP if se_cluster is not None else None
    ci_low = mean_pp - Z95 * se_pp if se_pp is not None else None
    ci_high = mean_pp + Z95 * se_pp if se_pp is not None else None

    deff_emp = None
    if se_cluster is not None and se_iid not in (None, 0.0):
        deff_emp = (se_cluster / se_iid) ** 2

    iters = rho_iters if rho_iters is not None else bootstrap_B
    rho_d = estimate_rho(rho_pairs(pairs, "d"), iters, boot_seed)
    rho_D = estimate_rho(rho_pairs(pairs, "cand"), iters, boot_seed)
    deff_model = None
    if rho_d["point"] is not None:
        deff_model = 1.0 + 0.18 * rho_d["point"]

    flags = []
    model_misfit = None
    if deff_emp is not None and deff_model is not None:
        model_misfit = abs(deff_emp - deff_model) > MODEL_MISFIT_TOL
        if model_misfit:
            flags.append("model_misfit")
    elif deff_emp is not None:
        flags.append("model_misfit_unavailable")
    if deff_emp is not None and deff_emp < 1.0:
        # Legitimate when rho_d < 0, but then there is no positive design
        # effect to report -- "DEFF < 1" must never be read as a bonus.
        flags.append("deff_empirical_below_one")
    if deff_emp is not None and deff_emp > DEFF_IMPLAUSIBLE:
        flags.append("deff_empirical_implausible")
    if deff_emp is not None and deff_emp > DEFF_UNIVERSAL:
        flags.append("deff_empirical_above_universal_bound")
    if rho_d.get("interval_disagreement"):
        flags.append("rho_interval_wide")
    rd_ci = (rho_d["ci_low"], rho_d["ci_high"])
    if rd_ci[0] is not None and rd_ci[1] is not None and rd_ci[1] - rd_ci[0] > 0.6:
        if "rho_interval_wide" not in flags:
            flags.append("rho_interval_wide")
    if rho_d["point"] is None:
        flags.append("rho_d_undefined")

    se_ci_out = None
    if se_ci and se_ci[0] > 0:
        se_ci_out = bootstrap_se_ci(d_by_cluster(pairs), se_ci[0], se_ci[1], boot_seed)
        if se_ci_out:
            se_ci_out["ci_low"] = se_ci_out["ci_low"] * D_PP
            se_ci_out["ci_high"] = se_ci_out["ci_high"] * D_PP
    if se_ci_out is None:
        # The SE is an estimate too, so a record without its interval is
        # incomplete rather than merely shorter. Say so instead of omitting it.
        flags.append("se_ci_missing")

    state, detail = sensitivity_state(sd_pp, mean_pp, delta_pp)
    verdict = fold_verdict(mean_pp, ci_low, ci_high, delta_pp)

    return {
        "estimand": estimand,
        "pair": spec,
        "seed": seed,
        "n_rows": n,
        "n_clusters": len({p["cluster_id"] for p in pairs}),
        "n_pairs": len(rho_pairs(pairs, "d")),
        "row_ids": [p["row_id"] for p in pairs],
        "d_i": ds,
        "f1_cand": [p["f1_cand"] for p in pairs],
        "f1_ref": [p["f1_ref"] for p in pairs],
        "mean_d": mean_d,
        "mean_d_pp": mean_pp,
        "sd_d_row": sd_row,
        "sd_d_row_pp": sd_pp,
        "se_iid": se_iid,
        "se_iid_pp": se_iid * D_PP if se_iid is not None else None,
        "se_cluster_w": se_cluster,
        "se_cluster_w_pp": se_pp,
        "se_cluster_w_ci": se_ci_out,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": "normal_approx(mean +- 1.96*se_cluster_w)",
        "deff_empirical": deff_emp,
        "deff_model": deff_model,
        "deff_bound_universal": DEFF_UNIVERSAL,
        "deff_bound_model": DEFF_MODEL_MAX,
        "model_misfit": model_misfit,
        "rho_D": rho_D,
        "rho_d": rho_d,
        "sensitivity_state": state,
        "sensitivity_detail": detail,
        "bootstrap_B": bootstrap_B,
        "bootstrap_seed": boot_seed,
        "delta_pp": delta_pp,
        "verdict": verdict,
        "verdict_authority": False,
        "flags": flags,
    }


def aggregate_seeds(records, delta_pp=None):
    """Seed-level summary. `sigma_s` needs K >= 2; nothing is invented from K=1."""
    means_pp = [r["mean_d_pp"] for r in records]
    out = {
        "n_seeds": len(records),
        "mean_d_pp": _mean(means_pp),
        "sigma_s_pp": _sd(means_pp) if len(records) > 1 else None,
        "sigma_s_note": ("seed-level sd of the per-seed mean; requires K>=2 "
                         "and the new pipeline's seed stream" if len(records) > 1
                         else "K=1: sigma_s undefined, do not report a value"),
        "sigma_D_by_seed_pp": {str(r["seed"]): r["sd_d_row_pp"] for r in records},
        "rho_d_by_seed": {str(r["seed"]): r["rho_d"] for r in records},
        "per_seed_verdict": {str(r["seed"]): r["verdict"] for r in records},
    }
    sds = [r["sd_d_row_pp"] for r in records if r["sd_d_row_pp"] is not None]
    out["sigma_D_median_pp"] = _pct(sorted(sds), 50) if sds else None
    out["sigma_D_range_pp"] = [min(sds), max(sds)] if sds else None
    primary = records[0]
    out["primary_seed"] = primary["seed"]
    out["primary_ci"] = {"ci_low": primary["ci_low"], "ci_high": primary["ci_high"],
                         "se_cluster_w_pp": primary["se_cluster_w_pp"],
                         "note": "single-seed CI; seed variance not included"}
    out["verdict"] = primary["verdict"]
    out["verdict_authority"] = False
    return out


def check_derived_consistency(derived):
    """G5 / G12 / G13 on the derived layer."""
    gates = []
    ests = derived.get("estimands", {})

    incomplete = []
    for name, block in ests.items():
        for rec in block.get("per_seed", []):
            for field in ("rho_D", "rho_d"):
                rho = rec.get(field) or {}
                miss = [k for k in RHO_REQUIRED if k not in rho]
                if miss:
                    incomplete.append((name, rec.get("seed"), field, miss))
    if incomplete:
        gates.append(_gate("G5", "INCOMPLETE_RHO", "FAIL",
                           "%d rho block(s) missing required keys, first=%r"
                           % (len(incomplete), incomplete[0])))
    else:
        gates.append(_gate("G5", "OK_RHO_BLOCKS", "PASS",
                           "rho_D / rho_d carry method+point+interval+B+seed"))

    # G12: one d_i per estimand, never a shared column
    conflation = []
    names = sorted(ests)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = ests[names[i]].get("per_seed", [])
            b = ests[names[j]].get("per_seed", [])
            for ra in a:
                for rb in b:
                    if ra.get("seed") == rb.get("seed") and ra.get("d_i") == rb.get("d_i"):
                        conflation.append((names[i], names[j], ra.get("seed")))
    bare = [k for k, _ in _walk_keys(derived.get("estimands", {}))
            if k.split(".")[-1] == "sigma_D"]
    if conflation:
        gates.append(_gate("G12", "FAIL_ESTIMAND_CONFLATION", "FAIL",
                           "two estimands share an identical d_i column: %r" % (conflation[:2],)))
    elif bare:
        gates.append(_gate("G12", "FAIL_ESTIMAND_CONFLATION", "FAIL",
                           "bare sigma_D in derived layer"))
    else:
        gates.append(_gate("G12", "OK_ESTIMAND_SEPARATION", "PASS",
                           "each estimand keeps its own d_i / sd / rho_d / se"))
    return gates


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def validate_pilot(rows, meta, delta_pp=None, bootstrap_B=BOOTSTRAP_B,
                   boot_seed=BOOTSTRAP_SEED, se_ci=(0, 0)):
    """Run every gate, then (only if nothing blocks) the statistics.

    Returns a report dict. `blocking` mirrors "any FAIL"; when it is True the
    `derived` block is None on purpose -- a record whose gates fail must not
    emit numbers that someone could quote.
    """
    gates = []
    gates.extend(check_meta(meta))
    gates.extend(check_rows(rows))
    blocking = any(g["status"] == "FAIL" for g in gates)

    report = {
        "schema_version": SCHEMA_VERSION,
        "validator": "validate_pilot",
        "gates": gates,
        "blocking": blocking,
        "derived": None,
        "flags": [],
        "delta_pp": delta_pp,
        "verdict_authority": False,
        "may_not_be_cited_as": ["NI_PASS", "NI_REJECTED", "H2 判决", "质量主张"],
    }

    if blocking or not rows:
        report["flags"] = sorted({g["code"] for g in gates if g["status"] == "FAIL"})
        return report

    idx, _dupes = index_rows(rows)
    det_arms = tuple(meta.get("deterministic_arms", DEFAULT_DETERMINISTIC_ARMS))
    report["deterministic_arms"] = list(det_arms)
    ests = {}
    pair_problems = []
    for spec in meta["estimand_pairs"]:
        name = spec["estimand"]
        seeds = seed_axis(rows, spec, det_arms)
        per_seed = []
        for seed in seeds:
            pairs, problems = pair_rows(idx, spec, seed, det_arms)
            if problems:
                pair_problems.extend([{"estimand": name, "seed": seed, **p}
                                      for p in problems[:5]])
                continue
            per_seed.append(derive_record(name, spec, seed, pairs, delta_pp,
                                          bootstrap_B, boot_seed, se_ci))
        if per_seed:
            ests[name] = {"per_seed": per_seed,
                          "aggregate": aggregate_seeds(per_seed, delta_pp)}

    if pair_problems:
        gates.append(_gate("G15", "INVALID_PAIRED_SOURCE", "FAIL",
                           "%d pairing problem(s): %r"
                           % (len(pair_problems), pair_problems[:2])))
        report["gates"] = gates
        report["blocking"] = True
        report["flags"] = sorted({g["code"] for g in gates if g["status"] == "FAIL"})
        return report

    derived = {"estimands": ests, "delta_pp": delta_pp,
               "verdict_authority": False}
    gates.extend(check_derived_consistency(derived))

    flags = set()
    for block in ests.values():
        for rec in block["per_seed"]:
            flags.update(rec["flags"])
    report["gates"] = gates
    report["blocking"] = any(g["status"] == "FAIL" for g in gates)
    report["derived"] = None if report["blocking"] else derived
    report["flags"] = sorted(flags)
    return report


def dump_report(report):
    return json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
