"""2C / 2D: AM-HighestAttnKeys construction chain (B' = shared queries, C = per-destination).

Fidelity target
---------------
Zweiger, Fu, Guo, Kim. *Fast KV Compaction via Attention Matching.*
ICML 2026, PMLR 306 (arXiv:2602.16284v2).

Every step below cites the section it reproduces:

  key selection   S3.3  a_i = softmax(q_i K^T); s_j = RMS over queries of a_i,j
                        (NOT RMS of the raw logits -- that was the old skeleton)
  beta            S3.2  w_j = exp(beta_j);  min_{w>=0} ||A w - m||^2
                  C.2   the HighestAttnKeys branch replaces that with a
                        *bounded* least-squares over e^-3 <= w_j <= e^3, solved
                        by projected gradient with iters = 2. (The OMP branch
                        uses iters = 0 and a different -7/+7 rule; this module
                        is on the HighestAttnKeys branch, so -3/+3 is correct.)
  value           S3.2  C_v = argmin ||X C_v - Y||_F  (ordinary least squares)
                  C.2   l2 regularization "degraded performance across all
                        positive values of lambda" -> no ridge on this path.
  logical length  S2    the compact cache keeps a logical length T so appended
                        tokens receive the same position IDs / RoPE phases.

Deliberately NOT collapsed: `beta_uniform` (one scalar shift for the whole block)
is kept only as the *ablation arm*. It is not AM and must never be labelled AM.
See tests/test_am_properties.py for the mathematical difference: a uniform shift
cancels inside a single block's softmax, so it changes the block's mass but not
the intra-block distribution, whereas per-key beta changes both.

Attribution constraint (2C): B' and C run this identical chain and differ in
exactly one thing -- whether the reference query set is shared across
destinations or taken per destination.

Precondition enforced, not papered over: ordinary least squares needs
n_ref >= t and full column rank. A rank-deficient or underdetermined design
matrix raises `RankDeficientError` with its statistics attached. There is no
silent ridge fallback -- C.2 says positive lambda degrades the estimate, and
quietly adding it would also hide exactly the failure this module exists to
surface.
"""

import hashlib
import math
import random as _random

from .state import NEG_INF

# --- frozen AM constants (S3.2 / C.2) --------------------------------------
BETA_LO = -3.0          # HighestAttnKeys box
BETA_HI = 3.0
PGD_ITERS = 2           # C.2: "For Highest Attention Keys, we use iters = 2"
BIND_TOL = 1e-6         # tolerance, in beta units, for "sits on the bound"
DEAD_REL = 1e-6         # a key carrying < this share of block mass counts as dead
N_REF_MULT = 2          # U3 ruling: n_ref = 2t at the main budget

# --- frozen statistic conventions (2026-11-25) -----------------------------
# Every one of these is echoed into the per-edge record so that a cross-budget
# comparison can *prove* both sides were produced with the same convention.
# None of them is a per-run parameter: choosing a tolerance after seeing the
# numbers is exactly the failure mode the frozen doc bans.
MASS_RESIDUAL_AGG = "pooled_over_all_calibration_queries"
DEAD_SCOPE = ("per retained key j: c_j = w_j * sum_i A_ij; c_j < DEAD_REL * sum_j c_j. "
              "The denominator is the sum over retained keys only, and A is the "
              "beta design matrix stacked over all n_ref calibration queries.")

# legacy ridge value, kept only for the ridge ablation arm
LAM_DEFAULT = 1e-3

_EPS = 2.220446049250313e-16


class RankDeficientError(ValueError):
    """The design matrix does not have full column rank, or n_ref < t.

    Carries `.stats` so a caller can record the diagnosis instead of guessing.
    """

    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = stats or {}


# ---------------------------------------------------------------------------
# basic linear algebra (pure stdlib)
# ---------------------------------------------------------------------------

def _matmul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for k in range(m):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(p):
                oi[j] += aik * bk[j]
    return out


def _transpose(a):
    return [list(col) for col in zip(*a)]


def _solve(a, b):
    """Gauss-Jordan with partial pivoting. Used by the ridge ablation only."""
    n = len(a)
    m = len(b[0])
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-14:
            raise ValueError(
                "singular system in ridge value regression at column %d. "
                "This is the ablation path, not the AM chain: AM does not "
                "regularize. Use value_mode='ols' with n_ref >= t." % col)
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for j in range(col, n + m):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f == 0.0:
                continue
            for j in range(col, n + m):
                aug[r][j] -= f * aug[col][j]
    return [row[n:] for row in aug]


def _gram(a):
    n, p = len(a), len(a[0])
    g = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            v = 0.0
            for k in range(n):
                v += a[k][i] * a[k][j]
            g[i][j] = v
            g[j][i] = v
    return g


def _qr_rdiag(x):
    """|R_ii| of the Householder QR of X (no right-hand side).

    |R_ii| is the norm of column i's tail before its reflection, which is what
    the rank test needs. This is O(n p^2); a Jacobi eigen-decomposition of the
    Gram would give true singular values but costs O(p^3) per sweep and squares
    the condition number anyway, so it is not used in the hot path.
    """
    n, p = len(x), len(x[0])
    a = [list(row) for row in x]
    rdiag = []
    colmax = 0.0
    for j in range(p):
        nrm0 = math.sqrt(sum(a[i][j] * a[i][j] for i in range(j, n)))
        if nrm0 > colmax:
            colmax = nrm0
    for j in range(p):
        tail = [a[i][j] for i in range(j, n)]
        nrm = math.sqrt(sum(v * v for v in tail))
        rdiag.append(nrm)
        if nrm == 0.0 or nrm <= 1e-12 * colmax:
            continue
        alpha = -nrm if tail[0] >= 0.0 else nrm
        v = tail[:]
        v[0] -= alpha
        vv = sum(t * t for t in v)
        if vv <= 0.0:
            continue
        for c in range(j, p):
            dot = 0.0
            for i in range(j, n):
                dot += v[i - j] * a[i][c]
            f = 2.0 * dot / vv
            if f == 0.0:
                continue
            for i in range(j, n):
                a[i][c] -= f * v[i - j]
    return rdiag


def _stats_from_rdiag(rdiag, n, p):
    smax = max(rdiag) if rdiag else 0.0
    smin = min(rdiag) if rdiag else 0.0
    tol = max(n, p) * _EPS * smax
    rank = sum(1 for d in rdiag if d > tol)
    return {
        "n": n, "p": p, "rank": rank,
        "sigma_min": smin, "sigma_max": smax,
        "cond": (smax / smin) if smin > 0.0 else float("inf"),
        "tol": tol,
        "r_min": smin, "r_max": smax,
        "rank_deficient": rank < p,
    }


def singular_stats(x):
    """Rank / sigma_min / sigma_max / condition number of a design matrix.

    sigma_min and sigma_max are the |R_ii| extremes of the QR, i.e. estimates
    that bound the true singular values from below; `cond` is the corresponding
    ratio. They are reported for diagnosis, not as exact spectral quantities.
    Never raises.
    """
    n, p = len(x), len(x[0])
    if n == 0 or p == 0:
        return {"n": n, "p": p, "rank": 0, "sigma_min": 0.0, "sigma_max": 0.0,
                "cond": float("inf"), "tol": 0.0, "r_min": 0.0, "r_max": 0.0,
                "rank_deficient": True}
    return _stats_from_rdiag(_qr_rdiag(x), n, p)


def qr_lstsq(x, y):
    """Minimum-residual solution of min ||X C - Y|| via Householder QR.

    Requires n >= p and full column rank; raises RankDeficientError otherwise.
    No regularization anywhere (AM C.2 uses torch.linalg.lstsq).
    """
    n, p = len(x), len(x[0])
    if n < p:
        raise RankDeficientError(
            "n_ref (%d) < t (%d): OLS is underdetermined. AM fits t parameters "
            "from n reference queries; use n_ref >= 2t (U3 ruling)." % (n, p),
            {"n": n, "p": p, "rank": None, "reason": "n_lt_p"})

    m = len(y[0])
    a = [list(x[i]) + list(y[i]) for i in range(n)]
    rdiag = []

    # scale of the problem, used to decide when a column is numerically zero
    colmax = 0.0
    for j in range(p):
        nrm0 = math.sqrt(sum(a[i][j] * a[i][j] for i in range(j, n)))
        if nrm0 > colmax:
            colmax = nrm0

    for j in range(p):
        tail = [a[i][j] for i in range(j, n)]
        nrm = math.sqrt(sum(v * v for v in tail))
        # |R_jj| equals the norm of the column tail *before* the reflection.
        # Reading the post-reflection diagonal instead lets round-off mask a
        # genuinely dependent column, which would defeat the rank check.
        rdiag.append(nrm)
        if nrm == 0.0 or nrm <= 1e-12 * colmax:
            continue
        alpha = -nrm if tail[0] >= 0.0 else nrm
        v = tail[:]
        v[0] -= alpha
        vv = sum(t * t for t in v)
        if vv <= 0.0:
            continue
        # H = I - 2 v v^T / v^T v applied to every remaining column.
        # v indexes rows j..n-1; the column index is separate. Conflating the
        # two silently produces a different (wrong) transform.
        for c in range(j, p + m):
            dot = 0.0
            for i in range(j, n):
                dot += v[i - j] * a[i][c]
            f = 2.0 * dot / vv
            if f == 0.0:
                continue
            for i in range(j, n):
                a[i][c] -= f * v[i - j]

    smax = max(rdiag) if rdiag else 0.0
    rtol = max(n, p) * _EPS * smax
    rmin = min(rdiag) if rdiag else 0.0
    if smax == 0.0 or rmin <= rtol:
        stats = singular_stats(x)
        stats.update({"r_min": rmin, "r_max": smax, "reason": "rank_deficient_in_qr"})
        raise RankDeficientError(
            "design matrix is rank deficient (min |R_ii| = %.3e, tol = %.3e): the "
            "OLS solution is not unique and must not be reported as one."
            % (rmin, rtol), stats)

    z = [[a[i][p + k] for k in range(m)] for i in range(p)]
    c = [[0.0] * m for _ in range(p)]
    for i in range(p - 1, -1, -1):
        s = list(z[i])
        for j in range(i + 1, p):
            rij = a[i][j]
            if rij != 0.0:
                cj = c[j]
                for k in range(m):
                    s[k] -= rij * cj[k]
        rii = a[i][i]
        for k in range(m):
            c[i][k] = s[k] / rii
    return c, {"r_min": rmin, "r_max": smax}


def _spectral_norm2(a, iters=40):
    """||A||_2^2 by power iteration (AM C.2 estimates the step size the same way).

    At convergence ||A^T A v|| = lambda_max(A^T A) = ||A||_2^2 for a unit v, so
    the last norm is already the quantity wanted -- do not square it again.
    """
    n, p = len(a), len(a[0])
    v = [1.0 / math.sqrt(p)] * p
    lam = 0.0
    for _ in range(iters):
        # one application of A^T A: A v (length n), then A^T (A v) (length p).
        # Writing this as at[j] . v instead is dimensionally wrong whenever n != p
        # and silently yields a value too small, which makes the step overshoot.
        u = [sum(a[i][j] * v[j] for j in range(p)) for i in range(n)]
        w = [sum(a[i][j] * u[i] for i in range(n)) for j in range(p)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm == 0.0:
            return 0.0
        v = [x / nrm for x in w]
        lam = nrm
    return lam


# ---------------------------------------------------------------------------
# attention primitives
# ---------------------------------------------------------------------------

def _logsumexp(vals):
    m = max(vals)
    if m == NEG_INF:
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def scaled_logits(q_rows, k):
    """q K^T / sqrt(d), the argument AM writes as the logit."""
    d = len(q_rows[0])
    s = 1.0 / math.sqrt(d)
    return [[sum(qv[t] * kj[t] for t in range(d)) * s for kj in k] for qv in q_rows]


def attention_output(q, k, v, beta=None):
    """Plain attention with an optional additive logit bias per key."""
    dk = len(q[0])
    scale = 1.0 / math.sqrt(dk)
    out = []
    for qv in q:
        scores = [sum(qv[d] * k[j][d] for d in range(dk)) * scale for j in range(len(k))]
        if beta is not None:
            scores = [s + b for s, b in zip(scores, beta)]
        lse = _logsumexp(scores)
        w = [math.exp(s - lse) for s in scores]
        row = [0.0] * len(v[0])
        for j, wj in enumerate(w):
            if wj == 0.0:
                continue
            vj = v[j]
            for d in range(len(row)):
                row[d] += wj * vj[d]
        out.append(row)
    return out


def true_mass(q_rows, k):
    """Exact sum of exp of raw logits per query (no shift needed for comparison)."""
    dk = len(q_rows[0])
    scale = 1.0 / math.sqrt(dk)
    masses = []
    for qv in q_rows:
        scores = [sum(qv[d] * k[j][d] for d in range(dk)) * scale for j in range(len(k))]
        m = max(scores)
        masses.append(math.exp(m) * sum(math.exp(s - m) for s in scores))
    return masses


# ---------------------------------------------------------------------------
# step 1: key selection
# ---------------------------------------------------------------------------

def select_keys(q_fit, k, budget, selector="softmax_rms"):
    """Pick the retained key subset.

    selector="softmax_rms"  AM S3.3: a_i = softmax(q_i K^T), s_j = RMS_i(a_i,j).
                            Frozen default for both B' and C.
    selector="logit_rms"    the previous skeleton's rule (RMS of raw logits).
                            Kept only to show the two rules disagree; not AM.

    Returns a sorted list of indices.
    """
    if selector == "softmax_rms":
        lg = scaled_logits(q_fit, k)
        T = len(k)
        acc = [0.0] * T
        for row in lg:
            m = max(row)
            ex = [math.exp(x - m) for x in row]
            z = sum(ex)
            for j in range(T):
                a = ex[j] / z
                acc[j] += a * a
        score = [math.sqrt(v / len(q_fit)) for v in acc]
    elif selector == "logit_rms":
        dk = len(q_fit[0])
        scale = 1.0 / math.sqrt(dk)
        acc = [0.0] * len(k)
        for qv in q_fit:
            for j in range(len(k)):
                s = sum(qv[d] * k[j][d] for d in range(dk)) * scale
                acc[j] += s * s
        score = [v / len(q_fit) for v in acc]
    else:
        raise ValueError("unknown selector: %r" % (selector,))

    order = sorted(range(len(k)), key=lambda j: (-score[j], j))
    return sorted(order[:budget])


def selection_overlap(idx_a, idx_b):
    """Jaccard overlap of two index sets (used to show the two selectors differ)."""
    sa, sb = set(idx_a), set(idx_b)
    return len(sa & sb) / len(sa | sb)


def query_hash(rows):
    """Stable sha256 of a query set.

    Lives here rather than in record.py so the builders can stamp the hash of the
    rows they actually drew, instead of a caller re-deriving the sample with the
    same seed and hoping the two agree. Floats use .17g so the hash round-trips
    through JSON text exactly.
    """
    h = hashlib.sha256()
    if not rows:
        return h.hexdigest()
    h.update(("%d|%d|" % (len(rows), len(rows[0]))).encode())
    for r in rows:
        h.update(",".join(format(x, ".17g") for x in r).encode())
        h.update(b";")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# step 2: beta
# ---------------------------------------------------------------------------

def _mass_system(q_fit, k_sel, k_full):
    """Row-shifted (A, m) for the beta fit.

    Argument order deliberately mirrors `beta_box_ls(q_fit, k_sel, k_full)` so a
    transposed call is visible at the call site rather than inside.

    A_ij = exp(q_i (C_k)_j^T / sqrt(d)),  m_i = sum_{all keys} exp(q_i K^T / sqrt(d)).
    Row i of both A and m is scaled by exp(-max logit); a common positive factor
    per row leaves that row's residual scaled but does not move the minimiser,
    and it keeps every exponent in [0, 1].
    """
    lg_sel = scaled_logits(q_fit, k_sel)
    lg_all = scaled_logits(q_fit, k_full)
    shifts = [max(row) for row in lg_all]
    A = [[math.exp(lg_sel[i][j] - shifts[i]) for j in range(len(k_sel))]
         for i in range(len(q_fit))]
    m = [sum(math.exp(x - shifts[i]) for x in lg_all[i]) for i in range(len(q_fit))]
    return A, m


def beta_uniform(q_fit, k_sel, k_full):
    """ABLATION ARM ONLY -- one scalar shift so the block mass matches.

    A uniform shift cancels inside a single block's softmax; it only moves the
    block's unnormalised mass. This is what the pre-upgrade skeleton did and it
    is NOT AM's beta. Kept so the "no per-key beta" arm can be run.
    """
    dk = len(q_fit[0])
    scale = 1.0 / math.sqrt(dk)
    total = 0.0
    for qv in q_fit:
        tot_m = true_mass([qv], k_full)[0]
        sel = [sum(qv[d] * k_sel[j][d] for d in range(dk)) * scale for j in range(len(k_sel))]
        lse = _logsumexp(sel)
        total += math.log(tot_m) - lse
    return max(0.0, total / len(q_fit))


def count_on_bound(beta, lo=BETA_LO, hi=BETA_HI, tol=BIND_TOL):
    """(n_on_lower, n_on_upper) with the frozen tolerance.

    Extracted so the boundary rule can be tested directly instead of inferred
    from whatever beta a random fixture happens to produce. `tol` is deliberately
    a keyword with a frozen default and is never plumbed from a run config: a
    tolerance chosen after seeing the numbers is not a tolerance.
    """
    return (sum(1 for b in beta if b <= lo + tol),
            sum(1 for b in beta if b >= hi - tol))


def key_contributions(A, w):
    """c_j = w_j * sum_i A_ij, plus the denominator sum_j c_j.

    A is the beta design matrix stacked over the calibration queries, so c_j is
    the mass key j accounts for across the whole calibration set, not per query.
    """
    contrib = [w[j] * sum(A[i][j] for i in range(len(A))) for j in range(len(w))]
    return contrib, sum(contrib)


def count_dead(contrib, dead_rel=DEAD_REL):
    """Keys whose aggregate contribution is below dead_rel of the block's total."""
    tot = sum(contrib)
    if tot <= 0.0:
        return 0
    return sum(1 for c in contrib if c < dead_rel * tot)


def beta_box_ls(q_fit, k_sel, k_full, lo=BETA_LO, hi=BETA_HI, iters=PGD_ITERS):
    """AM C.2 HighestAttnKeys: bounded least squares over w = exp(beta).

        w* = argmin_{e^lo <= w_j <= e^hi} ||A w - m||^2

    solved as the paper describes: unconstrained least squares, clamp into the
    box, then `iters` projected-gradient steps with fixed step 1/||A||^2.
    It is NOT "solve NNLS then clip afterwards" -- that is a third algorithm and
    must not carry AM's name.

    Returns (beta_list, stats). stats records bound saturation, the dead-key
    fraction, the mass residual and the design-matrix diagnostics that U2 asks
    to be reported per budget point.
    """
    A, m = _mass_system(q_fit, k_sel, k_full)
    w_lo, w_hi = math.exp(lo), math.exp(hi)

    w0, qr_info = qr_lstsq(A, [[v] for v in m])
    w = [min(max(row[0], w_lo), w_hi) for row in w0]

    at = _transpose(A)
    lsq = _spectral_norm2(A)
    eta = (1.0 / lsq) if lsq > 0.0 else 0.0
    for _ in range(iters):
        fit = [sum(A[i][j] * w[j] for j in range(len(w))) for i in range(len(A))]
        resid = [f - t for f, t in zip(fit, m)]
        grad = [sum(at[j][i] * resid[i] for i in range(len(resid))) for j in range(len(w))]
        w = [min(max(wj - eta * g, w_lo), w_hi) for wj, g in zip(w, grad)]

    beta = [math.log(v) for v in w]
    fit = [sum(A[i][j] * w[j] for j in range(len(w))) for i in range(len(A))]
    resid = [f - t for f, t in zip(fit, m)]
    num = math.sqrt(sum(v * v for v in resid))
    den = math.sqrt(sum(v * v for v in m)) + 1e-300

    # effective mass each retained key carries, for the dead-key statistic.
    # NOTE: with the box in force no key can reach w = 0, so "dead" is defined
    # as "carries < DEAD_REL of the block's reconstructed mass", which is a
    # different quantity from lower-bound saturation (that is bind_lo).
    contrib, tot_contrib = key_contributions(A, w)
    dead = count_dead(contrib, DEAD_REL)
    on_lo, on_hi = count_on_bound(beta, lo=lo, hi=hi)

    dstats = singular_stats(A)
    dstats.update(qr_info)
    dstats["problem"] = "beta_box_ls"
    stats = {
        "mode": "box",
        "n_ref": len(q_fit),
        "t": len(k_sel),
        "iters": iters,
        "lo": lo, "hi": hi,
        "beta_min": min(beta),
        "beta_max": max(beta),
        "beta_mean": sum(beta) / len(beta),
        "bind_hi_count": on_hi,
        "bind_hi_frac": on_hi / len(beta),
        "bind_lo_count": on_lo,
        "bind_lo_frac": on_lo / len(beta),
        "dead_count": dead,
        "dead_frac": dead / len(beta),
        "mass_residual": num / den,
        # raw and denominator are stored alongside the ratio so a reader can tell
        # a well-conditioned large-mass fit from a small-mass one; the ratio alone
        # cannot distinguish them.
        "mass_residual_raw": num,
        "mass_residual_denom": den,
        "mass_residual_agg": MASS_RESIDUAL_AGG,
        "bind_tol": BIND_TOL,
        "dead_rel": DEAD_REL,
        "dead_denominator": tot_contrib,
        "dead_scope": DEAD_SCOPE,
        "problem": "beta_box_ls",
        # beta_design is the canonical name; beta_fit is the pre-2026-11-25 key
        # and points at the same dict. They are NOT the value design matrix --
        # that one is a different least-squares problem with its own rank/cond.
        "beta_design": dstats,
        "beta_fit": dstats,
        "solver_status": "ok" if not dstats["rank_deficient"] else "underdetermined_beta",
    }
    return beta, stats


def mass_residual(q_fit, k, k_sel, beta):
    """||A w - m||_2 / ||m||_2 for an arbitrary beta (used by the ablation tests)."""
    A, m = _mass_system(q_fit, k_sel, k)
    w = [math.exp(b) for b in beta]
    fit = [sum(A[i][j] * w[j] for j in range(len(w))) for i in range(len(A))]
    resid = [f - t for f, t in zip(fit, m)]
    return math.sqrt(sum(v * v for v in resid)) / (math.sqrt(sum(v * v for v in m)) + 1e-300)


# ---------------------------------------------------------------------------
# step 3: value
# ---------------------------------------------------------------------------

def _value_design(q_fit, k_sel, beta):
    """X_ij = exp(q_i (C_k)_j^T + beta_j) / sum_j exp(...), per query."""
    dk = len(q_fit[0])
    scale = 1.0 / math.sqrt(dk)
    X = []
    for qv in q_fit:
        row = [sum(qv[d] * k_sel[j][d] for d in range(dk)) * scale + beta[j]
               for j in range(len(k_sel))]
        lse = _logsumexp(row)
        X.append([math.exp(s - lse) for s in row])
    return X


def value_ols(q_fit, k_sel, v_full, y_true, beta):
    """AM S3.2: C_v = argmin ||X C_v - Y||_F. No regularization."""
    X = _value_design(q_fit, k_sel, beta)
    C, qr_info = qr_lstsq(X, y_true)
    stats = singular_stats(X)
    stats.update(qr_info)
    stats["problem"] = "value_ols"
    stats["n_ref"] = len(q_fit)
    stats["t"] = len(k_sel)
    stats["solver_status"] = "ok"
    return C, stats


def value_regression(q_fit, k_sel, v_full, y_true, beta, lam=LAM_DEFAULT):
    """ABLATION ARM ONLY -- ridge. AM C.2 reports lambda > 0 degrades both beta
    and C_v, so this must never be labelled the paper's Value fitting."""
    if isinstance(beta, (int, float)):
        beta = [float(beta)] * len(k_sel)
    X = _value_design(q_fit, k_sel, beta)
    gram = _gram(X)
    for i in range(len(gram)):
        gram[i][i] += lam
    rhs = _matmul(_transpose(X), y_true)
    return _solve(gram, rhs)


# ---------------------------------------------------------------------------
# assembling one compact representation
# ---------------------------------------------------------------------------

def build(q_fit, k, v, budget, *, selector="softmax_rms", beta_mode="box",
          value_mode="ols", lam=LAM_DEFAULT, position_range=None,
          logical_len=None):
    """One compact representation from one reference query set.

    beta_mode   "box"     AM C.2 bounded LS, per-key beta   (default, frozen)
                "uniform" ablation: one scalar shift, invariant intra-block
                "none"    beta = 0, the no-beta ablation
    value_mode  "ols"     AM S3.2                            (default, frozen)
                "ridge"   ablation
    position_range  (pos_start, pos_end) in *global* positions. Defaults to the
                whole source span. It is carried, not used: AM S2 keeps the
                logical length so that appended tokens get the same RoPE phase,
                and 2D-06 checks that the metadata survives serialisation. It
                does NOT verify the phase itself -- that needs the real model.

    Every returned dict carries `diagnostics`, so a run can never silently omit
    the bound-saturation / mass-residual / rank numbers U2 requires.
    """
    if value_mode == "ols" and len(q_fit) < budget:
        raise RankDeficientError(
            "n_ref (%d) < t (%d): OLS underdetermined. n_ref >= 2t is the ruling."
            % (len(q_fit), budget),
            {"n": len(q_fit), "p": budget, "reason": "n_lt_p"})

    idx = select_keys(q_fit, k, budget, selector=selector)
    k_sel = [k[j] for j in idx]
    y_true = attention_output(q_fit, k, v)

    if beta_mode == "box":
        beta, beta_stats = beta_box_ls(q_fit, k_sel, k)
    elif beta_mode == "uniform":
        b = beta_uniform(q_fit, k_sel, k)
        beta = [b] * len(k_sel)
        beta_stats = {"mode": "uniform", "beta": b, "bind_hi_frac": None,
                      "bind_lo_frac": None, "dead_frac": None,
                      "mass_residual": None, "beta_fit": None,
                      "solver_status": "ablation_uniform"}
    elif beta_mode == "none":
        beta = [0.0] * len(k_sel)
        beta_stats = {"mode": "none", "bind_hi_frac": None, "bind_lo_frac": None,
                      "dead_frac": None, "mass_residual": None,
                      "beta_fit": None, "solver_status": "ablation_none"}
    else:
        raise ValueError("unknown beta_mode: %r" % (beta_mode,))

    if value_mode == "ols":
        v_sel, value_stats = value_ols(q_fit, k_sel, v, y_true, beta)
    elif value_mode == "ridge":
        v_sel = value_regression(q_fit, k_sel, v, y_true, beta, lam)
        value_stats = {"solver_status": "ablation_ridge", "lam": lam}
    else:
        raise ValueError("unknown value_mode: %r" % (value_mode,))

    return {
        "indices": idx,
        "k_sel": k_sel,
        "v_sel": v_sel,
        "beta": beta,
        "budget": budget,
        "n_queries": len(q_fit),
        # stamped by the builder that actually drew the rows, so a record never
        # has to re-derive the sample from a seed and hope it matches
        "query_hash": query_hash(q_fit),
        # AM S2: the physical size shrinks but the logical length T is kept, so a
        # shard of a longer sequence must carry the *source's* T, not its own.
        "logical_len": len(k) if logical_len is None else int(logical_len),
        "source_len": len(k),
        "position_range": ([0, len(k)] if position_range is None
                           else [int(position_range[0]), int(position_range[1])]),
        "selector": selector,
        "beta_mode": beta_mode,
        "value_mode": value_mode,
        "diagnostics": {"beta": beta_stats, "value": value_stats},
    }


def error_on(q_eval, k, v, compact):
    """rel-L2 of compact-KV attention vs dense attention, on held-out queries."""
    ref = attention_output(q_eval, k, v)
    got = attention_output(q_eval, compact["k_sel"], compact["v_sel"],
                           beta=compact["beta"])
    num = den = 0.0
    for r, g in zip(ref, got):
        for a, b in zip(g, r):
            num += (a - b) ** 2
            den += b * b
    return math.sqrt(num) / (math.sqrt(den) + 1e-300)


# ---------------------------------------------------------------------------
# query pools: B' (shared, balanced) vs C (per destination)
# ---------------------------------------------------------------------------

def allocate_balanced(n_total, k):
    """Split n_total over k destinations as evenly as possible.

    3274 over 3 -> [1092, 1091, 1091], the allocation the U3 ruling specifies
    for the shared arm.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    base = n_total // k
    rem = n_total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def compute_n_ref(budget, multiplier=N_REF_MULT):
    """Per-edge reference query count for a given budget. U3: n_ref = 2t.

    multiplier=8 gives the 13096 used by the future four-budget sweep, where all
    budgets share one n_ref so that the budget is the only thing that changes.
    """
    return budget * multiplier


def sample_indices(n_pool, n_want, seed=0, exclude=()):
    """Deterministic distinct indices, never returning anything in `exclude`."""
    banned = set(exclude)
    pool = [i for i in range(n_pool) if i not in banned]
    if n_want > len(pool):
        raise ValueError("pool too small after exclusion: want %d, have %d"
                         % (n_want, len(pool)))
    idx = pool[:]
    _random.Random(seed).shuffle(idx)
    return sorted(idx[:n_want])


def sample_pool(pool, n_want, seed=0, exclude=()):
    idx = sample_indices(len(pool), n_want, seed=seed, exclude=exclude)
    return [pool[i] for i in idx], idx


def split_pool(pool, n_calib, n_heldout, seed=0):
    """Split one pool into strictly disjoint calibration / held-out blocks.

    Returns (calib_rows, heldout_rows, calib_idx, heldout_idx). The held-out
    block is drawn first and then excluded from the calibration draw, so the two
    index sets cannot intersect no matter what.
    """
    held_idx = sample_indices(len(pool), n_heldout, seed=seed + 7)
    cali_idx = sample_indices(len(pool), n_calib, seed=seed, exclude=held_idx)
    return ([pool[i] for i in cali_idx], [pool[i] for i in held_idx],
            cali_idx, held_idx)


def build_shared(q_lists, k, v, budget, n_ref=None, seed=0, **kwargs):
    """B' arm: one compact representation from one shared reference query set.

    n_ref=None  use every query in every pool (legacy behaviour)
    n_ref=N     draw N queries split across the destination pools as evenly as
                possible, then build one representation.
    """
    if n_ref is None:
        shared = [qv for qs in q_lists for qv in qs]
        allocation = [len(qs) for qs in q_lists]
    else:
        allocation = allocate_balanced(n_ref, len(q_lists))
        shared = []
        for di, (qs, want) in enumerate(zip(q_lists, allocation)):
            rows, _ = sample_pool(qs, want, seed=seed + 1000 + di)
            shared.extend(rows)
    rep = build(shared, k, v, budget, **kwargs)
    rep["arm"] = "shared"
    rep["n_ref"] = len(shared)
    rep["allocation"] = allocation
    return rep


def build_per_destination(q_lists, k, v, budget, n_ref=None, seed=0, **kwargs):
    """C arm: one compact representation per destination, from its own queries.

    Same budget, same chain, same per-edge n_ref as B'. The only intended
    difference from build_shared is *whose* queries are used.
    """
    out = []
    for di, qs in enumerate(q_lists):
        if n_ref is None:
            use = list(qs)
        else:
            use, _ = sample_pool(qs, n_ref, seed=seed + 2000 + di)
        rep = build(use, k, v, budget, **kwargs)
        rep["arm"] = "per_destination"
        rep["destination"] = di
        rep["n_ref"] = len(use)
        out.append(rep)
    return out


# ---------------------------------------------------------------------------
# merge-level diagnostics (AM Eq.1+2: what happens after concatenation)
# ---------------------------------------------------------------------------

def merged_attention(blocks, q_rows):
    """Attention over the concatenation of [(k, v, beta_or_None), ...].

    A uniform beta cancels inside a single block but not here; this is where a
    block's beta becomes observable.
    """
    d = len(q_rows[0])
    scale = 1.0 / math.sqrt(d)
    K, V, B = [], [], []
    for k, v, beta in blocks:
        K.extend(k)
        V.extend(v)
        B.extend([0.0] * len(k) if beta is None else beta)
    out = []
    for qv in q_rows:
        sc = [sum(qv[t] * K[j][t] for t in range(d)) * scale + B[j] for j in range(len(K))]
        lse = _logsumexp(sc)
        w = [math.exp(x - lse) for x in sc]
        row = [0.0] * len(V[0])
        for j, wj in enumerate(w):
            if wj == 0.0:
                continue
            vj = V[j]
            for t in range(len(row)):
                row[t] += wj * vj[t]
        out.append(row)
    return out


def block_mass(blocks, q_rows):
    """Per-block unnormalised mass for each query, after concatenation."""
    d = len(q_rows[0])
    scale = 1.0 / math.sqrt(d)
    per_block = []
    for k, v, beta in blocks:
        B = [0.0] * len(k) if beta is None else beta
        vals = []
        for qv in q_rows:
            sc = [sum(qv[t] * k[j][t] for t in range(d)) * scale + B[j] for j in range(len(k))]
            m = max(sc)
            vals.append(math.exp(m) * sum(math.exp(x - m) for x in sc))
        per_block.append(vals)
    return per_block


def merge_rel_l2(q_eval, k_src, v_src, k_fix, v_fix, compact):
    """rel-L2 of (compacted prefix + fixed block) against the dense concatenation."""
    ref = merged_attention([(k_src, v_src, None), (k_fix, v_fix, None)], q_eval)
    got = merged_attention([(compact["k_sel"], compact["v_sel"], compact["beta"]),
                            (k_fix, v_fix, None)], q_eval)
    num = den = 0.0
    for r, g in zip(ref, got):
        for a, b in zip(g, r):
            num += (a - b) ** 2
            den += b * b
    return math.sqrt(num) / (math.sqrt(den) + 1e-300)
