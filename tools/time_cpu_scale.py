"""Measure how the pure-Python construction path scales with the budget t.

Why this exists
---------------
The record template is *configurable* for t = 1637, but the CPU reference cannot
*execute* it: the Householder QR is O(n_ref * t^2) with n_ref = 2t, so cost grows
like t^3 in pure Python. Without a number, "1637 is too slow on CPU" is an
opinion. With a measured curve and a fitted exponent, it is a fact that can be
checked.

This is a timing measurement, not a quality result. It reports wall-clock on
whatever machine ran it and says so in the output.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import compact as C  # noqa: E402

D = 8
SOURCE_LEN = 512
BUDGETS = (16, 32, 64, 96, 128, 192, 256)
TARGET = 1637


def world(seed, t=SOURCE_LEN, d=D):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    v = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(t)]
    return k, v


def queries(n, seed, d=D):
    rnd = random.Random(seed)
    return [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)]


def main():
    rows = []
    for t in BUDGETS:
        k, v = world(11)
        q = queries(2 * t, 23)
        t0 = time.perf_counter()
        rep = C.build(q, k, v, t)
        dt = time.perf_counter() - t0
        rows.append({"t": t, "n_ref": 2 * t, "seconds": dt,
                     "beta_cond": rep["diagnostics"]["beta"]["beta_fit"]["cond"],
                     "value_cond": rep["diagnostics"]["value"]["cond"]})
        print("t=%4d  n_ref=%5d  %8.2fs" % (t, 2 * t, dt))

    # local exponent between consecutive points. A single fit over the last two
    # points can understate the growth if the asymptotic term only dominates
    # later, so the trend is reported instead of one number that hides it.
    import math
    local = []
    for i in range(1, len(rows)):
        t1, s1 = rows[i - 1]["t"], rows[i - 1]["seconds"]
        t2, s2 = rows[i]["t"], rows[i]["seconds"]
        local.append({"from_t": t1, "to_t": t2,
                      "exponent": math.log(s2 / s1) / math.log(t2 / t1)})

    # extrapolate with the steepest observed exponent: the conservative choice,
    # since under-estimating the cost is what would make the CPU path look viable
    p = max(x["exponent"] for x in local)
    p_last = local[-1]["exponent"]
    est = rows[-1]["seconds"] * (TARGET / rows[-1]["t"]) ** p
    est_last = rows[-1]["seconds"] * (TARGET / rows[-1]["t"]) ** p_last

    out = {
        "meta": {
            "status": "timing_only",
            "note": ("wall clock of the pure-Python CPU reference on this machine; "
                     "NOT a GPU number and NOT a quality result"),
            "backend": "cpu_ref_f64",
            "d_model": D,
            "source_len": SOURCE_LEN,
            "budgets": list(BUDGETS),
            "n_ref_rule": "2t",
        },
        "rows": rows,
        "fit": {
            "local_exponents": local,
            "exponent_used": p,
            "exponent_last_pair": p_last,
            "target_t": TARGET,
            "extrapolated_seconds_per_edge": est,
            "extrapolated_seconds_per_3_edges": est * 3,
            "extrapolated_seconds_per_edge_last_pair": est_last,
            "caveat": ("extrapolation from t<=%d. The Householder QR is O(n_ref t^2) "
                       "with n_ref=2t, so the asymptotic cost is cubic; the exponent is "
                       "fitted from measurement because lower-order terms dominate at "
                       "small t. Treat the result as an order of magnitude, and treat the "
                       "larger of the two estimates as the planning figure."
                       % max(BUDGETS)),
        },
    }

    dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "cpu", "cpu_scale_timing.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nlocal exponents (cost ~ t^p between consecutive points):")
    for x in local:
        print("  t %4d -> %4d : p = %.2f" % (x["from_t"], x["to_t"], x["exponent"]))
    print("\nusing p = %.2f (the steepest observed): per edge ~%.1f min, three edges ~%.1f min"
          % (p, est / 60.0, est * 3 / 60.0))
    print("using last pair p = %.2f          : per edge ~%.1f min"
          % (p_last, est_last / 60.0))
    print("-> %s" % dest)


if __name__ == "__main__":
    main()
