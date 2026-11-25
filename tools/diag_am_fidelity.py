"""Diagnostic: how much the pre-upgrade skeleton differed from AM, in numbers.

Runs four arms through the *real* construction chain in src/v2/cpu_ref/compact.py
(no duplicated linear algebra here -- an earlier version of this script carried
its own copy and drifted from the module, which is how three solver bugs went
unnoticed).

  legacy        logit-RMS select, uniform beta, ridge 1e-3
  +box beta     logit-RMS select, box beta, ridge 1e-3
  +OLS value    logit-RMS select, uniform beta, OLS
  AM            softmax-RMS select, box beta [-3,3], OLS      <- frozen chain

Reported per arm: single-block rel-L2, merged rel-L2 against a fixed block
(AM Eq.1+2), mass residual over the block, and the bound-saturation fraction.

Status of every number: synthetic_random_gaussian, advisory only. Not for Table 4.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import compact as C  # noqa: E402

T = 300
D = 12
FIXED = 48
N_EVAL = 24
SEEDS = (7, 23, 41)
RATIOS = (("40x", 40), ("20x", 20), ("10x", 10), ("5x", 5))

ARMS = {
    "legacy(logitRMS, uniform, ridge)": dict(selector="logit_rms", beta_mode="uniform",
                                             value_mode="ridge"),
    "+box beta(logitRMS, box, ridge)": dict(selector="logit_rms", beta_mode="box",
                                            value_mode="ridge"),
    "+OLS value(logitRMS, uniform, ols)": dict(selector="logit_rms", beta_mode="uniform",
                                               value_mode="ols"),
    "AM(softmaxRMS, box, ols)": dict(selector="softmax_rms", beta_mode="box",
                                     value_mode="ols"),
}


def make_world(n, seed):
    rnd = random.Random(seed)
    return ([[rnd.gauss(0, 1) for _ in range(D)] for _ in range(n)],
            [[rnd.gauss(0, 1) for _ in range(D)] for _ in range(n)])


def make_queries(n, seed):
    rnd = random.Random(seed)
    return [[rnd.gauss(0, 1) for _ in range(D)] for _ in range(n)]


def main():
    out = {"status": "advisory_only_synthetic", "T": T, "d": D,
           "beta_box": [C.BETA_LO, C.BETA_HI], "rows": [], "sweep": []}

    for seed in SEEDS:
        k, v = make_world(T, seed)
        k_fix, v_fix = make_world(FIXED, seed + 500)
        q_eval = make_queries(N_EVAL, seed + 900)

        for label, ratio in RATIOS:
            t = max(4, T // ratio)
            n_ref = C.compute_n_ref(t)
            q_fit = make_queries(n_ref, seed + 1)
            for name, cfg in ARMS.items():
                try:
                    rep = C.build(q_fit, k, v, t, **cfg)
                    single = C.error_on(q_eval, k, v, rep)
                    merged = C.merge_rel_l2(q_eval, k, v, k_fix, v_fix, rep)
                    diag = rep["diagnostics"]["beta"]
                except C.RankDeficientError as exc:
                    out["rows"].append({"seed": seed, "ratio": label, "t": t, "arm": name,
                                        "error": str(exc)[:80], "rel_l2": None,
                                        "merge_rel_l2": None, "bind_hi": None,
                                        "mass_residual": None, "beta_max": None})
                    continue
                out["rows"].append({
                    "seed": seed, "ratio": label, "t": t, "n_ref": n_ref, "arm": name,
                    "rel_l2": round(single, 6),
                    "merge_rel_l2": round(merged, 6),
                    "bind_hi": None if diag.get("bind_hi_frac") is None else round(diag["bind_hi_frac"], 4),
                    "mass_residual": None if diag.get("mass_residual") is None else round(diag["mass_residual"], 4),
                    "beta_max": None if diag.get("beta_max") is None else round(diag["beta_max"], 4),
                })

        # how many reference queries does a per-key beta need?
        t = 20
        for mult in (1, 2, 4, 8):
            q_fit = make_queries(mult * t, seed + 3)
            for name in ("legacy(logitRMS, uniform, ridge)", "AM(softmaxRMS, box, ols)"):
                try:
                    rep = C.build(q_fit, k, v, t, **ARMS[name])
                    m = C.merge_rel_l2(q_eval, k, v, k_fix, v_fix, rep)
                except C.RankDeficientError:
                    m = None
                out["sweep"].append({"seed": seed, "t": t, "n_over_t": mult, "arm": name,
                                     "merge_rel_l2": None if m is None else round(m, 6)})

    dest = os.path.join(os.path.dirname(__file__), "..", "results", "cpu",
                        "diag_am_fidelity.json")
    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    def mean(vals):
        vals = [x for x in vals if x is not None]
        return sum(vals) / len(vals) if vals else None

    print("T=%d  box=[%.0f,%.0f]  n_ref=2t per arm" % (T, C.BETA_LO, C.BETA_HI))
    for metric in ("rel_l2", "merge_rel_l2"):
        print("\n[%s]  (lower is better)" % metric)
        print("%-38s" % "arm" + "".join("%12s" % r for r, _ in RATIOS))
        for name in ARMS:
            line = "%-38s" % name
            for label, _ in RATIOS:
                v = mean([r[metric] for r in out["rows"]
                          if r["arm"] == name and r["ratio"] == label])
                line += "%12s" % ("n/a" if v is None else "%.4f" % v)
            print(line)

    print("\n[beta bound saturation on the AM arm]  fraction pinned at e^+3")
    for label, _ in RATIOS:
        v = mean([r["bind_hi"] for r in out["rows"]
                  if r["arm"] == "AM(softmaxRMS, box, ols)" and r["ratio"] == label])
        bmax = mean([r["beta_max"] for r in out["rows"]
                     if r["arm"] == "AM(softmaxRMS, box, ols)" and r["ratio"] == label])
        print("  %-5s bind_hi=%s  beta_max=%s"
              % (label, "n/a" if v is None else "%.2f" % v,
                 "n/a" if bmax is None else "%.2f" % bmax))

    print("\n[reference-query sweep]  merged rel-L2 vs n_ref/t  (t=20)")
    print("%-38s" % "arm" + "".join("%10s" % ("%dx" % m) for m in (1, 2, 4, 8)))
    for name in ("legacy(logitRMS, uniform, ridge)", "AM(softmaxRMS, box, ols)"):
        line = "%-38s" % name
        for mult in (1, 2, 4, 8):
            v = mean([r["merge_rel_l2"] for r in out["sweep"]
                      if r["arm"] == name and r["n_over_t"] == mult])
            line += "%10s" % ("n/a" if v is None else "%.4f" % v)
        print(line)

    print("\nwritten: %s\nstatus: advisory_only_synthetic (NOT for Table 4)" % dest)


if __name__ == "__main__":
    main()
