"""Measure destination-specialization on the CPU reference (NOT a paper result).

Writes results/cpu/2c_specialization.json.

This is a mechanism probe on *synthetic* inputs: it tells us whether the 2C/2D
attribution path runs and how the B'-vs-C gap moves with budget. It is not, and
must never be labelled as, a quality result on real data.

Both arms run the frozen AM-HighestAttnKeys chain (since 2026-11-25) and differ
only in the reference query set:

  B'  one shared set, n_ref = 2t total, split across the three destination pools
  C   one set per destination, n_ref = 2t each

Records are emitted per edge (record.py), not per experiment: B' produces one
construction serving three destinations, C produces three, and that asymmetry is
itself the thing being attributed.

A warning that belongs next to the numbers: with n_ref = 2t, raising the budget
also raises the fitting sample, so a budget trend here confounds the two. The
four-budget sweep fixes n_ref = 13096 for exactly this reason.
"""

import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import adapter_compact as AC  # noqa: E402
from src.v2.cpu_ref import compact as C           # noqa: E402
from src.v2.cpu_ref import envelope as E          # noqa: E402
from src.v2.cpu_ref import record as R            # noqa: E402

SEEDS = (23, 41, 77)
BUDGETS = (4, 6, 8, 12, 24)
N_EVAL = 8
D_MODEL = 8
L = 96


def world(seed):
    rnd = random.Random(seed)
    k = [[rnd.gauss(0, 1) for _ in range(D_MODEL)] for _ in range(L)]
    v = [[rnd.gauss(0, 1) for _ in range(D_MODEL)] for _ in range(L)]
    return k, v


def subspace_queries(dims, n, seed):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        row = [0.0] * D_MODEL
        for d in dims:
            row[d] = rnd.gauss(0, 1)
        out.append(row)
    return out


def diag_slice(rec):
    """The diagnostic fields a cross-budget comparison needs, per edge."""
    bb = rec["diagnostics"]["beta_bounds"]
    bd = rec["diagnostics"]["beta_dead"]
    mr = rec["diagnostics"]["mass_residual"]
    bdes = rec["diagnostics"]["beta_design"]
    vdes = rec["diagnostics"]["value_design"]
    return {
        "beta_bind_hi_count": bb["bind_hi_count"],
        "beta_bind_hi_frac": bb["bind_hi_frac"],
        "beta_bind_lo_count": bb["bind_lo_count"],
        "beta_bind_lo_frac": bb["bind_lo_frac"],
        "beta_dead_count": bd["dead_count"],
        "beta_dead_frac": bd["dead_frac"],
        "mass_residual": mr["normalized"],
        "mass_residual_raw": mr["raw"],
        "beta_design_rank": bdes["rank"],
        "beta_design_cond": bdes["cond"],
        "value_design_rank": vdes["rank"],
        "value_design_cond": vdes["cond"],
        "payload_bytes": rec["payload"]["bytes"],
    }


def main():
    rows = []
    shared_records = []
    per_records = []
    stage_rec = None      # one representative B' construction, for the record

    for seed in SEEDS:
        k, v = world(seed)
        for budget in BUDGETS:
            n_ref = C.compute_n_ref(budget)          # 2t, the U3 ruling
            pool_n = 3 * n_ref                        # headroom for sampling
            pools = [subspace_queries((2 * d, 2 * d + 1), pool_n, seed + d)
                     for d in range(3)]
            evl = [subspace_queries((2 * d, 2 * d + 1), N_EVAL, seed + 100 + d)
                   for d in range(3)]

            shared = C.build_shared(pools, k, v, budget, n_ref=n_ref, seed=seed)
            per = C.build_per_destination(pools, k, v, budget, n_ref=n_ref, seed=seed)

            # B' is ONE construction feeding three destinations -> one record
            rec_s = R.edge_record(shared, source=seed, destination="all",
                                  calib_queries=None, store_tensors=False)
            R.validate(rec_s, strict=True)
            shared_records.append({"seed": seed, "budget": budget, **rec_s})
            if stage_rec is None:
                stage_rec = {"seed": seed, "budget": budget, "record": rec_s}

            for d in range(3):
                rec_d = R.edge_record(per[d], source=seed, destination=d,
                                      calib_queries=None, store_tensors=False)
                R.validate(rec_d, strict=True)
                per_records.append({"seed": seed, "budget": budget, **rec_d})

                e_shared = C.error_on(evl[d], k, v, shared)
                e_dcc = C.error_on(evl[d], k, v, per[d])
                row = {
                    "seed": seed,
                    "budget": budget,
                    "destination": d,
                    "n_ref_shared_total": shared["n_ref"],
                    "n_ref_per_edge": per[d]["n_ref"],
                    "rel_l2_shared_Bprime": e_shared,
                    "rel_l2_dcc_C": e_dcc,
                    "delta_dcc_minus_shared": e_dcc - e_shared,
                    "dcc_better": e_dcc <= e_shared,
                }
                row.update(diag_slice(rec_d))
                rows.append(row)

    # One 2C envelope, same shell as the 2A and 2D records. Not the whole sweep:
    # an envelope per cell would be 15 copies of the same statement, and the
    # sweep's own rows are already in this file.
    env_2c = AC.from_2c(stage_rec["record"], experiment_id="2c-cpu-sweep-v1",
                        label="B_prime_shared_seed%d_budget%d"
                              % (stage_rec["seed"], stage_rec["budget"]),
                        notes=["representative cell of the budget sweep",
                               "synthetic inputs; not a quality result"])

    wins = sum(1 for r in rows if r["dcc_better"])
    out = {
        "meta": {
            "status": "advisory_only_synthetic",
            "note": "synthetic inputs, CPU reference; NOT a paper result, NOT data for Table 4",
            "chain": "AM-HighestAttnKeys (softmax-RMS select, box beta [-3,3] iters=2, OLS value)",
            "backend": "cpu_ref_f64",
            "dtype_bytes": R.FLOAT_BYTES,
            "record_schema": R.SCHEMA_VERSION,
            "definitions": dict(R.DEFINITIONS),
            "L": L,
            "d_model": D_MODEL,
            "budgets": list(BUDGETS),
            "seeds": list(SEEDS),
            "n_ref_rule": "2t per edge (C); 2t total across destinations (B')",
            "n_heldout_queries": N_EVAL,
            "confound": ("budget and n_ref move together here, so the budget trend is "
                         "confounded with the fitting-sample size. The four-budget sweep "
                         "must use a fixed n_ref = 13096."),
            "envelope_schema": E.SCHEMA_VERSION,
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        },
        "summary": {
            "cells": len(rows),
            "dcc_better_cells": wins,
            "dcc_worse_cells": len(rows) - wins,
            "constructions_shared_Bprime": len(shared_records),
            "constructions_per_destination_C": len(per_records),
            "per_budget_win_table": {
                str(b): [all(r["dcc_better"] for r in rows
                             if r["budget"] == b and r["destination"] == d) for d in range(3)]
                for b in BUDGETS
            },
        },
        "rows": rows,
        "records_shared": shared_records,
        "records_per_destination": per_records,
        "stage_record": env_2c,
        "stage_record_summary": E.summary(env_2c),
    }

    dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "results", "cpu", "2c_specialization.json"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("budget | n_ref/edge | mean shared | mean dcc | dcc<shared | bind_hi | dead | bcond/vcond")
    for b in BUDGETS:
        sel = [r for r in rows if r["budget"] == b]
        ms = sum(r["rel_l2_shared_Bprime"] for r in sel) / len(sel)
        md = sum(r["rel_l2_dcc_C"] for r in sel) / len(sel)
        w = sum(1 for r in sel if r["dcc_better"])
        bh = sum(r["beta_bind_hi_frac"] for r in sel) / len(sel)
        dd = sum(r["beta_dead_frac"] for r in sel) / len(sel)
        bc = sum(r["beta_design_cond"] for r in sel) / len(sel)
        vc = sum(r["value_design_cond"] for r in sel) / len(sel)
        print("  %4d | %10d | %11.4f | %8.4f | %3d/%d     | %7.2f | %.2f | %8.1f/%8.1f"
              % (b, C.compute_n_ref(b), ms, md, w, len(sel), bh, dd, bc, vc))
    print("\ncells=%d  dcc_better=%d  dcc_worse=%d" % (len(rows), wins, len(rows) - wins))
    print("records: B'=%d  C=%d" % (len(shared_records), len(per_records)))
    sv = env_2c["verdict"]
    print("record 2C %-29s partial=%s" % (sv["overall"], sv["partial"]))
    print("-> %s" % dest)


if __name__ == "__main__":
    main()
