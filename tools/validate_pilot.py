"""Validate a pilot record against the `pilot-1` contract.

Reads the row-level jsonl plus its metadata block and runs the gates in
`design/23` §5 (G1..G17), then -- only when nothing blocks -- the derived
statistics the design permits.

    python tools/validate_pilot.py results/pilot        # dir with the two files
    python tools/validate_pilot.py --rows r.jsonl --meta m.json
    python tools/validate_pilot.py results/pilot --delta-pp 1.5 --out d.json

What it will never do, no matter what flags are passed:

  * invent or adjust `delta` -- a margin only ever comes from `--delta-pp`
    (i.e. from the preregistration), never from the record itself;
  * emit `NI_PASS` / `NI_REJECTED` with any authority -- every record carries
    `verdict_authority = false`, and with K=1 seed the CI excludes seed
    variance by construction;
  * write to registry / unlock / ledger / prereg, or submit anything.

Exit codes: 0 all gates pass, 1 a gate failed (or no data), 2 usage error.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import pilot as P  # noqa: E402

META_NAMES = ("pilot_meta.json", "meta.json", "pilot.meta.json")
ROWS_NAMES = ("pilot_rows.jsonl", "rows.jsonl", "pilot.rows.jsonl")


def find(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


def load_rows(path):
    """Load jsonl, a json array, or a json object holding a row list.

    A summary-only file stays empty on purpose: `{"summary": ...}` must not be
    coerced into rows, since refusing it is exactly what G6 exists for.
    """
    if path is None:
        return [], "no rows file found"
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return [], "rows file is empty"

    # A jsonl line also starts with '{', so try the whole document first and
    # fall back to line-by-line only when that fails.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj, "%d row(s) from json array" % len(obj)
        if isinstance(obj, dict):
            for k in ("rows", "records", "rows_by_arm"):
                v = obj.get(k)
                if isinstance(v, list):
                    return v, "%d row(s) from json object[%r]" % (len(v), k)
            return [], "json object without a row list (summary only)"
    except ValueError:
        pass

    rows, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            bad += 1
    return rows, "%d row(s) from jsonl%s" % (
        len(rows), (", %d unparsable line(s)" % bad) if bad else "")


def load_meta(path):
    if path is None:
        return None, "no meta file found"
    with open(path, encoding="utf-8") as f:
        return json.load(f), "meta loaded"


def print_gates(report):
    print("%-4s %-34s %-5s %s" % ("gate", "code", "state", "detail"))
    print("-" * 108)
    for g in report["gates"]:
        detail = g["detail"]
        if len(detail) > 60:
            detail = detail[:57] + "..."
        print("%-4s %-34s %-5s %s" % (g["id"], g["code"], g["status"], detail))


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def print_derived(derived):
    print()
    print("=== derived (pp; verdict_authority = false) ===")
    print("these numbers estimate variance only. They may not be cited as an H2")
    print("verdict or as a quality claim, and a single-seed CI excludes seed noise.")
    for name in sorted(derived["estimands"]):
        block = derived["estimands"][name]
        for rec in block["per_seed"]:
            print("\n[%s] seed=%s  n=%d/%d/%d" % (name, rec["seed"], rec["n_rows"],
                                                  rec["n_clusters"], rec["n_pairs"]))
            print("  mean_d            %s pp" % fmt(rec["mean_d_pp"]))
            print("  sd_d_row          %s pp   (sigma_D_by_seed, not sigma_D)"
                  % fmt(rec["sd_d_row_pp"]))
            print("  se_iid            %s pp   [diagnostic]" % fmt(rec["se_iid_pp"]))
            print("  se_cluster_w      %s pp   [the SE used for CI]"
                  % fmt(rec["se_cluster_w_pp"]))
            if rec.get("se_cluster_w_ci"):
                ci = rec["se_cluster_w_ci"]
                print("  se_cluster_w_ci   [%s, %s] pp  %s"
                      % (fmt(ci["ci_low"]), fmt(ci["ci_high"]),
                         ci.get("ci_method", "")))
            print("  CI                [%s, %s] on mean +- 1.96*se_cluster_w"
                  % (fmt(rec["ci_low"]), fmt(rec["ci_high"])))
            print("  deff_empirical    %s   [measured]" % fmt(rec["deff_empirical"]))
            print("  deff_model        %s   [1+0.18*rho_d, diagnostic]"
                  % fmt(rec["deff_model"]))
            print("  deff_bound        2.0000 universal / 1.1800 model-internal")
            print("  rho_d             %s  ci [%s, %s] over %d pairs"
                  % (fmt(rec["rho_d"]["point"], 3), fmt(rec["rho_d"]["ci_low"], 3),
                     fmt(rec["rho_d"]["ci_high"], 3), rec["rho_d"]["n_pairs"]))
            print("  rho_D             %s  (diagnostic only)"
                  % fmt(rec["rho_D"]["point"], 3))
            print("  sensitivity       %s" % rec["sensitivity_state"])
            print("  verdict           %s   (delta=%s; not citable)"
                  % (rec["verdict"], fmt(rec["delta_pp"], 2)))
            if rec["flags"]:
                print("  flags             %s" % ", ".join(rec["flags"]))
        agg = block["aggregate"]
        print("\n  -- %s aggregate: K=%d  mean=%s pp  sigma_s=%s pp  primary_seed=%s"
              % (name, agg["n_seeds"], fmt(agg["mean_d_pp"]), fmt(agg["sigma_s_pp"]),
                 agg["primary_seed"]))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", help="directory holding meta + rows")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--delta-pp", type=float, default=None,
                    help="frozen NI margin in pp, from the preregistration only")
    ap.add_argument("--bootstrap-b", type=int, default=P.BOOTSTRAP_B)
    ap.add_argument("--bootstrap-seed", type=int, default=P.BOOTSTRAP_SEED)
    ap.add_argument("--se-ci-outer", type=int, default=60,
                    help="nested bootstrap outer size for an interval on SE "
                         "(0 = skip; the record is then flagged se_ci_missing)")
    ap.add_argument("--se-ci-inner", type=int, default=100)
    ap.add_argument("--out", default=None, help="write the derived record here")
    ap.add_argument("--json", action="store_true", help="print the report as json")
    args = ap.parse_args(argv[1:])

    root = args.path or "."
    meta_path = args.meta or find(root, META_NAMES)
    rows_path = args.rows or find(root, ROWS_NAMES)

    meta, meta_note = load_meta(meta_path)
    rows, rows_note = load_rows(rows_path)

    report = P.validate_pilot(
        rows, meta if meta is not None else {},
        delta_pp=args.delta_pp,
        bootstrap_B=args.bootstrap_b,
        boot_seed=args.bootstrap_seed,
        se_ci=(args.se_ci_outer, args.se_ci_inner),
    )
    report["inputs"] = {"meta": meta_path, "rows": rows_path,
                        "meta_note": meta_note, "rows_note": rows_note}

    if args.json:
        print(P.dump_report(report))
    else:
        print("pilot validator  schema=%s  verdict_authority=false" % P.SCHEMA_VERSION)
        print("meta: %s (%s)" % (meta_path, meta_note))
        print("rows: %s (%s)" % (rows_path, rows_note))
        print()
        print_gates(report)
        print()
        print("blocking = %s" % report["blocking"])
        if report["flags"]:
            print("flags    = %s" % ", ".join(report["flags"]))
        if report["derived"]:
            print_derived(report["derived"])
        elif report["blocking"]:
            print("\nblocked: no statistics emitted (a record that fails a gate "
                  "must not produce numbers that could be quoted).")

    if args.out and report["derived"]:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report["derived"], f, ensure_ascii=False, indent=1)
        print("\nderived record written to %s" % args.out)
    elif args.out:
        print("\nno derived record written: gates blocked or no data")

    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
