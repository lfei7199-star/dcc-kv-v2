"""Build `data/pilot/clusters.json`: row_id -> cluster_id / input_sha256 map.

Why this file exists
--------------------
`design/23` §2.1 makes `cluster_id` a *derived* value: the first 16 hex of
`sha256(serialize_input(context, question))`. A record may not be handed a
cluster id by a human, and the 182/164/18 split is a claim about the data that
has to be recomputed rather than asserted. This tool is that recomputation.

      python tools/build_pilot_clusters.py \
          --raw   PATH/TO/narrativeqa.jsonl \
          --index PATH/TO/longbench_qa.jsonl \
          --out   data/pilot/clusters.json

    python tools/build_pilot_clusters.py --raw ... --index ... --check

Hard checks (a failure exits 1 and writes nothing):

  * both files hold exactly 200 rows;
  * row order agrees -- `raw[i]["_id"] == index[i]["id"]` for every i, so
    `row_id` means the same thing in both files;
  * `raw[i]["length"] == index[i]["official_length_field"]`;
  * the grouping is 182 clusters of sizes {1: 164, 2: 18}, matching the frozen
    constants in `pilot.py` (N_CLUSTERS / CLUSTER_HISTOGRAM);
  * no two distinct 64-hex inputs share a 16-hex cluster id.

`row_id` is 0..199 and is the only key. `id` / `source_record_id` is carried as
provenance and explicitly flagged as *not* a key: the 200 records have 200
distinct ids but only 182 distinct inputs, so using `id` to deduplicate would
destroy every cluster.

Stdlib only. Imports `pilot` for the frozen serializer -- never re-implements
the prefix rule, so a change to `pilot-input/v1` cannot leave this tool behind.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import pilot as P  # noqa: E402

MAP_SCHEMA = "pilot-cluster-map/1"
DEFAULT_OUT = os.path.join("data", "pilot", "clusters.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise SystemExit("bad json at %s:%d: %s" % (path, lineno, exc))
    return rows


def build(raw_path, index_path):
    """Return (document, report). Raises SystemExit on a hard-check failure."""
    raw = load_jsonl(raw_path)
    index = load_jsonl(index_path)
    problems = []

    if len(raw) != P.N_ROWS:
        problems.append("raw rows = %d, expected %d" % (len(raw), P.N_ROWS))
    if len(index) != P.N_ROWS:
        problems.append("index rows = %d, expected %d" % (len(index), P.N_ROWS))

    n = min(len(raw), len(index))
    order_bad = [i for i in range(n) if raw[i].get("_id") != index[i].get("id")]
    if order_bad:
        problems.append("row order differs at %d rows, first: %s"
                        % (len(order_bad), order_bad[:5]))
    length_bad = [i for i in range(n) if raw[i].get("length") != index[i].get("official_length_field")]
    if length_bad:
        problems.append("official length differs at %d rows, first: %s"
                        % (len(length_bad), length_bad[:5]))

    shas, empty_ctx = [], []
    for i in range(n):
        ctx = raw[i].get("context")
        q = raw[i].get("input")
        if not isinstance(ctx, str) or not ctx.strip():
            empty_ctx.append(i)
            continue
        if not isinstance(q, str) or not q.strip():
            empty_ctx.append(i)
            continue
        shas.append(P.input_sha256(ctx, q))
    if empty_ctx:
        problems.append("empty context/question at rows %s" % (empty_ctx[:5],))

    if problems:
        for p in problems:
            print("FAIL  " + p)
        raise SystemExit(1)

    # group, keeping first-appearance order for stable output
    order, members = [], {}
    for i, sha in enumerate(shas):
        if sha not in members:
            members[sha] = []
            order.append(sha)
        members[sha].append(i)

    hist = {}
    for sha in order:
        m = len(members[sha])
        hist[m] = hist.get(m, 0) + 1

    collisions = P.short_id_collisions(order)

    if len(order) != P.N_CLUSTERS:
        print("FAIL  clusters = %d, frozen constant is %d" % (len(order), P.N_CLUSTERS))
        raise SystemExit(1)
    if hist != P.CLUSTER_HISTOGRAM:
        print("FAIL  histogram = %r, frozen is %r" % (hist, P.CLUSTER_HISTOGRAM))
        raise SystemExit(1)
    if collisions:
        print("FAIL  %d short-id collisions: %r" % (len(collisions), collisions[:3]))
        raise SystemExit(1)

    rows, pairs, gold_equal = [], [], []
    for sha in order:
        idxs = members[sha]
        cid = P.cluster_id_from_sha(sha)
        if len(idxs) > 1:
            pairs.append(tuple(idxs))
            seen = set()
            for i in idxs:
                key = tuple(raw[i].get("answers") or ())
                if key in seen:
                    gold_equal.append(idxs)
                    break
                seen.add(key)
        for gold_index, i in enumerate(idxs):
            rows.append({
                "row_id": i,
                "cluster_id": cid,
                "input_sha256": sha,
                "cluster_size": len(idxs),
                "gold_index": gold_index,
                "source_record_id": raw[i]["_id"],
                "context_chars": index[i].get("context_chars"),
                "prompt_tokens_full": index[i].get("prompt_tokens_full"),
            })

    rows.sort(key=lambda r: r["row_id"])

    doc = {
        "schema": MAP_SCHEMA,
        "serialization": P.CLUSTER_SERIALIZATION,
        "cluster_id_rule": "input_sha256[:16]",
        "row_key": "row_id",
        "source_record_id_is_not_a_key": True,
        "note": "input identity map only; carries no scores and no verdict authority",
        "dataset": "narrativeqa",
        "split": "test",
        "n_rows": len(rows),
        "n_clusters": len(order),
        "cluster_size_histogram": {str(k): hist[k] for k in sorted(hist)},
        "max_cluster_size": max(hist),
        "short_id_collisions": len(collisions),
        "sources": {
            "raw": {"name": os.path.basename(raw_path),
                    "bytes": os.path.getsize(raw_path),
                    "sha256": sha256_file(raw_path)},
            "index": {"name": os.path.basename(index_path),
                      "bytes": os.path.getsize(index_path),
                      "sha256": sha256_file(index_path)},
        },
        "rows": rows,
    }
    report = {
        "problems": [],
        "pairs": sorted(pairs),
        "gold_equal_pairs": gold_equal,
        "hist": hist,
        "clusters": len(order),
        "collisions": len(collisions),
    }
    return doc, report


def render(doc):
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="build/verify the pilot cluster map")
    ap.add_argument("--raw", required=True, help="narrativeqa.jsonl (original, unfiltered)")
    ap.add_argument("--index", required=True, help="longbench_qa.jsonl (200-row index)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if the file on disk differs")
    args = ap.parse_args(argv)

    doc, rep = build(args.raw, args.index)
    text = render(doc)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print("rows             %d" % len(doc["rows"]))
    print("clusters         %d" % rep["clusters"])
    print("histogram        %s" % json.dumps(doc["cluster_size_histogram"]))
    print("short-id dupes   %d" % rep["collisions"])
    print("size-2 pairs     %d" % len(rep["pairs"]))
    print("pairs            %s" % rep["pairs"])
    if rep["gold_equal_pairs"]:
        print("WARN identical gold inside cluster: %s" % rep["gold_equal_pairs"])
    print("raw   sha256     %s" % doc["sources"]["raw"]["sha256"])
    print("index sha256     %s" % doc["sources"]["index"]["sha256"])
    print("clusters.json    %s" % digest)

    if args.check:
        if not os.path.exists(args.out):
            print("FAIL  %s does not exist" % args.out)
            return 1
        with open(args.out, encoding="utf-8") as fh:
            on_disk = fh.read()
        if on_disk != text:
            print("FAIL  %s differs from a fresh build" % args.out)
            return 1
        print("CHECK OK  %s is byte-identical to a fresh build" % args.out)
        return 0

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("WROTE %s (%d bytes)" % (args.out, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
