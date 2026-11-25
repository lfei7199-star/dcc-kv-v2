"""Re-read the stored stage records and validate them from the file.

This is the check that a record is *readable*, not just writable: it goes through
`envelope.from_json` (schema gate first) and then `validate`, exactly as a later
reader would. Nothing is recomputed and no solver is imported.

Usage:
    python tools/validate_records.py [results_dir]

Exit code 0 when every stored record validates, 1 otherwise.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.v2.cpu_ref import envelope as E  # noqa: E402

SOURCES = (
    ("2d_closure.json", ("records",)),
    ("2c_specialization.json", ("stage_record",)),
)


def collect(path, keypath):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    node = obj
    for k in keypath:
        node = node.get(k) if isinstance(node, dict) else None
    if node is None:
        return []
    if isinstance(node, dict) and node.get("schema_version"):
        return [(keypath[-1], node)]
    return [(k, v) for k, v in node.items()
            if isinstance(v, dict) and v.get("schema_version")]


def main(argv):
    root = os.path.abspath(argv[1] if len(argv) > 1
                           else os.path.join(os.path.dirname(__file__), "..",
                                             "results", "cpu"))
    rows, bad = [], 0
    for name, keypath in SOURCES:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            rows.append((name, "-", "ABSENT", "file not found"))
            continue
        for label, rec in collect(path, keypath):
            try:
                rec = E.from_json(json.dumps(rec))
                problems = E.validate(rec, strict=False)
            except E.SchemaMismatchError as exc:
                rows.append((name, label, "SCHEMA", str(exc)))
                bad += 1
                continue
            v = rec["verdict"]
            rows.append((name, "%s (%s)" % (label, rec["stage"]), v["overall"],
                         "ok" if not problems else "; ".join(problems)))
            if problems:
                bad += 1

    print("%-24s %-14s %-30s %s" % ("file", "record", "verdict", "problems"))
    for r in rows:
        print("%-24s %-14s %-30s %s" % r)
    print("\n%d record(s), %d with problems" % (len(rows), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
