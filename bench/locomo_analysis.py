# -*- coding: utf-8 -*-
"""Analysis for E19 (LoCoMo + Mem0): system table, where Mem0 loses, and
the full-history vs retrieval length curve.

    python -m bench.locomo_analysis --all all.json --prefix s5.json,s12.json

Loss decomposition for a retrieval-based memory M (Mem0), per question,
using the judge's labels (F1 >= 0.5 when no judge has run):
    write loss       M's whole store in the prompt (mem0all) is wrong, while the
                     whole conversation (full) is right: the information was
                     lost when the conversation was rewritten into memories
    retrieval loss   mem0all is right but M's top-k (mem0:k) is wrong
    reader loss      even the whole conversation is wrong
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.locomo import CATEGORIES  # noqa: E402


def ok(row: Dict) -> float:
    return row["judge"] if "judge" in row else (1.0 if row["f1"] >= 0.5 else 0.0)


def paired(a: List[Dict], b: List[Dict], n: int = 10000, seed: int = 0) -> Dict:
    """b - a over the questions both answered."""
    ia, ib = {r["id"]: ok(r) for r in a}, {r["id"]: ok(r) for r in b}
    ids = sorted(set(ia) & set(ib))
    d = [ib[i] - ia[i] for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {"delta": round(sum(d) / len(d), 4), "ci95": [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)],
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "n": len(ids)}


def table(results: Dict[str, Dict]) -> List[str]:
    lines = ["| context | n | J (or F1>=0.5) | F1 | " + " | ".join(CATEGORIES.values()) + " | prompt chars |",
             "|---|---|---|---|" + "---|" * len(CATEGORIES) + "---|"]
    for spec, res in results.items():
        rows = res["rows"]
        by = defaultdict(list)
        for r in rows:
            by[r["category"]].append(ok(r))
        acc = sum(ok(r) for r in rows) / len(rows)
        cats = " | ".join(f"{sum(by[c]) / len(by[c]):.3f}" if by[c] else "-" for c in CATEGORIES)
        lines.append(f"| `{spec}` | {len(rows)} | {acc:.3f} | {res['F1']:.3f} | {cats} | {res['context_chars']} |")
    return lines


def decompose(results: Dict[str, Dict], system: str = "mem0:10") -> Dict:
    need = [system, "mem0all", "full"]
    if not all(k in results for k in need):
        return {}
    by = {k: {r["id"]: r for r in results[k]["rows"]} for k in need}
    ids = set.intersection(*(set(v) for v in by.values()))
    out = Counter()
    per_cat = defaultdict(Counter)
    for i in ids:
        if ok(by[system][i]):
            c = "correct"
        elif ok(by["mem0all"][i]):
            c = "retrieval loss"
        elif ok(by["full"][i]):
            c = "write loss"
        else:
            c = "reader loss"
        out[c] += 1
        per_cat[CATEGORIES[by[system][i]["category"]]][c] += 1
    return {"total": dict(out), "by_category": {k: dict(v) for k, v in per_cat.items()}, "n": len(ids)}


def evidence_vs_correct(results: Dict[str, Dict], spec: str = "raw:10") -> Dict:
    rows = results.get(spec, {}).get("rows", [])
    if not rows or "evidence_recall" not in rows[0]:
        return {}
    full_hit = [r for r in rows if r["evidence_recall"] == 1.0]
    miss = [r for r in rows if r["evidence_recall"] == 0.0]
    avg = lambda xs: round(sum(ok(r) for r in xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"all evidence retrieved": {"n": len(full_hit), "acc": avg(full_hit)},
            "no evidence retrieved": {"n": len(miss), "acc": avg(miss)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True)
    ap.add_argument("--prefix", default="", help="comma-separated answer files of --sessions N runs")
    args = ap.parse_args(argv)
    res = json.load(open(args.all))["results"]
    print("\n".join(table(res)))
    print()
    for a, b in [("mem0:10", "raw:10"), ("full", "raw:10"), ("mem0all", "full"), ("mem0:10", "mem0all")]:
        if a in res and b in res:
            print(f"{b} - {a}: {paired(res[a]['rows'], res[b]['rows'])}")
    print("\nloss decomposition (mem0:10):", json.dumps(decompose(res), ensure_ascii=False))
    print("raw:10 evidence vs correctness:", evidence_vs_correct(res))
    for path in [p for p in args.prefix.split(",") if p]:
        d = json.load(open(path))
        r = d["results"]
        print(f"\nfirst {d.get('sessions')} sessions:")
        print("\n".join(table(r)))
        if "full" in r and "raw:10" in r:
            print(f"raw:10 - full: {paired(r['full']['rows'], r['raw:10']['rows'])}")


if __name__ == "__main__":
    main()
