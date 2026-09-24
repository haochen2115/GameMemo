# -*- coding: utf-8 -*-
"""SOTA gate: may a candidate system replace what is on ``main``?

    python -m bench.run_retrieval --out bench/results/candidate.json
    python -m bench.gate bench/results/candidate.json --system v1-hybrid

Rules (``bench/sota.json``):
- the primary metric must beat the recorded SOTA by at least ``min_gain``;
- no guard metric may drop by more than its tolerance.
Exit code 0 = promotable, 1 = not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "bench", "sota.json")


def check(candidate: dict, sota: dict):
    primary = sota["primary"]
    lines, ok = [], True
    gain = candidate[primary] - sota["metrics"][primary]
    passed = gain >= sota["min_gain"]
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  {primary:12s} {sota['metrics'][primary]:.3f} -> "
                 f"{candidate[primary]:.3f}  (need >= +{sota['min_gain']:.3f})")
    for metric, tol in sota["guards"].items():
        drop = sota["metrics"][metric] - candidate[metric]
        passed = drop <= tol
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {metric:12s} {sota['metrics'][metric]:.3f} -> "
                     f"{candidate[metric]:.3f}  (max drop {tol:.3f})")
    return ok, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--system", required=True)
    args = ap.parse_args(argv)

    with open(SOTA, encoding="utf-8") as f:
        sota = json.load(f)
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    if results.get("split") != sota["split"]:
        sys.exit(f"results are for split={results.get('split')!r}, gate needs {sota['split']!r}")

    ok, lines = check(results["results"][args.system], sota)
    print(f"candidate: {args.system}   vs SOTA: {sota['system']} ({sota['ref']})")
    print("\n".join(lines))
    print("=> PROMOTABLE" if ok else "=> NOT PROMOTABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
