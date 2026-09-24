# -*- coding: utf-8 -*-
"""SOTA gate: may a candidate system replace what is on ``main``?

    python -m bench.run_retrieval --out bench/results/candidate.json
    python -m bench.gate bench/results/candidate.json --system v2-hybrid \
        --sota <the SOTA record of the branch you want to enter>

In CI the record comes from the base branch (``git show origin/main:bench/sota.json``),
because a promoting PR updates its own ``bench/sota.json`` to the new numbers.

Rules:
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
    """Guards are ``{metric: tolerance}`` (higher is better) or
    ``{metric: {"tol": t, "lower_is_better": true}}``."""
    primary = sota["primary"]
    lines, ok = [], True
    gain = candidate[primary] - sota["metrics"][primary]
    passed = gain >= sota["min_gain"]
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  {primary:12s} {sota['metrics'][primary]:.3f} -> "
                 f"{candidate[primary]:.3f}  (need >= +{sota['min_gain']:.3f})")
    for metric, spec in sota["guards"].items():
        tol = spec["tol"] if isinstance(spec, dict) else spec
        lower = isinstance(spec, dict) and spec.get("lower_is_better", False)
        worse = (candidate[metric] - sota["metrics"][metric]) if lower else (sota["metrics"][metric] - candidate[metric])
        passed = worse <= tol + 1e-9
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {metric:12s} {sota['metrics'][metric]:.3f} -> "
                     f"{candidate[metric]:.3f}  (max {'rise' if lower else 'drop'} {tol:.3f})")
    return ok, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--system", required=True)
    ap.add_argument("--sota", default=SOTA)
    args = ap.parse_args(argv)

    with open(args.sota, encoding="utf-8") as f:
        sota = json.load(f)
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    if results.get("split") != sota["split"] or results.get("dataset") != sota["dataset"]:
        sys.exit(f"results are for {results.get('dataset')}:{results.get('split')}, "
                 f"gate needs {sota['dataset']}:{sota['split']}")

    ok, lines = check(results["results"][args.system], sota)
    print(f"candidate: {args.system}   vs SOTA: {sota['system']} ({sota['ref']})")
    print("\n".join(lines))
    print("=> PROMOTABLE" if ok else "=> NOT PROMOTABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
