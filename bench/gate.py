# -*- coding: utf-8 -*-
"""SOTA gate: may this branch replace what is on ``main``?

A promotion must *improve* at least one benchmark and *hold* (not regress
on) every other one. The candidate is declared in ``bench/candidate.json``,
committed before the test run, which doubles as the pre-registration:

    {"retrieval": {"system": "memory-default", "mode": "hold"},
     "e2e": {"system": "v2+po+history", "mode": "improve",
             "results": "bench/results/e2e_v1_test.json"}}

    python -m bench.gate --candidate bench/candidate.json \
        --retrieval-results bench/results/candidate.json --base-dir /tmp/base

``--base-dir`` holds the base branch's SOTA records (``sota.json``,
``sota_e2e.json``); CI extracts them with ``git show origin/main:...``.
A missing record, or one measured on a different dataset, falls back to
the baseline shipped in this checkout (base code on the new dataset).

Rules per benchmark record:
- improve: primary metric >= SOTA + ``min_gain``;
- hold: primary metric >= SOTA - ``hold_tol``;
- always: no guard metric worse than its tolerance.
Exit code 0 = promotable, 1 = not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, "bench", "sota.json")
RECORDS = {"retrieval": ("sota.json", "baseline_v0.json"),
           "e2e": ("sota_e2e.json", "baseline_e2e.json")}


def check(candidate: dict, sota: dict, mode: str = "improve"):
    """Guards are ``{metric: tolerance}`` (higher is better) or
    ``{metric: {"tol": t, "lower_is_better": true}}``."""
    primary = sota["primary"]
    lines, ok = [], True
    delta = candidate[primary] - sota["metrics"][primary]
    if mode == "improve":
        passed, need = delta >= sota["min_gain"] - 1e-9, f">= +{sota['min_gain']:.3f}"
    else:
        tol = sota.get("hold_tol", 0.02)
        passed, need = delta >= -tol - 1e-9, f">= -{tol:.3f}"
    ok &= passed
    lines.append(f"{'PASS' if passed else 'FAIL'}  {primary:12s} {sota['metrics'][primary]:.3f} -> "
                 f"{candidate[primary]:.3f}  ({mode}: {need})")
    for metric, spec in sota["guards"].items():
        tol = spec["tol"] if isinstance(spec, dict) else spec
        lower = isinstance(spec, dict) and spec.get("lower_is_better", False)
        worse = (candidate[metric] - sota["metrics"][metric]) if lower else (sota["metrics"][metric] - candidate[metric])
        passed = worse <= tol + 1e-9
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {metric:12s} {sota['metrics'][metric]:.3f} -> "
                     f"{candidate[metric]:.3f}  (max {'rise' if lower else 'drop'} {tol:.3f})")
    return ok, lines


def load_record(bench: str, base_dir: str, dataset: str = "") -> dict:
    """The base branch's SOTA record for ``bench``. When the base has none,
    or it was measured on another dataset (the benchmark moved to a fresh
    sealed test set), fall back to the baseline shipped with this branch,
    which must be the base code measured on the new dataset."""
    name, fallback = RECORDS[bench]
    path = os.path.join(base_dir, name) if base_dir else os.path.join(ROOT, "bench", name)
    record = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    if record is None or (dataset and record.get("dataset") != dataset):
        with open(os.path.join(ROOT, "bench", fallback), encoding="utf-8") as f:
            record = json.load(f)
    return record


def _candidate_dataset(bench: str, entry: dict, retrieval_results: str) -> str:
    path = retrieval_results if bench == "retrieval" else os.path.join(ROOT, entry["results"])
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("dataset", "")


def candidate_metrics(bench: str, entry: dict, retrieval_results: str, record: dict) -> dict:
    if bench == "retrieval":
        with open(retrieval_results, encoding="utf-8") as f:
            res = json.load(f)
        _same_data(res, record)
        return res["results"][entry["system"]]
    # e2e: never trust stored numbers; re-score the stored evidence.
    from bench.run_e2e import rejudge

    path = os.path.join(ROOT, entry["results"])
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    _same_data(res, record)
    if res.get("model") != record.get("model"):
        sys.exit(f"e2e results use model {res.get('model')!r}, record needs {record.get('model')!r}")
    return rejudge(path, os.path.join(ROOT, record["dataset"]), show_errors=False)[entry["system"]]


def _same_data(res: dict, record: dict) -> None:
    if res.get("split") != record["split"] or res.get("dataset") != record["dataset"]:
        sys.exit(f"results are for {res.get('dataset')}:{res.get('split')}, "
                 f"gate needs {record['dataset']}:{record['split']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", help="(single-benchmark mode) retrieval results file")
    ap.add_argument("--system", help="(single-benchmark mode) retrieval system name")
    ap.add_argument("--sota", default=SOTA, help="(single-benchmark mode) SOTA record")
    ap.add_argument("--candidate", help="bench/candidate.json")
    ap.add_argument("--retrieval-results", default=os.path.join(ROOT, "bench", "results", "candidate.json"))
    ap.add_argument("--base-dir", default="", help="directory with the base branch's SOTA records")
    args = ap.parse_args(argv)

    if not args.candidate:  # single-benchmark mode
        with open(args.sota, encoding="utf-8") as f:
            sota = json.load(f)
        with open(args.results, encoding="utf-8") as f:
            results = json.load(f)
        _same_data(results, sota)
        ok, lines = check(results["results"][args.system], sota)
        print(f"candidate: {args.system}   vs SOTA: {sota['system']} ({sota['ref']})")
        print("\n".join(lines))
        print("=> PROMOTABLE" if ok else "=> NOT PROMOTABLE")
        return 0 if ok else 1

    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)
    all_ok, improved = True, False
    for bench, entry in cand.items():
        if bench.startswith("_"):
            continue
        record = load_record(bench, args.base_dir, _candidate_dataset(bench, entry, args.retrieval_results))
        metrics = candidate_metrics(bench, entry, args.retrieval_results, record)
        ok, lines = check(metrics, record, entry["mode"])
        print(f"[{bench}] candidate {entry['system']} ({entry['mode']})   "
              f"vs SOTA: {record['system']} ({record['ref']})")
        print("\n".join("  " + l for l in lines))
        all_ok &= ok
        improved |= ok and entry["mode"] == "improve"
    if not improved:
        print("no benchmark is marked 'improve' and passing")
    ok = all_ok and improved
    print("=> PROMOTABLE" if ok else "=> NOT PROMOTABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
