# -*- coding: utf-8 -*-
"""Paired read-path comparison on stored dev memories (no LLM).

    GAMEMEMO_RUNS=<dir with e2e result files> python bench/replay_dev.py as-written,concepts

The first variant is the baseline; every other variant is evaluated on the
same questions and the same stored memories (paired bootstrap). SETS lists
the result files used in docs/EXPERIMENTS.md.
"""
import io, contextlib, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.run_e2e import replay
S = os.environ.get("GAMEMEMO_RUNS", "runs")  # directory with stored e2e result files
SETS = [("c3_e2e_v4dev", "e2e_v4", "p1"), ("c3_e2e_v3dev", "e2e_v3", "p1"), ("c2_e2e_v3dev", "e2e_v3", "p1"),
        ("c2_e2e_v2dev", "e2e_v2", "p1"), ("p1b_e2e_v2dev", "e2e_v2", "p1"), ("p1b_e2e_v1dev", "e2e_v1", "p1"),
        ("f0_e2e_v5dev", "e2e_v5", "p2"),
        # e2e_v5 test players, retired to development after the P3 report
        ("e2e_v5_test", "e2e_v5", "p3"),
        # e2e_v6 test players, retired to development after the P4 report
        ("e2e_v6_test", "e2e_v6", "p4")]
variants = sys.argv[1].split(",")
base = variants[0]
rows = {v: [] for v in variants}
for n, d, sysname in SETS:
    if not os.path.exists(f"{S}/{n}.json"):
        continue
    line = f"{n:16s}"
    for v in variants:
        with contextlib.redirect_stdout(io.StringIO()):
            r = replay(f"{S}/{n}.json", f"bench/data/{d}.json", sysname, v, False)[f"{sysname}/{v}"]
        rows[v] += r["rows"]
        line += f" {v}={r['E2EScore']:.3f}(fact {r['by_type'].get('fact',0):.2f}, neg {r['by_type'].get('negative',0):.2f})"
    print(line, flush=True)
for v in variants[1:]:
    d = [a["success"] - b["success"] for a, b in zip(rows[v], rows[base])]
    n = len(d); random.seed(0)
    bs = sorted(sum(random.choice(d) for _ in range(n)) / n for _ in range(5000))
    print(f"{v:22s} vs {base}: {sum(d)/n:+.3f} CI [{bs[125]:+.3f},{bs[4875]:+.3f}] better/worse {sum(x>0 for x in d)}/{sum(x<0 for x in d)} n={n}")
