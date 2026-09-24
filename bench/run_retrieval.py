# -*- coding: utf-8 -*-
"""Offline retrieval benchmark.

Compares the v0 keyword scorer (the original ``game_memory.py``) with the
v1 hybrid retriever on ``bench/data/retrieval_v1.json``. No LLM needed.

    python -m bench.run_retrieval                  # all systems, test split
    python -m bench.run_retrieval --split dev      # tune on dev only
    python -m bench.run_retrieval --no-dense       # skip the embedding model

Primary metric ``MemScore@3``: mean over queries of recall@3 for queries
that have gold memories, and 1/0 for "returned nothing" on negative
queries. It rewards finding the right memories *and* staying quiet when
nothing relevant exists, since both matter once memories go in a prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gamememo.personal.model import MemoryRecord, parse_time  # noqa: E402
from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402

DATA = os.path.join(ROOT, "bench", "data", "retrieval_v1.json")
K = 3

Searcher = Callable[[str], List[str]]


def load(path: str = DATA):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = [MemoryRecord.from_dict(m) for m in data["memories"]]
    return data, records, parse_time(data["now"])


# ---------------------------------------------------------------- systems

def v0_searcher(records: List[MemoryRecord]) -> Searcher:
    """The original scorer, run as-is.

    v0 asks an LLM for 2-5 query keywords before scoring. With no LLM in
    the loop we substitute jieba's TF-IDF keyword extractor, which is the
    closest deterministic stand-in; see docs/BENCHMARK.md for the caveat.
    """
    import jieba.analyse
    import game_memory as v0

    tmp = tempfile.mkdtemp(prefix="gm_v0_")
    gm = v0.GameMemory(user_id="bench", storage_dir=tmp)
    gm.memories = []
    for r in records:
        if not r.is_active:
            continue  # v0 overwrote or soft-deleted old versions
        m = v0.Memory(keywords=", ".join(r.keywords), content=r.content,
                      source=r.source, priority=6 - r.importance, mem_id=r.id)
        gm.memories.append(m)
    gm.save = lambda: None

    def fake_llm(prompt, debug=False):
        text = prompt.split("文本：", 1)[1].split("请以JSON", 1)[0].strip()
        return json.dumps({"keywords": jieba.analyse.extract_tags(text, topK=5)},
                          ensure_ascii=False)

    gm._call_llm = fake_llm

    def search(q: str) -> List[str]:
        return [m.id for m in gm.retrieval_relevant_memory(q, top_k=K)]

    return search


def v1_searcher(records, now, embedder=None, **cfg) -> Searcher:
    config = RetrievalConfig(top_k=K, **cfg) if embedder else RetrievalConfig.lexical_only(top_k=K, **cfg)
    retriever = HybridRetriever(embedder=embedder, config=config)

    def search(q: str) -> List[str]:
        return [s.record.id for s in retriever.search(q, records, now=now)]

    return search


# ---------------------------------------------------------------- metrics

def evaluate(search: Searcher, queries) -> Dict:
    per_type = defaultdict(list)
    rows = []
    t0 = time.time()
    for q in queries:
        got = search(q["query"])
        gold = q["gold"]
        if gold:
            recall = len(set(got[:K]) & set(gold)) / len(gold)
            rr = next((1.0 / (i + 1) for i, g in enumerate(got) if g in gold), 0.0)
            success = recall
        else:
            recall, rr = None, None
            success = 1.0 if not got else 0.0
        rows.append({"id": q["id"], "query": q["query"], "type": q["type"],
                     "gold": gold, "got": got, "success": success})
        per_type[q["type"]].append(success)
    latency_ms = (time.time() - t0) * 1000 / max(1, len(queries))

    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    return {
        "MemScore@3": mean(r["success"] for r in rows),
        "Recall@3": mean(r["success"] for r in pos),
        "MRR": mean(next((1.0 / (i + 1) for i, g in enumerate(r["got"]) if g in r["gold"]), 0.0)
                    for r in pos),
        "Abstain": mean(r["success"] for r in neg),
        "by_type": {t: round(mean(v), 4) for t, v in sorted(per_type.items())},
        "n": len(rows),
        "latency_ms": round(latency_ms, 2),
        "rows": rows,
    }


def mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


# ---------------------------------------------------------------- main

def build_systems(records, now, dense: bool):
    systems = {"v0-keyword (baseline)": (v0_searcher(records), 1)}
    systems["v1-lexical"] = (v1_searcher(records, now), 0)
    if dense:
        from gamememo.personal.embed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        systems["v1-dense"] = (v1_searcher(records, now, emb, use_lexical=False), 0)
        systems["v1-hybrid"] = (v1_searcher(records, now, emb), 0)
    return systems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default=None, help="write full JSON results here")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)

    data, records, now = load()
    queries = [q for q in data["queries"] if args.split == "all" or q["split"] == args.split]
    systems = build_systems(records, now, dense=not args.no_dense)

    results = {}
    for name, (search, llm_calls) in systems.items():
        res = evaluate(search, queries)
        res["llm_calls_per_query"] = llm_calls
        results[name] = res

    types = sorted({q["type"] for q in queries})
    print(f"\nsplit={args.split}  queries={len(queries)}  k={K}\n")
    head = f"{'system':24s} {'MemScore@3':>10s} {'Recall@3':>9s} {'MRR':>6s} {'Abstain':>8s} {'LLM/q':>6s}"
    print(head + "".join(f" {t[:10]:>10s}" for t in types))
    for name, r in results.items():
        line = (f"{name:24s} {r['MemScore@3']:10.3f} {r['Recall@3']:9.3f} {r['MRR']:6.3f} "
                f"{r['Abstain']:8.3f} {r['llm_calls_per_query']:6d}")
        print(line + "".join(f" {r['by_type'].get(t, 0):10.3f}" for t in types))

    if args.show_errors:
        for name, r in results.items():
            print(f"\n--- misses: {name}")
            for row in r["rows"]:
                if row["success"] < 1:
                    print(f"  {row['id']} [{row['type']}] {row['query']}  gold={row['gold']} got={row['got']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dataset": os.path.relpath(DATA, ROOT), "split": args.split,
                       "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "results": results}, f, ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    main()
