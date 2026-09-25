# -*- coding: utf-8 -*-
"""Offline retrieval benchmark. No LLM needed.

    python -m bench.run_retrieval                         # default systems, test split
    python -m bench.run_retrieval --split dev             # tune on dev only
    python -m bench.run_retrieval --systems v0,v1-hybrid  # pick systems
    python -m bench.run_retrieval --list                  # show registered systems

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
from functools import lru_cache
from typing import Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gamememo.personal.model import MemoryRecord, parse_time  # noqa: E402
from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402

DATA = os.path.join(ROOT, "bench", "data", "retrieval_v2.json")
K = 3

Searcher = Callable[[str], List[str]]
Factory = Callable[[List[MemoryRecord], datetime], Searcher]


def load(path: str = DATA):
    """Return (data, {player_id: (records, now)}). Accepts v1 single-player files."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "players" not in data:  # v1 format
        data = dict(data, players=[{"id": "p0", "now": data["now"], "memories": data["memories"]}],
                    queries=[dict(q, player="p0") for q in data["queries"]])
    banks = {p["id"]: ([MemoryRecord.from_dict(m) for m in p["memories"]], parse_time(p["now"]))
             for p in data["players"]}
    return data, banks


# ---------------------------------------------------------------- systems

def v0_factory(records: List[MemoryRecord], now: datetime) -> Searcher:
    """The original v0 scorer, run as-is.

    v0 asks an LLM for 2-5 query keywords before scoring. With no LLM in the
    loop we substitute jieba's TF-IDF keyword extractor, the closest
    deterministic stand-in (see docs/BENCHMARK.md for the caveat).
    """
    import jieba.analyse
    import game_memory as v0

    gm = v0.GameMemory(user_id="bench", storage_dir=tempfile.mkdtemp(prefix="gm_v0_"))
    gm.memories = [v0.Memory(keywords=", ".join(r.keywords), content=r.content, source=r.source,
                             priority=6 - r.importance, mem_id=r.id)
                   for r in records if r.is_active]  # v0 overwrote or soft-deleted old versions
    gm.save = lambda: None

    def fake_llm(prompt, debug=False):
        text = prompt.split("文本：", 1)[1].split("请以JSON", 1)[0].strip()
        return json.dumps({"keywords": jieba.analyse.extract_tags(text, topK=5)}, ensure_ascii=False)

    gm._call_llm = fake_llm
    return lambda q: [m.id for m in gm.retrieval_relevant_memory(q, top_k=K)]


@lru_cache(maxsize=None)
def embedder(name: str):
    from gamememo.personal.embed import FastEmbedEmbedder
    return FastEmbedEmbedder(name)


@lru_cache(maxsize=None)
def reranker(name: str):
    from gamememo.personal.embed import FastEmbedReranker
    return FastEmbedReranker(name)


def v1_factory(emb: str = None, rr: str = None, **cfg) -> Factory:
    def make(records, now):
        e = embedder(emb) if emb else None
        config = RetrievalConfig.for_embedder(e, top_k=K, **cfg)
        retriever = HybridRetriever(embedder=e, config=config, reranker=reranker(rr) if rr else None)
        return lambda q: [s.record.id for s in retriever.search(q, records, now=now)]
    return make


def memory_factory(**opts) -> Factory:
    """The production read path: PersonalMemory.search (history recall etc.)."""
    def make(records, now):
        from gamememo.personal import PersonalMemory

        mem = PersonalMemory("bench", storage_dir=tempfile.mkdtemp(prefix="gm_ret_"),
                             embedder=embedder(JINA_ZH), clock=lambda: now, **opts)
        mem.store.extend(records)
        return lambda q: [s.record.id for s in mem.search(q, top_k=K)]
    return make


BGE_ZH = "BAAI/bge-small-zh-v1.5"
JINA_ZH = "jinaai/jina-embeddings-v2-base-zh"
BGE_RR = "BAAI/bge-reranker-base"
JINA_RR = "jinaai/jina-reranker-v2-base-multilingual"

SYSTEMS: Dict[str, Factory] = {
    "v0-keyword": v0_factory,
    "v1-lexical": v1_factory(),
    "v1-dense": v1_factory(BGE_ZH, use_lexical=False),
    "v1-hybrid": v1_factory(BGE_ZH),                 # bge-small-zh, P0 thresholds
    "v2-hybrid": v1_factory(JINA_ZH, require_specific=False),  # jina-v2-base-zh, as promoted
    "v2-hybrid+specific": v1_factory(JINA_ZH, require_specific=True, min_dense_alone=0.40),
    "v2-hybrid+rerank": v1_factory(BGE_ZH, rr=BGE_RR, min_rerank=-1.0, rerank_pool=5),
    "memory-default": memory_factory(),
    "memory-history": memory_factory(history_recall=True),
    "memory-subjects": memory_factory(subject_recall=True, interleave_fallback=True),
}
DEFAULT_SYSTEMS = ["v0-keyword", "v1-lexical", "v1-hybrid", "v2-hybrid"]


# ---------------------------------------------------------------- metrics

def run_system(factory: Factory, banks, queries) -> Dict:
    searchers = {pid: factory(recs, now) for pid, (recs, now) in banks.items()
                 if any(q["player"] == pid for q in queries)}
    t0 = time.time()
    rows = []
    for q in queries:
        rows.append({"id": q["id"], "player": q["player"], "query": q["query"], "type": q["type"],
                     "gold": q["gold"], "got": searchers[q["player"]](q["query"])})
    res = score(rows)
    res["latency_ms"] = round((time.time() - t0) * 1000 / max(1, len(rows)), 2)
    return res


def score(rows) -> Dict:
    per_type = defaultdict(list)
    for r in rows:
        if r["gold"]:
            r["success"] = len(set(r["got"][:K]) & set(r["gold"])) / len(r["gold"])
        else:
            r["success"] = 1.0 if not r["got"] else 0.0
        per_type[r["type"]].append(r["success"])
    pos = [r for r in rows if r["gold"]]
    neg = [r for r in rows if not r["gold"]]
    return {
        "MemScore@3": mean(r["success"] for r in rows),
        "Recall@3": mean(r["success"] for r in pos),
        "MRR": mean(next((1.0 / (i + 1) for i, g in enumerate(r["got"]) if g in r["gold"]), 0.0)
                    for r in pos),
        "Abstain": mean(r["success"] for r in neg),
        "by_type": {t: mean(v) for t, v in sorted(per_type.items())},
        "n": len(rows),
        "rows": rows,
    }


def mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def print_table(results: Dict[str, Dict], types: List[str]) -> None:
    head = f"{'system':30s} {'MemScore@3':>10s} {'Recall@3':>9s} {'MRR':>6s} {'Abstain':>8s} {'ms/q':>7s}"
    print(head + "".join(f" {t[:10]:>10s}" for t in types))
    for name, r in results.items():
        line = (f"{name:30s} {r['MemScore@3']:10.3f} {r['Recall@3']:9.3f} {r['MRR']:6.3f} "
                f"{r['Abstain']:8.3f} {r['latency_ms']:7.1f}")
        print(line + "".join(f" {r['by_type'].get(t, 0):10.3f}" for t in types))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    ap.add_argument("--out", default=None, help="write full JSON results here")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(SYSTEMS))
        return {}

    data, banks = load(args.data)
    queries = [q for q in data["queries"] if args.split == "all" or q["split"] == args.split]
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    results = {name: run_system(SYSTEMS[name], banks, queries) for name in names}

    print(f"\ndata={os.path.basename(args.data)}  split={args.split}  queries={len(queries)}  k={K}\n")
    print_table(results, sorted({q["type"] for q in queries}))

    if args.show_errors:
        for name, r in results.items():
            print(f"\n--- misses: {name}")
            for row in r["rows"]:
                if row["success"] < 1:
                    print(f"  {row['id']} [{row['type']}] {row['query']}  gold={row['gold']} got={row['got']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dataset": os.path.relpath(os.path.abspath(args.data), ROOT), "split": args.split,
                       "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "results": results}, f, ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    main()
