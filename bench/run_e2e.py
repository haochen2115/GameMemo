# -*- coding: utf-8 -*-
"""End-to-end memory benchmark: real LLM writes, then retrieval is judged.

    ollama serve && ollama pull qwen2.5:3b
    python -m bench.run_e2e --split dev --systems v0-pipeline,v2
    python -m bench.run_e2e --split test --model qwen3:4b --out bench/results/e2e_v1_test.json

For each player, sessions are ingested in date order (the system's clock
is set to each session's date), then every question is asked at
``ask_at``. The evidence is the top-3 retrieved memories (content plus
event date). A question is answered when the evidence contains any
``answer_any`` string. Update questions additionally fail if the
evidence still contains an outdated value (``stale_any``). Negative
questions (never mentioned) must retrieve nothing.

This measures the whole write + read path. No LLM judge is involved, so
the scores are deterministic given what the memory system stored.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gamememo.llm import OllamaClient  # noqa: E402
from gamememo.personal.model import parse_time  # noqa: E402

DATA = os.path.join(ROOT, "bench", "data", "e2e_v1.json")
K = 3
NAMES = {"user": "玩家", "assistant": "助手"}

# A system is built per player and exposes ingest(transcript, when) and
# evidence(query, when) -> list of evidence strings, plus dump() for audit.


def transcript(session) -> str:
    return "\n".join(f"{NAMES[t['role']]}: {t['content']}" for t in session["turns"])


class V0System:
    """The original game_memory.py pipeline with a real LLM."""

    def __init__(self, model: str, base_url: str, workdir: str):
        import game_memory as v0
        from llm_client import OllamaClient as V0Client

        self.gm = v0.GameMemory(user_id="e2e", storage_dir=workdir,
                                llm_client=V0Client(model=model, base_url=base_url, timeout=600))

    def ingest(self, text: str, when: datetime) -> None:
        self.gm.update_personal_memory_with_messages(text)  # v0 has no notion of time

    def evidence(self, query: str, when: datetime) -> List[str]:
        return [m.content for m in self.gm.retrieval_relevant_memory(query, top_k=K)]

    def dump(self) -> List[Dict]:
        return [m.to_dict() for m in self.gm.memories]


class V2System:
    def __init__(self, model: str, base_url: str, workdir: str, embedder=None, seed: int = 0, **opts):
        from gamememo.personal import PersonalMemory

        self.now = datetime.now()
        self.mem = PersonalMemory("e2e", llm=OllamaClient(model=model, base_url=base_url, timeout=600,
                                                          seed=seed, think=False if "qwen3" in model else None),
                                  storage_dir=workdir, embedder=embedder, clock=lambda: self.now, **opts)

    def ingest(self, text: str, when: datetime) -> None:
        self.now = when
        self.mem.ingest(text, source="chat")

    def evidence(self, query: str, when: datetime) -> List[str]:
        self.now = when
        describe = getattr(self.mem, "describe", None)  # absent on older mains
        return [describe(r) if describe else r.content + (f"（{r.event_time}）" if r.event_time else "")
                for r in self.mem.retrieve(query, top_k=K, touch=False)]

    def dump(self) -> List[Dict]:
        return [r.to_dict() for r in self.mem.store.all()]


_JINA = []


def jina():
    from gamememo.personal.embed import FastEmbedEmbedder
    if not _JINA:
        _JINA.append(FastEmbedEmbedder())
    return _JINA[0]


def _nospec():
    from gamememo.personal.retrieval import RetrievalConfig
    return RetrievalConfig.for_embedder(jina(), require_specific=False)


SYSTEMS: Dict[str, Callable[..., object]] = {
    "v0-pipeline": lambda model, url, wd: V0System(model, url, wd),
    "v2-lexical": lambda model, url, wd: V2System(model, url, wd),
    "v2": lambda model, url, wd: V2System(model, url, wd, embedder=jina()),
    "v2+player-only": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), player_only=True),
    "v2+po+history": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), player_only=True,
                                                      history_recall=True),
    "v2+po+history+turn": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), player_only=True,
                                                           history_recall=True, per_turn=True),
    "v2+po+history-nospec": lambda model, url, wd: V2System(
        model, url, wd, embedder=jina(), player_only=True, history_recall=True,
        retrieval_config=_nospec()),
    "v2+history": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), history_recall=True),
    "v2-slots": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), write_mode="slots"),
    "v2-slots+history": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), write_mode="slots",
                                                         history_recall=True),
    "v2-slots+po+history": lambda model, url, wd: V2System(model, url, wd, embedder=jina(), write_mode="slots",
                                                            player_only=True, history_recall=True),
}


_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def normalize_dates(text: str) -> str:
    """Append a Chinese spelling of every ISO date so "2026-06-28" also
    matches an answer key written as "6月28" (grader fix found on dev)."""
    extra = [f"{int(y)}年{int(m)}月{int(d)}日 {int(m)}月{int(d)}日"
             for y, m, d in _ISO.findall(text)]
    return text + ("\n" + " ".join(extra) if extra else "")


def judge(q: Dict, evidence: List[str]) -> Tuple[float, bool]:
    text = normalize_dates("\n".join(evidence))
    if q["type"] == "negative":
        return (1.0 if not evidence else 0.0), False
    if q.get("answer_all"):  # trajectories: every state must be recalled
        hit = all(a in text for a in q["answer_all"])
    else:
        hit = any(a in text for a in q["answer_any"])
    stale = any(s in text for s in q.get("stale_any", []))
    ok = hit and not (q["type"] == "update" and stale)
    return (1.0 if ok else 0.0), stale


def run_player(system, player, errors: List[str]) -> List[Dict]:
    for s in player["sessions"]:
        try:
            system.ingest(transcript(s), parse_time(s["date"]))
        except Exception as e:  # a failed write loses that session, as it would in production
            errors.append(f"{player['id']} {s['date']}: {e}")
    ask = parse_time(player["ask_at"])
    rows = []
    for q in player["questions"]:
        ev = system.evidence(q["query"], ask)
        success, stale = judge(q, ev)
        rows.append({"id": q["id"], "player": player["id"], "type": q["type"], "query": q["query"],
                     "answer_any": q["answer_any"] or q.get("answer_all", []), "evidence": ev, "success": success, "stale": stale})
    return rows


def summarize(rows) -> Dict:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["success"])
    upd = [r for r in rows if r["type"] == "update"]
    neg = [r for r in rows if r["type"] == "negative"]
    pos = [r for r in rows if r["type"] != "negative"]
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    return {
        "E2EScore": avg([r["success"] for r in rows]),
        "Answer@3": avg([r["success"] for r in pos]),
        "StaleRate": avg([1.0 if r["stale"] else 0.0 for r in upd]),
        "Abstain": avg([r["success"] for r in neg]),
        "by_type": {t: avg(v) for t, v in sorted(by_type.items())},
        "n": len(rows),
    }


def print_results(results: Dict, header: str, show_errors: bool) -> None:
    types = sorted({r["type"] for res in results.values() for r in res["rows"]})
    print(f"\n{header}\n")
    print(f"{'system':22s} {'E2EScore':>8s} {'Answer@3':>8s} {'Stale':>6s} {'Abstain':>7s}"
          + "".join(f" {t[:9]:>9s}" for t in types))
    for name, r in results.items():
        print(f"{name:22s} {r['E2EScore']:8.3f} {r['Answer@3']:8.3f} {r['StaleRate']:6.3f} "
              f"{r['Abstain']:7.3f}" + "".join(f" {r['by_type'].get(t, 0):9.3f}" for t in types))
    if show_errors:
        for name, r in results.items():
            print(f"\n--- misses: {name}")
            for row in r["rows"]:
                if row["success"] < 1:
                    print(f"  {row['id']} [{row['type']}] {row['query']} want={row['answer_any']} "
                          f"{'STALE ' if row['stale'] else ''}got={row['evidence']}")


def rejudge(path: str, data_path: str, show_errors: bool) -> Dict:
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    with open(data_path, encoding="utf-8") as f:
        qs = {q["id"]: q for p in json.load(f)["players"] for q in p["questions"]}
    results = {}
    for name, res in saved["results"].items():
        rows = []
        for row in res["rows"]:
            q = qs[row["id"]]
            success, stale = judge(q, row["evidence"])
            rows.append(dict(row, answer_any=q["answer_any"] or q.get("answer_all", []),
                             success=success, stale=stale))
        results[name] = dict(summarize(rows), rows=rows)
    print_results(results, f"rejudged {os.path.basename(path)}  model={saved.get('model')}", show_errors)
    return results


def save(args, results) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": os.path.relpath(os.path.abspath(args.data), ROOT), "split": args.split,
                   "model": args.model, "seeds": args.seeds, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, f, ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--systems", default="v0-pipeline,v2")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--seeds", default="0", help="comma-separated LLM seeds for v2 systems; "
                    "rows of all seeds are pooled, so metrics are seed-averaged")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", default=None, help="directory to keep each run's memory files")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--rejudge", default=None,
                    help="re-score a saved results file with the current judge and answer keys (no LLM)")
    args = ap.parse_args(argv)

    if args.rejudge:
        return rejudge(args.rejudge, args.data, args.show_errors)

    if not OllamaClient(base_url=args.base_url).is_available():
        sys.exit(f"Ollama is not reachable at {args.base_url}")
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    players = [p for p in data["players"] if args.split == "all" or p["split"] == args.split]

    seeds = [int(x) for x in args.seeds.split(",")]
    results = {}
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        rows, dumps, errors, t0 = [], {}, [], time.time()
        for seed in seeds:
            for p in players:
                wd = tempfile.mkdtemp(prefix=f"e2e_{name}_")
                system = SYSTEMS[name](args.model, args.base_url, wd)
                if isinstance(system, V2System):
                    system.mem.llm.seed = seed
                rows.extend(dict(r, seed=seed) for r in run_player(system, p, errors))
                dumps[f"{p['id']}@{seed}"] = system.dump()
                if args.keep:
                    shutil.copytree(wd, os.path.join(args.keep, name, f"{p['id']}@{seed}"), dirs_exist_ok=True)
                shutil.rmtree(wd, ignore_errors=True)
            if not isinstance(system, V2System):
                break  # v0 has no seed control; one run
        res = summarize(rows)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        res["ingest_errors"] = errors
        res["rows"], res["memories"] = rows, dumps
        results[name] = res
        print(f"[{name}] done in {res['minutes']} min, {len(errors)} failed sessions", flush=True)
        if args.out:  # save after every system so a crash never loses finished runs
            save(args, results)

    print_results(results, f"model={args.model}  split={args.split}  "
                           f"questions={results[next(iter(results))]['n']}", args.show_errors)

    return results


if __name__ == "__main__":
    main()
