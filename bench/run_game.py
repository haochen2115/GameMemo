# -*- coding: utf-8 -*-
"""Answer-level evaluation on the game-memory benchmark (e2e_g1).

    python -m bench.run_game --data bench/data/e2e_g1.json --split dev --contexts gamememo,chat,rag,full

Contexts (what the reply model sees):
    none        nothing
    chat        chat memory only (RawMemory / P8), no match data: a chat-only companion
    rag         one retrieval index over chat exchanges AND match records (a general memory
                system given the match log as more text), top-10, dated, in time order
    full        every chat and every match record in the prompt
    card        the ledger's game card only (no chat)
    gamememo    GameMemory: game card + named-hero records + chat of event-anchored
                days + verbatim chat recall + pending promises

Judging (answer_type): rank (first rank named must equal the key's tier and
sub-level), percent (a "%" number within ±3), count (a number within ±5%, at
least ±2), count_exact, name (hero / position, aliases allowed), keyword
(any answer_any), negative (must decline). Update-style stale values count
when a stale rank is the first rank named.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from bench.run_answer import _ABSTAIN, answer, paired_bootstrap
from bench.run_e2e import LLMCache, jina, transcript
from bench.gamesim import match_line
from gamememo.game.memory import GameMemory
from gamememo.game.ontology import POSITION_ALIASES, parse_ranks
from gamememo.llm import OllamaClient
from gamememo.personal.model import MemoryRecord, parse_time

_DATEISH = re.compile(r"\d{4}[-年./]\d{1,2}[-月./]\d{1,2}日?|\d{1,2}月\d{1,2}[日号]?|\d{1,2}:\d{2}")
_CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _numbers(text: str) -> List[float]:
    text = _DATEISH.sub(" ", text)
    out = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    for m in re.finditer(r"([一二两三四五六七八九十]{1,3})\s*(?:把|场|局|次)", text):
        s = m.group(1)
        if s == "十":
            out.append(10)
        elif s.startswith("十"):
            out.append(10 + _CN[s[1]])
        elif len(s) == 2 and s[1] == "十":
            out.append(_CN[s[0]] * 10)
        elif len(s) == 3 and s[1] == "十":
            out.append(_CN[s[0]] * 10 + _CN[s[2]])
        elif len(s) == 1:
            out.append(_CN[s])
    return out


def judge(q: Dict, known: bool, ans: str) -> float:
    declined = (not known) or not ans.strip() or bool(_ABSTAIN.search(ans) and len(ans) < 40)
    if q["type"] == "negative":
        return 1.0 if declined else 0.0
    if not known:
        return 0.0
    t, key = q["answer_type"], q["key"]
    if t == "rank":
        named = parse_ranks(ans)
        want = parse_ranks(key)[0]
        if not named:
            return 0.0
        first = named[0]
        return 1.0 if first[0] == want[0] and (want[1] is None or first[1] == want[1]) else 0.0
    if t == "percent":
        pct = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", ans)]
        pct += [float(x) for x in re.findall(r"百分之(\d+)", ans)]
        return 1.0 if any(abs(p - key) <= 3 for p in pct) else 0.0
    if t == "count":
        tol = max(2, 0.05 * key)
        return 1.0 if any(abs(n - key) <= tol for n in _numbers(ans)) else 0.0
    if t == "count_exact":
        return 1.0 if key in _numbers(ans) else 0.0
    if t == "name":
        names = POSITION_ALIASES.get(key, (key,))
        return 1.0 if any(n in ans for n in names) else 0.0
    return 1.0 if any(k in ans for k in key) else 0.0


# ---------------------------------------------------------------- contexts

class Player:
    def __init__(self, p: Dict, workdir: str):
        self.p = p
        self.now = parse_time(p["ask_at"])
        self.clock = {"now": self.now}
        self.mem = GameMemory(p["id"], storage_dir=workdir, embedder=jina(), clock=lambda: self.clock["now"])
        for s in sorted(p["sessions"], key=lambda s: s["date"]):
            self.clock["now"] = parse_time(s["date"])
            self.mem.ingest_chat(transcript(s))
        self.clock["now"] = self.now
        self.mem.ingest_matches(p["matches"])
        self._rag = None

    def context(self, kind: str, query: str) -> str:
        if kind == "none":
            return "关于这位玩家，你没有任何记录。"
        if kind == "gamememo":
            return self.mem.context(query)
        if kind == "card":
            return self.mem.ledger.card(self.now) or "关于这位玩家，你没有任何记录。"
        if kind == "chat":
            rec = self.mem.chat.retrieve(query, top_k=8, touch=False)
            lines = [self.mem.chat.describe(r) for r in rec] + [self.mem.chat.describe(r) for r in self.mem.chat.pending_promises()]
            return "关于这位玩家，你想起了这些聊天：\n" + "\n".join(lines) if lines else "关于这位玩家，你没有任何记录。"
        if kind == "full":
            chats = [f"【{s['date'][:16]} 的聊天】\n{transcript(s)}" for s in sorted(self.p["sessions"], key=lambda s: s["date"])]
            games = [match_line(m) for m in self.p["matches"]]
            return ("下面是你和这位玩家过去所有的聊天记录：\n\n" + "\n\n".join(chats) +
                    "\n\n下面是系统记录的这位玩家全部对局（时间 模式 英雄（位置）胜负 击杀/死亡/助攻 段位变化）：\n" + "\n".join(games))
        if kind == "rag":
            from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig
            if self._rag is None:
                docs = [r for r in self.mem.chat.store.active() if r.kind == "turn"]
                docs += [MemoryRecord(content="系统对局记录：" + match_line(m), kind="turn", created_at=m["time"])
                         for m in self.p["matches"]]
                self._rag = (HybridRetriever(jina(), RetrievalConfig.for_embedder(jina()).for_candidates()), docs)
            r, docs = self._rag
            hits = sorted((h.record for h in r.search(query, docs, top_k=10, now=self.now)), key=lambda x: x.created_at)
            return "下面是和这个问题相关的聊天和对局记录：\n" + "\n".join(f"【{h.created_at[:16]}】{h.content}" for h in hits)
        raise SystemExit(f"unknown context {kind}")


def summarize(rows: List[Dict]) -> Dict:
    by = defaultdict(list)
    for r in rows:
        by[r["subtype"]].append(r["success"])
    group = defaultdict(list)
    for r in rows:
        group[r["type"]].append(r["success"])
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"score": avg([r["success"] for r in rows]), "n": len(rows),
            "by_group": {k: avg(v) for k, v in sorted(group.items())},
            "by_subtype": {k: avg(v) for k, v in sorted(by.items())},
            "context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--contexts", default="none,chat,card,gamememo,rag,full")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--llm-cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)
    data = json.load(open(args.data, encoding="utf-8"))
    players = [p for p in data["players"] if args.split == "all" or p["split"] == args.split]
    llm = OllamaClient(model=args.model, timeout=1800, seed=0, num_ctx=args.num_ctx)
    if args.llm_cache:
        LLMCache(args.llm_cache).wrap(llm)
    built = [Player(p, tempfile.mkdtemp(prefix="g1_")) for p in players]
    out = {}
    for kind in [c.strip() for c in args.contexts.split(",") if c.strip()]:
        rows, t0 = [], time.time()
        for pl in built:
            for q in pl.p["questions"]:
                ctx = pl.context(kind, q["query"])
                known, ans = answer(llm, ctx, pl.now, q["query"])
                rows.append({"id": q["id"], "player": pl.p["id"], "type": q["type"], "subtype": q["subtype"],
                             "query": q["query"], "key": q["key"], "known": known, "answer": ans,
                             "success": judge(q, known, ans), "context_chars": len(ctx)})
        out[kind] = summarize(rows) | {"rows": rows, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"[{kind}] {out[kind]['score']} {out[kind]['by_group']} chars={out[kind]['context_chars']}", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump({"data": args.data, "split": args.split, "model": args.model, "results": out},
                      open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    names = list(out)
    print()
    for k in names:
        print(f"{k:10s} {out[k]['score']:.3f}  {out[k]['by_subtype']}")
    if "gamememo" in out:
        for k in names:
            if k != "gamememo":
                print(f"gamememo - {k}: {paired_bootstrap(out[k]['rows'], out['gamememo']['rows'])}")
    if args.show_errors:
        for k in names:
            print(f"\n--- misses: {k}")
            for r in out[k]["rows"]:
                if not r["success"]:
                    print(f"  {r['id']} [{r['subtype']}] {r['query']} key={r['key']} got={r['answer'][:80]}")
    return out


if __name__ == "__main__":
    main()
