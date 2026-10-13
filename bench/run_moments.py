# -*- coding: utf-8 -*-
"""Proactive recall at game moments (e2e_g2): right after a match, does the
companion bring up the thing the player said before that this game is the cue
for, and stay quiet otherwise?

    python -m bench.run_moments --data bench/data/e2e_g2.json --split dev --contexts gamememo,full,rag
    python -m bench.run_moments --data ... --split dev --mem0-build     # Mem0 writes + searches, in time order

Every context gets the same description of the game just played (the ledger's
``situation``: match, rank move, the evening so far) and the same instruction;
they differ in what memory comes with it:

    none          nothing
    full          every chat session before now, verbatim, dated
    full_ground   every exchange before now, each with the game it was said after
    full_log      every chat session and every match record before now, interleaved in time order
                  (all the information is there; nothing is indexed)
    rag           the 8 exchanges before now that best match the game description (hybrid retrieval)
    mem0          the 8 Mem0 memories that best match the game description (Mem0 fed sessions in time order)
    gamememo      the intentions this game is the cue for (GameMemory.moment); nothing due -> stays quiet, no LLM call
    gm_noground   same exchanges, without the game they were said after and the cue note
    gm_nofirst    cue fires on every matching game, not only the first since it was said
    gm_decide     GameMemory's exchanges, but the reply model decides whether to speak (as the others do)

GameMemory decides when to speak (an intention is due) and the reply model
only words it (``COMPOSE``); every other context leaves the decision to the
reply model, with the same game description and instruction (``SYSTEM``).

Scoring: at a positive moment the companion must speak and name the planted
intention (any of the author's answer_any words); at a negative moment it must
stay quiet. ``repeat`` moments (a second game on a hero after its reminder was
due) are reported apart and not in the primary score.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from bench.run_answer import paired_bootstrap
from bench.run_e2e import LLMCache, jina, transcript
from gamememo.game.intentions import IntentionIndex
from gamememo.game.memory import GameMemory
from gamememo.llm import OllamaClient, parse_json
from gamememo.personal.model import parse_time

SYSTEM = """你是一位《王者荣耀》玩家的游戏陪伴助手，平时陪他聊天。他每打完一局，你可以主动跟他说一句话，也可以不说。

现在是{now}。
【刚才这局】
{situation}

{context}

只有当你记得的某件事和刚才这局直接相关、现在提起正合适时（比如他以前说过的话、定下的目标或规矩、你答应过要提醒他的事），才主动跟他说，用一两句话，并且要说出那件事的具体内容。没有这样的事就不说；不要只是普通地夸他或安慰他。"""
# GameMemory decides *when* (an intention is due); the reply model only says it
COMPOSE = """你是一位《王者荣耀》玩家的游戏陪伴助手，平时陪他聊天。他刚打完一局，你要主动跟他说一句话。

现在是{now}。
【刚才这局】
{situation}

{context}

这件事正好和刚才这局对上了。用一两句话主动跟他提起，要说出那件事的具体内容（他当时说的关键内容），语气自然。"""
SCHEMA = {"type": "object", "properties": {"say": {"type": "boolean"}, "message": {"type": "string"}},
          "required": ["say", "message"]}
NOTHING = "关于这位玩家，你没有任何记录。"
PRIMARY_NEG = ("opp_hero", "friend_goal", "near_rank", "short_streak", "random")


def speak(llm, now: str, situation: str, context: str, compose: bool = False) -> Tuple[bool, str]:
    system = (COMPOSE if compose else SYSTEM).format(now=now[:16], situation=situation, context=context)
    raw = llm.chat(system=system, messages=[{"role": "user", "content": "（刚打完一局。要主动说点什么吗？）"}],
                   temperature=0.0, max_tokens=200, json_schema=SCHEMA)
    data = parse_json(raw)
    if not isinstance(data, dict):
        return False, str(raw)
    msg = str(data.get("message", "")).strip()
    said = (compose or bool(data.get("say"))) and bool(msg) and msg not in ("无", "不说", "无。")
    return said, msg if said else ""


def judge(x: Dict, said: bool, msg: str) -> float:
    if not x["gold"]:
        return 0.0 if said else 1.0
    return 1.0 if said and any(k in msg for k in x["answer_any"]) else 0.0


# ---------------------------------------------------------------- one player's memories

class Player:
    def __init__(self, p: Dict, workdir: str, mem0: Optional[Dict] = None):
        self.p = p
        self.by_time = {m["time"]: m for m in p["matches"]}
        self.clock = {"now": datetime(2026, 1, 1)}
        self.gm = GameMemory(p["id"], storage_dir=workdir, embedder=jina(), clock=lambda: self.clock["now"])
        self.gm.ingest_matches(p["matches"])
        self.sessions = sorted(p["sessions"], key=lambda s: s["date"])
        for s in self.sessions:
            self.clock["now"] = parse_time(s["date"])
            self.gm.ingest_chat(transcript(s))
        self.gm._index()
        self.nofirst = IntentionIndex(self.gm.ledger, first_only=False)
        for it in self.gm.intentions.items:
            self.nofirst.items.append(it)
        self.turns = sorted((r for r in self.gm.chat.store.active() if r.kind == "turn"), key=lambda r: r.created_at)
        self.mem0 = mem0 or {}
        self._rag = None

    def gold_records(self, x: Dict) -> set:
        """Record ids of the exchanges that carry the moment's planted intention."""
        texts = [t["content"] for s in self.sessions for t in s["turns"] if t.get("item") in x["gold"]]
        return {r.id for r in self.turns if any(f"玩家: {t}" in r.content for t in texts)}

    def context(self, kind: str, x: Dict) -> Optional[str]:
        m, now = self.by_time[x["match_time"]], x["now"]
        before = [r for r in self.turns if r.created_at < now]
        if kind == "none":
            return NOTHING
        if kind == "full":
            chats = [f"【{s['date'][:16]} 的聊天】\n{transcript(s)}" for s in self.sessions if s["date"] < now]
            return "下面是你和这位玩家过去所有的聊天记录：\n\n" + "\n\n".join(chats) if chats else NOTHING
        if kind == "full_log":
            from bench.gamesim import match_line
            events = [(s["date"][:16], f"【{s['date'][:16]} 的聊天】\n{transcript(s)}") for s in self.sessions if s["date"] < now]
            events += [(g["time"], "对局：" + match_line(g)) for g in self.p["matches"] if g["time"] <= m["time"]]
            return ("下面是这位玩家过去所有的对局记录和你们的聊天，按时间排列：\n" +
                    "\n".join(e for _, e in sorted(events, key=lambda e: e[0])))
        if kind == "full_ground":
            lines = [f"【{self.gm.intentions.grounding(r.created_at[:16])}】" + r.content.replace("\n", " ") for r in before]
            return "下面是你和这位玩家过去所有的聊天（每句前面是说这话时刚打完的那局）：\n" + "\n".join(lines) if lines else NOTHING
        if kind == "rag":
            from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig
            if self._rag is None:
                self._rag = HybridRetriever(jina(), RetrievalConfig.for_embedder(jina()).for_candidates())
            if not before:
                return NOTHING
            hits = sorted((h.record for h in self._rag.search(x["situation"], before, top_k=8, now=parse_time(now))),
                          key=lambda r: r.created_at)
            return "你想起了和这位玩家的这些聊天：\n" + "\n".join(self.gm.chat.describe(r) for r in hits)
        if kind == "mem0":
            res = self.mem0.get(x["id"])
            if res is None:
                raise SystemExit(f"no Mem0 search for {x['id']}: run --mem0-build first")
            return "你记得关于这位玩家的这些事：\n" + "\n".join(f"- {r}" for r in res) if res else NOTHING
        if kind in ("gamememo", "gm_noground", "gm_nofirst", "gm_decide"):
            index = self.nofirst if kind == "gm_nofirst" else self.gm.intentions
            due = index.due(m)
            if not due:
                return None
            ground = kind != "gm_noground"
            lines = list(dict.fromkeys(index.describe(it, ground=ground) for it in due))
            return "【他以前说过的、和刚才这局有关的话】\n" + "\n".join(lines)
        raise SystemExit(f"unknown context {kind}")

    def due_records(self, x: Dict) -> set:
        return {it.record_id for it in self.gm.intentions.due(self.by_time[x["match_time"]])}


# ---------------------------------------------------------------- Mem0, fed in time order

def mem0_build(players: List[Dict], base: str, out_path: str) -> None:
    from bench.locomo import mem0_config
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    from mem0 import Memory
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for p in players:
        if all(x["id"] in out for x in p["moments"]):
            continue
        path = os.path.join(base, p["id"])
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        mem = Memory.from_config(mem0_config(path, "g2"))
        events = [(s["date"], "s", s) for s in p["sessions"]] + [(x["now"], "m", x) for x in p["moments"]]
        for when, kind, obj in sorted(events, key=lambda e: (e[0], e[1])):
            t0 = time.time()
            if kind == "s":
                msgs = [{"role": "user", "content": f"（这段聊天发生在{when[:16]}。）"}]
                msgs += [{"role": t["role"], "content": t["content"]} for t in obj["turns"]]
                mem.add(msgs, user_id=p["id"], metadata={"date": when[:16]})
                print(p["id"], "add", when, round(time.time() - t0), "s", flush=True)
            else:
                res = mem.search(obj["situation"], filters={"user_id": p["id"]}, top_k=8).get("results", [])
                out[obj["id"]] = [f"{r['memory']}（{(r.get('metadata') or {}).get('date', '')[:10]}）" for r in res]
                json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- scoring

def summarize(rows: List[Dict]) -> Dict:
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    pos = [r["success"] for r in rows if r["gold"]]
    neg = [r["success"] for r in rows if not r["gold"] and r["category"] in PRIMARY_NEG]
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r["success"])
    primary = [r for r in rows if r["gold"] or r["category"] in PRIMARY_NEG]
    named = [r["success"] for r in rows if r["gold"] and r.get("named") is True]
    deictic = [r["success"] for r in rows if r["gold"] and r.get("named") is False]
    return {"score": avg([r["success"] for r in primary]), "n": len(primary),
            "recall": avg(pos), "quiet": avg(neg), "repeat_quiet": avg(cats.get("repeat", [])),
            "hero_named": avg(named), "hero_deictic": avg(deictic),
            "by_category": {k: avg(v) for k, v in sorted(cats.items())},
            "llm_calls": sum(r["llm"] for r in rows),
            "context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--contexts", default="none,full,full_ground,rag,gamememo,gm_noground,gm_nofirst")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--llm-cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mem0-build", action="store_true")
    ap.add_argument("--mem0", default=None, help="Mem0 search results (json), from --mem0-build")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--sample", type=int, default=0,
                    help="per player, only this many positive and this many primary-negative moments (seeded)")
    args = ap.parse_args(argv)
    data = json.load(open(args.data, encoding="utf-8"))
    players = [p for p in data["players"] if args.split == "all" or p["split"] == args.split]
    mem0_path = args.mem0 or f"bench/results/mem0_g2_{args.split}.json"
    if args.mem0_build:
        mem0_build(players, "bench/.mem0_g2", mem0_path)
        return
    mem0 = json.load(open(mem0_path)) if os.path.exists(mem0_path) else {}
    llm = OllamaClient(model=args.model, timeout=1800, seed=0, num_ctx=args.num_ctx)
    if args.llm_cache:
        LLMCache(args.llm_cache).wrap(llm)
    if args.sample:
        import random
        for p in players:
            rng = random.Random(p["id"])
            pos = [x for x in p["moments"] if x["gold"]]
            neg = [x for x in p["moments"] if not x["gold"] and x["category"] in PRIMARY_NEG]
            keep = {x["id"] for x in rng.sample(pos, args.sample) + rng.sample(neg, args.sample)}
            p["moments"] = [x for x in p["moments"] if x["id"] in keep]
    built = [Player(p, tempfile.mkdtemp(prefix="g2_"), mem0) for p in players]
    items = {(pl.p["id"], it["id"]): it for pl in built for it in pl.p["items"]}
    out = json.load(open(args.out)) if args.out and os.path.exists(args.out) else {}
    out = out.get("results", {})
    for kind in [c.strip() for c in args.contexts.split(",") if c.strip()]:
        rows, t0 = [], time.time()
        for pl in built:
            for x in pl.p["moments"]:
                ctx = pl.context(kind, x)
                if ctx is None:
                    said, msg, calls = False, "", 0
                else:
                    said, msg = speak(llm, x["now"], x["situation"], ctx,
                                      compose=kind in ("gamememo", "gm_noground", "gm_nofirst"))
                    calls = 1
                it = items.get((pl.p["id"], x["item"])) if x["item"] else None
                rows.append({"id": x["id"], "player": pl.p["id"], "category": x["category"], "gold": x["gold"],
                             "named": it.get("named") if it and x["gold"] and it["kind"].startswith("hero") else None,
                             "said": said, "message": msg, "success": judge(x, said, msg), "llm": calls,
                             "context_chars": len(ctx or "")})
        out[kind] = summarize(rows) | {"rows": rows, "minutes": round((time.time() - t0) / 60, 1)}
        s = out[kind]
        print(f"[{kind}] score={s['score']} recall={s['recall']} quiet={s['quiet']} repeat={s['repeat_quiet']} "
              f"named={s['hero_named']} deictic={s['hero_deictic']} calls={s['llm_calls']} chars={s['context_chars']}",
              flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump({"data": args.data, "split": args.split, "model": args.model, "results": out},
                      open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # the index alone: is the planted exchange among what is due, and nothing due at negatives?
    hit = [bool(pl.gold_records(x) & pl.due_records(x)) for pl in built for x in pl.p["moments"] if x["gold"]]
    quiet = [not pl.due_records(x) for pl in built for x in pl.p["moments"]
             if not x["gold"] and x["category"] in PRIMARY_NEG]
    print(f"\nindex alone: recall {sum(hit)}/{len(hit)}, quiet {sum(quiet)}/{len(quiet)}")
    names = list(out)
    print()
    for k in names:
        print(f"{k:12s} {out[k]['score']:.3f}  {out[k]['by_category']}")
    if "gamememo" in out:
        prim = lambda rows: [r for r in rows if r["gold"] or r["category"] in PRIMARY_NEG]  # noqa: E731
        for k in names:
            if k != "gamememo":
                print(f"gamememo - {k}: {paired_bootstrap(prim(out[k]['rows']), prim(out['gamememo']['rows']))}")
    if args.show:
        for k in names:
            print(f"\n--- misses: {k}")
            for r in out[k]["rows"]:
                if not r["success"]:
                    print(f"  {r['id']} [{r['category']}] said={r['said']} {r['message'][:90]}")
    return out


if __name__ == "__main__":
    main()
