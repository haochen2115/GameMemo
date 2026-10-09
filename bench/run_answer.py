# -*- coding: utf-8 -*-
"""Answer-level benchmark: an LLM answers every e2e question from a given
context, and the *answer* is judged.

``run_e2e`` judges the top-3 retrieved memories, which only works for
memory systems. Here every system is reduced to the context it would put
in front of the reply model, so a memory system can be compared with the
trivial alternative of putting the whole conversation history in the
prompt:

    none              the question alone (lower bound; abstains on everything)
    full              every past session verbatim, with dates
    evidence:<sys>    the top-3 memories <sys> retrieved in a run_e2e results file
    chatbot:<sys>     what MemoryChatBot injects: core profile + pending
                      promises + top-3, replayed on <sys>'s stored memories
    allmem:<sys>      every stored memory of <sys> (active and superseded, described
                      with dates and staleness) in date order: no retrieval at all
    recall:<sys>      the top-3 memories <sys> retrieves, replayed on its stored memories
    rag:<k>[:<ck>]    no memory writing: the top-k raw exchanges (player turn +
                      assistant reply) by the same hybrid retriever, in date order;
                      questions about change over time get the top-<ck> instead

    python -m bench.run_answer --data bench/data/e2e_v10.json --split test \\
        --contexts none,full,evidence:p6b --results bench/results/e2e_v10_test.json \\
        --seeds 0,1,2 --llm-cache bench/.cache/answer.sqlite --out bench/results/answer_v10_test.json

The answerer returns {"known": bool, "answer": str}. Judging is
deterministic: negatives must say "not known"; trajectories must name every
state; update answers fail when they give an outdated value without
marking it as past. Memory contexts pair storage seed s with answer seed s.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.run_e2e import LLMCache, SYSTEMS, normalize_dates, transcript  # noqa: E402
from gamememo.llm import OllamaClient, parse_json  # noqa: E402
from gamememo.personal.model import parse_time  # noqa: E402

WEEKDAYS = "一二三四五六日"

SYSTEM = """你是一个游戏陪伴助手，正在和一位相处了很久的玩家聊天。今天是{today}（星期{weekday}）。

{context}

请只根据上面的内容回答玩家的问题，不要编造。
- 如果上面的内容里找不到答案，known 填 false，answer 简单说明不记得。
- 能回答时 known 填 true，answer 要包含具体的名称、地点、数值或日期。
- 问"现在"的情况时，回答最新的状态；问"怎么变的""换过哪些"时，按时间顺序列出每一个状态。
- 问"你答应过 / 说过要……"时，只回答助手确实说过的事；助手没说过就 known 填 false。
只输出 JSON：{{"known": true 或 false, "answer": "……"}}"""

SCHEMA = {"type": "object", "properties": {"known": {"type": "boolean"}, "answer": {"type": "string"}},
          "required": ["known", "answer"]}

# Rank tiers are written 黄金三 / 黄金3 / 黄金III; answer keys use Chinese numerals.
_TIER = re.compile(r"(青铜|白银|黄金|铂金|钻石|星耀)\s*(IV|V|I{1,3}|[1-5])(?![0-9])")
_NUM = {"1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "I": "一", "II": "二", "III": "三", "IV": "四", "V": "五"}
# An outdated value is fine in an answer when it is marked as past ("以前打乒乓，现在打羽毛球").
_PAST = ("以前", "之前", "原来", "原先", "曾", "过去", "从前", "后来", "换成", "改成", "变成", "换到", "搬到",
         "升到", "→", "->", "之后", "不再", "不打", "不玩", "不住")
_ABSTAIN = re.compile(r"不记得|不知道|没有提到|没提到|没提过|没有提过|没说过|没有说过|没有答应|没答应|没有记录|找不到|无法确定")


# "how did it change" questions need every state, not the best few mentions
CHANGE_INTENT = re.compile(r"怎么(变|换|升|降|走|起伏|上来)|变化|换过|一路|一步步|历程|前后")


def normalize_answer(text: str) -> str:
    text = _TIER.sub(lambda m: m.group(1) + _NUM[m.group(2)], text)
    return normalize_dates(text)


def judge_answer(q: Dict, known: bool, answer: str) -> Tuple[float, bool]:
    """(success, stale) for one answer."""
    text = normalize_answer(answer)
    abstained = (not known) or (not answer.strip()) or bool(_ABSTAIN.search(answer) and len(answer) < 40)
    stale = any(s in text for s in q.get("stale_any", []))
    if q["type"] == "negative":
        return (1.0 if abstained else 0.0), False
    if not known:
        return 0.0, False
    if q.get("answer_all"):
        hit = all(a in text for a in q["answer_all"])
    else:
        hit = any(a in text for a in q["answer_any"])
    bad_stale = q["type"] == "update" and stale and not any(p in answer for p in _PAST)
    return (1.0 if hit and not bad_stale else 0.0), stale


# ---------------------------------------------------------------- contexts

def full_context(player) -> str:
    parts = []
    for s in player["sessions"]:
        parts.append(f"【{s['date'][:16]} 的聊天】\n{transcript(s)}")
    return "下面是你和这位玩家过去所有的聊天记录（按时间顺序）：\n\n" + "\n\n".join(parts)


def memory_context(lines: List[str]) -> str:
    if not lines:
        return "关于这位玩家，你没有想起任何相关的记忆。"
    return "关于这位玩家，你想起了这些记忆：\n" + "\n".join(f"- {x}" for x in lines)


class ContextProvider:
    def __init__(self, spec: str, results: Optional[Dict], players: Dict[str, Dict]):
        self.spec = spec
        self.kind, _, self.system = spec.partition(":")
        self.players = players
        self._evidence: Dict[Tuple[str, int], List[str]] = {}
        self._mem: Dict[Tuple[str, int], object] = {}
        if self.kind in ("evidence", "chatbot", "recall", "allmem"):
            if results is None or self.system not in results["results"]:
                raise SystemExit(f"{spec}: needs --results containing system {self.system!r}")
            res = results["results"][self.system]
            for row in res["rows"]:
                self._evidence[(row["id"], row.get("seed", 0))] = row["evidence"]
            self._memories = res["memories"]
        elif self.kind == "rag":
            k, _, ck = (self.system or "5").partition(":")
            self.k, self.change_k = int(k), int(ck or k)
            self._rag: Dict[str, Tuple[object, List]] = {}
        elif self.kind not in ("none", "full"):
            raise SystemExit(f"unknown context {spec}")

    def _replayed(self, pid: str, seed: int):
        """The stored memories of pid@seed loaded into the system's read path."""
        key = (pid, seed)
        if key not in self._mem:
            self._mem.clear()  # one player at a time: long histories hold many embedded records
            from gamememo.personal.model import MemoryRecord
            p = self.players[pid]
            wd = tempfile.mkdtemp(prefix="answer_")
            sysobj = SYSTEMS[self.system]("unused", "http://localhost:0", wd)
            sysobj.now = parse_time(p["ask_at"])
            sysobj.mem.store.extend(MemoryRecord.from_dict(m) for m in self._memories[f"{pid}@{seed}"])
            self._mem[key] = sysobj
        return self._mem[key]

    def _rag_context(self, player, q) -> str:
        from bench.run_e2e import jina
        from gamememo.personal.model import MemoryRecord
        from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig
        if player["id"] not in self._rag:
            recs = []
            for s in player["sessions"]:
                turns = s["turns"]
                for i, t in enumerate(turns):
                    if t["role"] != "user":
                        continue
                    text = f"玩家: {t['content']}"
                    if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                        text += f"\n助手: {turns[i + 1]['content']}"
                    recs.append(MemoryRecord(content=text, created_at=s["date"], updated_at=s["date"]))
            self._rag[player["id"]] = (HybridRetriever(jina(), RetrievalConfig.for_embedder(jina())), recs)
        retriever, recs = self._rag[player["id"]]
        k = self.change_k if CHANGE_INTENT.search(q["query"]) else self.k
        hits = [h.record for h in retriever.search(q["query"], recs, top_k=k, now=parse_time(player["ask_at"]))]
        if not hits:
            return "你在过去的聊天记录里没有找到相关的内容。"
        hits.sort(key=lambda r: r.created_at)
        return "下面是过去聊天记录里和这个问题相关的片段：\n\n" + "\n\n".join(
            f"【{r.created_at[:16]}】\n{r.content}" for r in hits)

    def __call__(self, player, q, seed: int) -> str:
        if self.kind == "none":
            return "关于这位玩家，你没有任何记录。"
        if self.kind == "full":
            return full_context(player)
        if self.kind == "evidence":
            return memory_context(self._evidence[(q["id"], seed)])
        if self.kind == "rag":
            return self._rag_context(player, q)
        sysobj = self._replayed(player["id"], seed)
        mem = sysobj.mem
        if self.kind == "allmem":
            recs = sorted(mem.store.all(), key=lambda r: (r.event_time or r.created_at or "", r.created_at or ""))
            lines = [(f"你答应过玩家：{r.content}（{r.created_at[:10]}）" if r.kind == "promise" else mem.describe(r))
                     for r in recs]
            return memory_context(lines)
        if self.kind == "recall":
            return memory_context([mem.describe(r) for r in mem.retrieve(q["query"], top_k=3, touch=False)])
        profile, promises = mem.core_profile(), mem.pending_promises()
        shown = {r.id for r in profile + promises}
        retrieved = [r for r in mem.retrieve(q["query"], top_k=3, touch=False) if r.id not in shown]
        lines = [mem.describe(r) for r in profile + retrieved]
        lines += [f"你答应过玩家：{r.content}（{r.created_at[:10]}）" for r in promises]
        return memory_context(lines)


# ---------------------------------------------------------------- running

def summarize(rows) -> Dict:
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["success"])
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0  # noqa: E731
    upd = [r for r in rows if r["type"] == "update"]
    neg = [r for r in rows if r["type"] == "negative"]
    pos = [r for r in rows if r["type"] != "negative"]
    return {
        "AnswerScore": avg([r["success"] for r in rows]),
        "Answered": avg([r["success"] for r in pos]),
        "Abstain": avg([r["success"] for r in neg]),
        "StaleMention": avg([1.0 if r["stale"] else 0.0 for r in upd]),
        "by_type": {t: avg(v) for t, v in sorted(by_type.items())},
        "context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows))),
        "n": len(rows),
    }


def paired_bootstrap(a_rows, b_rows, n=10000, seed=0) -> Dict:
    """b - a, resampling questions (all seeds of a question move together)."""
    a = defaultdict(list)
    b = defaultdict(list)
    for r in a_rows:
        a[r["id"]].append(r["success"])
    for r in b_rows:
        b[r["id"]].append(r["success"])
    ids = sorted(set(a) & set(b))
    d = [sum(b[i]) / len(b[i]) - sum(a[i]) / len(a[i]) for i in ids]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {"delta": round(sum(d) / len(d), 4), "ci95": [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)],
            "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "questions": len(ids)}


def rejudge(path: str, data_path: str) -> Dict[str, Dict]:
    """Re-score saved answers with the current judge and answer keys (no LLM)."""
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    with open(data_path, encoding="utf-8") as f:
        qs = {q["id"]: q for p in json.load(f)["players"] for q in p["questions"]}
    out = {}
    for name, res in saved["results"].items():
        rows = []
        for row in res["rows"]:
            success, stale = judge_answer(qs[row["id"]], row["known"], row["answer"])
            rows.append(dict(row, success=success, stale=stale))
        out[name] = dict(summarize(rows), rows=rows)
    return out


def answer(llm, context: str, today, query: str) -> Tuple[bool, str]:
    system = SYSTEM.format(today=today.strftime("%Y-%m-%d"), weekday=WEEKDAYS[today.weekday()], context=context)
    raw = llm.chat(system=system, messages=[{"role": "user", "content": query}], temperature=0.0,
                   max_tokens=256, json_schema=SCHEMA)
    data = parse_json(raw)
    if not isinstance(data, dict):
        return False, str(raw)
    return bool(data.get("known")), str(data.get("answer", ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--players", default=None)
    ap.add_argument("--contexts", default="none,full")
    ap.add_argument("--results", default=None, help="run_e2e results file (for evidence:/chatbot: contexts)")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--llm-cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    players = [p for p in data["players"] if args.split == "all" or p["split"] == args.split]
    if args.players:
        wanted = set(args.players.split(","))
        players = [p for p in players if p["id"] in wanted]
    by_id = {p["id"]: p for p in data["players"]}
    results_file = None
    if args.results:
        with open(args.results, encoding="utf-8") as f:
            results_file = json.load(f)
    cache = LLMCache(args.llm_cache) if args.llm_cache else None
    if cache:  # systems that call an LLM at read time (P11) use the same cache
        from bench import run_e2e
        run_e2e._CACHE[:] = [cache]
    seeds = [int(x) for x in args.seeds.split(",")]

    out: Dict[str, Dict] = {}
    for spec in [c.strip() for c in args.contexts.split(",") if c.strip()]:
        provider = ContextProvider(spec, results_file, by_id)
        rows, t0 = [], time.time()
        for seed in seeds:
            llm = OllamaClient(model=args.model, base_url=args.base_url, timeout=900, seed=seed,
                               think=False if "qwen3" in args.model else None, num_ctx=args.num_ctx)
            if cache:
                cache.wrap(llm)
            for p in players:
                today = parse_time(p["ask_at"])
                for q in p["questions"]:
                    ctx = provider(p, q, seed)
                    known, ans = answer(llm, ctx, today, q["query"])
                    success, stale = judge_answer(q, known, ans)
                    rows.append({"id": q["id"], "player": p["id"], "seed": seed, "type": q["type"],
                                 "query": q["query"], "answer_any": q["answer_any"] or q.get("answer_all", []),
                                 "known": known, "answer": ans, "success": success, "stale": stale,
                                 "context_chars": len(ctx)})
            print(f"[{spec}] seed {seed} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res = summarize(rows)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        res["rows"] = rows
        out[spec] = res
        if args.out:
            _save(args, out)

    names = list(out)
    types = sorted({r["type"] for res in out.values() for r in res["rows"]})
    print(f"\nmodel={args.model} split={args.split} questions={out[names[0]]['n']}\n")
    print(f"{'context':22s} {'Score':>6s} {'Answ':>6s} {'Abst':>6s} {'Stale':>6s} {'chars':>6s}"
          + "".join(f" {t[:9]:>9s}" for t in types))
    for name, r in out.items():
        print(f"{name:22s} {r['AnswerScore']:6.3f} {r['Answered']:6.3f} {r['Abstain']:6.3f} "
              f"{r['StaleMention']:6.3f} {r['context_chars']:6d}" + "".join(f" {r['by_type'].get(t, 0):9.3f}" for t in types))
    if len(names) > 1:
        print("\npaired bootstrap (row - first):")
        for name in names[1:]:
            print(f"  {name:22s} {paired_bootstrap(out[names[0]]['rows'], out[name]['rows'])}")
    if args.show_errors:
        for name, r in out.items():
            print(f"\n--- misses: {name}")
            for row in r["rows"]:
                if row["success"] < 1:
                    print(f"  {row['id']}@{row['seed']} [{row['type']}] {row['query']} want={row['answer_any']} "
                          f"known={row['known']} got={row['answer']}")
    return out


def _save(args, out) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": os.path.relpath(os.path.abspath(args.data), ROOT), "split": args.split,
                   "model": args.model, "seeds": args.seeds, "results_file": args.results,
                   "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": out}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
