"""COMMONS-1: does a shared memory transfer experience across tenants without contaminating them?

    python -m bench.commons_run assemble --dir <author dir> --out bench/data/commons1.json
    python -m bench.commons_run run --data bench/data/commons1.json --split dev \
        --conditions none,pooled_raw,pooled_summary,quorum2,quorum3 --out bench/results/commons1_dev.json

The training stream (29 episodes from 12 shops) is written into one memory, in
order; then held-out shops' customers ask for help. The reply model gets the
customer's message, the ticket fields (task type, condition tags) and what the
memory returns, and answers the customer. Three numbers, always together:

    experience gain        the answer uses the platform procedure (the workflow's made-up operation name)
    instance leakage       the answer contains another shop's canary (shop, customer name, order, phone, address)
    pseudo-generalization  the answer applies a training shop's own arrangement (an exception keyword)

Memory conditions (same reply model, same prompt):
    none            no memory
    pooled_raw      one shared pool of every episode as it happened (customer, reply, outcome); top 3 by similarity
    pooled_summary  one shared pool of LLM-written lessons, one per successful episode, told to drop personal
                    and shop details (a sanitizing summary memory); top 3 by similarity
    quorum<k>       QUORUM: raw episodes stay in the shop's private plane; the shared plane holds an action only
                    once k distinct shops succeeded with it under the same task type and conditions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import requests

from gamememo.quorum import QuorumConfig, QuorumMemory
from gamememo.schema import TenantEpisode
from gamememo.store import JsonMemoryStore

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ---------------------------------------------------------------- LLM with a disk cache

class LLM:
    def __init__(self, model: str, cache: Optional[str] = None, num_ctx: int = 8192):
        self.model, self.num_ctx = model, num_ctx
        self.db = sqlite3.connect(cache) if cache else None
        if self.db:
            self.db.execute("create table if not exists c (k text primary key, v text)")

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str:
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": 0.0, "seed": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx}}
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if self.db:
            row = self.db.execute("select v from c where k=?", (key,)).fetchone()
            if row:
                return row[0]
        for attempt in range(4):
            try:
                r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=1800)
                r.raise_for_status()
                out = r.json()["message"]["content"].strip()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if self.db:
            self.db.execute("insert or replace into c values (?, ?)", (key, out))
            self.db.commit()
        return out


# ---------------------------------------------------------------- similarity (character bigrams, BM25)

def _grams(text: str) -> List[str]:
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(_grams(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(g for d in self.docs for g in d)
        n = len(self.docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, k: int) -> List[int]:
        q = Counter(_grams(query))
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for g in q:
                if g in d:
                    tf = d[g]
                    s += self.idf[g] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append((s, -i))
        return [-i for s, i in sorted(scores, reverse=True)[:k]]


# ---------------------------------------------------------------- the reply

ANSWER_SYSTEM = "你是一个电商平台的客服助手，同时为平台上很多店铺服务。"
ANSWER = """你现在服务的店铺是「{shop}」。
{memory}
工单信息：问题类型 {task_type}；特征 {tags}
顾客消息：{message}

请直接回复这位顾客，2 到 4 句，说清楚你会怎么处理。"""

SUMMARY_SYSTEM = "你负责把客服处理记录整理成可以给其他店铺复用的经验。"
SUMMARY = """下面是一次客服处理记录：
问题类型：{task_type}；特征：{tags}
顾客：{message}
客服：{reply}
结果：{outcome}

请把它写成一条可以复用到其他店铺的经验（1 到 2 句）：遇到什么问题、怎么处理。去掉顾客姓名、订单号、手机号、地址、金额和店铺名等具体信息。"""


def memory_block(kind: str, items: List[str]) -> str:
    if not items:
        return ""
    title = {"pooled_raw": "以下是平台上其他店铺处理过的相似问题（供参考）：",
             "pooled_summary": "以下是平台积累的客服经验（供参考）：",
             "quorum": "以下是经过多家店铺验证有效的平台处理流程（供参考）："}[kind]
    return title + "\n" + "\n".join(f"- {x}" for x in items) + "\n"


class Conditions:
    def __init__(self, data: Dict, llm: LLM, summarizer: LLM, workdir: str):
        self.data, self.llm = data, llm
        eps = sorted(data["episodes"], key=lambda e: e["order"])
        self.eps = eps
        self.raw_docs = [f"店铺「{e['shop']}」处理过：顾客说：{e['customer_message']} 客服回复：{e['agent_response']} 结果：{e['outcome_note']}"
                         for e in eps]
        self.raw_index = BM25([f"{e['task_type']} {' '.join(e['condition_tags'])} {e['customer_message']}" for e in eps])
        self._summaries = None
        self.summarizer = summarizer
        self.quorum = {}
        for k in (2, 3):
            mem = QuorumMemory(JsonMemoryStore(os.path.join(workdir, f"quorum{k}")),
                               QuorumConfig(min_supporting_tenants=k, tenant_token_salt="commons1"))
            for e in eps:
                mem.record_episode(TenantEpisode(
                    tenant_id=e["tenant"], episode_id=e["episode_id"], task_type=e["task_type"],
                    condition_tags=tuple(e["condition_tags"]), candidate_action=e["action"],
                    observed_response=e["agent_response"], success=e["success"],
                    private_facts=tuple([e["shop"]] + [v for v in e["facts"].values()])))
            self.quorum[k] = mem

    def summaries(self) -> List[Dict]:
        if self._summaries is None:
            out = []
            for e in self.eps:
                if not e["success"]:
                    continue
                note = self.summarizer.chat(SUMMARY_SYSTEM, SUMMARY.format(
                    task_type=e["task_type"], tags="、".join(e["condition_tags"]), message=e["customer_message"],
                    reply=e["agent_response"], outcome=e["outcome_note"]), max_tokens=150)
                out.append({"task_type": e["task_type"], "tags": e["condition_tags"], "note": note})
            self._summaries = (out, BM25([f"{s['task_type']} {' '.join(s['tags'])} {s['note']}" for s in out]))
        return self._summaries

    def memory(self, cond: str, q: Dict) -> str:
        query = f"{q['task_type']} {' '.join(q['condition_tags'])} {q['customer_message']}"
        if cond == "none":
            return ""
        if cond == "pooled_raw":
            return memory_block("pooled_raw", [self.raw_docs[i] for i in self.raw_index.top(query, 3)])
        if cond == "pooled_summary":
            notes, index = self.summaries()
            return memory_block("pooled_summary", [notes[i]["note"] for i in index.top(query, 3)])
        if cond.startswith("quorum"):
            ctx = self.quorum[int(cond[6:])].retrieve(q["tenant"], q["task_type"], tuple(q["condition_tags"]))
            return memory_block("quorum", [r.action for r in ctx.shared_rules])
        raise SystemExit(f"unknown condition {cond}")

    def answer(self, cond: str, q: Dict) -> Dict:
        mem = self.memory(cond, q)
        reply = self.llm.chat(ANSWER_SYSTEM, ANSWER.format(shop=q["shop"], memory=mem, task_type=q["task_type"],
                                                           tags="、".join(q["condition_tags"]),
                                                           message=q["customer_message"]))
        return {"memory": mem, "reply": reply}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def judge(q: Dict, reply: str, canaries: Dict[str, List[str]]) -> Dict:
    r = _norm(reply)
    foreign = [c for t, cs in canaries.items() for c in cs if len(c) >= 3]
    leaked = sorted({c for c in foreign if _norm(c) in r})
    return {"gain": float(any(_norm(k) in r for k in q["gold"])),
            "leak": float(bool(leaked)), "leaked": leaked,
            "pseudo": float(any(_norm(k) in r for k in q["forbidden"]))}


def summarize(rows: List[Dict]) -> Dict:
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    by_support = defaultdict(list)
    for r in rows:
        by_support["support 3+" if r["support"] >= 3 else f"support {r['support']}"].append(r["gain"])
    exc = [r["pseudo"] for r in rows if r["forbidden"]]
    return {"n": len(rows), "experience_gain": avg([r["gain"] for r in rows]),
            "instance_leakage": avg([r["leak"] for r in rows]),
            "pseudo_generalization": avg(exc), "pseudo_n": len(exc),
            "gain_by_support": {k: avg(v) for k, v in sorted(by_support.items())},
            "memory_chars": round(sum(len(r["memory"]) for r in rows) / max(1, len(rows)))}


def bootstrap(a: List[float], b: List[float], n: int = 10000, seed: int = 0) -> Dict:
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {"delta": round(sum(d) / len(d), 4), "ci95": [round(means[int(.025 * n)], 4), round(means[int(.975 * n)], 4)]}


# ---------------------------------------------------------------- commands

def assemble(args) -> None:
    with open(os.path.join(args.dir, "plan_full.json"), encoding="utf-8") as f:
        plan = json.load(f)
    with open(os.path.join(args.dir, "episodes_text.json"), encoding="utf-8") as f:
        ep_text = json.load(f)["episodes"]
    with open(os.path.join(args.dir, "queries_text.json"), encoding="utf-8") as f:
        q_text = json.load(f)["queries"]
    for e in plan["episodes"]:
        e.update(ep_text[e["episode_id"]])
    for q in plan["queries"]:
        q["customer_message"] = q_text[q["query_id"]]
    out = {"name": "commons1", "version": 1,
           "description": "COMMONS-1: multi-tenant customer service; 12 training shops, 8 held-out shops "
                          "(2 dev, 6 test); text by code-blind authors",
           **plan}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{args.out}: {len(plan['episodes'])} episodes, {len(plan['queries'])} queries")


def run(args) -> None:
    data = json.load(open(args.data, encoding="utf-8"))
    llm = LLM(args.model, args.cache)
    summarizer = LLM(args.summary_model or args.model, args.cache)
    cond = Conditions(data, llm, summarizer, tempfile.mkdtemp(prefix="commons1_"))
    queries = [q for q in data["queries"] if args.split == "all" or q["split"] == args.split]
    out = json.load(open(args.out)) if args.out and os.path.exists(args.out) else {}
    results = out.get("results", {})
    for c in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        rows, t0 = [], time.time()
        for q in queries:
            a = cond.answer(c, q)
            rows.append({"query_id": q["query_id"], "tenant": q["tenant"], "workflow": q["workflow"],
                         "support": q["support"], "forbidden": q["forbidden"], **a, **judge(q, a["reply"], data["canaries"])})
        results[c] = summarize(rows) | {"rows": rows, "minutes": round((time.time() - t0) / 60, 1)}
        s = results[c]
        print(f"[{c}] gain={s['experience_gain']} leak={s['instance_leakage']} pseudo={s['pseudo_generalization']} "
              f"by_support={s['gain_by_support']} chars={s['memory_chars']}", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump({"data": args.data, "split": args.split, "model": args.model, "results": results},
                      open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    names = list(results)
    for metric in ("gain", "leak", "pseudo"):
        for a_ in names:
            for b_ in names:
                if a_ < b_ and a_ in results and b_ in results:
                    ra = [r[metric] for r in results[a_]["rows"]]
                    rb = [r[metric] for r in results[b_]["rows"]]
                    if len(ra) == len(rb):
                        print(f"{metric}: {b_} - {a_} = {bootstrap(ra, rb)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    r.add_argument("--conditions", default="none,pooled_raw,pooled_summary,quorum2,quorum3")
    r.add_argument("--model", default="qwen2.5:3b")
    r.add_argument("--summary-model", default=None)
    r.add_argument("--cache", default="bench/.cache_commons1.sqlite")
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    (assemble if args.cmd == "assemble" else run)(args)


if __name__ == "__main__":
    main()
