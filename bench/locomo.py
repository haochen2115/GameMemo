# -*- coding: utf-8 -*-
"""External check on LoCoMo (Maharana et al., 2024): do the GameMemo findings
(E11-E13) hold on a public long-conversation benchmark, against Mem0?

    python -m bench.locomo answer --data /path/locomo10.json --contexts full,raw:10 --out ...
    python -m bench.locomo mem0-ingest --data ... --out-dir bench/.mem0     # Mem0 writes (LLM)
    python -m bench.locomo answer --contexts mem0:10,mem0all --mem0-dir bench/.mem0 ...
    python -m bench.locomo judge --answers ... --judge qwen2.5:7b             # LLM judge

Contexts (what the reply model sees before answering):
    full          every session verbatim, with its date
    raw:<k>       the k turns most relevant to the question, verbatim, dated, in time
                  order (no LLM on the write path; RawMemory's idea, without its
                  game-specific parts)
    mem0:<k>      the k memories Mem0 retrieves for the question
    mem0all       every memory Mem0 wrote for the conversation (no retrieval)

``--sessions N`` keeps the first N sessions of every conversation and only the
questions whose evidence lies in them: the same questions over histories of
growing length.

Categories (LoCoMo numbering): 1 multi-hop, 2 temporal, 3 open-domain,
4 single-hop; 5 (adversarial) is excluded, as in most memory papers.
Scores: token F1 against the gold answer (LoCoMo's metric) and, after
``judge``, an LLM judge's CORRECT/WRONG (the Mem0 paper's "J").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.run_e2e import LLMCache  # noqa: E402
from gamememo.llm import OllamaClient  # noqa: E402

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

ANSWER_SYSTEM = """You are answering questions about a long conversation between two people, {a} and {b}.

{context}

Answer the question using only the information above. Use the dates to resolve relative time
("yesterday", "last week") into concrete dates. Give a short answer (a few words or a short phrase),
not a full sentence. If the information above does not contain the answer, say "unknown"."""


# ---------------------------------------------------------------- data

def load(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sessions(sample: Dict) -> List[Tuple[int, str, List[Dict]]]:
    conv = sample["conversation"]
    out = []
    n = 1
    while f"session_{n}" in conv:
        out.append((n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]))
        n += 1
    return out


def turn_text(t: Dict) -> str:
    text = f"{t['speaker']}: {t['text']}"
    if t.get("blip_caption"):
        text += f" [shares a photo: {t['blip_caption']}]"
    return text


def questions(sample: Dict, max_session: Optional[int]) -> List[Dict]:
    out = []
    for i, q in enumerate(sample["qa"]):
        if q.get("category") not in CATEGORIES or "answer" not in q:
            continue
        ev = [int(m.group(1)) for e in q.get("evidence", []) for m in [re.match(r"D(\d+):", e)] if m]
        if max_session and (not ev or max(ev) > max_session):
            continue
        out.append(dict(q, id=f"{sample['sample_id']}_q{i:03d}", answer=str(q["answer"])))
    return out


# ---------------------------------------------------------------- scoring

def _norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm_tokens(pred), _norm_tokens(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- contexts

class Contexts:
    def __init__(self, sample: Dict, max_session: Optional[int], mem0_dir: Optional[str]):
        self.sample = sample
        self.sess = [s for s in sessions(sample) if not max_session or s[0] <= max_session]
        self.max_session = max_session
        self.mem0_dir = mem0_dir
        self._raw = None
        self._mem0 = None

    def full(self) -> str:
        parts = [f"[Session {n}, {date}]\n" + "\n".join(turn_text(t) for t in turns) for n, date, turns in self.sess]
        return "Here is the whole conversation so far, in order:\n\n" + "\n\n".join(parts)

    def raw(self, query: str, k: int) -> Tuple[str, List[str]]:
        from bench.run_e2e import jina
        from gamememo.personal.model import MemoryRecord
        from gamememo.personal.retrieval import HybridRetriever, RetrievalConfig
        if self._raw is None:
            recs = []
            for n, date, turns in self.sess:
                for i, t in enumerate(turns):
                    recs.append(MemoryRecord(content=turn_text(t), id=t["dia_id"], created_at=f"{n:03d}.{i:03d}",
                                             event_time=date))
            # every LoCoMo question is about the conversation: recall-oriented, no relevance gate
            cfg = RetrievalConfig.for_embedder(jina()).for_candidates()
            self._raw = (HybridRetriever(jina(), cfg), recs)
        retriever, recs = self._raw
        hits = sorted((h.record for h in retriever.search(query, recs, top_k=k)), key=lambda r: r.created_at)
        lines = [f"[{r.event_time}] {r.content}" for r in hits]
        return ("Relevant parts of the conversation, in time order:\n" + "\n".join(lines)), [r.id for r in hits]

    def _mem0_store(self):
        if self._mem0 is None:
            self._mem0 = mem0_client(self.mem0_dir, self.sample["sample_id"], self.max_session)
        return self._mem0

    def mem0(self, query: str, k: int) -> str:
        m, uid = self._mem0_store()
        res = m.search(query, filters={"user_id": uid}, limit=k).get("results", [])
        return "Memories about the conversation:\n" + "\n".join(f"- {r['memory']}" for r in res)

    def mem0all(self) -> str:
        m, uid = self._mem0_store()
        res = m.get_all(filters={"user_id": uid}, limit=10000).get("results", [])
        return "Memories about the conversation:\n" + "\n".join(f"- {r['memory']}" for r in res)


# ---------------------------------------------------------------- Mem0

def mem0_config(path: str, collection: str) -> Dict:
    return {
        "llm": {"provider": "ollama", "config": {"model": "qwen2.5:3b", "temperature": 0.0, "max_tokens": 1500,
                                                 "ollama_base_url": "http://localhost:11434"}},
        "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text",
                                                      "ollama_base_url": "http://localhost:11434"}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": collection, "path": path,
                                                          "embedding_model_dims": 768, "on_disk": True}},
        "history_db_path": os.path.join(path, "history.db"),
    }


def mem0_client(base: str, sample_id: str, max_session: Optional[int]):
    # Mem0's telemetry opens a second local qdrant store (~/.mem0/migrations_qdrant)
    # per Memory instance, and local qdrant allows one client per process.
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    from mem0 import Memory
    path = os.path.join(base, sample_id)
    return Memory.from_config(mem0_config(path, "locomo")), sample_id


def mem0_ingest(args) -> None:
    """Write every session into Mem0, one add() per session, the session's date
    stated in the text (otherwise Mem0 resolves "yesterday" against today)."""
    data = load(args.data)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "ingest_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for sample in data:
        sid = sample["sample_id"]
        if args.samples and sid not in args.samples.split(","):
            continue
        m, uid = mem0_client(args.out_dir, sid, None)
        a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
        for n, date, turns in sessions(sample):
            key = f"{sid}:{n}"
            if key in log:
                continue
            msgs = [{"role": "user", "content": f"(This conversation between {a} and {b} took place on {date}.)"}]
            msgs += [{"role": "user", "content": turn_text(t)} for t in turns]
            t0 = time.time()
            try:
                res = m.add(msgs, user_id=uid, metadata={"session": n, "date": date})
                log[key] = {"events": [r.get("event") for r in res.get("results", [])], "seconds": round(time.time() - t0, 1)}
            except Exception as e:
                log[key] = {"error": str(e)[:300], "seconds": round(time.time() - t0, 1)}
            with open(log_path, "w") as f:
                json.dump(log, f, indent=1)
            print(key, log[key], flush=True)


# ---------------------------------------------------------------- answering

def answer_all(args) -> None:
    data = load(args.data)
    cache = LLMCache(args.llm_cache) if args.llm_cache else None
    llm = OllamaClient(model=args.model, timeout=1800, seed=0, num_ctx=args.num_ctx)
    if cache:
        cache.wrap(llm)
    out = json.load(open(args.out)) if args.out and os.path.exists(args.out) else {"results": {}}
    out.update({"data": os.path.basename(args.data), "model": args.model, "sessions": args.sessions})
    specs = [c.strip() for c in args.contexts.split(",") if c.strip()]
    for spec in specs:
        rows = out["results"].get(spec, {}).get("rows", [])
        done = {r["id"] for r in rows}
        t0 = time.time()
        for sample in data:
            if args.samples and sample["sample_id"] not in args.samples.split(","):
                continue
            ctx = Contexts(sample, args.sessions, args.mem0_dir)
            a, b = sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]
            for q in questions(sample, args.sessions):
                if q["id"] in done:
                    continue
                kind, _, k = spec.partition(":")
                retrieved = None
                if kind == "full":
                    context = ctx.full()
                elif kind == "raw":
                    context, retrieved = ctx.raw(q["question"], int(k or 10))
                elif kind == "mem0":
                    context = ctx.mem0(q["question"], int(k or 10))
                elif kind == "mem0all":
                    context = ctx.mem0all()
                elif kind == "none":
                    context = "(no information)"
                else:
                    raise SystemExit(f"unknown context {spec}")
                pred = llm.chat(system=ANSWER_SYSTEM.format(a=a, b=b, context=context),
                                messages=[{"role": "user", "content": q["question"]}],
                                temperature=0.0, max_tokens=64).strip()
                row = {"id": q["id"], "category": q["category"], "question": q["question"], "gold": q["answer"],
                       "evidence": q.get("evidence", []), "pred": pred, "f1": round(f1(pred, q["answer"]), 4),
                       "context_chars": len(context)}
                if retrieved is not None:
                    row["retrieved"] = retrieved
                    row["evidence_recall"] = (sum(e in retrieved for e in q.get("evidence", [])) /
                                              max(1, len(q.get("evidence", []))))
                rows.append(row)
            if args.out:
                out["results"][spec] = summarize(rows) | {"rows": rows}
                _save(args.out, out)
            print(f"[{spec}] {sample['sample_id']} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        out["results"][spec] = summarize(rows) | {"rows": rows}
        if args.out:
            _save(args.out, out)
    report(out)


def summarize(rows: List[Dict]) -> Dict:
    by = defaultdict(list)
    for r in rows:
        by[CATEGORIES[r["category"]]].append(r)
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    res = {"n": len(rows), "F1": avg([r["f1"] for r in rows]),
           "F1_by": {c: avg([r["f1"] for r in rs]) for c, rs in sorted(by.items())},
           "context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows)))}
    if rows and all("judge" in r for r in rows):
        res["J"] = avg([r["judge"] for r in rows])
        res["J_by"] = {c: avg([r["judge"] for r in rs]) for c, rs in sorted(by.items())}
    if rows and "evidence_recall" in rows[0]:
        res["evidence_recall"] = avg([r["evidence_recall"] for r in rows])
    return res


def report(out: Dict) -> None:
    print(f"\nmodel={out.get('model')} sessions={out.get('sessions')}")
    for spec, res in out["results"].items():
        extra = f" J={res['J']:.3f}" if res.get("J") is not None else ""
        extra += f" evRecall={res['evidence_recall']:.3f}" if res.get("evidence_recall") is not None else ""
        print(f"{spec:10s} n={res['n']:4d} F1={res['F1']:.3f}{extra} chars={res['context_chars']}  {res['F1_by']}")


def _save(path: str, out: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- judging

JUDGE_PROMPT = """Your task is to label an answer to a question as CORRECT or WRONG.
You will be given a question, a gold (ground truth) answer, and a generated answer.
Be generous: if the generated answer refers to the same thing or the same time as the gold answer,
even with different wording or format (e.g. "May 7th" vs "7 May 2023"), it is CORRECT.
If it says it does not know, or gives a different thing or time, it is WRONG.

Question: {question}
Gold answer: {gold}
Generated answer: {pred}

Reply with one word: CORRECT or WRONG."""


def judge(args) -> None:
    cache = LLMCache(args.llm_cache) if args.llm_cache else None
    llm = OllamaClient(model=args.judge, timeout=600, seed=0, num_ctx=2048)
    if cache:
        cache.wrap(llm)
    out = json.load(open(args.answers))
    for spec, res in out["results"].items():
        for r in res["rows"]:
            if "judge" in r:
                continue
            reply = llm.chat(prompt=JUDGE_PROMPT.format(question=r["question"], gold=r["gold"], pred=r["pred"]),
                             temperature=0.0, max_tokens=8)
            r["judge"] = 1.0 if "CORRECT" in reply.upper() and "WRONG" not in reply.upper() else 0.0
        out["results"][spec] = summarize(res["rows"]) | {"rows": res["rows"]}
        _save(args.answers, out)
        print(f"[{spec}] judged", flush=True)
    out["judge_model"] = args.judge
    _save(args.answers, out)
    report(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer")
    a.add_argument("--data", required=True)
    a.add_argument("--contexts", default="full,raw:10")
    a.add_argument("--model", default="qwen2.5:3b")
    a.add_argument("--num-ctx", type=int, default=32768)
    a.add_argument("--sessions", type=int, default=None)
    a.add_argument("--samples", default=None)
    a.add_argument("--mem0-dir", default=None)
    a.add_argument("--llm-cache", default=None)
    a.add_argument("--out", default=None)
    i = sub.add_parser("mem0-ingest")
    i.add_argument("--data", required=True)
    i.add_argument("--samples", default=None)
    i.add_argument("--out-dir", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--answers", required=True)
    j.add_argument("--judge", default="qwen2.5:7b")
    j.add_argument("--llm-cache", default=None)
    args = ap.parse_args(argv)
    {"answer": answer_all, "mem0-ingest": mem0_ingest, "judge": judge}[args.cmd](args)


if __name__ == "__main__":
    main()
