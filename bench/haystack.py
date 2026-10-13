# -*- coding: utf-8 -*-
"""Long-horizon e2e data: pad every player's history with neutral filler
sessions (LongMemEval-style haystack) so that putting the whole history in
the prompt stops being free.

    # 1. build the padded dataset (no LLM)
    python -m bench.haystack build --data bench/data/e2e_v10.json --split test --sizes 40,160
    # 2. write every filler session once into an empty memory (needs Ollama)
    python -m bench.haystack ingest --system p6b --seeds 0,1,2 --out bench/results/haystack_v1_p6b.json
    # 3. merge filler memories into a run_e2e results file for a padded dataset
    python -m bench.haystack merge --results bench/results/e2e_v10_test.json --system p6b \\
        --fill bench/results/haystack_v1_p6b.json --data bench/data/e2e_v10_h160.json \\
        --out bench/results/e2e_v10_h160_test.json

Filler sessions are spread over the player's real timeline, between the
first session and ``ask_at``; each player gets its own sample and order.

Writing the filler with the full write path would cost ~50 s per session
on CPU (hours per player). Step 2 instead writes each filler session once,
into an empty memory, and step 3 adds the resulting records (re-dated) to
the player's stored memories. This ignores interactions between filler
and real memories; filler is checked to carry no personal facts, so the
interaction should be small (validated on a small scale in E13).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
from datetime import timedelta
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gamememo.personal.model import parse_time  # noqa: E402

HAYSTACK = os.path.join(ROOT, "bench", "data", "haystack_v1.json")
FILL_DATE = "2026-05-01 21:00:00"  # nominal date when a filler session is written alone


def load_haystack(path: str = HAYSTACK) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["sessions"]


def check_haystack(sessions: List[Dict], datasets: List[str]) -> List[str]:
    """Filler lines that mention any answer key / stale value of the given
    datasets, a rank tier, or look like an assistant promise."""
    from gamememo.personal.system import promises_from_rules
    from bench.run_e2e import transcript
    keys = set()
    for path in datasets:
        with open(path, encoding="utf-8") as f:
            for p in json.load(f)["players"]:
                for q in p["questions"]:
                    keys.update(q.get("answer_any", []) + q.get("answer_all", []) + q.get("stale_any", []))
    keys = {k for k in keys if len(k) >= 2}
    tier = re.compile(r"(青铜|白银|黄金|铂金|钻石|星耀)[一二三四五1-5]|荣耀王者|王者\d+星")
    problems = []
    for s in sessions:
        for t in s["turns"]:
            hit = sorted(k for k in keys if k in t["content"])
            if hit:
                problems.append(f"{s['id']} answer key {hit}: {t['content']}")
            if tier.search(t["content"]):
                problems.append(f"{s['id']} rank tier: {t['content']}")
        for line in promises_from_rules(transcript(s)):
            problems.append(f"{s['id']} promise-like: {line}")
    return problems


def pad_player(player: Dict, filler: List[Dict], size: int) -> Dict:
    """A copy of player with ``size`` filler sessions spread over its timeline."""
    rng = random.Random(f"{player['id']}:{size}")
    picked = rng.sample(filler, size) if size <= len(filler) else [rng.choice(filler) for _ in range(size)]
    start = parse_time(player["sessions"][0]["date"])
    end = parse_time(player["ask_at"]) - timedelta(days=1)
    span = (end - start).total_seconds()
    real_days = {s["date"][:10] for s in player["sessions"]}
    sessions = list(player["sessions"])
    for i, h in enumerate(picked):
        when = start + timedelta(seconds=span * (i + rng.random()) / size)
        when = when.replace(hour=rng.choice([12, 13, 20, 21, 22, 23]), minute=rng.randrange(60), second=0)
        while when.strftime("%Y-%m-%d") in real_days:  # never share a day with a real session
            when += timedelta(days=1)
        sessions.append({"date": when.strftime("%Y-%m-%d %H:%M:%S"), "turns": h["turns"], "filler": h["id"]})
    sessions.sort(key=lambda s: s["date"])
    return dict(player, sessions=sessions)


def build(args) -> None:
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    filler = load_haystack(args.haystack)
    # drop filler that mentions this dataset's answer keys (a hero met as an
    # opponent must not make a wrong answer match by accident)
    bad = {line.split()[0] for line in check_haystack(filler, [args.data])}
    filler = [h for h in filler if h["id"] not in bad]
    print(f"{len(filler)} filler sessions usable ({len(bad)} dropped: {sorted(bad)})")
    base = os.path.splitext(args.data)[0]
    for size in [int(x) for x in args.sizes.split(",")]:
        players = [pad_player(p, filler, size) for p in data["players"]
                   if args.split == "all" or p["split"] == args.split]
        out = dict(data, name=f"{data.get('name', '')}_h{size}", players=players,
                   description=f"{data.get('description', '')} + {size} neutral filler sessions per player "
                               f"({os.path.basename(args.haystack)})")
        path = f"{base}_h{size}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        chars = [sum(len(t["content"]) for s in p["sessions"] for t in s["turns"]) for p in players]
        print(f"{path}: {len(players)} players, {size + 8} sessions each, ~{sum(chars) // len(chars)} chars")


def ingest(args) -> None:
    from bench.run_e2e import LLMCache, SYSTEMS, V2System, _CACHE, transcript
    if args.llm_cache:
        _CACHE[:] = [LLMCache(args.llm_cache)]
    filler = load_haystack(args.haystack)
    out = {"haystack": os.path.relpath(args.haystack, ROOT), "system": args.system, "model": args.model,
           "memories": {}, "errors": []}
    if os.path.exists(args.out):  # resume
        with open(args.out, encoding="utf-8") as f:
            out = json.load(f)
    for seed in [int(x) for x in args.seeds.split(",")]:
        for h in filler:
            key = f"{h['id']}@{seed}"
            if key in out["memories"]:
                continue
            system = SYSTEMS[args.system](args.model, args.base_url, tempfile.mkdtemp(prefix="fill_"))
            assert isinstance(system, V2System)
            system.mem.llm.seed = seed
            try:
                system.ingest(transcript(h), parse_time(FILL_DATE))
            except Exception as e:
                out["errors"].append(f"{key}: {e}")
            out["memories"][key] = system.dump()
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"seed {seed}: {len(filler)} filler sessions written", flush=True)


def _redate(rec: Dict, date: str) -> Dict:
    rec = dict(rec)
    day = FILL_DATE[:10]
    for k in ("created_at", "updated_at", "valid_to", "last_accessed_at"):
        if rec.get(k):
            rec[k] = rec[k].replace(day, date[:10])
    if rec.get("event_time") and rec["event_time"].startswith(day):
        rec["event_time"] = date[:10] + rec["event_time"][10:]
    return rec


def merge(args) -> None:
    """Add each padded player's filler memories to its stored memories in
    a run_e2e results file; rows are dropped (they must be re-judged)."""
    with open(args.results, encoding="utf-8") as f:
        saved = json.load(f)
    with open(args.fill, encoding="utf-8") as f:
        fill = json.load(f)["memories"]
    with open(args.data, encoding="utf-8") as f:
        players = {p["id"]: p for p in json.load(f)["players"]}
    res = saved["results"][args.system]
    merged = {}
    for key, mems in res["memories"].items():
        pid, seed = key.split("@")
        if pid not in players:
            continue
        extra = []
        for s in players[pid]["sessions"]:
            if "filler" in s:
                # a filler session can be reused across players: give its records fresh ids
                recs = fill[f"{s['filler']}@{seed if args.fill_seed is None else args.fill_seed}"]
                ids = {m["id"]: f"{m['id']}-{s['date'][:10]}" for m in recs}
                for m in recs:
                    m = _redate(m, s["date"])
                    for k in ("id", "supersedes", "superseded_by"):
                        if m.get(k):
                            m[k] = ids.get(m[k], m[k])
                    extra.append(m)
        merged[key] = mems + extra
    out = dict(saved, dataset=os.path.relpath(os.path.abspath(args.data), ROOT),
               results={args.system: {"memories": merged, "rows": [],
                                      "note": f"merged {os.path.basename(args.fill)} into {os.path.basename(args.results)}"}})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{args.out}: {len(merged)} player-runs, "
          f"{sum(len(v) for v in merged.values()) / max(1, len(merged)):.0f} memories each")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", required=True)
    b.add_argument("--split", default="test")
    b.add_argument("--sizes", default="40,160")
    b.add_argument("--haystack", default=HAYSTACK)
    c = sub.add_parser("check")
    c.add_argument("--haystack", default=HAYSTACK)
    c.add_argument("--datasets", required=True, help="comma-separated e2e datasets whose answer keys must not leak")
    i = sub.add_parser("ingest")
    i.add_argument("--system", default="p6b")
    i.add_argument("--model", default="qwen2.5:3b")
    i.add_argument("--base-url", default="http://localhost:11434")
    i.add_argument("--seeds", default="0")
    i.add_argument("--haystack", default=HAYSTACK)
    i.add_argument("--llm-cache", default=None)
    i.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--results", required=True)
    m.add_argument("--system", required=True)
    m.add_argument("--fill", required=True)
    m.add_argument("--data", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--fill-seed", type=int, default=None, help="use this seed's filler memories for every seed")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args)
    elif args.cmd == "check":
        problems = check_haystack(load_haystack(args.haystack), args.datasets.split(","))
        print("\n".join(problems) or "ok")
    elif args.cmd == "ingest":
        ingest(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
