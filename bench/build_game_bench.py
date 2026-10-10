# -*- coding: utf-8 -*-
"""Build the game-memory benchmark e2e_g1: chat + match data, game answers
computed from the matches.

    python -m bench.build_game_bench plan --out <dir>        # 1. simulate players, write the chat plan
    (an author who does not read the code writes chats.json and phrasings.json from the plan)
    python -m bench.build_game_bench assemble --dir <dir> --out bench/data/e2e_g1.json

Question types:
- game state, answered from match data (the answer key is computed, exact):
  current_rank, season_peak, last_season_peak, hero_winrate, hero_games,
  recent_top_hero, main_position, last_season_games, longest_losing_streak
- chat linked by a game event: what the player said on the day they first
  reached a tier / during the longest losing streak (link_promotion, link_streak)
- chat only (written by the chat author): fact, promise, negative

Each player's chat contains one inflated claim about rank ("brag"); the claim
is the stale value for the question it contradicts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, datetime, timedelta
from typing import Dict, List

from bench.gamesim import ASK_AT, END, START, make_specs, match_line, simulate, timeline
from gamememo.game.ledger import GameLedger
from gamememo.game.ontology import (POSITION_ALIASES, TIER_NAMES, level_start, previous_season, rank_name,
                                    season_of, tier_of)

N_DEV, N_TEST = 2, 8
SEED = 7


def _players():
    specs = make_specs(N_DEV + N_TEST, seed=SEED, prefix="g")
    for i, s in enumerate(specs):
        s.id = f"g1_d{i + 1}" if i < N_DEV else f"g1_t{i - N_DEV + 1}"
    return specs


def plan(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    out = []
    for spec in _players():
        ms = simulate(spec)
        led = GameLedger(ms)
        events = timeline(ms)
        rng = random.Random(spec.seed)
        play_days = sorted({m["time"][:10] for m in ms})
        anchor = [e for e in events if e["kind"] in ("promotion", "losing_streak", "main_switch")]
        chat_days = {e["date"] for e in anchor}
        while len(chat_days) < 13:
            chat_days.add(rng.choice(play_days))
        days = []
        for d in sorted(chat_days):
            dms = [m for m in ms if m["time"][:10] == d]
            day_ev = [e["text"] for e in events if e["date"] == d]
            summary = (f"当天{len(dms)}局，{sum(m['win'] for m in dms)}胜{len(dms) - sum(m['win'] for m in dms)}负，"
                       f"用了{'、'.join(sorted({m['hero'] for m in dms}))}") if dms else "当天没玩"
            last = [m for m in ms if m["time"][:10] <= d and m["mode"] == "排位"]
            days.append({"date": d, "game_summary": summary, "rank_after_that_day": rank_name(last[-1]["rank_after"], stars=False) if last else "",
                         "events": day_ev, "must_link_topic": bool({"promotion", "losing_streak"} & {e["kind"] for e in events if e["date"] == d})})
        last_season = previous_season(season_of(ASK_AT))
        peak_last = led.peak(ASK_AT, last_season)
        claim_tier_idx = min(len(TIER_NAMES) - 1, TIER_NAMES.index(tier_of(peak_last[0])) + 1)
        brag_day = rng.choice([d["date"] for d in days if d["date"] > last_season.end.isoformat()] or [days[-1]["date"]])
        out.append({"id": spec.id, "position": spec.position, "mains_in_order": [h for _, h in spec.phases],
                    "duo_partner": spec.duo, "chat_days": days,
                    "brag": {"date": brag_day, "true_last_season_peak": rank_name(peak_last[0], stars=False),
                             "claim": f"上赛季（{last_season.name}）打到了{TIER_NAMES[claim_tier_idx]}",
                             "claimed_tier": TIER_NAMES[claim_tier_idx]}})
    with open(os.path.join(args.out, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} player plans to {args.out}/plan.json")


# ---------------------------------------------------------------- answers computed from data

def game_questions(spec, ms: List[Dict], phr: Dict[str, List[str]], rng: random.Random, chat: Dict) -> List[Dict]:
    led = GameLedger(ms)
    now = ASK_AT
    season = season_of(now)
    last = previous_season(season)
    qs = []

    def q(kind, answer_type, key, stale=(), **fmt):
        text = rng.choice(phr[kind]).format(**fmt)
        qs.append({"type": "game", "subtype": kind, "query": text, "answer_type": answer_type,
                   "key": key, "stale_any": list(stale)})

    cur = led.current_rank(now)[0]
    q("current_rank", "rank", rank_name(cur, stars=False))
    q("season_peak", "rank", rank_name(led.peak(now, season)[0], stars=False))
    brag = chat.get("brag_claimed_tier")
    true_last = rank_name(led.peak(now, last)[0], stars=False)
    q("last_season_peak", "rank", true_last, stale=[brag] if brag and brag not in true_last else [])
    top = led.heroes(now)[0][0]
    hero2 = led.heroes(now)[1][0]
    rec = led.record(now, hero=top)
    q("hero_winrate", "percent", round(100 * rec.winrate), hero=top)
    q("hero_games", "count", led.record(now, hero=hero2).games, hero=hero2)
    q("recent_top_hero", "name", led.heroes(now, since=(now - timedelta(days=30)).date())[0][0])
    pos = led.positions(now)[0][0]
    q("main_position", "name", pos)
    q("last_season_games", "count", led.record(now, season=last, ranked_only=True).games)
    q("longest_losing_streak", "count_exact", led.losing_streak(now)[0])
    return qs


def assemble(args) -> None:
    with open(os.path.join(args.dir, "chats.json"), encoding="utf-8") as f:
        chats = {p["id"]: p for p in json.load(f)["players"]}
    with open(os.path.join(args.dir, "phrasings.json"), encoding="utf-8") as f:
        phr = json.load(f)
    players = []
    for spec in _players():
        ms = simulate(spec)
        chat = chats[spec.id]
        split = "dev" if spec.id.startswith("g1_d") else "test"
        rng = random.Random(spec.seed + 1)
        qs = game_questions(spec, ms, phr[split], rng, chat)
        led = GameLedger(ms)
        for kind, day_kind in (("link_promotion", "promotion"), ("link_streak", "losing_streak")):
            link = next((l for l in chat.get("links", []) if l["kind"] == day_kind), None)
            if not link:
                continue
            fmt = {}
            if day_kind == "promotion":
                fmt["tier"] = link.get("tier", "")
            qs.append({"type": "link", "subtype": kind, "query": rng.choice(phr[split][kind]).format(**fmt),
                       "answer_type": "keyword", "key": link["answer_any"], "stale_any": [], "date": link["date"]})
        for cq in chat["questions"]:
            qs.append({"type": cq["type"], "subtype": cq["type"], "query": cq["query"], "answer_type": "keyword",
                       "key": cq.get("answer_any", []), "stale_any": cq.get("stale_any", [])})
        for i, x in enumerate(qs):
            x["id"] = f"{spec.id}_q{i + 1:02d}"
        players.append({"id": spec.id, "split": split, "ask_at": ASK_AT.strftime("%Y-%m-%d %H:%M:%S"),
                        "sessions": chat["sessions"], "matches": ms, "questions": qs})
    out = {"name": "e2e_g1", "version": 1,
           "description": "game memory: chat + simulated 王者荣耀 match data; game answers computed from matches",
           "players": players}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"{args.out}: {len(players)} players, {sum(len(p['questions']) for p in players)} questions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    (plan if args.cmd == "plan" else assemble)(args)


if __name__ == "__main__":
    main()
