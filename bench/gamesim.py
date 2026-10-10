# -*- coding: utf-8 -*-
"""Simulated 王者荣耀 match history for the game-memory benchmark.

Each player gets six months of matches (deterministic given the spec's seed):
ranked games move stars along the ladder of ``gamememo.game.ontology``;
seasons reset ranks; the main hero changes in phases; win rate on a hero rises
with practice; the player plays some evenings and more at weekends, sometimes
on tilt after losses. Casual (匹配) games do not move rank.

    python -m bench.gamesim --players 3 --show     # print a digest per player

``timeline`` turns a match history into the dated events a person would talk
about (promotions, losing streaks, switching main hero, season ends); the
benchmark's chat author writes the chats from it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from gamememo.game.ontology import (HERO_POSITION, POSITIONS, RESET_FLOOR, SEASON_DROP, SEASONS, TIER_NAMES,
                                    level_start, rank_name, season_of, tier_of)

START = date(2026, 3, 1)
END = date(2026, 9, 30)
ASK_AT = datetime(2026, 10, 3, 21, 0)

_KDA = {  # position -> (kills, deaths, assists) means for a loss; wins shift them
    "发育路": (5.5, 4.5, 4.0), "打野": (6.0, 4.5, 5.0), "中路": (4.5, 4.0, 6.0),
    "对抗路": (3.0, 4.5, 5.0), "游走": (1.0, 5.0, 8.5)}


@dataclass
class PlayerSpec:
    id: str
    seed: int
    start_points: int
    position: str
    phases: List[Tuple[str, str]]          # (from date, main hero)
    second_position: str
    activity: float = 0.5                  # chance of playing on a weekday evening
    skill: float = 0.0                     # baseline edge over the field
    comfort: int = 60                      # star index the player's skill naturally sits at
    duo: Optional[str] = None              # friend they sometimes queue with


def _main_on(spec: PlayerSpec, d: date) -> str:
    main = spec.phases[0][1]
    for since, hero in spec.phases:
        if date.fromisoformat(since) <= d:
            main = hero
    return main


def simulate(spec: PlayerSpec, start: date = START, end: date = END) -> List[Dict]:
    rng = random.Random(spec.seed)
    points = spec.start_points
    season = season_of(start)
    played: Counter = Counter()
    losses_in_row = 0
    matches: List[Dict] = []
    pool_main = [h for h, p in HERO_POSITION.items() if p == spec.position]
    pool_second = [h for h, p in HERO_POSITION.items() if p == spec.second_position]
    d = start
    while d <= end:
        if season_of(d) != season:
            season = season_of(d)
            points = max(RESET_FLOOR, points - SEASON_DROP) if points > RESET_FLOOR else points
        weekend = d.weekday() >= 5
        if rng.random() < (min(0.95, spec.activity + 0.3) if weekend else spec.activity):
            t = datetime(d.year, d.month, d.day, rng.choice([13, 15, 20, 21]) if weekend else rng.choice([20, 21, 22]),
                         rng.randrange(60))
            for _ in range(rng.choice([1, 2, 2, 3, 3, 4, 5, 6])):
                main = _main_on(spec, d)
                r = rng.random()
                hero = main if r < 0.55 else rng.choice(pool_main if r < 0.85 else pool_second)
                ranked = rng.random() < 0.8
                practice = 0.10 * (1 - math.exp(-played[hero] / 40))
                pressure = max(-0.15, min(0.15, (points - spec.comfort) / 120))
                tilt = 0.04 if losses_in_row >= 2 else 0.0
                p_win = 0.47 + spec.skill + practice - (pressure if ranked else 0) - tilt
                win = rng.random() < p_win
                pos = HERO_POSITION[hero]
                k, de, a = _KDA[pos]
                bonus = 1.6 if win else 1.0
                kills = max(0, round(rng.gauss(k * bonus, k * 0.5)))
                deaths = max(0, round(rng.gauss(de / bonus, 1.5)))
                assists = max(0, round(rng.gauss(a * bonus, a * 0.4)))
                before = points
                if ranked:
                    points = points + 1 if win else max(0, points - 1)
                matches.append({
                    "time": t.strftime("%Y-%m-%d %H:%M"), "season": season.name, "mode": "排位" if ranked else "匹配",
                    "hero": hero, "position": pos, "win": win, "k": kills, "d": deaths, "a": assists,
                    "mvp": win and rng.random() < 0.25, "rank_before": before, "rank_after": points,
                    "duo": spec.duo if spec.duo and rng.random() < 0.2 else None})
                played[hero] += 1
                losses_in_row = 0 if win else losses_in_row + 1
                t += timedelta(minutes=rng.randrange(14, 24))
        d += timedelta(days=1)
    return matches


def match_line(m: Dict) -> str:
    """One match as a player-facing record line."""
    res = "胜" if m["win"] else "负"
    rank = f" {rank_name(m['rank_before'])}→{rank_name(m['rank_after'])}" if m["mode"] == "排位" else ""
    return (f"{m['time']} {m['mode']} {m['hero']}（{m['position']}）{res} {m['k']}/{m['d']}/{m['a']}"
            f"{' MVP' if m['mvp'] else ''}{' 和' + m['duo'] + '双排' if m['duo'] else ''}{rank}")


# ---------------------------------------------------------------- what a player would talk about

def timeline(matches: List[Dict]) -> List[Dict]:
    """Dated events worth mentioning in chat."""
    events = []
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    for m in matches:
        by_day[m["time"][:10]].append(m)
    best = None  # highest tier reached so far (index in TIER_NAMES)
    streak = 0
    worst = (0, None)
    last_main = None
    hero_count: Counter = Counter()
    first_day = date.fromisoformat(min(by_day))
    for day in sorted(by_day):
        ms = by_day[day]
        for m in ms:
            if m["mode"] != "排位":
                continue
            t = TIER_NAMES.index(tier_of(m["rank_after"]))
            if best is not None and t > best:
                events.append({"date": day, "kind": "promotion", "text": f"首次打到{TIER_NAMES[t]}"})
            best = t if best is None else max(best, t)
            streak = 0 if m["win"] else streak + 1
            if streak > worst[0]:
                worst = (streak, day)
        wins = sum(m["win"] for m in ms)
        if len(ms) >= 4 and wins == 0:
            events.append({"date": day, "kind": "bad_day", "text": f"一天{len(ms)}把全输"})
        for m in ms:
            hero_count[m["hero"]] += 1
        top = hero_count.most_common(1)[0][0] if hero_count else None
        recent = Counter(m["hero"] for m in matches if day >= m["time"][:10] >= (date.fromisoformat(day) - timedelta(days=21)).isoformat())
        main_now = recent.most_common(1)[0][0] if recent else top
        if date.fromisoformat(day) - first_day < timedelta(days=21):
            last_main = main_now  # too early to call a switch
        elif main_now != last_main and recent[main_now] >= 8:
            events.append({"date": day, "kind": "main_switch", "text": f"最近主玩从{last_main}换成了{main_now}"})
            last_main = main_now
    if worst[1]:
        events.append({"date": worst[1], "kind": "losing_streak", "text": f"排位连输{worst[0]}把（最长的一次）"})
    for s in SEASONS:
        ms = [m for m in matches if m["season"] == s.name and m["mode"] == "排位"]
        if ms and s.end <= END:
            peak = max(m["rank_after"] for m in ms)
            events.append({"date": s.end.isoformat(), "kind": "season_end",
                           "text": f"{s.name}赛季结束，最高{rank_name(peak, stars=False)}，结束时{rank_name(ms[-1]['rank_after'], stars=False)}"})
    return sorted(events, key=lambda e: e["date"])


# ---------------------------------------------------------------- player specs

_DUOS = ["阿杰", "小周", "老三", "胖虎", "表弟", "室友", "阿伟", None, None, None]


def make_specs(n: int, seed: int = 0, prefix: str = "g") -> List[PlayerSpec]:
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        pos = rng.choice(POSITIONS)
        second = rng.choice([p for p in POSITIONS if p != pos])
        pool = [h for h, p in HERO_POSITION.items() if p == pos]
        mains = rng.sample(pool, 3)
        switch1 = START + timedelta(days=rng.randrange(40, 80))
        switch2 = switch1 + timedelta(days=rng.randrange(40, 70))
        comfort = rng.randrange(level_start("铂金"), level_start("星耀") + 10)
        specs.append(PlayerSpec(
            id=f"{prefix}{i + 1}", seed=seed * 1000 + i, start_points=comfort - rng.randrange(0, 15), position=pos,
            phases=[(START.isoformat(), mains[0]), (switch1.isoformat(), mains[1]), (switch2.isoformat(), mains[2])],
            second_position=second, activity=rng.uniform(0.3, 0.7), skill=rng.uniform(-0.02, 0.05),
            comfort=comfort, duo=rng.choice(_DUOS)))
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    for spec in make_specs(args.players, args.seed):
        ms = simulate(spec)
        ranked = [m for m in ms if m["mode"] == "排位"]
        print(f"\n{spec.id}: {spec.position} mains={[h for _, h in spec.phases]} matches={len(ms)} "
              f"ranked={len(ranked)} wr={sum(m['win'] for m in ranked) / len(ranked):.2f} "
              f"start={rank_name(spec.start_points)} end={rank_name(ms[-1]['rank_after'])} "
              f"peak={rank_name(max(m['rank_after'] for m in ms))}")
        if args.show:
            for e in timeline(ms):
                print("  ", e["date"], e["kind"], e["text"])
            print("  sample:", match_line(ms[0]))


if __name__ == "__main__":
    main()
