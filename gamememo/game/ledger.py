# -*- coding: utf-8 -*-
"""The player's game state, computed exactly from match records.

A general memory system only sees chat, and chat is a poor source for game
facts: players round up ("我早就王者了"), forget ("我孙尚香胜率挺高的吧"), and a
memory system cannot count matches it was never told about. The ledger keeps
every match and answers game questions by computation: current and peak rank,
per-season results, per-hero records, streaks, and the days things happened.

It also gives chat a game-time axis: "我第一次上钻石那天" or "连输最多那次" is
resolved to a date here, and the chat of that day can then be recalled.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from .ontology import (POSITIONS, SEASONS, TIER_NAMES, Season, heroes_in, previous_season, rank_name, season_of,
                       tier_of)


@dataclass
class Record:
    games: int = 0
    wins: int = 0
    k: int = 0
    d: int = 0
    a: int = 0
    mvp: int = 0

    def add(self, m: Dict) -> None:
        self.games += 1
        self.wins += bool(m["win"])
        self.k += m["k"]
        self.d += m["d"]
        self.a += m["a"]
        self.mvp += bool(m["mvp"])

    @property
    def winrate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def text(self) -> str:
        if not self.games:
            return "0场"
        g = self.games
        return (f"{g}场，胜率{round(100 * self.winrate)}%（{self.wins}胜{g - self.wins}负），"
                f"场均{self.k / g:.1f}/{self.d / g:.1f}/{self.a / g:.1f}，MVP{self.mvp}次")


class GameLedger:
    def __init__(self, matches: Iterable[Dict] = ()):
        self.matches: List[Dict] = []
        self.extend(matches)

    def extend(self, matches: Iterable[Dict]) -> None:
        self.matches.extend(matches)
        self.matches.sort(key=lambda m: m["time"])

    # ---------------------------------------------------------------- queries

    def until(self, now: datetime) -> List[Dict]:
        stamp = now.strftime("%Y-%m-%d %H:%M")
        return [m for m in self.matches if m["time"] <= stamp]

    def ranked(self, now: datetime, season: Optional[Season] = None) -> List[Dict]:
        return [m for m in self.until(now) if m["mode"] == "排位" and (season is None or m["season"] == season.name)]

    def current_rank(self, now: datetime) -> Optional[Tuple[int, str]]:
        r = self.ranked(now)
        if not r:
            return None
        return r[-1]["rank_after"], r[-1]["time"][:10]

    def peak(self, now: datetime, season: Optional[Season] = None) -> Optional[Tuple[int, str]]:
        r = self.ranked(now, season)
        if not r:
            return None
        best = max(r, key=lambda m: (m["rank_after"], [-ord(c) for c in m["time"]]))
        return best["rank_after"], best["time"][:10]

    def record(self, now: datetime, hero: Optional[str] = None, season: Optional[Season] = None,
               since: Optional[date] = None, ranked_only: bool = False) -> Record:
        rec = Record()
        for m in self.until(now):
            if hero and m["hero"] != hero:
                continue
            if season and m["season"] != season.name:
                continue
            if since and m["time"][:10] < since.isoformat():
                continue
            if ranked_only and m["mode"] != "排位":
                continue
            rec.add(m)
        return rec

    def heroes(self, now: datetime, since: Optional[date] = None) -> List[Tuple[str, int]]:
        c = Counter(m["hero"] for m in self.until(now) if not since or m["time"][:10] >= since.isoformat())
        return c.most_common()

    def positions(self, now: datetime) -> List[Tuple[str, float]]:
        ms = self.until(now)
        c = Counter(m["position"] for m in ms)
        return [(p, n / len(ms)) for p, n in c.most_common()] if ms else []

    def losing_streak(self, now: datetime) -> Tuple[int, Optional[str]]:
        best, cur, day = 0, 0, None
        for m in self.ranked(now):
            cur = 0 if m["win"] else cur + 1
            if cur > best:
                best, day = cur, m["time"][:10]
        return best, day

    def first_reached(self, now: datetime) -> List[Tuple[str, str]]:
        """(tier, date) the first time each higher tier was reached in the data."""
        out, best = [], None
        for m in self.ranked(now):
            t = TIER_NAMES.index(tier_of(m["rank_after"]))
            if best is not None and t > best:
                out.append((TIER_NAMES[t], m["time"][:10]))
            best = t if best is None else max(best, t)
        return out

    def days_played(self, now: datetime) -> Dict[str, Record]:
        out: Dict[str, Record] = defaultdict(Record)
        for m in self.until(now):
            out[m["time"][:10]].add(m)
        return out

    # ---------------------------------------------------------------- what goes in the prompt

    def card(self, now: datetime) -> str:
        """The player's game profile, independent of the question."""
        if not self.until(now):
            return ""
        lines = [f"【游戏数据（系统记录，截至{now:%Y-%m-%d}）】"]
        cur = self.current_rank(now)
        if cur:
            lines.append(f"当前段位：{rank_name(cur[0])}（{cur[1]}最后一局排位后）")
        season = season_of(now)
        for label, s in (("本赛季", season), ("上赛季", previous_season(season))):
            if s is None:
                continue
            rec = self.record(now, season=s, ranked_only=True)
            if rec.games:
                pk = self.peak(now, s)
                lines.append(f"{label}最高段位：{rank_name(pk[0])}（{pk[1]}达到；{s.name}赛季{s.start}至{s.end}）")
                lines.append(f"{label}排位：{rec.games}场，胜率{round(100 * rec.winrate)}%")
        pk = self.peak(now)
        lines.append(f"有记录以来最高段位：{rank_name(pk[0])}（{pk[1]}）")
        top = self.heroes(now)[:3]
        lines.append("最常用英雄：" + "；".join(f"{h} {self.record(now, hero=h).text()}" for h, _ in top))
        recent = self.heroes(now, since=(now - timedelta(days=30)).date())[:2]
        if recent:
            lines.append("最近30天用得最多：" + "、".join(f"{h}（{n}场）" for h, n in recent))
        pos = self.positions(now)[:2]
        lines.append("位置：" + "、".join(f"{p}{round(100 * x)}%" for p, x in pos))
        streak, day = self.losing_streak(now)
        if streak:
            lines.append(f"最长排位连败：{streak}把（{day}）")
        reached = self.first_reached(now)
        if reached:
            lines.append("首次升到：" + "、".join(f"{t}（{d}）" for t, d in reached))
        return "\n".join(lines)

    def views(self, query: str, now: datetime) -> List[str]:
        """Exact records for the heroes a question names."""
        out = []
        for h in heroes_in(query):
            rec = self.record(now, hero=h)
            if not rec.games:
                out.append(f"{h}：没有对局记录")
                continue
            per = []
            for s in SEASONS:
                r = self.record(now, hero=h, season=s)
                if r.games:
                    per.append(f"{s.name} {r.games}场胜率{round(100 * r.winrate)}%")
            used = [m["time"][:10] for m in self.until(now) if m["hero"] == h]
            out.append(f"{h}：{rec.text()}；{'，'.join(per)}；第一次用{used[0]}，最近一次{used[-1]}")
        return out

    def rank_on(self, day: str) -> Optional[Tuple[int, int]]:
        """(lowest, highest) rank the player held on a day, from ranked games."""
        r = [m for m in self.matches if m["mode"] == "排位" and m["time"][:10] == day]
        if not r:
            before = [m for m in self.matches if m["mode"] == "排位" and m["time"][:10] < day]
            return (before[-1]["rank_after"],) * 2 if before else None
        vals = [m["rank_before"] for m in r] + [m["rank_after"] for m in r]
        return min(vals), max(vals)

    def check_claim(self, said: str, day: str) -> Optional[str]:
        """A note when a rank the player claims on ``day`` is above the record."""
        from .ontology import TIER_NAMES, parse_ranks
        claims = parse_ranks(said)
        if not claims:
            return None
        when = datetime.fromisoformat(day + " 23:59")
        if "上赛季" in said or "上个赛季" in said:
            s = previous_season(season_of(when))
            pk = self.peak(when, s) if s else None
            if not pk:
                return None
            truth, label = pk[0], f"上赛季（{s.name}）最高其实是{rank_name(pk[0], stars=False)}"
        else:
            pk = self.peak(when)
            if not pk:
                return None
            truth, label = pk[0], f"到那天为止最高其实是{rank_name(pk[0], stars=False)}"
        top = max(TIER_NAMES.index(t) for t, _ in claims)
        if top > TIER_NAMES.index(tier_of(truth)):
            return f"（与系统记录不符：{label}）"
        return None

    def event_dates(self, query: str, now: datetime) -> List[str]:
        """Dates a question anchors to by a game event, e.g. 第一次上钻石那天 / 连输最多那次."""
        dates = []
        for tier, day in self.first_reached(now):
            if tier in query and any(w in query for w in ("上", "到", "升", "打到")):
                dates.append(day)
        if any(w in query for w in ("连输", "连败", "连跪")):
            _, day = self.losing_streak(now)
            if day:
                dates.append(day)
        for s in SEASONS:
            if s.name in query or ("赛季末" in query or "赛季结束" in query) and s.end <= now.date() < s.end + timedelta(days=90):
                dates.append(s.end.isoformat())
        return sorted(set(dates))
