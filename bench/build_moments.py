# -*- coding: utf-8 -*-
"""Build e2e_g2: does the companion bring up the right memory at the right game moment?

A general memory system answers when asked. A game companion also has to
speak first: right after a match, when something the player said before
applies to what just happened ("你不是说再也不玩这英雄了吗", "上钻石了，说好的日料
呢", "已经连输三把了，你说过这时候就下线"). Nobody asks; the cue is a game event.
In psychology this is event-based prospective memory: an intention tied to a
future cue.

Each player has six months of simulated matches and about 12 chat sessions.
Some sessions carry a planted *intention* tied to a game cue; the moment the
cue next happens in the match data is a positive moment, where the companion
should bring it up. Other moments are negatives, where it should say nothing.

Intentions (positives):
    hero_quit    after a bad game, the player swears off the hero      cue: next game on that hero
    hero_tip     after a bad game, a lesson to remember for the hero   cue: next game on that hero
    rank_goal    "上{tier}就…"                                          cue: the game that enters the tier
    streak_rule  "以后一晚上连输{n}把就…"                               cue: the next evening that reaches n
    duo_note     something to remember for playing with a friend       cue: next game queued with that friend
For hero intentions the plan says whether the player names the hero or only
says "这英雄 / 这把" (the companion saw the match); half each.

Negatives: the opponent's hero ("对面的X太恶心") when the player next plays X;
a friend's goal ("阿杰说他上星耀就请客") when the player enters that tier;
promotions inside a tier before the goal tier; evenings that stop at two
losses after a three-loss rule; games on heroes never talked about. A
second game on a hero after its intention was due is reported apart
("repeat"): the reminder belonged to the first one.

    python -m bench.build_moments plan --out <dir>
    (an author who does not read the code writes chats.json from plan.json)
    python -m bench.build_moments assemble --dir <dir> --out bench/data/e2e_g2.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bench.gamesim import END, make_specs, simulate
from gamememo.game.ledger import GameLedger, entered_tier, night_of
from gamememo.game.ontology import TIER_NAMES, rank_name, tier_of

N_DEV, N_TEST = 2, 8
SEED = 11
N_SESSIONS = 12
FIRST_DAY = "2026-03-20"
FRIENDS = ["阿杰", "小周", "老三", "胖虎", "表弟", "室友", "阿伟", "大刘", "小林", "浩子"]


def _players():
    """The first players (in seed order) whose six months hold every planted cue."""
    out = []
    for s in make_specs(40, seed=SEED, prefix="g"):
        if not s.duo:  # every player has a friend they sometimes queue with
            s.duo = FRIENDS[int(s.id[1:]) % len(FRIENDS)]
        try:
            Planner(s).plan()
        except PlanError:
            continue
        out.append(s)
        if len(out) == N_DEV + N_TEST:
            break
    for i, s in enumerate(out):
        s.id = f"g2_d{i + 1}" if i < N_DEV else f"g2_t{i - N_DEV + 1}"
    return out


class PlanError(Exception):
    pass


def _end(m: Dict) -> datetime:
    return datetime.fromisoformat(m["time"]) + timedelta(minutes=18)


def _chat_time(m: Dict, rng: random.Random) -> str:
    return (_end(m) + timedelta(minutes=rng.randrange(3, 25))).strftime("%Y-%m-%d %H:%M")


def _last_of_night(ms: List[Dict], m: Dict) -> bool:
    later = [x for x in ms if x["time"] > m["time"]]
    return not later or night_of(later[0]["time"]) != night_of(m["time"])


def _after(ms: List[Dict], t: str) -> List[Dict]:
    return [x for x in ms if x["time"] > t]


class Planner:
    """Picks the planted intentions for one player, each on its own evening."""

    def __init__(self, spec):
        self.spec = spec
        self.ms = simulate(spec)
        self.led = GameLedger(self.ms)
        self.rng = random.Random(spec.seed + 99)
        self.nights: set = set()
        self.heroes: set = set()
        self.cues: set = set()  # match times used as moments
        self.items: List[Dict] = []

    def free(self, m: Dict) -> bool:
        return night_of(m["time"]) not in self.nights and m["time"][:10] >= FIRST_DAY

    def take(self, item: Dict, anchor: Dict, cue: Optional[Dict]) -> None:
        self.nights.add(night_of(anchor["time"]))
        if cue:
            self.cues.add(cue["time"])
        self.items.append(item)

    def next_on_hero(self, t: str, hero: str) -> List[Dict]:
        return [x for x in _after(self.ms, t) if x["hero"] == hero]

    def hero_item(self, kind: str, idx: int, named: bool) -> bool:
        cands = []
        for m in self.ms:
            if m["win"] or m["d"] < 5 or not self.free(m) or m["hero"] in self.heroes or not _last_of_night(self.ms, m):
                continue
            later = self.next_on_hero(m["time"], m["hero"])
            if len(later) < 2 or night_of(later[0]["time"]) == night_of(m["time"]) or later[0]["time"] in self.cues:
                continue
            if later[0]["time"] > END.isoformat() or later[0]["time"][:10] > "2026-09-28":
                continue
            cands.append((m, later))
        if not cands:
            return False
        m, later = self.rng.choice(cands)
        at = _chat_time(m, self.rng)
        self.heroes.add(m["hero"])
        what = {"hero_quit": "这局打得很崩，玩家发誓以后再也不用这个英雄了（或者说很长时间都不碰了）。助手回应。",
                "hero_tip": "玩家总结了一条这个英雄的教训（具体是什么你来定，比如出装、对线、什么时候进场），"
                            "让助手下次他用这个英雄时提醒他。助手答应。"}[kind]
        naming = (f"玩家说话时要说出英雄名“{m['hero']}”。" if named else
                  f"玩家刚打完这一局，知道助手看得到刚才的对局，所以说话时不要说出英雄名，用“这英雄”“这把”之类代替；"
                  f"助手的回复也不要说出英雄名。")
        self.take({"id": f"{kind}{idx}", "kind": kind, "polarity": "positive", "chat_at": at, "hero": m["hero"],
                   "named": named, "anchor": m["time"], "cue": later[0]["time"], "repeat": later[1]["time"],
                   "instruction": f"刚打完的这局：{m['hero']}（{m['position']}），{m['mode']}，输了，"
                                  f"{m['k']}/{m['d']}/{m['a']}。{what}{naming}"}, m, later[0])
        return True

    def rank_goal(self) -> bool:
        cands = []
        for c in self.ms:
            tier = entered_tier(c)
            if not tier or c["time"] in self.cues or c["time"][:10] > "2026-09-28":
                continue
            for m in self.ms:
                gap = (datetime.fromisoformat(c["time"]) - datetime.fromisoformat(m["time"])).days
                if not (7 <= gap <= 45) or not self.free(m) or not _last_of_night(self.ms, m):
                    continue
                if TIER_NAMES.index(tier_of(m["rank_after"])) >= TIER_NAMES.index(tier):
                    continue
                first = next((x for x in _after(self.ms, m["time"]) if entered_tier(x) == tier), None)
                if first is c:
                    cands.append((m, c, tier))
        if not cands:
            return False
        m, c, tier = self.rng.choice(cands)
        near = [x for x in self.ms if m["time"] < x["time"] < c["time"] and x["mode"] == "排位" and not entered_tier(x)
                and rank_name(x["rank_after"], stars=False) != rank_name(x["rank_before"], stars=False)
                and x["rank_after"] > x["rank_before"]]
        self.take({"id": "rank_goal1", "kind": "rank_goal", "polarity": "positive", "chat_at": _chat_time(m, self.rng),
                   "tier": tier, "anchor": m["time"], "cue": c["time"], "near": [x["time"] for x in near],
                   "instruction": f"玩家现在是{rank_name(m['rank_after'], stars=False)}。玩家给自己定了个目标：上{tier}就"
                                  f"（一件具体的奖励或计划，你来定，比如买某个皮肤、请谁吃某家店、去做某件事）。"
                                  f"说出段位名“{tier}”。助手回应。"}, m, c)
        return True

    def streak_rule(self, n: int = 3) -> bool:
        def reaches(m):
            return self.led.night_losses(m) == n
        cands = []
        for m in self.ms:
            if not reaches(m) or not self.free(m):
                continue
            last = [x for x in self.ms if night_of(x["time"]) == night_of(m["time"])][-1]
            at = _chat_time(last, self.rng)
            nxt = next((x for x in self.ms if x["time"] > at and reaches(x)), None)
            if nxt and nxt["time"][:10] <= "2026-09-28" and nxt["time"] not in self.cues:
                cands.append((m, last, at, nxt))
        if not cands:
            return False
        m, last, at, nxt = self.rng.choice(cands)
        # evenings that stop at two losses in a row (rule not reached) before the cue
        two = [x for x in self.ms if at < x["time"] < nxt["time"] and self.led.night_losses(x) == n - 1
               and not any(self.led.night_losses(y) >= n for y in self.ms
                           if night_of(y["time"]) == night_of(x["time"]) and y["time"] > x["time"])]
        nightly = [x for x in self.ms if night_of(x["time"]) == night_of(m["time"])]
        w = sum(x["win"] for x in nightly)
        self.take({"id": "streak_rule1", "kind": "streak_rule", "polarity": "positive", "chat_at": at, "n": n,
                   "anchor": last["time"], "cue": nxt["time"], "short": [x["time"] for x in two],
                   "instruction": f"今晚打了{len(nightly)}局，{w}胜{len(nightly) - w}负，中间排位连输了{n}把。玩家给自己定了个规矩："
                                  f"以后一晚上排位连输{n}把就（下线 / 去睡觉 / 去做别的，你来定），让助手到时候提醒他。助手答应。"},
                  last, nxt)
        return True

    def duo_note(self) -> bool:
        p = self.spec.duo
        duos = [m for m in self.ms if m["duo"] == p]
        cands = []
        for m in duos:
            night = [x for x in self.ms if night_of(x["time"]) == night_of(m["time"])]
            last = night[-1]
            if not self.free(m):
                continue
            at = _chat_time(last, self.rng)
            nxt = next((x for x in duos if x["time"] > at), None)
            if nxt and night_of(nxt["time"]) != night_of(m["time"]) and nxt["time"][:10] <= "2026-09-28" \
                    and nxt["time"] not in self.cues:
                cands.append((m, last, at, nxt))
        if not cands:
            return False
        m, last, at, nxt = self.rng.choice(cands)
        self.take({"id": "duo_note1", "kind": "duo_note", "polarity": "positive", "chat_at": at, "partner": p,
                   "anchor": last["time"], "cue": nxt["time"],
                   "instruction": f"今晚和朋友{p}双排了。玩家说了一件下次跟{p}双排时要注意或者约好的事（你来定，比如先说好分路、"
                                  f"谁不许玩什么、输了不许甩锅），让助手下次他们双排时提醒。助手答应。"}, last, nxt)
        return True

    def opp_hero(self) -> bool:
        cands = []
        for m in self.ms:
            if not self.free(m) or not _last_of_night(self.ms, m):
                continue
            seen = {x["hero"] for x in self.ms if x["time"] <= m["time"]}
            for h in sorted({x["hero"] for x in _after(self.ms, m["time"])} - self.heroes):
                nxt = self.next_on_hero(m["time"], h)
                if nxt and night_of(nxt[0]["time"]) != night_of(m["time"]) and nxt[0]["time"] not in self.cues \
                        and nxt[0]["time"][:10] <= "2026-09-28" and h not in seen:
                    cands.append((m, h, nxt[0]))
        if not cands:
            return False
        m, h, nxt = self.rng.choice(cands)
        self.heroes.add(h)
        self.take({"id": "opp_hero1", "kind": "opp_hero", "polarity": "negative", "chat_at": _chat_time(m, self.rng),
                   "hero": h, "anchor": m["time"], "cue": nxt["time"],
                   "instruction": f"玩家吐槽今晚对面（对手）的{h}特别难打，具体怎么被针对你来定。要说清楚是对面的{h}；"
                                  f"玩家不要说自己会玩或想玩这个英雄。"}, m, nxt)
        return True

    def friend_goal(self, own: Optional[str]) -> bool:
        friend = next(f for f in self.rng.sample(FRIENDS, len(FRIENDS)) if f != self.spec.duo)
        cands = []
        for c in self.ms:
            tier = entered_tier(c)
            if not tier or c["time"] in self.cues or c["time"][:10] > "2026-09-28":
                continue
            for m in self.ms:
                gap = (datetime.fromisoformat(c["time"]) - datetime.fromisoformat(m["time"])).days
                if not (5 <= gap <= 45) or not self.free(m) or not _last_of_night(self.ms, m):
                    continue
                first = next((x for x in _after(self.ms, m["time"]) if entered_tier(x) == tier), None)
                if first is c:
                    cands.append((m, c, tier))
        if not cands:
            return False
        m, c, tier = self.rng.choice(cands)
        self.take({"id": "friend_goal1", "kind": "friend_goal", "polarity": "negative", "chat_at": _chat_time(m, self.rng),
                   "tier": tier, "friend": friend, "anchor": m["time"], "cue": c["time"],
                   "instruction": f"玩家顺口提到朋友{friend}：{friend}说他自己上{tier}就要（做件什么事，你来定）。"
                                  f"这是{friend}的目标，不是玩家的；玩家自己不要定目标。"}, m, c)
        return True

    def plan(self) -> Dict:
        named = self.rng.random() < 0.5
        ok = [self.hero_item("hero_quit", 1, named), self.hero_item("hero_tip", 1, not named),
              self.hero_item("hero_tip", 2, named), self.rank_goal(), self.streak_rule(), self.duo_note(),
              self.opp_hero(), self.friend_goal(None)]
        if not all(ok):
            raise PlanError(f"{self.spec.id}: {ok}")
        sessions = [{"chat_at": it["chat_at"], "items": [it["id"]]} for it in self.items]
        nights = sorted({night_of(m["time"]) for m in self.ms if m["time"][:10] >= FIRST_DAY} - self.nights)
        while len(sessions) < N_SESSIONS:
            day = self.rng.choice(nights)
            nights.remove(day)
            last = [m for m in self.ms if night_of(m["time"]) == day][-1]
            sessions.append({"chat_at": _chat_time(last, self.rng), "items": []})
        sessions.sort(key=lambda s: s["chat_at"])
        for s in sessions:
            before = [m for m in self.ms if m["time"] < s["chat_at"] and night_of(m["time"]) == night_of(s["chat_at"])]
            w = sum(m["win"] for m in before)
            rank = [m for m in self.ms if m["time"] < s["chat_at"] and m["mode"] == "排位"]
            s["evening"] = (f"今晚打了{len(before)}局，{w}胜{len(before) - w}负；现在段位"
                            f"{rank_name(rank[-1]['rank_after'], stars=False) if rank else '未定级'}")
        return {"id": self.spec.id, "friend_duo": self.spec.duo, "heroes_allowed": sorted(self.heroes),
                "sessions": sessions, "items": self.items, "matches": self.ms}


def plan(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    out = [Planner(spec).plan() for spec in _players()]
    for_author = []
    for p in out:
        items = {it["id"]: it for it in p["items"]}
        for_author.append({
            "id": p["id"], "friend_you_duo_with": p["friend_duo"], "heroes_you_may_name": p["heroes_allowed"],
            "sessions": [{"date": s["chat_at"], "evening": s["evening"],
                          "planted": [{"item": i, "instruction": items[i]["instruction"]} for i in s["items"]]}
                         for s in p["sessions"]]})
    with open(os.path.join(args.out, "plan_full.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in p.items() if k != "matches"} for p in out], f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(for_author, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} plans to {args.out}")
    for p in out:
        print(p["id"], Counter(it["kind"] for it in p["items"]), [s["chat_at"] for s in p["sessions"]][:3], "...")


# ---------------------------------------------------------------- moments

def moments(p: Dict, ms: List[Dict], rng: random.Random) -> List[Dict]:
    by_time = {m["time"]: m for m in ms}
    out = []

    def add(t, category, gold, item=None):
        if any(x["match_time"] == t for x in out):
            return
        out.append({"match_time": t, "category": category, "gold": gold, "item": item})

    first_chat = min(s["chat_at"] for s in p["sessions"])
    mentioned = set(p["heroes_allowed"])
    for it in p["items"]:  # positives first: a negative never displaces a cue
        if it["polarity"] == "positive":
            add(it["cue"], it["kind"], [it["id"]], it["id"])
    for it in p["items"]:
        if it["polarity"] == "positive":
            if it.get("repeat"):
                add(it["repeat"], "repeat", [], it["id"])
            for t in rng.sample(it.get("near", []), min(2, len(it.get("near", [])))):
                add(t, "near_rank", [], it["id"])
            for t in rng.sample(it.get("short", []), min(2, len(it.get("short", [])))):
                add(t, "short_streak", [], it["id"])
        else:
            add(it["cue"], it["kind"], [], it["id"])
    used = {x["match_time"] for x in out}
    led = GameLedger(ms)
    pool = [m for m in ms if m["time"] > first_chat and m["time"] not in used and m["hero"] not in mentioned
            and not entered_tier(m) and led.night_losses(m) < 2 and m["time"][:10] <= "2026-09-28"
            and m["duo"] is None]
    for m in rng.sample(pool, 4):
        add(m["time"], "random", [])
    for x in out:
        x["situation"] = led.situation(by_time[x["match_time"]])
        x["now"] = _end(by_time[x["match_time"]]).strftime("%Y-%m-%d %H:%M:%S")
    return sorted(out, key=lambda x: x["match_time"])


def assemble(args) -> None:
    with open(os.path.join(args.dir, "plan_full.json"), encoding="utf-8") as f:
        plans = {p["id"]: p for p in json.load(f)}
    with open(os.path.join(args.dir, "chats.json"), encoding="utf-8") as f:
        chats = {p["id"]: p for p in json.load(f)["players"]}
    players = []
    for spec in _players():
        if spec.id not in chats:
            continue
        p, chat = plans[spec.id], chats[spec.id]
        keys = chat.get("items", {})
        ms = simulate(spec)
        mo = moments(p, ms, random.Random(spec.seed + 5))
        for i, x in enumerate(mo):
            x["id"] = f"{spec.id}_m{i + 1:02d}"
            x["answer_any"] = sorted({k for g in x["gold"] for k in keys.get(g, {}).get("answer_any", [])})
        sessions = [{"date": s["date"] + (":00" if len(s["date"]) == 16 else ""), "turns": s["turns"]}
                    for s in chat["sessions"]]
        players.append({"id": spec.id, "split": "dev" if "_d" in spec.id else "test", "persona": chat.get("persona", ""),
                        "sessions": sessions, "matches": ms, "items": p["items"], "moments": mo})
    out = {"name": "e2e_g2", "version": 1,
           "description": "proactive recall at game moments: intentions planted in chat, cues from simulated matches",
           "players": players}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    n = Counter(x["category"] for p in players for x in p["moments"])
    print(f"{args.out}: {len(players)} players, {sum(n.values())} moments {dict(n)}")


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
