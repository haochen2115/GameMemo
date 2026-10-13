# -*- coding: utf-8 -*-
"""Intentions tied to game cues: what the companion should bring up, and when.

A general memory system is retrospective: it answers when asked. A game
companion also needs event-based prospective memory: the player says "这英雄
我再也不碰了", "上星耀就去吃那家日料", "以后连输三把你就让我下线", and the right
time to bring it up is a game event (the next game on that hero, the game that
enters 星耀, the third loss of an evening), when nobody asks anything.

Three things a general memory cannot do here, and this module does:

1. Ground what was said in the match it was said about. Chat right after a
   game is deictic: the companion saw the match, so the player says "这英雄",
   "这把". The exchange is aligned to the match timeline when it is written,
   so "这英雄" is resolved to the hero just played.
2. Turn an intention into a cue the match stream can fire, using the game's
   own structure: a hero the player plays (not the opponent's "对面的韩信"), a
   tier on the ladder (the player's goal, not a friend's), a count of losses
   in one evening, a friend queued with.
3. Fire on the first occurrence since it was said. The match record says
   whether the player has already played that hero since, so a reminder is
   due once, not every game.

Nothing here calls an LLM; nothing said is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .ledger import GameLedger, entered_tier, night_of
from .ontology import TIER_NAMES, heroes_in, parse_ranks

# a forward-looking statement: a vow, a goal, a rule, a request to be reminded
INTENT = re.compile(r"再也不|再也|不再|以后|今后|下次|下回|下把|提醒|记得|记住|别让我|一定要|一定得|目标|发誓|戒了|说好|约好|"
                    r"不碰|不玩了|不用了|得先|要先|先说好|就(去|请|买|给|吃|换|下线|睡|休息|关|退|停|奖励|犒劳)|等我|要是|如果")
# "这英雄 / 这把": the player points at the match the companion just saw
DEICTIC = re.compile(r"这英雄|这个英雄|这角色|这个角色|这把|这局|刚才那|刚那|这玩意|这货|这个位置|这位置|英雄")
# a hero named as the opponent's, not the player's
_OPPONENT = re.compile(r"(对面|对方|敌方|对手|人家|他们|被)[^，。！？,!?\s]{0,3}$")
# a goal said on someone else's behalf ("阿杰说他上星耀就请客")
_OTHERS = re.compile(r"说他|说她|他说|她说|他自己|她自己|他要|她要|他上|她上")
_STREAK = re.compile(r"连(输|跪|败|掉)\s*([0-9一二两三四五六七八九十]+)?\s*(把|局|场|次)?")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_CLAUSE = re.compile(r"[。！？!?；;\n]")


@dataclass
class Intention:
    record_id: str
    said_at: str          # "YYYY-MM-DD HH:MM"
    cue: str              # hero | tier | streak | duo
    key: object           # hero name, tier name, loss count, friend name
    grounding: str        # the match the exchange was said right after, as a line
    content: str          # the exchange, verbatim


def _player_lines(content: str) -> str:
    return "\n".join(l.split(":", 1)[1].strip() for l in content.splitlines() if l.startswith("玩家"))


def _assistant_lines(content: str) -> str:
    return "\n".join(l.split(":", 1)[1].strip() for l in content.splitlines() if l.startswith("助手"))


def _self_heroes(text: str) -> List[str]:
    out = []
    for h in heroes_in(text):
        i = text.find(h)
        if i >= 0 and _OPPONENT.search(text[:i]):
            continue
        out.append(h)
    return out


def _streak_n(text: str) -> Optional[int]:
    m = _STREAK.search(text)
    if not m:
        return None
    s = m.group(2)
    if not s:
        return 3
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    return _CN_NUM.get(s[-1]) if len(s) == 1 else None


_GOAL_TIER = re.compile(r"(上|到|冲|进|升)了?\s*(青铜|白银|黄金|铂金|钻石|星耀|王者)")


def _own_goal_tiers(text: str) -> List[str]:
    """Tiers the player sets out to reach ("上星耀就…"), not the one they are in ("我现在钻石二")."""
    tiers = []
    for clause in _CLAUSE.split(text):
        if _OTHERS.search(clause):
            continue
        for m in _GOAL_TIER.finditer(clause):
            if m.group(2) not in tiers:
                tiers.append(m.group(2))
    return tiers


_DUO = re.compile(r"双排|开黑|组队|一起打|一起排|一块打|一块排")


class IntentionIndex:
    """Exchanges that carry an intention, keyed by the game cue that should bring them back."""

    def __init__(self, ledger: GameLedger, ground: bool = True, first_only: bool = True, extractor=None):
        """extractor: None for the rules below, or an object with ``extract(content, anchor_line)``
        returning (cue, key) pairs, e.g. ``LLMIntentExtractor``. Grounding and triggering are the same."""
        self.ledger = ledger
        self.ground = ground
        self.first_only = first_only
        self.extractor = extractor
        self.items: List[Intention] = []

    # ---------------------------------------------------------------- write

    def anchor(self, said_at: str) -> Optional[Dict]:
        """The match an exchange was said right after: the evening's last game before it."""
        before = [m for m in self.ledger.matches if m["time"] < said_at]
        if not before or night_of(before[-1]["time"]) != night_of(said_at):
            return None
        return before[-1]

    def grounding(self, said_at: str) -> str:
        m = self.anchor(said_at)
        if not m:
            return f"{said_at}"
        res = "赢了" if m["win"] else "输了"
        night = self.ledger.night(m)
        w = sum(x["win"] for x in night)
        duo = f"，和{m['duo']}双排" if m["duo"] else ""
        return (f"{said_at}，刚用{m['hero']}（{m['position']}）打完一局{m['mode']}，{res}，{m['k']}/{m['d']}/{m['a']}{duo}；"
                f"那晚{w}胜{len(night) - w}负")

    def add(self, record_id: str, said_at: str, content: str) -> List[Intention]:
        if self.extractor is not None:
            return self._add_extracted(record_id, said_at, content)
        said, reply = _player_lines(content), _assistant_lines(content)
        if not INTENT.search(said) and not re.search(r"提醒|记着|记住", reply):
            return []
        found: List[Intention] = []
        g = self.grounding(said_at)

        def put(cue, key):
            if not any(i.cue == cue and i.key == key for i in found):
                found.append(Intention(record_id, said_at, cue, key, g, content))

        heroes = _self_heroes(said)
        for h in heroes:
            put("hero", h)
        if not heroes and DEICTIC.search(said):
            m = self.anchor(said_at)
            if m:
                put("hero", m["hero"])
        n = _streak_n(said)
        if n:
            put("streak", n)
        for t in _own_goal_tiers(said):
            put("tier", t)
        partners = {m["duo"] for m in self.ledger.matches if m["duo"] and m["time"] < said_at}
        named = [p for p in partners if p in said and not _OTHERS.search(said)]
        for p in named:
            put("duo", p)
        if not named and _DUO.search(said):
            # "下次双排他…": the friend is the one the player queued with tonight
            tonight = {m["duo"] for m in self.ledger.night(self.anchor(said_at))} if self.anchor(said_at) else set()
            for p in sorted(tonight - {None}):
                put("duo", p)
        if found and found[0].cue == "hero" and (n or any(i.cue in ("tier", "duo") for i in found)) \
                and not _self_heroes(said):
            found = [i for i in found if i.cue != "hero"]  # "这把" in a rule or a goal points at no hero
        self.items.extend(found)
        return found

    def _add_extracted(self, record_id: str, said_at: str, content: str) -> List[Intention]:
        """Cues proposed by an extractor, resolved against the match timeline and checked by the ontology."""
        m = self.anchor(said_at)
        anchor_line = self.grounding(said_at) if m else "（这段对话之前当晚没有对局）"
        found: List[Intention] = []
        g = self.grounding(said_at)
        partners = {x["duo"] for x in self.ledger.matches if x["duo"] and x["time"] < said_at}
        tonight = {x["duo"] for x in self.ledger.night(m)} - {None} if m else set()
        for cue, key in self.extractor.extract(content, anchor_line):
            key = str(key).strip()
            if cue == "hero":
                named = heroes_in(key)
                key = named[0] if named else (m["hero"] if m else None)
            elif cue == "tier":
                key = next((t for t in TIER_NAMES if t in key), None)
            elif cue == "streak":
                key = _streak_n(f"连输{key}把") if key else None
            elif cue == "duo":
                key = next((p for p in partners if p in key), None) or (sorted(tonight)[0] if tonight else None)
            else:
                key = None
            if key and not any(i.cue == cue and i.key == key for i in found):
                found.append(Intention(record_id, said_at, cue, key, g, content))
        self.items.extend(found)
        return found

    # ---------------------------------------------------------------- read

    def _between(self, said_at: str, m: Dict) -> List[Dict]:
        return [x for x in self.ledger.matches if said_at < x["time"] < m["time"]]

    def fires(self, it: Intention, m: Dict) -> bool:
        if it.said_at >= m["time"]:
            return False
        if it.cue == "hero":
            hit, again = m["hero"] == it.key, lambda x: x["hero"] == it.key
        elif it.cue == "tier":
            hit, again = entered_tier(m) == it.key, lambda x: entered_tier(x) == it.key
        elif it.cue == "streak":
            hit = self.ledger.night_losses(m) == it.key
            again = lambda x: self.ledger.night_losses(x) == it.key  # noqa: E731
        else:
            hit, again = m["duo"] == it.key, lambda x: x["duo"] == it.key
        if not hit:
            return False
        return not (self.first_only and any(again(x) for x in self._between(it.said_at, m)))

    def due(self, m: Dict) -> List[Intention]:
        return [it for it in self.items if self.fires(it, m)]

    @staticmethod
    def cue_note(it: Intention) -> str:
        return {"hero": f"这是那次之后他第一次再用{it.key}",
                "tier": f"这是那次之后他第一次升到{it.key}",
                "streak": f"这是那次之后他第一次一晚上排位连输{it.key}把",
                "duo": f"这是那次之后他第一次和{it.key}双排"}[it.cue]

    def describe(self, it: Intention, ground: Optional[bool] = None) -> str:
        ground = self.ground if ground is None else ground
        text = it.content.replace("\n", " ")
        if not ground:
            return f"【{it.said_at}】{text}"
        return f"【{it.grounding}】{text}（{self.cue_note(it)}）" if self.first_only else f"【{it.grounding}】{text}"


# ---------------------------------------------------------------- an LLM in place of the intent rules

EXTRACT_SYSTEM = "你从游戏陪伴的聊天里找出玩家的意图，只根据对话内容，不编造。严格输出 JSON。"
EXTRACT_PROMPT = """玩家刚打完的那局：{anchor}
这段对话：
{content}

这段对话里，玩家有没有说出一个和以后某个游戏情境有关的打算？包括：发誓以后不用某个英雄；某个英雄的教训、下次用它要注意什么；上某个段位就做某件事；一晚上排位连输几把就怎样；下次和某个朋友双排要注意或约好的事。
只算玩家自己的打算：朋友的目标不算，吐槽对面（对手）的英雄不算，只说现在的情况也不算。
如果有，写出会让这件事该被想起的游戏情境：
- cue 为 hero：玩家下次用某个英雄时。key 是英雄名；玩家只说"这英雄 / 这把"时，key 写"刚才那局的英雄"。
- cue 为 tier：玩家升到某个段位时。key 是段位档位（青铜、白银、黄金、铂金、钻石、星耀、王者之一）。
- cue 为 streak：玩家一晚上排位连输 N 把时。key 是 N。
- cue 为 duo：玩家下次和某个朋友双排时。key 是朋友的名字；没点名就写"今晚一起双排的朋友"。
没有就输出空列表。"""
EXTRACT_SCHEMA = {"type": "object", "properties": {"intentions": {"type": "array", "items": {
    "type": "object", "properties": {"cue": {"type": "string", "enum": ["hero", "tier", "streak", "duo"]},
                                     "key": {"type": "string"}}, "required": ["cue", "key"]}}},
    "required": ["intentions"]}


class LLMIntentExtractor:
    """One LLM call per exchange decides whether it carries an intention and what its cue is.
    The index then resolves the cue against the match timeline and the game ontology as usual."""

    def __init__(self, llm):
        self.llm = llm

    def extract(self, content: str, anchor_line: str):
        from ..llm import parse_json
        raw = self.llm.chat(system=EXTRACT_SYSTEM, temperature=0.0, max_tokens=200, json_schema=EXTRACT_SCHEMA,
                            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(anchor=anchor_line, content=content)}])
        data = parse_json(raw)
        items = data.get("intentions", []) if isinstance(data, dict) else []
        return [(i.get("cue"), i.get("key", "")) for i in items if isinstance(i, dict)]
