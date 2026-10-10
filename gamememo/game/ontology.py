# -*- coding: utf-8 -*-
"""What a 王者荣耀 companion knows about the game itself.

The game world is closed and structured: ranks are an ordered ladder of tiers,
sub-levels and stars; seasons cut time into named periods; every hero has a
lane. Questions about the player's game life ("这赛季最高打到哪", "我孙尚香胜率
多少") can therefore be answered exactly from match data instead of being
guessed from chat. This module is that shared vocabulary; it holds no player
data.

Simplifications (documented, used consistently by the simulator and the
memory): a win is +1 star and a loss -1 star in ranked games; a season reset
drops a player by ``SEASON_DROP`` stars, not below 黄金四.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

# (tier, sub-levels, stars per sub-level). Sub-levels count down: 黄金四 < 黄金一.
TIERS: Tuple[Tuple[str, int, int], ...] = (
    ("青铜", 3, 3), ("白银", 3, 3), ("黄金", 4, 4), ("铂金", 4, 4), ("钻石", 5, 5), ("星耀", 5, 5))
TOP_TIER = "王者"
TIER_NAMES = tuple(t for t, _, _ in TIERS) + (TOP_TIER,)
_NUM = "一二三四五"
_NUM_VALUE = {c: i + 1 for i, c in enumerate(_NUM)}

# cumulative star index where each (tier, sub-level) starts
_LADDER: List[Tuple[str, int, int, int]] = []  # (tier, sub, start, stars)
_pos = 0
for _tier, _subs, _stars in TIERS:
    for _sub in range(_subs, 0, -1):
        _LADDER.append((_tier, _sub, _pos, _stars))
        _pos += _stars
KING_START = _pos  # 王者0星
SEASON_DROP = 12
RESET_FLOOR = next(start for tier, sub, start, _ in _LADDER if tier == "黄金" and sub == 4)


def rank_name(points: int, stars: bool = True) -> str:
    """铂金二3星 / 王者12星 (``stars=False``: 铂金二 / 王者)."""
    if points >= KING_START:
        return f"{TOP_TIER}{points - KING_START}星" if stars else TOP_TIER
    for tier, sub, start, n in reversed(_LADDER):
        if points >= start:
            name = f"{tier}{_NUM[sub - 1]}"
            return f"{name}{points - start + 1}星" if stars else name
    return "青铜三1星" if stars else "青铜三"


def tier_of(points: int) -> str:
    return TOP_TIER if points >= KING_START else rank_name(points, stars=False)[:2]


def level_start(tier: str, sub: Optional[int] = None) -> int:
    """Star index where 铂金二 (or 铂金, at its lowest sub-level) starts."""
    if tier == TOP_TIER:
        return KING_START
    subs = [(s, start) for t, s, start, _ in _LADDER if t == tier]
    if sub is None:
        return min(start for _, start in subs)
    return next(start for s, start in subs if s == sub)


_RANK_TEXT = re.compile(r"(青铜|白银|黄金|铂金|钻石|星耀)\s*([一二三四五1-5]|IV|V|I{1,3})?|(?<![打玩])王者")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


def parse_ranks(text: str) -> List[Tuple[str, Optional[int]]]:
    """(tier, sub) pairs named in text: "上铂金2了" -> [("铂金", 2)]."""
    out = []
    for m in _RANK_TEXT.finditer(text):
        if m.group(1):
            s = m.group(2)
            sub = None if not s else (_NUM_VALUE.get(s) or _ROMAN.get(s) or int(s))
            out.append((m.group(1), sub))
        else:
            out.append((TOP_TIER, None))
    return out


# ---------------------------------------------------------------- seasons

@dataclass(frozen=True)
class Season:
    name: str
    start: date
    end: date  # inclusive


SEASONS: Tuple[Season, ...] = (
    Season("S38", date(2025, 10, 9), date(2026, 1, 7)),
    Season("S39", date(2026, 1, 8), date(2026, 4, 8)),
    Season("S40", date(2026, 4, 9), date(2026, 7, 8)),
    Season("S41", date(2026, 7, 9), date(2026, 10, 14)),
    Season("S42", date(2026, 10, 15), date(2027, 1, 13)),
)


def season_of(when) -> Season:
    d = when.date() if isinstance(when, datetime) else when
    for s in SEASONS:
        if s.start <= d <= s.end:
            return s
    raise ValueError(f"no season covers {d}")


def previous_season(s: Season) -> Optional[Season]:
    i = SEASONS.index(s)
    return SEASONS[i - 1] if i else None


# ---------------------------------------------------------------- lanes and heroes

POSITIONS = ("对抗路", "打野", "中路", "发育路", "游走")
POSITION_ALIASES = {"对抗路": ("对抗路", "上单", "边路"), "打野": ("打野",), "中路": ("中路", "中单", "法师位"),
                    "发育路": ("发育路", "射手", "下路", "ADC", "adc"), "游走": ("游走", "辅助")}

HERO_POSITION: Dict[str, str] = {}
for _p, _names in {
    "发育路": "鲁班七号 孙尚香 后羿 虞姬 狄仁杰 李元芳 马可波罗 公孙离 黄忠 百里守约 伽罗 蒙犽 艾琳 戈娅",
    "中路": "妲己 安琪拉 小乔 王昭君 貂蝉 甄姬 武则天 诸葛亮 干将莫邪 上官婉儿 嫦娥 西施 沈梦溪 不知火舞 米莱狄 女娲 周瑜 高渐离 张良 嬴政 海月",
    "打野": "李白 韩信 赵云 娜可露露 兰陵王 阿轲 百里玄策 孙悟空 镜 澜 裴擒虎 云中君 暃 典韦 曜 镜",
    "对抗路": "亚瑟 吕布 关羽 花木兰 马超 老夫子 狂铁 夏侯惇 曹操 杨戬 李信 蒙恬 刘备 铠 程咬金 夏洛特",
    "游走": "张飞 蔡文姬 大乔 孙膑 明世隐 瑶 鬼谷子 庄周 刘禅 钟馗 盾山 牛魔 太乙真人 东皇太一 廉颇 鲁班大师 少司缘 朵莉亚",
}.items():
    for _h in _names.split():
        HERO_POSITION.setdefault(_h, _p)
HEROES_BY_LENGTH = tuple(sorted(HERO_POSITION, key=len, reverse=True))


def heroes_in(text: str) -> List[str]:
    """Heroes named in text, longest name first ("鲁班大师" is not "鲁班七号")."""
    found, rest = [], text
    for h in HEROES_BY_LENGTH:
        if h in rest:
            found.append(h)
            rest = rest.replace(h, " ")
    return found


def positions_in(text: str) -> List[str]:
    return [p for p, words in POSITION_ALIASES.items() if any(w in text for w in words)]


_GAME_TALK = ("排位", "上分", "掉分", "连跪", "连败", "连胜", "段位", "赛季", "战令", "皮肤", "开黑", "双排", "五排",
              "对面", "队友", "打野", "补位", "团战", "推塔", "大招", "出装", "铭文", "MVP", "mvp", "巅峰赛", "王者",
              "钻石", "星耀", "铂金", "黄金", "白银", "青铜", "上星", "掉星", "这把", "那把", "一把", "几把")


def is_game_talk(text: str) -> bool:
    """Does a line talk about the game (matches, heroes, rank) rather than life?"""
    return bool(heroes_in(text)) or any(w in text for w in _GAME_TALK) or bool(positions_in(text))
