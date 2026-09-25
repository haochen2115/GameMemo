# -*- coding: utf-8 -*-
"""Consolidation: link facts that are successive values of one attribute.

Small models often *add* "玩家上钻石了" instead of *updating* "玩家段位是铂金",
leaving both active: "what is my rank now" then returns stale values, and
"how did my rank change" finds no version chain. Like memory consolidation
during sleep, this pass reorganises what was stored: for attributes whose
values form a closed set (rank tiers, device brands) it detects the value
in each active fact and supersedes older values with newer ones, so the
chain oldest -> newest becomes the attribute's history.

It is deterministic, cheap, and only touches attributes it can recognise
with high precision; everything else is left to the LLM's own UPDATEs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Pattern, Sequence, Tuple

from .model import MemoryRecord


@dataclass(frozen=True)
class Attribute:
    name: str
    values: Tuple[str, ...]
    exclude: Pattern            # statements about wishes/goals are not the current value
    strip: Tuple[str, ...] = ()  # substrings removed before matching (e.g. the game's name)
    require: Optional[Pattern] = None
    suffix: Optional[Pattern] = None  # sub-level kept in the value: 钻石 + 三 -> 钻石三


# "冲王者" is a wish; "冲到星耀四了" / "冲上王者" is an achievement.
_WISH = re.compile(r"目标|想|打算|冲(?![到上进])|希望|准备|梦想|争取|计划")

ATTRIBUTES: Tuple[Attribute, ...] = (
    Attribute("段位", ("青铜", "白银", "黄金", "铂金", "钻石", "星耀", "王者"),
              exclude=_WISH, strip=("王者荣耀", "荣耀王者"), suffix=re.compile(r"[一二三四五六七八九十0-9]+星?")),
    Attribute("设备", ("iPhone", "iPad", "华为", "小米", "红米", "OPPO", "vivo", "一加", "三星", "Redmi"),
              exclude=re.compile(r"想买|打算买|准备买|想换|打算换"),
              require=re.compile(r"手机|设备|平板|用|换|iPhone|iPad", re.I)),
)


# 王者荣耀 heroes. Only used for recall ("我主玩的英雄是怎么变的"), not for the
# write-side checks, and only when the fact says the player plays the hero.
HEROES = tuple(sorted((
    "鲁班七号 鲁班大师 孙尚香 后羿 虞姬 狄仁杰 李元芳 马可波罗 公孙离 黄忠 百里守约 伽罗 蒙犽 艾琳 戈娅 敖隐 莱西奥 "
    "成吉思汗 孙策 "
    "妲己 安琪拉 小乔 王昭君 貂蝉 甄姬 武则天 诸葛亮 干将莫邪 上官婉儿 嫦娥 西施 沈梦溪 杨玉环 不知火舞 米莱狄 弈星 "
    "女娲 周瑜 高渐离 扁鹊 张良 墨子 嬴政 姜子牙 司马懿 海月 金蝉 王者 元歌 露娜 海诺 "
    "李白 韩信 赵云 娜可露露 兰陵王 阿轲 百里玄策 孙悟空 镜 澜 裴擒虎 云中君 暃 司空震 大司命 宫本武藏 橘右京 "
    "典韦 曜 "
    "亚瑟 吕布 关羽 花木兰 马超 老夫子 狂铁 夏侯惇 曹操 杨戬 李信 盘古 蒙恬 刘备 达摩 铠 哪吒 猪八戒 程咬金 苏烈 "
    "项羽 刘邦 廉颇 张飞 白起 东皇太一 牛魔 太乙真人 蔡文姬 大乔 孙膑 明世隐 瑶 鬼谷子 庄周 刘禅 钟馗 盾山 少司缘 "
    "朵莉亚 桑启 鲁班 梦奇 钟无艳 芈月 雅典娜 宫本 阿古朵 姬小满 云缨 影 司马 空空儿"
).split(), key=len, reverse=True))
_PLAYS = re.compile(r"主玩|常用|本命|改玩|在练|练|最近玩|拿手|擅长|换成|玩的是|玩得最多|主打|一直玩|只玩|基本只玩|上手|喜欢用|常玩|用")

HERO = Attribute("英雄", HEROES, exclude=re.compile(r"想玩|打算玩|克|被|对面|敌方|怕"), require=_PLAYS)
RECALL_ATTRIBUTES: Tuple[Attribute, ...] = ATTRIBUTES + (HERO,)


_SUB = re.compile(r"[一二三四五六七八九十0-9]+星?$")


def base_value(value: str) -> str:
    """钻石三 / 钻石三星 -> 钻石 (sub-levels are phrased inconsistently)."""
    return _SUB.sub("", value)


def detect(rec: MemoryRecord, attributes: Sequence[Attribute] = ATTRIBUTES) -> List[Tuple[str, str]]:
    """(attribute, value) pairs a fact states; the last-mentioned value wins
    ("从黄金升到钻石" -> 钻石)."""
    found = []
    for attr in attributes:
        text = rec.content
        for s in attr.strip:
            text = text.replace(s, "")
        if attr.exclude.search(text) or (attr.require and not attr.require.search(text)):
            continue
        low = text.lower()
        hits = [(low.rfind(v.lower()), v) for v in attr.values if v.lower() in low]
        if hits:
            pos, value = max(hits)
            if attr.suffix:
                m = attr.suffix.match(text, pos + len(value))
                if m:
                    value += m.group(0)
            found.append((attr.name, value))
    return found


_CLAUSE = re.compile(r"[，,；;、]|并且|而且|同时")


def single_attribute(rec: MemoryRecord) -> bool:
    """Only short, single-clause facts are retired by consolidation: a
    composite fact ("玩家是大一新生，在武汉，段位黄金一") also carries other
    information that must not be forgotten when the rank changes."""
    return not _CLAUSE.search(rec.content) and len(rec.content) <= 24


def _when(r: MemoryRecord) -> Tuple[str, str]:
    return (r.event_time or r.created_at[:10], r.created_at)


def consolidate(records: Sequence[MemoryRecord], history: Callable[[str], List[MemoryRecord]],
                now: str, attributes: Sequence[Attribute] = ATTRIBUTES) -> List[MemoryRecord]:
    """Link active facts of the same attribute into one version chain, in
    time order. ``history(id)`` returns a record's chain (oldest first).
    Returns the records that were retired.
    """
    by_attr: Dict[str, List[MemoryRecord]] = {}
    for r in records:
        if r.kind != "fact" or not r.is_active:
            continue
        for name, _ in detect(r, attributes):
            by_attr.setdefault(name, []).append(r)

    retired = []
    # Composite facts may be the newest value but are never retired.
    for recs in by_attr.values():
        if len(recs) < 2:
            continue
        recs.sort(key=_when)
        newest = recs[-1]
        for old in recs[:-1]:
            if not old.is_active or not single_attribute(old) or history(old.id)[-1].id == newest.id:
                continue
            _insert(old, history(newest.id), now)
            retired.append(old)
    return retired


def _insert(old: MemoryRecord, chain: List[MemoryRecord], now: str) -> None:
    """Retire ``old`` by placing it into ``chain`` at its point in time."""
    later = [r for r in chain if _when(r) > _when(old)]
    nxt = later[0] if later else chain[-1]
    prev_id = nxt.supersedes
    if prev_id and prev_id != old.id and not old.supersedes:
        # nxt had an older version: splice old between it and nxt.
        prev = next(r for r in chain if r.id == prev_id)
        prev.superseded_by = old.id
        old.supersedes = prev.id
    nxt.supersedes = old.id
    old.superseded_by = nxt.id
    old.valid_to = old.updated_at = now
