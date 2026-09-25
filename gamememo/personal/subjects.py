# -*- coding: utf-8 -*-
"""Who a question or a memory is about.

"我姐姐是做什么的" asks about the sister; a memory about the brother's job
is not an answer, however close it looks. A question that names a person
only considers memories that mention that person; a question about the
player ignores facts whose subject is somebody else.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Pattern, Tuple

# canonical person -> surface forms (longest first). Cousins (表/堂) are
# different people from siblings, so a bare 哥/姐/弟/妹 must not follow 表 or 堂.
_FORMS: Dict[str, Tuple[str, ...]] = {
    # One partner, many words: "女朋友" before the wedding, "老婆" after,
    # "爱人" / "对象" for either. Listed first so every partner word maps here.
    "伴侣": ("女朋友", "男朋友", "老婆", "老公", "妻子", "丈夫", "媳妇", "太太", "爱人", "对象", "女友", "男友"),
    "妈妈": ("妈妈", "母亲", "老妈", "妈"),
    "爸爸": ("爸爸", "父亲", "老爸", "爸"),
    "哥哥": ("哥哥", "大哥", "哥"),
    "姐姐": ("姐姐", "大姐", "姐"),
    "弟弟": ("弟弟", "弟"),
    "妹妹": ("妹妹", "妹"),
    "儿子": ("儿子",),
    "女儿": ("女儿", "闺女"),
    "孩子": ("孩子", "儿子", "女儿", "闺女", "娃"),
    "室友": ("室友", "舍友"),
    "同事": ("同事",),
    "同学": ("同学",),
    "朋友": ("朋友", "兄弟", "闺蜜", "哥们"),
    "表哥": ("表哥",), "表姐": ("表姐",), "表弟": ("表弟",), "表妹": ("表妹",),
    "堂哥": ("堂哥",), "堂姐": ("堂姐",), "堂弟": ("堂弟",), "堂妹": ("堂妹",),
    "公公": ("公公",), "婆婆": ("婆婆",),
    "岳父": ("岳父", "老丈人"), "岳母": ("岳母", "丈母娘"),
    "爷爷": ("爷爷",), "奶奶": ("奶奶",),
    "外公": ("外公", "姥爷"), "外婆": ("外婆", "姥姥"),
    "导师": ("导师",), "师父": ("师父", "师傅"), "徒弟": ("徒弟",),
    "队友": ("队友",), "老板": ("老板", "领导"), "邻居": ("邻居",),
}


def _pattern(forms: Tuple[str, ...]) -> Pattern:
    alts = sorted(forms, key=len, reverse=True)
    return re.compile("|".join(f"(?<![表堂]){re.escape(f)}" if len(f) == 1 else re.escape(f) for f in alts))


_MENTION: Dict[str, Pattern] = {p: _pattern(f) for p, f in _FORMS.items()}
# surface form -> the first person listing it ("女儿" is a daughter, not just a child)
_SURFACE: Dict[str, str] = {}
for _p, _fs in _FORMS.items():
    for _f in _fs:
        _SURFACE.setdefault(_f, _p)
_ASKED = re.compile(r"我(?:的)?(" + "|".join(sorted(_SURFACE, key=len, reverse=True)) + r")")
# "玩家的妹妹…", "玩家老婆…": the fact is about that person.
_FACT_OTHER = re.compile(r"^玩家(?:的)?(" + "|".join(sorted((f for f in _SURFACE if len(f) > 1), key=len, reverse=True)) + r")")


def query_subject(query: str) -> Optional[str]:
    """The person a question asks about, or None for the player."""
    m = _ASKED.search(query)
    return _SURFACE[m.group(1)] if m else None


def asked_word(query: str) -> Optional[str]:
    """The word the question uses for the person ("姐姐" in "我姐姐…")."""
    m = _ASKED.search(query)
    return m.group(1) if m else None


def strip_subject(query: str) -> str:
    """The question without the person it names ("我姐姐做什么的" -> "我做什么的").
    Once memories are filtered to that person, the word itself only adds
    noise: a memory may call her "姐" or "大姐"."""
    return _ASKED.sub("我", query, count=1)


def mentions(text: str, person: str) -> bool:
    return bool(_MENTION[person].search(text))


def fact_subject(text: str) -> Optional[str]:
    """Whom a fact is about: a person, or None for the player."""
    m = _FACT_OTHER.match(text.strip())
    return _SURFACE[m.group(1)] if m else None
