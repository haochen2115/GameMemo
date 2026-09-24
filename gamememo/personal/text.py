# -*- coding: utf-8 -*-
"""Chinese-aware tokenization for lexical retrieval.

Uses jieba when installed and falls back to character bigrams otherwise,
so the package keeps working with zero optional dependencies.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

try:  # optional dependency
    import jieba  # type: ignore

    jieba.setLogLevel(60)
    HAS_JIEBA = True
except ImportError:  # pragma: no cover - exercised only without jieba
    jieba = None
    HAS_JIEBA = False


_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)

STOPWORDS: Set[str] = set(
    "的 了 是 我 你 他 她 它 们 我们 你们 他们 在 和 与 及 就 都 也 还 又 很 太 最 更 吧 吗 呢 啊 呀 哦 嗯 "
    "这 那 这个 那个 这些 那些 一个 一下 一些 有 没 没有 不 会 能 要 想 让 把 被 给 对 从 到 为 以 "
    "什么 怎么 怎么样 怎样 如何 哪 哪个 哪些 哪里 多少 几 吗 么 啥 嘛 请 记住 知道 觉得 感觉 "
    "玩家 助手 一般 通常 平时 经常 比较 可以 应该 时候 现在 之前 以后 最近 其实 就是 还是 或者 "
    "上 下 中 里 个 次 场 局 过 着 得 地 所以 因为 但是 然后 如果 自己 用".split()
)


# Function characters; a bigram containing one is almost always noise
# ("是我", "的生"), so bigram expansion skips it.
STOP_CHARS: Set[str] = set("的了是我你他她它们在和与就都也还又很太最吧吗呢啊呀哦嗯这那个有没不会能要想让把被给对从到为以什么怎哪几啥嘛请过着得地")


def add_words(words: Iterable[str]) -> None:
    """Teach the segmenter domain terms (hero names, ranks, skins...)."""
    if not HAS_JIEBA:
        return
    for w in words:
        w = w.strip()
        if 1 < len(w) <= 12:
            jieba.add_word(w, freq=20000)


def _bigrams(text: str) -> List[str]:
    out: List[str] = []
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.append(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def tokenize(text: str, bigrams: bool = False) -> List[str]:
    """Split text into retrieval terms.

    With jieba: search-mode words (long words also yield their sub-words),
    minus stopwords and punctuation. ``bigrams=True`` adds CJK character
    bigrams as a recall safety net for words jieba segments differently.
    """
    text = text.lower()
    tokens: List[str] = []
    if HAS_JIEBA:
        for tok in jieba.lcut_for_search(text):
            tok = tok.strip()
            if not tok or _PUNCT.match(tok) or tok in STOPWORDS:
                continue
            if len(tok) == 1 and _CJK.fullmatch(tok):
                continue
            tokens.append(tok)
    else:
        tokens.extend(t for t in _bigrams(text) if t not in STOPWORDS)
        tokens.extend(_WORD.findall(text))
        return tokens
    if bigrams:
        tokens.extend(t for t in _bigrams(text)
                      if len(t) == 2 and not (set(t) & STOP_CHARS) and t not in tokens)
    return tokens
