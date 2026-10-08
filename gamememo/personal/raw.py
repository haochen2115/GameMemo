# -*- coding: utf-8 -*-
"""Raw-first memory: keep what was said, verbatim, and retrieve it.

E11 (docs/EXPERIMENTS.md) showed that having a small LLM rewrite chats
into "facts" loses information: the whole fact store of P6b in the prompt
still scored 0.16 below the raw history, and retrieving raw exchanges beat
the full P6b system. So this memory never rewrites anything:

- write: every exchange (one player line + the assistant's reply) is stored
  as it was said, with its time. No LLM call, nothing to hallucinate, and
  a later message never overwrites an earlier one; the reply model sees
  the dates and works out what is current.
- the assistant's promises are its own sentences, found by pattern
  (``promises_from_rules``), stored verbatim and kept in the prompt like a
  to-do list.
- read: hybrid retrieval (BM25 + dense + relevance gate) over exchanges,
  shown oldest first; "how did it change" questions get a wider window,
  since every state of a trajectory is a separate exchange.

It exposes the read API ``MemoryChatBot`` uses, so it is a drop-in
replacement for ``PersonalMemory``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Callable, List, Optional, Sequence

from .embed import Embedder
from .model import MemoryRecord, fmt_time
from .retrieval import HybridRetriever, RetrievalConfig
from .consolidate import RECALL_ATTRIBUTES, detect
from .store import JsonMemoryStore
from .system import ATTRIBUTE_QUERY, HERO_QUERY, IngestReport, promises_from_rules

_SPEAKER = re.compile(r"^\s*(玩家|助手)\s*[:：]\s*")
# "how did it change" questions need every state, not the best few mentions
CHANGE_INTENT = re.compile(r"怎么(变|换|升|降|走|起伏|上来)|变化|换过|一路|一步步|历程|前后")


TIMELINE_NOTE = "（下面是玩家先后{n}次提到这件事的原话，按时间排列；问变化过程时，要按时间顺序逐一说出每个阶段）"


def exchanges(text: str) -> List[str]:
    """Split a transcript into exchanges: a player line plus the assistant
    lines that answer it. Assistant lines before any player line are kept
    as their own exchange (they may hold a promise)."""
    out: List[List[str]] = []
    for line in text.splitlines():
        m = _SPEAKER.match(line)
        if not m or not line[m.end():].strip():
            continue
        if m.group(1) == "玩家" or not out:
            out.append([])
        out[-1].append(f"{m.group(1)}: {line[m.end():].strip()}")
    return ["\n".join(x) for x in out]


class RawMemory:
    def __init__(self,
                 user_id: str,
                 storage_dir: str = "./memory_data",
                 embedder: Optional[Embedder] = None,
                 retrieval_config: Optional[RetrievalConfig] = None,
                 clock: Callable[[], datetime] = datetime.now,
                 min_k: int = 8,
                 change_k: int = 12,
                 promises: bool = True,
                 max_promises: int = 3,
                 timeline: bool = True,
                 attribute_index: bool = False,
                 state_line: bool = True):
        """
        min_k: retrieve at least this many exchanges, whatever top_k the
            caller asks for (one exchange is much shorter than a summary).
        change_k: window for "how did it change" questions.
        promises: detect the assistant's promises and keep them in mind.
        timeline: for change questions, lead the recalled exchanges with a
            note that they are every mention in time order, so a small reply
            model lists each stage instead of summarising the last one.
        attribute_index: for a change question about a closed-set game
            attribute (rank, main hero, phone), recall every exchange in which
            the player states a value of it, not only the best-matching ones;
            otherwise unrelated chat about heroes and ranked games fills the
            window once the history is long (E15). The index only points at
            exchanges; nothing is rewritten.
        state_line: with the attribute index, lead with one line listing the
            values the player stated, in time order (taken from the player's
            own words by the attribute patterns); a small reply model lists
            these, while it summarises a dozen raw exchanges to the last one.
        """
        self.user_id = user_id
        self.clock = clock
        self.min_k = min_k
        self.change_k = change_k
        self.promises = promises
        self.max_promises = max_promises
        self.timeline = timeline
        self.attribute_index = attribute_index
        self.state_line = state_line
        self.store = JsonMemoryStore(os.path.join(storage_dir, f"{user_id}_raw.json"))
        if retrieval_config is None:
            retrieval_config = RetrievalConfig.for_embedder(embedder)
        self.retriever = HybridRetriever(embedder=embedder, config=retrieval_config)

    # ================================================================ write

    def ingest(self, text: str, source: str = "chat") -> IngestReport:
        now = fmt_time(self.clock())
        report = IngestReport()
        for ex in exchanges(text):
            rec = MemoryRecord(content=ex, kind="turn", source=source, importance=3,
                               created_at=now, updated_at=now, event_time=now[:10])
            self.store.put(rec)
            report.added.append(rec)
        if self.promises:
            known = {r.content for r in self.store.active() if r.kind == "promise"}
            for p in promises_from_rules(text):
                if p not in known:
                    rec = MemoryRecord(content=p, kind="promise", source=source, importance=4,
                                       created_at=now, updated_at=now)
                    self.store.put(rec)
                    report.added.append(rec)
                    known.add(p)
        self.store.save()
        return report

    def forget(self, mem_id: str, hard: bool = False) -> bool:
        rec = self.store.get(mem_id)
        if rec is None:
            return False
        if hard:
            del self.store.records[mem_id]
        elif rec.is_active:
            rec.valid_to = rec.updated_at = fmt_time(self.clock())
        self.store.save()
        return True

    # ================================================================ read

    def retrieve(self, query: str, top_k: int = 5, touch: bool = True) -> List[MemoryRecord]:
        change = bool(CHANGE_INTENT.search(query))
        k = max(top_k, self.change_k if change else self.min_k)
        turns = [r for r in self.store.active() if r.kind == "turn"]
        hits = [h.record for h in self.retriever.search(query, turns, top_k=k, now=self.clock())]
        attr = self._asked_attribute(query) if change and self.attribute_index else None
        states: List[MemoryRecord] = []
        if attr:
            states = sorted((r for r in turns if self._states(r, attr)), key=lambda r: r.created_at)
            others = [r for r in hits if r not in states]
            hits = states + others[:max(0, k - len(states))]
        hits.sort(key=lambda r: r.created_at)
        if states and self.state_line:
            line = self._state_line(attr, states)
            if line:
                hits.insert(0, MemoryRecord(content=line, kind="note", source="recall",
                                            created_at=states[0].created_at, id="state-line"))
                return hits
        if change and self.timeline and len(hits) > 1:
            note = MemoryRecord(content=TIMELINE_NOTE.format(n=len(hits)), kind="note", source="recall",
                                created_at=hits[0].created_at, id="timeline-note")
            hits.insert(0, note)
        return hits

    @staticmethod
    def _asked_attribute(query: str) -> Optional[str]:
        for name, pattern in ATTRIBUTE_QUERY + (HERO_QUERY,):
            if pattern.search(query):
                return name
        return None

    @staticmethod
    def _states(rec: MemoryRecord, attr: str) -> bool:
        """Does the player, in this exchange, state a value of ``attr``?"""
        said = "\n".join(l for l in rec.content.splitlines() if l.startswith("玩家"))
        return any(name == attr for name, _ in detect(MemoryRecord(content=said), RECALL_ATTRIBUTES))

    @staticmethod
    def _state_line(attr: str, states: List[MemoryRecord]) -> str:
        """Every value the player stated, oldest first, each as said and dated:
        "按时间顺序，玩家说过的段位：黄金三（2026-01-06）→ 铂金二（2026-03-12）"."""
        parts, last = [], None
        for r in states:
            said = "\n".join(l for l in r.content.splitlines() if l.startswith("玩家"))
            value = next((v for n, v in detect(MemoryRecord(content=said), RECALL_ATTRIBUTES) if n == attr), "")
            if value and value != last:
                parts.append(f"{value}（{r.created_at[:10]}）")
                last = value
        if len(parts) < 2:
            return ""
        return f"按时间顺序，玩家说过的{attr}：" + " → ".join(parts) + "（下面是原话）"

    def core_profile(self, limit: int = 6) -> List[MemoryRecord]:
        return []  # nothing is distilled; everything is recalled on demand

    def pending_promises(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        promises = [r for r in self.store.active() if r.kind == "promise"]
        promises.sort(key=lambda r: r.created_at, reverse=True)
        return promises[:limit or self.max_promises]

    def active(self) -> List[MemoryRecord]:
        return self.store.active()

    @staticmethod
    def describe(r: MemoryRecord) -> str:
        if r.kind == "note":
            return r.content
        if r.kind == "turn":
            return f"【{r.created_at[:16]}】" + r.content.replace("\n", " ")
        return f"{r.content}（{r.created_at[:10]}）"

    @classmethod
    def format_for_prompt(cls, records: Sequence[MemoryRecord]) -> str:
        return "\n".join(f"- {cls.describe(r)}" for r in records)
