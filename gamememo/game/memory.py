# -*- coding: utf-8 -*-
"""GameMemory: what a game companion remembers, from two sources.

- Match data -> ``GameLedger``: rank, seasons, heroes, streaks, computed
  exactly. It is the system's record, so when a player's own account
  disagrees ("上赛季我可是打到星耀了"), the record is what holds.
- Chat -> ``RawMemory``: the player's life, feelings and the assistant's
  promises, kept verbatim with their dates (see docs/FINDINGS.md for why it
  is not rewritten into facts).

The two meet on the game's own time axis: a question anchored by a game event
("我第一次上钻石那天…", "连输最多那次…") is resolved to a date by the ledger, and
that day's chat is recalled.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from ..personal.model import MemoryRecord
from ..personal.raw import RawMemory
from .ledger import GameLedger

RECORD_NOTE = "（以上是系统记录的对局数据，比玩家自己的说法准；两者不一致时以系统记录为准）"


class GameMemory:
    def __init__(self, user_id: str, storage_dir: str = "./memory_data", embedder=None,
                 clock: Callable[[], datetime] = datetime.now, **raw_opts):
        self.clock = clock
        # game attributes (rank, heroes) come from the ledger, not from what was said in chat
        raw_opts.setdefault("attribute_index", False)
        self.chat = RawMemory(user_id, storage_dir=storage_dir, embedder=embedder, clock=clock, **raw_opts)
        self.ledger = GameLedger()

    # ---------------------------------------------------------------- write

    def ingest_chat(self, text: str):
        return self.chat.ingest(text, source="chat")

    def ingest_matches(self, matches: Iterable[Dict]) -> None:
        self.ledger.extend(matches)

    # ---------------------------------------------------------------- read

    def day_exchanges(self, day: str) -> List[MemoryRecord]:
        return sorted((r for r in self.chat.store.active() if r.kind == "turn" and r.created_at.startswith(day)),
                      key=lambda r: r.created_at)

    def describe(self, r: MemoryRecord) -> str:
        """A chat exchange, with a note when the player's rank claim contradicts the record."""
        text = self.chat.describe(r)
        if r.kind == "turn":
            said = "\n".join(l for l in r.content.splitlines() if l.startswith("玩家"))
            note = self.ledger.check_claim(said, r.created_at[:10])
            if note:
                text += note
        return text

    def context(self, query: str, top_k: int = 8) -> str:
        now = self.clock()
        days = self.ledger.event_dates(query, now)
        linked = [(day, self.day_exchanges(day)) for day in days]
        linked = [(day, ex) for day, ex in linked if ex]
        if linked:
            # the question is anchored by a game event: answer from that day's chat
            return "【问题提到的那一天（由对局记录确定）】\n" + "\n\n".join(
                f"{day}那天的聊天：\n" + "\n".join(self.describe(r) for r in ex) for day, ex in linked)
        parts = []
        card = self.ledger.card(now)
        if card:
            views = self.ledger.views(query, now)
            parts.append(card + ("\n" + "\n".join(views) if views else "") + "\n" + RECORD_NOTE)
        recalled = self.chat.retrieve(query, top_k=top_k, touch=False)
        if recalled:
            parts.append("【相关的聊天原话】\n" + "\n".join(self.describe(r) for r in recalled))
        promises = self.chat.pending_promises()
        if promises:
            parts.append("【你答应过玩家的事】\n" + "\n".join(self.describe(r) for r in promises))
        return "\n\n".join(parts) if parts else "关于这位玩家，你没有任何记录。"
