# -*- coding: utf-8 -*-
"""A game-assistant chatbot wired to PersonalMemory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..llm import LLMClient
from . import prompts
from .model import MemoryRecord
from .system import IngestReport, PersonalMemory

WEEKDAYS = "一二三四五六日"


@dataclass
class ChatTurn:
    reply: str
    profile: List[MemoryRecord] = field(default_factory=list)
    retrieved: List[MemoryRecord] = field(default_factory=list)
    report: Optional[IngestReport] = None  # set on turns that ran extraction
    errors: List[str] = field(default_factory=list)


class MemoryChatBot:
    """Each turn: retrieve → answer → (every ``extract_every`` turns) ingest.

    Extraction only reads turns it has not processed yet, plus a little
    preceding context, so nothing is re-extracted and duplicated.
    """

    FALLBACK_REPLY = "抱歉，我这边刚才卡了一下，你能再说一次吗？"

    def __init__(self, memory: PersonalMemory, llm: LLMClient,
                 extract_every: int = 3, history_turns: int = 10,
                 top_k: int = 3, context_turns: int = 2):
        self.memory = memory
        self.llm = llm
        self.extract_every = extract_every
        self.history_turns = history_turns
        self.top_k = top_k
        self.context_turns = context_turns
        self.messages: List[Dict[str, str]] = []
        self._extracted_upto = 0  # index into self.messages
        self._user_turns = 0

    def chat(self, user_input: str, temperature: float = 0.8) -> ChatTurn:
        self._user_turns += 1
        profile = self.memory.core_profile()
        promises = self.memory.pending_promises()
        shown = {r.id for r in profile + promises}
        retrieved = [r for r in self.memory.retrieve(self._query(user_input), top_k=self.top_k)
                     if r.id not in shown]

        self.messages.append({"role": "user", "content": user_input})
        errors: List[str] = []
        try:
            reply = self.llm.chat(system=self._system_prompt(profile, retrieved, promises),
                                  messages=self.messages[-2 * self.history_turns:],
                                  temperature=temperature)
        except Exception as e:  # a model hiccup must not kill the conversation
            errors.append(f"reply: {e}")
            reply = self.FALLBACK_REPLY
        self.messages.append({"role": "assistant", "content": reply})

        turn = ChatTurn(reply=reply, profile=profile, retrieved=retrieved, errors=errors)
        if self._user_turns % self.extract_every == 0:
            try:
                turn.report = self.flush()
            except Exception as e:  # unprocessed turns stay pending and are retried
                errors.append(f"memory: {e}")
        return turn

    def flush(self) -> Optional[IngestReport]:
        """Extract memories from turns not processed yet. On failure the
        turns stay pending, so the next flush retries them."""
        if self._extracted_upto >= len(self.messages):
            return None
        start = max(0, self._extracted_upto - 2 * self.context_turns)
        end = len(self.messages)
        report = self.memory.ingest(self._transcript(start, end), source="chat")
        self._extracted_upto = end
        return report

    # ---- helpers ----

    def _query(self, user_input: str) -> str:
        # Very short inputs ("那后羿呢？") lean on the previous user turn.
        if len(user_input) < 6:
            prev = [m["content"] for m in self.messages if m["role"] == "user"][-1:]
            return " ".join(prev + [user_input])
        return user_input

    def _transcript(self, start: int, end: int) -> str:
        names = {"user": "玩家", "assistant": "助手"}
        return "\n".join(f"{names[m['role']]}: {m['content']}" for m in self.messages[start:end])

    def _system_prompt(self, profile: List[MemoryRecord], retrieved: List[MemoryRecord],
                       promises: List[MemoryRecord] = ()) -> str:
        now = self.memory.clock()
        parts = [prompts.CHAT_PERSONA,
                 f"现在是{now.strftime('%Y-%m-%d %H:%M')}，星期{WEEKDAYS[now.weekday()]}。"]
        if profile:
            parts.append("【玩家档案】\n" + self.memory.format_for_prompt(profile))
        if retrieved:
            parts.append("【与当前话题相关的记忆】\n" + self.memory.format_for_prompt(retrieved))
        if promises:
            parts.append("【你答应过玩家的事】（合适的时候主动兑现或提起）\n"
                         + self.memory.format_for_prompt(promises))
        if profile or retrieved or promises:
            parts.append(prompts.CHAT_MEMORY_RULES)
        return "\n\n".join(parts)
