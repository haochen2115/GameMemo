# -*- coding: utf-8 -*-
"""PersonalMemory: long-term memory about one player.

Write path (``ingest``), two LLM calls per batch of conversation or data:
  1. extract atomic facts, with relative dates resolved against "today";
  2. for those facts, retrieve only the *related* existing memories, show
     them under short numeric aliases, and ask for ADD/UPDATE/DELETE/NOOP.
Every operation is validated before it touches the store: unknown targets
are rejected (not silently "succeeded"), and near-duplicate ADDs are
dropped. UPDATE supersedes instead of overwriting, so history is kept.

Read path (``retrieve``): the local hybrid retriever, no LLM call.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..llm import LLMClient, parse_json
from . import prompts
from .consolidate import consolidate as consolidate_versions
from .consolidate import base_value, detect as detect_values
from .embed import Embedder
from .model import MemoryRecord, clamp_importance, fmt_time
from .retrieval import HybridRetriever, RetrievalConfig, ScoredMemory
from .store import JsonMemoryStore


@dataclass
class IngestReport:
    facts: List[str] = field(default_factory=list)
    added: List[MemoryRecord] = field(default_factory=list)
    updated: List[Tuple[MemoryRecord, MemoryRecord]] = field(default_factory=list)  # (old, new)
    deleted: List[MemoryRecord] = field(default_factory=list)
    noop: int = 0
    rejected: List[Tuple[Dict, str]] = field(default_factory=list)  # (op, why)
    consolidated: List[MemoryRecord] = field(default_factory=list)  # retired by consolidation

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)


_NORM = re.compile(r"[\s\W_]+", re.UNICODE)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Questions about the past ("什么时候升的铂金") need superseded versions too.
PAST_INTENT = re.compile(r"什么时候|哪天|以前|之前|原来|曾经|上次|当时|那时|第一次")
# Recall modes (P1). A person answers these by recalling differently:
# "上次聊了什么" -> the latest conversation; "你答应过我什么" -> own promises;
# "我段位是怎么变的" -> the whole story of one attribute, not its current value.
RECENT_TALK = re.compile(r"(上次|上一次|最近一次|刚才|前几天).{0,6}(聊|说|讲)")
PROMISE_INTENT = re.compile(r"答应|承诺|说过要|说好|保证过")
# Episodes describe the state *at that time*; "我现在什么段位" must be answered
# from semantic memory, so episodes are searched only for questions about a
# particular time or event.
EPISODIC_INTENT = re.compile(r"那天|那次|那阵子|那段时间|那时候|发生了什么|哪次|当时|上次|什么时候|聊过|说过")
TRAJECTORY_INTENT = re.compile(r"怎么变|变化|一路|一步步|这几个月|历程|怎么升|怎么上来|成长")


def _distinct_versions(chain: List[MemoryRecord]) -> List[MemoryRecord]:
    """Drop versions that repeat the previous value, keeping the earliest
    (when it first became true): 白银→黄金→铂金→铂金 tells 白银→黄金→铂金."""
    out: List[MemoryRecord] = []
    for r in chain:
        if out:
            prev = out[-1]
            same_value = detect_values(r) and detect_values(r) == detect_values(prev)
            if same_value or SequenceMatcher(None, _norm(r.content), _norm(prev.content)).ratio() >= 0.9:
                continue
        out.append(r)
    return out


def _grounded(content: str, source_text: str) -> bool:
    """Closed-set values (rank tier, device brand) in a new memory must
    appear in the text it was written from."""
    probe = MemoryRecord(content=content)
    low = source_text.lower()
    return all(base_value(value).lower() in low for _, value in detect_values(probe))


def _norm(text: str) -> str:
    return _NORM.sub("", text.lower())


class PersonalMemory:
    def __init__(self,
                 user_id: str,
                 llm: Optional[LLMClient] = None,
                 storage_dir: str = "./memory_data",
                 embedder: Optional[Embedder] = None,
                 retrieval_config: Optional[RetrievalConfig] = None,
                 clock: Callable[[], datetime] = datetime.now,
                 related_per_fact: int = 3,
                 max_candidates: int = 15,
                 duplicate_ratio: float = 0.9,
                 write_mode: str = "ops",
                 player_only: bool = True,
                 history_recall: bool = True,
                 per_turn: bool = False,
                 max_output_tokens: int = 1024,
                 episodes: bool = True,
                 promises: bool = True,
                 recall_modes: bool = True,
                 consolidate: bool = True):
        """
        write_mode: "ops" = extract facts, then the LLM decides
            ADD/UPDATE/DELETE/NOOP against related memories; "slots" = facts
            are extracted with an aspect, and a new value of a single-valued
            aspect supersedes the old one without a second LLM call.
        player_only: extract from the player's lines only (assistant lines
            are dropped before the LLM sees the transcript).
        history_recall: questions about the past also search superseded
            versions (ranked lower and marked as outdated).
        per_turn: extract facts from each player line separately (shorter
            inputs help small models), then decide operations once.
        max_output_tokens: cap on every write-path LLM reply; small models
            can loop forever inside structured output without it.
        episodes: also store a one-sentence summary of every conversation
            (episodic memory: "what happened that day").
        promises: also store what the assistant promised the player
            (the assistant's own autobiographical memory).
        recall_modes: route "last time" / "promised" / "how did it change"
            questions to the matching kind of recall.
        consolidate: after each write, link successive values of closed-set
            attributes (rank, device) into version chains (consolidate.py).
        """
        if write_mode not in ("ops", "slots"):
            raise ValueError(f"unknown write_mode {write_mode!r}")
        self.write_mode = write_mode
        self.player_only = player_only
        self.history_recall = history_recall
        self.per_turn = per_turn
        self.max_output_tokens = max_output_tokens
        self.episodes = episodes
        self.promises = promises
        self.recall_modes = recall_modes
        self.consolidate = consolidate
        self.user_id = user_id
        self.llm = llm
        self.clock = clock
        self.related_per_fact = related_per_fact
        self.max_candidates = max_candidates
        self.duplicate_ratio = duplicate_ratio
        self.store = JsonMemoryStore(os.path.join(storage_dir, f"{user_id}_memory.json"))
        if retrieval_config is None:
            retrieval_config = RetrievalConfig.for_embedder(embedder)
        self.retriever = HybridRetriever(embedder=embedder, config=retrieval_config)
        self.candidate_retriever = HybridRetriever(embedder=embedder,
                                                   config=retrieval_config.for_candidates())

    # ================================================================ read

    def search(self, query: str, top_k: int = 5,
               include_history: Optional[bool] = None) -> List[ScoredMemory]:
        if self.recall_modes:
            routed = self._recall_mode(query, top_k)
            if routed:
                return routed
        if include_history is None:
            include_history = self.history_recall and bool(PAST_INTENT.search(query))
        records = self.store.all() if include_history else self.store.active()
        if not EPISODIC_INTENT.search(query):
            records = [r for r in records if r.kind != "episode"]
        # Promises are recalled when asked about, and shown to the chatbot via
        # pending_promises(); in ordinary search they only crowd out facts.
        records = [r for r in records if r.kind != "promise" or not self.recall_modes]
        return self.retriever.search(query, records, top_k=top_k, now=self.clock(),
                                     include_inactive=include_history)

    def _recall_mode(self, query: str, top_k: int) -> List[ScoredMemory]:
        now = self.clock()
        if PROMISE_INTENT.search(query):
            promises = [r for r in self.store.active() if r.kind == "promise"]
            hits = self.retriever.search(query, promises, top_k=top_k, now=now)
            if not hits:  # "你答应过我什么" names no topic: recall them all, newest first
                promises.sort(key=lambda r: r.created_at, reverse=True)
                hits = [ScoredMemory(r, 1.0) for r in promises[:top_k]]
            return hits
        if RECENT_TALK.search(query):
            episodes = sorted((r for r in self.store.active() if r.kind == "episode"),
                              key=lambda r: (r.event_time or "", r.created_at), reverse=True)
            if episodes:
                return [ScoredMemory(episodes[0], 1.0)]
        if TRAJECTORY_INTENT.search(query):
            facts = [r for r in self.store.all() if r.kind == "fact"]
            # Finding the attribute is a recall problem: use the loose retriever.
            hits = self.candidate_retriever.search(query, facts, top_k=top_k, now=now,
                                                   include_inactive=True)
            if hits:
                chain = _distinct_versions(self.store.history(hits[0].record.id))
                if len(chain) > 1:  # the attribute's versions, oldest first
                    return [ScoredMemory(r, 1.0) for r in chain[-top_k:]]
            # No version chain (the model ADDed instead of UPDATEd): rebuild the
            # story from facts and episodes of different times, in time order.
            pool = facts + [r for r in self.store.active() if r.kind == "episode"]
            hits = self.candidate_retriever.search(query, pool, top_k=top_k, now=now,
                                                   include_inactive=True)
            hits.sort(key=lambda h: h.record.event_time or h.record.created_at[:10])
            return hits
        return []

    def retrieve(self, query: str, top_k: int = 5, touch: bool = True,
                 include_history: Optional[bool] = None) -> List[MemoryRecord]:
        """Memories relevant to ``query``; empty when nothing is relevant."""
        recs = [s.record for s in self.search(query, top_k, include_history)]
        if touch and recs:
            now = self.clock()
            for r in recs:
                r.touch(now)
            self.store.save()
        return recs

    def core_profile(self, limit: int = 6) -> List[MemoryRecord]:
        """Identity-level facts (importance 5) that always go in the prompt."""
        core = [r for r in self.store.active() if r.importance >= 5 and r.kind == "fact"]
        core.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
        return core[:limit]

    def pending_promises(self, limit: int = 3) -> List[MemoryRecord]:
        """The assistant's most recent promises, kept in mind like a to-do list."""
        promises = [r for r in self.store.active() if r.kind == "promise"]
        promises.sort(key=lambda r: r.created_at, reverse=True)
        return promises[:limit]

    def active(self) -> List[MemoryRecord]:
        return self.store.active()

    def history(self, mem_id: str) -> List[MemoryRecord]:
        return self.store.history(mem_id)

    @staticmethod
    def describe(r: MemoryRecord) -> str:
        """One memory as text for a prompt, with its date and staleness."""
        if r.kind == "episode":
            return f"（{r.event_time or r.created_at[:10]} 的聊天）{r.content}"
        notes = []
        if r.event_time:
            notes.append(r.event_time)
        elif r.created_at:
            # No event date: say when we learned it, as a person would
            # ("you told me on 08-20 that you reached Starlight").
            notes.append(f"记于{r.created_at[:10]}")
        if not r.is_active:
            notes.append(f"已过时，{(r.valid_to or '')[:10]}被新信息取代")
        return r.content + (f"（{'；'.join(notes)}）" if notes else "")

    @classmethod
    def format_for_prompt(cls, records: Sequence[MemoryRecord]) -> str:
        return "\n".join(f"- {cls.describe(r)}" for r in records)

    def stats(self) -> Dict:
        active = self.store.active()
        by_source: Dict[str, int] = {}
        for r in active:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        return {
            "user_id": self.user_id,
            "active": len(active),
            "archived": len(self.store.records) - len(active),
            "core": len([r for r in active if r.importance >= 5]),
            "by_source": by_source,
        }

    # =============================================================== write

    def ingest(self, text: str, source: str = "chat") -> IngestReport:
        """Extract facts from a conversation or trajectory and store them.
        For chats, optionally also an episode and the assistant's promises."""
        raw = text
        if source == "chat" and self.player_only:
            text = "\n".join(l for l in text.splitlines()
                             if not l.lstrip().startswith(("助手:", "助手：")))
        if source == "chat" and self.write_mode == "slots":
            report = self._ingest_slots(text, source)
        else:
            if source == "chat" and self.per_turn:
                lines = [l for l in text.splitlines() if l.lstrip().startswith(("玩家:", "玩家："))]
                facts = list(dict.fromkeys(f for l in lines for f in self.extract_facts(l, source)))
            else:
                facts = self.extract_facts(text, source)
            report = IngestReport(facts=facts)
            if facts:
                candidates = self._related(facts)
                ops = self._decide(facts, candidates)
                self.apply(ops, candidates, source, report, source_text=text)
        if source == "chat" and self.episodes:
            self._write_episode(text, report)
        if source == "chat" and self.promises:
            self._write_promises(raw, report)
        if self.consolidate:
            report.consolidated = consolidate_versions(self.store.all(), self.store.history,
                                                       fmt_time(self.clock()))
            if report.consolidated:
                self.store.save()
        return report

    def _write_episode(self, text: str, report: IngestReport) -> None:
        today = self.clock()
        data = parse_json(self.llm.chat(
            prompt=prompts.EPISODE.format(text=text.strip(), today=today.strftime("%Y-%m-%d")),
            system=prompts.SYSTEM_JSON, temperature=0.1, json_schema=prompts.EPISODE_SCHEMA,
            max_tokens=self.max_output_tokens))
        summary = str(data.get("summary") or "").strip() if isinstance(data, dict) else ""
        if not summary:
            return
        now = fmt_time(today)
        rec = MemoryRecord(content=summary, keywords=[str(k) for k in (data.get("keywords") or [])][:6],
                           source="chat", importance=2, kind="episode", created_at=now, updated_at=now,
                           event_time=today.strftime("%Y-%m-%d"))
        self.store.put(rec)
        self.store.save()
        report.added.append(rec)

    def _write_promises(self, text: str, report: IngestReport) -> None:
        if not any(l.lstrip().startswith(("助手:", "助手：")) for l in text.splitlines()):
            return
        today = self.clock()
        data = parse_json(self.llm.chat(
            prompt=prompts.PROMISES.format(text=text.strip(), today=today.strftime("%Y-%m-%d")),
            system=prompts.SYSTEM_JSON, temperature=0.1, json_schema=prompts.PROMISES_SCHEMA,
            max_tokens=self.max_output_tokens))
        items = data.get("promises", []) if isinstance(data, dict) else []
        now = fmt_time(today)
        changed = False
        for p in items:
            content = str(p).strip() if isinstance(p, str) else ""
            if not content.startswith("助手") or self._find_duplicate(content, kind="promise"):
                continue
            rec = MemoryRecord(content=content, source="chat", importance=4, kind="promise",
                               created_at=now, updated_at=now)
            self.store.put(rec)
            report.added.append(rec)
            changed = True
        if changed:
            self.store.save()

    def extract_facts(self, text: str, source: str = "chat") -> List[str]:
        self._need_llm()
        today = self.clock()
        template = prompts.EXTRACT_CHAT if source == "chat" else prompts.EXTRACT_TRAJECTORY
        prompt = template.format(text=text.strip(), today=today.strftime("%Y-%m-%d"),
                                 yesterday=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
                                 before_yesterday=(today - timedelta(days=2)).strftime("%Y-%m-%d"))
        data = parse_json(self.llm.chat(prompt=prompt, system=prompts.SYSTEM_JSON,
                                        temperature=0.1, json_schema=prompts.FACTS_SCHEMA,
                                        max_tokens=self.max_output_tokens))
        facts = data.get("facts", []) if isinstance(data, dict) else []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def apply(self, ops: Sequence[Dict], candidates: Sequence[MemoryRecord],
              source: str = "chat", report: Optional[IngestReport] = None,
              source_text: Optional[str] = None) -> IngestReport:
        """Validate and execute operations. ``target`` is a 1-based index into
        ``candidates`` — the only memories the model was shown. With
        ``source_text``, a rank/device value the text never mentions is
        rejected as invented."""
        report = report or IngestReport()
        now = fmt_time(self.clock())
        touched: set = set()

        for op in ops:
            kind = str(op.get("op", "")).upper()
            target = self._resolve(op.get("target"), candidates)
            content = str(op.get("content") or "").strip()

            if kind == "NOOP":
                report.noop += 1
                continue
            if kind in ("UPDATE", "DELETE"):
                if target is None:
                    report.rejected.append((op, "target is not one of the shown memories"))
                    continue
                if not target.is_active or target.id in touched:
                    report.rejected.append((op, "target was already changed in this batch"))
                    continue
            if kind in ("ADD", "UPDATE") and not content:
                report.rejected.append((op, "empty content"))
                continue
            known = source_text if kind == "ADD" or target is None else f"{source_text}\n{target.content}"
            if kind in ("ADD", "UPDATE") and source_text is not None and not _grounded(content, known):
                report.rejected.append((op, "states a value the conversation never mentions"))
                continue

            if kind == "ADD":
                dup = self._find_duplicate(content)
                if dup is not None:
                    report.noop += 1
                    continue
                rec = self._new_record(op, content, source, now)
                self.store.put(rec)
                report.added.append(rec)
            elif kind == "UPDATE":
                a, b = _norm(content), _norm(target.content)
                if a == b or SequenceMatcher(None, a, b).ratio() >= self.duplicate_ratio:
                    report.noop += 1  # a restatement ("玩家的段位是铂金") is not a new version
                    continue
                rec = self._new_record(op, content, source, now)
                self._supersede(target, rec, now)
                self.store.put(rec)
                touched.update({target.id, rec.id})
                report.updated.append((target, rec))
            elif kind == "DELETE":
                target.valid_to = now
                target.updated_at = now
                touched.add(target.id)
                report.deleted.append(target)
            else:
                report.rejected.append((op, f"unknown op {kind!r}"))

        if report.changed:
            self.store.save()
        return report

    def add(self, content: str, keywords: Sequence[str] = (), importance: int = 3,
            source: str = "manual", event_time: Optional[str] = None) -> MemoryRecord:
        now = fmt_time(self.clock())
        rec = MemoryRecord(content=content, keywords=list(keywords), source=source,
                           importance=clamp_importance(importance), created_at=now,
                           updated_at=now, event_time=event_time)
        self.store.put(rec)
        self.store.save()
        return rec

    def forget(self, mem_id: str, hard: bool = False) -> bool:
        """Retire a memory. ``hard=True`` erases it and its whole version
        chain from disk (for privacy / right-to-be-forgotten requests)."""
        rec = self.store.get(mem_id)
        if rec is None:
            return False
        if hard:
            for r in self.store.history(mem_id):
                del self.store.records[r.id]
        elif rec.is_active:
            rec.valid_to = rec.updated_at = fmt_time(self.clock())
        self.store.save()
        return True

    def _ingest_slots(self, text: str, source: str) -> IngestReport:
        self._need_llm()
        today = self.clock()
        prompt = prompts.EXTRACT_SLOTS.format(
            text=text.strip(), today=today.strftime("%Y-%m-%d"),
            yesterday=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
            before_yesterday=(today - timedelta(days=2)).strftime("%Y-%m-%d"),
            aspects="、".join(prompts.ASPECTS))
        data = parse_json(self.llm.chat(prompt=prompt, system=prompts.SYSTEM_JSON,
                                        temperature=0.1, json_schema=prompts.SLOTS_SCHEMA,
                                        max_tokens=self.max_output_tokens))
        items = data.get("facts", []) if isinstance(data, dict) else []
        items = [f for f in items if isinstance(f, dict) and str(f.get("statement") or "").strip()]
        report = IngestReport(facts=[str(f["statement"]).strip() for f in items])
        now = fmt_time(today)

        for f in items:
            content = str(f["statement"]).strip()
            aspect = f.get("aspect") if f.get("aspect") in prompts.ASPECTS else "其他"
            if self._find_duplicate(content) is not None:
                report.noop += 1
                continue
            event = f.get("event_date")
            op = {"keywords": f.get("keywords") or [], "importance": f.get("importance", 3),
                  "event_time": event if isinstance(event, str) and _DATE.match(event) else None}
            rec = self._new_record(op, content, source, now)
            rec.aspect = aspect
            olds = ([r for r in self.store.active() if r.aspect == aspect]
                    if aspect in prompts.SINGLE_ASPECTS else [])
            for old in olds:
                self._supersede(old, rec, now)
                report.updated.append((old, rec))
            if not olds:
                report.added.append(rec)
            self.store.put(rec)

        if report.changed:
            self.store.save()
        return report

    # ============================================================ helpers

    @staticmethod
    def _supersede(old: MemoryRecord, new: MemoryRecord, now: str) -> None:
        new.supersedes = old.id
        new.access_count = max(new.access_count, old.access_count)
        new.last_accessed_at = new.last_accessed_at or old.last_accessed_at
        old.valid_to = now
        old.superseded_by = new.id
        old.updated_at = now

    def _need_llm(self) -> None:
        if self.llm is None:
            raise RuntimeError("this operation needs an LLM client")

    def _related(self, facts: Sequence[str]) -> List[MemoryRecord]:
        seen: Dict[str, MemoryRecord] = {}
        active = [r for r in self.store.active() if r.kind == "fact"]
        for fact in facts:
            for s in self.candidate_retriever.search(fact, active, top_k=self.related_per_fact,
                                                    now=self.clock()):
                seen.setdefault(s.record.id, s.record)
                if len(seen) >= self.max_candidates:
                    return list(seen.values())
        return list(seen.values())

    def _decide(self, facts: Sequence[str], candidates: Sequence[MemoryRecord]) -> List[Dict]:
        existing = "\n".join(f"[{i}] {r.content}" for i, r in enumerate(candidates, 1)) or "（无）"
        prompt = prompts.DECIDE_OPS.format(
            today=self.clock().strftime("%Y-%m-%d"),
            existing=existing,
            facts="\n".join(f"- {f}" for f in facts))
        data = parse_json(self.llm.chat(prompt=prompt, system=prompts.SYSTEM_JSON,
                                        temperature=0.1, json_schema=prompts.OPS_SCHEMA,
                                        max_tokens=self.max_output_tokens))
        ops = data.get("operations", []) if isinstance(data, dict) else []
        return [o for o in ops if isinstance(o, dict)]

    @staticmethod
    def _resolve(target, candidates: Sequence[MemoryRecord]) -> Optional[MemoryRecord]:
        try:
            idx = int(str(target).strip("[] "))
        except (TypeError, ValueError):
            return None
        return candidates[idx - 1] if 1 <= idx <= len(candidates) else None

    def _find_duplicate(self, content: str, kind: str = "fact") -> Optional[MemoryRecord]:
        key = _norm(content)
        for r in self.store.active():
            if r.kind != kind:
                continue
            other = _norm(r.content)
            if key == other or SequenceMatcher(None, key, other).ratio() >= self.duplicate_ratio:
                return r
        return None

    @staticmethod
    def _new_record(op: Dict, content: str, source: str, now: str) -> MemoryRecord:
        kws = op.get("keywords") or []
        if isinstance(kws, str):
            kws = [k.strip() for k in re.split(r"[,，、]", kws) if k.strip()]
        event_time = op.get("event_time") or None
        return MemoryRecord(content=content, keywords=[str(k) for k in kws][:6], source=source,
                            importance=clamp_importance(op.get("importance", 3)),
                            created_at=now, updated_at=now,
                            event_time=str(event_time) if event_time else None)
