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

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)


_NORM = re.compile(r"[\s\W_]+", re.UNICODE)


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
                 duplicate_ratio: float = 0.9):
        self.user_id = user_id
        self.llm = llm
        self.clock = clock
        self.related_per_fact = related_per_fact
        self.max_candidates = max_candidates
        self.duplicate_ratio = duplicate_ratio
        self.store = JsonMemoryStore(os.path.join(storage_dir, f"{user_id}_memory.json"))
        if retrieval_config is None:
            retrieval_config = RetrievalConfig() if embedder else RetrievalConfig.lexical_only()
        self.retriever = HybridRetriever(embedder=embedder, config=retrieval_config)
        self.candidate_retriever = HybridRetriever(embedder=embedder,
                                                   config=retrieval_config.for_candidates())

    # ================================================================ read

    def search(self, query: str, top_k: int = 5) -> List[ScoredMemory]:
        return self.retriever.search(query, self.store.active(), top_k=top_k, now=self.clock())

    def retrieve(self, query: str, top_k: int = 5, touch: bool = True) -> List[MemoryRecord]:
        """Memories relevant to ``query``; empty when nothing is relevant."""
        recs = [s.record for s in self.search(query, top_k)]
        if touch and recs:
            now = self.clock()
            for r in recs:
                r.touch(now)
            self.store.save()
        return recs

    def core_profile(self, limit: int = 6) -> List[MemoryRecord]:
        """Identity-level facts (importance 5) that always go in the prompt."""
        core = [r for r in self.store.active() if r.importance >= 5]
        core.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
        return core[:limit]

    def active(self) -> List[MemoryRecord]:
        return self.store.active()

    def history(self, mem_id: str) -> List[MemoryRecord]:
        return self.store.history(mem_id)

    @staticmethod
    def format_for_prompt(records: Sequence[MemoryRecord]) -> str:
        lines = []
        for r in records:
            when = f"（{r.event_time}）" if r.event_time else ""
            lines.append(f"- {r.content}{when}")
        return "\n".join(lines)

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
        """Extract facts from a conversation or trajectory and store them."""
        facts = self.extract_facts(text, source)
        report = IngestReport(facts=facts)
        if not facts:
            return report
        candidates = self._related(facts)
        ops = self._decide(facts, candidates)
        self.apply(ops, candidates, source, report)
        return report

    def extract_facts(self, text: str, source: str = "chat") -> List[str]:
        self._need_llm()
        today = self.clock()
        template = prompts.EXTRACT_CHAT if source == "chat" else prompts.EXTRACT_TRAJECTORY
        prompt = template.format(text=text.strip(), today=today.strftime("%Y-%m-%d"),
                                 yesterday=(today - timedelta(days=1)).strftime("%Y-%m-%d"))
        data = parse_json(self.llm.chat(prompt=prompt, system=prompts.SYSTEM_JSON,
                                        temperature=0.1, json_schema=prompts.FACTS_SCHEMA))
        facts = data.get("facts", []) if isinstance(data, dict) else []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def apply(self, ops: Sequence[Dict], candidates: Sequence[MemoryRecord],
              source: str = "chat", report: Optional[IngestReport] = None) -> IngestReport:
        """Validate and execute operations. ``target`` is a 1-based index into
        ``candidates`` — the only memories the model was shown."""
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

            if kind == "ADD":
                dup = self._find_duplicate(content)
                if dup is not None:
                    report.noop += 1
                    continue
                rec = self._new_record(op, content, source, now)
                self.store.put(rec)
                report.added.append(rec)
            elif kind == "UPDATE":
                if _norm(content) == _norm(target.content):
                    report.noop += 1
                    continue
                rec = self._new_record(op, content, source, now)
                rec.supersedes = target.id
                rec.access_count = target.access_count
                rec.last_accessed_at = target.last_accessed_at
                target.valid_to = now
                target.superseded_by = rec.id
                target.updated_at = now
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

    # ============================================================ helpers

    def _need_llm(self) -> None:
        if self.llm is None:
            raise RuntimeError("this operation needs an LLM client")

    def _related(self, facts: Sequence[str]) -> List[MemoryRecord]:
        seen: Dict[str, MemoryRecord] = {}
        active = self.store.active()
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
                                        temperature=0.1, json_schema=prompts.OPS_SCHEMA))
        ops = data.get("operations", []) if isinstance(data, dict) else []
        return [o for o in ops if isinstance(o, dict)]

    @staticmethod
    def _resolve(target, candidates: Sequence[MemoryRecord]) -> Optional[MemoryRecord]:
        try:
            idx = int(str(target).strip("[] "))
        except (TypeError, ValueError):
            return None
        return candidates[idx - 1] if 1 <= idx <= len(candidates) else None

    def _find_duplicate(self, content: str) -> Optional[MemoryRecord]:
        key = _norm(content)
        for r in self.store.active():
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
