# -*- coding: utf-8 -*-
"""Memory record and time helpers."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


TIME_FMT = "%Y-%m-%d %H:%M:%S"


def fmt_time(t: datetime) -> str:
    return t.strftime(TIME_FMT)


def parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in (TIME_FMT, "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


@dataclass
class MemoryRecord:
    """One piece of long-term memory about a player.

    Facts are never overwritten in place. An update closes the old record
    (``valid_to`` + ``superseded_by``) and opens a new one that points back
    via ``supersedes``, so the system can still answer "what was my rank
    last month?" and narrate how a player changed over time.

    ``importance`` runs 1 (trivia) .. 5 (core identity, e.g. birthday).
    """

    content: str
    keywords: List[str] = field(default_factory=list)
    source: str = "chat"
    importance: int = 3
    aspect: Optional[str] = None      # e.g. "当前段位"; single-valued aspects supersede
    id: str = field(default_factory=new_id)
    created_at: str = ""
    updated_at: str = ""
    event_time: Optional[str] = None
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    access_count: int = 0
    last_accessed_at: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.valid_to is None

    def touch(self, now: datetime) -> None:
        self.access_count += 1
        self.last_accessed_at = fmt_time(now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        if "priority" in d or "create_time" in d:
            return cls._from_v0(d)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        known["importance"] = clamp_importance(known.get("importance", 3))
        return cls(**known)

    @classmethod
    def _from_v0(cls, d: Dict[str, Any]) -> "MemoryRecord":
        """Load a record written by the v0 ``game_memory.py`` format.

        v0 used ``priority`` 1 (core) .. 5 (general); invert it.
        """
        keywords = d.get("keywords", "")
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        created = d.get("create_time", "")
        updated = d.get("update_time", created)
        return cls(
            id=d.get("id") or new_id(),
            content=d.get("content", ""),
            keywords=keywords,
            source=d.get("source", "chat"),
            importance=clamp_importance(6 - int(d.get("priority", 3))),
            created_at=created,
            updated_at=updated,
            valid_to=None if d.get("valid", 1) else updated,
            access_count=int(d.get("access_count", 0)),
            last_accessed_at=d.get("last_access_time"),
        )


def clamp_importance(v: Any) -> int:
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 3
