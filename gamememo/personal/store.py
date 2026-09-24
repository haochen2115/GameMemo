# -*- coding: utf-8 -*-
"""JSON file storage for one player's memories."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Iterable, List, Optional

from .model import MemoryRecord


SCHEMA_VERSION = 1


class JsonMemoryStore:
    """All records for one user in a single JSON file.

    Saves are atomic (write to a temp file, then ``os.replace``) so a crash
    mid-write never leaves a truncated memory file behind. Reads accept the
    v0 file format and upgrade it on the next save.
    """

    def __init__(self, path: str):
        self.path = path
        self.records: Dict[str, MemoryRecord] = {}
        self.load()

    # ---- queries ----

    def get(self, mem_id: str) -> Optional[MemoryRecord]:
        return self.records.get(mem_id)

    def all(self) -> List[MemoryRecord]:
        return list(self.records.values())

    def active(self) -> List[MemoryRecord]:
        return [r for r in self.records.values() if r.is_active]

    def history(self, mem_id: str) -> List[MemoryRecord]:
        """Oldest-to-newest chain of versions that ``mem_id`` belongs to."""
        rec = self.records.get(mem_id)
        if rec is None:
            return []
        seen = {rec.id}
        while rec.supersedes and rec.supersedes in self.records and rec.supersedes not in seen:
            rec = self.records[rec.supersedes]
            seen.add(rec.id)
        chain = [rec]
        seen = {rec.id}
        while rec.superseded_by and rec.superseded_by in self.records and rec.superseded_by not in seen:
            rec = self.records[rec.superseded_by]
            seen.add(rec.id)
            chain.append(rec)
        return chain

    # ---- mutations ----

    def put(self, rec: MemoryRecord) -> None:
        self.records[rec.id] = rec

    def extend(self, recs: Iterable[MemoryRecord]) -> None:
        for r in recs:
            self.put(r)

    # ---- persistence ----

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = {}
        for d in data.get("memories", []):
            rec = MemoryRecord.from_dict(d)
            self.records[rec.id] = rec

    def save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        data = {
            "schema_version": SCHEMA_VERSION,
            "memories": [r.to_dict() for r in self.records.values()],
        }
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
