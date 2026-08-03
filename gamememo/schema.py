"""Typed records for the tenant and shared memory planes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any


def _clean_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = {tag.strip().lower().replace(" ", "_") for tag in tags if tag.strip()}
    return tuple(sorted(cleaned))


@dataclass(frozen=True)
class TenantEpisode:
    """A private customer interaction plus a proposed reusable rule.

    ``observed_response`` and ``private_facts`` never enter the shared plane.
    Only the structured ``candidate_action`` may be considered for promotion.
    """

    tenant_id: str
    episode_id: str
    task_type: str
    condition_tags: tuple[str, ...]
    candidate_action: str
    observed_response: str
    success: bool
    private_facts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id.strip() or not self.episode_id.strip():
            raise ValueError("tenant_id and episode_id are required")
        if not self.task_type.strip() or not self.candidate_action.strip():
            raise ValueError("task_type and candidate_action are required")
        object.__setattr__(self, "task_type", self.task_type.strip().lower())
        object.__setattr__(self, "condition_tags", _clean_tags(self.condition_tags))

    @property
    def proposal_key(self) -> str:
        payload = {
            "task_type": self.task_type,
            "condition_tags": self.condition_tags,
            "candidate_action": self.candidate_action.strip().lower(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
        return sha256(encoded).hexdigest()[:20]

    @property
    def evidence_hash(self) -> str:
        payload = f"{self.tenant_id}|{self.episode_id}|{self.proposal_key}"
        return sha256(payload.encode()).hexdigest()

    def to_private_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["condition_tags"] = list(self.condition_tags)
        data["private_facts"] = list(self.private_facts)
        return data

    @classmethod
    def from_private_dict(cls, data: dict[str, Any]) -> "TenantEpisode":
        return cls(
            tenant_id=data["tenant_id"],
            episode_id=data["episode_id"],
            task_type=data["task_type"],
            condition_tags=tuple(data.get("condition_tags", ())),
            candidate_action=data["candidate_action"],
            observed_response=data.get("observed_response", ""),
            success=bool(data["success"]),
            private_facts=tuple(data.get("private_facts", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SharedRule:
    rule_id: str
    task_type: str
    condition_tags: tuple[str, ...]
    action: str
    support: int
    failures: int
    success_rate: float
    evidence_hashes: tuple[str, ...]

    def matches(self, task_type: str, condition_tags: tuple[str, ...]) -> bool:
        query_tags = set(_clean_tags(condition_tags))
        return self.task_type == task_type.strip().lower() and set(self.condition_tags) <= query_tags

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["condition_tags"] = list(self.condition_tags)
        data["evidence_hashes"] = list(self.evidence_hashes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedRule":
        return cls(
            rule_id=data["rule_id"],
            task_type=data["task_type"],
            condition_tags=tuple(data["condition_tags"]),
            action=data["action"],
            support=int(data["support"]),
            failures=int(data["failures"]),
            success_rate=float(data["success_rate"]),
            evidence_hashes=tuple(data.get("evidence_hashes", ())),
        )


@dataclass(frozen=True)
class MemoryContext:
    tenant_id: str
    private_episodes: tuple[TenantEpisode, ...]
    shared_rules: tuple[SharedRule, ...]
