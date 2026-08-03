"""QUORUM promotion and retrieval policy."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import re

from .schema import MemoryContext, SharedRule, TenantEpisode
from .store import JsonMemoryStore


@dataclass(frozen=True)
class QuorumConfig:
    min_supporting_tenants: int = 3
    min_success_rate: float = 0.75
    max_failures: int = 1
    tenant_token_salt: str = "replace-me-in-production"

    def __post_init__(self) -> None:
        if self.min_supporting_tenants < 2:
            raise ValueError("shared rules require support from at least two tenants")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between zero and one")


@dataclass(frozen=True)
class RecordOutcome:
    proposal_key: str
    status: str
    support: int
    failures: int
    reason: str = ""


class RuleSafetyPolicy:
    """Reject obvious instance material before it reaches the candidate ledger."""

    PATTERNS = (
        re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
        re.compile(r"\b(?:order|ticket|account|case)[-_ #:]?[a-z0-9]{4,}\b", re.I),
        re.compile(r"\b\+?\d[\d ()-]{7,}\d\b"),
        re.compile(r"\b\d{1,5}\s+[A-Z][\w.-]+\s+(?:st|street|rd|road|ave|avenue)\b", re.I),
    )

    def validate(self, episode: TenantEpisode) -> tuple[bool, str]:
        candidate = " ".join((episode.candidate_action, *episode.condition_tags))
        lowered = candidate.casefold()
        if len(episode.tenant_id.strip()) >= 3 and episode.tenant_id.casefold() in lowered:
            return False, "candidate contains its tenant identifier"
        for fact in episode.private_facts:
            fact = fact.strip()
            if len(fact) >= 3 and fact.casefold() in lowered:
                return False, "candidate contains a declared private fact"
        if any(pattern.search(candidate) for pattern in self.PATTERNS):
            return False, "candidate resembles instance-identifying material"
        return True, ""


class QuorumMemory:
    """Promote reusable rules while retaining raw instances per tenant."""

    def __init__(
        self,
        store: JsonMemoryStore,
        config: QuorumConfig | None = None,
        safety_policy: RuleSafetyPolicy | None = None,
    ) -> None:
        self.store = store
        self.config = config or QuorumConfig()
        self.safety_policy = safety_policy or RuleSafetyPolicy()

    def _tenant_token(self, tenant_id: str) -> str:
        return hmac.new(
            self.config.tenant_token_salt.encode(),
            tenant_id.encode(),
            sha256,
        ).hexdigest()

    def record_episode(self, episode: TenantEpisode) -> RecordOutcome:
        self.store.append_private_episode(episode)
        safe, reason = self.safety_policy.validate(episode)
        if not safe:
            return RecordOutcome(episode.proposal_key, "rejected", 0, 0, reason)

        ledger = self.store.load_candidate_ledger()
        candidates = ledger.setdefault("candidates", {})
        candidate = candidates.setdefault(
            episode.proposal_key,
            {
                "task_type": episode.task_type,
                "condition_tags": list(episode.condition_tags),
                "action": episode.candidate_action,
                "tenant_outcomes": {},
                "evidence_hashes": [],
            },
        )
        token = self._tenant_token(episode.tenant_id)
        candidate["tenant_outcomes"][token] = bool(episode.success)
        shared_evidence_hash = sha256(
            f"{token}|{episode.episode_id}|{episode.proposal_key}".encode()
        ).hexdigest()
        if shared_evidence_hash not in candidate["evidence_hashes"]:
            candidate["evidence_hashes"].append(shared_evidence_hash)
        self.store.save_candidate_ledger(ledger)

        successes = sum(candidate["tenant_outcomes"].values())
        failures = len(candidate["tenant_outcomes"]) - successes
        total = successes + failures
        success_rate = successes / total if total else 0.0
        promoted = (
            successes >= self.config.min_supporting_tenants
            and success_rate >= self.config.min_success_rate
            and failures <= self.config.max_failures
        )
        status = "promoted" if promoted else "candidate"
        if promoted:
            self._upsert_rule(episode.proposal_key, candidate, successes, failures, success_rate)
        else:
            self._remove_rule(episode.proposal_key)
        return RecordOutcome(episode.proposal_key, status, successes, failures)

    def _remove_rule(self, proposal_key: str) -> None:
        rules = self.store.load_shared_rules()
        retained = [rule for rule in rules if rule.rule_id != proposal_key]
        if len(retained) != len(rules):
            self.store.save_shared_rules(retained)

    def _upsert_rule(
        self,
        proposal_key: str,
        candidate: dict,
        successes: int,
        failures: int,
        success_rate: float,
    ) -> None:
        rules = [rule for rule in self.store.load_shared_rules() if rule.rule_id != proposal_key]
        rules.append(
            SharedRule(
                rule_id=proposal_key,
                task_type=candidate["task_type"],
                condition_tags=tuple(candidate["condition_tags"]),
                action=candidate["action"],
                support=successes,
                failures=failures,
                success_rate=success_rate,
                evidence_hashes=tuple(candidate["evidence_hashes"]),
            )
        )
        self.store.save_shared_rules(sorted(rules, key=lambda rule: rule.rule_id))

    def retrieve(
        self,
        tenant_id: str,
        task_type: str,
        condition_tags: tuple[str, ...],
    ) -> MemoryContext:
        private = tuple(
            episode
            for episode in self.store.load_private_episodes(tenant_id)
            if episode.task_type == task_type.strip().lower()
        )
        shared = tuple(
            sorted(
                (
                    rule
                    for rule in self.store.load_shared_rules()
                    if rule.matches(task_type, condition_tags)
                ),
                key=lambda rule: (-len(rule.condition_tags), -rule.support, rule.rule_id),
            )
        )
        return MemoryContext(tenant_id=tenant_id, private_episodes=private, shared_rules=shared)

    def audit_shared_plane(self) -> list[str]:
        """Return violations instead of silently claiming the boundary is safe."""
        violations: list[str] = []
        ledger = self.store.load_candidate_ledger()
        forbidden_keys = {"tenant_id", "private_facts", "observed_response", "metadata"}
        for key, candidate in ledger.get("candidates", {}).items():
            overlap = forbidden_keys & set(candidate)
            if overlap:
                violations.append(f"candidate {key} contains private keys: {sorted(overlap)}")
            text = " ".join((candidate.get("action", ""), *candidate.get("condition_tags", [])))
            if any(pattern.search(text) for pattern in RuleSafetyPolicy.PATTERNS):
                violations.append(f"candidate {key} contains instance-like text")
        return violations
