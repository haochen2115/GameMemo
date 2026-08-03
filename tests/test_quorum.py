from dataclasses import replace
import json

from gamememo import JsonMemoryStore, QuorumConfig, QuorumMemory, TenantEpisode


def make_episode(tenant: str, index: int, *, action: str = "verify reference before escalation") -> TenantEpisode:
    return TenantEpisode(
        tenant_id=tenant,
        episode_id=f"ep-{index}",
        task_type="refund",
        condition_tags=("pending", "card"),
        candidate_action=action,
        observed_response=f"Order ORD-{index}991 belongs to {tenant}.",
        success=True,
        private_facts=(f"ORD-{index}991", tenant),
    )


def test_promotes_only_after_distinct_tenant_quorum(tmp_path):
    memory = QuorumMemory(
        JsonMemoryStore(tmp_path),
        QuorumConfig(min_supporting_tenants=3, tenant_token_salt="test"),
    )
    assert memory.record_episode(make_episode("a", 1)).status == "candidate"
    assert memory.record_episode(make_episode("a", 2)).support == 1
    assert memory.record_episode(make_episode("b", 3)).status == "candidate"
    assert memory.record_episode(make_episode("c", 4)).status == "promoted"


def test_retrieval_never_crosses_private_plane(tmp_path):
    memory = QuorumMemory(
        JsonMemoryStore(tmp_path),
        QuorumConfig(min_supporting_tenants=2, tenant_token_salt="test"),
    )
    memory.record_episode(make_episode("a", 1))
    memory.record_episode(make_episode("b", 2))
    context = memory.retrieve("heldout", "refund", ("pending", "card"))
    assert context.private_episodes == ()
    assert len(context.shared_rules) == 1
    serialized = json.dumps(context.shared_rules[0].to_dict())
    assert "ORD-" not in serialized
    assert not memory.audit_shared_plane()


def test_rejects_instance_material_from_shared_candidate(tmp_path):
    memory = QuorumMemory(JsonMemoryStore(tmp_path))
    episode = make_episode("a", 1, action="look up order ORD-1991")
    outcome = memory.record_episode(episode)
    assert outcome.status == "rejected"
    assert memory.store.load_candidate_ledger() == {"candidates": {}}
    assert len(memory.store.load_private_episodes("a")) == 1


def test_rule_conditions_must_match(tmp_path):
    memory = QuorumMemory(
        JsonMemoryStore(tmp_path),
        QuorumConfig(min_supporting_tenants=2, tenant_token_salt="test"),
    )
    memory.record_episode(make_episode("a", 1))
    memory.record_episode(make_episode("b", 2))
    assert not memory.retrieve("c", "refund", ("pending",)).shared_rules


def test_promoted_rule_is_revoked_when_evidence_drops_below_threshold(tmp_path):
    memory = QuorumMemory(
        JsonMemoryStore(tmp_path),
        QuorumConfig(min_supporting_tenants=2, min_success_rate=0.75, tenant_token_salt="test"),
    )
    memory.record_episode(make_episode("a", 1))
    memory.record_episode(make_episode("b", 2))
    assert memory.retrieve("c", "refund", ("pending", "card")).shared_rules
    failed = replace(make_episode("b", 3), success=False)
    assert memory.record_episode(failed).status == "candidate"
    assert not memory.retrieve("c", "refund", ("pending", "card")).shared_rules


def test_tenant_identifier_cannot_be_promoted(tmp_path):
    memory = QuorumMemory(JsonMemoryStore(tmp_path))
    episode = make_episode("merchant-alpha", 1, action="apply merchant-alpha's refund workflow")
    outcome = memory.record_episode(episode)
    assert outcome.status == "rejected"
    assert "tenant identifier" in outcome.reason
