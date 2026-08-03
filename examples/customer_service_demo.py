"""Minimal multi-tenant example; no LLM is required."""

from gamememo import JsonMemoryStore, QuorumConfig, QuorumMemory, TenantEpisode


memory = QuorumMemory(
    JsonMemoryStore("runs/customer-service"),
    QuorumConfig(min_supporting_tenants=3, tenant_token_salt="demo-only"),
)

for tenant in ("shop-a", "shop-b", "shop-c"):
    memory.record_episode(
        TenantEpisode(
            tenant_id=tenant,
            episode_id=f"{tenant}-refund-1",
            task_type="refund",
            condition_tags=("refund_pending", "card_payment"),
            candidate_action="verify the transaction reference before escalation",
            observed_response=f"A private, tenant-specific response for {tenant}.",
            success=True,
            private_facts=(tenant,),
        )
    )

context = memory.retrieve(
    tenant_id="new-shop",
    task_type="refund",
    condition_tags=("refund_pending", "card_payment"),
)

assert not context.private_episodes
assert context.shared_rules
print(context.shared_rules[0].action)
