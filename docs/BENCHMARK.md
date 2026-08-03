# COMMONS benchmark specification

COMMONS evaluates a memory system across an ordered stream of tenants. Training episodes accumulate in one persistent memory state; held-out tenants are then evaluated without resetting that shared state.

## Evaluation unit

Each item contains:

- a tenant identity visible only to the private plane;
- task and condition tags;
- a private interaction and outcome;
- a generalized candidate action;
- foreign private facts and exception actions used only by the verifier.

## Metrics

Let a held-out set contain \(N\) queries.

- **Experience gain**: fraction of queries on which the response contains the verified reusable action, reported relative to the same agent without accumulated memory in model experiments.
- **Instance leakage**: fraction containing any foreign tenant canary or private fact.
- **Pseudo-generalization**: fraction applying an explicitly scoped exception outside its valid scope.

Always report all three. A scalar utility may be convenient for search, but it must not hide movement along the transfer–contamination frontier.

## Partitions for a full release

1. In-domain transfer: familiar task, new tenant.
2. Cross-domain transfer: related task language, new tenant and organization.
3. Instance canaries: synthetic identifiers that must never cross tenants.
4. Rare valid exceptions: successful actions whose scope is intentionally narrow.
5. Contradictory evidence: policies that change by region, plan, or time.
6. Deletion and revocation: remove a tenant and recompute support/provenance.

The checked-in fixture is deliberately small and deterministic. It is a contract test for scoring and state boundaries, not a substitute for the real corpus.
