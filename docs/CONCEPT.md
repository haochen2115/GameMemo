# From personal memory to shared memory

Personal memory has one principal: remember facts about the current user accurately. Shared memory has two principals that can conflict:

1. **Transfer** — reuse successful experience from earlier tenants on a new tenant.
2. **Isolation** — never carry an earlier tenant's instance or unjustified exception into the new tenant.

GameMemo 2.0 treats this as a granularity problem rather than a storage problem. Raw episodes remain in a physically separate tenant plane. The shared plane receives only structured, conditional action candidates. A candidate becomes a rule after independent successful support from enough tenants.

## Two contamination modes

| Failure | Definition | Why it matters |
|---|---|---|
| Instance leakage | Another tenant's name, order, address, account, or other concrete fact appears in the current context or answer. | Safety and privacy failure. |
| Pseudo-generalization | A valid tenant-specific exception is promoted into a rule for tenants that do not satisfy its scope. | Capability and reliability failure. |

Access-control labels alone do not solve either problem. They can hide a row while still allowing a summarizer to write its contents into a globally readable summary. QUORUM therefore constrains the *write path*, not only the read path.

## QUORUM contract

- A private episode stores the observed response, private facts, outcome, and a proposed reusable action.
- The candidate ledger stores no raw tenant identifier, observed response, private fact, or metadata.
- Support counts distinct salted tenant tokens, not repeated episodes from one tenant.
- A promoted rule is conditional: task type + condition tags + generalized action.
- Retrieval joins only the current tenant's private plane with matching promoted rules.
- The included regex policy is defense in depth, not a complete PII detector. Production deployments should add domain-specific classification, deletion, encryption, and access auditing.

## Non-claims

This repository is an executable research preview. Its fixture proves that the API exposes the intended isolation and promotion semantics. It does **not** prove that QUORUM improves a particular model or production system. That requires the model-based protocol in [EXPERIMENTS.md](EXPERIMENTS.md).
