# Threat model

## Protected boundary

Tenant A's observed responses, identifiers, addresses, account facts, and metadata must not be retrievable by tenant B. A shared rule may reveal a reusable workflow, but not the tenant instances that supported it.

## Covered by this preview

- Separate files for private tenant episodes and the shared plane.
- Hashed tenant filenames and salted tenant tokens in support counts.
- Candidate rejection for declared private facts and common identifier patterns.
- Conditional rules and distinct-tenant promotion thresholds.
- An audit that rejects private-schema keys in the shared ledger.

## Not yet covered

- Encryption at rest or a hardware-backed secret for tenant-token hashing.
- Robust multilingual PII detection.
- Poisoning, collusion, Sybil tenants, or malicious feedback.
- Provenance deletion when a tenant invokes erasure rights.
- Time-bounded policies, regional policy conflicts, and rollback.
- Side channels through embeddings, logs, caches, or model weights.

Treat the current JSON store as an inspectable reference adapter. Production deployments should place the two planes in separately authorized services and make promotion an audited job.
