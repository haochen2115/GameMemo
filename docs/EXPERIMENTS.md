# Experiment plan

No model results are claimed in this preview. When running experiments, use Ollama so the model boundary is explicit and reproducible.

## Suggested first matrix

| Axis | Initial values |
|---|---|
| Model | `qwen3.5:9b`, `qwen3.5:27b`, `qwen3.5:35b` |
| Memory | no memory, per-tenant only, pooled raw, pooled summary, QUORUM |
| Quorum support | 2, 3, 5 tenants |
| Temperature | 0 for extraction/judging; 0.2 for response generation |
| Seeds | at least 3 |

Start with the 9B model to close the full data → memory → response → verifier loop. Use 27B/35B only after the metrics and canaries are stable.

## Required controls

- Frozen prompts and model digests.
- Identical ordered streams for every memory condition.
- A no-memory base run and a per-tenant-only run.
- Canary matching plus semantic review for leakage.
- Separate reporting of transfer gain, instance leakage, and pseudo-generalization.
- Cost, latency, memory growth, and promotion delay.
- Bootstrap confidence intervals over tenants, not only over individual turns.

## Commands available now

```bash
python -m gamememo commons --output runs/commons-contract.json
python -m gamememo demo --storage runs/quorum-demo
```

The next implementation milestone is an Ollama-backed runner over a real, redistributable customer-service corpus. Until that corpus and its licenses are fixed, adding a model loop would create an impressive-looking but scientifically weak number.
