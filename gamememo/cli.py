"""Command-line entry points for the research preview."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .commons import run_commons, write_results
from .quorum import QuorumConfig, QuorumMemory
from .schema import TenantEpisode
from .store import JsonMemoryStore


def _demo(storage: Path) -> int:
    memory = QuorumMemory(
        JsonMemoryStore(storage),
        QuorumConfig(min_supporting_tenants=3, tenant_token_salt="demo-only"),
    )
    action = "verify the transaction reference before escalation"
    for index, tenant in enumerate(("alpha", "bravo", "charlie"), start=1):
        outcome = memory.record_episode(
            TenantEpisode(
                tenant_id=tenant,
                episode_id=f"refund-{index}",
                task_type="refund",
                condition_tags=("refund_pending", "card_payment"),
                candidate_action=action,
                observed_response=f"Private response for {tenant}, order ORD-{index}004.",
                success=True,
                private_facts=(f"ORD-{index}004", tenant),
            )
        )
        print(f"{tenant:>7}: {outcome.status} (support={outcome.support})")
    context = memory.retrieve("new-tenant", "refund", ("card_payment", "refund_pending"))
    print("\nHeld-out tenant context")
    print(f"  private episodes: {len(context.private_episodes)}")
    print(f"  shared rules:     {len(context.shared_rules)}")
    if context.shared_rules:
        print(f"  action:           {context.shared_rules[0].action}")
    violations = memory.audit_shared_plane()
    print(f"  isolation audit:  {'PASS' if not violations else 'FAIL'}")
    return int(bool(violations))


def _commons(output: Path | None) -> int:
    results = run_commons()
    print(json.dumps([asdict(row) for row in results], indent=2))
    if output:
        write_results(output, results)
        print(f"\nwrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gamememo", description="GameMemo 2.0 research preview")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the QUORUM promotion demo")
    demo.add_argument("--storage", type=Path, default=Path("runs/demo"))
    commons = subparsers.add_parser("commons", help="run the deterministic COMMONS fixture")
    commons.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _demo(args.storage)
    return _commons(args.output)
