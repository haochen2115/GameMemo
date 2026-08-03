"""COMMONS: a deterministic contract benchmark for shared-memory semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile

from .quorum import QuorumConfig, QuorumMemory
from .schema import TenantEpisode
from .store import JsonMemoryStore


@dataclass(frozen=True)
class CommonsQuery:
    tenant_id: str
    task_type: str
    condition_tags: tuple[str, ...]
    expected_action: str
    forbidden_actions: tuple[str, ...]
    foreign_private_facts: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    policy: str
    experience_gain: float
    instance_leakage: float
    pseudo_generalization: float

    @property
    def contamination(self) -> float:
        return self.instance_leakage + self.pseudo_generalization


def commons_fixture() -> tuple[list[TenantEpisode], list[CommonsQuery]]:
    verify = "verify the transaction reference before escalation"
    trace = "open a carrier trace after confirming the delivery address"
    credit = "issue an immediate goodwill credit"
    episodes = [
        TenantEpisode("acme", "b1", "refund", ("refund_pending", "card_payment"), verify,
                      "I checked order ORD-A104 for Ana and will verify its transaction reference.", True,
                      ("ORD-A104", "Ana")),
        TenantEpisode("boreal", "b2", "refund", ("refund_pending", "card_payment"), verify,
                      "For ticket TKT-B778, verify the reference before escalating the $42 refund.", True,
                      ("TKT-B778", "$42")),
        TenantEpisode("cinder", "b3", "refund", ("refund_pending", "card_payment"), verify,
                      "I will verify account ACCT-C991's transaction reference before escalation.", True,
                      ("ACCT-C991",)),
        TenantEpisode("delta", "d1", "delivery", ("package_delayed", "carrier_scan_missing"), trace,
                      "For order ORD-D440 to 12 River St, I opened a carrier trace.", True,
                      ("ORD-D440", "12 River St")),
        TenantEpisode("ember", "d2", "delivery", ("package_delayed", "carrier_scan_missing"), trace,
                      "I confirmed 8 Cedar Road for Mia and opened trace TKT-E220.", True,
                      ("8 Cedar Road", "Mia", "TKT-E220")),
        TenantEpisode("fjord", "d3", "delivery", ("package_delayed", "carrier_scan_missing"), trace,
                      "Carrier trace TKT-F661 is open for order ORD-F119.", True,
                      ("TKT-F661", "ORD-F119")),
        TenantEpisode("vip-one", "x1", "refund", ("refund_pending", "card_payment", "bespoke_sla"), credit,
                      "Under Nimbus Corp's bespoke SLA, issue an immediate $200 goodwill credit.", True,
                      ("Nimbus Corp", "$200")),
    ]
    all_private = tuple(fact for episode in episodes for fact in episode.private_facts)
    queries = [
        CommonsQuery("heldout-a", "refund", ("refund_pending", "card_payment"), verify, (credit,), all_private),
        CommonsQuery("heldout-b", "refund", ("refund_pending", "card_payment"), verify, (credit,), all_private),
        CommonsQuery("heldout-c", "delivery", ("package_delayed", "carrier_scan_missing"), trace, (), all_private),
        CommonsQuery("heldout-d", "delivery", ("package_delayed", "carrier_scan_missing"), trace, (), all_private),
    ]
    return episodes, queries


def _pooled_raw_response(query: CommonsQuery, episodes: list[TenantEpisode]) -> str:
    matches = [episode for episode in episodes if episode.task_type == query.task_type and episode.success]
    if not matches:
        return "follow the standard support workflow"
    episode = matches[-1]
    # A naive pooled system reuses both the inferred action and the raw successful instance.
    return f"{episode.candidate_action}\n{episode.observed_response}"


def _score(policy: str, responses: list[str], queries: list[CommonsQuery]) -> BenchmarkResult:
    total = len(queries)
    helpful = sum(query.expected_action.casefold() in response.casefold() for response, query in zip(responses, queries))
    leaks = sum(
        any(fact.casefold() in response.casefold() for fact in query.foreign_private_facts if len(fact) >= 3)
        for response, query in zip(responses, queries)
    )
    pseudo = sum(
        any(action.casefold() in response.casefold() for action in query.forbidden_actions)
        for response, query in zip(responses, queries)
    )
    return BenchmarkResult(policy, helpful / total, leaks / total, pseudo / total)


def run_commons(storage_dir: str | Path | None = None) -> list[BenchmarkResult]:
    """Run the fixture. These values validate mechanics, not model quality."""
    episodes, queries = commons_fixture()
    temporary = None
    if storage_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="gamememo-commons-")
        storage_dir = temporary.name
    try:
        memory = QuorumMemory(
            JsonMemoryStore(storage_dir),
            QuorumConfig(min_supporting_tenants=3, tenant_token_salt="commons-fixture"),
        )
        for episode in episodes:
            memory.record_episode(episode)

        no_memory = ["follow the standard support workflow" for _ in queries]
        pooled = [_pooled_raw_response(query, episodes) for query in queries]
        quorum = []
        for query in queries:
            context = memory.retrieve(query.tenant_id, query.task_type, query.condition_tags)
            quorum.append(context.shared_rules[0].action if context.shared_rules else no_memory[0])
        return [
            _score("no_memory", no_memory, queries),
            _score("pooled_raw", pooled, queries),
            _score("quorum", quorum, queries),
        ]
    finally:
        if temporary is not None:
            temporary.cleanup()


def write_results(path: str | Path, results: list[BenchmarkResult]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"kind": "deterministic_contract_fixture", "results": [asdict(row) for row in results]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
