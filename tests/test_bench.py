# -*- coding: utf-8 -*-
"""Guards on the benchmark itself (data integrity + lexical smoke run)."""

from bench.run_retrieval import evaluate, load, v1_searcher


def test_dataset_integrity():
    data, records, now = load()
    ids = {r.id for r in records}
    active = {r.id for r in records if r.is_active}
    assert now is not None
    for q in data["queries"]:
        assert q["split"] in ("dev", "test")
        assert set(q["gold"]) <= active, f"{q['id']} points at a missing or superseded memory"
        assert bool(q["gold"]) == (q["type"] != "negative")
    for r in records:
        if r.superseded_by:
            assert r.superseded_by in ids and not r.is_active


def test_lexical_retriever_runs_without_embeddings():
    data, records, now = load()
    res = evaluate(v1_searcher(records, now), data["queries"])
    assert res["Abstain"] == 1.0
    assert res["Recall@3"] > 0.5
