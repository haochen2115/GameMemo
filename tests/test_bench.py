# -*- coding: utf-8 -*-
"""Guards on the benchmark itself (data integrity + lexical smoke run)."""

import os

import pytest

from bench.run_retrieval import ROOT, SYSTEMS, load, run_system

DATASETS = ["retrieval_v1.json", "retrieval_v2.json"]


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_integrity(name):
    data, banks = load(os.path.join(ROOT, "bench", "data", name))
    for pid, (records, now) in banks.items():
        assert now is not None
        ids = {r.id for r in records}
        for r in records:
            if r.superseded_by:
                assert r.superseded_by in ids and not r.is_active
    qids = [q["id"] for q in data["queries"]]
    assert len(qids) == len(set(qids))
    for q in data["queries"]:
        active = {r.id for r in banks[q["player"]][0] if r.is_active}
        assert q["split"] in ("dev", "test")
        assert set(q["gold"]) <= active, f"{q['id']} points at a missing or superseded memory"
        assert bool(q["gold"]) == (q["type"] != "negative")


def test_v2_test_players_are_disjoint_from_dev():
    data, _ = load()
    dev = {q["player"] for q in data["queries"] if q["split"] == "dev"}
    test = {q["player"] for q in data["queries"] if q["split"] == "test"}
    assert dev and test and not dev & test


def test_lexical_retriever_runs_without_embeddings():
    data, banks = load()
    dev = [q for q in data["queries"] if q["split"] == "dev"]
    res = run_system(SYSTEMS["v1-lexical"], banks, dev)
    assert res["Recall@3"] > 0.5
    assert res["Abstain"] > 0.8
