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


def test_e2e_dataset_and_judge():
    import json

    from bench.run_e2e import DATA, judge, transcript

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    splits = {p["split"] for p in data["players"]}
    assert splits == {"dev", "test"}
    for p in data["players"]:
        dates = [s["date"] for s in p["sessions"]]
        assert dates == sorted(dates) and dates[-1] < p["ask_at"]
        assert transcript(p["sessions"][0]).startswith("玩家: ")
        for q in p["questions"]:
            assert bool(q["answer_any"]) == (q["type"] != "negative")

    upd = {"type": "update", "answer_any": ["钻石"], "stale_any": ["铂金"]}
    assert judge(upd, ["玩家段位是钻石"]) == (1.0, False)
    assert judge(upd, ["玩家段位是钻石", "玩家段位是铂金"]) == (0.0, True)
    assert judge({"type": "fact", "answer_any": ["护士"]}, ["玩家是护士"])[0] == 1.0
    assert judge({"type": "negative", "answer_any": []}, [])[0] == 1.0
    assert judge({"type": "negative", "answer_any": []}, ["x"])[0] == 0.0
