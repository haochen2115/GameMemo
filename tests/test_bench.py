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


def test_e2e_judge_accepts_iso_dates_for_chinese_keys():
    from bench.run_e2e import judge
    assert judge({"type": "fact", "answer_any": ["6月28"]}, ["玩家的生日是2026-06-28"])[0] == 1.0
    assert judge({"type": "temporal", "answer_any": ["2026-07"]}, ["玩家升到铂金（2026-07-05）"])[0] == 1.0


# SHA-256 of the test players in bench/data/e2e_v1.json as sealed in c58c35a.
E2E_V1_TEST_SHA256 = "a841901abee86647f7454dce0cfeb23daaaaa777478e06f8f3a835ac54a583fa"


def test_e2e_test_split_is_unchanged_since_sealing():
    import hashlib
    import json

    from bench.run_e2e import DATA

    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    test = [p for p in data["players"] if p["split"] == "test"]
    digest = hashlib.sha256(json.dumps(test, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == E2E_V1_TEST_SHA256


def test_gate_rules():
    from bench.gate import check

    sota = {"primary": "S", "min_gain": 0.02, "metrics": {"S": 0.5, "A": 0.9, "St": 0.1},
            "guards": {"A": 0.05, "St": {"tol": 0.05, "lower_is_better": True}}}
    assert check({"S": 0.6, "A": 0.9, "St": 0.1}, sota)[0]
    assert not check({"S": 0.51, "A": 0.9, "St": 0.1}, sota)[0]      # gain too small
    assert not check({"S": 0.6, "A": 0.8, "St": 0.1}, sota)[0]       # guard dropped
    assert not check({"S": 0.6, "A": 0.9, "St": 0.2}, sota)[0]       # stale rate rose
    assert check({"S": 0.6, "A": 0.9, "St": 0.0}, sota)[0]           # stale rate fell: fine
