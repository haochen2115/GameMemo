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


@pytest.mark.parametrize("name", ["e2e_v1.json", "e2e_v2.json", "e2e_v3.json", "e2e_v4.json"])
def test_e2e_dataset_integrity(name):
    import json

    from bench.run_e2e import transcript

    with open(os.path.join(ROOT, "bench", "data", name), encoding="utf-8") as f:
        data = json.load(f)
    assert {p["split"] for p in data["players"]} == {"dev", "test"}
    ids = [q["id"] for p in data["players"] for q in p["questions"]]
    assert len(ids) == len(set(ids))
    for p in data["players"]:
        dates = [s["date"] for s in p["sessions"]]
        assert dates == sorted(dates) and dates[-1] < p["ask_at"]
        assert transcript(p["sessions"][0]).startswith("玩家: ")
        for q in p["questions"]:
            has_key = bool(q["answer_any"] or q.get("answer_all"))
            assert has_key == (q["type"] != "negative")


def test_e2e_judge():
    from bench.run_e2e import judge

    upd = {"type": "update", "answer_any": ["钻石"], "stale_any": ["铂金"]}
    assert judge(upd, ["玩家段位是钻石"]) == (1.0, False)
    assert judge(upd, ["玩家段位是钻石", "玩家段位是铂金"]) == (0.0, True)
    assert judge({"type": "fact", "answer_any": ["护士"]}, ["玩家是护士"])[0] == 1.0
    assert judge({"type": "negative", "answer_any": []}, [])[0] == 1.0
    assert judge({"type": "negative", "answer_any": []}, ["x"])[0] == 0.0
    traj = {"type": "trajectory", "answer_any": [], "answer_all": ["黄金", "钻石"]}
    assert judge(traj, ["玩家升到黄金", "玩家上了钻石"])[0] == 1.0
    assert judge(traj, ["玩家上了钻石"])[0] == 0.0


def test_e2e_judge_accepts_iso_dates_for_chinese_keys():
    from bench.run_e2e import judge
    assert judge({"type": "fact", "answer_any": ["6月28"]}, ["玩家的生日是2026-06-28"])[0] == 1.0
    assert judge({"type": "temporal", "answer_any": ["2026-07"]}, ["玩家升到铂金（2026-07-05）"])[0] == 1.0


# SHA-256 of each dataset's test players as sealed (e2e_v1: c58c35a, e2e_v2: eb8add4, e2e_v3: b15eccc, e2e_v4: 995e09d).
SEALED_TEST_SHA256 = {
    "e2e_v1.json": "a841901abee86647f7454dce0cfeb23daaaaa777478e06f8f3a835ac54a583fa",
    "e2e_v4.json": "3d28c81e90ff352a9ead28c61498ae52b8719762ad5db837bb97871f95ccf802",
    "e2e_v3.json": "ef2596fb3325c78d3c35c0e162576651d72f219f4d0c358b998926671eede455",
    "e2e_v2.json": "7af43f5752f334ce7227663478132049c42b352eb0869690d6043597eb7dc7a9",
}


@pytest.mark.parametrize("name", sorted(SEALED_TEST_SHA256))
def test_e2e_test_split_is_unchanged_since_sealing(name):
    import hashlib
    import json

    with open(os.path.join(ROOT, "bench", "data", name), encoding="utf-8") as f:
        data = json.load(f)
    test = [p for p in data["players"] if p["split"] == "test"]
    digest = hashlib.sha256(json.dumps(test, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == SEALED_TEST_SHA256[name]


def test_gate_rules():
    from bench.gate import check

    sota = {"primary": "S", "min_gain": 0.02, "metrics": {"S": 0.5, "A": 0.9, "St": 0.1},
            "guards": {"A": 0.05, "St": {"tol": 0.05, "lower_is_better": True}}}
    assert check({"S": 0.6, "A": 0.9, "St": 0.1}, sota)[0]
    assert not check({"S": 0.51, "A": 0.9, "St": 0.1}, sota)[0]      # gain too small
    assert not check({"S": 0.6, "A": 0.8, "St": 0.1}, sota)[0]       # guard dropped
    assert not check({"S": 0.6, "A": 0.9, "St": 0.2}, sota)[0]       # stale rate rose
    assert check({"S": 0.6, "A": 0.9, "St": 0.0}, sota)[0]           # stale rate fell: fine


def test_gate_hold_mode_and_candidate_file(tmp_path):
    import json

    from bench.gate import check, main

    sota = {"primary": "S", "min_gain": 0.02, "hold_tol": 0.02, "metrics": {"S": 0.5}, "guards": {}}
    assert check({"S": 0.49}, sota, "hold")[0]
    assert not check({"S": 0.47}, sota, "hold")[0]
    assert not check({"S": 0.51}, sota, "improve")[0]

    # A candidate that only "holds" everything is not promotable.
    rec = {"system": "base", "ref": "x", "dataset": "d.json", "split": "test", "primary": "S",
           "min_gain": 0.02, "metrics": {"S": 0.5}, "guards": {}}
    base = tmp_path / "base"
    base.mkdir()
    (base / "sota.json").write_text(json.dumps(rec))
    (tmp_path / "r.json").write_text(json.dumps({"dataset": "d.json", "split": "test",
                                                 "results": {"sys": {"S": 0.5}}}))
    (tmp_path / "c.json").write_text(json.dumps({"retrieval": {"system": "sys", "mode": "hold"}}))
    argv = ["--candidate", str(tmp_path / "c.json"), "--retrieval-results", str(tmp_path / "r.json"),
            "--base-dir", str(base)]
    assert main(argv) == 1
    (tmp_path / "c.json").write_text(json.dumps({"retrieval": {"system": "sys", "mode": "improve"}}))
    (tmp_path / "r.json").write_text(json.dumps({"dataset": "d.json", "split": "test",
                                                 "results": {"sys": {"S": 0.6}}}))
    assert main(argv) == 0


def test_retrieval_v2_test_split_is_unchanged_since_sealing():
    """Test queries and test players as sealed in e2d9bce (dev may grow)."""
    import hashlib
    import json

    with open(os.path.join(ROOT, "bench", "data", "retrieval_v2.json"), encoding="utf-8") as f:
        d = json.load(f)
    sealed = [q for q in d["queries"] if q["split"] == "test"] + \
             [p for p in d["players"] if p["id"] != "p_archer"]
    digest = hashlib.sha256(json.dumps(sealed, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == "4763e81c58a336a82149b5e7b543bf697a338d59f6632d193cad2b29c86151ac"


def test_gate_falls_back_when_benchmark_moves_to_a_new_dataset(tmp_path):
    import json

    from bench.gate import load_record

    (tmp_path / "sota_e2e.json").write_text(json.dumps({"dataset": "bench/data/e2e_v1.json", "system": "old"}))
    rec = load_record("e2e", str(tmp_path), "bench/data/some_new_dataset.json")
    assert rec["system"] != "old"          # shipped baseline, not the stale record
    rec = load_record("e2e", str(tmp_path), "bench/data/e2e_v1.json")
    assert rec["system"] == "old"
