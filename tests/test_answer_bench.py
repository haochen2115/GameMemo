# -*- coding: utf-8 -*-
"""Guards on the answer-level judge (bench/run_answer.py) and the
long-horizon padding (bench/haystack.py). No LLM needed."""

import json
import os

from bench.haystack import pad_player
from bench.run_answer import full_context, judge_answer
from bench.run_e2e import ROOT


def q(type_, any_=(), all_=(), stale=()):
    return {"type": type_, "answer_any": list(any_), "answer_all": list(all_), "stale_any": list(stale)}


def test_fact_needs_known_and_key():
    assert judge_answer(q("fact", ["株洲"]), True, "你在株洲开理发店")[0] == 1.0
    assert judge_answer(q("fact", ["株洲"]), False, "株洲？不太确定")[0] == 0.0
    assert judge_answer(q("fact", ["株洲"]), True, "你在长沙")[0] == 0.0


def test_negative_needs_abstention():
    assert judge_answer(q("negative"), False, "不记得你说过")[0] == 1.0
    assert judge_answer(q("negative"), True, "我没有答应过帮你订票")[0] == 1.0  # short refusal counts
    assert judge_answer(q("negative"), True, "你穿42码的鞋")[0] == 0.0


def test_update_stale_only_ok_when_marked_past():
    item = q("update", ["羽毛球"], stale=["乒乓"])
    assert judge_answer(item, True, "羽毛球")[0] == 1.0
    assert judge_answer(item, True, "你以前打乒乓球，现在改打羽毛球了")[0] == 1.0
    assert judge_answer(item, True, "你打乒乓球和羽毛球")[0] == 0.0


def test_trajectory_needs_every_state_and_tier_spellings():
    item = q("trajectory", all_=["黄金三", "铂金二", "钻石五"])
    assert judge_answer(item, True, "黄金3 → 铂金II → 钻石5")[0] == 1.0
    assert judge_answer(item, True, "从黄金三升到了钻石五")[0] == 0.0


def test_padding_keeps_real_sessions_and_spreads_filler():
    with open(os.path.join(ROOT, "bench", "data", "e2e_v10.json"), encoding="utf-8") as f:
        player = [p for p in json.load(f)["players"] if p["split"] == "test"][0]
    filler = [{"id": f"h{i}", "turns": [{"role": "user", "content": f"填充{i}"}]} for i in range(30)]
    padded = pad_player(player, filler, 20)
    assert len(padded["sessions"]) == len(player["sessions"]) + 20
    real = [s for s in padded["sessions"] if "filler" not in s]
    assert real == player["sessions"]
    dates = [s["date"] for s in padded["sessions"]]
    assert dates == sorted(dates) and dates[-1] < player["ask_at"]
    assert not {s["date"][:10] for s in real} & {s["date"][:10] for s in padded["sessions"] if "filler" in s}
    assert pad_player(player, filler, 20) == padded  # deterministic
    assert "填充" in full_context(padded)
