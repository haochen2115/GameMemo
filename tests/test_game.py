# -*- coding: utf-8 -*-
"""Game ontology, ledger, GameMemory and the game benchmark's judge. No LLM."""

from datetime import datetime

from bench.gamesim import make_specs, simulate
from bench.run_game import judge
from gamememo.game.ledger import GameLedger
from gamememo.game.memory import GameMemory
from gamememo.game.ontology import KING_START, level_start, parse_ranks, rank_name, season_of


def m(time, hero="孙尚香", win=True, before=40, after=41, mode="排位", pos="发育路"):
    return {"time": time, "season": season_of(datetime.fromisoformat(time)).name, "mode": mode, "hero": hero,
            "position": pos, "win": win, "k": 5, "d": 3, "a": 6, "mvp": False, "rank_before": before,
            "rank_after": after, "duo": None}


def test_rank_ladder_round_trips():
    for tier, sub in (("黄金", 4), ("铂金", 2), ("钻石", 1), ("星耀", 5)):
        p = level_start(tier, sub)
        assert rank_name(p, stars=False) == f"{tier}{'一二三四五'[sub - 1]}"
        assert parse_ranks(rank_name(p)) == [(tier, sub)]
    assert rank_name(KING_START + 3) == "王者3星"
    assert parse_ranks("晚上打王者，上铂金2了") == [("铂金", 2)]


def test_ledger_answers_exactly():
    d4 = level_start("钻石", 5) - 1
    led = GameLedger([
        m("2026-06-01 21:00", before=d4 - 1, after=d4),
        m("2026-06-02 21:00", before=d4, after=d4 + 1),                       # first 钻石
        m("2026-06-03 21:00", hero="后羿", win=False, before=d4 + 1, after=d4),
        m("2026-06-03 21:20", hero="后羿", win=False, before=d4, after=d4 - 1),
        m("2026-06-03 21:40", win=False, before=d4 - 1, after=d4 - 2),
        m("2026-06-04 21:00", mode="匹配", before=d4 - 2, after=d4 - 2),
    ])
    now = datetime(2026, 6, 10, 21)
    assert led.current_rank(now)[0] == d4 - 2
    assert rank_name(led.peak(now)[0], stars=False) == "钻石五" and led.peak(now)[1] == "2026-06-02"
    assert led.record(now, hero="孙尚香").games == 4 and led.record(now, hero="后羿").wins == 0
    assert led.losing_streak(now) == (3, "2026-06-03")
    assert led.first_reached(now) == [("钻石", "2026-06-02")]
    assert led.event_dates("我第一次上钻石那天聊了啥", now) == ["2026-06-02"]
    assert led.event_dates("我连跪最多那次在干嘛", now) == ["2026-06-03"]
    card = led.card(now)
    assert "当前段位：铂金一" in card and "最长排位连败：3把" in card and "本赛季最高段位：钻石五" in card


def test_game_memory_links_event_day_to_that_days_chat(tmp_path):
    clock = {"now": datetime(2026, 6, 2, 22)}
    gm = GameMemory("p", storage_dir=str(tmp_path), clock=lambda: clock["now"])
    gm.ingest_chat("玩家: 终于上钻石了！今天还去面试了\n助手: 双喜临门")
    clock["now"] = datetime(2026, 6, 5, 22)
    gm.ingest_chat("玩家: 今天吃了火锅\n助手: 好吃吗")
    d = level_start("钻石", 5)
    gm.ingest_matches([m("2026-06-01 21:00", before=d - 2, after=d - 1), m("2026-06-02 21:00", before=d - 1, after=d)])
    clock["now"] = datetime(2026, 6, 10, 21)
    ctx = gm.context("我第一次上钻石那天跟你说了啥")
    assert "2026-06-02那天的聊天" in ctx and "面试" in ctx and "火锅" not in ctx  # only that day
    ctx = gm.context("我现在什么段位")
    assert "当前段位：钻石五" in ctx and "系统记录" in ctx


def test_rank_claims_above_the_record_are_flagged(tmp_path):
    clock = {"now": datetime(2026, 8, 2, 22)}
    gm = GameMemory("p", storage_dir=str(tmp_path), clock=lambda: clock["now"])
    gm.ingest_chat("玩家: 上赛季我可是打到星耀的\n助手: 厉害\n玩家: 这赛季在钻石五\n助手: 加油")
    d = level_start("钻石", 3)
    gm.ingest_matches([m("2026-05-01 21:00", before=d - 1, after=d), m("2026-08-01 21:00", before=d - 13, after=d - 12)])
    clock["now"] = datetime(2026, 9, 1, 21)
    ctx = gm.context("上赛季我最高到哪")
    assert "上赛季我可是打到星耀的" in ctx and "与系统记录不符：上赛季（S40）最高其实是钻石三" in ctx
    assert "上赛季最高段位：钻石三" in ctx
    assert ctx.count("与系统记录不符") == 1  # 钻石五 is not above the record


def test_judge_by_answer_type():
    rank = {"type": "game", "answer_type": "rank", "key": "钻石五"}
    assert judge(rank, True, "上赛季最高钻石5，你说的星耀不对") == 1.0
    assert judge(rank, True, "你说过打到星耀了，其实是钻石五") == 0.0  # the stale claim comes first
    pct = {"type": "game", "answer_type": "percent", "key": 55}
    assert judge(pct, True, "孙尚香打了120场，胜率57%") == 1.0 and judge(pct, True, "胜率60%") == 0.0
    cnt = {"type": "game", "answer_type": "count", "key": 150}
    assert judge(cnt, True, "2026-07-08之前打了148场") == 1.0
    ex = {"type": "game", "answer_type": "count_exact", "key": 7}
    assert judge(ex, True, "最多连输了七把") == 1.0 and judge(ex, True, "连输6把") == 0.0
    pos = {"type": "game", "answer_type": "name", "key": "游走"}
    assert judge(pos, True, "你主要打辅助") == 1.0
    neg = {"type": "negative", "answer_type": "keyword", "key": []}
    assert judge(neg, False, "不记得") == 1.0 and judge(neg, True, "你说过要去北京") == 0.0


def test_simulation_is_deterministic_and_rank_moves_by_stars():
    spec = make_specs(1, seed=3)[0]
    a, b = simulate(spec), simulate(spec)
    assert a == b and len(a) > 100
    for x in a:
        if x["mode"] == "匹配":
            assert x["rank_before"] == x["rank_after"]
        elif x["rank_before"] > 0:
            assert x["rank_after"] - x["rank_before"] == (1 if x["win"] else -1)
