# -*- coding: utf-8 -*-
"""RawMemory (P7): verbatim exchanges, rule promises, chatbot drop-in."""

from datetime import datetime

from gamememo.llm import FakeLLM
from gamememo.personal import MemoryChatBot
from gamememo.personal.raw import RawMemory, exchanges


def make(tmp_path, when=datetime(2026, 3, 1, 21, 0)):
    clock = {"now": when}
    mem = RawMemory("p1", storage_dir=str(tmp_path), clock=lambda: clock["now"])
    return mem, clock


def test_exchanges_pair_player_line_with_reply():
    text = "助手: 欢迎回来\n玩家: 我在株洲开理发店\n助手: 辛苦了\n\n玩家: 段位黄金三"
    assert exchanges(text) == ["助手: 欢迎回来", "玩家: 我在株洲开理发店\n助手: 辛苦了", "玩家: 段位黄金三"]


def test_ingest_is_verbatim_and_never_overwrites(tmp_path):
    mem, clock = make(tmp_path)
    mem.ingest("玩家: 周末去打乒乓球\n助手: 挺好")
    clock["now"] = datetime(2026, 6, 1, 21, 0)
    mem.ingest("玩家: 现在周末改打羽毛球了\n助手: 换换口味")
    turns = sorted((r for r in mem.store.all() if r.kind == "turn"), key=lambda r: r.created_at)
    assert [t.content for t in turns] == ["玩家: 周末去打乒乓球\n助手: 挺好", "玩家: 现在周末改打羽毛球了\n助手: 换换口味"]
    assert all(t.is_active for t in turns)
    hits = mem.retrieve("我周末打什么球")
    assert [h.created_at[:10] for h in hits] == ["2026-03-01", "2026-06-01"]  # oldest first, dated
    assert mem.describe(hits[-1]).startswith("【2026-06-01 21:00】玩家: 现在")
    reloaded = RawMemory("p1", storage_dir=str(tmp_path))
    assert len(reloaded.store.all()) == 2


def test_promises_are_kept_once_and_offered_to_the_chatbot(tmp_path):
    mem, _ = make(tmp_path)
    text = "玩家: 下周要交燃气费\n助手: 好，周五晚上我提醒你交燃气费"
    mem.ingest(text)
    mem.ingest(text)
    promises = mem.pending_promises()
    assert len(promises) == 1 and "燃气费" in promises[0].content
    assert mem.core_profile() == []


def test_change_questions_get_a_wider_window(tmp_path):
    mem, clock = make(tmp_path)
    for month, tier in enumerate(["黄金三", "铂金二", "铂金四", "钻石五", "钻石三", "星耀五"], start=1):
        clock["now"] = datetime(2026, month, 1, 21, 0)
        mem.ingest(f"玩家: 我段位到{tier}了\n助手: 恭喜")
    mem.min_k = 3
    assert len(mem.retrieve("我段位是多少", top_k=3)) == 3
    hits = mem.retrieve("我段位是怎么变的", top_k=3)
    assert [h.kind for h in hits] == ["note"] + ["turn"] * 6 and "6次" in mem.describe(hits[0])


def test_hard_forget_removes_the_exchange(tmp_path):
    mem, _ = make(tmp_path)
    rec = mem.ingest("玩家: 我身份证号是123\n助手: 收到").added[0]
    assert mem.forget(rec.id, hard=True)
    assert RawMemory("p1", storage_dir=str(tmp_path)).store.all() == []


def test_drop_in_for_memory_chatbot(tmp_path):
    mem, _ = make(tmp_path)
    mem.ingest("玩家: 我在株洲开理发店\n助手: 两个人撑一家店，挺能干的。好，周五我提醒你交燃气费")
    llm = FakeLLM(lambda prompt, system: "记得呀")
    bot = MemoryChatBot(mem, llm, extract_every=1)
    turn = bot.chat("我在哪里开理发店来着")
    system = llm.calls[-1]["system"]
    assert "株洲" in system and "燃气费" in system
    assert turn.report is not None and any("我在哪里开理发店来着" in r.content for r in turn.report.added)
