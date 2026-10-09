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
    hits = [h for h in mem.retrieve("我周末打什么球") if h.kind == "turn"]
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
    mem.min_k, mem.attribute_index, mem.recency = 3, False, False  # P7 behaviour: generic timeline note
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


def test_attribute_index_recalls_every_stated_value(tmp_path):
    mem, clock = make(tmp_path)
    mem.attribute_index, mem.change_k = True, 3
    lines = ["我段位到黄金三了", "对面小乔大招好疼", "我主玩小乔，现在铂金二", "刚才排位又输了", "今天天气不错",
             "上钻石五了！", "队友挂机了", "排位赛季快结束了"]
    for month, line in enumerate(lines, start=1):
        clock["now"] = datetime(2026, month, 1, 21, 0)
        mem.ingest(f"玩家: {line}\n助手: 好的")
    said = " ".join(mem.describe(h) for h in mem.retrieve("我的段位是怎么变的", top_k=3))
    assert all(t in said for t in ("黄金三", "铂金二", "钻石五"))
    heroes = [h for h in mem.retrieve("我主玩的英雄是怎么变的", top_k=3) if h.kind == "turn"]
    assert "主玩小乔" in heroes[0].content or any("主玩小乔" in h.content for h in heroes)
    assert not any("对面小乔" in h.content for h in heroes[:1])


def test_registered_raw_systems_keep_their_definitions():
    """A candidate's baseline must not drift when RawMemory's defaults change
    (P8 silently inherited P9's default once; see EXPERIMENTS E16)."""
    from bench.run_e2e import SYSTEMS
    flags = {}
    for name in ("p7", "p8", "p9", "p10"):
        mem = SYSTEMS[name]("m", "http://localhost:0", str(tmp_dir(name))).mem
        flags[name] = (mem.attribute_index, mem.state_line, mem.current, mem.recency)
    assert flags == {"p7": (False, True, False, False), "p8": (True, True, False, False),
                     "p9": (True, True, True, False), "p10": (True, True, False, True)}


def tmp_dir(name):
    import tempfile
    return tempfile.mkdtemp(prefix=f"sys_{name}_")
