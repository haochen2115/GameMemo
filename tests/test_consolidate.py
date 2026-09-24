# -*- coding: utf-8 -*-
from gamememo.personal.consolidate import consolidate, detect
from gamememo.personal.model import MemoryRecord
from gamememo.personal.store import JsonMemoryStore


def rec(content, day, **kw):
    return MemoryRecord(content=content, created_at=f"{day} 21:00:00", updated_at=f"{day} 21:00:00", **kw)


def test_detect_rank_and_device():
    assert detect(rec("玩家段位是白银一", "2026-01-01")) == [("段位", "白银一")]
    assert detect(rec("玩家从黄金升到了钻石", "2026-01-01")) == [("段位", "钻石")]
    assert detect(rec("玩家的目标是冲王者", "2026-01-01")) == []          # a wish, not a rank
    assert detect(rec("玩家很喜欢王者荣耀", "2026-01-01")) == []          # the game's name
    assert detect(rec("玩家手机换成了小米14", "2026-01-01")) == [("设备", "小米")]
    assert detect(rec("玩家想买iPhone", "2026-01-01")) == []


def test_consolidate_builds_time_ordered_chain(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "m.json"))
    silver = rec("玩家段位是白银", "2026-01-10")
    plat = rec("玩家上铂金了", "2026-05-10")
    gold = rec("玩家升黄金了", "2026-02-18")
    other = rec("玩家常用瑶", "2026-01-10")
    store.extend([silver, plat, gold, other])
    retired = consolidate(store.all(), store.history, "2026-06-01 00:00:00")
    assert {r.id for r in retired} == {silver.id, gold.id}
    assert [r.content for r in store.history(plat.id)] == ["玩家段位是白银", "玩家升黄金了", "玩家上铂金了"]
    assert plat.is_active and other.is_active and not gold.is_active
    # idempotent
    assert consolidate(store.all(), store.history, "2026-06-02 00:00:00") == []


def test_consolidate_splices_into_existing_llm_chain(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "m.json"))
    gold = rec("玩家段位是黄金", "2026-02-01")
    diamond = rec("玩家上钻石了", "2026-06-01")
    gold.valid_to, gold.superseded_by, diamond.supersedes = "2026-06-01 21:00:00", diamond.id, gold.id
    plat = rec("玩家升铂金了", "2026-04-01")                              # ADDed, not UPDATEd
    store.extend([gold, diamond, plat])
    consolidate(store.all(), store.history, "2026-06-02 00:00:00")
    assert [r.content for r in store.history(diamond.id)] == ["玩家段位是黄金", "玩家升铂金了", "玩家上钻石了"]
    assert not plat.is_active and diamond.is_active


def test_write_path_rejects_invented_values_and_restatements(tmp_path):
    from datetime import datetime
    from gamememo.llm import FakeLLM
    from gamememo.personal import PersonalMemory

    mem = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: datetime(2026, 6, 15),
                         episodes=False, promises=False)
    old = mem.add("玩家段位是黄金", ["段位"], 4)
    ops = [{"op": "UPDATE", "target": 1, "content": "玩家的段位是黄金"},     # restatement
           {"op": "ADD", "content": "玩家的段位是钻石"}]                     # never said
    mem.llm = FakeLLM(lambda p, s: {"operations": ops} if "【新提取的事实】" in p else {"facts": ["玩家儿子中考结束", "玩家段位是黄金"]})
    rep = mem.ingest("玩家: 我儿子中考结束了\n助手: 恭喜")
    assert rep.changed == 0 and rep.noop == 1
    assert [why for _, why in rep.rejected] == ["states a value the conversation never mentions"]
    assert [r.content for r in mem.active()] == ["玩家段位是黄金"] and old.is_active


def test_trajectory_recall_collapses_repeated_values(tmp_path):
    from datetime import datetime
    from gamememo.personal import PersonalMemory

    mem = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: datetime(2026, 9, 1))
    chain = [rec("玩家段位是白银一", "2026-02-01"), rec("玩家段位是黄金三", "2026-02-25"),
             rec("玩家段位是铂金", "2026-05-06"), rec("玩家的段位是铂金", "2026-08-02")]
    for old, new in zip(chain, chain[1:]):
        PersonalMemory._supersede(old, new, new.created_at)
    mem.store.extend(chain)
    got = [r.content for r in mem.retrieve("我的段位这半年是怎么变的", top_k=3)]
    assert got == ["玩家段位是白银一", "玩家段位是黄金三", "玩家段位是铂金"]


def test_rank_values_keep_sub_tier():
    assert detect(rec("玩家升到钻石一了", "2026-01-01")) == [("段位", "钻石一")]
    assert detect(rec("玩家段位是星耀三星", "2026-01-01")) == [("段位", "星耀三星")]
    assert detect(rec("玩家上王者了", "2026-01-01")) == [("段位", "王者")]


def test_composite_facts_are_not_retired(tmp_path):
    store = JsonMemoryStore(str(tmp_path / "m.json"))
    first = rec("玩家是大一新生，在武汉读书，段位黄金一", "2026-02-05")
    plat = rec("玩家升铂金了", "2026-03-10")
    store.extend([first, plat])
    assert consolidate(store.all(), store.history, "2026-04-01 00:00:00") == []
    assert first.is_active and plat.is_active


def test_grounding_ignores_sub_tier_phrasing():
    from gamememo.personal.system import _grounded
    assert _grounded("玩家段位是钻石三星", "玩家: 我升到钻石三了")
    assert not _grounded("玩家段位是星耀", "玩家: 我升到钻石三了")


def test_achievements_with_chong_are_not_wishes():
    assert detect(rec("玩家暑假冲到星耀四了", "2026-01-01")) == [("段位", "星耀四")]
    assert detect(rec("玩家冲上王者了", "2026-01-01")) == [("段位", "王者")]
    assert detect(rec("玩家这赛季要冲王者", "2026-01-01")) == []
