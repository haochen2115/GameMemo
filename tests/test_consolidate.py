# -*- coding: utf-8 -*-
from gamememo.personal.consolidate import consolidate, detect
from gamememo.personal.model import MemoryRecord
from gamememo.personal.store import JsonMemoryStore


def rec(content, day, **kw):
    return MemoryRecord(content=content, created_at=f"{day} 21:00:00", updated_at=f"{day} 21:00:00", **kw)


def test_detect_rank_and_device():
    assert detect(rec("玩家段位是白银一", "2026-01-01")) == [("段位", "白银")]
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
