# -*- coding: utf-8 -*-
"""Rule-based promises and one-line trajectories (P6 candidate)."""
from datetime import datetime

from gamememo.personal import MemoryRecord, PersonalMemory
from gamememo.personal.system import PROMISE_INTENT_WIDE, promises_from_rules

NOW = datetime(2026, 9, 28, 21, 0)


def test_promise_patterns():
    text = "玩家: 戈娅老被抓\n助手: 下次我给你整理一份戈娅走位躲技能的思路。\n助手: 加油，你可以的！"
    assert promises_from_rules(text) == ["助手答应下次我给你整理一份戈娅走位躲技能的思路"]
    assert promises_from_rules("玩家: 下次我帮你带奶茶") == []      # the player's own lines are not promises
    assert PROMISE_INTENT_WIDE.search("你之前说要帮我整理什么")
    assert not PROMISE_INTENT_WIDE.search("我说要去北京")


def test_trajectory_is_one_line_with_every_state(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW, trajectory_summary=True)
    for day, text in [("2026-01-05", "玩家段位黄金二"), ("2026-03-05", "玩家升到铂金五"),
                      ("2026-05-05", "玩家掉到铂金二"), ("2026-08-05", "玩家上了钻石四")]:
        m.store.put(MemoryRecord(content=text, created_at=f"{day} 21:00:00", updated_at=f"{day} 21:00:00"))
    first = m.retrieve("我的段位是怎么变的")[0].content
    assert all(v in first for v in ("黄金二", "铂金五", "铂金二", "钻石四"))
    assert m.retrieve("我现在什么段位")[0].content == "玩家上了钻石四"   # a real memory when it is unambiguous
    m.store.put(MemoryRecord(content="玩家从钻石四掉回铂金一了", created_at="2026-09-01 21:00:00",
                             updated_at="2026-09-01 21:00:00"))
    assert m.retrieve("我现在什么段位")[0].content.startswith("现在的段位：铂金一")


def test_main_hero_changes_are_tracked(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW, trajectory_summary=True)
    for day, text in [("2026-01-05", "玩家主玩鲁班七号"), ("2026-03-05", "玩家对面的关羽太烦了"),
                      ("2026-04-05", "玩家最近改玩狄仁杰"), ("2026-07-05", "玩家现在只玩戈娅")]:
        m.store.put(MemoryRecord(content=text, created_at=f"{day} 21:00:00", updated_at=f"{day} 21:00:00"))
    first = m.retrieve("我主玩的英雄是怎么变的")[0].content
    assert "鲁班七号" in first and "狄仁杰" in first and "戈娅" in first and "关羽" not in first
