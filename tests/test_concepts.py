# -*- coding: utf-8 -*-
from datetime import datetime

from gamememo.personal import PersonalMemory, RetrievalConfig
from gamememo.personal.concepts import memory_concepts, query_concepts
from gamememo.personal.model import MemoryRecord


def test_concept_detection():
    assert memory_concepts("玩家是公交车司机") == ["职业"]
    assert "城市" in memory_concepts("玩家住在长沙")
    assert query_concepts("我是做什么工作的") == ["职业"]
    assert query_concepts("我喜欢什么颜色") == []


def test_concept_tags_bridge_category_questions(tmp_path):
    def mem(**kw):
        m = PersonalMemory("p", storage_dir=str(tmp_path / str(len(kw))), clock=lambda: datetime(2026, 9, 1),
                           retrieval_config=RetrievalConfig.lexical_only(**kw))
        m.store.extend([MemoryRecord(content="玩家是护士", keywords=["护士"]),
                        MemoryRecord(content="玩家常用瑶", keywords=["瑶"])])
        return m
    assert mem().retrieve("我是做什么工作的") == []
    assert [r.content for r in mem(concept_tags=True).retrieve("我是做什么工作的")] == ["玩家是护士"]
    assert mem(concept_tags=True).retrieve("我喜欢什么颜色") == []


def test_episode_fallback_only_when_no_fact(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: datetime(2026, 9, 1), episode_fallback=True)
    m.store.put(MemoryRecord(content="玩家说膝盖做了个小手术，在家休养", keywords=["膝盖", "手术"],
                             kind="episode", event_time="2026-03-20", created_at="2026-03-20 10:00:00"))
    assert [r.kind for r in m.retrieve("我膝盖怎么了")] == ["episode"]
    m.episode_fallback = False
    assert m.retrieve("我膝盖怎么了") == []
