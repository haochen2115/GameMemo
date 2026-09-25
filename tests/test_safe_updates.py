# -*- coding: utf-8 -*-
"""Write-side guards against lossy updates (safe_updates)."""
from datetime import datetime

from gamememo.personal import MemoryRecord, PersonalMemory
from gamememo.personal.concepts import category_terms
from gamememo.personal.retrieval import RetrievalConfig

NOW = datetime(2026, 9, 1, 21, 0)


def make(tmp_path, *facts, **kw):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW, safe_updates=True, **kw)
    recs = [m.add(c, k, 3, source="chat") for c, k in facts]
    return m, recs


def active(m):
    return sorted(r.content for r in m.store.active())


def test_update_keeps_clauses_the_new_fact_does_not_touch(tmp_path):
    m, (old,) = make(tmp_path, ("玩家主玩打野，常用赵云", ["打野", "赵云"]))
    m.apply([{"op": "UPDATE", "target": 1, "content": "玩家常用赵云", "keywords": ["赵云"]}], [old],
            source_text="玩家: 最近还是常用赵云")
    assert active(m) == ["玩家主玩打野", "玩家常用赵云"]


def test_real_change_retires_the_old_value(tmp_path):
    m, (old,) = make(tmp_path, ("玩家主玩法师，最喜欢西施", ["法师", "西施"]))
    m.apply([{"op": "UPDATE", "target": 1, "content": "玩家主玩打野", "keywords": ["打野"]}], [old],
            source_text="玩家: 我现在改玩打野了")
    assert active(m) == ["玩家主玩打野", "玩家最喜欢西施"]
    m2, (rank,) = make(tmp_path / "b", ("玩家的段位是钻石五", ["钻石"]))
    m2.apply([{"op": "UPDATE", "target": 1, "content": "玩家的段位是钻石三", "keywords": ["钻石"]}], [rank],
             source_text="玩家: 打到钻石三了")
    assert active(m2) == ["玩家的段位是钻石三"]


def test_fact_about_someone_else_does_not_replace_the_players(tmp_path):
    m, (old,) = make(tmp_path, ("玩家是牙医，在厦门开诊所", ["牙医", "厦门"]))
    m.apply([{"op": "UPDATE", "target": 1, "content": "玩家老婆是银行柜员", "keywords": ["银行"]}], [old],
            source_text="玩家: 我老婆在银行当柜员")
    assert active(m) == ["玩家是牙医，在厦门开诊所", "玩家老婆是银行柜员"]


def test_delete_needs_the_conversation_to_mention_the_memory(tmp_path):
    m, (team, rank) = make(tmp_path, ("玩家是学校篮球队的", ["篮球队"]), ("玩家段位是钻石四", ["钻石四"]))
    rep = m.apply([{"op": "DELETE", "target": 1}], [team, rank], source_text="玩家: 我上钻石一了")
    assert rep.rejected and team.is_active
    m.apply([{"op": "DELETE", "target": 1}], [team, rank], source_text="玩家: 我退出篮球队了")
    assert not team.is_active


def test_guards_are_off_by_default(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW)
    old = m.add("玩家主玩打野，常用赵云", ["打野", "赵云"], 3, source="chat")
    m.apply([{"op": "UPDATE", "target": 1, "content": "玩家常用赵云"}], [old], source_text="常用赵云")
    assert active(m) == ["玩家常用赵云"]


def test_category_word_is_covered_by_the_concept_tag(tmp_path):
    assert category_terms("我是做什么工作的", ["工作"]) == {"职业": ["工作"]}
    assert category_terms("我对什么过敏", ["过敏"]) == {}      # a value word must match literally

    def mem(**kw):
        m = PersonalMemory("p", storage_dir=str(tmp_path / str(len(kw))), clock=lambda: NOW,
                           retrieval_config=RetrievalConfig.lexical_only(concept_tags=True, **kw))
        m.store.extend([MemoryRecord(content="玩家是消防员", keywords=["消防员"]),
                        MemoryRecord(content="玩家说工作日晚上才有空", keywords=["工作日"]),
                        MemoryRecord(content="玩家常用张飞", keywords=["张飞"])])
        return m
    assert mem().retrieve("我是做什么工作的")[0].content != "玩家是消防员"
    assert "玩家是消防员" in [r.content for r in mem(concept_cover=True).retrieve("我是做什么工作的")]
    assert mem(concept_cover=True).retrieve("我对什么过敏") == []


def test_concept_fallback_looks_in_episodes_when_facts_are_off_concept(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW, concept_fallback=True,
                       retrieval_config=RetrievalConfig.lexical_only(concept_tags=True, concept_cover=True,
                                                                     min_lexical_coverage=0.05))
    m.store.extend([MemoryRecord(content="玩家上班很累", keywords=["上班"]),
                    MemoryRecord(content="玩家说自己是会计，住在苏州，主玩辅助", keywords=["苏州"],
                                 kind="episode", event_time="2026-03-12")])
    assert "苏州" in " ".join(r.content for r in m.retrieve("我在哪个城市上班"))
    m.concept_fallback = False
    assert "苏州" not in " ".join(r.content for r in m.retrieve("我在哪个城市上班"))
