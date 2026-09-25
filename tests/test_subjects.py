# -*- coding: utf-8 -*-
"""Subject-aware recall: who a question is about."""
from datetime import datetime

from gamememo.personal import MemoryRecord, PersonalMemory
from gamememo.personal.retrieval import RetrievalConfig
from gamememo.personal.subjects import fact_subject, mentions, query_subject

NOW = datetime(2026, 9, 1, 21, 0)


def test_question_and_fact_subjects():
    assert query_subject("我姐姐是做什么的") == "姐姐"
    assert query_subject("我哥是做什么的") == "哥哥"
    assert query_subject("我女儿几岁了") == "女儿"
    assert query_subject("我老婆做什么的") == query_subject("我女朋友呢") == "伴侣"
    assert query_subject("我是做什么的") is None
    assert fact_subject("玩家的妹妹在读大二") == "妹妹"
    assert fact_subject("玩家是牙医") is None
    assert mentions("玩家的弟弟在读书", "弟弟")
    assert not mentions("玩家表弟要来住几天", "弟弟")      # a cousin is someone else
    assert not mentions("玩家婆婆是医生", "公公")


def make(tmp_path, *facts, **kw):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW, subject_recall=True,
                       retrieval_config=RetrievalConfig.lexical_only(concept_tags=True, concept_cover=True), **kw)
    m.store.extend(MemoryRecord(content=c, keywords=k) for c, k in facts)
    return m


def test_question_about_a_person_only_sees_that_person(tmp_path):
    m = make(tmp_path, ("玩家哥哥是警察", ["警察"]), ("玩家是程序员", ["程序员"]))
    assert m.retrieve("我姐姐是做什么的") == []
    assert [r.content for r in m.retrieve("我哥是做什么的")] == ["玩家哥哥是警察"]


def test_question_about_the_player_ignores_other_peoples_facts(tmp_path):
    m = make(tmp_path, ("玩家的妹妹在读大二", ["大二"]), ("玩家是程序员", ["程序员"]))
    assert "玩家的妹妹在读大二" not in [r.content for r in m.retrieve("我在读大几")]


def test_partner_words_are_interchangeable(tmp_path):
    m = make(tmp_path, ("玩家的女朋友在奶茶店当店长", ["奶茶店", "店长"]))
    assert [r.content for r in m.retrieve("我老婆做什么的")] == ["玩家的女朋友在奶茶店当店长"]


def test_off_by_default(tmp_path):
    m = PersonalMemory("p", storage_dir=str(tmp_path), clock=lambda: NOW)
    recs = [MemoryRecord(content="玩家哥哥是警察", keywords=["警察"])]
    assert m._about_subject("我姐姐是做什么的", recs) == recs
