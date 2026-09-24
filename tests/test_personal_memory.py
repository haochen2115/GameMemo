# -*- coding: utf-8 -*-
import json
from datetime import datetime

import pytest

from gamememo.llm import FakeLLM, parse_json
from gamememo.personal import MemoryChatBot, MemoryRecord, PersonalMemory
from gamememo.personal.store import JsonMemoryStore

NOW = datetime(2026, 9, 1, 21, 0, 0)


def scripted(facts=(), ops=(), reply="好的"):
    """FakeLLM answering extraction / decision / chat prompts."""
    def handler(prompt, system):
        if "【新提取的事实】" in prompt:
            return {"operations": list(ops)}
        if "值得长期记住" in prompt:
            return {"facts": list(facts)}
        return reply
    return FakeLLM(handler)


def make(tmp_path, llm=None, **kw):
    return PersonalMemory("p1", llm=llm, storage_dir=str(tmp_path), clock=lambda: NOW, **kw)


def seed(mem, *items):
    recs = []
    for content, kws, imp in items:
        recs.append(mem.add(content, kws, imp, source="chat"))
    return recs


# ------------------------------------------------------------------ write

def test_ingest_adds_facts_and_injects_today(tmp_path):
    llm = scripted(facts=["玩家生日是2月12日"],
                   ops=[{"op": "ADD", "content": "玩家生日是2月12日", "keywords": ["生日"], "importance": 5}])
    mem = make(tmp_path, llm)
    report = mem.ingest("玩家: 我生日2月12日")
    assert [r.content for r in report.added] == ["玩家生日是2月12日"]
    assert mem.active()[0].importance == 5
    assert "2026-09-01" in llm.calls[0]["prompt"]           # today is in the prompt
    assert "2026-08-31" in llm.calls[0]["prompt"]           # so is yesterday
    assert llm.calls[0]["json_schema"] is not None           # structured output


def test_update_supersedes_and_keeps_history(tmp_path):
    mem = make(tmp_path)
    old, = seed(mem, ("玩家段位是星耀三星", ["段位", "星耀"], 4))
    old.access_count = 7
    mem.llm = scripted(facts=["玩家升到了王者段位"],
                       ops=[{"op": "UPDATE", "target": 1, "content": "玩家段位是王者", "keywords": ["段位", "王者"]}])
    report = mem.ingest("玩家: 我上王者了！")

    (was, now_rec), = report.updated
    assert was.id == old.id and not was.is_active and was.superseded_by == now_rec.id
    assert now_rec.supersedes == old.id and now_rec.access_count == 7
    assert [r.content for r in mem.active()] == ["玩家段位是王者"]
    assert [r.content for r in mem.history(now_rec.id)] == ["玩家段位是星耀三星", "玩家段位是王者"]


def test_hallucinated_target_is_rejected_not_counted(tmp_path):
    mem = make(tmp_path)
    seed(mem, ("玩家段位是星耀三星", ["段位"], 4))
    mem.llm = scripted(facts=["玩家段位是王者"],
                       ops=[{"op": "UPDATE", "target": 9, "content": "玩家段位是王者"},
                            {"op": "DELETE", "target": "mem_nonexistent"}])
    report = mem.ingest("...")
    assert report.changed == 0
    assert len(report.rejected) == 2
    assert [r.content for r in mem.active()] == ["玩家段位是星耀三星"]


def test_decision_prompt_only_shows_related_memories(tmp_path):
    mem = make(tmp_path)
    seed(mem,
         ("玩家段位是星耀三星", ["段位", "星耀"], 4),
         ("玩家最讨厌遇到挂机的队友", ["挂机", "队友"], 3),
         ("玩家拥有后羿的皮肤精灵王", ["皮肤", "后羿"], 2))
    llm = scripted(facts=["玩家段位升到王者"], ops=[])
    mem.llm = llm
    mem.ingest("玩家: 我段位升到王者了")
    decide_prompt = llm.calls[1]["prompt"]
    assert "星耀三星" in decide_prompt
    assert "挂机" not in decide_prompt and "精灵王" not in decide_prompt


def test_near_duplicate_add_is_skipped(tmp_path):
    mem = make(tmp_path)
    seed(mem, ("玩家生日是2月12日", ["生日"], 5))
    mem.llm = scripted(facts=["玩家生日是2月12日。"],
                       ops=[{"op": "ADD", "content": "玩家生日是2月12日。", "keywords": ["生日"]}])
    report = mem.ingest("...")
    assert report.added == [] and report.noop == 1
    assert len(mem.active()) == 1


def test_delete_retires_but_hard_forget_erases(tmp_path):
    mem = make(tmp_path)
    a, b = seed(mem, ("玩家用安卓手机", ["手机"], 2), ("玩家生日是2月12日", ["生日"], 5))
    assert mem.forget(a.id)
    assert not mem.store.get(a.id).is_active
    assert mem.forget(b.id, hard=True)
    reloaded = JsonMemoryStore(mem.store.path)
    assert b.id not in reloaded.records and a.id in reloaded.records


def test_garbage_llm_output_changes_nothing(tmp_path):
    mem = make(tmp_path, FakeLLM(lambda p, s: "抱歉，我无法完成"))
    report = mem.ingest("玩家: 你好")
    assert report.facts == [] and report.changed == 0


# ------------------------------------------------------------------ read

def test_retrieve_is_relevant_and_can_abstain(tmp_path):
    mem = make(tmp_path)
    seed(mem,
         ("玩家生日是2月12日", ["生日"], 5),
         ("玩家最常用鲁班七号，胜率62%", ["鲁班七号", "常用英雄"], 4),
         ("玩家最讨厌遇到挂机的队友", ["挂机", "队友"], 3))
    got = mem.retrieve("鲁班七号怎么出装")
    assert got and got[0].content.startswith("玩家最常用鲁班七号")
    # a core memory no longer wins just by being core
    assert all("生日" not in r.content for r in got)
    assert mem.retrieve("Python怎么读取文件") == []
    assert mem.store.get(got[0].id).access_count == 1


def test_retrieval_skips_superseded(tmp_path):
    mem = make(tmp_path)
    old, = seed(mem, ("玩家段位是星耀三星", ["段位"], 4))
    mem.llm = scripted(facts=["玩家段位升到王者"], ops=[{"op": "UPDATE", "target": 1, "content": "玩家段位是王者", "keywords": ["段位"]}])
    mem.ingest("玩家: 我段位升到王者了")
    assert [r.content for r in mem.retrieve("我现在什么段位")] == ["玩家段位是王者"]


# ------------------------------------------------------------------ store

def test_store_reads_v0_format(tmp_path):
    v0 = {"user_id": "p1", "memories": [
        {"id": "mem_1", "keywords": "生日, 2月12日", "content": "玩家生日是2月12日", "source": "chat",
         "valid": 1, "priority": 1, "create_time": "2026-01-01 10:00:00",
         "update_time": "2026-01-01 10:00:00", "access_count": 3, "last_access_time": "2026-01-02 10:00:00"},
        {"id": "mem_2", "keywords": "段位", "content": "旧段位", "valid": 0, "priority": 3,
         "create_time": "2026-01-01 10:00:00", "update_time": "2026-01-05 10:00:00"},
    ]}
    (tmp_path / "p1_memory.json").write_text(json.dumps(v0, ensure_ascii=False), encoding="utf-8")
    mem = make(tmp_path)
    r1, r2 = mem.store.get("mem_1"), mem.store.get("mem_2")
    assert r1.keywords == ["生日", "2月12日"] and r1.importance == 5 and r1.access_count == 3
    assert not r2.is_active
    mem.store.save()
    assert json.loads((tmp_path / "p1_memory.json").read_text(encoding="utf-8"))["schema_version"] == 1


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('好的：{"a": [1, 2]} 以上') == {"a": [1, 2]}
    assert parse_json("no json") is None


# ------------------------------------------------------------------ chatbot

def test_chatbot_uses_memory_and_extracts_only_new_turns(tmp_path):
    mem = make(tmp_path)
    seed(mem, ("玩家生日是2月12日", ["生日"], 5), ("玩家最常用鲁班七号", ["鲁班七号"], 4))
    llm = scripted(facts=[], reply="收到")
    mem.llm = llm
    bot = MemoryChatBot(mem, llm, extract_every=2, context_turns=0)

    t1 = bot.chat("鲁班七号怎么玩")
    system = llm.calls[-1]["system"]
    assert "2026-09-01" in system and "星期二" in system
    assert "玩家生日是2月12日" in system                      # core profile
    assert [r.content for r in t1.retrieved] == ["玩家最常用鲁班七号"]
    assert t1.report is None

    t2 = bot.chat("好的谢谢你")
    assert t2.report is not None
    first_extract = [c["prompt"] for c in llm.calls if "值得长期记住" in c["prompt"]][0]
    assert "鲁班七号怎么玩" in first_extract

    bot.chat("第三句话")
    bot.chat("第四句话")
    second_extract = [c["prompt"] for c in llm.calls if "值得长期记住" in c["prompt"]][1]
    assert "第三句话" in second_extract and "鲁班七号怎么玩" not in second_extract


def test_record_roundtrip():
    r = MemoryRecord(content="x", keywords=["a"], importance=9)
    assert MemoryRecord.from_dict(r.to_dict()) == MemoryRecord(**{**r.to_dict(), "importance": 5})


def test_chatbot_survives_llm_failures_and_retries_extraction(tmp_path):
    state = {"fail": True}

    def handler(prompt, system):
        if state["fail"]:
            raise RuntimeError("timed out")
        if "【新提取的事实】" in prompt:
            return {"operations": [{"op": "ADD", "content": "玩家主玩打野", "keywords": ["打野"]}]}
        if "值得长期记住" in prompt:
            return {"facts": ["玩家主玩打野"]}
        return "好的"

    llm = FakeLLM(handler)
    mem = make(tmp_path, llm)
    bot = MemoryChatBot(mem, llm, extract_every=1)
    t1 = bot.chat("我主玩打野")
    assert t1.reply == MemoryChatBot.FALLBACK_REPLY
    assert len(t1.errors) == 2 and t1.report is None

    state["fail"] = False
    t2 = bot.chat("记住哦")
    assert t2.errors == []
    extract_prompt = [c["prompt"] for c in llm.calls if "值得长期记住" in c["prompt"]][-1]
    assert "我主玩打野" in extract_prompt            # the failed turn was retried
    assert [r.content for r in mem.active()] == ["玩家主玩打野"]


# ------------------------------------------------------------------ slot mode / history

def slot_llm(facts):
    return FakeLLM(lambda p, s: {"facts": facts})


def test_slots_supersede_single_valued_aspects_without_second_call(tmp_path):
    mem = make(tmp_path, write_mode="slots", episodes=False, promises=False)
    mem.llm = slot_llm([{"aspect": "当前段位", "statement": "玩家段位是铂金", "keywords": ["段位", "铂金"],
                         "event_date": "2026-07-05"},
                        {"aspect": "游戏伙伴", "statement": "玩家常和老公双排", "keywords": ["老公"]}])
    rep = mem.ingest("玩家: 我铂金了，和老公双排")
    assert len(rep.added) == 2 and len(mem.llm.calls) == 1
    assert mem.llm.calls[0]["json_schema"]["properties"]["facts"]["items"]["properties"]["aspect"]["enum"]

    mem.llm = slot_llm([{"aspect": "当前段位", "statement": "玩家上了钻石", "keywords": ["段位", "钻石"],
                         "event_date": "not a date"},
                        {"aspect": "游戏伙伴", "statement": "玩家的好友阿杰玩射手", "keywords": ["阿杰"]}])
    rep = mem.ingest("玩家: 我上钻石了，阿杰玩射手")
    (old, new), = rep.updated
    assert old.content == "玩家段位是铂金" and not old.is_active and new.event_time is None
    assert sorted(r.content for r in mem.active()) == ["玩家上了钻石", "玩家常和老公双排", "玩家的好友阿杰玩射手"]


def test_player_only_drops_assistant_lines(tmp_path):
    mem = make(tmp_path, write_mode="slots", player_only=True)
    mem.llm = slot_llm([])
    mem.ingest("玩家: 我是护士\n助手: 你可以多练练站位\n玩家: 好的")
    prompt = mem.llm.calls[0]["prompt"]
    assert "我是护士" in prompt and "多练练站位" not in prompt


def test_history_recall_answers_questions_about_the_past(tmp_path):
    mem = make(tmp_path, write_mode="slots", history_recall=True)
    mem.llm = slot_llm([{"aspect": "当前段位", "statement": "玩家升到了铂金段位", "keywords": ["段位", "铂金"],
                         "event_date": "2026-07-05"}])
    mem.ingest("...")
    mem.llm = slot_llm([{"aspect": "当前段位", "statement": "玩家升到了钻石段位", "keywords": ["段位", "钻石"],
                         "event_date": "2026-09-01"}])
    mem.ingest("...")

    now_q = [r.content for r in mem.retrieve("我现在什么段位")]
    assert now_q == ["玩家升到了钻石段位"]
    past = mem.retrieve("我什么时候升的铂金")
    assert past[0].content == "玩家升到了铂金段位" and not past[0].is_active
    text = mem.format_for_prompt(past)
    assert "2026-07-05" in text and "已过时" in text


def test_per_turn_extraction_reads_each_player_line(tmp_path):
    llm = FakeLLM(lambda p, s: {"facts": []} if "值得长期记住" in p else {"operations": []})
    mem = make(tmp_path, llm, per_turn=True, episodes=False, promises=False)
    mem.ingest("玩家: 我是护士\n助手: 辛苦了\n玩家: 我主玩瑶")
    prompts_seen = [c["prompt"] for c in llm.calls]
    assert len(prompts_seen) == 2
    assert "我是护士" in prompts_seen[0] and "我主玩瑶" not in prompts_seen[0]


def test_candidate_retrieval_stays_recall_oriented():
    from gamememo.personal.retrieval import RetrievalConfig

    strict = RetrievalConfig(require_specific=True, min_dense_alone=0.4)
    loose = strict.for_candidates()
    assert not loose.require_specific and loose.min_lexical_coverage == 0.0


# ------------------------------------------------------------------ P1: episodes, promises, recall modes

def p1_llm(facts=(), ops=(), summary="", promises=()):
    def handler(prompt, system):
        if "【新提取的事实】" in prompt:
            return {"operations": list(ops)}
        if "概括这次聊天" in prompt:
            return {"summary": summary, "keywords": ["生日"]}
        if "答应玩家" in prompt:
            return {"promises": list(promises)}
        if "值得长期记住" in prompt:
            return {"facts": list(facts)}
        return "好的"
    return FakeLLM(handler)


def test_episode_and_promise_are_written_per_conversation(tmp_path):
    mem = make(tmp_path, episodes=True, promises=True)
    mem.llm = p1_llm(facts=["玩家今天生日"], ops=[{"op": "ADD", "content": "玩家今天生日", "keywords": ["生日"]}],
                     summary="玩家说今天是生日，用狄仁杰拿了五杀",
                     promises=["助手答应下次帮玩家复盘", "祝你生日快乐"])
    rep = mem.ingest("玩家: 今天我生日，拿了五杀\n助手: 生日快乐！下次我帮你复盘")
    kinds = sorted(r.kind for r in rep.added)
    assert kinds == ["episode", "fact", "promise"]            # the greeting is not a promise
    ep = next(r for r in mem.active() if r.kind == "episode")
    assert ep.event_time == "2026-09-01" and "（2026-09-01 的聊天）" in mem.describe(ep)
    # the promise prompt sees the assistant's words even with player_only
    assert "下次我帮你复盘" in [c["prompt"] for c in mem.llm.calls if "答应玩家" in c["prompt"]][0]
    # episodes and promises never become UPDATE targets or core profile
    assert all(r.kind == "fact" for r in mem.core_profile())


def test_recall_modes(tmp_path):
    mem = make(tmp_path, recall_modes=True)
    old = mem.add("玩家段位是黄金", ["段位", "黄金"], 4)
    mid = MemoryRecord(content="玩家升到了铂金", keywords=["段位", "铂金"], created_at="2026-07-01 10:00:00")
    new = MemoryRecord(content="玩家上了钻石", keywords=["段位", "钻石"], created_at="2026-09-01 10:00:00")
    PersonalMemory._supersede(old, mid, "2026-07-01 10:00:00")
    PersonalMemory._supersede(mid, new, "2026-09-01 10:00:00")
    mem.store.extend([mid, new])
    for i, day in enumerate(["2026-08-01", "2026-08-20"]):
        mem.store.put(MemoryRecord(content=f"玩家聊了第{i + 1}件事", kind="episode", event_time=day,
                                   created_at=day + " 10:00:00"))
    mem.store.put(MemoryRecord(content="助手答应下次帮玩家复盘", kind="promise", created_at="2026-08-01 10:00:00"))

    assert [r.content for r in mem.retrieve("我的段位是怎么一路变化的")] == \
        ["玩家段位是黄金", "玩家升到了铂金", "玩家上了钻石"]
    assert [r.content for r in mem.retrieve("上次我们聊了什么")] == ["玩家聊了第2件事"]
    assert [r.content for r in mem.retrieve("你答应过我什么")] == ["助手答应下次帮玩家复盘"]
    assert [r.content for r in mem.retrieve("我现在什么段位")] == ["玩家上了钻石"]


def test_episodes_only_answer_questions_about_a_time(tmp_path):
    mem = make(tmp_path, recall_modes=True)
    mem.add("玩家段位是钻石", ["段位", "钻石"], 4)
    mem.store.put(MemoryRecord(content="玩家说自己段位是黄金，还拿了五杀", keywords=["段位", "五杀"],
                               kind="episode", event_time="2026-03-02", created_at="2026-03-02 10:00:00"))
    assert [r.content for r in mem.retrieve("我现在什么段位")] == ["玩家段位是钻石"]
    assert any(r.kind == "episode" for r in mem.retrieve("我拿五杀那天发生了什么"))


def test_promises_are_kept_in_mind_not_in_ordinary_search(tmp_path):
    mem = make(tmp_path, recall_modes=True)
    mem.add("玩家最近在练镜", ["镜", "练习"], 3)
    mem.store.put(MemoryRecord(content="助手答应每周帮玩家总结一次战绩", keywords=["战绩", "总结"],
                               kind="promise", created_at="2026-08-01 10:00:00"))
    assert all(r.kind != "promise" for r in mem.retrieve("我最近在练什么英雄，战绩怎么样"))
    llm = scripted(reply="好")
    bot = MemoryChatBot(mem, llm)
    bot.chat("在吗")
    assert "【你答应过玩家的事】" in llm.calls[-1]["system"]
    assert "总结一次战绩" in llm.calls[-1]["system"]
