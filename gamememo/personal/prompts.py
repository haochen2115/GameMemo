# -*- coding: utf-8 -*-
"""Prompts and JSON schemas for the memory write path."""

SYSTEM_JSON = "你是游戏AI助手的记忆模块，负责准确地整理关于玩家的长期信息。严格按要求输出JSON。"

EXTRACT_CHAT = """今天是{today}。

下面是玩家与游戏助手的对话。请提取其中关于【玩家本人】、值得长期记住的信息，每条写成一句以"玩家"开头的简洁事实。

规则：
- 只提取玩家自己说出或明确确认的信息；助手的建议、客套话不算。
- 关注：个人属性（生日、身份、设备等）、游戏偏好（英雄、位置、模式）、水平与段位、目标、社交关系、习惯与情绪模式、重要事件（成就、特殊经历）。
- "昨天""前天""上周"这类相对时间一律换算成具体日期（昨天 = {yesterday}，前天 = {before_yesterday}）。
- 只写对话里真实出现的信息，不要补充对话中没有的数字或细节。
- 闲聊、一次性的提问、没有信息量的内容不要提取。没有可提取的信息时返回空列表。

对话：
{text}

返回 JSON：{{"facts": ["玩家……", "玩家……"]}}"""

EXTRACT_TRAJECTORY = """今天是{today}。

下面是玩家的游戏行为数据。请提炼出值得长期记住的稳定特征和显著变化，每条写成一句以"玩家"开头的简洁事实，保留关键数字。

规则：
- 优先：常用英雄及胜率、位置偏好、出装偏好、活跃时段、打法风格、主要短板、成就纪录、社交关系。
- 数据给出的统计周期要写明（如"近30天""本周"）。
- 不要逐条复述原始数据，合并同类信息，最多15条。

游戏数据：
{text}

返回 JSON：{{"facts": ["玩家……", "玩家……"]}}"""

FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

DECIDE_OPS = """今天是{today}。

【已有的相关记忆】（方括号内是编号）
{existing}

【新提取的事实】
{facts}

对每条新事实，决定如何写入记忆库，输出一个操作：
- ADD：全新的信息。
- UPDATE：新事实取代了某条已有记忆的旧值（例如段位变化、换了设备、目标更新、数据刷新），target 填那条记忆的编号，content 写完整的新事实。
- DELETE：新事实表明某条已有记忆不再成立、且没有新值可替代，target 填编号。
- NOOP：已有记忆已经包含这条信息。

要求：
- target 只能使用上面列出的编号，没有合适的就不要用 UPDATE/DELETE。
- content 只写简洁事实，以"玩家"开头，形如"玩家的<某方面>是<具体内容>"；只能使用新事实里出现的信息，不要添加任何新数字或细节。
- keywords 给 2-5 个检索用关键词（英雄名、段位名等专有名词要原样保留）。
- importance：5=身份核心信息（生日、身份）；4=长期偏好与水平；3=一般偏好和习惯；2=具体事件和数据；1=琐碎信息。
- event_time：事实描述的是某个具体日期发生的事件时填 YYYY-MM-DD，否则为 null。

返回 JSON：{{"operations": [{{"op": "ADD", "target": null, "content": "...", "keywords": ["..."], "importance": 3, "event_time": null, "reason": "..."}}]}}"""

OPS_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
                    "target": {"type": ["integer", "null"]},
                    "content": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "event_time": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["operations"],
}

CHAT_PERSONA = ("你是一个专业又贴心的王者荣耀游戏助手，擅长给出游戏建议，也会像老朋友一样陪玩家聊天。"
                "说话口语化、简短，一般不超过3-4句话；玩家明确要攻略时再展开。不确定的游戏细节不要编造。")

CHAT_MEMORY_RULES = """使用记忆的原则：
1. 这些是你对这位玩家的真实了解，需要时自然地用上，不要生硬复述或炫耀“我记得”。
2. 与当前话题无关的记忆不要提。
3. 记忆与玩家当下说法冲突时，以玩家当下说法为准，可以顺带确认。
4. 标注了日期的事件，结合今天的日期理解（例如今天恰好是玩家生日）。"""

# ---------------------------------------------------------------- slot mode
# Facts are extracted together with the aspect of the player they describe.
# For a single-valued aspect a new fact replaces the old one automatically,
# so a small model never has to reason about which memory to UPDATE.

SINGLE_ASPECTS = ["生日", "年龄", "身份职业", "学校年级", "所在城市", "游戏时段", "设备",
                  "当前段位", "主玩英雄", "主玩位置", "正在学的英雄", "当前目标", "游戏昵称"]
MULTI_ASPECTS = ["游戏伙伴", "喜好", "厌恶", "成就与事件", "习惯", "近况", "其他"]
ASPECTS = SINGLE_ASPECTS + MULTI_ASPECTS

EXTRACT_SLOTS = """今天是{today}（昨天 = {yesterday}，前天 = {before_yesterday}）。

下面是玩家与游戏助手的对话。请只根据【玩家说的话】提取值得长期记住的、关于玩家本人的信息。

规则：
- 助手说的建议、客套、提问都不是玩家的信息，不要提取。
- 玩家的一次性提问（比如"推荐个英雄"）不要提取。
- 每条信息写成一句以"玩家"开头的完整事实，只能使用玩家原话里出现的内容，不要补充任何数字或细节。
- aspect 从这个列表里选最贴切的一项：{aspects}
- 如果玩家说的是某个时间点发生的事（升段、拿成就、换设备、过生日等），event_date 填具体日期 YYYY-MM-DD；"今天"就是{today}，"昨天""前天"等要换算。否则为 null。
- keywords 给 2-4 个检索用关键词，英雄名、段位名等专有名词保持原样。
- importance：5=身份核心（生日、身份），4=长期偏好与水平，3=一般习惯，2=具体事件，1=琐碎。
- 没有可提取的信息时返回空列表。

对话：
{text}

返回 JSON：{{"facts": [{{"aspect": "...", "statement": "玩家……", "keywords": ["..."], "importance": 3, "event_date": null}}]}}"""

SLOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "enum": ASPECTS},
                    "statement": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "event_date": {"type": ["string", "null"]},
                },
                "required": ["aspect", "statement"],
            },
        }
    },
    "required": ["facts"],
}


# ---------------------------------------------------------------- P1: episodes & promises

EPISODE = """今天是{today}。

下面是玩家和游戏助手今天的一段对话。请用一到两句话概括这次聊天里【玩家】讲了什么、发生了什么，写成第三人称，以"玩家"开头。
- 保留关键的事件、人物、英雄、段位和数字；
- 不写助手的建议和客套话；
- 不要编造对话里没有的内容。

对话：
{text}

返回 JSON：{{"summary": "玩家……", "keywords": ["...", "..."]}}"""

EPISODE_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"},
                   "keywords": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary"],
}

PROMISES = """今天是{today}。

下面是玩家和游戏助手的一段对话。请找出【助手】明确答应玩家、将来要去做的事（承诺），比如"下次帮你复盘""每天提醒你休息"。
- 每条写成以"助手答应"开头的一句话，写清楚要为玩家做什么；
- 只算助手明确说出口的承诺，建议、祝福、客套话都不算；
- 没有承诺就返回空列表。

对话：
{text}

返回 JSON：{{"promises": ["助手答应……"]}}"""

PROMISES_SCHEMA = {
    "type": "object",
    "properties": {"promises": {"type": "array", "items": {"type": "string"}}},
    "required": ["promises"],
}
