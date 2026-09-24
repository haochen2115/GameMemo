# -*- coding: utf-8 -*-
"""Concept tags: bridge category questions to value statements.

"我是做什么工作的" and "玩家是护士" share no word, and a small embedding
model does not see them as close either. People organise facts under
concepts (job, city, health...), so this module tags a memory with the
concepts its values belong to, and a question with the concepts it asks
about; retrieval then matches the shared tag.

The lexicons are deliberately plain and game-player oriented. A value
word only ever *adds* a tag, so a missing word just means no bridge.
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

_OCCUPATIONS = (
    "护士 医生 牙医 大夫 药剂师 老师 教师 教授 讲师 司机 快递 快递员 外卖 骑手 分拣 会计 出纳 程序员 工程师 "
    "设计师 销售 商务 运营 客服 厨师 主厨 学徒 服务员 店长 收银 消防 消防员 警察 律师 法官 公务员 银行 柜员 "
    "保安 工人 电工 农民 老板 创业 自由职业 主播 模特 演员 歌手 作家 记者 编辑 翻译 导游 空姐 飞行员 军人 "
    "保姆 家政 全职妈妈 全职主妇 退休 学生 研究生 博士 实习 打工 上班族 白领 会计师"
).split()

_CITIES = (
    "北京 上海 天津 重庆 广州 深圳 杭州 南京 苏州 成都 武汉 西安 郑州 长沙 青岛 厦门 福州 合肥 济南 沈阳 大连 "
    "哈尔滨 长春 昆明 贵阳 南宁 南昌 太原 石家庄 兰州 西宁 银川 乌鲁木齐 拉萨 呼和浩特 海口 三亚 宁波 温州 无锡 "
    "常州 佛山 东莞 珠海 洛阳 开封 徐州 扬州 绍兴 嘉兴 泉州 烟台 潍坊 唐山 保定 桂林 丽江 大理 香港 澳门 台北 "
    "广东 浙江 江苏 四川 湖北 湖南 河南 河北 山东 山西 陕西 福建 安徽 江西 云南 贵州 广西 海南 辽宁 吉林 "
    "黑龙江 内蒙古 新疆 西藏 甘肃 青海 宁夏"
).split()

_ROLES = "射手 法师 打野 辅助 坦克 中路 对抗路 发育路 边路 上单 战士 刺客 游走".split()

_HEALTH = (
    "过敏 近视 远视 散光 腰 腰椎 颈椎 膝盖 韧带 骨折 拉伤 扭伤 手术 住院 感冒 发烧 失眠 头晕 头疼 胃 胃病 "
    "低血糖 高血压 血压 血糖 糖尿病 哮喘 鼻炎 牙疼 视力 手汗 腱鞘炎 怀孕"
).split()

_STUDY = (
    "专业 计算机 软件 软件工程 土木 材料 会计 金融 经济 法律 医学 临床 护理 物理 化学 数学 生物 英语 "
    "中文 历史 机械 电子 电气 建筑 设计 美术 音乐 体育 新闻 心理 管理"
).split()

# concept -> (question trigger, value words)
CONCEPTS: Dict[str, Tuple[Pattern, Tuple[str, ...]]] = {
    "职业": (re.compile(r"工作|职业|做什么的|干什么的|干啥的|是做什么|做什么工作|什么工作|上班|身份|从事"),
           tuple(_OCCUPATIONS)),
    "城市": (re.compile(r"城市|哪里人|在哪(个城市|里)?(上班|工作|住|读书|上学)|住在哪|住哪|老家|家乡|哪个地方"),
           tuple(_CITIES)),
    "位置": (re.compile(r"位置|哪条路|分路|打什么(位)?|主打"), tuple(_ROLES)),
    "健康": (re.compile(r"身体|健康|毛病|不舒服|病|伤|哪里疼|过敏|眼睛"), tuple(_HEALTH)),
    "专业": (re.compile(r"专业|学什么|学的什么|读什么"), tuple(_STUDY)),
}


def tag_text(concept: str) -> str:
    return f"概念{concept}"


def memory_concepts(text: str) -> List[str]:
    return [c for c, (_, values) in CONCEPTS.items() if any(v in text for v in values)]


def query_concepts(query: str) -> List[str]:
    return [c for c, (trigger, _) in CONCEPTS.items() if trigger.search(query)]
