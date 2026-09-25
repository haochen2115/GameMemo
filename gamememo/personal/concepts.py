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

# Prefecture-level cities (and a few county-level ones people name as home).
_MORE_CITIES = (
    "廊坊 沧州 衡水 邢台 邯郸 承德 张家口 秦皇岛 大同 阳泉 长治 晋城 朔州 晋中 运城 忻州 临汾 吕梁 包头 乌海 赤峰 "
    "通辽 鄂尔多斯 呼伦贝尔 巴彦淖尔 乌兰察布 鞍山 抚顺 本溪 丹东 锦州 营口 阜新 辽阳 盘锦 铁岭 朝阳 葫芦岛 吉林市 "
    "四平 辽源 通化 白山 松原 白城 延边 齐齐哈尔 鸡西 鹤岗 双鸭山 大庆 伊春 佳木斯 七台河 牡丹江 黑河 绥化 "
    "南通 连云港 淮安 盐城 镇江 泰州 宿迁 湖州 金华 衢州 舟山 台州 丽水 义乌 芜湖 蚌埠 淮南 马鞍山 淮北 铜陵 安庆 "
    "黄山 滁州 阜阳 宿州 六安 亳州 池州 宣城 莆田 三明 漳州 南平 龙岩 宁德 景德镇 萍乡 九江 新余 鹰潭 赣州 吉安 "
    "宜春 抚州 上饶 淄博 枣庄 东营 济宁 泰安 威海 日照 临沂 德州 聊城 滨州 菏泽 平顶山 安阳 鹤壁 新乡 焦作 濮阳 "
    "许昌 漯河 三门峡 南阳 商丘 信阳 周口 驻马店 黄石 十堰 宜昌 襄阳 鄂州 荆门 孝感 荆州 黄冈 咸宁 随州 恩施 株洲 "
    "湘潭 衡阳 邵阳 岳阳 常德 张家界 益阳 郴州 永州 怀化 娄底 韶关 汕头 江门 湛江 茂名 肇庆 惠州 梅州 汕尾 河源 "
    "阳江 清远 中山 潮州 揭阳 云浮 柳州 梧州 北海 防城港 钦州 贵港 玉林 百色 贺州 河池 来宾 崇左 儋州 三沙 自贡 "
    "攀枝花 泸州 德阳 绵阳 广元 遂宁 内江 乐山 南充 眉山 宜宾 广安 达州 雅安 巴中 资阳 六盘水 遵义 安顺 毕节 铜仁 "
    "曲靖 玉溪 保山 昭通 普洱 临沧 西双版纳 铜川 宝鸡 咸阳 渭南 延安 汉中 榆林 安康 商洛 嘉峪关 金昌 白银 天水 "
    "武威 张掖 平凉 酒泉 庆阳 定西 陇南 海东 石嘴山 吴忠 固原 中卫 克拉玛依 吐鲁番 哈密 喀什 伊犁 库尔勒 阿克苏 "
    "日喀则 林芝 昆山 江阴 常熟 张家港 晋江 石狮 顺德 番禺 宜兴 慈溪 余姚 诸暨 海宁 桐乡 温岭 瑞安 乐清"
).split()

_ROLES = "射手 法师 打野 辅助 坦克 中路 对抗路 发育路 边路 上单 战士 刺客 游走".split()

_HEALTH = (
    "过敏 近视 远视 散光 腰 腰椎 颈椎 膝盖 韧带 骨折 拉伤 扭伤 手术 住院 感冒 发烧 失眠 头晕 头疼 胃 胃病 "
    "低血糖 高血压 血压 血糖 糖尿病 哮喘 鼻炎 牙疼 视力 手汗 腱鞘炎 怀孕"
).split()

_STUDY = (
    "计算机 软件 软件工程 土木 材料 会计 金融 经济 法律 医学 临床 护理 物理 化学 数学 生物 英语 "
    "中文 历史 机械 电子 电气 建筑 设计 美术 音乐 体育 新闻 心理 管理"
).split()

# concept -> (question trigger, value words)
CONCEPTS: Dict[str, Tuple[Pattern, Tuple[str, ...]]] = {
    "职业": (re.compile(r"工作|职业|做什么的|干什么的|干啥的|是做什么|做什么工作|什么工作|身份|从事"),
           tuple(_OCCUPATIONS)),
    "城市": (re.compile(r"城市|哪里人|在哪(个城市|里)?(上班|工作|住|读书|上学)|住在哪|住哪|老家|家乡|哪个地方"),
           tuple(_CITIES + _MORE_CITIES)),
    "位置": (re.compile(r"位置|哪条路|分路|打什么(位)?|主打"), tuple(_ROLES)),
    "健康": (re.compile(r"身体|健康|毛病|不舒服|病|伤|哪里疼|过敏|眼睛"), tuple(_HEALTH)),
    "专业": (re.compile(r"专业|学什么|学的什么|读什么"), tuple(_STUDY)),
}


def tag_text(concept: str) -> str:
    return f"概念{concept}"


# Jobs the word list misses, recognised by their suffix ("烘焙师", "海员",
# "焊工"); words that end the same way but are not jobs are excluded.
_JOB_SUFFIX = re.compile(r"[一-鿿]{1,4}(?:师|员|医生|司机|经理|主管|店长|老板|教练)|[焊电钳车瓦木漆]工")
_NOT_JOB = re.compile(r"成员|会员|队员|学员|球员|动员|人员|官员|满员|演员表|大师兄|师兄|师姐|师弟|师妹|老师傅")


def memory_concepts(text: str) -> List[str]:
    found = [c for c, (_, values) in CONCEPTS.items() if any(v in text for v in values)]
    if "职业" not in found and _JOB_SUFFIX.search(_NOT_JOB.sub("", text)):
        found.insert(0, "职业")
    return found


def query_concepts(query: str) -> List[str]:
    return [c for c, (trigger, _) in CONCEPTS.items() if trigger.search(query)]


def category_terms(query: str, terms: List[str]) -> Dict[str, List[str]]:
    """Query terms that only name the category a concept asks about
    ("工作" in "我是做什么工作的"). A memory carrying the concept's tag
    answers them, even though it never contains the word itself. Value
    words ("过敏") are never absorbed: they must match literally."""
    out: Dict[str, List[str]] = {}
    for c, (trigger, values) in CONCEPTS.items():
        spans = [m.group() for m in trigger.finditer(query)]
        hit = [t for t in terms if t not in values and any(t in s for s in spans)]
        if hit:
            out[c] = hit
    return out
