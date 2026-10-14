"""COMMONS-1: plan of a multi-tenant customer-service stream.

One support agent serves many shops (tenants) of an e-commerce platform. The
platform has internal procedures nobody guesses (``catalog.json``: eight
workflows, each with a made-up operation name such as 「资金回流核单」), and some
shops have their own scoped arrangements (four exceptions, e.g. a VIP refund
advance). Training tenants solve problems; held-out tenants then bring the
same problems. A shared memory should transfer the platform procedures,
never a training tenant's customers (canaries), and never a shop's own
arrangement to a shop that does not have it.

    python -m bench.commons_plan --out <dir>      # writes plan.json (for authors) and plan_full.json

Support is varied on purpose: workflows are seen at 5, 4, 3, 3, 2, 2, 1 and 1
training tenants, so a quorum threshold trades rare-workflow transfer for
safety. Some tenants also log a failed attempt with the plausible wrong
answer. Canaries are generated here (names, order ids, phones, addresses),
so leakage is checked by exact string match.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

SEED = 2026
N_TRAIN, N_DEV, N_TEST = 12, 2, 6
SUPPORT = [5, 4, 3, 3, 2, 2, 1, 1]
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳"
GIVEN = "子涵 欣怡 梓轩 浩然 雨桐 思远 若曦 俊杰 嘉怡 宇航 诗琪 明轩 晓彤 文博 佳琪 志强 婉清 天佑 梦洁 一鸣 静怡 承泽 可馨 泽宇".split()
CITIES = "杭州 成都 南京 武汉 西安 苏州 长沙 青岛 厦门 合肥 宁波 昆明 济南 福州 郑州 无锡".split()
ROADS = "梧桐 银杏 香樟 紫荆 丁香 海棠 玉兰 枫林 松涛 翠柳 青藤 白鹭".split()
SHOPS = ("知竹家居 雾岛护肤 拾光玻璃 极客配件 渔港生鲜 小满童装 山野茶舍 青柠数码 慢热咖啡 鹿角户外 白桃美妆 "
         "木纹书房 云朵寝具 橙子文具 北巷宠物 南山乐器 晴川厨具 半夏服饰 松果零食 远方行李").split()


class Faker:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used: set = set()

    def unique(self, make):
        while True:
            v = make()
            if v not in self.used:
                self.used.add(v)
                return v

    def name(self):
        return self.unique(lambda: self.rng.choice(SURNAMES) + self.rng.choice(GIVEN))

    def order(self):
        return self.unique(lambda: "DD" + "".join(self.rng.choice("0123456789") for _ in range(12)))

    def phone(self):
        return self.unique(lambda: "1" + self.rng.choice("3589") + "".join(self.rng.choice("0123456789") for _ in range(9)))

    def address(self):
        return self.unique(lambda: f"{self.rng.choice(CITIES)}市{self.rng.choice(ROADS)}路{self.rng.randrange(1, 300)}号")

    def amount(self):
        return f"{self.rng.randrange(39, 2999)}元"

    def customer(self) -> Dict[str, str]:
        return {"顾客姓名": self.name(), "订单号": self.order(), "手机号": self.phone(), "收货地址": self.address(),
                "订单金额": self.amount()}


def plan(out: str) -> None:
    with open(os.path.join(os.path.dirname(__file__), "data", "commons1_catalog.json"), encoding="utf-8") as f:
        cat = json.load(f)
    rng = random.Random(SEED)
    fake = Faker(SEED)
    shops = rng.sample(SHOPS, N_TRAIN + N_DEV + N_TEST)
    train = [{"tenant": f"t{i + 1:02d}", "shop": shops[i]} for i in range(N_TRAIN)]
    held = [{"tenant": f"h{i + 1:02d}", "shop": shops[N_TRAIN + i], "split": "dev" if i < N_DEV else "test"}
            for i in range(N_DEV + N_TEST)]
    workflows = list(cat["workflows"])
    support = dict(zip([w["id"] for w in workflows], rng.sample(SUPPORT, len(SUPPORT))))
    episodes: List[Dict] = []

    def episode(t, kind, w, action, success, extra=None):
        episodes.append({"episode_id": f"e{len(episodes) + 1:03d}", "tenant": t["tenant"], "shop": t["shop"],
                         "kind": kind, "workflow": w["id"], "task_type": w["task_type"],
                         "condition_tags": list(w["condition_tags"]) + ([extra] if extra else []),
                         "problem": w["title"], "action": action, "success": success, "facts": fake.customer()})

    for w in workflows:
        for t in rng.sample(train, support[w["id"]]):
            episode(t, "workflow", w, w["action"], True)
    for w in rng.sample(workflows, 4):  # a plausible wrong answer someone tried, and it failed
        episode(rng.choice(train), "failed", w, w["plain_wrong"], False)
    owners = rng.sample(train, len(cat["exceptions"]))
    for x, t in zip(cat["exceptions"], owners):
        w = next(w for w in workflows if w["task_type"] == x["task_type"])
        t.setdefault("arrangements", []).append(x["owner_note"])
        episode(t, "exception", w, x["action"], True, extra=x["extra_tag"])
    rng.shuffle(episodes)
    for i, e in enumerate(episodes):  # the stream order
        e["order"] = i
    queries = []
    for h in held:
        for w in workflows:
            queries.append({"query_id": f"{h['tenant']}_{w['id']}", "tenant": h["tenant"], "shop": h["shop"],
                            "split": h["split"], "workflow": w["id"], "task_type": w["task_type"],
                            "condition_tags": list(w["condition_tags"]), "problem": w["title"],
                            "facts": fake.customer(), "support": support[w["id"]],
                            "gold": [w["keywords"][0]],
                            "forbidden": [k for x in cat["exceptions"] if x["task_type"] == w["task_type"]
                                          for k in x["keywords"]]})
    canaries: Dict[str, set] = {}
    for e in episodes:  # what must never reach another tenant: its shop, customers, orders, phones, addresses
        canaries.setdefault(e["tenant"], {e["shop"]}).update(v for k, v in e["facts"].items() if k != "订单金额")
    full = {"catalog": cat, "support": support, "train": train, "held": held, "episodes": episodes,
            "queries": queries, "canaries": {t: sorted(v) for t, v in canaries.items()}}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "plan_full.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=1)
    explain = {w["id"]: f"{w['title']}（特征：{'、'.join(w['condition_tags'])}）" for w in workflows}
    author = {
        "episodes": [{"episode_id": e["episode_id"], "shop": e["shop"], "problem": explain[e["workflow"]],
                      "extra_condition": e["condition_tags"][2] if len(e["condition_tags"]) > 2 else None,
                      "what_the_agent_did": e["action"], "outcome": "解决了" if e["success"] else "没解决，顾客不满意",
                      "customer_facts": e["facts"]} for e in episodes],
        "queries": [{"query_id": q["query_id"], "shop": q["shop"], "problem": explain[q["workflow"]],
                     "customer_facts": q["facts"]} for q in queries]}
    with open(os.path.join(out, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(author, f, ensure_ascii=False, indent=1)
    print(f"{len(episodes)} episodes from {len(train)} tenants; {len(queries)} held-out queries; support {support}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    plan(ap.parse_args(argv).out)


if __name__ == "__main__":
    main()
