# 🎮 GameMemo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.com/)

Long-term, human-like memory for game AI assistants. It remembers who the player is, what they went through, and how they changed over time, and every change is measured on a benchmark before it reaches `main`.

给游戏 AI 助手用的长期记忆系统。目标是让助手像老朋友一样记住玩家：玩家是谁、一起经历过什么、这段时间有什么变化。所有改进都要先在评测上证明有效，才能进入 `main`。

## 现在能做什么

- **写入**：从对话或游戏数据中抽取事实（相对时间自动换算成绝对日期）→ 只把相关的旧记忆交给 LLM → 决定 ADD / UPDATE / DELETE / NOOP → 逐条校验后执行。LLM 编造的目标 id 会被拒绝，近似重复的内容会被跳过。
- **更新保留历史**：段位从星耀升到王者时，旧记录失效但保留，可以回答"我以前什么段位"。
- **检索不调 LLM**：jieba + BM25 + 中文向量（可选），用 RRF 融合；无关的问题返回空，不往 prompt 里塞凑数的记忆。
- **聊天机器人**：每轮都带当前日期、玩家档案和相关记忆；只从还没处理过的新对话里抽取，不重复写入。
- **像人一样回忆**：每段对话会留下一条情景记忆（"那天发生了什么"）；助手记得自己答应过的事，并在聊天时带在身边；问"上次聊了什么""你答应过我什么""我段位是怎么变的"时，会切换到对应的回忆方式；被问到过去时，也能想起已经过时的旧状态，并标注出来。
- **记得自己的变化**：问"我的段位是怎么升上来的""我换过哪些手机"时，会从所有记忆（包括每次聊天的情景摘要）里按时间重建这个属性的每一次变化；写入时拒绝对话里没出现过的段位 / 设备值，防止小模型编造。
- **只从玩家的话里记事**：助手自己的建议和客套话不会被当成玩家的事实。
- **隐私**：`forget(id, hard=True)` 从磁盘彻底删除一条记忆及其全部历史版本。

## 快速开始

```bash
pip install -e '.[dev]'          # 含 pytest 和本地向量模型 fastembed（默认 jina-v2-base-zh）
pytest                           # 单元测试，不需要 LLM
python -m bench.run_retrieval    # 离线检索评测，不需要 LLM
```

与 Ollama 一起跑：

```bash
ollama serve
python examples/chat.py                         # 9 轮脚本演示
python examples/chat.py -i --user alice         # 交互模式（/memory /history <id> /forget <id>）
python examples/chat.py --embedder fastembed    # 打开向量检索
```

代码里使用：

```python
from gamememo import OllamaClient, PersonalMemory, MemoryChatBot
from gamememo.personal import FastEmbedEmbedder

llm = OllamaClient(model="deepseek-v3.1:671b-cloud")
mem = PersonalMemory("player_001", llm=llm, embedder=FastEmbedEmbedder())

report = mem.ingest("玩家: 我生日是2月12日，最近刚上了王者！")
print([r.content for r in report.added], report.updated)

for r in mem.retrieve("我现在什么段位？"):
    print(r.content)

bot = MemoryChatBot(mem, llm)
print(bot.chat("今天是我生日！").reply)
```

## 评测与分支

- 评测集、指标和结果见 [docs/BENCHMARK.md](docs/BENCHMARK.md)。在封存的 test 集（两位新玩家、102 条查询）上，当前 SOTA `v2-hybrid` 的 MemScore@3 为 0.892，v0 为 0.750（95% CI [+0.049, +0.235]）；召回和拒答都更好，而且检索不再调用 LLM。
- 端到端评测（真实 LLM 写入，再按规则判分）：P1 在 e2e_v2 test 上从 0.448 升到 0.759；P2 在 e2e_v4 test 上从 0.693 升到 0.765（段位 / 手机的变化轨迹 0.333 → 1.000）；P3 在 e2e_v5 test 上从 0.794 升到 0.825（事实题，边缘显著）；P4 在更难的 e2e_v6 test 上从 0.482 升到 0.563（无损更新，事实题 0.560 → 0.720）；P5b 在 e2e_v8 test 上从 0.451 升到 0.586（按主语回忆：问"我姐姐…"不再返回哥哥的事，过时率 0.542 → 0.306）；P6b 在 e2e_v10 test 上从 0.461 升到 0.536（记得自己答应过什么、段位 / 英雄的完整变化）。实验全过程（包括没通过的尝试）见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
- `main` 只放当前 SOTA。分支命名和门禁规则见 [docs/BRANCHING.md](docs/BRANCHING.md)。
- 类人记忆的路线图见 [docs/ROADMAP.md](docs/ROADMAP.md)：情景记忆、遗忘曲线、离线巩固、程序性记忆……

## 目录

```
gamememo/
  llm.py                 LLM 接口：OllamaClient / FakeLLM / JSON 解析
  personal/
    system.py            PersonalMemory：写入链路、检索、遗忘
    retrieval.py         混合检索（BM25 + 向量 + RRF + 相关性门槛）
    model.py, store.py   记忆记录（双时态字段）与原子写入的 JSON 存储
    chatbot.py           带记忆的聊天机器人
    prompts.py, text.py, embed.py
bench/                   离线评测、数据集、SOTA 门禁（sota.json）
tests/                   单元测试（FakeLLM，无需联网）
examples/chat.py         演示
game_memory.py, llm_client.py, chatbot_demo.py, test_memory.py, mock_data.py
                         v0 原始实现，保留用于评测对比，后续移除
```

## License

MIT
