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
- **隐私**：`forget(id, hard=True)` 从磁盘彻底删除一条记忆及其全部历史版本。

## 快速开始

```bash
pip install -e '.[dev]'          # 含 pytest 和本地向量模型 fastembed
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

- 评测集、指标和结果见 [docs/BENCHMARK.md](docs/BENCHMARK.md)。当前 v1 混合检索在 test 上 Recall@3 为 0.912（v0 为 0.765），但拒答率回退，所以还没进 `main`。
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
