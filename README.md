<div align="center">

# 🎮 GameMemo · GameMemory

### Remember at the Right Game: Event-Triggered Memory for Game Companions

**打到那一局就想起：面向游戏陪伴的事件触发记忆**

[![Project page](https://img.shields.io/badge/project-page-2f5bd3)](https://haochen2115.github.io/GameMemo/game-memory/)
[![Paper](https://img.shields.io/badge/paper-docs%2FPAPER.md-1f7a4d)](docs/PAPER.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5-orange.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[项目主页](https://haochen2115.github.io/GameMemo/) · [论文](docs/PAPER.md) · [实验记录](docs/EXPERIMENTS.md) · [共享记忆（shared-memory 分支）](https://github.com/haochen2115/GameMemo/tree/shared-memory)

</div>

---

通用的长期记忆（Mem0、MemGPT、对聊天做检索、把历史全塞进 prompt）都是**问了才想起**。游戏陪伴不一样：它同时看得到聊天和**对局流**，而最需要记忆的时刻往往没人提问——玩家发誓不再碰的英雄又被选上的那一局、定了奖励的段位终于升上的那一局、说过"连输三把就让我下线"之后的第三次连输。

**GameMemory** 按游戏数据的特点设计，把对局流用作记忆的三样东西：

| 对局流的作用 | 解决的问题 | 结果（封存 test 集，`qwen2.5:3b`） |
|---|---|---|
| **系统记录**：游戏状态由对局账本精确计算 | 胜率、场次、赛季最高段位问不出来；玩家夸大的段位被复述 | 游戏数据题 **0.944**，通用做法 0.153–0.306；夸大说法 0 次复述（通用 8/8） |
| **对齐坐标**：聊天写入时钉在它紧跟的那一局 | "这英雄""下次双排他…"这类指代无法解析 | 只说"这英雄"的意图，通用做法召回 0.000–0.067 |
| **触发器**：意图变成英雄 / 档位 / 连败 / 双排线索，由对局流在"说完后第一次"触发 | 没有提问时，记忆根本不会被读出 | 主动回忆 **0.861**，Mem0 0.339、检索 0.504、全部聊天 0.583；索引本身命中 47/48、负例 0 误触发 |

```python
from gamememo.game.memory import GameMemory
from gamememo.personal import FastEmbedEmbedder

mem = GameMemory("player_001", embedder=FastEmbedEmbedder())
mem.ingest_matches(matches)       # 对局：time / mode / hero / position / win / k / d / a / rank_before / rank_after / duo
mem.ingest_chat("玩家: 这英雄我再也不碰了\n助手: 好，先冷静一下")

mem.moment(match)                 # 一局结束：这局该主动提起的原话，或 None（不开口，不调用 LLM）
mem.context("我上赛季最高到哪了")   # 被问时：游戏档案卡（以记录为准）+ 相关原话 + 助手的承诺
```

## 研究内容

| 实验 | 问题 | 结论 |
|---|---|---|
| [E21](docs/EXPERIMENTS.md) · e2e_g2 | 游戏时刻的主动回忆 | GameMemory 0.861 vs Mem0 0.339（+0.522 [+0.435, +0.617]）；通用做法召回 ≤ 0.063 |
| [E20](docs/EXPERIMENTS.md) · e2e_g1 | 游戏状态 + 聊天问答 | GameMemory 0.833 vs 检索 0.460（+0.373 [+0.287, +0.460]） |
| [E11–E19](docs/FINDINGS.md) | 记忆该不该让模型改写 | 原话记忆在本项目和 LoCoMo 上都显著好于改写式（LoCoMo 比 Mem0 高 0.181）；改写的损失在写入 |

所有结论来自预登记的实验：测试集在任何系统运行之前封存，聊天由不看代码的作者编写，假设在跑测试集之前写入实验记录，没成立的假设照样报告。完整论文见 [docs/PAPER.md](docs/PAPER.md)。

## 快速开始

```bash
pip install -e '.[dev]'           # 含 pytest 和本地中文向量模型
pytest                            # 单元测试，不需要 LLM

ollama pull qwen2.5:3b            # 复现实验
python -m bench.run_moments --data bench/data/e2e_g2.json --split test --contexts gamememo,full,rag,none   # E21
python -m bench.run_game    --data bench/data/e2e_g1.json --split test --contexts gamememo,chat,rag,full   # E20
python -m bench.gamesim --players 3 --show                                                                # 看模拟的对局
```

## 目录

```
gamememo/
  game/
    ontology.py          游戏本体：段位阶梯、赛季、位置与英雄、文本里的段位 / 英雄 / 位置识别
    ledger.py            对局账本：精确计算游戏状态、游戏档案卡、每局情境、核对玩家说法
    intentions.py        意图索引：聊天对齐对局、意图 → 游戏线索、由对局流按"第一次"触发
    memory.py            GameMemory：账本 + 原话日志 + 意图索引
  personal/
    raw.py               RawMemory：原话 + 时间保存，混合检索，助手承诺
    retrieval.py         BM25 + 向量 + RRF + 相关性门槛
    system.py, ...       早期的改写式记忆（P1–P6b），保留用于对照
bench/
  gamesim.py             对局模拟器
  build_moments.py, run_moments.py      e2e_g2：游戏时刻的主动回忆
  build_game_bench.py, run_game.py      e2e_g1：游戏状态 + 聊天问答
  locomo.py              LoCoMo + Mem0 外部验证
  data/, results/        封存的数据集和结果
site/                    项目主页（GitHub Pages）
docs/
  PAPER.md               论文
  EXPERIMENTS.md         全部实验记录（E1–E21，含预登记和失败的尝试）
  FINDINGS.md            原话记忆 vs 改写式记忆
  BRANCHING.md           分支说明
```

## 分支

| 分支 | 内容 |
|---|---|
| `main` | **GameMemory 正式版**（本页） |
| [`shared-memory`](https://github.com/haochen2115/GameMemo/tree/shared-memory) | **Shared Memory Is Harder**：多租户客服智能体的共享记忆（QUORUM / COMMONS，研究预览） |
| `exp/*` | 历史实验分支（个人记忆 P1–P6、原话记忆和游戏记忆的开发记录），内容都已在上面两个分支里 |
| `archive/v0-keyword-baseline` | v0 的冻结快照 |

详见 [docs/BRANCHING.md](docs/BRANCHING.md)。

## License

MIT
