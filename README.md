<div align="center">

# 🎮 GameMemo · GameMemory

### Event-Grounded Memory for Game Companions

**面向游戏陪伴的事件流记忆**

[![Project page](https://img.shields.io/badge/project-page-2f5bd3)](https://haochen2115.github.io/GameMemo/game-memory/)
[![Paper](https://img.shields.io/badge/paper-docs%2FPAPER.md-1f7a4d)](docs/PAPER.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-qwen2.5-orange.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[项目主页](https://haochen2115.github.io/GameMemo/) · [论文](docs/PAPER.md) · [实验记录](docs/EXPERIMENTS.md) · [共享记忆（shared-memory 分支）](https://github.com/haochen2115/GameMemo/tree/shared-memory)

</div>

---

通用的长期记忆（Mem0、MemGPT、对话检索、长上下文）把交互历史当作**文本**，按**查询**检索。游戏陪伴的输入有不同的结构：主观、稀疏的对话流与客观、稠密的对局流并存；对话紧随对局、以指代引用对局实体；最需要记忆的时刻常由对局事件而非提问界定。

**事件流记忆**（Event-Grounded Memory）把对局流转换为事件流，用作记忆的系统记录、对话的锚点和意图的触发器。在**输入完全相同**（全部对话 + 全部对局记录）的比较中：

| 任务（封存 test 集，`qwen2.5:3b`） | 事件流记忆 | 检索 | Mem0 | 长上下文 |
|---|---|---|---|---|
| 事件触发回忆（E23，主分 / 召回） | **0.765 / 0.438** | 0.452 / 0.000 | 0.252 / 0.021 | 0.417 / 0.083 |
| 游戏状态问答（E20，游戏数据题） | **0.944** | 0.236 | — | 0.306 |

差距来自记忆结构而非信息量；换 7B 回复模型、或把意图识别从规则换成 LLM，结论不变。技术报告见 [docs/PAPER.md](docs/PAPER.md)。

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
| [E23](docs/EXPERIMENTS.md) · e2e_g2 | 事件触发回忆，信息对等 | 事件流记忆 0.765 vs 检索 0.452、长上下文 0.417、Mem0 0.252（领先 +0.313 至 +0.513）；通用做法召回 ≤ 0.083 |
| [E21–E22](docs/EXPERIMENTS.md) · e2e_g2 | 事件触发回忆，消融与稳健性 | 0.861（基线仅对话）；7B 回复 0.930；LLM 意图识别 0.844 |
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
