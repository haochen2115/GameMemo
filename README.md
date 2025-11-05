# 🎮 GameMemo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.com/)

An intelligent memory management system for game AI assistants, powered by LLMs through Ollama.

[English](#english) | [中文](#中文)

---

## English

### 📖 Overview

GameMemo is a sophisticated memory management system designed for game AI assistants. It automatically extracts, stores, and retrieves player information from conversations and game data, enabling personalized and context-aware interactions.

### ✨ Features

- 🧠 **Smart Memory Extraction** - Automatically extracts key information from conversations and game trajectories
- 🔄 **Dynamic Updates** - Supports add, update, and delete operations on memories
- 🔍 **Semantic Retrieval** - Finds relevant memories using keyword and semantic matching
- 📝 **Detailed Logging** - Tracks all operations for debugging and analysis
- 🎯 **Priority System** - Organizes memories by importance (core, important, general)
- 💾 **Persistent Storage** - Saves memories to JSON files for long-term retention

### 🚀 Quick Start

#### Prerequisites

1. **Python 3.8+**
2. **Ollama** - [Download and install](https://ollama.com/download)
3. **A language model** - e.g., `deepseek-v3.1:671b-cloud`, `llama3.2:7b`

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GameMemo.git
cd GameMemo
```

#### Running the Demo

```bash
# Automatic demo (9 conversation turns)
python chatbot_demo.py

# Interactive mode
python chatbot_demo.py --interactive

# Run tests
python test_memory.py
```

### 📋 Usage Example

```python
from game_memory import GameMemory
from llm_client import OllamaClient

# 初始化
memory_system = GameMemory(
    user_id="player_001",
    model="deepseek-v3.1:671b-cloud"
)

# 添加对话记忆
conversation = """
玩家: 你好，我是新手玩家
助手: 欢迎！
玩家: 我的生日是2月12日
玩家: 我喜欢玩射手英雄
"""

count = memory_system.update_personal_memory_with_messages(conversation)
print(f"提取了 {count} 条记忆")

# 检索相关记忆
query = "玩家喜欢什么英雄？"
memories = memory_system.retrieval_relevant_memory(query, top_k=3)
for mem in memories:
    print(f"- {mem.content}")

# 保存
memory_system.save()
```

### 🔧 Configuration

You can customize the LLM client:

```python
from llm_client import OllamaClient

# Custom Ollama instance
client = OllamaClient(
    model="deepseek-v3.1:671b-cloud",
    base_url="http://localhost:11434",
    timeout=120
)

# Use with GameMemory
memory = GameMemory(
    user_id="player_001",
    llm_client=client
)
```

### 📊 Memory Priority Levels

| Priority | Level | Use Case |
|----------|-------|----------|
| 1 | Core | Critical information (birthday, username) |
| 2-3 | Important | Game preferences, frequently used heroes |
| 4-5 | General | Casual information, temporary notes |

---

## 中文

### 📖 概述

GameMemo 是一个为游戏 AI 助手设计的智能记忆管理系统。它能够自动从对话和游戏数据中提取、存储和检索玩家信息，从而实现个性化和上下文感知的交互。

### ✨ 功能特性

- 🧠 **智能记忆提取** - 自动从对话和游戏轨迹中提取关键信息
- 🔄 **动态更新** - 支持记忆的新增、更新和删除操作
- 🔍 **语义检索** - 使用关键词和语义匹配查找相关记忆
- 📝 **详细日志** - 记录所有操作，便于调试和分析
- 🎯 **优先级系统** - 按重要性组织记忆（核心、重要、一般）
- 💾 **持久化存储** - 将记忆保存到 JSON 文件，长期保留

### 🚀 快速开始

#### 前置要求

1. **Python 3.8+**
2. **Ollama** - [下载并安装](https://ollama.com/download)
3. **语言模型** - 例如 `deepseek-v3.1:671b-cloud`、`llama3.2:7b`

#### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/GameMemo.git
cd GameMemo
```

#### 运行演示

```bash
# 自动演示（9 轮对话）
python chatbot_demo.py

# 交互模式
python chatbot_demo.py --interactive

# 运行测试
python test_memory.py
```

### 📋 使用示例

```python
from game_memory import GameMemory
from llm_client import OllamaClient

# 初始化
memory_system = GameMemory(
    user_id="player_001",
    model="deepseek-v3.1:671b-cloud"
)

# 添加对话记忆
conversation = """
玩家: 你好，我是新手玩家
助手: 欢迎！
玩家: 我的生日是2月12日
玩家: 我喜欢玩射手英雄
"""

count = memory_system.update_personal_memory_with_messages(conversation)
print(f"提取了 {count} 条记忆")

# 检索相关记忆
query = "玩家喜欢什么英雄？"
memories = memory_system.retrieval_relevant_memory(query, top_k=3)
for mem in memories:
    print(f"- {mem.content}")

# 保存
memory_system.save()
```

### 🔧 配置

你可以自定义 LLM 客户端：

```python
from llm_client import OllamaClient

# 自定义 Ollama 实例
client = OllamaClient(
    model="deepseek-v3.1:671b-cloud",
    base_url="http://localhost:11434",
    timeout=120
)

# 与 GameMemory 一起使用
memory = GameMemory(
    user_id="player_001",
    llm_client=client
)
```

### 📊 记忆优先级

| 优先级 | 级别 | 使用场景 |
|--------|------|---------|
| 1 | 核心 | 关键信息（生日、用户名） |
| 2-3 | 重要 | 游戏偏好、常用英雄 |
| 4-5 | 一般 | 临时信息、随意记录 |

---

## 📞 Contact
- Email: haochen2115@gmail.com

