# -*- coding: utf-8 -*-
"""Interactive / scripted chat with a memory-backed game assistant.

    ollama serve
    python examples/chat.py                          # scripted 9-turn demo
    python examples/chat.py -i --user alice          # interactive
    python examples/chat.py --embedder fastembed     # add dense retrieval
    python examples/chat.py --ingest-trajectory      # seed from mock game data first

Interactive commands: /memory  /stats  /history <mem_id>  /forget <mem_id>  /quit
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamememo.llm import OllamaClient  # noqa: E402
from gamememo.personal import FastEmbedEmbedder, MemoryChatBot, OllamaEmbedder, PersonalMemory  # noqa: E402

SCRIPT = [
    "你好，我是新手玩家",
    "我的生日是2月12日，请记住",
    "我喜欢玩射手位置，特别是鲁班七号",
    "顺便说一下，我通常晚上8点到10点玩游戏",
    "给我推荐一个英雄",
    "我最近想上分，有什么建议吗？",
    "我段位刚从星耀升到王者了！",
    "我经常被刺客杀死，应该怎么办？",
    "我现在什么段位来着？",
]


def show_turn(turn) -> None:
    if turn.retrieved:
        print("  🔍 相关记忆: " + " | ".join(r.content for r in turn.retrieved))
    print(f"🤖 {turn.reply}")
    rep = turn.report
    if rep is not None:
        for r in rep.added:
            print(f"  ➕ {r.content}")
        for old, new in rep.updated:
            print(f"  🔄 {old.content}  →  {new.content}")
        for r in rep.deleted:
            print(f"  ➖ {r.content}")
        for op, why in rep.rejected:
            print(f"  ⚠️  rejected {op.get('op')}: {why}")


def show_memories(mem: PersonalMemory) -> None:
    for r in sorted(mem.active(), key=lambda r: -r.importance):
        print(f"  [{r.id}] ★{r.importance} {r.content}  ({r.source}, 访问{r.access_count}次)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--user", default="demo_player")
    ap.add_argument("--model", default="deepseek-v3.1:671b-cloud")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--embedder", choices=["none", "fastembed", "ollama"], default="none")
    ap.add_argument("--storage", default="./memory_data")
    ap.add_argument("--ingest-trajectory", action="store_true")
    args = ap.parse_args()

    llm = OllamaClient(model=args.model, base_url=args.base_url)
    if not llm.is_available():
        sys.exit(f"Ollama is not reachable at {args.base_url}. Run: ollama serve")

    embedder = None
    if args.embedder == "fastembed":
        embedder = FastEmbedEmbedder()
    elif args.embedder == "ollama":
        embedder = OllamaEmbedder(llm, model="bge-m3")

    mem = PersonalMemory(args.user, llm=llm, storage_dir=args.storage, embedder=embedder)
    bot = MemoryChatBot(mem, llm)
    print(f"已加载 {len(mem.active())} 条记忆")

    if args.ingest_trajectory:
        from mock_data import MOCK_TRAJECTORY_1
        rep = mem.ingest(MOCK_TRAJECTORY_1, source="trajectory")
        print(f"从游戏数据写入 {rep.changed} 条记忆")

    try:
        if not args.interactive:
            for line in SCRIPT:
                print(f"\n👤 {line}")
                show_turn(bot.chat(line))
            return

        while True:
            line = input("\n👤 ").strip()
            if not line:
                continue
            if line in ("/quit", "quit"):
                break
            if line == "/memory":
                show_memories(mem)
            elif line == "/stats":
                print(mem.stats())
            elif line.startswith("/history "):
                for r in mem.history(line.split()[1]):
                    print(f"  {r.created_at}  {r.content}{'' if r.is_active else '  (已过期)'}")
            elif line.startswith("/forget "):
                print("已删除" if mem.forget(line.split()[1], hard=True) else "没有这条记忆")
            else:
                show_turn(bot.chat(line))
    finally:
        rep = bot.flush()
        if rep is not None and rep.changed:
            print(f"\n会话结束，补充写入 {rep.changed} 条记忆")


if __name__ == "__main__":
    main()
