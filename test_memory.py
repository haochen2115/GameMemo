# -*- coding: utf-8 -*-
"""
Memory System Tests - Clear demonstration of memory update process

Test content:
1. Extract memories from conversations
2. Extract memories from game trajectories  
3. Memory retrieval tests
"""

import os
from game_memory import GameMemory
from llm_client import OllamaClient
from mock_data import MOCK_CHAT_1, MOCK_TRAJECTORY_1


def print_divider(title=""):
    """Print divider line"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'-'*70}\n")


def show_memory_change(memory_system: GameMemory, before_count: int):
    """Show memory changes"""
    after_count = len([m for m in memory_system.memories if m.valid == 1])
    new_count = after_count - before_count
    
    print(f"\n📊 Memory changes:")
    print(f"   Previous memories: {before_count} items")
    print(f"   Current memories: {after_count} items")
    print(f"   New memories: {new_count} items")
    
    if new_count > 0:
        print(f"\n📝 Details of new memories:")
        valid_memories = [m for m in memory_system.memories if m.valid == 1]
        for mem in sorted(valid_memories, key=lambda x: x.create_time, reverse=True)[:new_count]:
            print(f"   • ID: {mem.id}")
            print(f"     Keywords: {mem.keywords}")
            print(f"     Content: {mem.content}")
            print(f"     Priority: {mem.priority} | Source: {mem.source}")
            print()


def test_1_chat_memory():
    """Test 1: Extract memories from conversations"""
    print_divider("Test 1: Extract Memories from Conversations")
    
    memory = GameMemory(user_id="test_001", storage_dir="./memory_data", model="deepseek-v3.1:671b-cloud")
    
    print("📥 Input conversation:")
    print(MOCK_CHAT_1.strip())
    print_divider()
    
    before_count = len([m for m in memory.memories if m.valid == 1])
    
    print("🔄 Starting memory extraction...")
    print("\n📌 Step 1: Summarize key information")
    key_info = memory.summarize_key_info(MOCK_CHAT_1, "chat", debug=False)
    print(f"   Extracted key information:")
    for i, info in enumerate(key_info, 1):
        print(f"   {i}. {info}")
    
    print("\n📌 Step 2: Convert to structured memories and store")
    count = memory.update_personal_memory_with_messages(MOCK_CHAT_1, debug=False)
    
    show_memory_change(memory, before_count)
    
    return memory


def test_2_trajectory_memory(memory: GameMemory):
    """Test 2: Extract memories from game trajectories"""
    print_divider("Test 2: Extract Memories from Game Trajectories")
    
    print("📥 Input trajectory:")
    print(MOCK_TRAJECTORY_1.strip()[:200] + "...")
    print_divider()
    
    before_count = len([m for m in memory.memories if m.valid == 1])
    
    print("🔄 Starting memory extraction...")
    print("\n📌 Step 1: Summarize key behaviors")
    key_info = memory.summarize_key_info(MOCK_TRAJECTORY_1, "trajectory", debug=False)
    print(f"   Extracted key behaviors:")
    for i, info in enumerate(key_info, 1):
        print(f"   {i}. {info}")
    
    print("\n📌 Step 2: Convert to structured memories and store")
    count = memory.update_personal_memory_with_trajectories(MOCK_TRAJECTORY_1, debug=False)
    
    show_memory_change(memory, before_count)
    
    return memory


def test_3_memory_retrieval(memory: GameMemory):
    """Test 3: Memory retrieval"""
    print_divider("Test 3: Memory Retrieval")
    
    queries = [
        "玩家的生日是什么时候？",
        "玩家喜欢用什么英雄？",
        "玩家的游戏时间是什么时候？"
    ]
    
    for query in queries:
        print(f"🔍 Query: {query}")
        memories = memory.retrieval_relevant_memory(query, top_k=3, debug=False)
        
        if memories:
            print(f"   Found {len(memories)} relevant memories:")
            for i, mem in enumerate(memories, 1):
                print(f"   {i}. {mem.content}")
        else:
            print(f"   No relevant memories found")
        print()


def test_4_show_all_memories(memory: GameMemory):
    """Test 4: View all memories"""
    print_divider("Test 4: All Memories Overview")
    
    valid_memories = [m for m in memory.memories if m.valid == 1]
    
    print(f"共有 {len(valid_memories)} 条有效记忆:\n")
    
    for mem in sorted(valid_memories, key=lambda x: x.priority):
        print(f"[优先级 {mem.priority}] [{mem.source}]")
        print(f"  关键词: {mem.keywords}")
        print(f"  内容: {mem.content}")
        print(f"  创建时间: {mem.create_time}")
        print()


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  游戏记忆系统测试")
    print("="*70)
    
    # Check if Ollama is available
    print("\n🔍 检查 Ollama 可用性...")
    client = OllamaClient()
    if not client.is_available():
        print("\n❌ 错误：Ollama 未运行！")
        print("请先启动 Ollama：")
        print("  1. 运行: ollama serve")
        print("  2. 确保有模型: ollama pull deepseek-v3.1:671b-cloud")
        return
    
    print("✅ Ollama 正在运行\n")
    
    # Clean old data
    if os.path.exists("./memory_data/test_001_memory.json"):
        os.remove("./memory_data/test_001_memory.json")
        print("🧹 已清理旧的测试数据\n")
    
    # Test 1: Chat memory
    memory = test_1_chat_memory()
    
    # Test 2: Trajectory memory
    memory = test_2_trajectory_memory(memory)
    
    # Test 3: Memory retrieval
    test_3_memory_retrieval(memory)
    
    # Test 4: View all memories
    test_4_show_all_memories(memory)
    
    # Save memories
    print_divider("保存记忆")
    memory.save()
    print(f"✅ 记忆已保存至: memory_data/test_001_memory.json")
    
    print("\n" + "="*70)
    print("  测试完成")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_tests()
