# -*- coding: utf-8 -*-
"""
Memory-Based Game Chatbot Demo

Features:
- Real-time memory extraction and updates during conversations
- Memory insertion into prompt context
- Clear demonstration of memory changes per conversation turn
"""

import os
import time
from game_memory import GameMemory
from llm_client import OllamaClient


class MemoryChatBot:
    """Memory-based chatbot for game assistance"""
    
    def __init__(self, 
                 user_id: str, 
                 model: str = "deepseek-v3.1:671b-cloud",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize chatbot
        
        Args:
            user_id: User ID
            model: LLM model name
            ollama_base_url: Ollama API base URL
        """
        self.user_id = user_id
        self.model = model
        
        # Initialize LLM client
        self.llm_client = OllamaClient(model=model, base_url=ollama_base_url)
        
        # Check if Ollama is available
        if not self.llm_client.is_available():
            raise RuntimeError(
                "Ollama is not available. Please ensure Ollama is running.\n"
                "Run: ollama serve"
            )
        
        # Initialize memory system
        self.memory = GameMemory(
            user_id=user_id, 
            storage_dir="./memory_data",
            llm_client=self.llm_client
        )
        
        self.current_conversation = []  # Current conversation history
        self.turn_count = 0
        
        # Create log file
        log_time = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("./logs", exist_ok=True)
        self.log_file = f"./logs/chatbot_{user_id}_{log_time}.log"
        self._log(f"Initialized chatbot: user_id={user_id}, model={model}")
        self._log(f"Loaded existing memories: {len([m for m in self.memory.memories if m.valid == 1])} items")
    
    def _log(self, message: str):
        """Write to log file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
        print(f"   [LOG] {message}")
    
    def chat(self, user_input: str) -> str:
        """
        Chat and update memories
        
        Args:
            user_input: User input text
            
        Returns:
            Assistant response
        """
        self.turn_count += 1
        print(f"\n{'='*80}")
        print(f"Turn {self.turn_count}")
        print(f"{'='*80}")
        
        # 1. Record user input
        print(f"\n👤 Player input: {user_input}")
        self.current_conversation.append(f"Player: {user_input}")
        
        # 2. Retrieve relevant memories
        print(f"\n🔍 Retrieving relevant memories...")
        self._log(f"Starting memory retrieval, query context: {user_input}")
        
        recent_context = "\n".join(self.current_conversation[-5:])
        relevant_memories = self.memory.retrieval_relevant_memory(recent_context, top_k=3, debug=False)
        
        if relevant_memories:
            print(f"   Found {len(relevant_memories)} relevant memories:")
            self._log(f"Retrieved {len(relevant_memories)} relevant memories:")
            for i, mem in enumerate(relevant_memories, 1):
                print(f"   {i}. [Priority{mem.priority}] {mem.content}")
                self._log(f"  Memory{i}: ID={mem.id}, Keywords={mem.keywords}, Content={mem.content}")
        else:
            print(f"   No relevant memories found")
            self._log("No relevant memories found")
        
        # 3. Build prompt with memories
        system_prompt = "你是一个专业的游戏助手，擅长提供游戏建议和陪伴玩家。"
        
        if relevant_memories:
            memory_text = "\n".join([f"- {mem.content}" for mem in relevant_memories])
            system_prompt += f"\n\n【关于这位玩家的信息】\n{memory_text}\n\n使用说明：\n1. 这些是关于玩家的真实信息，你确实知道\n2. 不要主动炫耀你知道这些，保持自然对话\n3. 当玩家问起相关话题时，可以自然地回答\n4. 例如：玩家问\"我生日是什么时候\"→你可以回答\"2月12日啊\"\n5. 不要说\"我不记得\"或\"我没有存储\"，因为你确实知道"
        
        # 4. Call LLM to generate response
        print(f"\n🤖 生成回复...")
        conversation_context = "\n".join(self.current_conversation[-10:])
        
        try:
            response = self.llm_client.chat(
                prompt=f"{conversation_context}\n助手:",
                system=system_prompt,
                temperature=0.8,
                debug=False
            )
        except Exception as e:
            print(f"   ❌ 生成回复时出错: {e}")
            self._log(f"生成回复时出错: {e}")
            return "抱歉，我遇到了一个错误，请重试。"
        
        print(f"   助手回复: {response}")
        self.current_conversation.append(f"助手: {response}")
        
        # 5. Extract and update memories
        print(f"\n💾 Updating memories...")
        old_memory_count = len([m for m in self.memory.memories if m.valid == 1])
        
        # Extract memory every 3 turns (avoid too frequent)
        if self.turn_count % 3 == 0:
            self._log(f"Triggered memory extraction (turn {self.turn_count})")
            conversation_text = "\n".join(self.current_conversation)
            
            # Record extracted key information
            key_info = self.memory.summarize_key_info(conversation_text, "chat", debug=False)
            self._log(f"Extracted key information: {key_info}")
            
            new_count = self.memory.update_personal_memory_with_messages(conversation_text, debug=False)
            new_memory_count = len([m for m in self.memory.memories if m.valid == 1])
            
            print(f"   Previous memories: {old_memory_count} items")
            print(f"   Current memories: {new_memory_count} items")
            print(f"   This extraction: {new_count} new memories")
            self._log(f"Memory update complete: {old_memory_count} items -> {new_memory_count} items (added {new_count} items)")
            
            if new_count > 0:
                print(f"\n   📝 Memory change details:")
                # Show all memory changes (including add, update, delete)
                valid_memories = [m for m in self.memory.memories if m.valid == 1]
                invalid_memories = [m for m in self.memory.memories if m.valid == 0]
                
                # Show added
                new_mems = sorted([m for m in valid_memories if m.create_time == m.update_time], 
                                 key=lambda x: x.create_time, reverse=True)[:new_count]
                if new_mems:
                    print(f"      ➕ Added:")
                    for mem in new_mems:
                        print(f"         • [Priority{mem.priority}] {mem.content}")
                        self._log(f"Added memory: ID={mem.id}, Keywords={mem.keywords}, Content={mem.content}, Priority={mem.priority}")
                
                # Show updated
                updated_mems = [m for m in valid_memories if m.create_time != m.update_time 
                               and m.update_time > self.memory.statistics.get("last_update", "")]
                if updated_mems:
                    print(f"      🔄 Updated:")
                    for mem in updated_mems:
                        print(f"         • [Priority{mem.priority}] {mem.content}")
                        self._log(f"Updated memory: ID={mem.id}, New content={mem.content}, Priority={mem.priority}")
                
                # Show deleted
                recent_deleted = [m for m in invalid_memories 
                                 if m.update_time > self.memory.statistics.get("last_update", "")]
                if recent_deleted:
                    print(f"      ❌ Deleted:")
                    for mem in recent_deleted:
                        print(f"         • {mem.content}")
                        self._log(f"Deleted memory: ID={mem.id}, Content={mem.content}")
        else:
            print(f"   (Memory extracted every 3 turns, skipping this turn)")
            self._log(f"Skipped memory extraction (turn {self.turn_count}, not extraction cycle)")
        
        return response
    
    def show_all_memories(self):
        """Display all memories"""
        print(f"\n{'='*80}")
        print(f"当前所有记忆 (共 {len([m for m in self.memory.memories if m.valid == 1])} 条)")
        print(f"{'='*80}\n")
        
        valid_memories = [m for m in self.memory.memories if m.valid == 1]
        for mem in sorted(valid_memories, key=lambda x: x.priority):
            print(f"[优先级 {mem.priority}] [{mem.source}] {mem.content}")
            print(f"   关键词: {mem.keywords}")
            print(f"   访问次数: {mem.access_count} | 创建时间: {mem.create_time}\n")


def demo_conversation():
    """Demo complete conversation flow"""
    print("\n" + "="*80)
    print("游戏AI助手 - 记忆系统演示")
    print("="*80)
    
    # Initialize chatbot
    try:
        bot = MemoryChatBot(user_id="demo_player_2024", model="deepseek-v3.1:671b-cloud")
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Simulate a series of conversations
    conversations = [
        "你好，我是新手玩家",
        "我的生日是2月12日，请记住",
        "我喜欢玩射手位置，特别是鲁班",
        "顺便说一下，我通常晚上8点到10点玩游戏",
        "给我推荐一个英雄",  # Should use previous memories
        "我最近想上分，有什么建议吗？",  # Should use memories too
        "我的技术水平怎么样？",
        "我经常被刺客杀死，应该怎么办？",
        "今天是我的生日！",  # Test if it remembers birthday
    ]
    
    for i, user_input in enumerate(conversations, 1):
        bot.chat(user_input)
        if i < len(conversations):
            print(f"\n{'='*80}")
            print(f"⏸️  Completed {i}/{len(conversations)} turns, press Enter to continue...")
            print(f"{'='*80}")
            input()
    
    # Display all memories
    bot.show_all_memories()
    
    # Save memories
    print(f"\n💾 保存记忆到文件...")
    bot.memory.save()
    print(f"✅ 记忆已保存至: memory_data/{bot.user_id}_memory.json")


def interactive_mode():
    """Interactive conversation mode"""
    print("\n" + "="*80)
    print("游戏AI助手 - 交互式对话模式")
    print("="*80)
    print("\n输入 'quit' 退出")
    print("输入 'memory' 查看所有记忆")
    print("输入 'stats' 查看统计信息\n")
    
    user_id = input("请输入玩家ID (默认: test_player): ").strip() or "test_player"
    
    try:
        bot = MemoryChatBot(user_id=user_id, model="deepseek-v3.1:671b-cloud")
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Load existing memories
    existing_count = len([m for m in bot.memory.memories if m.valid == 1])
    if existing_count > 0:
        print(f"\n✅ 已加载 {existing_count} 条历史记忆\n")
    
    while True:
        user_input = input("\n你: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            bot.memory.save()
            print("\n👋 再见！记忆已保存。")
            break
        
        if user_input.lower() == 'memory':
            bot.show_all_memories()
            continue
        
        if user_input.lower() == 'stats':
            stats = bot.memory.get_statistics()
            print(f"\n📊 统计信息:")
            print(f"   总记忆: {stats['total_memories']} 条")
            print(f"   核心记忆: {stats['core_memories']} 条")
            print(f"   对话记忆: {stats['chat_memories']} 条")
            print(f"   轨迹记忆: {stats['trajectory_memories']} 条")
            continue
        
        bot.chat(user_input)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Default: run demo
        demo_conversation()
        
        # Ask if user wants to enter interactive mode
        choice = input("\n是否进入交互式对话模式？(y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
