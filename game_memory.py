# -*- coding: utf-8 -*-
"""
Game Memory System
An intelligent memory management system for game AI assistants

Features:
- Extract player attributes from conversations
- Extract player preferences from game trajectories  
- Retrieve relevant memories using keyword and semantic matching
- Support CRUD operations on memories
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from difflib import SequenceMatcher

from llm_client import OllamaClient


class Memory:
    """Memory object representing a single memory entry"""
    
    def __init__(self, 
                 keywords: str,
                 content: str,
                 source: str = "chat",
                 priority: int = 3,
                 mem_id: str = None):
        """
        Initialize a memory entry
        
        Args:
            keywords: Comma-separated keywords for retrieval
            content: Memory content (factual information)
            source: Memory source ("chat" or "trajectory")
            priority: Priority level (1=core, 2-3=important, 4-5=general)
            mem_id: Memory ID (auto-generated if not provided)
        """
        self.id = mem_id if mem_id else f"mem_{uuid.uuid4().hex[:8]}"
        self.keywords = keywords
        self.content = content
        self.source = source
        self.valid = 1
        self.priority = priority
        self.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.update_time = self.create_time
        self.access_count = 0
        self.last_access_time = self.create_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "keywords": self.keywords,
            "content": self.content,
            "source": self.source,
            "valid": self.valid,
            "priority": self.priority,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Memory':
        """Create memory from dictionary"""
        mem = Memory(
            keywords=data["keywords"],
            content=data["content"],
            source=data.get("source", "chat"),
            priority=data.get("priority", 3),
            mem_id=data.get("id")
        )
        mem.valid = data.get("valid", 1)
        mem.create_time = data.get("create_time", mem.create_time)
        mem.update_time = data.get("update_time", mem.update_time)
        mem.access_count = data.get("access_count", 0)
        mem.last_access_time = data.get("last_access_time", mem.last_access_time)
        return mem
    
    def update_access(self):
        """Update access information"""
        self.access_count += 1
        self.last_access_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class GameMemory:
    """Game Memory System for managing player memories"""
    
    # Prompt 模板
    PROMPT_SUMMARIZE_CHAT = """你是一个游戏AI助手，负责从玩家的对话中提取关键信息。

请从以下对话中提取关键信息点，每个信息点用一句话概括。
重点关注：个人属性（生日、职业等）、游戏偏好（喜欢的英雄、打法等）、重要事件（成就、特殊经历等）。

对话记录：
{messages}

请以JSON格式返回，格式如下：
{{"key_info": ["信息点1", "信息点2", ...]}}

只返回JSON，不要有其他内容。"""

    PROMPT_SUMMARIZE_TRAJECTORY = """你是一个游戏AI助手，负责从玩家的游戏轨迹中提取关键行为模式。

请从以下游戏轨迹中提取玩家的偏好和习惯，每个特征用一句话概括。
重点关注：英雄使用频率、胜率、出装偏好、操作习惯、活跃时间等。

游戏轨迹：
{trajectories}

请以JSON格式返回，格式如下：
{{"key_info": ["特征1", "特征2", ...]}}

只返回JSON，不要有其他内容。"""

    PROMPT_EXTRACT_MEMORY_FROM_INFO = """你是一个游戏AI助手，负责将关键信息转换为结构化记忆。

【已有记忆】
{existing_memories}

【新的关键信息】
{key_info}

请分析新信息与已有记忆的关系，决定操作类型：

1. 如果是全新信息 → "action": "add"
2. 如果是更新已有信息 → "action": "update", "update_id": "mem_xxx"
3. 如果新信息表明旧信息已过期/错误 → "action": "delete", "delete_id": "mem_xxx"

返回JSON格式：
{{
  "operations": [
    {{"action": "add", "keywords": "关键词1, 关键词2", "content": "简洁事实", "priority": 1-5}},
    {{"action": "update", "update_id": "mem_xxx", "keywords": "新关键词", "content": "更新后的内容", "priority": 2}},
    {{"action": "delete", "delete_id": "mem_yyy", "reason": "信息已过期"}}
  ]
}}

要求：
- content只写简洁事实，不加修饰
- 例如："玩家生日是2月12日"、"常用鲁班七号，胜率62%"
- 判断是否需要更新（如段位变化、胜率变化等）
- 判断是否需要删除（如已退游戏、偏好改变等）

优先级：1=核心信息，2-3=重要信息，4-5=一般信息

只返回JSON。"""

    PROMPT_EXTRACT_KEYWORDS = """请从以下文本中提取2-5个关键词，用于记忆检索。

文本：
{text}

请以JSON格式返回，格式如下：
{{"keywords": ["关键词1", "关键词2", ...]}}

只返回JSON，不要有其他内容。"""

    def __init__(self, 
                 user_id: str, 
                 storage_dir: str = "./memory_data",
                 llm_client: Optional[OllamaClient] = None,
                 model: str = "deepseek-v3.1:671b-cloud"):
        """
        Initialize GameMemory system
        
        Args:
            user_id: User ID
            storage_dir: Directory for storing memory files
            llm_client: LLM client instance (creates default if not provided)
            model: Model name for LLM (used if llm_client is not provided)
        """
        self.user_id = user_id
        self.storage_dir = storage_dir
        self.llm_client = llm_client if llm_client else OllamaClient(model=model)
        self.memories: List[Memory] = []
        self.statistics = {
            "total_memories": 0,
            "core_memories": 0,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing memories
        self.load()
    
    def _call_llm(self, prompt: str, debug: bool = False) -> str:
        """
        Call LLM
        
        Args:
            prompt: Prompt text
            debug: Print debug information
            
        Returns:
            Model output
        """
        system_prompt = "你是一个专业的游戏AI助手，擅长理解玩家需求和游戏相关内容。请严格按照指令返回JSON格式。"
        
        try:
            return self.llm_client.chat(
                prompt=prompt,
                system=system_prompt,
                temperature=0.3,  # Lower temperature for accuracy
                debug=debug
            )
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON response
        
        Args:
            response: Model response
            
        Returns:
            Parsed dictionary, None if failed
        """
        if not response:
            return None
        
        try:
            # Try direct parsing
            return json.loads(response)
        except:
            # Try to extract JSON part
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
        
        print(f"[WARNING] Unable to parse JSON response: {response[:100]}...")
        return None
    
    # ================ Core APIs ================
    
    def summarize_key_info(self, text: str, text_type: str = "chat", debug: bool = False) -> List[str]:
        """
        Summarize key information from text
        
        Args:
            text: Raw text (conversation or trajectory)
            text_type: Text type ("chat" or "trajectory")
            debug: Print debug information
            
        Returns:
            List of key information
        """
        if text_type == "chat":
            prompt = self.PROMPT_SUMMARIZE_CHAT.format(messages=text)
        else:
            prompt = self.PROMPT_SUMMARIZE_TRAJECTORY.format(trajectories=text)
        
        response = self._call_llm(prompt, debug)
        result = self._parse_json_response(response)
        
        if result and "key_info" in result:
            return result["key_info"]
        
        return []
    
    def retrieval_relevant_memory(self, query: str, top_k: int = 5, debug: bool = False) -> List[Memory]:
        """
        Retrieve relevant memories
        
        Args:
            query: Query text
            top_k: Return top-k memories
            debug: Print debug information
            
        Returns:
            List of relevant memories
        """
        if not self.memories:
            return []
        
        # Extract query keywords
        prompt = self.PROMPT_EXTRACT_KEYWORDS.format(text=query)
        response = self._call_llm(prompt, debug)
        result = self._parse_json_response(response)
        
        query_keywords = []
        if result and "keywords" in result:
            query_keywords = result["keywords"]
        
        # Calculate relevance scores
        scored_memories = []
        for mem in self.memories:
            if mem.valid == 0:  # Skip deleted memories
                continue
            
            score = self._calculate_relevance(query, query_keywords, mem)
            scored_memories.append((score, mem))
        
        # Sort and return top-k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Update access information
        result_memories = []
        for score, mem in scored_memories[:top_k]:
            if score > 0:  # Only return relevant memories
                mem.update_access()
                result_memories.append(mem)
        
        return result_memories
    
    def _calculate_relevance(self, query: str, query_keywords: List[str], memory: Memory) -> float:
        """
        Calculate relevance score
        
        Args:
            query: Query text
            query_keywords: Query keywords
            memory: Memory object
            
        Returns:
            Relevance score
        """
        score = 0.0
        query_lower = query.lower()
        content_lower = memory.content.lower()
        mem_keywords = [k.strip() for k in memory.keywords.split(',')]
        
        # 1. Query contains memory keywords (weight 1.0, highest)
        for mk in mem_keywords:
            if mk.lower() in query_lower:
                score += 1.0
        
        # 2. Memory content contains query keywords (weight 0.8)
        for qk in query_keywords:
            if qk.lower() in content_lower:
                score += 0.8
        
        # 3. Exact keyword match (weight 0.6)
        for qk in query_keywords:
            if qk in mem_keywords:
                score += 0.6
        
        # 4. Fuzzy keyword match (weight 0.4)
        for qk in query_keywords:
            for mk in mem_keywords:
                similarity = SequenceMatcher(None, qk.lower(), mk.lower()).ratio()
                if similarity > 0.7:
                    score += 0.4 * similarity
        
        # 5. Query words in memory content (weight 0.3)
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 1 and word in content_lower:
                score += 0.3
        
        # 6. Priority weighting (core memories are more important)
        priority_weight = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.6}
        score *= priority_weight.get(memory.priority, 1.0)
        
        return score
    
    def update_memory(self, operation: str, memory: Memory = None, mem_id: str = None, debug: bool = False) -> bool:
        """
        Update memory
        
        Args:
            operation: Operation type ("add", "update", "delete")
            memory: Memory object (required for add/update)
            mem_id: Memory ID (required for update/delete)
            debug: Print debug information
            
        Returns:
            Success status
        """
        if operation == "add":
            if not memory:
                print("[ERROR] add operation requires memory object")
                return False
            
            # Check for duplicate memories
            similar_mem = self._find_similar_memory(memory)
            if similar_mem:
                if debug:
                    print(f"[INFO] Found similar memory, merging update: {similar_mem.id}")
                # Merge memory content
                if len(memory.content) > len(similar_mem.content):
                    similar_mem.content = memory.content
                similar_mem.keywords = memory.keywords
                similar_mem.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                similar_mem.priority = min(similar_mem.priority, memory.priority)
            else:
                self.memories.append(memory)
                if debug:
                    print(f"[INFO] Added new memory: {memory.id}")
        
        elif operation == "update":
            if not mem_id or not memory:
                print("[ERROR] update operation requires mem_id and memory object")
                return False
            
            for i, mem in enumerate(self.memories):
                if mem.id == mem_id:
                    memory.id = mem_id
                    memory.create_time = mem.create_time
                    memory.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.memories[i] = memory
                    if debug:
                        print(f"[INFO] Updated memory: {mem_id}")
                    break
        
        elif operation == "delete":
            if not mem_id:
                print("[ERROR] delete operation requires mem_id")
                return False
            
            for mem in self.memories:
                if mem.id == mem_id:
                    mem.valid = 0
                    mem.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if debug:
                        print(f"[INFO] Deleted memory: {mem_id}")
                    break
        
        # Check core memory limit
        self._check_core_memory_limit()
        
        # Save
        self.save()
        return True
    
    def _find_similar_memory(self, memory: Memory, threshold: float = 0.8) -> Optional[Memory]:
        """
        Find similar memory
        
        Args:
            memory: Memory to check
            threshold: Similarity threshold
            
        Returns:
            Similar memory object, None if not found
        """
        mem_keywords = set([k.strip() for k in memory.keywords.split(',')])
        
        for existing_mem in self.memories:
            if existing_mem.valid == 0:
                continue
            
            existing_keywords = set([k.strip() for k in existing_mem.keywords.split(',')])
            
            # Calculate keyword similarity (Jaccard similarity)
            intersection = len(mem_keywords & existing_keywords)
            union = len(mem_keywords | existing_keywords)
            
            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    return existing_mem
        
        return None
    
    def _check_core_memory_limit(self, max_core: int = 20):
        """
        Check core memory limit
        
        Args:
            max_core: Maximum number of core memories
        """
        core_memories = [m for m in self.memories if m.priority == 1 and m.valid == 1]
        
        if len(core_memories) > max_core:
            # Sort by access count, keep most frequently used
            core_memories.sort(key=lambda x: x.access_count, reverse=True)
            for mem in core_memories[max_core:]:
                mem.priority = 2  # Downgrade to important
                print(f"[INFO] Core memory limit exceeded, downgraded: {mem.id}")
    
    # ================ Combined APIs ================
    
    def update_personal_memory_with_messages(self, messages: str, debug: bool = False) -> int:
        """
        Update personal memory from conversation
        
        Args:
            messages: Conversation text
            debug: Print debug information
            
        Returns:
            Number of memories added/updated
        """
        # 1. Summarize key information
        key_info_list = self.summarize_key_info(messages, "chat", debug)
        if not key_info_list:
            return 0
        
        if debug:
            print(f"[INFO] Extracted {len(key_info_list)} key information items")
        
        # 2. Convert to structured memories (with existing memory context)
        key_info_text = "\n".join([f"- {info}" for info in key_info_list])
        
        # Get existing memories
        existing_mems = [m for m in self.memories if m.valid == 1]
        existing_text = "None" if not existing_mems else "\n".join([
            f"[{m.id}] {m.content}" for m in existing_mems
        ])
        
        prompt = self.PROMPT_EXTRACT_MEMORY_FROM_INFO.format(
            existing_memories=existing_text,
            key_info=key_info_text
        )
        response = self._call_llm(prompt, debug)
        result = self._parse_json_response(response)
        
        if not result or "operations" not in result:
            return 0
        
        # 3. Execute operations (add/update/delete)
        count = 0
        for op in result["operations"]:
            action = op.get("action")
            
            if action == "add":
                memory = Memory(
                    keywords=op.get("keywords", ""),
                    content=op.get("content", ""),
                    source="chat",
                    priority=op.get("priority", 3)
                )
                if self.update_memory("add", memory=memory, debug=debug):
                    count += 1
                    
            elif action == "update":
                update_id = op.get("update_id")
                if update_id:
                    memory = Memory(
                        keywords=op.get("keywords", ""),
                        content=op.get("content", ""),
                        source="chat",
                        priority=op.get("priority", 3)
                    )
                    if self.update_memory("update", memory=memory, mem_id=update_id, debug=debug):
                        if debug:
                            print(f"[INFO] Updated memory: {update_id}")
                        count += 1
                        
            elif action == "delete":
                delete_id = op.get("delete_id")
                if delete_id:
                    if self.update_memory("delete", mem_id=delete_id, debug=debug):
                        if debug:
                            print(f"[INFO] Deleted memory: {delete_id}, reason: {op.get('reason', 'unknown')}")
                        count += 1
        
        if debug:
            print(f"[INFO] Successfully updated {count} memories")
        
        return count
    
    def update_personal_memory_with_trajectories(self, trajectories: str, debug: bool = False) -> int:
        """
        Update personal memory from game trajectories
        
        Args:
            trajectories: Game trajectory text
            debug: Print debug information
            
        Returns:
            Number of memories added/updated
        """
        # 1. Summarize key behaviors
        key_info_list = self.summarize_key_info(trajectories, "trajectory", debug)
        if not key_info_list:
            return 0
        
        if debug:
            print(f"[INFO] Extracted {len(key_info_list)} key behaviors")
        
        # 2. Convert to structured memories (with existing memory context)
        key_info_text = "\n".join([f"- {info}" for info in key_info_list])
        
        # Get existing memories
        existing_mems = [m for m in self.memories if m.valid == 1]
        existing_text = "None" if not existing_mems else "\n".join([
            f"[{m.id}] {m.content}" for m in existing_mems
        ])
        
        prompt = self.PROMPT_EXTRACT_MEMORY_FROM_INFO.format(
            existing_memories=existing_text,
            key_info=key_info_text
        )
        response = self._call_llm(prompt, debug)
        result = self._parse_json_response(response)
        
        if not result or "operations" not in result:
            return 0
        
        # 3. Execute operations (add/update/delete)
        count = 0
        for op in result["operations"]:
            action = op.get("action")
            
            if action == "add":
                memory = Memory(
                    keywords=op.get("keywords", ""),
                    content=op.get("content", ""),
                    source="trajectory",
                    priority=op.get("priority", 3)
                )
                if self.update_memory("add", memory=memory, debug=debug):
                    count += 1
                    
            elif action == "update":
                update_id = op.get("update_id")
                if update_id:
                    memory = Memory(
                        keywords=op.get("keywords", ""),
                        content=op.get("content", ""),
                        source="trajectory",
                        priority=op.get("priority", 3)
                    )
                    if self.update_memory("update", memory=memory, mem_id=update_id, debug=debug):
                        if debug:
                            print(f"[INFO] Updated memory: {update_id}")
                        count += 1
                        
            elif action == "delete":
                delete_id = op.get("delete_id")
                if delete_id:
                    if self.update_memory("delete", mem_id=delete_id, debug=debug):
                        if debug:
                            print(f"[INFO] Deleted memory: {delete_id}, reason: {op.get('reason', 'unknown')}")
                        count += 1
        
        if debug:
            print(f"[INFO] Successfully updated {count} memories")
        
        return count
    
    def retrieval_personal_memory_with_messages(self, messages: str, top_k: int = 5, debug: bool = False) -> str:
        """
        Retrieve relevant memories based on conversation context and format
        
        Args:
            messages: Current conversation context
            top_k: Return top-k memories
            debug: Print debug information
            
        Returns:
            Formatted memory text
        """
        # 1. Extract key information
        key_info_list = self.summarize_key_info(messages, "chat", debug)
        if not key_info_list:
            return ""
        
        query = " ".join(key_info_list)
        
        # 2. Retrieve relevant memories
        memories = self.retrieval_relevant_memory(query, top_k, debug)
        
        if not memories:
            return ""
        
        # 3. Format output
        formatted = "### 相关记忆 ###\n"
        for i, mem in enumerate(memories, 1):
            formatted += f"{i}. {mem.content}\n"
        
        return formatted
    
    # ================ Storage ================
    
    def save(self):
        """Save memories to file"""
        data = {
            "user_id": self.user_id,
            "memories": [m.to_dict() for m in self.memories],
            "statistics": {
                "total_memories": len([m for m in self.memories if m.valid == 1]),
                "core_memories": len([m for m in self.memories if m.valid == 1 and m.priority == 1]),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        file_path = os.path.join(self.storage_dir, f"{self.user_id}_memory.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Load memories from file"""
        file_path = os.path.join(self.storage_dir, f"{self.user_id}_memory.json")
        
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.memories = [Memory.from_dict(m) for m in data.get("memories", [])]
            self.statistics = data.get("statistics", self.statistics)
            
            print(f"[INFO] Loaded {len(self.memories)} memories")
        except Exception as e:
            print(f"[ERROR] Failed to load memories: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        valid_memories = [m for m in self.memories if m.valid == 1]
        
        return {
            "user_id": self.user_id,
            "total_memories": len(valid_memories),
            "core_memories": len([m for m in valid_memories if m.priority == 1]),
            "chat_memories": len([m for m in valid_memories if m.source == "chat"]),
            "trajectory_memories": len([m for m in valid_memories if m.source == "trajectory"]),
            "most_accessed": sorted(valid_memories, key=lambda x: x.access_count, reverse=True)[:5],
            "last_update": self.statistics.get("last_update", "")
        }
    
    def print_all_memories(self):
        """Print all memories"""
        valid_memories = [m for m in self.memories if m.valid == 1]
        
        print(f"\n{'='*60}")
        print(f"用户 {self.user_id} 的记忆库 (共 {len(valid_memories)} 条)")
        print(f"{'='*60}\n")
        
        for mem in sorted(valid_memories, key=lambda x: x.priority):
            print(f"[ID: {mem.id}] [优先级: {mem.priority}] [来源: {mem.source}]")
            print(f"关键词: {mem.keywords}")
            print(f"内容: {mem.content}")
            print(f"创建: {mem.create_time} | 访问: {mem.access_count}次")
            print(f"{'-'*60}\n")


if __name__ == "__main__":
    # 简单测试
    print("测试 GameMemory 系统...")
    print("=" * 60)
    
    # 检查 Ollama 是否可用
    client = OllamaClient()
    if not client.is_available():
        print("\n❌ 错误：Ollama 未运行！")
        print("请先启动 Ollama：")
        print("  1. 运行: ollama serve")
        print("  2. 确保有模型: ollama pull deepseek-v3.1:671b-cloud")
        exit(1)
    
    print("✅ Ollama 正在运行\n")
    
    # 初始化记忆系统
    memory_system = GameMemory(user_id="test_player_001", model="deepseek-v3.1:671b-cloud")
    
    # 测试对话记忆
    test_messages = """
    玩家: 你好，我是新手玩家
    助手: 你好，欢迎来到游戏！
    玩家: 我的生日是2月12日，请记住
    助手: 好的，你的生日是2月12日
    玩家: 我喜欢玩射手英雄，特别是鲁班
    """
    
    print("测试从对话中提取记忆...")
    count = memory_system.update_personal_memory_with_messages(test_messages, debug=True)
    print(f"\n成功提取 {count} 条记忆\n")
    
    # 打印所有记忆
    memory_system.print_all_memories()
