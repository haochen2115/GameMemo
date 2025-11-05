# -*- coding: utf-8 -*-
"""
LLM Client for Ollama
Provides a simple interface to interact with Ollama models
"""

import json
import requests
from typing import Optional, Dict, Any


class OllamaClient:
    """Ollama LLM Client"""
    
    def __init__(self, 
                 model: str = "deepseek-v3.1:671b-cloud",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120):
        """
        Initialize Ollama client
        
        Args:
            model: Model name (e.g., "deepseek-v3.1:671b-cloud", "llama3.2:7b")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_endpoint = f"{self.base_url}/api/chat"
    
    def chat(self, 
             prompt: str,
             system: str = "",
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             debug: bool = False) -> str:
        """
        Send a chat request to Ollama
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (optional)
            debug: Print debug information
            
        Returns:
            Model response text
            
        Raises:
            RuntimeError: If the request fails
        """
        # Construct messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        if debug:
            print(f"\n[DEBUG] Request to Ollama:")
            print(f"  Model: {self.model}")
            print(f"  Temperature: {temperature}")
            print(f"  System: {system[:100]}..." if len(system) > 100 else f"  System: {system}")
            print(f"  Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"  Prompt: {prompt}")
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            message_content = result.get("message", {}).get("content", "")
            
            if debug:
                print(f"\n[DEBUG] Response from Ollama:")
                print(f"  Content: {message_content[:200]}..." if len(message_content) > 200 else f"  Content: {message_content}")
            
            return message_content
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request to Ollama timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running (try: ollama serve)"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request to Ollama failed: {str(e)}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse Ollama response: {str(e)}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """
        List available models
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []


def test_ollama_connection():
    """Test Ollama connection and list available models"""
    print("测试 Ollama 连接...")
    
    client = OllamaClient()
    
    # Check availability
    if not client.is_available():
        print("❌ Ollama 不可用。请确保 Ollama 正在运行。")
        print("   运行: ollama serve")
        return False
    
    print("✅ Ollama 正在运行！")
    
    # List models
    models = client.list_models()
    if models:
        print(f"\n📦 可用模型 ({len(models)}):")
        for model in models:
            print(f"   - {model}")
    else:
        print("\n⚠️  未找到模型。请先拉取模型。")
        print("   示例: ollama pull deepseek-v3.1:671b-cloud")
        return False
    
    # Test chat
    print("\n🧪 测试聊天功能...")
    try:
        response = client.chat(
            prompt="用一句话说'你好，世界！'",
            system="你是一个有帮助的助手。",
            temperature=0.1
        )
        print(f"✅ 聊天测试成功！")
        print(f"   回复: {response}")
        return True
    except Exception as e:
        print(f"❌ 聊天测试失败: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_ollama_connection()
    
    if success:
        print("\n✅ 所有测试通过！Ollama 客户端已准备就绪。")
    else:
        print("\n❌ 测试失败。请修复上述问题。")


