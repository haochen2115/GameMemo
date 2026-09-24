# -*- coding: utf-8 -*-
"""GameMemo: long-term memory for game AI assistants."""

from .llm import FakeLLM, OllamaClient
from .personal import MemoryChatBot, PersonalMemory

__version__ = "0.2.0.dev0"

__all__ = ["FakeLLM", "MemoryChatBot", "OllamaClient", "PersonalMemory"]
