# -*- coding: utf-8 -*-
"""Text embedders for dense retrieval.

An embedder maps texts to L2-normalised vectors. Results are cached per
text, so re-ranking the same memory bank never re-embeds stored memories.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol, Sequence

Vector = List[float]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        ...

    def embed_query(self, text: str) -> Vector:
        ...


def normalize(v: Sequence[float]) -> Vector:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class _CachedEmbedder:
    name = "base"
    query_prefix = ""

    def __init__(self):
        self._cache: Dict[str, Vector] = {}

    def _embed(self, texts: List[str]) -> List[Vector]:
        raise NotImplementedError

    def _get(self, texts: Sequence[str]) -> List[Vector]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._embed(missing)):
                self._cache[t] = normalize(v)
        return [self._cache[t] for t in texts]

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return self._get(texts)

    def embed_query(self, text: str) -> Vector:
        return self._get([self.query_prefix + text])[0]


class OllamaEmbedder(_CachedEmbedder):
    """Embeddings served by Ollama, e.g. ``ollama pull bge-m3``."""

    def __init__(self, client, model: str = "bge-m3", query_prefix: str = ""):
        super().__init__()
        self.client = client
        self.model = model
        self.name = f"ollama:{model}"
        self.query_prefix = query_prefix

    def _embed(self, texts: List[str]) -> List[Vector]:
        return self.client.embed(texts, model=self.model)


class FastEmbedEmbedder(_CachedEmbedder):
    """Local ONNX embeddings via ``fastembed`` (no server, no torch).

    The default ``BAAI/bge-small-zh-v1.5`` is ~100 MB and runs on CPU; it is
    what the offline benchmark uses so results are reproducible anywhere.
    """

    BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5", query_prefix: Optional[str] = None):
        super().__init__()
        from fastembed import TextEmbedding  # optional dependency

        self.model = TextEmbedding(model)
        self.name = f"fastembed:{model}"
        if query_prefix is None:
            query_prefix = self.BGE_ZH_QUERY_PREFIX if "bge" in model and "zh" in model else ""
        self.query_prefix = query_prefix

    def _embed(self, texts: List[str]) -> List[Vector]:
        return [list(map(float, v)) for v in self.model.embed(texts)]
