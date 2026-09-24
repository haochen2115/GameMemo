# -*- coding: utf-8 -*-
"""Hybrid memory retrieval.

Pipeline, all local and without any LLM call:

1. Lexical: BM25 over jieba tokens of content + keywords.
2. Dense (optional): cosine similarity of embeddings.
3. Relevance gate: a memory is a candidate only if at least one signal is
   convincingly above noise; this is what lets the retriever return
   *nothing* for an unrelated query instead of padding the prompt.
4. Fusion: reciprocal-rank fusion of the signals, then a small (±10%)
   modulation by importance and recency so they break ties but never
   override relevance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .embed import Embedder, Reranker, cosine
from .model import MemoryRecord, parse_time
from .text import add_words, tokenize


# (min_dense, dense_margin) per embedding model, tuned on the retrieval_v2
# dev split only (docs/BENCHMARK.md). Cosine scales differ a lot by model.
CALIBRATED_DENSE = {
    "jina-embeddings-v2-base-zh": (0.25, 0.15),
    "bge-small-zh-v1.5": (0.38, 0.12),
    "bge-m3": (0.50, 0.12),
}


@dataclass
class RetrievalConfig:
    top_k: int = 5
    use_lexical: bool = True
    use_dense: bool = True
    bigrams: bool = False
    keyword_boost: int = 2          # keywords count this many times in BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    min_lexical_coverage: float = 0.34  # idf-weighted share of query terms matched
    min_dense: float = 0.25             # cosine floor (default embedder: jina-v2-base-zh)
    dense_margin: float = 0.15          # also require being near the best match
    w_importance: float = 0.10
    w_recency: float = 0.05
    recency_half_life_days: float = 30.0
    inactive_penalty: float = 0.7       # superseded versions, when history is searched
    rerank_pool: int = 10               # top-N by fused rank sent to the reranker
    min_rerank: float = 0.0             # reranker logit needed to be returned

    def for_candidates(self) -> "RetrievalConfig":
        """Recall-oriented variant for the write path: any lexical overlap or
        a looser semantic match qualifies, and the LLM decides what to do."""
        return replace(self, min_lexical_coverage=0.0, min_dense=self.min_dense - 0.05,
                       dense_margin=1.0, w_recency=0.0)

    @classmethod
    def for_embedder(cls, embedder, **kw) -> "RetrievalConfig":
        """Config with dense thresholds calibrated for this embedder."""
        if embedder is None:
            return cls.lexical_only(**kw)
        for key, (floor, margin) in CALIBRATED_DENSE.items():
            if key in getattr(embedder, "name", ""):
                kw.setdefault("min_dense", floor)
                kw.setdefault("dense_margin", margin)
                break
        return cls(**kw)

    @classmethod
    def lexical_only(cls, **kw) -> "RetrievalConfig":
        """Preset for running without an embedder (calibrated on dev split)."""
        kw.setdefault("min_lexical_coverage", 0.2)
        return cls(use_dense=False, **kw)


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float
    lexical: float = 0.0
    coverage: float = 0.0
    dense: float = 0.0


class HybridRetriever:
    def __init__(self, embedder: Optional[Embedder] = None,
                 config: Optional[RetrievalConfig] = None,
                 reranker: Optional[Reranker] = None):
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self._tok_cache: Dict[Tuple[str, str], List[str]] = {}
        self._known_keywords: set = set()

    # ---- public ----

    def search(self, query: str, records: Sequence[MemoryRecord],
               top_k: Optional[int] = None,
               now: Optional[datetime] = None,
               include_inactive: bool = False) -> List[ScoredMemory]:
        """``include_inactive`` also searches superseded versions (for
        questions about the past); they rank below current ones."""
        cfg = self.config
        top_k = cfg.top_k if top_k is None else top_k
        records = [r for r in records if (include_inactive or r.is_active) and r.content.strip()]
        if not records or not query.strip():
            return []
        self._learn_keywords(records)

        n = len(records)
        lex = [0.0] * n
        cov = [0.0] * n
        dense = [0.0] * n
        lists: List[List[int]] = []
        gate = [False] * n

        if cfg.use_lexical:
            lex, cov = self._bm25(query, records)
            # Only documents that matched something get a lexical rank.
            lists.append(sorted((i for i in range(n) if lex[i] > 0), key=lambda i: -lex[i]))
            for i in range(n):
                gate[i] |= lex[i] > 0 and cov[i] >= cfg.min_lexical_coverage
        if cfg.use_dense and self.embedder is not None:
            dense = self._dense(query, records)
            lists.append(sorted(range(n), key=lambda i: -dense[i]))
            # Absolute floor rejects unrelated queries; the margin keeps only
            # memories close to the best match so weak ones don't pad the prompt.
            floor = max(cfg.min_dense, max(dense) - cfg.dense_margin)
            for i in range(n):
                gate[i] |= dense[i] >= floor
        if not lists:
            return []

        rrf = [0.0] * n
        for order in lists:
            for rank, i in enumerate(order):
                rrf[i] += 1.0 / (cfg.rrf_k + rank + 1)

        now = now or datetime.now()
        if self.reranker is not None:
            # The cross-encoder replaces the heuristic gate: it sees the best
            # fused candidates and decides both order and abstention.
            pool = sorted((i for i in range(n) if rrf[i] > 0), key=lambda i: -rrf[i])[:cfg.rerank_pool]
            rr = self.reranker.score(query, [self._doc_text(records[i]) for i in pool])
            keep = [(i, s) for i, s in zip(pool, rr) if s >= cfg.min_rerank]
            base = {i: s for i, s in keep}
        else:
            base = {i: rrf[i] for i in range(n) if gate[i]}

        scored = []
        for i, b in base.items():
            rec = records[i]
            mod = 1.0 + cfg.w_importance * (rec.importance - 3) / 2.0 \
                + cfg.w_recency * self._recency(rec, now)
            if not rec.is_active:
                mod *= cfg.inactive_penalty
            # Rerank logits can be negative; modulate the margin above the floor.
            value = (b - cfg.min_rerank) * mod if self.reranker is not None else b * mod
            scored.append(ScoredMemory(rec, value, lex[i], cov[i], dense[i]))
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]

    # ---- signals ----

    def _doc_text(self, rec: MemoryRecord) -> str:
        # The aspect ("身份职业") bridges questions that name the category
        # ("做什么工作") to facts that only state the value ("是护士").
        aspect = f" {rec.aspect}" if rec.aspect and rec.aspect != "其他" else ""
        return rec.content + " " + " ".join(rec.keywords) + aspect

    def _tokens(self, key: str, text: str) -> List[str]:
        k = (key, text)
        if k not in self._tok_cache:
            self._tok_cache[k] = tokenize(text, bigrams=self.config.bigrams)
        return self._tok_cache[k]

    def _doc_tokens(self, rec: MemoryRecord) -> List[str]:
        toks = list(self._tokens("c", rec.content))
        kw = self._tokens("k", " ".join(rec.keywords))
        toks.extend(kw * self.config.keyword_boost)
        if rec.aspect and rec.aspect != "其他":
            toks.extend(self._tokens("a", rec.aspect))
        return toks

    def _learn_keywords(self, records: Sequence[MemoryRecord]) -> None:
        new = {k for r in records for k in r.keywords} - self._known_keywords
        if new:
            add_words(new)
            self._known_keywords |= new
            self._tok_cache.clear()

    def _bm25(self, query: str, records: Sequence[MemoryRecord]) -> Tuple[List[float], List[float]]:
        cfg = self.config
        docs = [self._doc_tokens(r) for r in records]
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / n or 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        q_terms = list(dict.fromkeys(tokenize(query, bigrams=cfg.bigrams)))
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
        q_mass = sum(idf.values()) or 1.0

        scores, coverage = [], []
        for d in docs:
            tf = Counter(d)
            s = 0.0
            matched = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                matched += idf[t]
                s += idf[t] * f * (cfg.bm25_k1 + 1) / (
                    f + cfg.bm25_k1 * (1 - cfg.bm25_b + cfg.bm25_b * len(d) / avgdl))
            scores.append(s)
            coverage.append(matched / q_mass)
        return scores, coverage

    def _dense(self, query: str, records: Sequence[MemoryRecord]) -> List[float]:
        q = self.embedder.embed_query(query)
        docs = self.embedder.embed_documents([self._doc_text(r) for r in records])
        return [cosine(q, d) for d in docs]

    def _recency(self, rec: MemoryRecord, now: datetime) -> float:
        t = parse_time(rec.last_accessed_at) or parse_time(rec.updated_at) or parse_time(rec.created_at)
        if t is None:
            return 0.0
        age = max(0.0, (now - t).total_seconds() / 86400.0)
        return 0.5 ** (age / self.config.recency_half_life_days)
