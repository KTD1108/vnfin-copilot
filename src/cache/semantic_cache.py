"""Entity-Aware Semantic Cache: Lọc chính xác Ticker/Year + Cosine >= 0.95 + TTL."""

import math
import time
from dataclasses import dataclass

from rag.indexing.embedder import DenseEmbedder

DEFAULT_SIMILARITY_THRESHOLD: float = 0.95
DEFAULT_HISTORICAL_TTL_SEC: int = 86400
DEFAULT_INTRADAY_TTL_SEC: int = 300


@dataclass
class CacheEntry:
    """Một mục lưu trữ trong Semantic Cache kèm Metadata thực thể và TTL."""

    query: str
    embedding: list[float]
    answer: str
    ticker: str | None
    year: int | None
    expires_at: float


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


class EntityAwareSemanticCache:
    """Bộ nhớ đệm ngữ nghĩa bắt buộc khớp Metadata (ticker, year) để tránh trả nhầm mã CK."""

    def __init__(
        self,
        embedder: DenseEmbedder,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._entries: list[CacheEntry] = []

    def get(
        self,
        query: str,
        ticker: str | None = None,
        year: int | None = None,
    ) -> str | None:
        """Tìm câu trả lời trong cache nếu khớp chính xác (ticker, year) và Cosine >= ngưỡng."""
        now = time.monotonic()
        norm_ticker = ticker.upper() if ticker else None
        query_vec = self._embedder.embed_query(query)

        best_score = -1.0
        best_answer: str | None = None

        for entry in self._entries:
            if entry.expires_at <= now:
                continue
            if entry.ticker != norm_ticker or entry.year != year:
                continue
            score = _cosine_similarity(query_vec, entry.embedding)
            if score >= self._threshold and score > best_score:
                best_score = score
                best_answer = entry.answer

        return best_answer

    def set(
        self,
        query: str,
        answer: str,
        ticker: str | None = None,
        year: int | None = None,
        ttl_seconds: int = DEFAULT_HISTORICAL_TTL_SEC,
    ) -> None:
        """Lưu kết quả mới vào cache với TTL tương ứng."""
        now = time.monotonic()
        norm_ticker = ticker.upper() if ticker else None
        vec = self._embedder.embed_query(query)
        self._entries.append(
            CacheEntry(
                query=query,
                embedding=vec,
                answer=answer,
                ticker=norm_ticker,
                year=year,
                expires_at=now + ttl_seconds,
            )
        )
