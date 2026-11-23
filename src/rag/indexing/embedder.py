"""Dense & Sparse BM25 Embedder cho Hybrid Search trên Qdrant."""

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from qdrant_client.http.models import SparseVector

from config import DENSE_VECTOR_SIZE

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_SPARSE_HASH_SPACE: int = 2**20


class DenseEmbedder(Protocol):
    """Giao diện chuẩn cho mọi mô hình Dense Embedding (bge-m3, OpenAI, Gemini)."""

    @property
    def vector_size(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class SparseBM25Encoder:
    """Bộ mã hóa từ khóa thưa (Sparse Vector) nhẹ theo công thức TF-IDF/BM25 sublinear."""

    def __init__(self, k1: float = 1.5) -> None:
        self.k1 = k1

    def encode(self, text: str) -> SparseVector:
        """Chuyển văn bản thành Qdrant SparseVector (indices ổn định qua Blake2b)."""
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return SparseVector(indices=[], values=[])

        counts = Counter(tokens)
        index_weights: dict[int, float] = {}
        for token, freq in counts.items():
            idx = self._token_to_index(token)
            tf_weight = (freq * (self.k1 + 1.0)) / (freq + self.k1)
            index_weights[idx] = index_weights.get(idx, 0.0) + tf_weight

        sorted_items = sorted(index_weights.items())
        return SparseVector(
            indices=[idx for idx, _ in sorted_items],
            values=[val for _, val in sorted_items],
        )

    @staticmethod
    def _token_to_index(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % _SPARSE_HASH_SPACE


class DeterministicHashDenseEmbedder:
    """Embedder nhẹ dùng để test/dev offline khi chưa tải model bge-m3 nặng."""

    def __init__(self, size: int = DENSE_VECTOR_SIZE) -> None:
        self._size = size

    @property
    def vector_size(self) -> int:
        return self._size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._size
        tokens = _TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
