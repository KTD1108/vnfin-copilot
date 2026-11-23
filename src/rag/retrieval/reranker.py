"""Stage 2: Cross-Encoder Reranker chấm điểm tinh chuẩn Top-K ứng viên."""

import logging
import re

from config import DEFAULT_RERANKER_MODEL
from rag.schemas import RankedCandidate

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class BGEReranker:
    """Cross-Encoder Reranker sử dụng BAAI/bge-reranker-v2-m3 qua sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Chưa cài đặt 'sentence-transformers'. Chạy: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name, device=device)
        logger.info("Đã nạp CrossEncoder Reranker: %s", model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RankedCandidate],
        top_k: int,
    ) -> list[RankedCandidate]:
        """Chấm điểm cặp (query, candidate.text) và trả về Top-K ứng viên điểm cao nhất."""
        if not candidates or top_k <= 0:
            return []

        pairs = [(query, c.metadata.text) for c in candidates]
        scores = self._model.predict(pairs)

        rescored = [
            RankedCandidate(
                chunk_id=cand.chunk_id,
                parent_id=cand.parent_id,
                score=float(score),
                metadata=cand.metadata,
            )
            for cand, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]


class LexicalOverlapReranker:
    """Reranker nhẹ tính tỷ lệ trùng từ khóa (dùng cho test/dev offline không cần GPU)."""

    def rerank(
        self,
        query: str,
        candidates: list[RankedCandidate],
        top_k: int,
    ) -> list[RankedCandidate]:
        """Chấm điểm dựa trên Jaccard/Overlap giữa query và nội dung chunk."""
        if not candidates or top_k <= 0:
            return []

        query_tokens = set(_WORD_PATTERN.findall(query.lower()))
        rescored: list[RankedCandidate] = []

        for cand in candidates:
            doc_tokens = set(_WORD_PATTERN.findall(cand.metadata.text.lower()))
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            rescored.append(
                RankedCandidate(
                    chunk_id=cand.chunk_id,
                    parent_id=cand.parent_id,
                    score=float(overlap),
                    metadata=cand.metadata,
                )
            )

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
