"""Two-Stage Hybrid Retriever: Metadata Pre-filtering + Dense/Sparse + RRF + Rerank."""

import logging
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from config import RRF_K_CONSTANT, STAGE1_TOP_K, STAGE2_TOP_K
from rag.indexing.embedder import DenseEmbedder, SparseBM25Encoder
from rag.indexing.qdrant_store import DENSE_VECTOR_NAME, QDRANT_COLLECTION_NAME, SPARSE_VECTOR_NAME
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.schemas import ChunkMetadata, FinancialDocType, RankedCandidate

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Protocol):
    """Giao diện chuẩn cho tầng Reranker (Stage 2)."""

    def rerank(
        self,
        query: str,
        candidates: list[RankedCandidate],
        top_k: int,
    ) -> list[RankedCandidate]: ...


def build_metadata_filter(
    ticker: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    doc_type: FinancialDocType | None = None,
) -> Filter | None:
    """Tạo Qdrant Filter cho phép lọc chính xác theo mã CK và kỳ báo cáo."""
    conditions: list[FieldCondition] = []
    if ticker:
        conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker.upper())))
    if year is not None:
        conditions.append(FieldCondition(key="year", match=MatchValue(value=year)))
    if quarter is not None:
        conditions.append(FieldCondition(key="quarter", match=MatchValue(value=quarter)))
    if doc_type is not None:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type.value)))

    return Filter(must=conditions) if conditions else None


class HybridRetriever:
    """Bộ truy xuất 2 giai đoạn kết hợp Qdrant Hybrid Search (RRF) và Reranker."""

    def __init__(
        self,
        client: QdrantClient,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseBM25Encoder | None = None,
        reranker: CrossEncoderReranker | None = None,
        collection_name: str = QDRANT_COLLECTION_NAME,
    ) -> None:
        self.client = client
        self.dense_embedder = dense_embedder
        self.sparse_encoder = sparse_encoder or SparseBM25Encoder()
        self.reranker = reranker
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
        doc_type: FinancialDocType | None = None,
        stage1_top_k: int = STAGE1_TOP_K,
        final_top_k: int = STAGE2_TOP_K,
    ) -> list[RankedCandidate]:
        """Thực hiện truy xuất 2 tầng: Hybrid Search (RRF) -> Cross-Encoder Rerank."""
        if not query.strip():
            raise ValueError("Câu truy vấn không được để trống.")

        query_filter = build_metadata_filter(ticker=ticker, year=year, quarter=quarter, doc_type=doc_type)
        dense_hits = self._search_dense(query, query_filter, stage1_top_k)
        sparse_hits = self._search_sparse(query, query_filter, stage1_top_k)

        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=RRF_K_CONSTANT,
            deduplicate_by_parent=True,
        )
        logger.info(
            "Stage 1 Hybrid Search: dense=%d, sparse=%d, sau RRF=%d ứng viên.",
            len(dense_hits),
            len(sparse_hits),
            len(fused),
        )

        if self.reranker is not None and fused:
            return self.reranker.rerank(query, fused[:stage1_top_k], top_k=final_top_k)
        return fused[:final_top_k]

    def _search_dense(
        self,
        query: str,
        query_filter: Filter | None,
        limit: int,
    ) -> list[RankedCandidate]:
        query_vec = self.dense_embedder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return self._to_candidates(response.points)

    def _search_sparse(
        self,
        query: str,
        query_filter: Filter | None,
        limit: int,
    ) -> list[RankedCandidate]:
        sparse_vec = self.sparse_encoder.encode(query)
        if not sparse_vec.indices:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return self._to_candidates(response.points)

    @staticmethod
    def _to_candidates(points: list) -> list[RankedCandidate]:
        candidates: list[RankedCandidate] = []
        for pt in points:
            meta = ChunkMetadata.model_validate(pt.payload)
            candidates.append(
                RankedCandidate(
                    chunk_id=meta.chunk_id,
                    parent_id=meta.parent_id,
                    score=float(pt.score),
                    metadata=meta,
                )
            )
        return candidates
