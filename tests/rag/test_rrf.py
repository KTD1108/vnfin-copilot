"""Unit tests cho thuật toán Reciprocal Rank Fusion (RRF) và Parent Deduplication."""

import pytest

from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.schemas import ChunkMetadata, FinancialDocType, RankedCandidate


def _make_candidate(chunk_id: str, parent_id: str, score: float = 0.9) -> RankedCandidate:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        parent_id=parent_id,
        ticker="VCB",
        year=2025,
        quarter=None,
        doc_type=FinancialDocType.BAO_CAO_THUONG_NIEN,
        is_table=False,
        page_number=10,
        section_title="Phân tích rủi ro",
        text=f"Nội dung {chunk_id}",
        parent_text=f"Nội dung cha {parent_id}",
    )
    return RankedCandidate(chunk_id=chunk_id, parent_id=parent_id, score=score, metadata=meta)


def test_rrf_boosts_document_appearing_in_both_lists() -> None:
    """Chunk xuất hiện ở cả Dense và Sparse phải đạt điểm RRF cao nhất."""
    dense = [_make_candidate("c1", "p1"), _make_candidate("c2", "p2")]
    sparse = [_make_candidate("c2", "p2"), _make_candidate("c3", "p3")]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, deduplicate_by_parent=True)

    assert [item.chunk_id for item in fused] == ["c2", "c1", "c3"]
    expected_c2 = (1.0 / 62.0) + (1.0 / 61.0)
    assert fused[0].score == pytest.approx(expected_c2, rel=1e-6)


def test_rrf_deduplicates_children_sharing_same_parent() -> None:
    """Nhiều chunk con thuộc cùng 1 parent_id chỉ được giữ lại 1 đại diện điểm cao nhất."""
    dense = [
        _make_candidate("c1_a", "parent_table_1"),
        _make_candidate("c1_b", "parent_table_1"),
        _make_candidate("c2", "parent_text_2"),
    ]
    fused = reciprocal_rank_fusion([dense], k=60, deduplicate_by_parent=True)

    assert len(fused) == 2
    assert [item.parent_id for item in fused] == ["parent_table_1", "parent_text_2"]
    assert fused[0].chunk_id == "c1_a"
