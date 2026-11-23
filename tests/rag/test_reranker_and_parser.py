"""Unit tests cho Currency Detector và Stage 2 Reranker."""

from rag.ingestion.pdf_parser import detect_currency_unit
from rag.retrieval.reranker import LexicalOverlapReranker
from rag.schemas import ChunkMetadata, FinancialDocType, RankedCandidate


def _make_candidate(chunk_id: str, text: str) -> RankedCandidate:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        parent_id=f"parent_{chunk_id}",
        ticker="FOX",
        year=2024,
        quarter=None,
        doc_type=FinancialDocType.BCTC_HOP_NHAT,
        is_table=False,
        page_number=12,
        section_title="Thuyết minh",
        text=text,
        parent_text=text,
    )
    return RankedCandidate(chunk_id=chunk_id, parent_id=meta.parent_id, score=0.1, metadata=meta)


def test_detect_currency_unit_extracts_vietnamese_units() -> None:
    """Nhận diện đúng đơn vị tiền tệ trong báo cáo tài chính tiếng Việt."""
    assert detect_currency_unit("Đơn vị tính: Triệu VNĐ") == "Triệu VNĐ"
    assert detect_currency_unit("Bảng cân đối kế toán (Đơn vị: tỷ đồng)") == "Tỷ Đồng"
    assert detect_currency_unit("Không có đơn vị") == "Không xác định"


def test_lexical_reranker_promotes_exact_matching_candidate() -> None:
    """Reranker Stage 2 phải đẩy đoạn văn khớp chính xác câu hỏi lên Top 1."""
    c1 = _make_candidate("c1", "Chi phí quản lý doanh nghiệp năm 2024.")
    c2 = _make_candidate("c2", "Doanh thu thuần hợp nhất năm 2024 đạt 17.610 tỷ đồng.")
    reranker = LexicalOverlapReranker()

    top = reranker.rerank(
        query="Doanh thu thuần hợp nhất năm 2024",
        candidates=[c1, c2],
        top_k=1,
    )

    assert len(top) == 1
    assert top[0].chunk_id == "c2"
    assert top[0].score > 0.5
