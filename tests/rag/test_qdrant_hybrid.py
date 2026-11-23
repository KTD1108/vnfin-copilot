"""Integration tests cho QdrantHybridStore & HybridRetriever (chạy trên Qdrant In-Memory)."""

from qdrant_client import QdrantClient

from rag.indexing.embedder import DeterministicHashDenseEmbedder
from rag.indexing.qdrant_store import QdrantHybridStore
from rag.ingestion.chunker import DocumentContext, TableAwareParentChildChunker
from rag.retrieval.retriever import HybridRetriever
from rag.schemas import DocumentSection, FinancialDocType


def _seed_sample_financial_docs(store: QdrantHybridStore) -> None:
    chunker = TableAwareParentChildChunker()

    fpt_sections = [
        DocumentSection(
            section_title="Kết quả kinh doanh hợp nhất FPT",
            content=(
                "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
                "| :--- | :---: | :---: |\n"
                "| Doanh thu thuần | 52.618 | 62.849 |\n"
                "| Lợi nhuận sau thuế | 7.792 | 9.420 |"
            ),
            page_number=6,
            is_table=True,
            currency_unit="Tỷ VNĐ",
        ),
        DocumentSection(
            section_title="Thuyết minh khối Công nghệ",
            content="Doanh thu chuyển đổi số thị trường Nhật Bản của FPT tăng trưởng mạnh 32%.",
            page_number=24,
            is_table=False,
            currency_unit="Tỷ VNĐ",
        ),
    ]
    vcb_sections = [
        DocumentSection(
            section_title="Kết quả kinh doanh hợp nhất Vietcombank",
            content=(
                "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
                "| :--- | :---: | :---: |\n"
                "| Thu nhập lãi thuần | 55.400 | 61.200 |\n"
                "| Lợi nhuận trước thuế | 41.200 | 45.800 |"
            ),
            page_number=4,
            is_table=True,
            currency_unit="Tỷ VNĐ",
        ),
    ]

    fpt_chunks = chunker.chunk_sections(
        fpt_sections,
        DocumentContext(ticker="FPT", year=2025, quarter=None, doc_type=FinancialDocType.BCTC_HOP_NHAT),
    )
    vcb_chunks = chunker.chunk_sections(
        vcb_sections,
        DocumentContext(ticker="VCB", year=2025, quarter=None, doc_type=FinancialDocType.BCTC_HOP_NHAT),
    )
    store.upsert_chunks(fpt_chunks + vcb_chunks)


def test_hybrid_retriever_filters_by_ticker_and_finds_relevant_table() -> None:
    """Truy vấn lọc theo mã FPT không được lẫn tài liệu VCB và phải tìm đúng bảng doanh thu."""
    client = QdrantClient(":memory:")
    embedder = DeterministicHashDenseEmbedder()
    store = QdrantHybridStore(client=client, dense_embedder=embedder)
    _seed_sample_financial_docs(store)

    retriever = HybridRetriever(client=client, dense_embedder=embedder)
    results = retriever.retrieve(
        query="Doanh thu thuần hợp nhất năm 2025 của FPT là bao nhiêu?",
        ticker="FPT",
        year=2025,
    )

    assert len(results) >= 1
    assert all(item.metadata.ticker == "FPT" for item in results)
    assert "62.849" in results[0].metadata.parent_text
