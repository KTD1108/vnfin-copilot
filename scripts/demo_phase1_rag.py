"""Script chạy thử (Demo) luồng Giai đoạn 1: Chunking -> Qdrant Hybrid Index -> RRF Retrieval."""

import argparse
import logging
import sys

from qdrant_client import QdrantClient

from rag.generation import build_rag_messages
from rag.indexing.embedder import DeterministicHashDenseEmbedder
from rag.indexing.qdrant_store import QdrantHybridStore
from rag.ingestion.chunker import DocumentContext, TableAwareParentChildChunker
from rag.retrieval.reranker import LexicalOverlapReranker
from rag.retrieval.retriever import HybridRetriever
from rag.schemas import DocumentSection, FinancialDocType

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh cho phép chọn Qdrant In-Memory hoặc Qdrant Server."""
    parser = argparse.ArgumentParser(description="Demo Giai đoạn 1 Hybrid RAG.")
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="URL của Qdrant Server (VD: http://localhost:6333). Bỏ trống để dùng :memory:.",
    )
    return parser.parse_args()


def main() -> None:
    """Nạp dữ liệu mẫu BCTC FPT & VCB vào Qdrant và truy vấn thử."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    client = QdrantClient(url=args.qdrant_url) if args.qdrant_url else QdrantClient(":memory:")
    embedder = DeterministicHashDenseEmbedder()
    store = QdrantHybridStore(client=client, dense_embedder=embedder)
    chunker = TableAwareParentChildChunker()

    sections_fpt = [
        DocumentSection(
            section_title="Báo cáo Kết quả Kinh doanh Hợp nhất",
            content=(
                "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
                "| :--- | :---: | :---: |\n"
                "| Doanh thu thuần | 52.618 | 62.849 |\n"
                "| Lợi nhuận gộp | 20.513 | 24.837 |\n"
                "| Lợi nhuận sau thuế | 7.792 | 9.420 |"
            ),
            page_number=6,
            is_table=True,
            currency_unit="Tỷ VNĐ",
        ),
        DocumentSection(
            section_title="Giải trình tăng trưởng khối Công nghệ",
            content=(
                "Doanh thu dịch vụ CNTT nước ngoài đạt mức tăng trưởng cao nhất nhờ nhu cầu "
                "chuyển đổi số và AI tại thị trường Nhật Bản tăng 32% so với cùng kỳ."
            ),
            page_number=24,
            is_table=False,
            currency_unit="Tỷ VNĐ",
        ),
    ]

    chunks = chunker.chunk_sections(
        sections_fpt,
        DocumentContext(ticker="FPT", year=2025, quarter=None, doc_type=FinancialDocType.BCTC_HOP_NHAT),
    )
    store.upsert_chunks(chunks)

    retriever = HybridRetriever(
        client=client,
        dense_embedder=embedder,
        reranker=LexicalOverlapReranker(),
    )
    queries = [
        "Doanh thu thuần và lợi nhuận sau thuế năm 2025 của FPT là bao nhiêu?",
        "Thị trường Nhật Bản tăng trưởng bao nhiêu phần trăm?",
    ]

    for query in queries:
        print(f"\n=== CÂU HỎI: {query} ===")
        results = retriever.retrieve(query=query, ticker="FPT", year=2025, final_top_k=2)
        for idx, item in enumerate(results, start=1):
            print(f"\n[Top {idx}] Rerank Score: {item.score:.5f} | Trang {item.metadata.page_number}")
            print(item.metadata.parent_text)
        messages = build_rag_messages(query, results[:1])
        print(f"\n[Stage 4 Prompt sẵn sàng gửi LLM ({len(messages[1]['content'])} ký tự)]")


if __name__ == "__main__":
    main()
