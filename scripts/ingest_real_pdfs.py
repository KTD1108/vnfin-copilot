"""Script nạp toàn bộ file PDF thật trong data/raw/ vào Qdrant bằng Docling."""

import argparse
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from rag.indexing.embedder import DeterministicHashDenseEmbedder
from rag.indexing.qdrant_store import QdrantHybridStore
from rag.ingestion.chunker import DocumentContext, TableAwareParentChildChunker
from rag.ingestion.pdf_parser import DoclingPDFParser
from rag.schemas import FinancialDocType

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nạp BCTC PDF thật vào Qdrant.")
    parser.add_argument("--pdf", type=Path, required=True, help="Đường dẫn tới file PDF.")
    parser.add_argument("--ticker", type=str, required=True, help="Mã CK (VD: FOX, FPT).")
    parser.add_argument("--year", type=int, required=True, help="Năm báo cáo.")
    parser.add_argument("--quarter", type=int, default=None, help="Quý (1..4) hoặc bỏ trống.")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    parser = DoclingPDFParser()
    sections = parser.parse(args.pdf)

    ctx = DocumentContext(
        ticker=args.ticker,
        year=args.year,
        quarter=args.quarter,
        doc_type=FinancialDocType.BCTC_HOP_NHAT,
    )
    chunks = TableAwareParentChildChunker().chunk_sections(sections, ctx)

    client = QdrantClient(url=args.qdrant_url)
    store = QdrantHybridStore(client=client, dense_embedder=DeterministicHashDenseEmbedder())
    total = store.upsert_chunks(chunks)
    print(f"Hoàn tất nạp {total} chunks từ {args.pdf.name} vào Qdrant ({args.qdrant_url}).")


if __name__ == "__main__":
    main()
