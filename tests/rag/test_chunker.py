"""Unit tests cho TableAwareParentChildChunker."""

import pytest

from rag.ingestion.chunker import DocumentContext, TableAwareParentChildChunker
from rag.schemas import DocumentSection, FinancialDocType


def _sample_context() -> DocumentContext:
    return DocumentContext(
        ticker="fpt",
        year=2025,
        quarter=4,
        doc_type=FinancialDocType.BCTC_HOP_NHAT,
    )


def test_small_table_kept_whole_with_context_header() -> None:
    """Bảng nhỏ hơn ngưỡng phải giữ nguyên vẹn và có đầy đủ Context Header."""
    table_md = (
        "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
        "| :--- | :---: | :---: |\n"
        "| Doanh thu thuần | 52.618 | 62.849 |\n"
        "| Lợi nhuận gộp | 20.513 | 24.837 |"
    )
    section = DocumentSection(
        section_title="Kết quả kinh doanh hợp nhất",
        content=table_md,
        page_number=5,
        is_table=True,
        currency_unit="Tỷ VNĐ",
    )
    chunker = TableAwareParentChildChunker()
    chunks = chunker.chunk_sections([section], _sample_context())

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.ticker == "FPT"
    assert chunk.is_table is True
    assert "[Mã CK: FPT | Loại BC: BCTC_HOP_NHAT | Kỳ: Q4/2025 |" in chunk.text
    assert "|" in chunk.text
    assert "62.849" in chunk.text
    assert chunk.text == chunk.parent_text


def test_long_table_duplicates_markdown_header_in_every_child() -> None:
    """Bảng dài bị tách nhiều chunk con phải nhân bản dòng tiêu đề cột ở mỗi chunk."""
    header = "| Chỉ tiêu | Mã số | Năm 2024 | Năm 2025 |\n| :--- | :---: | :---: | :---: |"
    rows = [f"| Khoản mục tài sản số {i} | {100 + i} | {1000 + i} | {2000 + i} |" for i in range(15)]
    long_table = "\n".join([header, *rows])

    section = DocumentSection(
        section_title="Bảng cân đối kế toán",
        content=long_table,
        page_number=3,
        is_table=True,
        currency_unit="Triệu VNĐ",
    )
    chunker = TableAwareParentChildChunker(
        child_max_tokens=80,
        parent_max_tokens=300,
        table_keep_whole_max_tokens=50,
    )
    chunks = chunker.chunk_sections([section], _sample_context())

    assert len(chunks) > 1
    parent_ids = {c.parent_id for c in chunks}
    assert len(parent_ids) == 1
    for c in chunks:
        assert "| Chỉ tiêu | Mã số | Năm 2024 | Năm 2025 |" in c.text
        assert "| :--- | :---: | :---: | :---: |" in c.text
        assert "Khoản mục tài sản số 14" in c.parent_text


def test_empty_sections_raises_value_error() -> None:
    """Đầu vào rỗng phải báo lỗi sớm (Fail Fast)."""
    chunker = TableAwareParentChildChunker()
    with pytest.raises(ValueError, match="không được rỗng"):
        chunker.chunk_sections([], _sample_context())
