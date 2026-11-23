"""Pydantic schemas cho dữ liệu tài liệu và kết quả truy xuất trong Giai đoạn 1."""

from enum import Enum
from pydantic import BaseModel, Field


class FinancialDocType(str, Enum):
    """Loại tài liệu tài chính chuẩn hóa."""

    BCTC_HOP_NHAT = "BCTC_HOP_NHAT"
    BAO_CAO_THUONG_NIEN = "BAO_CAO_THUONG_NIEN"
    NGHI_QUYET_DHDCD = "NGHI_QUYET_DHDCD"


class DocumentSection(BaseModel):
    """Một khối nội dung thô được bóc tách từ PDF (văn bản hoặc bảng Markdown)."""

    section_title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)
    is_table: bool = False
    currency_unit: str = Field(default="Không xác định")


class ChunkMetadata(BaseModel):
    """Metadata bắt buộc lưu trên Qdrant Payload cho mỗi point."""

    chunk_id: str = Field(..., min_length=1)
    parent_id: str = Field(..., min_length=1)
    ticker: str = Field(..., min_length=2, max_length=10)
    year: int = Field(..., ge=2000, le=2100)
    quarter: int | None = Field(default=None, ge=1, le=4)
    doc_type: FinancialDocType
    is_table: bool
    page_number: int = Field(..., ge=1)
    section_title: str = Field(..., min_length=1)
    currency_unit: str = Field(default="Không xác định")
    text: str = Field(..., min_length=1)
    parent_text: str = Field(..., min_length=1)

    @property
    def period_label(self) -> str:
        """Chuỗi hiển thị kỳ báo cáo, ví dụ: 'Q4/2025' hoặc 'Năm 2025'."""
        if self.quarter is not None:
            return f"Q{self.quarter}/{self.year}"
        return f"Năm {self.year}"


class RankedCandidate(BaseModel):
    """Ứng viên sau bước tìm kiếm đơn nhánh hoặc hợp nhất RRF."""

    chunk_id: str
    parent_id: str
    score: float
    metadata: ChunkMetadata
