"""Table-Aware & Parent-Child Chunker cho Báo cáo Tài chính Việt Nam."""

import hashlib
import logging
from dataclasses import dataclass

from config import (
    CHARS_PER_TOKEN_ESTIMATE,
    CHILD_CHUNK_MAX_TOKENS,
    PARENT_CHUNK_MAX_TOKENS,
    TABLE_KEEP_WHOLE_MAX_TOKENS,
)
from rag.schemas import ChunkMetadata, DocumentSection, FinancialDocType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentContext:
    """Thông tin ngữ cảnh cấp tài liệu dùng để làm giàu từng chunk."""

    ticker: str
    year: int
    quarter: int | None
    doc_type: FinancialDocType

    @property
    def period_label(self) -> str:
        """Nhãn kỳ báo cáo chuẩn hóa."""
        return f"Q{self.quarter}/{self.year}" if self.quarter is not None else f"Năm {self.year}"


def estimate_tokens(text: str) -> int:
    """Ước lượng nhanh số lượng token từ độ dài ký tự."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def build_context_header(ctx: DocumentContext, section: DocumentSection) -> str:
    """Tạo dòng tiền tố ngữ cảnh chuẩn cho mọi chunk."""
    return (
        f"[Mã CK: {ctx.ticker.upper()} | Loại BC: {ctx.doc_type.value} | "
        f"Kỳ: {ctx.period_label} | Mục: {section.section_title} | "
        f"Đơn vị: {section.currency_unit}]"
    )


def _deterministic_id(seed: str) -> str:
    """Sinh ID ổn định từ nội dung để tránh trùng lặp khi re-index."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _split_text_by_chars(text: str, max_chars: int) -> list[str]:
    """Tách văn bản theo ranh giới đoạn/câu không vượt quá max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            for start in range(0, len(para), max_chars):
                chunks.append(para[start : start + max_chars])
            current = ""

    if current:
        chunks.append(current)
    return chunks or [text]


def _split_markdown_table(table_md: str, max_chars: int) -> list[str]:
    """Tách bảng Markdown dài nhưng luôn nhân bản Header + Separator vào đầu mỗi chunk."""
    lines = [line.rstrip() for line in table_md.splitlines() if line.strip()]
    if len(lines) <= 2:
        return [table_md]

    header_block = "\n".join(lines[:2])
    data_rows = lines[2:]
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in data_rows:
        candidate = "\n".join([header_block, *current_rows, row])
        if len(candidate) <= max_chars or not current_rows:
            current_rows.append(row)
        else:
            chunks.append("\n".join([header_block, *current_rows]))
            current_rows = [row]

    if current_rows:
        chunks.append("\n".join([header_block, *current_rows]))
    return chunks


class TableAwareParentChildChunker:
    """Chia tài liệu thành Parent-Child Chunks, bảo toàn cấu trúc bảng biểu."""

    def __init__(
        self,
        child_max_tokens: int = CHILD_CHUNK_MAX_TOKENS,
        parent_max_tokens: int = PARENT_CHUNK_MAX_TOKENS,
        table_keep_whole_max_tokens: int = TABLE_KEEP_WHOLE_MAX_TOKENS,
    ) -> None:
        if child_max_tokens <= 0 or parent_max_tokens < child_max_tokens:
            raise ValueError("Tham số token chunk không hợp lệ.")
        self.child_max_chars = child_max_tokens * CHARS_PER_TOKEN_ESTIMATE
        self.parent_max_chars = parent_max_tokens * CHARS_PER_TOKEN_ESTIMATE
        self.table_keep_whole_tokens = table_keep_whole_max_tokens

    def chunk_sections(
        self,
        sections: list[DocumentSection],
        ctx: DocumentContext,
    ) -> list[ChunkMetadata]:
        """Chuyển danh sách DocumentSection thành danh sách ChunkMetadata sẵn sàng index."""
        if not sections:
            raise ValueError("Danh sách sections đầu vào không được rỗng.")

        all_chunks: list[ChunkMetadata] = []
        for idx, section in enumerate(sections):
            all_chunks.extend(self._chunk_single_section(section, ctx, idx))

        logger.info(
            "Đã chunk %d sections của %s (%s) thành %d child chunks.",
            len(sections),
            ctx.ticker,
            ctx.period_label,
            len(all_chunks),
        )
        return all_chunks

    def _chunk_single_section(
        self,
        section: DocumentSection,
        ctx: DocumentContext,
        section_idx: int,
    ) -> list[ChunkMetadata]:
        """Xử lý tách một section thành các cặp Parent-Child."""
        header = build_context_header(ctx, section)
        if section.is_table:
            return self._chunk_table_section(section, ctx, header, section_idx)
        return self._chunk_text_section(section, ctx, header, section_idx)

    def _chunk_table_section(
        self,
        section: DocumentSection,
        ctx: DocumentContext,
        header: str,
        section_idx: int,
    ) -> list[ChunkMetadata]:
        """Giữ nguyên bảng nhỏ hoặc tách bảng dài có nhân bản Markdown Header."""
        total_tokens = estimate_tokens(section.content)
        parent_text = f"{header}\n{section.content}"
        parent_id = _deterministic_id(f"{ctx.ticker}:{ctx.period_label}:t:{section_idx}:{parent_text}")

        if total_tokens <= self.table_keep_whole_tokens:
            child_pieces = [section.content]
        else:
            child_pieces = _split_markdown_table(section.content, self.child_max_chars)

        return self._build_metadata_list(child_pieces, parent_text, parent_id, header, section, ctx)

    def _chunk_text_section(
        self,
        section: DocumentSection,
        ctx: DocumentContext,
        header: str,
        section_idx: int,
    ) -> list[ChunkMetadata]:
        """Tách văn bản thường thành Parent chunks rồi tách tiếp thành Child chunks."""
        parent_pieces = _split_text_by_chars(section.content, self.parent_max_chars)
        results: list[ChunkMetadata] = []

        for p_idx, p_piece in enumerate(parent_pieces):
            parent_text = f"{header}\n{p_piece}"
            parent_id = _deterministic_id(
                f"{ctx.ticker}:{ctx.period_label}:p:{section_idx}:{p_idx}:{p_piece}"
            )
            child_pieces = _split_text_by_chars(p_piece, self.child_max_chars)
            results.extend(
                self._build_metadata_list(child_pieces, parent_text, parent_id, header, section, ctx)
            )
        return results

    def _build_metadata_list(
        self,
        child_pieces: list[str],
        parent_text: str,
        parent_id: str,
        header: str,
        section: DocumentSection,
        ctx: DocumentContext,
    ) -> list[ChunkMetadata]:
        """Đóng gói từng child chunk kèm Context Header và Metadata chuẩn."""
        items: list[ChunkMetadata] = []
        for c_idx, piece in enumerate(child_pieces):
            enriched_text = f"{header}\n{piece}"
            chunk_id = _deterministic_id(f"{parent_id}:c:{c_idx}:{piece}")
            items.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    ticker=ctx.ticker.upper(),
                    year=ctx.year,
                    quarter=ctx.quarter,
                    doc_type=ctx.doc_type,
                    is_table=section.is_table,
                    page_number=section.page_number,
                    section_title=section.section_title,
                    currency_unit=section.currency_unit,
                    text=enriched_text,
                    parent_text=parent_text,
                )
            )
        return items
