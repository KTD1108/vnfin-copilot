"""Layout-Aware PDF Parser chuyển Báo cáo Tài chính PDF sang DocumentSection (Markdown)."""

import logging
import re
from pathlib import Path

from rag.schemas import DocumentSection

logger = logging.getLogger(__name__)

_CURRENCY_PATTERN = re.compile(
    r"(triệu\s*(?:vnđ|đồng)|tỷ\s*(?:vnđ|đồng)|ngàn\s*(?:vnđ|đồng)|vnd)",
    re.IGNORECASE,
)


def detect_currency_unit(text: str, default: str = "Không xác định") -> str:
    """Nhận diện đơn vị tiền tệ (Triệu VNĐ, Tỷ VNĐ...) từ tiêu đề hoặc đầu bảng."""
    match = _CURRENCY_PATTERN.search(text)
    if not match:
        return default
    raw = re.sub(r"\s+", " ", match.group(1)).strip()
    return raw.title().replace("Vnđ", "VNĐ").replace("Vnd", "VNĐ")


class DoclingPDFParser:
    """Bóc tách PDF phức tạp giữ nguyên cấu trúc bảng biểu Markdown bằng Docling."""

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Chưa cài đặt 'docling'. Chạy: pip install docling"
            ) from exc
        self._converter = DocumentConverter()

    def parse(self, pdf_path: Path) -> list[DocumentSection]:
        """Đọc file PDF và trả về danh sách DocumentSection (văn bản & bảng Markdown)."""
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

        logger.info("Đang bóc tách PDF bằng Docling: %s", pdf_path.name)
        result = self._converter.convert(str(pdf_path))
        doc = result.document

        sections: list[DocumentSection] = []
        current_title = "Nội dung chung"
        current_currency = "Không xác định"

        for item, _level in doc.iterate_items():
            page_no = item.prov[0].page_no if getattr(item, "prov", None) else 1
            label = getattr(getattr(item, "label", None), "value", str(getattr(item, "label", "")))

            if "heading" in label.lower() or "title" in label.lower():
                heading_text = getattr(item, "text", "").strip()
                if heading_text:
                    current_title = heading_text
                    current_currency = detect_currency_unit(heading_text, current_currency)
                continue

            if "table" in label.lower() and hasattr(item, "export_to_markdown"):
                table_md = item.export_to_markdown(doc=doc).strip()
                if table_md:
                    unit = detect_currency_unit(table_md, current_currency)
                    sections.append(
                        DocumentSection(
                            section_title=current_title,
                            content=table_md,
                            page_number=max(1, int(page_no)),
                            is_table=True,
                            currency_unit=unit,
                        )
                    )
                continue

            text_val = getattr(item, "text", "").strip()
            if text_val:
                current_currency = detect_currency_unit(text_val, current_currency)
                sections.append(
                    DocumentSection(
                        section_title=current_title,
                        content=text_val,
                        page_number=max(1, int(page_no)),
                        is_table=False,
                        currency_unit=current_currency,
                    )
                )

        logger.info("Đã bóc tách %d sections từ %s.", len(sections), pdf_path.name)
        return sections
