"""Chuẩn bị dữ liệu ChatML, Khử trùng lặp & Chống Data Leakage cho Text-to-SQL."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

TEXT_TO_SQL_SYSTEM_PROMPT: str = (
    "Bạn là chuyên gia PostgreSQL tài chính Việt Nam. "
    "Chỉ sinh duy nhất 1 câu lệnh SELECT hợp lệ dựa trên Schema được cung cấp."
)


class TextToSQLSample(BaseModel):
    """Một mẫu huấn luyện Vietnamese Financial Text-to-SQL."""

    question: str = Field(..., min_length=1)
    db_schema: str = Field(..., min_length=1)
    sql_query: str = Field(..., min_length=1)


def to_chatml_record(sample: TextToSQLSample) -> dict[str, list[dict[str, str]]]:
    """Chuyển mẫu Text-to-SQL sang định dạng ChatML chuẩn cho Qwen-2.5."""
    user_prompt = f"DATABASE SCHEMA:\n{sample.db_schema}\n\nCÂU HỎI:\n{sample.question}"
    return {
        "messages": [
            {"role": "system", "content": TEXT_TO_SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": sample.sql_query.strip()},
        ]
    }


def deduplicate_and_prevent_leakage(
    train_samples: list[TextToSQLSample],
    test_questions: set[str],
) -> list[TextToSQLSample]:
    """Loại bỏ trùng lặp nội bộ và chặn tuyệt đối câu hỏi thuộc tập Test (Data Leakage)."""
    normalized_test = {q.strip().lower() for q in test_questions}
    seen: set[str] = set()
    clean: list[TextToSQLSample] = []

    for sample in train_samples:
        norm_q = sample.question.strip().lower()
        if norm_q in normalized_test or norm_q in seen:
            continue
        seen.add(norm_q)
        clean.append(sample)
    return clean


def export_jsonl(samples: list[TextToSQLSample], output_path: Path) -> int:
    """Xuất danh sách mẫu sang file JSONL chuẩn ChatML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(to_chatml_record(sample), ensure_ascii=False) + "\n")
    return len(samples)
