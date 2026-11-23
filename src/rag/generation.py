"""Stage 4: Financial RAG Prompt Builder & Citation-Enforced Generator."""

import logging
import os

import httpx

from rag.schemas import RankedCandidate

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_REQUEST_TIMEOUT_SEC: float = 60.0

SYSTEM_PROMPT = """Bạn là Chuyên viên Phân tích Tài chính AI của hệ thống VnFinCopilot.
Quy tắc BẮT BUỘC:
1. CHỈ trả lời dựa trên các đoạn tài liệu (Context) được cung cấp bên dưới. Tuyệt đối KHÔNG tự bịa đặt số liệu.
2. Nếu tài liệu không chứa thông tin để trả lời, hãy nói rõ: "Tài liệu hiện có không chứa thông tin về vấn đề này."
3. Ngay sau mỗi con số tài chính, BẮT BUỘC ghi rõ trích dẫn theo đúng định dạng:
   [Nguồn: {doc_type} {ticker} {period}, Trang {page_number}]
4. Giữ nguyên đơn vị tính (Tỷ VNĐ, Triệu VNĐ...) như trong tài liệu gốc."""


def format_context_with_citations(candidates: list[RankedCandidate]) -> str:
    """Định dạng Top-K Parent Chunks kèm nhãn nguồn rõ ràng để đưa vào Prompt."""
    if not candidates:
        return "Không có tài liệu liên quan."

    blocks: list[str] = []
    for idx, item in enumerate(candidates, start=1):
        meta = item.metadata
        citation_tag = f"{meta.doc_type.value} {meta.ticker} {meta.period_label}, Trang {meta.page_number}"
        blocks.append(f"--- TÀI LIỆU #{idx} [{citation_tag}] ---\n{meta.parent_text}")
    return "\n\n".join(blocks)


def build_rag_messages(query: str, candidates: list[RankedCandidate]) -> list[dict[str, str]]:
    """Tạo danh sách messages chuẩn OpenAI Chat Completions."""
    context_str = format_context_with_citations(candidates)
    user_content = f"NGỮ CẢNH TÀI LIỆU:\n{context_str}\n\nCÂU HỎI: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def generate_grounded_answer(
    query: str,
    candidates: list[RankedCandidate],
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
) -> str:
    """Gọi LLM (temperature=0.0) để sinh câu trả lời bám sát ngữ cảnh kèm trích dẫn."""
    if not query.strip():
        raise ValueError("Câu hỏi không được để trống.")

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("Thiếu OPENAI_API_KEY để gọi LLM.")

    messages = build_rag_messages(query, candidates)
    headers = {"Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
    }

    async with httpx.AsyncClient(timeout=DEFAULT_REQUEST_TIMEOUT_SEC) as client:
        response = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    logger.info("Đã sinh câu trả lời RAG với model=%s, temperature=%.1f", model, DEFAULT_TEMPERATURE)
    return str(data["choices"][0]["message"]["content"])
