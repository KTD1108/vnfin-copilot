"""Input Prompt Injection Defense & Output Financial Compliance Guardrails."""

import re
from dataclasses import dataclass

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"bỏ qua (?:mọi|tất cả) hướng dẫn", re.IGNORECASE),
    re.compile(r"ignore (?:all )?previous instructions", re.IGNORECASE),
    re.compile(r"(?:in ra|reveal|show) system prompt", re.IGNORECASE),
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
)

_COMPLIANCE_VIOLATION_PATTERN = re.compile(
    r"(?:cam kết|chắc chắn)\s+(?:lãi|tăng)\s+100%|hãy vay tiền mua ngay",
    re.IGNORECASE,
)

FINANCIAL_DISCLAIMER: str = (
    "\n\n*Lưu ý: Thông tin trên được tổng hợp tự động từ tài liệu công bố và chỉ mang tính tham khảo.*"
)


@dataclass(frozen=True)
class GuardrailResult:
    """Kết quả kiểm tra an toàn đầu vào/đầu ra."""

    allowed: bool
    reason: str | None = None
    sanitized_text: str = ""


def check_input_safety(sanitized_query: str) -> GuardrailResult:
    """Phát hiện và chặn Prompt Injection / Jailbreak ở cửa ngõ đầu vào."""
    if not sanitized_query.strip():
        return GuardrailResult(allowed=False, reason="Câu hỏi không được để trống.")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized_query):
            return GuardrailResult(
                allowed=False,
                reason="Phát hiện mẫu câu điều khiển hệ thống không hợp lệ (Prompt Injection).",
            )
    return GuardrailResult(allowed=True, sanitized_text=sanitized_query)


def apply_output_compliance(answer: str) -> str:
    """Kiểm tra tuân thủ quy định tư vấn tài chính và đính kèm miễn trừ trách nhiệm."""
    if _COMPLIANCE_VIOLATION_PATTERN.search(answer):
        return (
            "Hệ thống chỉ cung cấp số liệu phân tích khách quan từ báo cáo tài chính, "
            "không đưa ra cam kết lợi nhuận hay khuyến nghị mua/bán cụ thể."
            + FINANCIAL_DISCLAIMER
        )
    if FINANCIAL_DISCLAIMER.strip() not in answer:
        return answer + FINANCIAL_DISCLAIMER
    return answer
