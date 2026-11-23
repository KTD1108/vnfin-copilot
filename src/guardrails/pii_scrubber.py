"""Vietnamese PII Scrubber: Ẩn danh hóa SĐT, CCCD, Email và Secret Tokens."""

import re

_VN_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{8}(?!\d)")
_VN_CCCD_PATTERN = re.compile(r"(?<!\d)\d{12}(?!\d)")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_TOKEN_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{16,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*)\b")


def mask_vietnamese_pii(text: str) -> str:
    """Thay thế toàn bộ thông tin định danh cá nhân & bí mật bằng thẻ ẩn danh."""
    masked = _TOKEN_PATTERN.sub("<SECRET_TOKEN>", text)
    masked = _EMAIL_PATTERN.sub("<EMAIL>", masked)
    masked = _VN_PHONE_PATTERN.sub("<VN_PHONE>", masked)
    masked = _VN_CCCD_PATTERN.sub("<VN_CCCD>", masked)
    return masked
