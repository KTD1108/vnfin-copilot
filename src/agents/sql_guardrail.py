"""SQL Security Guardrail: Chốt chặn Read-Only và giới hạn tài nguyên cho Text-to-SQL."""

import re

from agents.state import DEFAULT_SQL_LIMIT

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE|CREATE|REPLACE)\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)


class SQLSecurityError(ValueError):
    """Ngoại lệ khi câu lệnh SQL vi phạm chính sách Read-Only."""


def validate_readonly_sql(sql: str, default_limit: int = DEFAULT_SQL_LIMIT) -> str:
    """Kiểm tra an toàn câu lệnh SQL và tự động gắn LIMIT nếu chưa có."""
    cleaned = sql.strip()
    if not cleaned:
        raise SQLSecurityError("Câu lệnh SQL không được để trống.")

    # Cấm nhiều câu lệnh ghép bằng dấu chấm phẩy
    stripped_trailing = cleaned.rstrip(";").strip()
    if ";" in stripped_trailing:
        raise SQLSecurityError("Không cho phép thực thi nhiều câu lệnh SQL cùng lúc.")

    if _FORBIDDEN_SQL_PATTERN.search(stripped_trailing):
        raise SQLSecurityError("Câu lệnh chứa từ khóa thay đổi dữ liệu bị cấm (Read-Only).")

    upper_sql = stripped_trailing.upper()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        raise SQLSecurityError("Chỉ cho phép câu lệnh truy vấn SELECT hoặc WITH ... SELECT.")

    if not _LIMIT_PATTERN.search(stripped_trailing):
        return f"{stripped_trailing} LIMIT {default_limit}"

    return stripped_trailing
