"""Text-to-SQL Agent kèm vòng lặp tự sửa lỗi (Self-Correction Loop)."""

import logging
from typing import Any, Protocol

from agents.sql_guardrail import SQLSecurityError, validate_readonly_sql
from agents.state import AgentState, MAX_SQL_RETRIES

logger = logging.getLogger(__name__)


class SQLBackend(Protocol):
    """Giao diện thực thi SQL Read-Only (PostgreSQL / SQLite / Mock)."""

    async def execute_readonly(self, sql: str) -> list[dict[str, Any]]: ...


class SQLGeneratorFn(Protocol):
    """Giao diện hàm sinh/sửa SQL từ câu hỏi và thông báo lỗi (nếu có)."""

    async def __call__(self, query: str, previous_sql: str | None, error: str | None) -> str: ...


class SQLExecutionError(RuntimeError):
    """Lỗi thực thi từ cơ sở dữ liệu (sai tên cột, sai cú pháp...)."""


async def run_sql_with_self_correction(
    state: AgentState,
    generator: SQLGeneratorFn,
    backend: SQLBackend,
    max_retries: int = MAX_SQL_RETRIES,
) -> AgentState:
    """Thực thi Text-to-SQL với vòng lặp tự sửa lỗi tối đa max_retries lần."""
    current_sql: str | None = state.sql_query
    last_error: str | None = state.sql_error

    for attempt in range(max_retries):
        raw_sql = await generator(state.user_query, current_sql, last_error)
        try:
            safe_sql = validate_readonly_sql(raw_sql)
            rows = await backend.execute_readonly(safe_sql)
            state.sql_query = safe_sql
            state.sql_result = rows
            state.sql_error = None
            state.sql_retry_count = attempt
            return state
        except (SQLSecurityError, SQLExecutionError) as exc:
            last_error = str(exc)
            current_sql = raw_sql
            state.sql_query = raw_sql
            state.sql_error = last_error
            state.sql_retry_count = attempt + 1
            logger.warning("SQL lần thử %d/%d thất bại: %s", attempt + 1, max_retries, last_error)

    return state
