"""Integration tests cho ReadOnlySQLiteBackend và Text-to-SQL thực thi thật."""

import asyncio
from pathlib import Path

import pytest

from agents.sqlite_backend import ReadOnlySQLiteBackend, seed_sample_financial_db
from agents.sql_agent import SQLExecutionError, run_sql_with_self_correction
from agents.state import AgentState


def test_sqlite_backend_readonly_and_self_correction(tmp_path: Path) -> None:
    """Thực thi SQL thật trên SQLite Read-Only và tự sửa lỗi sai tên cột."""
    db_path = tmp_path / "vnfin_test.db"
    seed_sample_financial_db(db_path, Path("src/agents/schema.sql"))
    backend = ReadOnlySQLiteBackend(db_path)

    # Kiểm tra Read-Only chặn ghi dữ liệu ở tầng DB
    with pytest.raises(SQLExecutionError):
        asyncio.run(backend.execute_readonly("DELETE FROM companies"))

    async def generator(query: str, prev_sql: str | None, error: str | None) -> str:
        if error and "no such column" in error.lower():
            return "SELECT ticker, roe_pct FROM quarterly_financials WHERE ticker = 'FPT' AND quarter = 1"
        return "SELECT ticker, roe_wrong_col FROM quarterly_financials WHERE ticker = 'FPT'"

    state = AgentState(user_query="ROE quý 1/2025 của FPT?")
    result_state = asyncio.run(run_sql_with_self_correction(state, generator, backend))

    assert result_state.sql_error is None
    assert result_state.sql_retry_count == 1
    assert result_state.sql_result == [{"ticker": "FPT", "roe_pct": 28.4}]
