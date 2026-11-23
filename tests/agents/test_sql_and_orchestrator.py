"""Unit & Integration tests cho Giai đoạn 3: SQL Guardrail, Self-Correction & Multi-Agent."""

import asyncio
from typing import Any

import pytest

from agents.orchestrator import run_agent_workflow
from agents.sql_agent import SQLExecutionError, run_sql_with_self_correction
from agents.sql_guardrail import SQLSecurityError, validate_readonly_sql
from agents.state import AgentIntent, AgentState, HallucinationGrade


def test_sql_guardrail_blocks_dangerous_queries() -> None:
    """SQL Guardrail phải chặn 100% lệnh phá hoại hoặc nhiều câu lệnh ghép."""
    dangerous = [
        "DROP TABLE companies;",
        "DELETE FROM quarterly_financials WHERE ticker = 'FPT'",
        "SELECT * FROM companies; DROP TABLE companies;",
        "UPDATE daily_stock_prices SET close_price = 0",
    ]
    for sql in dangerous:
        with pytest.raises(SQLSecurityError):
            validate_readonly_sql(sql)

    safe = validate_readonly_sql("SELECT ticker, roe FROM quarterly_financials WHERE year = 2025")
    assert safe.endswith("LIMIT 100")


def test_sql_self_correction_recovers_on_second_attempt() -> None:
    """Agent đọc lỗi sai tên cột ở lần 1 và tự sửa thành công ở lần 2."""

    class MockBackend:
        async def execute_readonly(self, sql: str) -> list[dict[str, Any]]:
            if "net_income" in sql:
                raise SQLExecutionError('column "net_income" does not exist. Use "net_profit".')
            return [{"ticker": "FPT", "net_profit": 9420}]

    async def mock_generator(query: str, prev_sql: str | None, error: str | None) -> str:
        if error and "net_profit" in error:
            return "SELECT ticker, net_profit FROM quarterly_financials WHERE ticker = 'FPT'"
        return "SELECT ticker, net_income FROM quarterly_financials WHERE ticker = 'FPT'"

    state = AgentState(user_query="Lợi nhuận ròng FPT?")
    updated = asyncio.run(run_sql_with_self_correction(state, mock_generator, MockBackend()))

    assert updated.sql_error is None
    assert updated.sql_retry_count == 1
    assert updated.sql_result == [{"ticker": "FPT", "net_profit": 9420}]


def test_hybrid_orchestrator_runs_rag_and_sql_and_passes_fact_grader() -> None:
    """Câu hỏi kết hợp phải chạy cả RAG + SQL và qua được Fact Grader."""

    class MockBackend:
        async def execute_readonly(self, sql: str) -> list[dict[str, Any]]:
            return [{"ticker": "FPT", "roe": 28.5}]

    async def mock_sql_gen(query: str, prev_sql: str | None, error: str | None) -> str:
        return "SELECT ticker, roe FROM quarterly_financials WHERE ticker = 'FPT'"

    async def mock_rag(query: str, ticker: str | None) -> list[str]:
        return ["Thuyết minh FPT: Động lực tăng trưởng đến từ mảng chuyển đổi số tại Nhật Bản (+32%)."]

    async def mock_synth(st: AgentState) -> str:
        return "FPT đạt ROE 28.5% nhờ động lực tăng trưởng tại Nhật Bản tăng 32%."

    state = AgentState(
        user_query="ROE của FPT là bao nhiêu và giải trình động lực tăng trưởng theo báo cáo?",
        ticker="FPT",
    )
    final_state = asyncio.run(
        run_agent_workflow(state, mock_rag, mock_sql_gen, MockBackend(), mock_synth)
    )

    assert final_state.intent == AgentIntent.HYBRID_BOTH
    assert final_state.hallucination_grade == HallucinationGrade.PASSED
