"""Multi-Agent Orchestrator: Định tuyến, Chạy song song RAG + SQL & Kiểm chứng Số liệu."""

import asyncio
import json
from collections.abc import Awaitable, Callable

from agents.sql_agent import SQLBackend, SQLGeneratorFn, run_sql_with_self_correction
from agents.state import AgentIntent, AgentState, HallucinationGrade
from evaluation.generation_metrics import compute_faithfulness_overlap

RAGSearchFn = Callable[[str, str | None], Awaitable[list[str]]]
SynthesizeFn = Callable[[AgentState], Awaitable[str]]


def route_intent(query: str) -> AgentIntent:
    """Phân loại ý định câu hỏi dựa trên từ khóa định lượng vs giải trình."""
    q_lower = query.lower()
    needs_sql = any(kw in q_lower for kw in ("p/e", "roe", "giá cổ phiếu", "top ", "trung bình", "sql", "cơ sở dữ liệu"))
    needs_rag = any(kw in q_lower for kw in ("giải trình", "nguyên nhân", "báo cáo", "thuyết minh", "rủi ro", "động lực"))

    if needs_sql and needs_rag:
        return AgentIntent.HYBRID_BOTH
    if needs_sql:
        return AgentIntent.SQL_ONLY
    return AgentIntent.RAG_ONLY


def grade_fact_consistency(state: AgentState) -> HallucinationGrade:
    """Kiểm chứng mọi con số trong draft_answer có khớp với SQL + RAG Context hay không."""
    if not state.draft_answer.strip():
        return HallucinationGrade.INCOMPLETE

    sources: list[str] = list(state.retrieved_contexts)
    if state.sql_result:
        sources.append(json.dumps(state.sql_result, ensure_ascii=False))

    score = compute_faithfulness_overlap(state.draft_answer, sources)
    return HallucinationGrade.PASSED if score >= 0.99 else HallucinationGrade.HALLUCINATED


async def run_agent_workflow(
    state: AgentState,
    rag_search: RAGSearchFn,
    sql_generator: SQLGeneratorFn,
    sql_backend: SQLBackend,
    synthesizer: SynthesizeFn,
) -> AgentState:
    """Thực thi luồng Multi-Agent hoàn chỉnh kèm Fact Grader."""
    state.intent = route_intent(state.user_query)

    if state.intent == AgentIntent.RAG_ONLY:
        state.retrieved_contexts = await rag_search(state.user_query, state.ticker)
    elif state.intent == AgentIntent.SQL_ONLY:
        state = await run_sql_with_self_correction(state, sql_generator, sql_backend)
    else:
        rag_task = rag_search(state.user_query, state.ticker)
        sql_task = run_sql_with_self_correction(state, sql_generator, sql_backend)
        contexts, updated_sql_state = await asyncio.gather(rag_task, sql_task)
        state.retrieved_contexts = contexts
        state.sql_query = updated_sql_state.sql_query
        state.sql_result = updated_sql_state.sql_result
        state.sql_error = updated_sql_state.sql_error
        state.sql_retry_count = updated_sql_state.sql_retry_count

    state.draft_answer = await synthesizer(state)
    state.hallucination_grade = grade_fact_consistency(state)
    return state
