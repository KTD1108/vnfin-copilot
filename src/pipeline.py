"""End-to-End VnFinCopilot Pipeline: Kết nối Hybrid RAG + Multi-Agent + SQLite/SQL + Guardrails."""

import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from agents.orchestrator import run_agent_workflow
from agents.sql_agent import SQLBackend
from agents.sqlite_backend import ReadOnlySQLiteBackend, seed_sample_financial_db
from agents.state import AgentState, HallucinationGrade
from rag.generation import generate_grounded_answer
from rag.indexing.embedder import DeterministicHashDenseEmbedder
from rag.indexing.qdrant_store import QdrantHybridStore
from rag.ingestion.chunker import DocumentContext, TableAwareParentChildChunker
from rag.retrieval.reranker import LexicalOverlapReranker
from rag.retrieval.retriever import HybridRetriever
from rag.schemas import DocumentSection, FinancialDocType, RankedCandidate

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/vnfin_local.db")
DEFAULT_SCHEMA_PATH = Path("src/agents/schema.sql")


class CopilotResponse(BaseModel):
    """Kết quả tổng hợp hoàn chỉnh từ End-to-End Pipeline."""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    intent: str
    sql_query: str | None = None
    hallucination_grade: str


def _seed_default_rag_corpus(store: QdrantHybridStore) -> None:
    """Nạp dữ liệu BCTC chuẩn ban đầu nếu Qdrant chưa có collection."""
    if store.client.collection_exists(store.collection_name):
        return

    chunker = TableAwareParentChildChunker()
    fpt_sections = [
        DocumentSection(
            section_title="Báo cáo Kết quả Kinh doanh Hợp nhất FPT",
            content=(
                "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
                "| :--- | :---: | :---: |\n"
                "| Doanh thu thuần | 52.618 | 62.849 |\n"
                "| Lợi nhuận gộp | 20.513 | 24.837 |\n"
                "| Lợi nhuận sau thuế | 7.792 | 9.420 |"
            ),
            page_number=6,
            is_table=True,
            currency_unit="Tỷ VNĐ",
        ),
        DocumentSection(
            section_title="Thuyết minh khối Công nghệ FPT",
            content=(
                "Động lực tăng trưởng chính của FPT đến từ mảng dịch vụ CNTT nước ngoài, "
                "đặc biệt thị trường Nhật Bản tăng trưởng 32% nhờ nhu cầu chuyển đổi số và AI."
            ),
            page_number=24,
            is_table=False,
            currency_unit="Tỷ VNĐ",
        ),
    ]
    vcb_sections = [
        DocumentSection(
            section_title="Báo cáo Kết quả Kinh doanh Vietcombank",
            content=(
                "| Chỉ tiêu | Năm 2024 | Năm 2025 |\n"
                "| :--- | :---: | :---: |\n"
                "| Thu nhập lãi thuần | 55.400 | 61.200 |\n"
                "| Lợi nhuận trước thuế | 41.200 | 45.800 |"
            ),
            page_number=4,
            is_table=True,
            currency_unit="Tỷ VNĐ",
        ),
    ]
    chunks = chunker.chunk_sections(
        fpt_sections,
        DocumentContext(ticker="FPT", year=2025, quarter=None, doc_type=FinancialDocType.BCTC_HOP_NHAT),
    ) + chunker.chunk_sections(
        vcb_sections,
        DocumentContext(ticker="VCB", year=2025, quarter=None, doc_type=FinancialDocType.BCTC_HOP_NHAT),
    )
    store.upsert_chunks(chunks)


def _rule_based_text_to_sql(query: str, ticker: str | None) -> str:
    """Sinh câu lệnh SQL chuẩn hóa (fallback khi chưa chạy model vLLM/OpenAI)."""
    q_lower = query.lower()
    target_ticker = ticker.upper() if ticker else None
    if not target_ticker:
        for code in ("FPT", "FOX", "VCB", "HPG"):
            if code.lower() in q_lower:
                target_ticker = code
                break

    if "p/e" in q_lower or "giá cổ phiếu" in q_lower:
        where = f"WHERE ticker = '{target_ticker}' " if target_ticker else ""
        return f"SELECT ticker, trade_date, close_price_vnd, pe_ratio, pb_ratio FROM daily_stock_prices {where}ORDER BY trade_date DESC LIMIT 5"

    if "top" in q_lower and "roe" in q_lower:
        return "SELECT ticker, year, quarter, roe_pct, net_profit_billion_vnd FROM quarterly_financials ORDER BY roe_pct DESC LIMIT 5"

    where = f"WHERE ticker = '{target_ticker}' " if target_ticker else ""
    return f"SELECT ticker, year, quarter, revenue_billion_vnd, net_profit_billion_vnd, roe_pct FROM quarterly_financials {where}ORDER BY year DESC, quarter DESC LIMIT 5"


class VnFinCopilotPipeline:
    """Pipeline End-to-End kết nối Hybrid RAG, Multi-Agent, Text-to-SQL và Citation."""

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        sql_backend: SQLBackend | None = None,
        db_path: Path = DEFAULT_DB_PATH,
    ) -> None:
        self.embedder = DeterministicHashDenseEmbedder()
        self.qdrant_client = qdrant_client or QdrantClient(":memory:")
        self.store = QdrantHybridStore(client=self.qdrant_client, dense_embedder=self.embedder)
        _seed_default_rag_corpus(self.store)

        self.retriever = HybridRetriever(
            client=self.qdrant_client,
            dense_embedder=self.embedder,
            reranker=LexicalOverlapReranker(),
        )
        if sql_backend is not None:
            self.sql_backend = sql_backend
        else:
            seed_sample_financial_db(db_path, DEFAULT_SCHEMA_PATH)
            self.sql_backend = ReadOnlySQLiteBackend(db_path)

    async def run(
        self,
        query: str,
        ticker: str | None = None,
        year: int | None = None,
    ) -> CopilotResponse:
        """Thực thi toàn bộ luồng Multi-Agent và trả về câu trả lời kèm trích dẫn."""
        last_candidates: list[RankedCandidate] = []

        async def rag_search_fn(q: str, t: str | None) -> list[str]:
            nonlocal last_candidates
            resolved_ticker = t or self._detect_ticker(q)
            last_candidates = self.retriever.retrieve(
                query=q,
                ticker=resolved_ticker,
                year=year,
                final_top_k=3,
            )
            return [c.metadata.parent_text for c in last_candidates]

        async def sql_gen_fn(q: str, prev_sql: str | None, error: str | None) -> str:
            return _rule_based_text_to_sql(q, ticker)

        async def synth_fn(st: AgentState) -> str:
            if os.getenv("OPENAI_API_KEY") and last_candidates:
                return await generate_grounded_answer(st.user_query, last_candidates)
            return self._offline_grounded_synthesis(st, last_candidates)

        init_state = AgentState(user_query=query, ticker=ticker)
        final_state = await run_agent_workflow(
            state=init_state,
            rag_search=rag_search_fn,
            sql_generator=sql_gen_fn,
            sql_backend=self.sql_backend,
            synthesizer=synth_fn,
        )

        citations = [
            {
                "doc_type": c.metadata.doc_type.value,
                "ticker": c.metadata.ticker,
                "period": c.metadata.period_label,
                "page": c.metadata.page_number,
                "section": c.metadata.section_title,
            }
            for c in last_candidates
        ]
        return CopilotResponse(
            answer=final_state.draft_answer,
            citations=citations,
            intent=final_state.intent.value,
            sql_query=final_state.sql_query,
            hallucination_grade=final_state.hallucination_grade.value,
        )

    @staticmethod
    def _detect_ticker(query: str) -> str | None:
        match = re.search(r"\b(FPT|FOX|VCB|HPG|VNM)\b", query.upper())
        return match.group(1) if match else None

    @staticmethod
    def _offline_grounded_synthesis(
        state: AgentState,
        candidates: list[RankedCandidate],
    ) -> str:
        """Tổng hợp câu trả lời bám sát 100% dữ liệu nguồn SQL & RAG khi chạy offline."""
        parts: list[str] = []
        if state.sql_result:
            rows_str = "; ".join(
                ", ".join(f"{k}: {v}" for k, v in row.items()) for row in state.sql_result[:3]
            )
            parts.append(f"**Số liệu truy vấn từ Database:** {rows_str}.")
        if candidates:
            top = candidates[0]
            parts.append(
                f"**Thông tin trích xuất từ tài liệu** "
                f"[Nguồn: {top.metadata.doc_type.value} {top.metadata.ticker} "
                f"{top.metadata.period_label}, Trang {top.metadata.page_number}]:\n"
                f"{top.metadata.parent_text}"
            )
        if not parts:
            return "Tài liệu và cơ sở dữ liệu hiện có không chứa thông tin về vấn đề này."
        return "\n\n".join(parts)
