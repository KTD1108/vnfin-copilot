"""State schemas cho Multi-Agent State Machine (Giai đoạn 3)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

MAX_SQL_RETRIES: int = 3
MAX_REWRITES: int = 2
DEFAULT_SQL_LIMIT: int = 100
DEFAULT_SQL_TIMEOUT_MS: int = 3000


class AgentIntent(str, Enum):
    """Phân loại ý định truy vấn bởi Router Agent."""

    RAG_ONLY = "RAG_ONLY"
    SQL_ONLY = "SQL_ONLY"
    HYBRID_BOTH = "HYBRID_BOTH"


class HallucinationGrade(str, Enum):
    """Kết quả kiểm chứng số liệu của Fact Grader."""

    PASSED = "PASSED"
    HALLUCINATED = "HALLUCINATED"
    INCOMPLETE = "INCOMPLETE"


class AgentState(BaseModel):
    """Trạng thái dùng chung xuyên suốt các Node trong đồ thị điều phối."""

    user_query: str = Field(..., min_length=1)
    ticker: str | None = None
    intent: AgentIntent = AgentIntent.HYBRID_BOTH
    sub_queries: list[str] = Field(default_factory=list)
    sql_query: str | None = None
    sql_result: list[dict[str, Any]] = Field(default_factory=list)
    sql_error: str | None = None
    sql_retry_count: int = Field(default=0, ge=0)
    retrieved_contexts: list[str] = Field(default_factory=list)
    draft_answer: str = ""
    hallucination_grade: HallucinationGrade = HallucinationGrade.INCOMPLETE
    rewrite_count: int = Field(default=0, ge=0)
