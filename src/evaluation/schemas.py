"""Pydantic schemas cho Golden Evaluation Dataset và Báo cáo Ablation Study."""

from enum import Enum
from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    """4 nhóm độ khó chuẩn của bộ kiểm thử tài chính."""

    SIMPLE_LOOKUP = "simple_lookup"
    TABLE_CALCULATION = "table_calculation"
    MULTI_HOP_COMPARISON = "multi_hop_comparison"
    UNANSWERABLE_TRAP = "unanswerable_trap"


class GoldenEvalSample(BaseModel):
    """Một mẫu chuẩn trong Golden Evaluation Dataset."""

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    category: EvalCategory
    expected_tickers: list[str] = Field(default_factory=list)
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    ground_truth_answer: str = Field(..., min_length=1)
    ground_truth_numbers: list[float] = Field(default_factory=list)


class AblationRow(BaseModel):
    """Kết quả benchmark của một cấu hình hệ thống."""

    config_name: str
    hit_rate_at_5: float
    recall_at_5: float
    mrr_at_5: float
    faithfulness: float
    numerical_accuracy: float
    avg_latency_ms: float
