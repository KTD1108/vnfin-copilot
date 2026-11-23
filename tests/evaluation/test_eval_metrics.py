"""Unit tests cho Giai đoạn 2: Retrieval Metrics, Vietnamese Number Parser & Ablation."""

import json
from pathlib import Path

import pytest

from evaluation.ablation import format_ablation_markdown
from evaluation.generation_metrics import (
    compute_faithfulness_overlap,
    numerical_accuracy,
    parse_vn_number,
)
from evaluation.retrieval_metrics import hit_rate_at_k, mrr_at_k, recall_at_k
from evaluation.schemas import AblationRow, GoldenEvalSample


def test_retrieval_metrics_math() -> None:
    """Kiểm tra chính xác công thức Hit Rate@K, Recall@K và MRR@K."""
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    gt = ["d2", "d4"]

    assert hit_rate_at_k(retrieved, gt, k=5) == 1.0
    assert hit_rate_at_k(retrieved, ["d9"], k=5) == 0.0
    assert recall_at_k(retrieved, gt, k=3) == pytest.approx(0.5)
    assert recall_at_k(retrieved, gt, k=5) == pytest.approx(1.0)
    assert mrr_at_k(retrieved, gt, k=5) == pytest.approx(0.5)  # d2 ở rank 2 -> 1/2
    assert mrr_at_k(["d2", "d1"], gt, k=5) == pytest.approx(1.0)


def test_vietnamese_number_parsing_and_accuracy() -> None:
    """Kiểm tra chuẩn hóa số phân cách kiểu Việt Nam (62.849 và 19,4)."""
    assert parse_vn_number("62.849") == pytest.approx(62849.0)
    assert parse_vn_number("19,4") == pytest.approx(19.4)
    assert parse_vn_number("1.250.000") == pytest.approx(1250000.0)

    ans = "Doanh thu thuần năm 2025 đạt 62.849 tỷ VNĐ, tăng 19,4%."
    assert numerical_accuracy(ans, [62849.0, 19.4]) == pytest.approx(1.0)
    assert compute_faithfulness_overlap(ans, ["Doanh thu năm 2025 đạt 62.849, tăng 19,4%."]) == pytest.approx(1.0)


def test_golden_dataset_and_ablation_table() -> None:
    """Kiểm tra schema Golden Dataset và định dạng bảng Ablation Markdown."""
    raw = json.loads(Path("data/eval/golden_dataset.json").read_text(encoding="utf-8"))
    samples = [GoldenEvalSample.model_validate(item) for item in raw]
    assert len(samples) == 4

    md = format_ablation_markdown(
        [
            AblationRow(
                config_name="Hybrid + Reranker",
                hit_rate_at_5=0.95,
                recall_at_5=0.892,
                mrr_at_5=0.78,
                faithfulness=0.96,
                numerical_accuracy=0.94,
                avg_latency_ms=215.4,
            )
        ]
    )
    assert "| Hybrid + Reranker | 0.950 | 0.892 | 0.780 |" in md
