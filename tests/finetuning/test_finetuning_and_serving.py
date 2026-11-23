"""Unit tests cho Giai đoạn 4: ChatML Data Prep, Leakage Prevention, QLoRA & vLLM Benchmark."""

from pathlib import Path

import pytest

from finetuning.data_prep import (
    TextToSQLSample,
    deduplicate_and_prevent_leakage,
    export_jsonl,
    to_chatml_record,
)
from finetuning.qlora_config import QLoRATrainingConfig
from serving.benchmark import build_vllm_serve_command, compute_benchmark_report


def test_data_prep_prevents_leakage_and_exports_chatml(tmp_path: Path) -> None:
    """Loại bỏ trùng lặp, chặn rò rỉ tập test và xuất đúng định dạng ChatML."""
    samples = [
        TextToSQLSample(
            question="ROE của FPT năm 2025?",
            db_schema="quarterly_financials(ticker, year, roe)",
            sql_query="SELECT roe FROM quarterly_financials WHERE ticker='FPT' AND year=2025",
        ),
        TextToSQLSample(
            question="roe của fpt năm 2025?",  # Trùng lặp khác hoa thường
            db_schema="quarterly_financials(ticker, year, roe)",
            sql_query="SELECT roe FROM quarterly_financials WHERE ticker='FPT' AND year=2025",
        ),
        TextToSQLSample(
            question="Top 5 ngân hàng có P/E thấp nhất?",  # Nằm trong tập test
            db_schema="daily_stock_prices(ticker, pe)",
            sql_query="SELECT ticker, pe FROM daily_stock_prices ORDER BY pe ASC LIMIT 5",
        ),
    ]
    clean = deduplicate_and_prevent_leakage(
        samples,
        test_questions={"Top 5 ngân hàng có P/E thấp nhất?"},
    )
    assert len(clean) == 1

    chatml = to_chatml_record(clean[0])
    assert [m["role"] for m in chatml["messages"]] == ["system", "user", "assistant"]

    out_file = tmp_path / "train.jsonl"
    assert export_jsonl(clean, out_file) == 1
    assert out_file.exists()


def test_qlora_config_and_vllm_benchmark() -> None:
    """Kiểm tra cấu hình QLoRA và tính toán báo cáo benchmark TTFT/ITL/Throughput."""
    cfg = QLoRATrainingConfig()
    assert cfg.lora_alpha == 2 * cfg.lora_r
    assert cfg.mask_user_prompt_loss is True

    cmd = build_vllm_serve_command("models/vnfin-qwen2.5-3b-awq")
    assert "--quantization" in cmd and "--enable-prefix-caching" in cmd

    report = compute_benchmark_report(
        concurrency=10,
        ttft_ms_list=[120.0, 140.0, 180.0, 250.0],
        itl_ms_list=[15.0, 16.0, 17.0],
        total_output_tokens=1000,
        total_wall_time_sec=5.0,
    )
    assert report.throughput_tokens_per_sec == pytest.approx(200.0)
    assert report.p95_ttft_ms == pytest.approx(250.0)
