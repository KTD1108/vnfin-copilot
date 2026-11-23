"""Benchmark hiệu năng Serving (TTFT, ITL, Throughput) & Cấu hình vLLM."""

from pydantic import BaseModel, Field


class ServingBenchmarkReport(BaseModel):
    """Báo cáo hiệu năng suy luận của mô hình khi chạy đồng thời nhiều request."""

    concurrency: int = Field(..., ge=1)
    total_requests: int = Field(..., ge=1)
    p50_ttft_ms: float = Field(..., ge=0.0)
    p95_ttft_ms: float = Field(..., ge=0.0)
    mean_itl_ms: float = Field(..., ge=0.0)
    throughput_tokens_per_sec: float = Field(..., ge=0.0)


def percentile(values: list[float], pct: float) -> float:
    """Tính phân vị (VD: P50, P95) chính xác từ danh sách độ trễ."""
    if not values:
        raise ValueError("Danh sách giá trị đo không được rỗng.")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def compute_benchmark_report(
    concurrency: int,
    ttft_ms_list: list[float],
    itl_ms_list: list[float],
    total_output_tokens: int,
    total_wall_time_sec: float,
) -> ServingBenchmarkReport:
    """Tổng hợp các chỉ số TTFT, ITL và System Throughput."""
    if total_wall_time_sec <= 0:
        raise ValueError("Tổng thời gian chạy phải lớn hơn 0.")
    return ServingBenchmarkReport(
        concurrency=concurrency,
        total_requests=len(ttft_ms_list),
        p50_ttft_ms=round(percentile(ttft_ms_list, 50.0), 2),
        p95_ttft_ms=round(percentile(ttft_ms_list, 95.0), 2),
        mean_itl_ms=round(sum(itl_ms_list) / max(1, len(itl_ms_list)), 2),
        throughput_tokens_per_sec=round(total_output_tokens / total_wall_time_sec, 2),
    )


def build_vllm_serve_command(
    model_path: str,
    quantization: str = "awq",
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
) -> list[str]:
    """Sinh lệnh khởi chạy vLLM OpenAI-compatible Server chuẩn tối ưu."""
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--quantization",
        quantization,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd
