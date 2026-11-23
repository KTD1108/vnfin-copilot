"""Chạy thực nghiệm đối chứng (Ablation Study) và xuất bảng Markdown chuẩn báo cáo."""

from evaluation.schemas import AblationRow


def format_ablation_markdown(rows: list[AblationRow]) -> str:
    """Định dạng kết quả Ablation Study thành bảng Markdown chuẩn."""
    header = (
        "| Cấu hình | Hit Rate@5 | Recall@5 | MRR@5 | Faithfulness | Numerical Acc | Avg Latency (ms) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row.config_name} | {row.hit_rate_at_5:.3f} | {row.recall_at_5:.3f} | "
            f"{row.mrr_at_5:.3f} | {row.faithfulness:.3f} | {row.numerical_accuracy:.3f} | "
            f"{row.avg_latency_ms:.1f} |"
        )
    return "\n".join(lines)
