"""Cài đặt toán học chuẩn xác cho các chỉ số Retrieval: Hit Rate@K, Recall@K, MRR@K."""


def hit_rate_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """Trả về 1.0 nếu có ít nhất 1 tài liệu chuẩn nằm trong Top-K, ngược lại 0.0."""
    if k <= 0:
        raise ValueError("Tham số k phải lớn hơn 0.")
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0

    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & set(ground_truth_ids)) else 0.0


def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """Tính tỷ lệ tài liệu chuẩn tìm được trong Top-K trên tổng số tài liệu chuẩn."""
    if k <= 0:
        raise ValueError("Tham số k phải lớn hơn 0.")
    if not ground_truth_ids:
        return 1.0

    top_k = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    return len(top_k & gt_set) / len(gt_set)


def mrr_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5) -> float:
    """Mean Reciprocal Rank: 1 / rank của tài liệu chuẩn đầu tiên trong Top-K."""
    if k <= 0:
        raise ValueError("Tham số k phải lớn hơn 0.")
    if not ground_truth_ids:
        return 1.0

    gt_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0
