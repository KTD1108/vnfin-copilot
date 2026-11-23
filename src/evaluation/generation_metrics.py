"""Đo lường tầng Generation: Chuẩn hóa số liệu tiếng Việt & Numerical Accuracy."""

import re

_NUMBER_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)*")
DEFAULT_NUMERICAL_TOLERANCE: float = 0.005  # Sai số tương đối tối đa 0.5%


def parse_vn_number(raw: str) -> float | None:
    """Chuẩn hóa chuỗi số tiếng Việt/quốc tế (VD: '62.849' -> 62849.0, '19,4' -> 19.4)."""
    cleaned = raw.strip()
    if not cleaned:
        return None

    # Dạng có dấu chấm ngăn cách hàng nghìn rõ ràng (VD: 62.849 hoặc 1.250.000)
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    elif "," in cleaned and "." in cleaned:
        # VD: 1,234.56 hoặc 1.234,56
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # VD: 19,4 -> 19.4 (hoặc 19,400 nếu 3 chữ số sau dấu phẩy)
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0].lstrip("-")) <= 3:
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numbers_vn(text: str) -> list[float]:
    """Trích xuất toàn bộ các con số hợp lệ trong câu trả lời."""
    numbers: list[float] = []
    for match in _NUMBER_PATTERN.findall(text):
        parsed = parse_vn_number(match)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def numerical_accuracy(
    predicted_answer: str,
    ground_truth_numbers: list[float],
    tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> float:
    """Tính tỷ lệ con số chuẩn xuất hiện chính xác trong câu trả lời dự đoán."""
    if not ground_truth_numbers:
        return 1.0

    pred_nums = extract_numbers_vn(predicted_answer)
    if not pred_nums:
        return 0.0

    matched = 0
    for gt in ground_truth_numbers:
        scale = max(1.0, abs(gt))
        if any(abs(pred - gt) / scale <= tolerance for pred in pred_nums):
            matched += 1
    return matched / len(ground_truth_numbers)


def compute_faithfulness_overlap(answer: str, contexts: list[str]) -> float:
    """Ước lượng độ trung thực dựa trên mức độ phủ con số & thực thể từ Context."""
    if not answer.strip():
        return 0.0
    if not contexts:
        return 0.0

    context_blob = " ".join(contexts)
    ans_nums = extract_numbers_vn(answer)
    ctx_nums = extract_numbers_vn(context_blob)
    if not ans_nums:
        return 1.0

    supported = sum(
        1 for n in ans_nums if any(abs(n - c) / max(1.0, abs(c)) <= DEFAULT_NUMERICAL_TOLERANCE for c in ctx_nums)
    )
    return supported / len(ans_nums)
