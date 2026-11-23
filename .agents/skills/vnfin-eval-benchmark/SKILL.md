---
name: vnfin-eval-benchmark
description: >-
  Use this skill when building or running Phase 2 of VnFinCopilot: creating the
  Vietnamese financial Golden Evaluation Dataset, computing retrieval metrics
  (Hit Rate@K, Recall@K, MRR@K, NDCG@K), running LLM-as-a-Judge / RAGAS generation
  metrics (Faithfulness, Answer Relevance, Numerical Accuracy), conducting Ablation
  Studies, or integrating Langfuse tracing.
---

# VnFinCopilot Evaluation & Observability (Giai đoạn 2)

Quy trình chuẩn để đo lường định lượng chất lượng hệ thống AI bằng toán học, thực hiện thực nghiệm đối chứng (**Ablation Study**) chuẩn nghiên cứu VinAI, và giám sát hệ thống bằng **Langfuse**.

## 1. Cấu trúc Chuẩn của Golden Evaluation Dataset

Mọi mẫu kiểm thử trong `data/eval/golden_dataset.json` phải tuân thủ Pydantic Schema sau:
- `id`: Mã câu hỏi (VD: `"eval_fpt_001"`).
- `question`: Câu hỏi tiếng Việt sát thực tế phân tích tài chính.
- `category`: Một trong 4 nhóm:
  1. `simple_lookup`: Tra cứu 1 chỉ tiêu đơn lẻ từ PDF/SQL.
  2. `table_calculation`: Đọc bảng kế toán và tính tỷ lệ tăng trưởng / biên lợi nhuận.
  3. `multi_hop_comparison`: So sánh nhiều quý hoặc nhiều mã chứng khoán (VD: FPT vs CMG).
  4. `unanswerable_trap`: Câu hỏi bẫy về thông tin không có trong tài liệu (để test khả năng từ chối trả lời, chống bịa đặt).
- `expected_tickers`: Danh sách mã CK liên quan (VD: `["FPT"]`).
- `ground_truth_chunk_ids`: Danh sách `chunk_id` hoặc `parent_id` chứa đáp án chuẩn.
- `ground_truth_answer`: Câu trả lời chuẩn đã được kiểm chứng.
- `ground_truth_numbers`: Danh sách các con số quan trọng bắt buộc phải xuất hiện chính xác trong câu trả lời (VD: `[62849.0, 19.4]`).

## 2. Quy trình Đo lường Định lượng (Evaluation Pipeline)

### Bước 1: Tự cài đặt Retrieval Metrics (`src/evaluation/retrieval_metrics.py`)
Không phụ thuộc mù quáng vào thư viện hộp đen; luôn cài đặt rõ ràng và có unit test cho các công thức:
1. **`hit_rate_at_k(retrieved_ids, ground_truth_ids, k)`:** Trả về `1.0` nếu giao của `retrieved_ids[:k]` và `ground_truth_ids` khác rỗng, ngược lại `0.0`.
2. **`recall_at_k(retrieved_ids, ground_truth_ids, k)`:**
   $$\text{Recall@K} = \frac{|\text{Retrieved}_{1..K} \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$
3. **`mrr_at_k(retrieved_ids, ground_truth_ids, k)`:**
   $$\text{MRR@K} = \frac{1}{\text{rank}_{\text{first\_relevant}}}$$ (Nếu không tìm thấy trong Top $K$, trả về `0.0`).

### Bước 2: Đo lường Generation & Numerical Accuracy (`src/evaluation/generation_metrics.py`)
1. **Faithfulness (Độ trung thực):** Kiểm tra tỷ lệ mệnh đề trong câu trả lời được suy ra trực tiếp từ Context (mục tiêu $\ge 0.92$).
2. **Numerical Exact/Tolerance Match:**
   - Trích xuất các con số tài chính trong câu trả lời của hệ thống (chuẩn hóa dấu chấm/phẩy phân cách hàng nghìn theo kiểu Việt Nam `62.849` vs `62,849`).
   - Đối chiếu với `ground_truth_numbers` với sai số tương đối cho phép $\le 0.5\%$.

### Bước 3: Chạy Ablation Study & Xuất Báo Cáo Markdown
Khi người dùng yêu cầu đánh giá hoặc so sánh các cải tiến kỹ thuật, chạy benchmark trên cùng một tập Golden Dataset qua 3 cấu hình:
1. `Baseline (Dense Only)`
2. `Hybrid Search (Dense + BM25 + RRF)`
3. `Hybrid Search + Cross-Encoder Reranker`

Xuất kết quả tự động ra bảng Markdown gồm các cột:
`| Cấu hình | Hit Rate@5 | Recall@5 | MRR@5 | Faithfulness | Numerical Acc | Avg Latency (ms) |`

## 3. Tích hợp Langfuse Observability (`src/observability/`)
- Mọi hàm gọi Embedding, Qdrant Search, Reranker, SQL Execution và LLM Generation đều phải được bọc tracing (sử dụng decorator `@observe()` của Langfuse hoặc context manager).
- Ghi nhận đầy đủ: `input_tokens`, `output_tokens`, `latency_ms`, và `model_name`.

## 4. Cổng Kiểm chứng (Verification Gate)
- Chạy `pytest tests/evaluation/ -v` để kiểm tra độ chính xác toán học của các hàm `Recall@K`, `MRR@K` và bộ chuẩn hóa số liệu tiếng Việt.
- Tham chiếu lý thuyết tại [03-GIAI-DOAN-2-EVALUATION-VA-OBSERVABILITY.md](../../docs/03-GIAI-DOAN-2-EVALUATION-VA-OBSERVABILITY.md).
