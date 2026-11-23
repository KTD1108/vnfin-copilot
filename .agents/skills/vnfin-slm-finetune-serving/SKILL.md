---
name: vnfin-slm-finetune-serving
description: >-
  Use this skill when working on Phase 4 of VnFinCopilot: generating synthetic
  Vietnamese financial Text-to-SQL or Query Routing datasets, fine-tuning a Small
  Language Model (Qwen-2.5-3B or 7B) using LoRA/QLoRA (Unsloth/TRL) and DPO,
  quantizing models to 4-bit AWQ/GGUF, deploying with vLLM, or benchmarking
  inference latency (TTFT, ITL) and throughput.
---

# VnFinCopilot SLM Fine-Tuning & High-Performance Serving (Giai đoạn 4)

Quy trình chuẩn bị dữ liệu, huấn luyện tinh chỉnh mô hình ngôn ngữ nhỏ (**SLM** như `Qwen-2.5-3B-Instruct` / `7B-Instruct`) bằng **QLoRA**, nén **4-bit AWQ** và phục vụ tốc độ cao bằng **vLLM** (Trọng tâm kiến thức kỹ thuật sâu chuẩn VinAI).

## 1. Quy trình Tạo & Kiểm duyệt Dữ liệu Huấn luyện (`src/finetuning/data_gen.py`)

1. **Nguyên tắc Chất lượng hơn Số lượng (Data-Centric AI):**
   - Đối với bài toán **Vietnamese Financial Text-to-SQL**, mỗi mẫu dữ liệu sinh ra (`instruction`, `db_schema`, `sql_output`) **bắt buộc phải được chạy thử (Execution Validation)** trên cơ sở dữ liệu PostgreSQL thực tế trước khi đưa vào tập `train.jsonl`.
   - Loại bỏ 100% các mẫu câu lệnh SQL chạy bị lỗi cú pháp hoặc trả về bảng rỗng vô lý.
2. **Chống Rò rỉ Dữ liệu (Data Leakage Prevention):**
   - Chia rõ tập `train` (80%), `val` (10%) và `test` (10%).
   - Kiểm tra khử trùng lặp (Deduplication) để không có câu hỏi nào trong tập `Golden Test Set` xuất hiện trong tập `train`.
3. **Định dạng ChatML Chuẩn:**
   - Định dạng mọi mẫu dữ liệu theo đúng chuẩn `ChatML` (`system`, `user`, `assistant`) của dòng mô hình `Qwen-2.5`.

## 2. Cấu hình Chuẩn cho QLoRA Fine-Tuning (`src/finetuning/train_qlora.py`)

Khi viết script huấn luyện với `Unsloth` hoặc `HuggingFace PEFT + TRL (SFTTrainer)`, luôn tách siêu tham số ra file cấu hình rõ ràng:
- **Quantization:** `load_in_4bit = True`, `bnb_4bit_quant_type = "nf4"`, `bnb_4bit_use_double_quant = True`, `bnb_4bit_compute_dtype = torch.bfloat16` (hoặc `float16` nếu GPU không hỗ trợ bf16).
- **LoRA Config:**
  - `r = 16` (hoặc `32`)
  - `lora_alpha = 32` (quy tắc chuẩn: $\alpha = 2 \times r$)
  - `lora_dropout = 0.0` (khi dùng Unsloth để tối ưu kernel nhanh nhất) hoặc `0.05`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (gắn LoRA vào toàn bộ các tầng tuyến tính của cả Attention và MLP để đạt độ chính xác cao nhất).
- **Loss Masking (`DataCollatorForCompletionOnlyLM`):**
  - Chỉ tính hàm mất mát (Cross-Entropy Loss) trên phần câu trả lời của `assistant` (câu lệnh SQL / JSON), không tính loss trên phần đề bài và schema database dài dòng của `user`.

## 3. Triển khai Suy luận với `vLLM` & Benchmark Hiệu năng (`src/serving/`)

### Bước 1: Triển khai vLLM OpenAI-Compatible Server
- Cấu hình các tham số quan trọng của `vLLM`:
  - `--quantization awq` (nếu dùng bản nén 4-bit AWQ để giảm nghẽn Memory Bandwidth).
  - `--gpu-memory-utilization 0.85` (dành 85% VRAM cho trọng số và PagedAttention KV Cache).
  - `--max-model-len 4096` (giới hạn chiều dài ngữ cảnh vừa đủ với bài toán để tối đa hóa số lượng KV Blocks phục vụ đồng thời).
  - `--enable-prefix-caching` (**Cực kỳ quan trọng cho Text-to-SQL / RAG:** Tự động giữ lại KV Cache của phần System Prompt và Database Schema dùng chung giữa mọi câu hỏi, giúp giảm **TTFT** đi 50–70%!).

### Bước 2: Script Benchmark Tốc độ & Độ chính xác (`src/serving/benchmark.py`)
Khi chạy benchmark so sánh mô hình trước/sau khi Fine-tune và nén, phải đo đủ 4 chỉ số bằng `asyncio`:
1. **Execution Accuracy (EX %):** Tỷ lệ câu lệnh SQL sinh ra chạy ra kết quả khớp 100% với đáp án chuẩn.
2. **TTFT (Time To First Token - ms):** Độ trễ pha Prefill (P50, P95, P99).
3. **ITL (Inter-Token Latency - ms/token):** Độ trễ trung bình giữa các chữ ở pha Decode.
4. **System Throughput (tokens/sec):** Tổng số token sinh ra trên mỗi giây khi bắn đồng thời `concurrency = 1, 10, 30` requests.

## 4. Cổng Kiểm chứng (Verification Gate)
- Kiểm tra file dữ liệu huấn luyện bằng script validator (đảm bảo 100% JSONL hợp lệ và không rò rỉ tập test).
- Tham chiếu lý thuyết chuyên sâu về LoRA, AWQ và PagedAttention tại [05-GIAI-DOAN-4-FINETUNING-VA-MODEL-SERVING.md](../../docs/05-GIAI-DOAN-4-FINETUNING-VA-MODEL-SERVING.md).
