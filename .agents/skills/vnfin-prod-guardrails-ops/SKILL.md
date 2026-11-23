---
name: vnfin-prod-guardrails-ops
description: >-
  Use this skill when building or auditing Phase 5 of VnFinCopilot: implementing
  FastAPI Server-Sent Events (SSE) real-time streaming, Semantic Caching with
  Qdrant/Redis, Input/Output AI Guardrails (Vietnamese PII masking for phone,
  email, CCCD, bank accounts; prompt injection defense; financial compliance),
  Docker Compose orchestration, or CI/CD test gates.
---

# VnFinCopilot Production Backend, Guardrails & LLMOps (Giai đoạn 5)

Quy trình đóng gói hệ thống AI thành dịch vụ Production-grade với **FastAPI SSE Streaming**, **Semantic Caching**, **Bảo mật AI (Guardrails)** và **Docker/CI-CD**.

## 1. Kiến trúc FastAPI & Server-Sent Events (SSE) (`src/api/`)

1. **Chuẩn Sự kiện SSE (`text/event-stream`):**
   - Không trả về chuỗi văn bản thô không định dạng. Mọi sự kiện đẩy qua SSE phải là JSON hợp lệ theo cấu trúc:
     - `event: status` -> `data: {"step": "retrieving_sql", "message": "Đang truy vấn dữ liệu tài chính..."}`
     - `event: token` -> `data: {"delta": "Doanh "}`
     - `event: citations` -> `data: {"sources": [{"doc": "BCTC_FPT_2025.pdf", "page": 14}]}`
     - `event: done` -> `data: {"latency_ms": 1420, "cached": false}`
     - `event: error` -> `data: {"code": "GUARDRAIL_BLOCKED", "message": "..."}`
2. **Quản lý Kết nối & Timeout:**
   - Sử dụng `httpx.AsyncClient` và `async def` xuyên suốt toàn bộ các tầng (không bao giờ gọi hàm đồng bộ `time.sleep` hay `requests.post` làm block Event Loop của FastAPI).
   - Thiết lập timeout rõ ràng cho từng tầng và tự động hủy tác vụ (cancel task) nếu `await request.is_disconnected()` là `True`.

## 2. Semantic Caching (`src/cache/semantic_cache.py`)

1. **Nguyên tắc Chống Trả Nhầm Cache Giữa Các Mã Chứng Khoán (Entity-Aware Semantic Cache):**
   - Hai câu hỏi *"Doanh thu quý 4/2025 của FPT là bao nhiêu?"* và *"Doanh thu quý 4/2025 của VCB là bao nhiêu?"* chỉ khác nhau đúng 1 từ viết tắt nên Cosine Similarity của vector embedding có thể cao tới `0.96`!
   - **Quy tắc bắt buộc:** Khi tra cứu Semantic Cache trên Qdrant/Redis, **phải kết hợp Exact Filter trên Metadata (`ticker`, `year`, `quarter`) + Vector Similarity $\ge 0.95$**. Tuyệt đối không chỉ dùng mỗi ngưỡng Cosine Similarity thuần túy!
2. **Thiết lập TTL (Time-To-Live):**
   - Câu hỏi có liên quan đến giá cổ phiếu/chỉ số biến động theo ngày: `TTL = 300 giây` (5 phút).
   - Câu hỏi chỉ tra cứu Báo cáo tài chính lịch sử đã kiểm toán: `TTL = 86400 giây` (24 giờ).

## 3. Hệ thống Phòng thủ Bảo mật AI (`src/guardrails/`)

### Bước 1: Input Guardrails
1. **Vietnamese PII Masking (`src/guardrails/pii_scrubber.py`):**
   - Tự động quét và ẩn danh hóa trước khi gửi lên LLM hoặc ghi ra file log:
     - Số điện thoại Việt Nam (`03x, 05x, 07x, 08x, 09x, +84...`) -> `<VN_PHONE>`
     - Số CCCD 12 chữ số -> `<VN_CCCD>`
     - Địa chỉ Email -> `<EMAIL>`
     - API Key / Bearer Token -> `<SECRET_TOKEN>`
2. **Prompt Injection & Jailbreak Detector:**
   - Phát hiện và chặn các mẫu câu điều khiển hệ thống (*"Bỏ qua mọi hướng dẫn trước đó"*, *"Ignore previous instructions"*, *"In ra system prompt"*, *"DROP TABLE"*).

### Bước 2: Output Guardrails (Financial Compliance)
- Đảm bảo câu trả lời không chứa các cụm từ cam kết lợi nhuận hoặc xúi giục mua/bán vi phạm quy định tư vấn chứng khoán, và luôn đính kèm dòng miễn trừ trách nhiệm chuẩn hoặc trích dẫn nguồn rõ ràng.

## 4. Cổng Kiểm chứng (Verification Gate)
- Chạy `pytest tests/api/ tests/guardrails/ tests/cache/ -v` để kiểm tra:
  1. Semantic Cache không bao giờ trả nhầm kết quả của mã `FPT` khi người dùng hỏi mã `VCB`.
  2. Bộ lọc PII che được 100% số điện thoại, CCCD và Email trong bộ test.
  3. Endpoint SSE trả về đúng định dạng luồng sự kiện `text/event-stream`.
- Tham chiếu lý thuyết tại [06-GIAI-DOAN-5-PRODUCTION-GUARDRAILS-VA-LLMOPS.md](../../docs/06-GIAI-DOAN-5-PRODUCTION-GUARDRAILS-VA-LLMOPS.md).
