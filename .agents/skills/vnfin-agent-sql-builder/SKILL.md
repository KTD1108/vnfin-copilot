---
name: vnfin-agent-sql-builder
description: >-
  Use this skill when designing or implementing Phase 3 of VnFinCopilot: building
  the LangGraph multi-agent state machine, creating the Vietnamese financial
  Text-to-SQL agent on PostgreSQL with read-only guardrails and self-correction
  loops, combining structured SQL data with unstructured PDF RAG, or implementing
  the Self-RAG Hallucination/Fact Grader node.
---

# VnFinCopilot Multi-Agent & Text-to-SQL Builder (Giai đoạn 3)

Quy trình thiết kế và triển khai hệ thống **Multi-Agent State Graph** (`LangGraph`) kết hợp truy vấn cơ sở dữ liệu định lượng (**PostgreSQL Text-to-SQL**) và tài liệu định tính (**Qdrant Hybrid RAG**).

## 1. Quy tắc Thiết kế Đồ thị Trạng thái (State Graph Rules)

1. **Quản lý State tường minh bằng TypedDict / Pydantic (`src/agents/state.py`):**
   - `user_query`: Câu hỏi gốc của người dùng.
   - `intent`: Phân loại bởi Router (`"RAG_ONLY"`, `"SQL_ONLY"`, `"HYBRID_BOTH"`).
   - `sub_queries`: Danh sách các câu hỏi con đã được phân rã.
   - `sql_query`: Câu lệnh PostgreSQL được sinh ra.
   - `sql_result`: Kết quả truy vấn từ PostgreSQL (dạng danh sách dict/Markdown table).
   - `sql_error`: Thông báo lỗi từ PostgreSQL (nếu có).
   - `sql_retry_count`: Số lần đã thử tự sửa lỗi SQL (tối đa `MAX_SQL_RETRIES = 3`).
   - `retrieved_chunks`: Danh sách các đoạn tài liệu PDF lấy từ Hybrid RAG.
   - `draft_answer`: Câu trả lời nháp của Synthesizer.
   - `hallucination_grade`: Kết quả kiểm chứng số liệu (`"PASSED"`, `"HALLUCINATED"`, `"INCOMPLETE"`).
   - `rewrite_count`: Số lần viết lại câu hỏi (tối đa `MAX_REWRITES = 2`).
2. **Chống Vòng lặp Vô tận (Anti-Infinite Loop):**
   - Mọi cạnh rẽ nhánh quay lui (Conditional Edges cho SQL Self-Correction hoặc Query Rewrite) **bắt buộc phải kiểm tra biến đếm** (`sql_retry_count < 3`, `rewrite_count < 2`). Nếu vượt ngưỡng, chuyển thẳng sang trạng thái trả về thông báo lỗi minh bạch kèm dữ liệu từng phần đã thu thập được.
3. **Thực thi Song song (`asyncio.gather`):**
   - Khi `intent == "HYBRID_BOTH"`, bắt buộc chạy nhánh `RAG Agent` và `Text-to-SQL Agent` bất đồng bộ song song để giảm một nửa tổng thời gian chờ.

## 2. Quy chuẩn An toàn & Chính xác cho Text-to-SQL (`src/agents/sql_agent.py`)

### Bước 1: SQL Security Guardrail (Chốt chặn trước khi thực thi)
1. **Kiểm tra AST / Từ khóa cấm:** Chặn tuyệt đối mọi câu lệnh chứa `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `GRANT`, `EXEC`, hoặc nhiều câu lệnh ghép bằng dấu `;`. Chỉ cho phép câu lệnh bắt đầu bằng `SELECT` hoặc `WITH ... SELECT`.
2. **Giới hạn Tài nguyên:**
   - Luôn thực thi qua kết nối Read-only với `statement_timeout = 3000` (3 giây).
   - Tự động gắn `LIMIT 100` nếu câu lệnh SQL chưa có mệnh đề `LIMIT` để chống tràn bộ nhớ context.

### Bước 2: Vòng lặp Tự sửa lỗi (Self-Correction Loop)
- Nếu PostgreSQL ném ngoại lệ (VD: sai tên cột, lỗi ép kiểu ngày tháng, chia cho 0):
  1. Bắt ngoại lệ cụ thể (`asyncpg.PostgresError` / `sqlalchemy.exc.SQLAlchemyError`).
  2. Lưu nguyên văn thông báo lỗi vào `state["sql_error"]` và tăng `sql_retry_count += 1`.
  3. Gửi lại `(Schema bảng + Câu SQL lỗi + Thông báo lỗi PostgreSQL)` vào Node `sql_corrector` để sinh lại câu SQL đúng.

### Bước 3: Quy tắc Tính toán Số học
- **Cấm để LLM tự nhẩm tính cộng/trừ/nhân/chia** trên bảng số liệu lớn.
- Mọi phép tính (Tăng trưởng YoY/QoQ, Biên lợi nhuận, ROE trung bình, CAGR) phải được thực hiện trực tiếp bằng biểu thức toán học trong câu lệnh SQL (`ROUND(..., 2)`, `NULLIF(..., 0)`) hoặc thông qua một **Python Calculator Tool** an toàn.

## 3. Node Kiểm chứng Số liệu (Fact & Hallucination Grader)
- Trước khi xuất câu trả lời cuối cùng, Node `fact_grader` đối chiếu từng con số trong `draft_answer` với `sql_result` và `retrieved_chunks`.
- Nếu phát hiện con số bị lệch hoặc suy diễn không có căn cứ, đánh dấu `"HALLUCINATED"` và yêu cầu Node `synthesizer` viết lại nghiêm ngặt theo đúng dữ liệu nguồn.

## 4. Cổng Kiểm chứng (Verification Gate)
- Chạy `pytest tests/agents/ -v` để kiểm tra:
  1. SQL Guardrail chặn được 100% các câu lệnh `DROP/DELETE/UPDATE` và SQL Injection.
  2. Vòng lặp SQL Self-Correction tự sửa được lỗi sai tên cột giả lập và dừng đúng ở lần thử thứ 3 nếu vẫn lỗi.
- Tham chiếu lý thuyết tại [04-GIAI-DOAN-3-AGENTIC-WORKFLOW-VA-TEXT-TO-SQL.md](../../docs/04-GIAI-DOAN-3-AGENTIC-WORKFLOW-VA-TEXT-TO-SQL.md).
