---
name: vnfin-rag-architect
description: >-
  Use this skill when building, debugging, or optimizing Phase 1 of VnFinCopilot:
  parsing Vietnamese financial PDF reports (Docling), implementing table-aware and
  parent-child chunking, indexing into Qdrant (Dense bge-m3 + Sparse BM25),
  performing Hybrid Search with Reciprocal Rank Fusion (RRF), or integrating the
  Cross-Encoder Reranker (bge-reranker-v2-m3).
---

# VnFinCopilot RAG Architect (Giai đoạn 1)

Quy trình chuẩn hóa để xây dựng và kiểm chứng hệ thống **Advanced Hybrid RAG** xử lý Báo cáo tài chính (BCTC) và Báo cáo thường niên PDF tiếng Việt trong dự án `VnFinCopilot`.

## 1. Quy tắc Kiến trúc Bắt buộc (Non-Negotiable Rules)

1. **Cấm dùng Naive PDF Reader & Fixed-Character Splitting cho Bảng biểu:**
   - Luôn sử dụng `Docling` (hoặc bộ parser nhận diện layout tương đương) để xuất bảng biểu kế toán sang chuẩn **Markdown Table** giữ nguyên cấu trúc hàng/cột.
   - Không bao giờ cắt ngang một bảng nhỏ (< 800 tokens). Nếu bảng dài bắt buộc phải tách thành nhiều chunk, **phải nhân bản dòng Table Header** và đơn vị tính (`Tỷ VNĐ`, `Triệu VNĐ`) vào đầu mỗi chunk con.
2. **Context-Enriched Header cho mọi Chunk:**
   - Trước khi đưa văn bản đi tạo vector embedding, bắt buộc gắn tiền tố ngữ cảnh chuẩn ở đầu chunk:
     `[Mã CK: {ticker} | Loại BC: {doc_type} | Kỳ: {period} | Mục: {section_title} | Đơn vị: {currency_unit}]`
3. **Cấu trúc Metadata bắt buộc trên Qdrant Payload:**
   - Mỗi point lưu vào Qdrant phải có cấu trúc Pydantic chặt chẽ gồm:
     - `chunk_id` (UUID/deterministic hash)
     - `parent_id` (ID của đoạn cha chứa ngữ cảnh rộng)
     - `ticker` (VD: `"FPT"`, `"VCB"`, `"HPG"`)
     - `year` (`int`, VD: `2025`)
     - `quarter` (`int | None`, `1..4` hoặc `None` nếu là báo cáo năm)
     - `doc_type` (`"BCTC_HOP_NHAT"`, `"BAO_CAO_THUONG_NIEN"`, `"NGHI_QUYET_DHDCD"`)
     - `is_table` (`bool`)
     - `page_number` (`int`)
     - `text` (Nội dung chunk con)
     - `parent_text` (Nội dung đoạn cha / toàn bộ bảng để gửi cho LLM)

## 2. Quy trình Thực thi Từng bước (Workflow)

### Bước 1: Ingestion & Table-Aware Chunking (`src/rag/ingestion/`)
1. Kiểm tra file PDF đầu vào tồn tại và hợp lệ.
2. Trích xuất cấu trúc tài liệu thành các khối (Sections, Paragraphs, Tables).
3. Áp dụng `ParentChildChunker`:
   - Tạo **Parent Chunks** (~800–1200 tokens hoặc trọn vẹn 1 bảng kế toán).
   - Tạo **Child Chunks** (~200–300 tokens hoặc từng nhóm dòng trong bảng kèm Header) liên kết với `parent_id`.
4. Kiểm chứng: Viết unit test xác nhận mọi chunk có `is_table=True` đều chứa ký tự phân tách bảng Markdown `|` và dòng tiêu đề cột.

### Bước 2: Hybrid Indexing trên Qdrant (`src/rag/indexing/`)
1. Tạo Qdrant Collection với cấu hình:
   - `vectors_config`: Dense vector (mặc định dùng `BAAI/bge-m3`, `size=1024`, `distance=Cosine`).
   - `sparse_vectors_config`: Sparse vector cho từ khóa chính xác / BM25.
2. Tạo Payload Index cho các trường lọc thường xuyên: `ticker` (Keyword), `year` (Integer), `quarter` (Integer), `doc_type` (Keyword).
3. Nạp dữ liệu theo lô (`batch_size=64`) kèm cơ chế retry và logging tiến độ.

### Bước 3: Two-Stage Retrieval (`src/rag/retrieval/`)
1. **Pre-filtering:** Trích xuất `ticker` và `year`/`quarter` từ câu hỏi (nếu có) để tạo `qdrant_client.http.models.Filter`.
2. **Stage 1 - Hybrid Search (RRF):**
   - Truy vấn đồng thời nhánh Dense (`top_k=25`) và nhánh Sparse BM25 (`top_k=25`).
   - Kết hợp thứ hạng bằng thuật toán **Reciprocal Rank Fusion (RRF)** với hằng số $k = 60$:
     $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{sparse}}(d)}$$
   - Gom nhóm theo `parent_id` (Deduplication) để không lấy trùng nhiều chunk con của cùng 1 bảng.
3. **Stage 2 - Cross-Encoder Reranking:**
   - Đưa Top 15–25 ứng viên từ Stage 1 qua mô hình `BAAI/bge-reranker-v2-m3`.
   - Chọn ra **Top 5 Parent Chunks** có điểm cao nhất đưa vào Prompt.

### Bước 4: Generation kèm Trích dẫn Nghiêm ngặt
- Đặt `temperature = 0.0`.
- Bắt buộc LLM trả lời kèm trích dẫn chính xác `[Nguồn: {doc_type} {ticker} {period}, Trang {page_number}]` ngay sau mỗi con số tài chính.

## 3. Cổng Kiểm chứng (Verification Gate)
Trước khi hoàn thành bất kỳ task nào thuộc Giai đoạn 1, chạy lệnh kiểm thử:
- `pytest tests/rag/ -v` để xác nhận Chunker, RRF Fusion và Metadata Filter hoạt động đúng 100%.
- Tham chiếu lý thuyết chi tiết tại [02-GIAI-DOAN-1-DATA-VA-ADVANCED-RAG.md](../../docs/02-GIAI-DOAN-1-DATA-VA-ADVANCED-RAG.md).
