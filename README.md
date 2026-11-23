# VnFinCopilot — Enterprise Vietnamese Financial AI Platform

Hệ thống Trợ lý Phân tích Tài chính & Chứng khoán Việt Nam đa tác vụ, được xây dựng theo lộ trình 5 giai đoạn chuẩn kỹ nghệ AI Engineer:

1. **Giai đoạn 1 — Advanced Hybrid RAG:** Bóc tách BCTC PDF giữ nguyên bảng biểu (`Docling`), Parent-Child Chunking, Qdrant Hybrid Search (Dense `bge-m3` + Sparse BM25 + RRF) và Cross-Encoder Reranker (`bge-reranker-v2-m3`).
2. **Giai đoạn 2 — Evaluation & Observability:** Bộ kiểm thử Golden Dataset (`Recall@K`, `MRR@K`, `Faithfulness`, `Numerical Accuracy`), Ablation Study và Langfuse Tracing.
3. **Giai đoạn 3 — Multi-Agent & Text-to-SQL:** Đồ thị trạng thái `LangGraph`, Text-to-SQL trên PostgreSQL (Read-only Guardrails + Self-Correction Loop) và Self-RAG Hallucination Grader.
4. **Giai đoạn 4 — Fine-tuning SLM & High-Performance Serving:** QLoRA Fine-tuning (`Qwen-2.5`), lượng tử hóa 4-bit AWQ và triển khai với `vLLM` (PagedAttention, Continuous Batching, Prefix Caching).
5. **Giai đoạn 5 — Production Backend, Guardrails & LLMOps:** FastAPI SSE Streaming, Entity-Aware Semantic Cache, AI Security Guardrails (Vietnamese PII Masking, Prompt Injection Defense), Docker Compose và CI/CD.

## Tài liệu học tập & Kiến trúc
Xem trọn bộ hướng dẫn từ con số 0 trong thư mục [`docs/`](./docs/00-TONG-QUAN-VA-LO-TRINH.md).
