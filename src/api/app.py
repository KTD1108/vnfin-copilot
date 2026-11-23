"""FastAPI Server-Sent Events (SSE) API cho VnFinCopilot."""

import json
import time
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cache.semantic_cache import EntityAwareSemanticCache
from guardrails.pii_scrubber import mask_vietnamese_pii
from guardrails.security import apply_output_compliance, check_input_safety
from rag.indexing.embedder import DeterministicHashDenseEmbedder


class ChatRequest(BaseModel):
    """Schema yêu cầu hỏi đáp tài chính."""

    query: str = Field(..., min_length=1)
    ticker: str | None = None
    year: int | None = None


def format_sse_event(event: str, data: dict) -> str:
    """Định dạng chuẩn 1 sự kiện SSE JSON."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(cache: EntityAwareSemanticCache | None = None) -> FastAPI:
    """Khởi tạo ứng dụng FastAPI với Guardrails, Semantic Cache và SSE Streaming."""
    app = FastAPI(title="VnFinCopilot API", version="0.1.0")
    semantic_cache = cache or EntityAwareSemanticCache(embedder=DeterministicHashDenseEmbedder())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            start = time.perf_counter()
            masked_query = mask_vietnamese_pii(req.query)
            safety = check_input_safety(masked_query)

            if not safety.allowed:
                yield format_sse_event(
                    "error",
                    {"code": "GUARDRAIL_BLOCKED", "message": safety.reason},
                )
                return

            yield format_sse_event(
                "status",
                {"step": "cache_lookup", "message": "Đang kiểm tra Semantic Cache..."},
            )
            cached = semantic_cache.get(masked_query, ticker=req.ticker, year=req.year)
            if cached is not None:
                yield format_sse_event("token", {"delta": cached})
                elapsed = int((time.perf_counter() - start) * 1000)
                yield format_sse_event("done", {"latency_ms": elapsed, "cached": True})
                return

            yield format_sse_event(
                "status",
                {"step": "retrieving", "message": "Đang truy xuất dữ liệu tài chính..."},
            )
            raw_answer = f"Kết quả phân tích cho truy vấn: {masked_query}"
            safe_answer = apply_output_compliance(raw_answer)
            semantic_cache.set(masked_query, safe_answer, ticker=req.ticker, year=req.year)

            for word in safe_answer.split(" "):
                yield format_sse_event("token", {"delta": f"{word} "})

            yield format_sse_event("citations", {"sources": []})
            elapsed = int((time.perf_counter() - start) * 1000)
            yield format_sse_event("done", {"latency_ms": elapsed, "cached": False})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


app = create_app()
