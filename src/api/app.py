"""FastAPI Production Backend với SSE Streaming, End-to-End Pipeline & Web Chat UI."""

import json
import time
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from cache.semantic_cache import EntityAwareSemanticCache
from guardrails.pii_scrubber import mask_vietnamese_pii
from guardrails.security import apply_output_compliance, check_input_safety
from pipeline import VnFinCopilotPipeline
from rag.indexing.embedder import DeterministicHashDenseEmbedder


class ChatRequest(BaseModel):
    """Schema yêu cầu hỏi đáp tài chính."""

    query: str = Field(..., min_length=1)
    ticker: str | None = None
    year: int | None = None


def format_sse_event(event: str, data: dict) -> str:
    """Định dạng chuẩn 1 sự kiện SSE JSON."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>VnFinCopilot — Financial AI Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 16px; background: #0f172a; color: #e2e8f0; }
    h1 { color: #38bdf8; margin-bottom: 4px; }
    .sub { color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
    .row { display: flex; gap: 8px; margin-bottom: 12px; }
    input, select, button { padding: 10px 12px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #f8fafc; font-size: 14px; }
    input { flex: 1; }
    button { background: #0284c7; border: none; cursor: pointer; font-weight: 600; }
    button:hover { background: #0369a1; }
    #status { color: #fbbf24; font-size: 13px; min-height: 20px; margin-bottom: 8px; }
    #output { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; min-height: 180px; white-space: pre-wrap; line-height: 1.6; }
    #meta { margin-top: 10px; font-size: 12px; color: #94a3b8; }
  </style>
</head>
<body>
  <h1>VnFinCopilot</h1>
  <div class="sub">Enterprise Vietnamese Financial AI — Hybrid RAG + Multi-Agent + Text-to-SQL + Guardrails</div>
  <div class="row">
    <select id="ticker"><option value="">Tự động</option><option value="FPT">FPT</option><option value="VCB">VCB</option><option value="HPG">HPG</option></select>
    <input id="query" placeholder="VD: Doanh thu thuần năm 2025 của FPT là bao nhiêu?" />
    <button onclick="sendQuery()">Gửi</button>
  </div>
  <div id="status"></div>
  <div id="output">Kết quả phân tích sẽ stream trực tiếp tại đây...</div>
  <div id="meta"></div>
  <script>
    async function sendQuery() {
      const query = document.getElementById('query').value.trim();
      const ticker = document.getElementById('ticker').value || null;
      if (!query) return;
      document.getElementById('output').textContent = '';
      document.getElementById('meta').textContent = '';
      document.getElementById('status').textContent = 'Đang kết nối...';
      const resp = await fetch('/v1/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, ticker})
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const blocks = buffer.split('\\n\\n');
        buffer = blocks.pop();
        for (const blk of blocks) {
          const lines = blk.split('\\n');
          const ev = (lines[0] || '').replace('event: ', '');
          const data = JSON.parse((lines[1] || 'data: {}').replace('data: ', ''));
          if (ev === 'status') document.getElementById('status').textContent = data.message;
          if (ev === 'token') document.getElementById('output').textContent += data.delta;
          if (ev === 'error') document.getElementById('output').textContent = 'Bị chặn: ' + data.message;
          if (ev === 'done') {
            document.getElementById('status').textContent = 'Hoàn tất!';
            document.getElementById('meta').textContent = `Latency: ${data.latency_ms}ms | Cached: ${data.cached} | Intent: ${data.intent || '-'}`;
          }
        }
      }
    }
  </script>
</body>
</html>"""


def create_app(
    cache: EntityAwareSemanticCache | None = None,
    pipeline: VnFinCopilotPipeline | None = None,
) -> FastAPI:
    """Khởi tạo ứng dụng FastAPI kết nối trọn vẹn Pipeline, Guardrails và Cache."""
    app = FastAPI(title="VnFinCopilot API", version="0.2.0")
    semantic_cache = cache or EntityAwareSemanticCache(embedder=DeterministicHashDenseEmbedder())
    copilot_pipeline = pipeline or VnFinCopilotPipeline()

    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        return _CHAT_UI_HTML

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
                yield format_sse_event("error", {"code": "GUARDRAIL_BLOCKED", "message": safety.reason})
                return

            yield format_sse_event("status", {"step": "cache_lookup", "message": "Đang kiểm tra Semantic Cache..."})
            cached = semantic_cache.get(masked_query, ticker=req.ticker, year=req.year)
            if cached is not None:
                yield format_sse_event("token", {"delta": cached})
                elapsed = int((time.perf_counter() - start) * 1000)
                yield format_sse_event("done", {"latency_ms": elapsed, "cached": True})
                return

            yield format_sse_event("status", {"step": "orchestrating", "message": "Đang chạy Multi-Agent (RAG + Text-to-SQL)..."})
            result = await copilot_pipeline.run(masked_query, ticker=req.ticker, year=req.year)
            safe_answer = apply_output_compliance(result.answer)
            semantic_cache.set(masked_query, safe_answer, ticker=req.ticker, year=req.year)

            for word in safe_answer.split(" "):
                yield format_sse_event("token", {"delta": f"{word} "})

            yield format_sse_event("citations", {"sources": result.citations})
            elapsed = int((time.perf_counter() - start) * 1000)
            yield format_sse_event(
                "done",
                {
                    "latency_ms": elapsed,
                    "cached": False,
                    "intent": result.intent,
                    "hallucination_grade": result.hallucination_grade,
                },
            )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


app = create_app()
