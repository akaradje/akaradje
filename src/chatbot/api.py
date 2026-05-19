"""FastAPI + WebSocket backend for the chatbot.

Endpoints:
    POST /chat           — one-shot chat (returns full response)
    POST /chat/stream    — streaming chat (SSE)
    WS   /chat/ws        — WebSocket for bidirectional streaming
    GET  /health         — health check
    GET  /metrics        — observability metrics summary

This is a thin HTTP layer over the Orchestrator. All intelligence
lives in the orchestrator; this just exposes it over HTTP/WS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from .config import Config, ReasoningEffort, setup_logging
from .memory import ConversationMemory
from .observability import MetricsCollector
from .orchestrator import Orchestrator
from .router import Complexity

log = logging.getLogger(__name__)


def create_app() -> Any:
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Run: pip install fastapi uvicorn"
        )

    cfg = Config.from_env()
    setup_logging(cfg.log_level)
    cfg.validate()

    # Shared state
    memory = ConversationMemory()
    metrics = MetricsCollector(log_path=".akaradje_metrics/queries.jsonl")
    orch = Orchestrator(cfg, memory=memory)

    app = FastAPI(
        title="akaradje",
        description="DeepSeek V4 Pro chatbot with Opus 4.7-class scaffolding",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Request/Response models ─────────────────────────────────────

    class ChatRequest(BaseModel):
        message: str
        effort: str | None = Field(None, description="low|medium|high|max")
        force_complexity: str | None = Field(None, description="TRIVIAL|STANDARD|COMPLEX")

    class ChatResponse(BaseModel):
        answer: str
        complexity: str
        thinking_mode: str
        reasoning_effort: str
        tokens_used: int
        diagnostics: dict[str, Any] = {}

    # ─── Endpoints ───────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": cfg.model}

    @app.get("/metrics")
    async def get_metrics():
        return metrics.summary()

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        force_complexity = None
        if req.force_complexity:
            force_complexity = Complexity(req.force_complexity.upper())

        force_effort = None
        if req.effort:
            force_effort = ReasoningEffort(req.effort.lower())

        query_id = str(uuid.uuid4())[:8]
        async with metrics.track_query(req.message, query_id) as m:
            answer = await orch.ask(
                req.message,
                force_complexity=force_complexity,
                force_effort=force_effort,
            )
            m.complexity = answer.complexity.value
            m.thinking_mode = answer.thinking_mode
            m.reasoning_effort = answer.reasoning_effort
            m.total_tokens = answer.tokens_used
            m.iterations = answer.iterations
            m.tool_calls = answer.tool_calls
            if answer.verdict:
                m.verifier_score = answer.verdict.overall
                m.verifier_passed = answer.verdict.passed

        return ChatResponse(
            answer=answer.text,
            complexity=answer.complexity.value,
            thinking_mode=answer.thinking_mode,
            reasoning_effort=answer.reasoning_effort,
            tokens_used=answer.tokens_used,
            diagnostics=answer.diagnostics,
        )

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """True SSE streaming endpoint — token-by-token from DeepSeek API.

        Event types sent to the client:
        - {type: "progress", stage: "...", message: "..."}
        - {type: "reasoning", text: "..."} (thinking chunks, if show_thinking)
        - {type: "content", text: "..."} (answer chunks)
        - {type: "tool_call", name: "...", status: "calling|done"}
        - {type: "done", complexity: "...", tokens: N}
        - {type: "error", message: "..."}
        """
        from .streaming import StreamingClient, ChunkType
        from .router import Router

        async def event_generator():
            try:
                force_complexity = None
                if req.force_complexity:
                    force_complexity = Complexity(req.force_complexity.upper())
                force_effort = None
                if req.effort:
                    force_effort = ReasoningEffort(req.effort.lower())

                # Phase 1: Route
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'routing', 'message': 'Classifying query...'})}\n\n"

                router = Router(orch._client, cfg)
                complexity = force_complexity or await router.classify(req.message)
                thinking_mode, effort = router.get_thinking_config(complexity)
                if force_effort:
                    effort = force_effort
                    from .config import ThinkingMode
                    if force_effort.value in ("high", "max"):
                        thinking_mode = ThinkingMode.ENABLED

                yield f"data: {json.dumps({'type': 'progress', 'stage': 'thinking', 'message': f'Thinking with effort={effort.value}...'})}\n\n"

                # Phase 2: Stream the response
                streaming_client = StreamingClient(cfg)
                messages = [
                    {"role": "system", "content": "You are a helpful, careful assistant. Think step by step when needed."},
                    {"role": "user", "content": req.message},
                ]

                total_content = ""
                total_reasoning = ""

                async for chunk in streaming_client.stream(
                    messages=messages,
                    thinking=thinking_mode,
                    reasoning_effort=effort,
                    max_tokens=8192,
                ):
                    if chunk.type == ChunkType.REASONING:
                        total_reasoning += chunk.text
                        yield f"data: {json.dumps({'type': 'reasoning', 'text': chunk.text})}\n\n"
                    elif chunk.type == ChunkType.CONTENT:
                        total_content += chunk.text
                        yield f"data: {json.dumps({'type': 'content', 'text': chunk.text})}\n\n"
                    elif chunk.type == ChunkType.TOOL_CALL:
                        if chunk.tool_name:
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': chunk.tool_name, 'status': 'calling'})}\n\n"
                    elif chunk.type == ChunkType.DONE:
                        pass  # handled below
                    elif chunk.type == ChunkType.ERROR:
                        yield f"data: {json.dumps({'type': 'error', 'message': chunk.text})}\n\n"

                # Final event
                yield f"data: {json.dumps({'type': 'done', 'complexity': complexity.value, 'effort': effort.value, 'tokens': len(total_content) // 3})}\n\n"

            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.websocket("/chat/ws")
    async def websocket_chat(ws: WebSocket):
        """WebSocket endpoint for bidirectional streaming."""
        await ws.accept()
        try:
            while True:
                data = await ws.receive_json()
                message = data.get("message", "")
                effort = data.get("effort")

                if not message:
                    await ws.send_json({"error": "empty message"})
                    continue

                force_effort = ReasoningEffort(effort) if effort else None

                answer = await orch.ask(message, force_effort=force_effort)

                await ws.send_json({
                    "type": "answer",
                    "text": answer.text,
                    "complexity": answer.complexity.value,
                    "thinking_mode": answer.thinking_mode,
                    "reasoning_effort": answer.reasoning_effort,
                    "tokens_used": answer.tokens_used,
                })

        except WebSocketDisconnect:
            log.info("ws: client disconnected")
        except Exception as exc:
            log.error("ws: error: %s", exc)
            try:
                await ws.send_json({"error": str(exc)})
            except Exception:
                pass

    return app


# Entry point for: uvicorn chatbot.api:app
if HAS_FASTAPI:
    app = create_app()
