"""FastAPI + WebSocket backend for the chatbot.

Endpoints:
    POST /chat           — one-shot chat (returns full response)
    POST /chat/stream    — streaming chat (SSE, full pipeline)
    WS   /chat/ws        — WebSocket for bidirectional streaming
    POST /upload         — upload a text file for context injection
    DELETE /upload/{id}  — remove an uploaded file
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
from pathlib import Path
from typing import Annotated, Any

try:
    from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import os

from .config import Config, ReasoningEffort, setup_logging
from .file_upload import FileUploadCache, is_text_file, MAX_FILE_SIZE
from .guardrails import GuardrailsManager, SafetyLevel
from .memory import ConversationMemory
from .observability import MetricsCollector
from .orchestrator import Orchestrator
from .projects import ProjectStore
from .router import Complexity
from .streaming_orchestrator import StreamingOrchestrator
from .vector_memory import VectorMemory

log = logging.getLogger(__name__)


# ─── Request/Response models ─────────────────────────────────────────

if HAS_FASTAPI:
    class ChatRequest(BaseModel):
        message: str
        effort: str | None = Field(None, description="low|medium|high|max")
        force_complexity: str | None = Field(None, description="TRIVIAL|STANDARD|COMPLEX")
        file_ids: list[str] | None = Field(None, description="IDs of uploaded files to include")
        project_id: str | None = Field(None, description="Project ID for context injection")

    class ChatResponse(BaseModel):
        answer: str
        complexity: str
        thinking_mode: str
        reasoning_effort: str
        tokens_used: int
        diagnostics: dict[str, Any] = {}
else:
    ChatRequest = None  # type: ignore
    ChatResponse = None  # type: ignore


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
    # Vector memory (semantic RAG) — disabled if VECTOR_MEMORY_ENABLED=0/false
    vector_memory_enabled = os.getenv("VECTOR_MEMORY_ENABLED", "1").strip().lower()
    vector_memory = None
    if vector_memory_enabled not in {"0", "false", "no", "off"}:
        try:
            vector_memory = VectorMemory(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                persist=True,
            )
            log.info("vector_memory: initialized (dim=%d)", vector_memory.vector_size)
        except Exception as exc:
            log.warning("vector_memory: initialization failed (%s), continuing without RAG", exc)
            vector_memory = None

    memory = ConversationMemory(vector_memory=vector_memory)
    metrics = MetricsCollector(log_path=".akaradje_metrics/queries.jsonl")
    orch = Orchestrator(cfg, memory=memory)
    guardrails = GuardrailsManager()
    file_cache = FileUploadCache()
    project_store = ProjectStore()
    streaming_orch = StreamingOrchestrator(cfg, memory=memory)

    app = FastAPI(
        title="akaradje",
        description="DeepSeek V4 Pro chatbot with Opus 4.7-class scaffolding",
        version="0.3.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files — serve frontend directory (prefer Vite dist if available)
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    dist_dir = frontend_dir / "dist"
    if (dist_dir / "index.html").exists():
        serve_dir = dist_dir
    else:
        serve_dir = frontend_dir

    app.mount("/frontend", StaticFiles(directory=str(serve_dir)), name="frontend")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        # Prefer Vite dist, fall back to legacy index.html
        if (dist_dir / "index.html").exists():
            index_path = dist_dir / "index.html"
        else:
            index_path = frontend_dir / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    # ─── Endpoints ───────────────────────────────────────────────────

    def _build_context(file_ids: list[str] | None, project_id: str | None) -> str | None:
        """Build combined file + project context for system prompt injection."""
        parts: list[str] = []

        # File upload context
        if file_ids:
            fc = file_cache.build_context(file_ids)
            if fc:
                parts.append(fc)

        # Project context (instructions + files)
        if project_id:
            try:
                project = project_store.get(project_id)
                if project is not None:
                    ctx = project.build_context()
                    if ctx:
                        parts.append(ctx)
            except Exception as exc:
                log.warning("_build_context: project load failed: %s", exc)

        return "\n\n".join(parts) if parts else None

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": cfg.model}

    @app.get("/metrics")
    async def get_metrics():
        return metrics.summary()

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: Annotated[ChatRequest, Body()]):
        # Input guardrail
        input_check = guardrails.check_input(req.message)
        if input_check.level == SafetyLevel.BLOCKED:
            return ChatResponse(
                answer=guardrails.get_blocked_response(input_check),
                complexity="TRIVIAL",
                thinking_mode="disabled",
                reasoning_effort="low",
                tokens_used=0,
                diagnostics={"guardrail": input_check.level.value, "reason": input_check.reason},
            )

        force_complexity = None
        if req.force_complexity:
            force_complexity = Complexity(req.force_complexity.upper())

        force_effort = None
        if req.effort:
            force_effort = ReasoningEffort(req.effort.lower())

        # Build combined context
        combined_context = _build_context(req.file_ids, req.project_id)

        query_id = str(uuid.uuid4())[:8]
        async with metrics.track_query(req.message, query_id) as m:
            answer = await orch.ask(
                req.message,
                force_complexity=force_complexity,
                force_effort=force_effort,
                file_context=combined_context,
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

        # Output guardrail — check and sanitize final answer
        final_answer = answer.text
        output_check = guardrails.check_output(final_answer)
        if output_check.modified_content is not None:
            final_answer = output_check.modified_content

        diagnostics = answer.diagnostics
        if output_check.level != SafetyLevel.SAFE:
            diagnostics["guardrail"] = output_check.level.value
            diagnostics["guardrail_reason"] = output_check.reason

        return ChatResponse(
            answer=final_answer,
            complexity=answer.complexity.value,
            thinking_mode=answer.thinking_mode,
            reasoning_effort=answer.reasoning_effort,
            tokens_used=answer.tokens_used,
            diagnostics=diagnostics,
        )

    @app.post("/chat/stream")
    async def chat_stream(req: Annotated[ChatRequest, Body()]):
        """SSE streaming endpoint — full pipeline with tools, verifier, memory.

        Event types sent to the client:
        - {type: "progress", stage: "...", message: "..."}
        - {type: "reasoning", text: "..."} (thinking chunks)
        - {type: "content", text: "..."} (answer chunks)
        - {type: "tool_call", id: "...", name: "...", args: {...}, status: "calling"}
        - {type: "tool_result", id: "...", name: "...", result: "...", duration_ms: N, status: "done"}
        - {type: "done", complexity: "...", effort: "...", tokens: N, ...}
        - {type: "error", message: "..."}
        """
        async def event_generator():
            try:
                force_complexity = None
                if req.force_complexity:
                    force_complexity = Complexity(req.force_complexity.upper())

                force_effort = None
                if req.effort:
                    force_effort = ReasoningEffort(req.effort.lower())

                # Build combined file + project context
                combined_context = _build_context(req.file_ids, req.project_id)

                async for event in streaming_orch.stream(
                    req.message,
                    force_complexity=force_complexity,
                    force_effort=force_effort,
                    file_context=combined_context,
                ):
                    yield f"data: {json.dumps(event)}\n\n"

            except Exception as exc:
                log.error("chat_stream: unhandled error: %s", exc, exc_info=True)
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

    # ─── Artifacts ───────────────────────────────────────────────────

    @app.get("/artifacts")
    async def list_artifacts():
        """List all generated artifacts (metadata only, no content)."""
        return {
            "artifacts": [
                a.to_dict(include_content=False)
                for a in streaming_orch.artifact_store.list()
            ]
        }

    @app.get("/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        """Get a specific artifact with full content and version history."""
        artifact = streaming_orch.artifact_store.get(artifact_id)
        if artifact is None:
            return {"error": f"Artifact not found: {artifact_id}"}
        return artifact.to_dict(include_content=True)

    @app.get("/artifacts/{artifact_id}/versions/{version}")
    async def get_artifact_version(artifact_id: str, version: int):
        """Get a specific version of an artifact."""
        av = streaming_orch.artifact_store.get_version(artifact_id, version)
        if av is None:
            return {"error": f"Version {version} not found for artifact {artifact_id}"}
        return {
            "artifact_id": artifact_id,
            "version": av.version,
            "content": av.content,
            "created_at": av.created_at,
        }

    @app.delete("/artifacts/{artifact_id}")
    async def delete_artifact(artifact_id: str):
        """Remove an artifact from the store."""
        removed = streaming_orch.artifact_store.remove(artifact_id)
        if removed:
            return {"status": "removed", "artifact_id": artifact_id}
        return {"status": "not_found", "artifact_id": artifact_id}

    # ─── File Upload ─────────────────────────────────────────────────

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a text file for context injection into chat."""
        filename = file.filename or "unknown"

        if not is_text_file(filename):
            return {"error": f"Unsupported file type: {filename}"}

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return {
                "error": (
                    f"File too large ({len(content)} bytes, "
                    f"max {MAX_FILE_SIZE} bytes)"
                ),
            }

        entry = file_cache.store(filename, content, file.content_type or "")
        return {
            "file_id": entry.id,
            "name": entry.name,
            "size": entry.size,
            "preview": entry.content[:500],
        }

    @app.delete("/upload/{file_id}")
    async def delete_file(file_id: str):
        """Remove an uploaded file from the cache."""
        removed = file_cache.remove(file_id)
        if removed:
            return {"status": "removed", "file_id": file_id}
        return {"status": "not_found", "file_id": file_id}

    # ─── Projects ────────────────────────────────────────────────────

    @app.post("/projects")
    async def create_project(req: dict[str, Any] = Body(...)):
        """Create a new project."""
        name = str(req.get("name", "")).strip()
        if not name:
            return {"error": "Project name is required"}
        project = project_store.create(
            name=name,
            description=str(req.get("description", "")).strip(),
            custom_instructions=str(req.get("custom_instructions", "")).strip(),
        )
        return project.to_dict(include_files=True)

    @app.get("/projects")
    async def list_projects():
        """List all projects (metadata only)."""
        return {
            "projects": [p.to_dict(include_files=False) for p in project_store.list()]
        }

    @app.get("/projects/{project_id}")
    async def get_project(project_id: str):
        """Get a project with full details and file listing."""
        project = project_store.get(project_id)
        if project is None:
            return {"error": f"Project not found: {project_id}"}
        return project.to_dict(include_files=True)

    @app.put("/projects/{project_id}")
    async def update_project(project_id: str, req: dict[str, Any] = Body(...)):
        """Update a project's name, description, or custom instructions."""
        project = project_store.update(
            project_id,
            name=req.get("name"),
            description=req.get("description"),
            custom_instructions=req.get("custom_instructions"),
        )
        if project is None:
            return {"error": f"Project not found: {project_id}"}
        return project.to_dict(include_files=True)

    @app.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        """Delete a project and all its associated files."""
        removed = project_store.delete(project_id)
        if removed:
            return {"status": "deleted", "project_id": project_id}
        return {"status": "not_found", "project_id": project_id}

    @app.post("/projects/{project_id}/files")
    async def add_project_file(project_id: str, file: UploadFile = File(...)):
        """Upload a file and associate it with a project."""
        project = project_store.get(project_id)
        if project is None:
            return {"error": f"Project not found: {project_id}"}

        filename = file.filename or "unknown"
        content = await file.read()
        content_str = content.decode("utf-8", errors="replace")

        pf = project_store.add_file(
            project_id=project_id,
            name=filename,
            content=content_str,
            content_type=file.content_type or "",
        )
        if pf is None:
            return {"error": "Failed to add file"}
        return pf.to_dict(include_content=False)

    @app.delete("/projects/{project_id}/files/{file_id}")
    async def remove_project_file(project_id: str, file_id: str):
        """Remove a file from a project."""
        removed = project_store.remove_file(project_id, file_id)
        if removed:
            return {"status": "removed", "file_id": file_id}
        return {"status": "not_found", "file_id": file_id}

    # ─── WebSocket ───────────────────────────────────────────────────

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
