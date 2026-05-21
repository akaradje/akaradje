# Changelog

All notable changes to the akaradje chatbot project.

## [0.7.0] — 2026-05-22

### Added — Phase 6: Multi-step Planner

- **Planner** (`src/chatbot/planner.py`): Strategic task planner that evaluates incoming queries and generates structured execution plans using LLM Structured Output (JSON Mode with strict schema). Trivial queries skip planning; complex queries receive a 3–7 step plan.
- **PlanStep dataclasses**: Each step carries index, description, tool_hints, expected_output, and status (pending → running → done / failed). Plan tracks completion state and current active step.
- **Orchestrator integration** (`src/chatbot/orchestrator.py`): `_handle_standard` and `_handle_complex` invoke the Planner before executor runs. Generated plan is prepended to system prompt context as formatted instructions. Plan metadata included in diagnostics.
- **StreamingOrchestrator integration** (`src/chatbot/streaming_orchestrator.py`): Planner invoked during `_stream_standard`. Emits `{type: "plan_update", plan: {...}, status: "created"}` SSE event when a plan is generated. Tracks tool completion against plan steps and emits subsequent `plan_update` events on step transitions (`step_started`, `step_completed`).
- **PlanPanel** (`frontend/src/components/PlanPanel.jsx`): Interactive checklist visualization with animated status indicators — ⏳ pending (dim dot), ⚡ running (pulsing accent ring), ✅ done (green checkmark), ❌ failed (red X). Includes progress bar with gradient fill, tool hint badges, expected output previews, and a footer status line.
- **useChat hook**: Captures `plan_update` SSE events, exposes `plan` state. Auto-clears plan on chat reset.
- **ChatArea integration**: PlanPanel renders above the message list when a plan is active.

### Configuration
- Planner uses JSON Mode via `response_format` parameter (OpenAI-compatible structured output)
- `_PLAN_SCHEMA` enforces strict typing: `needs_planning`, `reasoning`, `steps[]`
- Max steps configurable via `max_steps` parameter (default 7, max 10)

## [0.6.0] — 2026-05-22

### Added — Phase 5: Vector RAG Pipeline

- **VectorMemory** (`src/chatbot/vector_memory.py`): Semantic long-term memory backed by Qdrant vector search. Uses local Qdrant instance (in-memory or persistent `.akaradje_data/qdrant/`). Embeddings generated via DeepSeek API with tenacity retry backoff (exponential: 1s → 2s → 4s, max 15s, 3 attempts). Supports `add_memory(text, metadata)` for indexing and `recall(query, k)` for cosine-similarity search.
- **ConversationMemory integration** (`src/chatbot/memory.py`): Dual-layer architecture — short-term window (last K turns verbatim) + semantic long-term recall via VectorMemory. Keyword-based recall retained as automatic fallback when VectorMemory is unavailable or fails. All turns are auto-indexed into Qdrant asynchronously (fire-and-forget with pending flush queue).
- **Configurable via env vars**:
  - `VECTOR_MEMORY_ENABLED=0` to disable (default: enabled)
  - `DEEPSEEK_EMBEDDING_MODEL` for embedding model selection (default: `deepseek-chat`)
  - `EMBEDDING_DIMS` for vector size override (auto-detected from first embedding otherwise)
- **Async memory pipeline**: `add_user`/`add_assistant` are now async — they schedule background vector indexing via `asyncio.ensure_future` so conversation flow is never blocked. `recall()` and `build_prior_messages()` are also async, performing semantic search against Qdrant.
- **Qdrant proxy** (`vite.config.js`): Added `/projects` proxy (carried forward from Phase 4).

### Changed
- `ConversationMemory.add_user()` / `add_assistant()`: Now async. Synchronous fallbacks `add_user_sync()` / `add_assistant_sync()` provided for non-async contexts.
- `ConversationMemory.recall()`: Now async — uses VectorMemory semantic search when available, keyword fallback otherwise.
- `ConversationMemory.build_prior_messages()`: Now async.
- `Orchestrator.ask()`: `await` on all memory calls.
- `StreamingOrchestrator.stream()`: `await` on all memory calls, `_build_prior()` made async.
- `api.py`: VectorMemory instantiated and wired into ConversationMemory. Controlled by `VECTOR_MEMORY_ENABLED` env var with graceful degradation on init failure.
- `pyproject.toml`: Added `qdrant-client>=1.9.0` to core dependencies.

## [0.5.0] — 2026-05-22

### Added — Phase 4: Projects System

- **ProjectStore** (`src/chatbot/projects.py`): SQLite-backed persistent store for projects with full CRUD. Database at `.akaradje_data/projects.db`. Each project stores name, description, custom instructions, and associated files. Files are injected as reference blocks in the system prompt.
- **Project CRUD endpoints** (`src/chatbot/api.py`):
  - `POST /projects` — create project
  - `GET /projects` — list all projects
  - `GET /projects/{id}` — get project with file listing
  - `PUT /projects/{id}` — update project metadata
  - `DELETE /projects/{id}` — delete project and files
  - `POST /projects/{id}/files` — upload file to project
  - `DELETE /projects/{id}/files/{file_id}` — remove file from project
- **Context injection pipeline**: `ChatRequest.project_id` flows through orchestrator → executor → voter, prepending project custom instructions and file contents as formatted reference blocks into every system prompt.
- **Sidebar** (`frontend/src/components/Sidebar.jsx`): Glassmorphism sidebar with collapsible design, project selector dropdown, new project creation, session list, and access to project settings.
- **ProjectSettings** (`frontend/src/components/ProjectSettings.jsx`): Modal dialog for editing project name, description, custom instructions, managing project files (upload/remove), and deleting projects with confirmation.
- **useProjects hook** (`frontend/src/hooks/useProjects.js`): React state management for project list, active project switching, and all CRUD operations.
- **Header project indicator**: Active project name displayed as a pill in the header bar.

### Changed
- `Orchestrator.ask()`: Now accepts `file_context` parameter and passes it to all paths (TRIVIAL, STANDARD, COMPLEX)
- `Executor.run()`: Accepts `file_context` and prepends it to the system prompt
- `Voter.vote()`: Accepts `file_context` and passes it to each candidate executor
- `useChat` hook: Accepts and forwards `projectId` to the streaming endpoint
- `streamChat()` API function: Sends `project_id` in request body
- `App.jsx`: Full sidebar + project state integration, project ID passed to chat
- `vite.config.js`: Added `/projects` proxy

## [0.4.0] — 2026-05-22

### Added — Phase 3: Artifacts System

- **ArtifactStore** (`src/chatbot/artifacts.py`): In-memory store for generated artifacts with full version history. Supports create, update, get, list, and remove operations. Each artifact tracks title, type, language, content, and version timeline.
- **ArtifactTool** (`src/chatbot/artifacts.py`): LLM-invocable tool (`generate_artifact`) for explicit rich content generation. Parameters: title, type (code|html|markdown|svg|mermaid), language, content. Returns artifact metadata including ID for frontend linking.
- **StreamingOrchestrator integration**: Artifact SSE events (`{type: "artifact", id, title, artifact_type, ...}`) are emitted when the `generate_artifact` tool is invoked, triggering the frontend split-panel.
- **REST endpoints** (`src/chatbot/api.py`):
  - `GET /artifacts` — list all artifacts (metadata only)
  - `GET /artifacts/{id}` — get artifact with full content and version history
  - `GET /artifacts/{id}/versions/{version}` — get a specific version
  - `DELETE /artifacts/{id}` — remove an artifact
- **ArtifactPanel** (`frontend/src/components/ArtifactPanel.jsx`): React split-panel with tab navigation and type-specific renderers:
  - Code: syntax-highlighted via highlight.js with line numbers and Copy button
  - HTML: sandboxed iframe live preview
  - Markdown: rendered display via react-markdown
  - SVG: inline vector graphic rendering
  - Mermaid: diagram definition display
- **Glassmorphism styling**: Full `.glass-panel` design with backdrop blur, translucent backgrounds, and dark theme integration
- **Version navigation**: Tab-based version switching with visual indicator

### Changed
- `ToolRegistry`: Added `register()` method for dynamic tool registration
- `StreamingOrchestrator`: Auto-registers ArtifactTool on construction
- `useChat` hook: Exposes `artifactIds` state array populated from SSE artifact events
- `App.jsx`: Split-panel layout with auto-revealing ArtifactPanel on artifact generation
- `vite.config.js`: Added `/artifacts` proxy to backend

## [0.3.0] — 2026-05-21

### Added
- Streaming pipeline with full orchestration (routing → ReAct → verifier)
- Tool UX improvements: calculator, Python exec, web search, URL fetch, file read, shell exec
- File upload with text extraction and context injection
- FastAPI backend with REST + WebSocket endpoints

## [0.2.0] — 2026-05-20

### Added
- Tier 1-3 improvements: caching, streaming, compaction
- Sub-agents with delegation
- Self-correction loop
- Observability with metrics collection

## [0.1.0] — 2026-05-19

### Added
- Initial project scaffold with FastAPI backend
- Basic chat endpoint
- DeepSeek V4 Pro integration
- React frontend with Tailwind CSS
