# Project Status — akaradje

## Phase Completion

| Phase | Feature | Status | Completed |
|-------|---------|--------|-----------|
| 1 | Foundation (FastAPI, chat endpoint, DeepSeek integration) | ✅ Complete | 2026-05-19 |
| 2 | Tool UX & File Upload (calculator, Python, web search, file upload, streaming) | ✅ Complete | 2026-05-21 |
| 3 | **Artifacts System** (ArtifactStore, ArtifactTool, split-panel UI, type renderers) | ✅ Complete | 2026-05-22 |
| 4 | **Projects System** (ProjectStore, SQLite persistence, sidebar, context injection) | ✅ Complete | 2026-05-22 |
| 5 | **Vector RAG Pipeline** (VectorMemory, Qdrant, semantic search, auto-indexing) | ✅ Complete | 2026-05-22 |
| 6 | **Multi-step Planner** (Planner, PlanStep, structured output, plan visualization) | ✅ Complete | 2026-05-22 |
| 7 | Security & Performance (guardrails, caching, compaction) | ✅ Complete | 2026-05-21 |
| 8 | Evaluation & Testing (smoke tests, E2E stress tests, eval runner) | ✅ Complete | 2026-05-22 |

## Current Architecture

```
akaradje/
├── src/chatbot/
│   ├── api.py                    # FastAPI server + SSE + WS + REST endpoints
│   ├── artifacts.py              # ArtifactStore + ArtifactTool (Phase 3)
│   ├── projects.py               # ProjectStore + Project + ProjectFile (Phase 4)
│   ├── vector_memory.py          # VectorMemory + Qdrant + embeddings (Phase 5)
│   ├── planner.py                # Multi-step Planner + PlanStep (Phase 6)
│   ├── streaming_orchestrator.py # Full pipeline orchestrator with SSE
│   ├── orchestrator.py           # Non-streaming orchestrator (COMPLEX path)
│   ├── streaming.py              # Raw LLM streaming client
│   ├── tools.py                  # Tool base + registry + all tools
│   ├── router.py                 # Complexity classifier
│   ├── executor.py               # ReAct executor
│   ├── verifier.py               # Output verification (Best-of-N)
│   ├── guardrails.py             # Input/output safety guardrails
│   ├── memory.py                 # Conversation memory
│   ├── compaction.py             # Context compaction
│   ├── cache.py                  # Response caching
│   ├── self_correction.py        # Self-correction loop
│   ├── subagent.py               # Sub-agent delegation
│   ├── file_upload.py            # File upload handling
│   ├── observability.py          # Metrics & monitoring
│   ├── eval_runner.py            # Evaluation harness
│   └── ...                       # Config, CLI, etc.
├── frontend/
│   └── src/
│       ├── App.jsx               # Main app with sidebar + split-panel layout
│       ├── components/
│       │   ├── ArtifactPanel.jsx  # Artifact display panel (Phase 3)
│       │   ├── ChatArea.jsx       # Chat message list
│       │   ├── ChatMessage.jsx    # Individual message bubble
│       │   ├── Header.jsx         # App header with project indicator
│       │   ├── Sidebar.jsx        # Project selector + session list (Phase 4)
│       │   ├── ProjectSettings.jsx # Project edit modal (Phase 4)
│       │   ├── ChatInput.jsx      # Message input + file upload
│       │   ├── StatusBar.jsx      # Status bar with ping indicator
│       │   ├── MarkdownRenderer.jsx
│       │   ├── ThinkingBlock.jsx
│       │   └── ToolCallPanel.jsx
│       ├── hooks/
│       │   ├── useChat.js         # Chat state + SSE streaming
│       │   ├── useFileUpload.js   # File upload state
│       │   └── useProjects.js     # Project state management (Phase 4)
│       └── utils/
│           └── api.js             # API client functions
└── tests/
    ├── test_smoke.py             # Basic smoke tests
    └── test_e2e_stress.py        # End-to-end stress tests
```

## Active Branch
- `main` — current development

## Recent Commits
- `25ee829` Merge PR #4 — Opus 4.7 parity gaps
- `e66fe1e` feat: close Opus 4.7 parity gaps
- `5fc179b` Merge PR #3 — tools, frontend tiers
- `fd0dcef` feat: wire real web search, add tools, build frontend
- `aec1f5f` feat: Tier 1-3 improvements
