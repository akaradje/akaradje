"""Progress Updates — keep the user informed during long agent loops.

Problem: Claude.ai shows "Searching...", "Reading file...", "Running code..."
in real-time. Our current implementation shows nothing until the full answer
is ready — the user stares at "thinking..." for 30+ seconds.

Solution:
- Emit typed progress events at each stage of the pipeline
- Events can be consumed by CLI (status bar) or frontend (SSE)
- Each event has: stage, message, optional metadata (tool name, iteration #)

This is purely a UX layer — it doesn't change how the agent works,
it just makes the work visible.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable, Optional, Union


class ProgressStage(str, Enum):
    ROUTING = "routing"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    TOOL_EXECUTING = "tool_executing"
    VERIFYING = "verifying"
    CORRECTING = "correcting"
    VOTING = "voting"
    COMPACTING = "compacting"
    STREAMING = "streaming"
    DONE = "done"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """A single progress update event."""
    stage: ProgressStage
    message: str
    timestamp: float = field(default_factory=time.time)
    iteration: int | None = None
    tool_name: str | None = None
    tokens_so_far: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Format as SSE data line."""
        import json
        payload = {
            "type": "progress",
            "stage": self.stage.value,
            "message": self.message,
        }
        if self.iteration is not None:
            payload["iteration"] = self.iteration
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.tokens_so_far:
            payload["tokens_so_far"] = self.tokens_so_far
        return f"data: {json.dumps(payload)}\n\n"


# Type for progress callbacks
ProgressCallback = Callable[[ProgressEvent], Optional[Awaitable[None]]]


class ProgressTracker:
    """Collects and dispatches progress events.

    Usage in the pipeline:
        tracker = ProgressTracker()
        tracker.on_progress = my_callback  # or use as async iterator

        await tracker.emit(ProgressStage.ROUTING, "Classifying query complexity")
        await tracker.emit(ProgressStage.THINKING, "Reasoning with effort=high")
        await tracker.emit(ProgressStage.TOOL_CALLING, "Calling web_search", tool_name="web_search")
    """

    def __init__(self):
        self._callbacks: list[ProgressCallback] = []
        self._events: list[ProgressEvent] = []
        self._queue: asyncio.Queue[ProgressEvent] | None = None

    def add_callback(self, cb: ProgressCallback) -> None:
        """Register a callback for progress events."""
        self._callbacks.append(cb)

    def enable_queue(self) -> None:
        """Enable async queue mode for SSE streaming."""
        self._queue = asyncio.Queue()

    async def emit(
        self,
        stage: ProgressStage,
        message: str,
        *,
        iteration: int | None = None,
        tool_name: str | None = None,
        tokens_so_far: int = 0,
        **metadata: Any,
    ) -> None:
        """Emit a progress event to all listeners."""
        event = ProgressEvent(
            stage=stage,
            message=message,
            iteration=iteration,
            tool_name=tool_name,
            tokens_so_far=tokens_so_far,
            metadata=metadata,
        )
        self._events.append(event)

        # Dispatch to callbacks
        for cb in self._callbacks:
            result = cb(event)
            if asyncio.iscoroutine(result):
                await result

        # Push to queue if enabled
        if self._queue is not None:
            await self._queue.put(event)

    async def iter_events(self) -> AsyncIterator[ProgressEvent]:
        """Async iterator over progress events (for SSE streaming).

        Must call enable_queue() first.
        """
        if self._queue is None:
            raise RuntimeError("Call enable_queue() before iter_events()")

        while True:
            event = await self._queue.get()
            yield event
            if event.stage == ProgressStage.DONE or event.stage == ProgressStage.ERROR:
                break

    @property
    def events(self) -> list[ProgressEvent]:
        """All events emitted so far."""
        return list(self._events)

    @property
    def elapsed_ms(self) -> int:
        """Total time from first to last event."""
        if len(self._events) < 2:
            return 0
        return int((self._events[-1].timestamp - self._events[0].timestamp) * 1000)
