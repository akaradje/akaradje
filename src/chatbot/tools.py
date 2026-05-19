"""Tool layer.

Each tool exposes:
    - name
    - description
    - json_schema (OpenAI function-calling format)
    - async run(args: dict) -> str

The Executor turns these into the `tools=[...]` payload at chat time.
Tools must always return a string — that's what the model will see in the
`tool` role message.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import operator as op
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the `parameters` field

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> str: ...

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Calculator — safe AST-based arithmetic evaluator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression. "
        "Supports + - * / // % ** and parentheses. "
        "Use this for any numeric computation instead of guessing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(2**32) + 17 * 4'",
            }
        },
        "required": ["expression"],
    }

    async def run(self, args: dict[str, Any]) -> str:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return "ERROR: missing 'expression'"
        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            return f"{result}"
        except Exception as exc:
            return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Python execution — subprocess with timeout
# ---------------------------------------------------------------------------


class PythonExecTool(Tool):
    """Run short Python snippets in a fresh subprocess.

    This is NOT a hardened sandbox. It blocks the obvious time bomb (infinite
    loops) via timeout, and isolates state via a fresh process, but it does
    not stop the snippet from touching the filesystem or network. For
    production use, swap this implementation for an E2B / Modal / Docker
    sandbox while keeping the same public interface.
    """

    name = "python_exec"
    description = (
        "Execute a short Python 3 snippet and return its stdout. "
        "Use this for non-trivial computation, data manipulation, "
        "or anything that benefits from running real code. "
        "The snippet runs in a fresh subprocess with a 10-second timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python 3 source code. Use print() for output.",
            }
        },
        "required": ["code"],
    }

    timeout_seconds: float = 10.0

    async def run(self, args: dict[str, Any]) -> str:
        code = str(args.get("code", ""))
        if not code.strip():
            return "ERROR: missing 'code'"

        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "snippet.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmp,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return f"ERROR: execution timed out after {self.timeout_seconds}s"
            except Exception as exc:  # pragma: no cover
                return f"ERROR: failed to launch subprocess: {exc}"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return f"EXIT={proc.returncode}\nSTDERR:\n{err}\nSTDOUT:\n{out}"
        if err:
            return f"STDOUT:\n{out}\nSTDERR:\n{err}"
        return out or "(no stdout)"


# ---------------------------------------------------------------------------
# Web search — pluggable. Default is a stub.
# ---------------------------------------------------------------------------


class WebSearchTool(Tool):
    """Stub web search tool.

    The default implementation just echoes back a placeholder. Wire this to
    Tavily, Brave, SerpAPI, or your own crawler by overriding `run()` or
    constructing with a `provider` callable.
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information. "
        "Returns a short list of result snippets. "
        "Use this when the answer depends on current facts the model may not know."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: Any | None = None):
        # `provider` is an optional callable: async (query, k) -> list[dict]
        self._provider = provider

    async def run(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 5))
        if not query:
            return "ERROR: missing 'query'"
        k = max(1, min(k, 10))

        if self._provider is None:
            return json.dumps(
                {
                    "warning": "web_search is stubbed — wire a real provider in tools.WebSearchTool",
                    "query": query,
                    "results": [],
                },
                ensure_ascii=False,
            )
        try:
            results = await self._provider(query, k)
            return json.dumps({"query": query, "results": results}, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover
            return f"ERROR: web search provider failed: {exc}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    @classmethod
    def default(cls) -> ToolRegistry:
        return cls([CalculatorTool(), PythonExecTool(), WebSearchTool()])

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    async def dispatch(self, name: str, raw_args: str | dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'"
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid JSON arguments for {name}: {exc}"
        else:
            args = raw_args
        log.info("tool: %s args=%s", name, args)
        return await tool.run(args)
