"""Tool layer for DeepSeek V4 Pro function-calling.

DeepSeek V4 uses the standard OpenAI function-calling format. When thinking
mode is enabled, the model REASONS about which tool to call (visible in
reasoning_content) before emitting the tool_call. This is "thinking with
tools" — the same behavior as Opus 4.7's interleaved thinking.

Each tool exposes:
    - name, description, parameters (JSON schema)
    - async run(args: dict) -> str
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import operator as op
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

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


# ═══════════════════════════════════════════════════════════════════════════════
# Calculator — safe AST-based arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod, ast.Pow: op.pow,
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
        "Supports + - * / // % ** and parentheses."
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


# ═══════════════════════════════════════════════════════════════════════════════
# Python execution — subprocess sandbox
# ═══════════════════════════════════════════════════════════════════════════════

class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a short Python 3 snippet and return stdout. "
        "Use for non-trivial computation. 10-second timeout."
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
                    sys.executable, str(script_path),
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
                    return f"ERROR: timed out after {self.timeout_seconds}s"
            except Exception as exc:
                return f"ERROR: subprocess failed: {exc}"
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return f"EXIT={proc.returncode}\nSTDERR:\n{err}\nSTDOUT:\n{out}"
        if err:
            return f"STDOUT:\n{out}\nSTDERR:\n{err}"
        return out or "(no stdout)"


# ═══════════════════════════════════════════════════════════════════════════════
# Web search — stub (wire to Tavily/Brave/SerpAPI)
# ═══════════════════════════════════════════════════════════════════════════════

class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for up-to-date information. "
        "Returns result snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, provider: Any | None = None):
        self._provider = provider

    async def run(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 5))
        if not query:
            return "ERROR: missing 'query'"
        k = max(1, min(k, 10))
        if self._provider is None:
            return json.dumps({
                "warning": "web_search is stubbed — wire a real provider",
                "query": query, "results": [],
            }, ensure_ascii=False)
        try:
            results = await self._provider(query, k)
            return json.dumps({"query": query, "results": results}, ensure_ascii=False)
        except Exception as exc:
            return f"ERROR: web search failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

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
                return f"ERROR: invalid JSON for {name}: {exc}"
        else:
            args = raw_args
        log.info("tool: %s args=%s", name, args)
        return await tool.run(args)
