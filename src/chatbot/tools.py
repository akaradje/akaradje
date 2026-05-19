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
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
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
# Web Search — DuckDuckGo (free, no API key) + optional Tavily
# ═══════════════════════════════════════════════════════════════════════════════

async def _duckduckgo_search(query: str, k: int) -> list[dict[str, str]]:
    """Search via DuckDuckGo HTML (no API key required).

    Uses the DuckDuckGo Lite endpoint which returns plain HTML.
    We parse it with regex (no external deps like BeautifulSoup needed).
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"

    def _fetch() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")

    loop = asyncio.get_event_loop()
    html = await loop.run_in_executor(None, _fetch)

    # Parse results from DuckDuckGo Lite HTML
    results: list[dict[str, str]] = []

    # Extract links and snippets
    # DDG Lite format: <a class="result-link" href="...">title</a> ... <td class="result-snippet">...</td>
    link_pattern = re.compile(
        r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title) in enumerate(links[:k]):
        title_clean = re.sub(r"<[^>]+>", "", title).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
        if href and title_clean:
            results.append({
                "title": title_clean[:200],
                "url": href,
                "snippet": snippet[:300],
            })

    return results[:k]


async def _tavily_search(query: str, k: int) -> list[dict[str, str]]:
    """Search via Tavily API (requires TAVILY_API_KEY env var)."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": k,
        "include_answer": False,
    }).encode("utf-8")

    def _fetch() -> str:
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _fetch)
    data = json.loads(raw)

    results: list[dict[str, str]] = []
    for item in data.get("results", [])[:k]:
        results.append({
            "title": item.get("title", "")[:200],
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:300],
        })
    return results


class WebSearchTool(Tool):
    """Web search tool with real providers.

    Provider priority:
    1. Custom provider (if passed to constructor)
    2. Tavily (if TAVILY_API_KEY is set)
    3. DuckDuckGo Lite (free fallback, always available)
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information. "
        "Returns title, URL, and snippet for each result. "
        "Use this when the answer depends on current facts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results (default 5, max 10)", "default": 5},
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

        try:
            if self._provider is not None:
                results = await self._provider(query, k)
            elif os.getenv("TAVILY_API_KEY"):
                results = await _tavily_search(query, k)
            else:
                results = await _duckduckgo_search(query, k)

            return json.dumps({"query": query, "results": results}, ensure_ascii=False)
        except Exception as exc:
            log.warning("web_search: primary provider failed (%s), trying DuckDuckGo", exc)
            try:
                results = await _duckduckgo_search(query, k)
                return json.dumps({"query": query, "results": results}, ensure_ascii=False)
            except Exception as exc2:
                return f"ERROR: all search providers failed: {exc2}"


# ═══════════════════════════════════════════════════════════════════════════════
# URL Fetch — retrieve content from a specific URL
# ═══════════════════════════════════════════════════════════════════════════════

class UrlFetchTool(Tool):
    """Fetch and extract text content from a URL."""

    name = "url_fetch"
    description = (
        "Fetch the text content of a web page. "
        "Use after web_search to read full articles. "
        "Returns plain text (HTML tags stripped). Max 10KB returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to fetch (https preferred)"},
            "max_chars": {"type": "integer", "description": "Max characters to return (default 10000)", "default": 10000},
        },
        "required": ["url"],
    }

    async def run(self, args: dict[str, Any]) -> str:
        url = str(args.get("url", "")).strip()
        max_chars = int(args.get("max_chars", 10000))
        if not url:
            return "ERROR: missing 'url'"
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        def _fetch() -> str:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AkaradjeBot/1.0)",
                "Accept": "text/html,text/plain,application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read(500_000).decode("utf-8", errors="replace")
                return data, content_type

        try:
            loop = asyncio.get_event_loop()
            raw, content_type = await loop.run_in_executor(None, _fetch)
        except Exception as exc:
            return f"ERROR: fetch failed: {exc}"

        # Strip HTML tags for readability
        if "html" in content_type.lower() or raw.strip().startswith("<"):
            text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = raw.strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"

        return text or "(empty page)"


# ═══════════════════════════════════════════════════════════════════════════════
# File Read — read local files (for agentic workflows)
# ═══════════════════════════════════════════════════════════════════════════════

class FileReadTool(Tool):
    """Read content from a local file."""

    name = "file_read"
    description = (
        "Read the contents of a local file. "
        "Use for inspecting code, configs, or data files. "
        "Returns the file content as text. Max 50KB."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative or absolute)"},
            "start_line": {"type": "integer", "description": "Start reading from this line (1-indexed, optional)"},
            "end_line": {"type": "integer", "description": "Stop reading at this line (inclusive, optional)"},
        },
        "required": ["path"],
    }

    def __init__(self, *, allowed_dirs: list[str] | None = None):
        # Security: restrict to specific directories if needed
        self._allowed_dirs = allowed_dirs

    async def run(self, args: dict[str, Any]) -> str:
        path_str = str(args.get("path", "")).strip()
        if not path_str:
            return "ERROR: missing 'path'"

        path = Path(path_str).resolve()

        # Security check
        if self._allowed_dirs:
            allowed = any(str(path).startswith(d) for d in self._allowed_dirs)
            if not allowed:
                return f"ERROR: access denied — path not in allowed directories"

        if not path.exists():
            return f"ERROR: file not found: {path}"
        if not path.is_file():
            return f"ERROR: not a file: {path}"
        if path.stat().st_size > 50_000:
            return f"ERROR: file too large ({path.stat().st_size} bytes, max 50KB)"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"ERROR: read failed: {exc}"

        # Line range filtering
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            s = max(0, (int(start_line) - 1)) if start_line else 0
            e = int(end_line) if end_line else len(lines)
            content = "".join(lines[s:e])

        return content or "(empty file)"


# ═══════════════════════════════════════════════════════════════════════════════
# Shell Exec — run shell commands (sandboxed with timeout)
# ═══════════════════════════════════════════════════════════════════════════════

class ShellExecTool(Tool):
    """Execute a shell command and return output."""

    name = "shell_exec"
    description = (
        "Execute a shell command (bash) and return stdout + stderr. "
        "Use for: listing files, running tests, git commands, etc. "
        "15-second timeout. Do NOT use for long-running processes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "cwd": {"type": "string", "description": "Working directory (optional)"},
        },
        "required": ["command"],
    }
    timeout_seconds: float = 15.0

    def __init__(self, *, blocked_commands: list[str] | None = None):
        self._blocked = blocked_commands or [
            "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",
            "shutdown", "reboot", "poweroff",
        ]

    async def run(self, args: dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        cwd = args.get("cwd")
        if not command:
            return "ERROR: missing 'command'"

        # Basic safety check
        for blocked in self._blocked:
            if blocked in command:
                return f"ERROR: blocked command pattern: '{blocked}'"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
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
            return f"ERROR: shell failed: {exc}"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        # Truncate long output
        if len(out) > 10000:
            out = out[:5000] + f"\n...[truncated {len(out)-10000} chars]...\n" + out[-5000:]
        if len(err) > 5000:
            err = err[:2500] + "\n...[truncated]...\n" + err[-2500:]

        parts = []
        if out:
            parts.append(f"STDOUT:\n{out}")
        if err:
            parts.append(f"STDERR:\n{err}")
        if proc.returncode != 0:
            parts.insert(0, f"EXIT_CODE={proc.returncode}")

        return "\n".join(parts) or "(no output)"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    @classmethod
    def default(cls) -> ToolRegistry:
        """Default tool set for general-purpose chat agent."""
        return cls([
            CalculatorTool(),
            PythonExecTool(),
            WebSearchTool(),
            UrlFetchTool(),
            FileReadTool(),
            ShellExecTool(),
        ])

    @classmethod
    def safe(cls) -> ToolRegistry:
        """Restricted tool set (no file/shell access)."""
        return cls([
            CalculatorTool(),
            PythonExecTool(),
            WebSearchTool(),
            UrlFetchTool(),
        ])

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
