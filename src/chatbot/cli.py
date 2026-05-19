"""Interactive command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import Config, setup_logging
from .memory import ConversationMemory
from .orchestrator import Answer, Orchestrator
from .router import Complexity


console = Console()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="akaradje",
        description="Scaffolded DeepSeek chatbot — Router + ReAct + Verifier + Best-of-N",
    )
    p.add_argument("--once", help="One-shot question, then exit", default=None)
    p.add_argument(
        "--deep",
        action="store_true",
        help="Force the COMPLEX path (Best-of-N) regardless of router verdict",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Force the TRIVIAL path (single fast-model call)",
    )
    p.add_argument(
        "--memory-file",
        default=".akaradje_memory/history.jsonl",
        help="Where to persist conversation history (set to empty to disable)",
    )
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory persistence and recall for this session",
    )
    p.add_argument("--quiet", action="store_true", help="Hide diagnostics panels")
    return p.parse_args(argv)


def _force_from_args(args: argparse.Namespace) -> Complexity | None:
    if args.deep and args.fast:
        console.print("[red]--deep and --fast are mutually exclusive[/red]")
        sys.exit(2)
    if args.deep:
        return Complexity.COMPLEX
    if args.fast:
        return Complexity.TRIVIAL
    return None


def _render_answer(ans: Answer, *, quiet: bool) -> None:
    console.print(Panel(Markdown(ans.text or "(empty)"), title="Answer", border_style="green"))
    if quiet:
        return

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("complexity", ans.complexity.value)
    table.add_row("model", ans.model_used)
    table.add_row("iterations", str(ans.iterations))
    table.add_row("tool calls", str(ans.tool_calls))
    if ans.verdict is not None:
        v = ans.verdict
        table.add_row(
            "verdict",
            f"overall={v.overall} correct={v.correctness} complete={v.completeness} "
            f"clarity={v.clarity} pass={v.passed}",
        )
        if v.issues:
            table.add_row("issues", "\n".join(f"- {i}" for i in v.issues))
    if ans.voting is not None:
        scores = ", ".join(f"{round(c.score, 1)}" for c in ans.voting.candidates)
        table.add_row("voting", f"n={ans.voting.n} scores=[{scores}]")
    for k, v in ans.diagnostics.items():
        table.add_row(k, str(v))
    console.print(Panel(table, title="Diagnostics", border_style="dim"))


async def _run_once(orch: Orchestrator, query: str, force: Complexity | None, quiet: bool) -> int:
    try:
        ans = await orch.ask(query, force_complexity=force)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    _render_answer(ans, quiet=quiet)
    return 0


async def _run_repl(orch: Orchestrator, force: Complexity | None, quiet: bool) -> int:
    console.print(
        Panel.fit(
            "[bold]akaradje[/bold] — type your question, or [cyan]/help[/cyan]",
            border_style="cyan",
        )
    )
    while True:
        try:
            query = console.input("[bold magenta]you ›[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        if not query:
            continue
        if query in {"/quit", "/exit", "/q"}:
            return 0
        if query == "/help":
            console.print(
                "[cyan]/quit[/cyan] exit  "
                "[cyan]/reset[/cyan] clear memory window  "
                "[cyan]/deep <q>[/cyan] force best-of-N  "
                "[cyan]/fast <q>[/cyan] force trivial path"
            )
            continue
        if query == "/reset":
            orch.memory.reset()
            console.print("[dim]memory window cleared[/dim]")
            continue

        per_turn_force = force
        if query.startswith("/deep "):
            per_turn_force = Complexity.COMPLEX
            query = query[len("/deep ") :].strip()
        elif query.startswith("/fast "):
            per_turn_force = Complexity.TRIVIAL
            query = query[len("/fast ") :].strip()

        try:
            with console.status("[dim]thinking...[/dim]"):
                ans = await orch.ask(query, force_complexity=per_turn_force)
        except KeyboardInterrupt:
            console.print("[dim]interrupted[/dim]")
            continue
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")
            continue
        _render_answer(ans, quiet=quiet)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    cfg = Config.from_env()
    setup_logging(cfg.log_level)
    try:
        cfg.validate()
    except RuntimeError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    memory_path: Path | None = None
    if not args.no_memory and args.memory_file:
        memory_path = Path(args.memory_file)
    memory = ConversationMemory(store_path=memory_path)

    orch = Orchestrator(cfg, memory=memory)
    force = _force_from_args(args)

    if args.once is not None:
        return asyncio.run(_run_once(orch, args.once, force, args.quiet))
    return asyncio.run(_run_repl(orch, force, args.quiet))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
