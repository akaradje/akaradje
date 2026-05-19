"""Interactive CLI for the DeepSeek V4 Pro chatbot.

Flags mirror Opus 4.7's interface:
    --effort low|medium|high|max   Force a specific reasoning effort
    --budget 128000                Override task budget tokens
    --deep                         Shortcut for --effort max (COMPLEX path)
    --fast                         Shortcut for thinking disabled (TRIVIAL path)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import Config, ReasoningEffort, setup_logging
from .memory import ConversationMemory
from .orchestrator import Answer, Orchestrator
from .router import Complexity

console = Console()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="akaradje",
        description="DeepSeek V4 Pro chatbot with Opus 4.7-class scaffolding",
    )
    p.add_argument("--once", help="One-shot question, then exit", default=None)
    p.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="Force reasoning effort level (like Opus 4.7 effort parameter)",
    )
    p.add_argument("--deep", action="store_true", help="Force COMPLEX path + max effort")
    p.add_argument("--fast", action="store_true", help="Force TRIVIAL path (thinking OFF)")
    p.add_argument("--budget", type=int, default=None, help="Override task budget tokens")
    p.add_argument(
        "--memory-file",
        default=".akaradje_memory/history.jsonl",
        help="Conversation history file",
    )
    p.add_argument("--no-memory", action="store_true", help="Disable memory")
    p.add_argument("--quiet", action="store_true", help="Hide diagnostics")
    p.add_argument("--show-thinking", action="store_true", help="Show reasoning trace")
    return p.parse_args(argv)


def _force_from_args(args: argparse.Namespace) -> tuple[Complexity | None, ReasoningEffort | None]:
    if args.deep and args.fast:
        console.print("[red]--deep and --fast are mutually exclusive[/red]")
        sys.exit(2)
    complexity = None
    effort = None
    if args.deep:
        complexity = Complexity.COMPLEX
        effort = ReasoningEffort.MAX
    elif args.fast:
        complexity = Complexity.TRIVIAL
    elif args.effort:
        effort = ReasoningEffort(args.effort)
    return complexity, effort


def _render_answer(ans: Answer, *, quiet: bool, show_thinking: bool) -> None:
    console.print(Panel(Markdown(ans.text or "(empty)"), title="Answer", border_style="green"))

    if show_thinking and ans.reasoning_trace:
        for i, trace in enumerate(ans.reasoning_trace, 1):
            console.print(
                Panel(
                    trace[:2000] + ("..." if len(trace) > 2000 else ""),
                    title=f"Thinking #{i}",
                    border_style="dim yellow",
                )
            )

    if quiet:
        return

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("complexity", ans.complexity.value)
    table.add_row("thinking", ans.thinking_mode)
    table.add_row("effort", ans.reasoning_effort)
    table.add_row("iterations", str(ans.iterations))
    table.add_row("tool calls", str(ans.tool_calls))
    table.add_row("tokens used", f"{ans.tokens_used:,}")
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


async def _run_once(
    orch: Orchestrator, query: str,
    force_complexity: Complexity | None,
    force_effort: ReasoningEffort | None,
    quiet: bool, show_thinking: bool,
) -> int:
    try:
        ans = await orch.ask(
            query,
            force_complexity=force_complexity,
            force_effort=force_effort,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    _render_answer(ans, quiet=quiet, show_thinking=show_thinking)
    return 0


async def _run_repl(
    orch: Orchestrator,
    force_complexity: Complexity | None,
    force_effort: ReasoningEffort | None,
    quiet: bool, show_thinking: bool,
) -> int:
    console.print(
        Panel.fit(
            "[bold]akaradje[/bold] (DeepSeek V4 Pro) — "
            "type a question, or [cyan]/help[/cyan]",
            border_style="cyan",
        )
    )
    while True:
        try:
            query = console.input("[bold magenta]you >[/bold magenta] ").strip()
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
                "[cyan]/reset[/cyan] clear memory  "
                "[cyan]/deep <q>[/cyan] max effort  "
                "[cyan]/fast <q>[/cyan] no thinking  "
                "[cyan]/think <q>[/cyan] show reasoning trace"
            )
            continue
        if query == "/reset":
            orch.memory.reset()
            console.print("[dim]memory cleared[/dim]")
            continue

        per_complexity = force_complexity
        per_effort = force_effort
        per_show = show_thinking

        if query.startswith("/deep "):
            per_complexity = Complexity.COMPLEX
            per_effort = ReasoningEffort.MAX
            query = query[6:].strip()
        elif query.startswith("/fast "):
            per_complexity = Complexity.TRIVIAL
            query = query[6:].strip()
        elif query.startswith("/think "):
            per_show = True
            query = query[7:].strip()

        try:
            with console.status("[dim]thinking...[/dim]"):
                ans = await orch.ask(
                    query,
                    force_complexity=per_complexity,
                    force_effort=per_effort,
                )
        except KeyboardInterrupt:
            console.print("[dim]interrupted[/dim]")
            continue
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")
            continue
        _render_answer(ans, quiet=quiet, show_thinking=per_show)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    cfg = Config.from_env()
    setup_logging(cfg.log_level)

    # Override budget if specified
    if args.budget is not None:
        # Rebuild config with overridden budget
        import dataclasses
        cfg = dataclasses.replace(cfg, task_budget_tokens=args.budget)

    try:
        cfg.validate()
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return 2

    memory_path: Path | None = None
    if not args.no_memory and args.memory_file:
        memory_path = Path(args.memory_file)
    memory = ConversationMemory(store_path=memory_path)

    orch = Orchestrator(cfg, memory=memory)
    force_complexity, force_effort = _force_from_args(args)

    if args.once is not None:
        return asyncio.run(_run_once(
            orch, args.once, force_complexity, force_effort,
            args.quiet, args.show_thinking,
        ))
    return asyncio.run(_run_repl(
        orch, force_complexity, force_effort,
        args.quiet, args.show_thinking,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
