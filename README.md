# akaradje

A DeepSeek V4 Pro chatbot with **Opus 4.7-class scaffolding**.

Instead of inventing jargon, this implements the actual engineering patterns
that make frontier models feel smart — applied to DeepSeek's API:

```
Query → Router → Complexity tier (TRIVIAL / STANDARD / COMPLEX)
                        │
                        ▼
              Map to reasoning_effort (disabled / medium / high / max)
                        │
                        ▼
              Executor (ReAct + Thinking + Tools) ←── Task Budget countdown
                        │
                        ▼
              Verifier (LLM-as-Judge with thinking ON)
                        │
                        ▼
              [Best-of-N voting for COMPLEX queries]
                        │
                        ▼
                      Answer
```

## How it maps to Opus 4.7

| Claude Opus 4.7 Feature | DeepSeek V4 Pro Implementation |
|---|---|
| Adaptive thinking | `thinking: {type: "enabled"}` + `reasoning_effort` |
| `effort: xhigh/high/medium/low` | `reasoning_effort: "max"/"high"/"medium"/"low"` |
| Task budgets (token countdown) | Simulated: track tokens, inject budget message |
| Interleaved thinking + tools | Native "thinking with tools" in V4 |
| Sampling params locked | V4 ignores temp/top_p in thinking mode |

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | Env-driven config with effort mapping |
| `client.py` | OpenAI SDK → DeepSeek with thinking mode support |
| `router.py` | Complexity classifier → reasoning_effort mapper |
| `tools.py` | Calculator, Python exec, Web search (pluggable) |
| `executor.py` | ReAct loop with interleaved thinking + tools |
| `verifier.py` | LLM-as-Judge (separate context, thinking ON) |
| `voter.py` | Best-of-N with diversity via prompt variation |
| `task_budget.py` | Token countdown (simulated Opus 4.7 task_budget) |
| `memory.py` | Window + JSONL archive + keyword RAG |
| `orchestrator.py` | Pipeline glue |
| `cli.py` | Rich REPL with --effort, --deep, --budget flags |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Set DEEPSEEK_API_KEY
```

## Usage

```bash
# Interactive
akaradje

# One-shot with max effort (like Opus 4.7 xhigh)
akaradje --once "Design a rate limiter" --effort max

# Force deep reasoning path with full Best-of-N
akaradje --once "Debug this race condition" --deep

# Fast path (thinking OFF, like effort=low)
akaradje --once "Hello" --fast

# Custom budget (128k tokens for the agent loop)
akaradje --once "Refactor the auth module" --budget 128000

# Show the model's reasoning trace
akaradje --once "Prove that sqrt(2) is irrational" --effort high --show-thinking
```

## REPL commands

```
/deep <q>    Force max effort + COMPLEX path
/fast <q>    Thinking disabled
/think <q>   Show reasoning trace
/reset       Clear memory window
/quit        Exit
```

## Key design choices

1. **Single model, variable effort** — V4 Pro handles everything from
   greetings (thinking OFF) to system design (reasoning_effort=max).
   No need for fast/standard/deep model tiers.

2. **Task Budget via injection** — DeepSeek has no native task_budget
   parameter. We simulate it by tracking tokens and injecting a countdown
   message so the model self-moderates. Same behavioral effect.

3. **Diversity via prompt variation** — Since temp is ignored in thinking
   mode, Best-of-N achieves diversity by giving each candidate a different
   "approach angle" system suffix. More principled than temperature jitter.

4. **reasoning_content handling** — Per DeepSeek docs: pass it back when
   tool calls happened (it participates in context), strip it otherwise
   (API ignores it anyway, save tokens).

5. **Verifier uses thinking** — The judge runs with `reasoning_effort: high`
   so it actually reasons about correctness rather than rubber-stamping.
