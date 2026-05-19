# akaradje

A scaffolded chatbot built on top of the DeepSeek API. Implements the
production-style pattern that makes frontier-class assistants feel "smart" —
not by inventing buzzwords, but by composing well-known techniques:

```
Query → Router → (Fast | Standard | Deep)
                        │
                        ▼
              Planner → Executor (ReAct + Tools)
                        │
                        ▼
              Verifier → Best-of-N voting → Answer
```

## What's inside

| Component | Purpose |
|-----------|---------|
| `router.py` | Classifies query complexity to avoid wasting compute |
| `tools.py` | Calculator, Python sandbox, web search (pluggable) |
| `executor.py` | ReAct loop with tool use |
| `verifier.py` | LLM-as-Judge with a separate prompt context |
| `voter.py` | Best-of-N sampling + verifier-scored selection |
| `memory.py` | Conversation history + simple keyword RAG |
| `orchestrator.py` | Glues everything together |
| `cli.py` | Interactive REPL |

## Setup

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
```

## Run

```bash
# Interactive chat
akaradje

# One-shot question
akaradje --once "What's 2^32 + factorial(10)?"

# Force deep reasoning path
akaradje --once "Design a rate limiter for a multi-tenant API" --deep
```

## Tuning knobs

All in `.env`:

- `BEST_OF_N` — how many candidate answers to sample for COMPLEX queries
- `VERIFIER_ENABLED` — turn off to halve cost when prototyping
- `MODEL_DEEP` — point at a stronger reasoning model
- `MAX_REACT_ITERATIONS` — bound the agent loop

## Notes on the DeepSeek "v4 pro" naming

DeepSeek's public model identifiers are `deepseek-chat` and
`deepseek-reasoner`. If you have access to a different SKU, set
`MODEL_DEEP` (etc.) to whatever string your provider exposes — the API is
OpenAI-compatible and the code does not hard-code model names.

## Architectural choices, briefly

- **Router first.** A Haiku-sized classifier decides if a question even
  deserves the deep path. Cheap insurance against burning tokens on
  "hello".
- **Separate verifier context.** The judge does not see the generator's
  scratchpad. This is the easiest way to dodge the self-agreement failure
  mode where a model rubber-stamps its own mistakes.
- **Best-of-N over self-refine.** Parallel sampling + judge selection
  empirically outperforms iterative self-correction on most reasoning
  benchmarks, and it's trivially parallelizable.
- **No fake jargon.** No "DiffAdapt", no "Recursive Tournament Voting".
  Just classifier routing, ReAct, LLM-as-Judge, and best-of-N.
