"""Generate router training data using DeepSeek V4 Pro as teacher.

The teacher model classifies each query's difficulty on a continuous scale.
This creates the training dataset for the small LoRA router model.

Output format (JSONL):
    {"query": "...", "difficulty": 0.73, "label": "COMPLEX", "reasoning": "..."}

The difficulty score (0.0-1.0) is the primary target. Labels are derived:
    0.0-0.2 → TRIVIAL
    0.2-0.6 → STANDARD
    0.6-0.8 → COMPLEX
    0.8-1.0 → VERY_COMPLEX
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher prompt — asks V4 Pro to score query difficulty
# ═══════════════════════════════════════════════════════════════════════════════

TEACHER_PROMPT = """You are a query difficulty scorer for an AI assistant.

Score the following query on a scale from 0.0 to 1.0:
- 0.0-0.2: TRIVIAL — greetings, simple lookups, yes/no questions, basic math
- 0.2-0.4: EASY — single-fact explanations, simple code snippets, translations
- 0.4-0.6: STANDARD — multi-paragraph explanations, debugging, refactoring
- 0.6-0.8: COMPLEX — system design, multi-step reasoning, ambiguous tasks
- 0.8-1.0: VERY_COMPLEX — novel research, cross-domain synthesis, proof writing

Consider these factors:
1. Number of reasoning steps required
2. Amount of domain knowledge needed
3. Ambiguity level (more ambiguous = harder)
4. Whether tools/computation would help
5. Expected output length and structure

Output JSON only:
{"difficulty": <float 0.0-1.0>, "reasoning": "<1 sentence why>"}

QUERY: {query}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Seed queries — diverse examples to bootstrap data generation
# ═══════════════════════════════════════════════════════════════════════════════

SEED_QUERIES = [
    # Trivial (0.0-0.2)
    "Hi", "Hello!", "Thanks", "สวัสดี", "What's 2+2?", "What day is today?",
    "Bye!", "OK", "Yes", "No", "Good morning",
    
    # Easy (0.2-0.4)
    "What's the capital of France?",
    "How do I print hello world in Python?",
    "What is HTTP?",
    "Convert 100 Fahrenheit to Celsius",
    "What does async mean in JavaScript?",
    
    # Standard (0.4-0.6)
    "Explain how DNS resolution works step by step",
    "Write a function to merge two sorted arrays in Python",
    "What's the difference between TCP and UDP?",
    "How do I set up a virtual environment in Python?",
    "Debug this code: for i in range(10): print(i",
    "Explain the CAP theorem with examples",
    
    # Complex (0.6-0.8)
    "Design a URL shortening service like bit.ly",
    "How would you implement a distributed lock?",
    "Explain the trade-offs between microservices and monoliths for a startup",
    "Write a rate limiter using sliding window algorithm with Redis",
    "Debug a race condition in this concurrent Go code",
    "Design the database schema for a social media platform",
    
    # Very Complex (0.8-1.0)
    "Design a real-time collaborative editor like Google Docs from scratch",
    "Prove that P ≠ NP implies one-way functions exist",
    "Design a distributed consensus algorithm that handles Byzantine faults",
    "Build a compiler for a new programming language with type inference",
    "Architect a globally distributed database with strong consistency",
    "Design an ML pipeline that detects financial fraud in real-time at scale",
]

# Templates for generating more diverse queries
QUERY_TEMPLATES = [
    "How do I {action}?",
    "Explain {concept} in simple terms",
    "What is the difference between {a} and {b}?",
    "Write a {language} function that {task}",
    "Design a system that {requirement}",
    "Debug this error: {error}",
    "Optimize this code for {metric}: {code_snippet}",
    "Compare {approach_a} vs {approach_b} for {use_case}",
    "Implement {algorithm} with {constraint}",
    "What are the trade-offs of {decision}?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data generation logic
# ═══════════════════════════════════════════════════════════════════════════════

async def score_query(client: AsyncOpenAI, query: str, model: str) -> dict[str, Any] | None:
    """Use the teacher model to score a single query."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TEACHER_PROMPT.format(query=query)}],
            max_tokens=100,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        obj = json.loads(text)
        difficulty = float(obj.get("difficulty", 0.5))
        difficulty = max(0.0, min(1.0, difficulty))
        
        # Derive label from score
        if difficulty < 0.2:
            label = "TRIVIAL"
        elif difficulty < 0.6:
            label = "STANDARD"
        elif difficulty < 0.8:
            label = "COMPLEX"
        else:
            label = "VERY_COMPLEX"
        
        return {
            "query": query,
            "difficulty": round(difficulty, 3),
            "label": label,
            "reasoning": obj.get("reasoning", ""),
        }
    except Exception as exc:
        log.warning("Failed to score query '%s': %s", query[:50], exc)
        return None


async def generate_dataset(
    client: AsyncOpenAI,
    queries: list[str],
    model: str,
    *,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Score all queries in batches."""
    results: list[dict[str, Any]] = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        tasks = [score_query(client, q, model) for q in batch]
        batch_results = await asyncio.gather(*tasks)
        
        for r in batch_results:
            if r is not None:
                results.append(r)
        
        log.info("Progress: %d/%d scored (%d successful)", i + len(batch), len(queries), len(results))
    
    return results


def augment_queries(seed_queries: list[str], target_n: int) -> list[str]:
    """Augment seed queries to reach target count.
    
    In production, you'd use a more sophisticated augmentation strategy
    (e.g., ask the LLM to generate similar queries at each difficulty level).
    For now, we just duplicate seeds with minor variations.
    """
    queries = list(seed_queries)
    
    # Add variations
    prefixes = ["Please ", "Can you ", "I need help with: ", "Quick question: ", ""]
    suffixes = ["", " Thanks!", " Please explain.", " Show me how.", ""]
    
    while len(queries) < target_n:
        base = random.choice(seed_queries)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        queries.append(f"{prefix}{base}{suffix}".strip())
    
    return queries[:target_n]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate router training data")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="../data/router_train.jsonl")
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--api-key", type=str, default=None, help="DeepSeek API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    import os
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY or pass --api-key")
        sys.exit(1)
    
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    
    # Generate query list
    queries = augment_queries(SEED_QUERIES, args.n)
    random.shuffle(queries)
    log.info("Generated %d queries for labeling", len(queries))
    
    # Score queries
    results = asyncio.run(generate_dataset(client, queries, args.model, batch_size=args.batch_size))
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    log.info("Wrote %d samples to %s", len(results), output_path)
    
    # Stats
    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    log.info("Label distribution: %s", dict(label_counts))
    
    avg_diff = sum(r["difficulty"] for r in results) / max(1, len(results))
    log.info("Average difficulty: %.3f", avg_diff)


if __name__ == "__main__":
    main()
