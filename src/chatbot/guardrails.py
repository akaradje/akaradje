"""Guardrails — input/output safety filtering.

Runs between every user input and model output to catch:
1. Prompt injection attempts
2. PII leakage in outputs
3. Harmful content generation
4. Jailbreak patterns

Inspired by Claude Code's architecture: "A second opus prompt that runs as
a security classifier between turns."

We implement this as a lightweight rule-based first pass (instant, free)
plus an optional LLM-based classifier for ambiguous cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"  # allow but flag
    BLOCKED = "blocked"  # refuse to process


@dataclass
class SafetyResult:
    level: SafetyLevel
    reason: str = ""
    category: str = ""
    modified_content: str | None = None  # if content was sanitized


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern-based rules (instant, no API call)
# ═══════════════════════════════════════════════════════════════════════════════

# Prompt injection patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions?|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<\|im_start\|>", re.IGNORECASE),
]

# PII patterns (for output filtering)
_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

# Content categories to flag
_HARMFUL_PATTERNS = [
    (re.compile(r"\b(how\s+to\s+make|synthesize|manufacture)\s+(bomb|explosive|weapon)", re.IGNORECASE), "weapons"),
    (re.compile(r"\b(hack|exploit|breach)\s+(into|someone)", re.IGNORECASE), "hacking"),
    (re.compile(r"\bself[- ]harm\b", re.IGNORECASE), "self_harm"),
]


class InputGuardrail:
    """Checks user input before it reaches the model."""

    def __init__(self, *, strict: bool = False):
        self._strict = strict

    def check(self, user_input: str) -> SafetyResult:
        """Check user input for safety issues.

        Returns SafetyResult with level and reason.
        """
        if not user_input.strip():
            return SafetyResult(level=SafetyLevel.SAFE)

        # Check prompt injection
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(user_input):
                log.warning("guardrails: injection attempt detected")
                if self._strict:
                    return SafetyResult(
                        level=SafetyLevel.BLOCKED,
                        reason="Potential prompt injection detected",
                        category="injection",
                    )
                else:
                    return SafetyResult(
                        level=SafetyLevel.WARNING,
                        reason="Input contains patterns similar to prompt injection",
                        category="injection",
                    )

        # Check harmful content requests
        for pattern, category in _HARMFUL_PATTERNS:
            if pattern.search(user_input):
                log.warning("guardrails: harmful request — %s", category)
                return SafetyResult(
                    level=SafetyLevel.BLOCKED,
                    reason=f"Request involves potentially harmful content ({category})",
                    category=category,
                )

        return SafetyResult(level=SafetyLevel.SAFE)


class OutputGuardrail:
    """Checks model output before it reaches the user."""

    def __init__(self, *, redact_pii: bool = True):
        self._redact_pii = redact_pii

    def check(self, output: str) -> SafetyResult:
        """Check model output and optionally redact PII.

        Returns SafetyResult. If PII was found and redacted,
        modified_content contains the sanitized version.
        """
        if not output.strip():
            return SafetyResult(level=SafetyLevel.SAFE)

        if not self._redact_pii:
            return SafetyResult(level=SafetyLevel.SAFE)

        # Check for PII in output
        modified = output
        pii_found: list[str] = []

        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(modified)
            if matches:
                pii_found.append(f"{pii_type}({len(matches)})")
                # Redact
                modified = pattern.sub(f"[REDACTED_{pii_type.upper()}]", modified)

        if pii_found:
            log.info("guardrails: redacted PII in output: %s", ", ".join(pii_found))
            return SafetyResult(
                level=SafetyLevel.WARNING,
                reason=f"PII detected and redacted: {', '.join(pii_found)}",
                category="pii",
                modified_content=modified,
            )

        return SafetyResult(level=SafetyLevel.SAFE)


class GuardrailsManager:
    """Unified guardrails manager for the pipeline."""

    def __init__(self, *, strict: bool = False, redact_pii: bool = True):
        self._input = InputGuardrail(strict=strict)
        self._output = OutputGuardrail(redact_pii=redact_pii)

    def check_input(self, text: str) -> SafetyResult:
        return self._input.check(text)

    def check_output(self, text: str) -> SafetyResult:
        return self._output.check(text)

    def get_blocked_response(self, result: SafetyResult) -> str:
        """Generate a polite refusal message when input is blocked."""
        return (
            "I'm not able to help with that request. "
            f"Reason: {result.reason}. "
            "Please rephrase your question or ask something else."
        )
