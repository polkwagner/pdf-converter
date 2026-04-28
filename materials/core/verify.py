"""Cheap, format-agnostic verification primitives.

Spec §6.1 (cheap checks always run) and §6.4 (failure semantics).
Format-specific deep checks live in the per-format modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


@dataclass
class VerifyReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def overall(self) -> str:
        if any(r.status == "FAIL" for r in self.results):
            return "FAIL"
        if any(r.status == "WARN" for r in self.results):
            return "WARN"
        return "PASS"


def check_non_empty(markdown: str) -> CheckResult:
    """Output must not be empty. Empty output is a FAIL — caller should not write."""
    if not markdown.strip():
        return CheckResult("non_empty", "FAIL", "Output markdown is empty")
    return CheckResult("non_empty", "PASS", f"{len(markdown):,} chars")


def check_word_retention(source_words: int, output_words: int, min_ratio: float) -> CheckResult:
    """Word retention ratio must be at or above the format's minimum band.
    Below band = WARN (write the file, flag it). Empty source = N/A."""
    if source_words == 0:
        return CheckResult(
            "word_retention",
            "PASS",
            "source has no extractable words (skipping)",
        )
    ratio = output_words / source_words
    detail = f"{output_words:,} / {source_words:,} = {ratio:.0%} (min {min_ratio:.0%})"
    if ratio < min_ratio:
        return CheckResult("word_retention", "WARN", detail)
    return CheckResult("word_retention", "PASS", detail)


def count_words(text: str) -> int:
    """Whitespace-delimited word count. Used by every format's cheap verifier."""
    return len(text.split())
