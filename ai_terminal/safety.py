"""Command safety auditing utilities for Magic Terminal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class CommandWarning:
    """A structured warning about a potentially unsafe command."""

    severity: str
    message: str


class CommandAuditor:
    """Detect destructive or high-risk shell commands before execution."""

    HIGH_RISK_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\brm\s+-rf\s+/(\s|$)"),
        re.compile(r"\bsudo\s+rm\s+-rf\s+\S+"),
        re.compile(r"\bdd\s+if=/dev/zero\s+of=/dev/sd"),
        re.compile(r"\bmkfs(\.|\s)"),
        re.compile(r"(:\(\)\s*\{\s*: \| : \& \};\s*:)", re.UNICODE),
        re.compile(r"\bshutdown\b"),
        re.compile(r"\breboot\b"),
        re.compile(r"\bpoweroff\b"),
        re.compile(r"\bcc\s+\w+\s+--force"),
        re.compile(r"\bkill\s+-9\s+1\b"),
    )

    MEDIUM_RISK_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\bchown\s+-R\s+root\b"),
        re.compile(r"\bchmod\s+-R\s+7\d\d\b"),
        re.compile(r"\brm\s+-rf\s+\S+"),
        re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
        re.compile(r">\s*/dev/sd"),
    )

    def audit(self, command: str) -> List[CommandWarning]:
        """Return a list of warnings describing risky patterns in the command."""
        warnings: List[CommandWarning] = []

        normalized = command.strip()
        if not normalized:
            return warnings

        for pattern in self.HIGH_RISK_PATTERNS:
            if pattern.search(normalized):
                warnings.append(
                    CommandWarning(
                        severity="high",
                        message=f"Potentially destructive command detected: pattern '{pattern.pattern}'.",
                    )
                )

        for pattern in self.MEDIUM_RISK_PATTERNS:
            if pattern.search(normalized):
                warnings.append(
                    CommandWarning(
                        severity="medium",
                        message=f"Command may require caution: pattern '{pattern.pattern}'.",
                    )
                )

        return warnings

    def highest_severity(self, warnings: Iterable[CommandWarning]) -> str:
        """Return the highest severity string present in *warnings*."""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        level = 0
        for warning in warnings:
            level = max(level, severity_order.get(warning.severity, 0))
        if level == 3:
            return "high"
        if level == 2:
            return "medium"
        if level == 1:
            return "low"
        return "none"


__all__ = ["CommandAuditor", "CommandWarning"]
