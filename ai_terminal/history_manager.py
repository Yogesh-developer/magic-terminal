"""Command history management utilities for Magic Terminal."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Optional

try:
    import readline  # type: ignore
except ImportError:  # pragma: no cover - `readline` missing on some platforms
    readline = None  # type: ignore


logger = logging.getLogger(__name__)


class HistoryManager:
    """Load and persist shell history when readline is available."""

    def __init__(self, path: Path, *, max_length: int = 1000) -> None:
        self.path = path
        self.max_length = max_length
        self._initialized = False

    def load(self) -> None:
        """Load history from disk into readline."""
        if readline is None:
            logger.debug("readline not available; skipping history load")
            return

        try:
            if self.path.exists():
                readline.read_history_file(str(self.path))
            readline.set_history_length(self.max_length)
            self._initialized = True
            logger.debug("Loaded history from %s", self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load history %s: %s", self.path, exc)

        atexit.register(self.save)

    def save(self) -> None:
        """Persist current readline history to disk."""
        if readline is None:
            return
        if not self._initialized:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(self.path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write history %s: %s", self.path, exc)

    def add_entry(self, command: str) -> None:
        """Append a command to the history buffer if readline is active."""
        if readline is None:
            return
        if not command:
            return
        try:
            readline.add_history(command)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Unable to append history entry: %s", exc)

    def search(self, prefix: str) -> Optional[str]:
        """Return the latest history entry matching the given prefix."""
        if readline is None:
            return None
        try:
            length = readline.get_current_history_length()
            for index in range(length, 0, -1):
                entry = readline.get_history_item(index)
                if entry and entry.startswith(prefix):
                    return entry
        except Exception as exc:  # noqa: BLE001
            logger.debug("History search failed: %s", exc)
        return None


__all__ = ["HistoryManager"]
