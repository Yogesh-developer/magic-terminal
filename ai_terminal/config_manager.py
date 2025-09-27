"""Configuration loading, validation, and persistence utilities for Magic Terminal."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    from jsonschema import ValidationError, validate  # type: ignore
except ImportError:  # pragma: no cover - jsonschema is an optional dependency at runtime
    validate = None  # type: ignore
    ValidationError = Exception  # type: ignore


logger = logging.getLogger(__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "auto_confirm_safe": False,
    "use_trash": True,
    "max_history": 1000,
    "bookmarks": {},
    "aliases": {},
    "preferred_package_manager": None,
}

CONFIG_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "auto_confirm_safe": {"type": "boolean"},
        "use_trash": {"type": "boolean"},
        "max_history": {"type": "integer", "minimum": 0},
        "bookmarks": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "aliases": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "preferred_package_manager": {
            "type": ["string", "null"],
            "pattern": "^[a-zA-Z0-9_-]+$",
        },
    },
    "additionalProperties": True,
}


@dataclass
class ConfigManager:
    """Handle config persistence with schema validation and migrations."""

    path: Path
    defaults: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    def load(self) -> Dict[str, Any]:
        """Load configuration from disk, applying defaults and migrations."""
        data = self.defaults.copy()

        if not self.path.exists():
            logger.debug("Config file %s does not exist; using defaults", self.path)
            return data

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in config %s: %s. Regenerating defaults.", self.path, exc)
            return data
        except OSError as exc:
            logger.warning("Unable to read config %s: %s. Using defaults.", self.path, exc)
            return data

        # Apply migrations before validation to accommodate old formats
        payload = self._apply_migrations(payload)

        if validate:
            try:
                validate(instance=payload, schema=CONFIG_SCHEMA)  # type: ignore[arg-type]
            except ValidationError as exc:  # type: ignore[misc]
                logger.warning("Config validation failed: %s. Falling back to defaults + valid keys.", exc)
                payload = self._filter_valid_keys(payload)
        else:
            logger.debug("jsonschema unavailable; skipping config validation")

        data.update(payload)
        return data

    def save(self, config: Dict[str, Any]) -> None:
        """Persist configuration to disk after validation."""
        payload = self.defaults.copy()
        payload.update(config)

        if validate:
            try:
                validate(instance=payload, schema=CONFIG_SCHEMA)  # type: ignore[arg-type]
            except ValidationError as exc:  # type: ignore[misc]
                logger.error("Refusing to save invalid config: %s", exc)
                raise

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except OSError as exc:
            logger.error("Failed to write config %s: %s", self.path, exc)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_migrations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy config structures to the current schema."""
        migrated = payload.copy()

        # Example migration: accept legacy bool stored as string ("true"/"false")
        for key in ("auto_confirm_safe", "use_trash"):
            value = migrated.get(key)
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in {"true", "false"}:
                    migrated[key] = lowered == "true"

        if isinstance(migrated.get("max_history"), str):
            try:
                migrated["max_history"] = int(migrated["max_history"])
            except (ValueError, TypeError):
                migrated.pop("max_history", None)

        return migrated

    def _filter_valid_keys(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only keys present in the schema/defaults when validation fails."""
        allowed = set(self.defaults.keys())
        return {key: value for key, value in payload.items() if key in allowed}


__all__ = ["ConfigManager", "DEFAULT_CONFIG"]
