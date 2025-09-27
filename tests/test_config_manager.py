import json
from pathlib import Path

import pytest

from ai_terminal.config_manager import ConfigManager, DEFAULT_CONFIG


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def test_load_returns_defaults_when_file_missing(config_path: Path) -> None:
    manager = ConfigManager(config_path)
    loaded = manager.load()
    assert loaded == DEFAULT_CONFIG


def test_load_applies_migrations_and_validation(config_path: Path) -> None:
    payload = {
        "auto_confirm_safe": "true",
        "use_trash": "false",
        "max_history": "100",
        "aliases": {"ls": "ls -la"},
        "unknown": "value",
    }
    config_path.write_text(json.dumps(payload))
    manager = ConfigManager(config_path)
    loaded = manager.load()

    assert loaded["auto_confirm_safe"] is True
    assert loaded["use_trash"] is False
    assert loaded["max_history"] == 100
    assert loaded["aliases"] == {"ls": "ls -la"}
    # unknown key should be discarded when validation fails
    assert "unknown" not in loaded


def test_save_rejects_invalid_payload(config_path: Path) -> None:
    pytest.importorskip("jsonschema")
    manager = ConfigManager(config_path)
    bad_config = {"auto_confirm_safe": "not-a-bool"}
    with pytest.raises(Exception):
        manager.save(bad_config)
