from __future__ import annotations

from typing import Any

import pytest

from ai_terminal.core import PackageManager


@pytest.mark.parametrize(
    "system_os, expected",
    [
        ("darwin", ["brew"]),
        ("linux", ["apt"]),
        ("windows", ["winget"]),
    ],
)
def test_detect_available_managers_prefers_which(monkeypatch: pytest.MonkeyPatch, system_os: str, expected: list[str]) -> None:
    calls: list[str] = []

    def fake_which(name: str) -> str | None:
        calls.append(name)
        return f"/usr/bin/{name}" if name in expected else None

    def fake_run(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - fallback should not be hit here
        raise AssertionError("subprocess.run should not be called when shutil.which succeeds")

    monkeypatch.setattr("ai_terminal.core.shutil.which", fake_which)
    monkeypatch.setattr("ai_terminal.core.subprocess.run", fake_run)

    manager = PackageManager(system_os)
    assert manager.available_managers == expected
    # ensure we attempted to detect at least the expected command
    for manager_name in expected:
        assert manager_name in calls


def test_suggest_install_command_uses_available_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ai_terminal.core.shutil.which", lambda name: None)

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        class Dummy:
            returncode = 0

        return Dummy()

    monkeypatch.setattr("ai_terminal.core.subprocess.run", fake_run)

    manager = PackageManager("linux")
    manager.available_managers = ["apt", "pip"]

    commands = manager.suggest_install_command("djongo")
    assert "sudo apt install -y djongo" in commands
    assert "pip install djongo" in commands
