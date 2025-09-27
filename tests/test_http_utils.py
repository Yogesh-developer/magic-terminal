import types
from typing import List

import pytest

from ai_terminal.http_utils import request_with_retries


class DummyResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


def test_request_with_retries_eventual_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []

    def fake_request(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:
            class DummyException(Exception):
                pass

            raise DummyException("temporary failure")
        return DummyResponse(status_code=200)

    monkeypatch.setattr("ai_terminal.http_utils.requests.request", fake_request)

    response = request_with_retries("GET", "http://example.com", retries=2, timeout=0.1, backoff_factor=0)
    assert response.status_code == 200
    assert len(calls) == 2


def test_request_with_retries_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(**kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr("ai_terminal.http_utils.requests.request", always_fail)

    with pytest.raises(RuntimeError):
        request_with_retries("GET", "http://example.com", retries=2, timeout=0.1, backoff_factor=0)
