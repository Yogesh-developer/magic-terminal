"""HTTP helper utilities with retry and backoff support."""

from __future__ import annotations

import logging
import time
from typing import Iterable, Optional, Union

import requests

DEFAULT_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def request_with_retries(
    method: str,
    url: str,
    *,
    retries: int = 3,
    backoff_factor: float = 0.5,
    timeout: Union[int, float] = 10,
    retry_statuses: Optional[Iterable[int]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> requests.Response:
    """Execute an HTTP request with simple exponential backoff.

    Parameters
    ----------
    method: str
        HTTP method (``"GET"``, ``"POST"`` ...).
    url: str
        Target URL.
    retries: int, default 3
        Maximum number of attempts.
    backoff_factor: float, default 0.5
        Base delay used for exponential backoff (seconds).
    timeout: int | float, default 10
        Timeout handed to ``requests`` per attempt.
    retry_statuses: Iterable[int] | None
        HTTP status codes that should trigger a retry.
    logger: logging.Logger | None
        Optional logger for diagnostics.
    kwargs:
        Additional keyword arguments forwarded to ``requests.request``.

    Returns
    -------
    requests.Response
        Response object for the first successful attempt (non-retryable status).

    Raises
    ------
    requests.RequestException
        Propagated when all attempts fail due to a transport-level error.
    requests.HTTPError
        Raised when all attempts ended with retryable HTTP status codes.
    """

    if retries < 1:
        raise ValueError("retries must be at least 1")

    retry_statuses_set = (
        set(DEFAULT_RETRY_STATUS_CODES) if retry_statuses is None else set(retry_statuses)
    )
    log = logger or logging.getLogger(__name__)

    response: Optional[requests.Response] = None
    last_exception: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.request(method=method, url=url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:  # noqa: BLE001
            last_exception = exc
            log.warning(
                "HTTP %s %s failed on attempt %s/%s: %s",
                method,
                url,
                attempt,
                retries,
                exc,
            )
        else:
            if response.status_code in retry_statuses_set and attempt < retries:
                last_exception = requests.HTTPError(
                    f"Retryable status code {response.status_code}", response=response
                )
                log.warning(
                    "HTTP %s %s returned %s on attempt %s/%s; retrying.",
                    method,
                    url,
                    response.status_code,
                    attempt,
                    retries,
                )
            else:
                return response

        if attempt < retries:
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            log.debug(
                "Sleeping for %.2fs before retrying HTTP %s %s",
                sleep_time,
                method,
                url,
            )
            time.sleep(sleep_time)

    if last_exception:
        raise last_exception

    assert response is not None  # for type-checkers; unreachable otherwise
    return response


__all__ = ["request_with_retries", "DEFAULT_RETRY_STATUS_CODES"]
