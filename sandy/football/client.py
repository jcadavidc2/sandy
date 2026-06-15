"""API-Football (api-sports.io) HTTP client.

Mirrors :class:`sandy.ingest.client.MlbStatsClient`:
- Token-bucket rate limiter capped at ``config.football.max_rps``
- Exponential-backoff retry on HTTP 429 / 5xx with jitter
- Non-retryable classification for other 4xx and JSON decode errors
- API-key auth via the ``x-apisports-key`` header

API-Football quirks handled here:
- The API returns HTTP 200 even for auth/quota problems, signalling the issue
  in a non-empty ``errors`` field of the JSON body. We surface those as
  :class:`FootballApiError` so callers don't silently treat them as success.
- The response envelope is ``{"response": [...], "results": N, "errors": ...}``;
  :meth:`ApiFootballClient.get` returns the full envelope (caller reads
  ``["response"]``).
"""
from __future__ import annotations

import contextlib
import json
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from sandy.config import FootballConfig
from sandy.logging import get_logger

logger = get_logger("football.client")


# API-Football is served via Cloudflare with AAAA (IPv6) records. Hosts with
# broken IPv6 egress (e.g. this EC2) hang in urllib because it has no
# happy-eyeballs fallback. Force IPv4 resolution for the duration of a request.
@contextlib.contextmanager
def _force_ipv4():
    real_getaddrinfo = socket.getaddrinfo

    def ipv4_only(host, *args, **kwargs):
        results = real_getaddrinfo(host, *args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        return v4 or results

    socket.getaddrinfo = ipv4_only
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FootballApiError(Exception):
    """Non-retryable API error (4xx other than 429, bad JSON, body errors)."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retryable = False


class FootballApiRetryExhausted(Exception):
    """Raised when all retry attempts are exhausted on a retryable error."""

    def __init__(self, message: str, http_status: int | None = None, retries: int = 0) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retries = retries
        self.retryable = True


class MissingApiKeyError(FootballApiError):
    """Raised when an API call is attempted without APIFOOTBALL_KEY set."""

    def __init__(self) -> None:
        super().__init__(
            "APIFOOTBALL_KEY is not set. Export it (api-sports.io key) before "
            "running football ingestion."
        )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (same design as ingest.client._TokenBucket)
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Token bucket supporting sub-1-rps rates.

    Capacity is ``max(rate, 1.0)`` so that low rates (e.g. 0.15 rps for the
    API-Football free tier) can still accumulate the single token needed to
    proceed — capping at ``rate`` alone would deadlock when rate < 1.
    """
    rate: float
    _tokens: float = field(init=False)
    _capacity: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._capacity = max(self.rate, 1.0)
        self._tokens = self._capacity     # start full (one request allowed immediately)
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            sleep_for = (1.0 - self._tokens) / self.rate
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ApiFootballClient:
    """HTTP client for API-Football v3.

    Usage::

        client = ApiFootballClient(cfg.football)
        env = client.get("/fixtures", params={"league": "1", "season": "2022"})
        for item in env["response"]:
            ...
    """

    def __init__(self, cfg: FootballConfig) -> None:
        self._cfg = cfg
        self._bucket = _TokenBucket(rate=max(cfg.max_rps, 0.01))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """GET ``path`` (relative to base_url) and return the parsed envelope.

        ``path`` should start with ``/``, e.g. ``"/fixtures"``.

        Raises:
            MissingApiKeyError: no API key configured
            FootballApiError: non-retryable failure (4xx, bad JSON, body errors)
            FootballApiRetryExhausted: retryable failure after all attempts
        """
        if not self._cfg.api_key:
            raise MissingApiKeyError()
        url = self._build_url(path, params)
        return self._request_with_retry(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_url(self, path: str, params: dict[str, str] | None) -> str:
        url = self._cfg.base_url.rstrip("/") + path
        if params:
            query = "&".join(
                f"{urllib.parse.quote_plus(k)}={urllib.parse.quote_plus(str(v))}"
                for k, v in params.items()
            )
            url = f"{url}?{query}"
        return url

    def _request_with_retry(self, url: str) -> dict[str, Any]:
        cfg = self._cfg
        last_status: int | None = None
        last_error: str = ""

        for attempt in range(1, cfg.max_retries + 1):
            self._bucket.acquire()
            try:
                return self._do_get(url)

            except FootballApiError:
                raise

            except _RetryableHttpError as exc:
                last_status = exc.http_status
                last_error = str(exc)
                if attempt < cfg.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.debug(
                        "Retryable error, will retry",
                        extra={
                            "component": "football.client",
                            "url": url,
                            "attempt": attempt,
                            "http_status": last_status,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    time.sleep(delay)

            except Exception as exc:  # network-level — retryable
                last_error = str(exc)
                if attempt < cfg.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.debug(
                        "Network error, will retry",
                        extra={
                            "component": "football.client",
                            "url": url,
                            "attempt": attempt,
                            "error": last_error,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    time.sleep(delay)

        raise FootballApiRetryExhausted(
            f"All {cfg.max_retries} attempts failed for {url}: {last_error}",
            http_status=last_status,
            retries=cfg.max_retries,
        )

    def _do_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "x-apisports-key": self._cfg.api_key,
            },
        )
        try:
            with _force_ipv4(), urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 429 or status >= 500:
                raise _RetryableHttpError(f"HTTP {status} from {url}", http_status=status)
            raise FootballApiError(f"HTTP {status} from {url}", http_status=status)
        except urllib.error.URLError as exc:
            raise _NetworkError(str(exc)) from exc

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FootballApiError(f"JSON decode error from {url}: {exc}", http_status=None)

        self._raise_on_body_errors(envelope, url)
        return envelope

    @staticmethod
    def _raise_on_body_errors(envelope: Any, url: str) -> None:
        """API-Football signals auth/quota issues in a non-empty ``errors`` body."""
        if not isinstance(envelope, dict):
            raise FootballApiError(f"Unexpected non-object response from {url}")
        errors = envelope.get("errors")
        # Empty list (success) or empty/absent dict means no error.
        if isinstance(errors, list) and errors:
            raise FootballApiError(f"API-Football errors from {url}: {errors}")
        if isinstance(errors, dict) and errors:
            raise FootballApiError(f"API-Football errors from {url}: {errors}")

    def _backoff_delay(self, attempt: int) -> float:
        base = self._cfg.retry_base_delay_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0.75, 1.25)
        return base * jitter


# ---------------------------------------------------------------------------
# Internal exception types (not part of public API)
# ---------------------------------------------------------------------------


class _RetryableHttpError(Exception):
    def __init__(self, message: str, http_status: int) -> None:
        super().__init__(message)
        self.http_status = http_status


class _NetworkError(Exception):
    pass


__all__ = [
    "ApiFootballClient",
    "FootballApiError",
    "FootballApiRetryExhausted",
    "MissingApiKeyError",
]
