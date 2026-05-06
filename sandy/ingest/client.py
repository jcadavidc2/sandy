"""MLB Stats API HTTP client.

Task 5.1: MlbStatsClient with:
- Token-bucket rate limiter capped at config.ingest.max_rps (default 10 rps)
- Exponential-backoff retry on HTTP 429 / 5xx: base 1.0s, factor 2,
  jitter ±25%, up to max_retries attempts (default 5)
- Non-retryable classification for other 4xx and JSON decode errors
- Raises MlbApiError (retryable=False) or MlbApiRetryExhausted on failure
  so the ingestion service can decide whether to record to ingest_failures
  and continue, or propagate.

Requirements: 1.3, 1.7
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from sandy.config import IngestConfig
from sandy.logging import get_logger

logger = get_logger("ingest.client")

BASE_URL = "https://statsapi.mlb.com/api"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MlbApiError(Exception):
    """Non-retryable API error (4xx other than 429, JSON decode failure)."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retryable = False


class MlbApiRetryExhausted(Exception):
    """Raised when all retry attempts are exhausted on a retryable error."""

    def __init__(self, message: str, http_status: int | None = None, retries: int = 0) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retries = retries
        self.retryable = True


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


@dataclass
class _TokenBucket:
    """Simple single-threaded token-bucket limiter.

    Tokens refill continuously at ``rate`` per second. Each call to
    ``acquire()`` blocks (via ``time.sleep``) until a token is available.
    """
    rate: float          # tokens per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.rate          # start full
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # Sleep for the fraction of a second needed to earn one token
            sleep_for = (1.0 - self._tokens) / self.rate
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MlbStatsClient:
    """HTTP client for the public MLB Stats API.

    Usage::

        client = MlbStatsClient(cfg.ingest)
        data = client.get("/v1/schedule", params={"sportId": "1", ...})

    All requests go through the rate limiter and retry logic. On permanent
    failure the caller receives either :class:`MlbApiError` (non-retryable)
    or :class:`MlbApiRetryExhausted` (retryable but exhausted).
    """

    def __init__(self, cfg: IngestConfig) -> None:
        self._cfg = cfg
        self._bucket = _TokenBucket(rate=cfg.max_rps)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET ``path`` (relative to BASE_URL) and return parsed JSON.

        ``path`` should start with ``/``, e.g. ``"/v1/schedule"``.
        ``params`` are appended as a query string.

        Raises:
            MlbApiError: non-retryable failure (4xx other than 429, bad JSON)
            MlbApiRetryExhausted: retryable failure after all attempts used
        """
        url = self._build_url(path, params)
        return self._request_with_retry(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_url(path: str, params: dict[str, str] | None) -> str:
        url = BASE_URL + path
        if params:
            query = "&".join(
                f"{urllib.parse.quote_plus(k)}={urllib.parse.quote_plus(v)}"
                for k, v in params.items()
            )
            url = f"{url}?{query}"
        return url

    def _request_with_retry(self, url: str) -> Any:
        cfg = self._cfg
        last_status: int | None = None
        last_error: str = ""

        for attempt in range(1, cfg.max_retries + 1):
            self._bucket.acquire()
            try:
                data = self._do_get(url)
                return data

            except MlbApiError:
                # Non-retryable — propagate immediately
                raise

            except _RetryableHttpError as exc:
                last_status = exc.http_status
                last_error = str(exc)
                if attempt < cfg.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.debug(
                        "Retryable error, will retry",
                        extra={
                            "component": "ingest.client",
                            "url": url,
                            "attempt": attempt,
                            "http_status": last_status,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    time.sleep(delay)

            except Exception as exc:
                # Network-level errors (timeout, connection refused) are
                # treated as retryable.
                last_error = str(exc)
                if attempt < cfg.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.debug(
                        "Network error, will retry",
                        extra={
                            "component": "ingest.client",
                            "url": url,
                            "attempt": attempt,
                            "error": last_error,
                            "retry_delay_seconds": round(delay, 2),
                        },
                    )
                    time.sleep(delay)

        raise MlbApiRetryExhausted(
            f"All {cfg.max_retries} attempts failed for {url}: {last_error}",
            http_status=last_status,
            retries=cfg.max_retries,
        )

    def _do_get(self, url: str) -> Any:
        """Execute a single HTTP GET and return parsed JSON.

        Raises:
            MlbApiError: 4xx (not 429) or JSON decode failure
            _RetryableHttpError: 429 or 5xx
        """
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 429 or status >= 500:
                raise _RetryableHttpError(
                    f"HTTP {status} from {url}", http_status=status
                )
            # Other 4xx — non-retryable
            raise MlbApiError(
                f"HTTP {status} from {url}", http_status=status
            )
        except urllib.error.URLError as exc:
            # Network-level error — retryable
            raise _NetworkError(str(exc)) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MlbApiError(
                f"JSON decode error from {url}: {exc}", http_status=None
            )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with ±25% jitter.

        delay = base * (factor ^ (attempt-1)) * jitter_multiplier
        where jitter_multiplier ∈ [0.75, 1.25]
        """
        cfg = self._cfg
        base = cfg.retry_base_delay_seconds * (2 ** (attempt - 1))
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


# urllib.parse is used in _build_url — import it here to keep the top clean
import urllib.parse  # noqa: E402


__all__ = [
    "BASE_URL",
    "MlbApiError",
    "MlbApiRetryExhausted",
    "MlbStatsClient",
]
