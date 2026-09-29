"""ESPN public API client for MLS (keyless).

Endpoints (verified live 2026-07-04):
  - scoreboard: /apis/site/v2/sports/soccer/usa.1/scoreboard?dates=YYYYMMDD
      → events with status + competitors + final scores, historical back years.
  - summary:    /apis/site/v2/sports/soccer/usa.1/summary?event=<id>
      → boxscore.teams[].statistics incl. wonCorners, totalShots, possessionPct.

Same defensive posture as the API-Football client: token-bucket throttle
(polite 1 req/s — no documented cap, but we're a guest), retries with backoff,
forced IPv4 (this EC2 box has broken IPv6 egress).
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

BASE_TMPL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{code}"
BASE = BASE_TMPL.format(sport="soccer", code="usa.1")  # MLS default
MAX_RPS = 1.0
RETRIES = 3
# None = let urllib send its default ("Python-urllib/3.x"), which ESPN accepts.
# Verified 2026-09-28 (Akamai edge): no-UA / Python-urllib / python-requests / curl → 200;
# "sandy/1.0", "sandy-bot", "Mozilla/5.0 (compatible; …)" and a Chrome UA → 403.
USER_AGENTS: tuple[str | None, ...] = (None, "Python-urllib/3.11", "python-requests/2.32")


_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


class EspnClient:
    """League-generic ESPN client: pass code='esp.1' (soccer) or sport='basketball',
    code='nba'. Defaults to MLS for backwards compatibility."""

    def __init__(self, max_rps: float = MAX_RPS, retries: int = RETRIES,
                 sport: str = "soccer", code: str = "usa.1"):
        self._base = BASE_TMPL.format(sport=sport, code=code)
        self._min_interval = 1.0 / max_rps
        self._retries = retries
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, url: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self._retries):
            self._throttle()
            # User-Agent rotation: ESPN's edge started answering HTTP 403 to the
            # custom "sandy/1.0" UA on 2026-08-05 (and to browser UAs without the
            # rest of a browser's headers) while the plain urllib default keeps
            # getting 200 — verified 2026-09-28. First attempt = urllib default;
            # later attempts rotate through USER_AGENTS so a future block on one
            # string degrades to a retry instead of killing every vertical.
            ua = USER_AGENTS[attempt % len(USER_AGENTS)]
            headers = {"Accept": "application/json"}
            if ua:
                headers["User-Agent"] = ua
            try:
                socket.getaddrinfo = _ipv4_getaddrinfo  # force IPv4 for this call
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        return json.loads(resp.read().decode())
                finally:
                    socket.getaddrinfo = _orig_getaddrinfo
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                backoff = 2.0 * (attempt + 1)
                logger.warning("ESPN request failed (attempt %s, UA=%r): %s — retrying in %.0fs",
                               attempt + 1, ua or "urllib-default", e, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"ESPN request failed after {self._retries} attempts: {url}: {last_err}")

    def scoreboard(self, yyyymmdd: str) -> dict:
        """All MLS events on a calendar date (ESPN uses US/Eastern-ish day buckets)."""
        return self._get(f"{self._base}/scoreboard?dates={yyyymmdd}")

    def summary(self, event_id: int) -> dict:
        """Match summary (boxscore team statistics incl. corners) for one event."""
        return self._get(f"{self._base}/summary?event={event_id}")
