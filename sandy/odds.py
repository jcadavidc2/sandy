"""Odds/value layer — TheOddsAPI v4 client, storage, matching and value log.

The user does NOT bet: this is an analytical layer that compares OUR calibrated
probabilities against the market to (a) surface "value" picks and (b) track the
hypothetical ROI of a flat 1-unit stake on them.

CREDIT FRUGALITY IS CRITICAL (500 credits/month):
  * we fetch ONLY sports that have pending (non-backtest, outcome NULL)
    predictions TODAY in our DB;
  * ONE fetch per sport per day — if odds.market_odds already has a row for
    that sport_key fetched today (UTC), the fetch is skipped;
  * a single /odds call with regions=eu & markets=totals,h2h costs 2 credits
    (1 per market per region); /v4/sports is free and used once per run to
    check whether a Colombia Primera A key exists.
  * every response's x-requests-remaining / x-requests-used headers are logged.

PROBABILITY SEMANTICS (important, easy to confuse):
  * edge / EV use the pick's OWN probability `prob` — the base model's
    calibrated probability of the pick's SIDE (e.g. P(over 8.5) = 0.61).
  * they must NOT use the 🤖 meta score, which is P(the pick is CORRECT) —
    a different quantity (a second-stage reliability score, not a side prob).
  * edge = prob - consensus implied_novig (median across books, vig removed);
    EV   = prob * (cuota - 1) - (1 - prob)   with cuota = best decimal price.

Market mapping (only what the feed actually covers):
  * our totals lines (goals/points/runs "over_X_5") ↔ API `totals` at the SAME
    point — the line must match exactly, otherwise no odds are attached.
    Totals conventions agree: MLB/NBA/NHL API totals include OT (ours do too);
    soccer totals are regular time (ours too).
  * our NBA `winner` ↔ API `h2h` (home/away).
  * our soccer `double_chance` (1X vs 2) is DERIVED per book from the 3-way
    soccer h2h (the API has no direct double-chance market on the basic call):
      cuota_1X = 1 / (1/cuota_home + 1/cuota_draw)
    — the standard combination formula (split the stake across home and draw so
    both payouts are equal); implied(1X) = implied(home) + implied(draw), and
    de-vigging the (1X, 2) pair proportionally equals nv(home)+nv(draw) from
    the 3-way de-vig, since both cover the whole outcome space exactly once.
    Stored as market='double_chance', sides 'home_or_draw' / 'away'.
    Costs NO extra credits: h2h is already in the fetched markets.
  * btts / corners have no feed → those picks stay odds-less.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import statistics
import unicodedata
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text

from sandy.betmeta import SPECS, _correct, market_threshold, score_candidate
from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

API_BASE = "https://api.the-odds-api.com/v4"
REGIONS = "eu"
MARKETS = "totals,h2h"
MIN_EDGE = 0.03  # 3 percentage points

# All verticals use the America/Los_Angeles calendar date of kickoff as
# match_date (see sandy/mls/parsers.py DISPLAY_TZ); MLB's game_date coincides
# with it for every real start time. So events are dated in PT for matching.
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")

# our league key → TheOddsAPI sport key. Liga Colombia has NO key on
# TheOddsAPI (checked live via /v4/sports on 2026-07-05: no `colombia` entry);
# discover_colombia_key() keeps probing the free endpoint so the day they add
# it, soccer_col starts fetching automatically.
SPORT_KEYS = {
    "mlb": "baseball_mlb",
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
    "mls": "soccer_usa_mls",
    "soccer_eng": "soccer_epl",
    "soccer_esp": "soccer_spain_la_liga",
    "soccer_mex": "soccer_mexico_ligamx",
    "worldcup": "soccer_fifa_world_cup",
}

# leagues where a ±1-day date fallback is allowed when team pair is unique
# (kickoff-date conventions can drift a day; US leagues play series so NO).
DATE_SLACK_LEAGUES = {"mls", "soccer_eng", "soccer_esp", "soccer_mex", "soccer_col", "worldcup"}

# NHL abbrev → full name (nhl.teams only stores the city, the API uses full
# names). Includes the 2025 Utah rename; aliases below catch the old name.
NHL_FULL = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken",
    "STL": "St Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}

# Manual aliases: league → {normalized API name → OUR value (code/abbrev/name)}.
# Only needed where normalization + fuzzy can't bridge the gap; grow as the
# unmatched log surfaces cases.
ALIASES: dict[str, dict[str, str]] = {
    # our predictions still carry the pre-2025 'OAK' code even though raw.teams
    # renamed the franchise to 'ATH' — the alias must beat the automatic map
    "mlb": {"athletics": "OAK", "oakland athletics": "OAK"},
    "nhl": {"utah hockey club": "UTA", "st louis blues": "STL"},
    # TheOddsAPI uses ESPN-style full names for NFL; defensive short forms only
    "nfl": {"washington": "WSH", "la rams": "LAR", "la chargers": "LAC",
            "washington football team": "WSH"},
    "worldcup": {"united states": "USA", "usa": "USA", "south korea": "South Korea",
                 "ivory coast": "Ivory Coast"},
    "mls": {"los angeles galaxy": "LA Galaxy", "los angeles fc": "LAFC",
            "dc united": "D.C. United", "montreal impact": "CF Montréal"},
    "soccer_mex": {"club america": "América", "chivas guadalajara": "Guadalajara",
                   "juarez": "FC Juarez", "u n a m pumas": "Pumas UNAM",
                   "unam pumas": "Pumas UNAM", "monterrey": "Monterrey"},
    "soccer_esp": {"athletic bilbao": "Athletic Club", "betis": "Real Betis"},
    "soccer_eng": {"wolves": "Wolverhampton Wanderers", "spurs": "Tottenham Hotspur",
                   "nottm forest": "Nottingham Forest"},
}

_STOP_TOKENS = {"fc", "cf", "sc", "afc", "club"}


# ------------------------------------------------------------------ client --
def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set (source ~/.sandy_env)")
    return key


def _get(path: str, **params) -> tuple[object, dict]:
    params = {"apiKey": _api_key(), **params}
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        logger.info("odds-api credits: remaining=%s used=%s (%s)", remaining, used, path)
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
              f"odds-api {path}: credits remaining={remaining} used={used}")
    return resp.json(), dict(resp.headers)


def list_sports() -> list[dict]:
    """GET /v4/sports — FREE (does not count against the credit quota)."""
    data, _h = _get("/sports", all="false")
    return data if isinstance(data, list) else []


def discover_colombia_key() -> str | None:
    """Return a Colombia Primera A sport key if TheOddsAPI ever lists one."""
    try:
        for s in list_sports():
            key = s.get("key", "")
            if "colombia" in key or "colombia" in (s.get("title") or "").lower():
                return key
    except Exception:
        logger.exception("sports list probe failed")
    return None


def sport_map() -> dict[str, str]:
    """SPORT_KEYS plus soccer_col if the free /sports probe finds a key."""
    m = dict(SPORT_KEYS)
    col = discover_colombia_key()
    if col:
        m["soccer_col"] = col
        print(f"[odds] Colombia Primera A available on TheOddsAPI as '{col}' — enabled")
    else:
        logger.info("no Colombia Primera A key on TheOddsAPI — soccer_col has no odds feed")
    return m


# --------------------------------------------------------------- name match --
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t not in _STOP_TOKENS)


def _team_map(league: str, conn) -> dict[str, str]:
    """normalized FULL/API-style name → OUR stored value (code/abbrev/name)."""
    out: dict[str, str] = {}
    if league == "mlb":
        for code, name in conn.execute(text("SELECT team_code, name FROM raw.teams")):
            out[_norm(name)] = code.strip()
    elif league == "nba":
        for abbrev, name in conn.execute(text("SELECT abbrev, name FROM nba.teams")):
            out[_norm(name)] = abbrev
    elif league == "nfl":
        # predictions store the abbrev; the API uses ESPN-style full names
        for abbrev, name in conn.execute(text("SELECT abbrev, name FROM nfl.teams")):
            out[_norm(name)] = abbrev
    elif league == "nhl":
        for abbrev, full in NHL_FULL.items():
            out[_norm(full)] = abbrev
        for abbrev, city in conn.execute(text("SELECT abbrev, name FROM nhl.teams")):
            out.setdefault(_norm(city), abbrev)
    else:
        spec = SPECS[league]
        extra = f" WHERE {spec['where']}" if spec.get("where") else ""
        for (name,) in conn.execute(text(
                f"SELECT DISTINCT home_team FROM {spec['table']}{extra} "
                f"UNION SELECT DISTINCT away_team FROM {spec['table']}{extra}")):
            if name:
                out[_norm(name)] = name
    return out


def _match_team(api_name: str, our_map: dict[str, str], aliases: dict[str, str]) -> str | None:
    n = _norm(api_name)
    if n in aliases:  # manual corrections beat the automatic map
        return aliases[n]
    if n in our_map:
        return our_map[n]
    # token-subset containment, e.g. "utah" ⊆ "utah mammoth" (unique hit only)
    toks = set(n.split())
    subset = [v for k, v in our_map.items()
              if toks and (toks <= set(k.split()) or set(k.split()) <= toks)]
    if len(set(subset)) == 1:
        return subset[0]
    # fuzzy last resort
    best, best_r = None, 0.0
    for k, v in our_map.items():
        r = difflib.SequenceMatcher(None, n, k).ratio()
        if r > best_r:
            best, best_r = v, r
    return best if best_r >= 0.8 else None


# -------------------------------------------------------------------- fetch --
def _fetched_today(conn, sport_key: str) -> bool:
    utc_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return bool(conn.execute(text(
        "SELECT 1 FROM odds.market_odds WHERE sport_key = :k AND fetched_at >= :t LIMIT 1"),
        {"k": sport_key, "t": utc_midnight}).fetchone())


def _pending_today(conn, league: str, day: date) -> int:
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    bt = "TRUE" if spec.get("no_backtest_col") else "NOT is_backtest"
    return conn.execute(text(f"""
        SELECT count(*) FROM {spec['table']}
        WHERE match_date = :d AND outcome_filled_at_utc IS NULL AND {bt}{extra}
    """), {"d": day}).scalar() or 0


def _event_rows(league: str, sport_key: str, event: dict) -> list[dict]:
    """Flatten one API event into market_odds rows with implied + no-vig."""
    rows: list[dict] = []
    home, away = event.get("home_team"), event.get("away_team")
    for bm in event.get("bookmakers") or []:
        book = bm.get("key")
        for mk in bm.get("markets") or []:
            market = mk.get("key")
            if market not in ("h2h", "totals", "alternate_totals"):
                continue
            if market == "alternate_totals":
                market = "totals"  # same rows/matching as main totals — just more lines
            groups: dict[float | None, list[dict]] = {}
            for oc in mk.get("outcomes") or []:
                pt = oc.get("point")
                groups.setdefault(None if pt is None else round(float(pt), 2), []).append(oc)
            for pt, ocs in groups.items():
                implied = {}
                for oc in ocs:
                    price = float(oc.get("price") or 0)
                    if price > 1.0:
                        implied[oc["name"]] = 1.0 / price
                total = sum(implied.values())
                # a full outcome set: 2 for totals/US h2h, 3 for soccer h2h
                expected = 3 if (market == "h2h" and sport_key.startswith("soccer")) else 2
                complete = len(implied) >= expected and total > 0
                for oc in ocs:
                    price = float(oc.get("price") or 0)
                    if price <= 1.0:
                        continue
                    name = oc["name"]
                    if market == "totals":
                        side = name.lower()
                        if side not in ("over", "under"):
                            continue
                    else:
                        side = "home" if name == home else ("away" if name == away else None)
                        if side is None:  # Draw: participates in the no-vig sum only
                            continue
                    rows.append({
                        "sport_key": sport_key, "league": league, "event_id": event.get("id"),
                        "event_home": home, "event_away": away,
                        "commence_utc": event.get("commence_time"),
                        "book": book, "market": market, "point": pt, "side": side,
                        "price": price, "implied": implied[name],
                        "implied_novig": (implied[name] / total) if complete else None,
                    })
                # DERIVED double chance from the complete 3-way soccer h2h:
                # cuota_1X = 1/(1/cuota_H + 1/cuota_X)  (combination formula);
                # nv(1X) = (iH+iX)/total == nv(H)+nv(X); the '2' side reuses the
                # h2h away price. No extra API credits — pure derivation.
                if market == "h2h" and expected == 3 and complete:
                    i_by_side = {}
                    for oc in ocs:
                        nm = oc["name"]
                        key = ("home" if nm == home else
                               "away" if nm == away else "draw")
                        if nm in implied:
                            i_by_side[key] = (implied[nm], float(oc["price"]))
                    if set(i_by_side) == {"home", "draw", "away"}:
                        i1x = i_by_side["home"][0] + i_by_side["draw"][0]
                        base = {"sport_key": sport_key, "league": league,
                                "event_id": event.get("id"), "event_home": home,
                                "event_away": away,
                                "commence_utc": event.get("commence_time"),
                                "book": book, "market": "double_chance", "point": None}
                        rows.append({**base, "side": "home_or_draw",
                                     "price": 1.0 / i1x, "implied": i1x,
                                     "implied_novig": i1x / total})
                        rows.append({**base, "side": "away",
                                     "price": i_by_side["away"][1],
                                     "implied": i_by_side["away"][0],
                                     "implied_novig": i_by_side["away"][0] / total})
    # De-dup (book, market, point, side): the main line usually repeats inside
    # alternate_totals — keep the best (max) price for the bettor.
    dedup: dict[tuple, dict] = {}
    for r in rows:
        k = (r["book"], r["market"], r["point"], r["side"])
        if k not in dedup or r["price"] > dedup[k]["price"]:
            dedup[k] = r
    return list(dedup.values())


def fetch_league(league: str, sport_key: str, engine) -> dict:
    """One frugal fetch for a league (skipped if already fetched today, UTC)."""
    with engine.begin() as conn:
        if _fetched_today(conn, sport_key):
            logger.info("%s (%s): already fetched today — skipping (credit frugality)",
                        league, sport_key)
            return {"league": league, "skipped": "already_fetched_today"}
    events, _headers = _get(f"/sports/{sport_key}/odds",
                            regions=REGIONS, markets=MARKETS, oddsFormat="decimal")
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    skipped_live = 0
    for ev in events or []:
        # PRE-MATCH ONLY: the API also returns commenced games with IN-PLAY
        # odds — comparing our pre-game probs against those fabricates edge.
        ct = ev.get("commence_time")
        try:
            started = datetime.fromisoformat(str(ct).replace("Z", "+00:00")) <= now
        except (TypeError, ValueError):
            started = False
        if started:
            skipped_live += 1
            continue
        rows.extend(_event_rows(league, sport_key, ev))
    if skipped_live:
        logger.info("%s: skipped %d already-commenced events (in-play odds)",
                    league, skipped_live)
    if rows:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO odds.market_odds
                    (sport_key, league, event_id, event_home, event_away, commence_utc,
                     book, market, point, side, price, implied, implied_novig)
                VALUES (:sport_key, :league, :event_id, :event_home, :event_away,
                        :commence_utc, :book, :market, :point, :side, :price,
                        :implied, :implied_novig)
            """), rows)
    logger.info("%s: stored %d odds rows across %d events", league, len(rows), len(events or []))
    return {"league": league, "events": len(events or []), "rows": len(rows)}


# Alternate totals are only sold PER EVENT on this API (1 credit per event).
# Fetch them TARGETED: only pending events fetched today, capped, so credit
# spend stays ~<=ALT_MAX_EVENTS/day on top of the bulk fetch.
ALT_MAX_EVENTS = 10


def fetch_alternates(league: str, sport_key: str, engine, max_events: int = ALT_MAX_EVENTS) -> dict:
    """Per-event alternate_totals for today's PENDING events (more lines get
    cuotas → bigger pools for Valor/Portafolios). Skips events that already
    have alternate coverage today (>=4 distinct total lines)."""
    with engine.begin() as conn:
        evs = conn.execute(text("""
            SELECT event_id, MIN(commence_utc) AS c
            FROM odds.market_odds
            WHERE league = :lg AND commence_utc > now()
              AND fetched_at::date = CURRENT_DATE
            GROUP BY event_id
            HAVING COUNT(DISTINCT point) FILTER (WHERE market = 'totals') < 4
            ORDER BY 2 LIMIT :cap"""), {"lg": league, "cap": max_events}).fetchall()
    done, rows_total = 0, 0
    for eid, _c in evs:
        try:
            ev, _h = _get(f"/sports/{sport_key}/events/{eid}/odds",
                          regions=REGIONS, markets="alternate_totals", oddsFormat="decimal")
        except Exception as exc:  # noqa: BLE001 — best-effort per event
            logger.info("%s alt %s: %s", league, eid, exc)
            continue
        rows = _event_rows(league, sport_key, ev or {})
        if rows:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO odds.market_odds
                        (sport_key, league, event_id, event_home, event_away, commence_utc,
                         book, market, point, side, price, implied, implied_novig)
                    VALUES (:sport_key, :league, :event_id, :event_home, :event_away,
                            :commence_utc, :book, :market, :point, :side, :price,
                            :implied, :implied_novig)
                """), rows)
            rows_total += len(rows)
        done += 1
    logger.info("%s: alternates for %d events (%d rows)", league, done, rows_total)
    return {"league": league, "alt_events": done, "alt_rows": rows_total}


# -------------------------------------------------------------------- match --
def _rederive_double_chance(conn, league: str, event_id: str) -> None:
    """Rebuild one event's derived double_chance rows from its (already
    side-flipped) h2h rows. Needed when an event matches SWAPPED vs our row:
    the fetch-time 1X was API-home-or-draw, which after the swap would mean
    OUR X2 — not flippable in place. The draw implied is recovered exactly via
    total = implied/implied_novig; i_draw = total - i_home - i_away."""
    conn.execute(text("""
        DELETE FROM odds.market_odds
        WHERE market = 'double_chance' AND league = :lg AND event_id = :eid
    """), {"lg": league, "eid": event_id})
    conn.execute(text("""
        WITH h2h AS (
            SELECT sport_key, league, event_id, event_home, event_away, commence_utc,
                   book, fetched_at, matched, match_date, our_home, our_away,
                   MAX(CASE WHEN side='home' THEN implied END) AS ih,
                   MAX(CASE WHEN side='away' THEN implied END) AS ia,
                   MAX(CASE WHEN side='away' THEN price END)   AS pa,
                   MAX(CASE WHEN side='home' THEN implied/implied_novig END) AS total
            FROM odds.market_odds
            WHERE market='h2h' AND league = :lg AND event_id = :eid
              AND implied_novig IS NOT NULL
            GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
        ), calc AS (
            SELECT *, (total - ih - ia) AS ix FROM h2h
            WHERE ih IS NOT NULL AND ia IS NOT NULL AND total IS NOT NULL
              AND (total - ih - ia) > 0
        )
        INSERT INTO odds.market_odds
            (fetched_at, sport_key, league, event_id, event_home, event_away,
             commence_utc, book, market, point, side, price, implied, implied_novig,
             matched, match_date, our_home, our_away)
        SELECT fetched_at, sport_key, league, event_id, event_home, event_away,
               commence_utc, book, 'double_chance', NULL, s.side,
               CASE WHEN s.side='home_or_draw' THEN 1.0/(ih+ix) ELSE pa END,
               CASE WHEN s.side='home_or_draw' THEN (ih+ix) ELSE ia END,
               CASE WHEN s.side='home_or_draw' THEN (ih+ix)/total ELSE ia/total END,
               matched, match_date, our_home, our_away
        FROM calc CROSS JOIN (VALUES ('home_or_draw'), ('away')) AS s(side)
    """), {"lg": league, "eid": event_id})


def match_league(league: str, engine, days_back: int = 7) -> dict:
    """Link stored API events to OUR prediction rows (date + both teams).

    Re-runs over the last `days_back` days of fetches, so alias fixes take
    effect without refetching. Unmatched events are logged, never fatal.
    """
    spec = SPECS[league]
    extra = f" AND {spec['where']}" if spec.get("where") else ""
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    with engine.begin() as conn:
        events = conn.execute(text("""
            SELECT DISTINCT event_id, event_home, event_away, commence_utc
            FROM odds.market_odds WHERE league = :lg AND fetched_at >= :t
        """), {"lg": league, "t": since}).fetchall()
        if not events:
            return {"league": league, "events": 0, "matched": 0, "unmatched": []}
        our_map = _team_map(league, conn)
        aliases = ALIASES.get(league, {})
        dates = [ev.commence_utc.astimezone(DISPLAY_TZ).date() for ev in events]
        ours = conn.execute(text(f"""
            SELECT DISTINCT match_date, home_team, away_team FROM {spec['table']}
            WHERE match_date BETWEEN :a AND :b{extra}
        """), {"a": min(dates) - timedelta(days=1), "b": max(dates) + timedelta(days=1)}).fetchall()
        games = {(g.match_date, (g.home_team or "").strip(), (g.away_team or "").strip())
                 for g in ours}
        our_days = {g[0] for g in games}
        matched, in_window, unmatched = 0, 0, []
        for ev, d in zip(events, dates):
            slack = [0, -1, 1] if league in DATE_SLACK_LEAGUES else [0]
            # only events on days we actually predicted count against the rate
            # (the API also returns days ahead we have no rows for yet)
            windowed = d in our_days
            in_window += windowed
            h = _match_team(ev.event_home, our_map, aliases)
            a = _match_team(ev.event_away, our_map, aliases)
            hit, swapped = None, False
            if h and a:
                for dd in slack:
                    cand = d + timedelta(days=dd)
                    if (cand, h, a) in games:
                        hit = (cand, h, a)
                        break
                    if (cand, a, h) in games:
                        hit, swapped = (cand, a, h), True
                        break
            if not hit:
                if windowed:
                    unmatched.append(f"{ev.event_home} vs {ev.event_away} ({d})")
                    logger.warning("%s: unmatched event %s vs %s (%s) → mapped (%s, %s)",
                                   league, ev.event_home, ev.event_away, d, h, a)
                continue
            conn.execute(text("""
                UPDATE odds.market_odds
                SET matched = TRUE, match_date = :d, our_home = :h, our_away = :a
                WHERE league = :lg AND event_id = :eid
            """), {"d": hit[0], "h": hit[1], "a": hit[2], "lg": league, "eid": ev.event_id})
            if swapped:  # keep h2h side semantics relative to OUR home team
                conn.execute(text("""
                    UPDATE odds.market_odds
                    SET side = CASE side WHEN 'home' THEN 'away' WHEN 'away' THEN 'home'
                               ELSE side END
                    WHERE league = :lg AND event_id = :eid AND market = 'h2h'
                """), {"lg": league, "eid": ev.event_id})
                if _is_soccer(league):  # 1X can't be flipped — rebuild from h2h
                    _rederive_double_chance(conn, league, ev.event_id)
            matched += 1
    rate = matched / in_window if in_window else None
    logger.info("%s: matched %d/%d in-window events (%.0f%%; %d fetched total)",
                league, matched, in_window, 100 * (rate or 0), len(events))
    return {"league": league, "events": len(events), "in_window": in_window,
            "matched": matched, "rate": rate, "unmatched": unmatched}


# -------------------------------------------------------------------- value --
def _is_soccer(league: str) -> bool:
    return league in ("mls", "worldcup") or league.startswith("soccer")


def market_to_api(league: str, market: str) -> tuple[str, float | None] | None:
    """Our market key → (stored market, point) — or None when no feed covers it."""
    _pcol, kind, line = SPECS[league]["markets"][market]
    if kind in ("goals", "points", "runs"):
        return "totals", round(float(line), 2)
    if kind == "winner":
        return "h2h", None
    if kind == "result" and _is_soccer(league):
        # derived from the 3-way h2h (NHL's home-or-tie is regulation-time and
        # has NO 3-way feed here → stays odds-less)
        return "double_chance", None
    return None  # NHL result / btts / corners: no odds feed


def pick_side(kind: str, p: float) -> str:
    if kind == "winner":
        return "home" if p >= 0.5 else "away"
    if kind == "result":
        return "home_or_draw" if p >= 0.5 else "away"
    return "over" if p >= 0.5 else "under"


def _agg(rows) -> tuple[float, float | None, int] | None:
    """(best price, median implied_novig, n books) over latest-per-book rows."""
    if not rows:
        return None
    best = max(r.price for r in rows)
    novig = [r.implied_novig for r in rows if r.implied_novig is not None]
    return best, (statistics.median(novig) if novig else None), len(rows)


def best_odds_for(league: str, home: str, away: str, market: str,
                  line: float | None, side: str, day: date,
                  engine=None) -> tuple[float, float | None, int] | None:
    """Best decimal price across books + consensus (median) implied_novig for
    one matched game/market/side. `market` is the API market ('totals'/'h2h');
    `line` the totals point (None for h2h). Returns None when no odds stored."""
    engine = engine or create_engine(load_config())
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (event_id, book) price, implied_novig
            FROM odds.market_odds
            WHERE matched AND league = :lg AND our_home = :h AND our_away = :a
              AND match_date = :d AND market = :m AND side = :s
              AND (point = :pt OR (CAST(:pt AS REAL) IS NULL AND point IS NULL))
            ORDER BY event_id, book, fetched_at DESC
        """), {"lg": league, "h": home, "a": away, "d": day, "m": market,
               "s": side, "pt": line}).fetchall()
    return _agg(rows)


def odds_index(league: str, start: date, end: date, engine=None
               ) -> dict[tuple, tuple[float, float | None, int]]:
    """Bulk lookup for the dashboard: (match_date, home, away, api_market,
    point, side) → (best price, median implied_novig, n books). One query."""
    engine = engine or create_engine(load_config())
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (event_id, book, market, point, side)
                   match_date, our_home, our_away, market, point, side,
                   price, implied_novig
            FROM odds.market_odds
            WHERE matched AND league = :lg AND match_date BETWEEN :a AND :b
            ORDER BY event_id, book, market, point, side, fetched_at DESC
        """), {"lg": league, "a": start, "b": end}).fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        pt = None if r.point is None else round(float(r.point), 2)
        groups.setdefault(
            (r.match_date, r.our_home, r.our_away, r.market, pt, r.side), []).append(r)
    return {k: _agg(v) for k, v in groups.items()}


def value_stats(prob: float, hit: tuple[float, float | None, int] | None
                ) -> dict[str, float | None]:
    """cuota / mercado % / edge / EV for a pick. `prob` is the pick's own
    base-model calibrated SIDE probability (NOT the 🤖 meta P(correct))."""
    out = {"cuota": None, "mercado %": None, "edge": None, "EV": None}
    if not hit:
        return out
    cuota, novig, _n = hit
    out["cuota"] = round(cuota, 2)
    out["EV"] = round(prob * (cuota - 1) - (1 - prob), 4)
    if novig is not None:
        out["mercado %"] = round(novig, 4)
        out["edge"] = round(prob - novig, 4)
    return out


# ---------------------------------------------------------------- value log --
def log_value_picks(day: date | None = None, cfg: Config | None = None,
                    min_edge: float = MIN_EDGE) -> list[dict]:
    """Insert today's meta-approved picks with edge >= min_edge into
    odds.value_log (idempotent via the unique index). Returns what it saw."""
    cfg = cfg or load_config()
    day = day or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    found: list[dict] = []
    for league in SPECS:
        idx = odds_index(league, day, day, engine)
        if not idx:
            continue
        spec = SPECS[league]
        extra = f" AND {spec['where']}" if spec.get("where") else ""
        bt = "TRUE" if spec.get("no_backtest_col") else "NOT is_backtest"
        with engine.begin() as conn:
            rows = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date = :d AND outcome_filled_at_utc IS NULL AND {bt}{extra}
                ORDER BY match_date, id"""), {"d": day}).fetchall()
        for r in rows:
            rd = dict(r._mapping)
            home, away = (rd["home_team"] or "").strip(), (rd["away_team"] or "").strip()
            for market, (pcol, kind, line) in spec["markets"].items():
                p = rd.get(pcol)
                mapping = market_to_api(league, market)
                if p is None or mapping is None:
                    continue
                p = float(p)
                prob = p if p >= 0.5 else 1 - p  # the pick's own side prob (NOT 🤖)
                mp = score_candidate(league, cfg, rd, market, p)
                thr = market_threshold(league, cfg, market)
                if mp is None or thr is None or mp < thr:
                    continue  # only meta-approved (✅) picks are candidates
                api_market, pt = mapping
                side = pick_side(kind, p)
                st = value_stats(prob, idx.get((day, home, away, api_market, pt, side)))
                if st["edge"] is None or st["edge"] < min_edge:
                    continue
                rec = {"date": day, "league": league, "home": home, "away": away,
                       "market": market, "side": side, "line": pt, "prob": round(prob, 4),
                       "cuota": st["cuota"], "edge": st["edge"], "ev": st["EV"]}
                with engine.begin() as conn:
                    dup = conn.execute(text("""
                        SELECT 1 FROM odds.value_log
                        WHERE date = :date AND league = :league AND home = :home
                          AND away = :away AND market = :market AND side = :side
                          AND COALESCE(line, -1.0) = COALESCE(CAST(:line AS REAL), -1.0)
                        LIMIT 1
                    """), rec).fetchone()
                    if not dup:
                        conn.execute(text("""
                            INSERT INTO odds.value_log
                                (date, league, home, away, market, side, line, prob,
                                 cuota, edge, ev, stake)
                            VALUES (:date, :league, :home, :away, :market, :side, :line,
                                    :prob, :cuota, :edge, :ev, 1.0)
                        """), rec)
                found.append(rec)
                logger.info("VALUE pick %s %s vs %s %s %s: prob=%.2f cuota=%.2f edge=%.3f",
                            league, home, away, market, side, prob, st["cuota"], st["edge"])
    return found


def reconcile_value_log(cfg: Config | None = None) -> dict:
    """Fill result + units on settled value picks from OUR predictions'
    reconciled outcomes (win: +(cuota-1), lose: -1; flat 1u stake)."""
    cfg = cfg or load_config()
    engine = create_engine(cfg)
    settled, still_open = 0, 0
    with engine.begin() as conn:
        open_rows = conn.execute(text(
            "SELECT * FROM odds.value_log WHERE result IS NULL ORDER BY date")).fetchall()
        for v in open_rows:
            spec = SPECS.get(v.league)
            if not spec:
                continue
            extra = f" AND {spec['where']}" if spec.get("where") else ""
            g = conn.execute(text(f"""
                SELECT * FROM {spec['table']}
                WHERE match_date = :d AND btrim(home_team) = :h AND btrim(away_team) = :a
                  AND outcome_filled_at_utc IS NOT NULL{extra}
                ORDER BY id LIMIT 1
            """), {"d": v.date, "h": v.home, "a": v.away}).fetchone()
            if g is None:
                still_open += 1
                continue
            _pcol, kind, _line = spec["markets"][v.market]
            # _correct scores "the pick at prob p"; encode our logged side as p
            p_side = 0.99 if v.side in ("over", "home", "home_or_draw") else 0.01
            won = _correct(dict(g._mapping), kind,
                           None if v.line is None else float(v.line), p_side)
            if won is None:
                still_open += 1
                continue
            conn.execute(text("""
                UPDATE odds.value_log
                SET result = :r, units = :u, settled_at = now()
                WHERE id = :id
            """), {"r": "win" if won else "lose",
                   "u": round((v.cuota - 1.0) * v.stake, 4) if won else -v.stake,
                   "id": v.id})
            settled += 1
    logger.info("value_log reconcile: settled=%d still_open=%d", settled, still_open)
    return {"settled": settled, "open": still_open}


def roi_frame(cfg: Config | None = None):
    """Settled value_log rows for the 💰 page (date-ordered)."""
    import pandas as pd
    engine = create_engine(cfg or load_config())
    with engine.begin() as conn:
        return pd.read_sql(text("""
            SELECT date, league, home, away, market, side, line, prob, cuota,
                   edge, stake, result, units
            FROM odds.value_log ORDER BY date, id
        """), conn)


# -------------------------------------------------------------------- daily --
def run_daily(day: date | None = None, cfg: Config | None = None) -> dict:
    """The whole frugal daily pass: fetch (only sports with pending predictions
    today, once per day) → match → value log → reconcile. Never raises for a
    single league's failure; raises only if EVERYTHING is broken (no DB)."""
    cfg = cfg or load_config()
    day = day or datetime.now(DISPLAY_TZ).date()
    engine = create_engine(cfg)
    smap = sport_map()  # free /sports probe (also the Colombia check)
    report: dict = {"day": str(day), "fetched": [], "match": {}, "value": [], "reconcile": {}}
    for league, sport_key in smap.items():
        try:
            with engine.begin() as conn:
                pending = _pending_today(conn, league, day)
            if not pending:
                logger.info("%s: no pending predictions today — no fetch (credit frugality)",
                            league)
                continue
            report["fetched"].append(fetch_league(league, sport_key, engine))
            # Targeted alternate totals (per-event endpoint): more lines get a
            # cuota. Self-limiting: skips events already holding >=4 lines.
            report["fetched"].append(fetch_alternates(league, sport_key, engine))
            report["match"][league] = match_league(league, engine)
        except Exception:
            logger.exception("odds daily: league %s failed (non-fatal)", league)
    try:
        report["value"] = log_value_picks(day, cfg)
    except Exception:
        logger.exception("value log step failed (non-fatal)")
    try:
        report["reconcile"] = reconcile_value_log(cfg)
    except Exception:
        logger.exception("reconcile step failed (non-fatal)")
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Sandy odds/value layer (TheOddsAPI)")
    ap.add_argument("cmd", choices=["daily", "match", "reconcile", "report"],
                    help="daily = fetch+match+value+reconcile (frugal)")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today in America/Los_Angeles)")
    args = ap.parse_args()
    day = date.fromisoformat(args.date) if args.date else None
    if args.cmd == "daily":
        rep = run_daily(day)
        print(json.dumps(rep, default=str, indent=2))
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{ts}] ODDS DAILY COMPLETE")
    elif args.cmd == "match":
        engine = create_engine(load_config())
        for lg in SPECS:
            r = match_league(lg, engine)
            if r["events"]:
                print(json.dumps(r, default=str))
    elif args.cmd == "reconcile":
        print(json.dumps(reconcile_value_log(), default=str))
    else:
        df = roi_frame()
        settled = df[df["units"].notna()]
        units = settled["units"].sum() if len(settled) else 0.0
        staked = settled["stake"].sum() if len(settled) else 0.0
        print(f"value picks logged: {len(df)} | settled: {len(settled)} | "
              f"units: {units:+.2f} | ROI: {(units / staked * 100) if staked else 0:.1f}%")


if __name__ == "__main__":
    main()
