"""Game-time weather covariates for the MLB / NFL meta-models.

Source: open-meteo (keyless, free — https://open-meteo.com).
  - Historical (measured):  archive-api.open-meteo.com/v1/archive
    (hourly temperature_2m / wind_speed_10m / precipitation; ~5-day lag).
  - Recent + pending games: api.open-meteo.com/v1/forecast with past_days,
    so today's slate gets a forecast row and the last few days get measured
    values before the archive catches up. The daily job later REPLACES
    forecast rows with archive rows ('forecast' -> 'hist') once available.

Efficiency: fetches are grouped PER STADIUM PER SEASON DATE-RANGE (one archive
call covers months of hourly data), never per game. Games are then joined to
the hourly grid by the hour nearest kickoff (UTC). Feature values:
  temp_c / wind_kmh  at the hour nearest first pitch / kickoff
  precip_mm          summed over the kickoff hour + the next 2 (game window)
Fixed-roof (dome) games get NEUTRAL indoor constants (21 C, 0 km/h, 0 mm) and
is_dome=TRUE — the flag itself is the covariate. Retractable-roof stadiums
store REAL outdoor readings (roof position is not knowable historically);
their 0.5 "semi-dome" score comes from the RETRACTABLE sets at feature time
(see roof_score(), used by both the bulk and live betmeta paths).

Known approximation (documented): games are mapped to the HOME team's stadium
for that season. The handful of neutral-site games (London/Mexico City/Seoul
series, ~5 per league-season) therefore get home-stadium weather — noise well
under 0.3% of rows.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from sandy.config import Config, load_config
from sandy.db import create_engine

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,wind_speed_10m,precipitation"
DOME_NEUTRAL = (21.0, 0.0, 0.0)  # fixed-roof games: mild indoor constants
ARCHIVE_LAG_DAYS = 6             # archive trails ~5 days; younger dates -> forecast API
PRECIP_WINDOW_H = 3              # precip = sum over kickoff hour + next 2
THROTTLE_S = 0.8                 # ~1.2 req/s, gentle on open-meteo + this box

# ---------------------------------------------------------------- stadiums --
# Coordinates from the stadiums' Wikipedia pages / MLB StatsAPI venue data
# (cross-checked against raw.games venue_id majorities per team-season).
# roof: 'open' | 'retractable' | 'dome' (fixed).
MLB_PARKS: dict[str, tuple[float, float, str]] = {
    "ATL": (33.8908, -84.4678, "open"),         # Truist Park
    "AZ":  (33.4453, -112.0667, "retractable"), # Chase Field
    "BAL": (39.2839, -76.6217, "open"),         # Camden Yards
    "BOS": (42.3467, -71.0972, "open"),         # Fenway Park
    "CHC": (41.9484, -87.6553, "open"),         # Wrigley Field
    "CIN": (39.0975, -84.5066, "open"),         # Great American Ball Park
    "CLE": (41.4962, -81.6852, "open"),         # Progressive Field
    "COL": (39.7559, -104.9942, "open"),        # Coors Field
    "CWS": (41.8299, -87.6338, "open"),         # Rate Field
    "DET": (42.3390, -83.0485, "open"),         # Comerica Park
    "HOU": (29.7573, -95.3555, "retractable"),  # Daikin Park (ex Minute Maid)
    "KC":  (39.0517, -94.4803, "open"),         # Kauffman Stadium
    "LAA": (33.8003, -117.8827, "open"),        # Angel Stadium
    "LAD": (34.0739, -118.2400, "open"),        # Dodger Stadium
    "MIA": (25.7781, -80.2196, "retractable"),  # loanDepot park
    "MIL": (43.0280, -87.9712, "retractable"),  # American Family Field
    "MIN": (44.9817, -93.2776, "open"),         # Target Field
    "NYM": (40.7571, -73.8458, "open"),         # Citi Field
    "NYY": (40.8296, -73.9262, "open"),         # Yankee Stadium
    "OAK": (37.7516, -122.2005, "open"),        # Oakland Coliseum (A's 2022-24)
    "ATH": (38.5804, -121.5133, "open"),        # Sutter Health Park, Sacramento (A's 2025+)
    "PHI": (39.9061, -75.1665, "open"),         # Citizens Bank Park
    "PIT": (40.4469, -80.0057, "open"),         # PNC Park
    "SD":  (32.7076, -117.1570, "open"),        # Petco Park
    "SEA": (47.5914, -122.3325, "retractable"), # T-Mobile Park (umbrella canopy)
    "SF":  (37.7786, -122.3893, "open"),        # Oracle Park
    "STL": (38.6226, -90.1928, "open"),         # Busch Stadium
    "TB":  (27.7683, -82.6534, "dome"),         # Tropicana Field (2023-24, back 2026)
    "TEX": (32.7473, -97.0842, "retractable"),  # Globe Life Field
    "TOR": (43.6414, -79.3894, "retractable"),  # Rogers Centre
    "WSH": (38.8730, -77.0074, "open"),         # Nationals Park
}
# (team, season) exceptions — relocations. The A's need none: they changed
# CODE (OAK 2022-24 Oakland -> ATH 2025+ Sacramento), both mapped above.
MLB_PARK_OVERRIDES: dict[tuple[str, int], tuple[float, float, str]] = {
    # Rays 2025 at Steinbrenner Field, Tampa (open air) after Hurricane Milton
    # wrecked the Trop's roof; back at Tropicana Field for 2026 (confirmed by
    # raw.games venue_id: 12 -> 2523 (2025) -> 12 (2026)).
    ("TB", 2025): (27.9803, -82.5067, "open"),
}
# NFL stadiums (2022+). Note vs the rough spec list: Atlanta, Indianapolis and
# Houston are RETRACTABLE (Mercedes-Benz, Lucas Oil, NRG), not fixed — kept
# factual here; retractable scores 0.5 at feature time either way.
NFL_STADIUMS: dict[str, tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"),  # State Farm Stadium
    "ATL": (33.7554, -84.4010, "retractable"),   # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227, "open"),          # M&T Bank Stadium
    "BUF": (42.7738, -78.7870, "open"),          # Highmark Stadium
    "CAR": (35.2258, -80.8528, "open"),          # Bank of America Stadium
    "CHI": (41.8623, -87.6167, "open"),          # Soldier Field
    "CIN": (39.0954, -84.5160, "open"),          # Paycor Stadium
    "CLE": (41.5061, -81.6995, "open"),          # Huntington Bank Field
    "DAL": (32.7473, -97.0945, "retractable"),   # AT&T Stadium
    "DEN": (39.7439, -105.0201, "open"),         # Empower Field at Mile High
    "DET": (42.3400, -83.0456, "dome"),          # Ford Field
    "GB":  (44.5013, -88.0622, "open"),          # Lambeau Field
    "HOU": (29.6847, -95.4107, "retractable"),   # NRG Stadium
    "IND": (39.7601, -86.1639, "retractable"),   # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373, "open"),          # EverBank Stadium
    "KC":  (39.0489, -94.4839, "open"),          # GEHA Field at Arrowhead
    "LAC": (33.9535, -118.3392, "dome"),         # SoFi Stadium (fixed canopy)
    "LAR": (33.9535, -118.3392, "dome"),         # SoFi Stadium
    "LV":  (36.0909, -115.1833, "dome"),         # Allegiant Stadium
    "MIA": (25.9580, -80.2389, "open"),          # Hard Rock Stadium (field open)
    "MIN": (44.9738, -93.2577, "dome"),          # U.S. Bank Stadium
    "NE":  (42.0910, -71.2643, "open"),          # Gillette Stadium
    "NO":  (29.9511, -90.0812, "dome"),          # Caesars Superdome
    "NYG": (40.8135, -74.0745, "open"),          # MetLife Stadium
    "NYJ": (40.8135, -74.0745, "open"),          # MetLife Stadium
    "PHI": (39.9008, -75.1675, "open"),          # Lincoln Financial Field
    "PIT": (40.4468, -80.0158, "open"),          # Acrisure Stadium
    "SEA": (47.5952, -122.3316, "open"),         # Lumen Field (stands covered, field open)
    "SF":  (37.4030, -121.9700, "open"),         # Levi's Stadium
    "TB":  (27.9759, -82.5033, "open"),          # Raymond James Stadium
    "TEN": (36.1665, -86.7713, "open"),          # Nissan Stadium
    "WSH": (38.9077, -76.8645, "open"),          # Northwest Stadium
}
# Retractable-roof sets by league — drives the 0.5 wx_dome score. Derived from
# the tables above; kept as an explicit constant because roof_score() must be
# computable from a stored game_weather row alone (bulk/live equality).
RETRACTABLE = {
    "mlb": frozenset(t for t, (_a, _o, r) in MLB_PARKS.items() if r == "retractable"),
    "nfl": frozenset(t for t, (_a, _o, r) in NFL_STADIUMS.items() if r == "retractable"),
}


def park_for(league: str, team, season) -> tuple[float, float, str] | None:
    """(lat, lon, roof) of *team*'s home stadium in *season* (None if unknown)."""
    t = str(team or "").strip()
    if league == "mlb":
        try:
            ov = MLB_PARK_OVERRIDES.get((t, int(season)))
        except (TypeError, ValueError):
            ov = None
        return ov or MLB_PARKS.get(t)
    return NFL_STADIUMS.get(t)


def roof_score(league: str, stadium_team, is_dome) -> float:
    """The wx_dome covariate: 1.0 fixed roof, 0.5 retractable, 0.0 open sky.
    Depends ONLY on stored game_weather row fields (is_dome + stadium_team),
    so betmeta's bulk and live paths derive it identically. Handles TB's
    2025 open-air exile correctly: those rows carry is_dome=FALSE and TB is
    not in the retractable set -> 0.0."""
    if is_dome:
        return 1.0
    return 0.5 if str(stadium_team or "").strip() in RETRACTABLE.get(league, ()) else 0.0


# ------------------------------------------------------------------ plumbing --
def ensure_tables(engine) -> None:
    sql = (Path(__file__).parent / "migrations" / "add_weather_tables.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(sql))


_UPSERT = text("""
    INSERT INTO odds.game_weather (league, game_key, game_date, stadium_team,
                                   kickoff_utc, temp_c, wind_kmh, precip_mm,
                                   is_dome, source, fetched_at)
    VALUES (:league, :key, :gd, :team, :ko, :t, :w, :p, :dome, :src, now())
    ON CONFLICT (league, game_key) DO UPDATE SET
        game_date = EXCLUDED.game_date, stadium_team = EXCLUDED.stadium_team,
        kickoff_utc = EXCLUDED.kickoff_utc, temp_c = EXCLUDED.temp_c,
        wind_kmh = EXCLUDED.wind_kmh, precip_mm = EXCLUDED.precip_mm,
        is_dome = EXCLUDED.is_dome, source = EXCLUDED.source, fetched_at = now()
""")


def _upsert(conn, league: str, rows: list[dict]) -> None:
    for r in rows:
        conn.execute(_UPSERT, {"league": league, **r})


def _get_json(url: str, params: dict, retries: int = 3) -> dict | None:
    """GET with retries + throttle. None on any failure (callers stay NaN-safe)."""
    import requests
    for attempt in range(retries):
        try:
            time.sleep(THROTTLE_S)
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("open-meteo %s -> HTTP %s (%s)", url, resp.status_code,
                           resp.text[:200])
        except requests.RequestException as e:
            logger.warning("open-meteo %s failed (try %s): %s", url, attempt + 1, e)
        time.sleep(2.0 * (attempt + 1))
    return None


class _HourlyGrid:
    """Hour-indexed (UTC) view of one open-meteo hourly payload."""

    def __init__(self, js: dict | None):
        h = (js or {}).get("hourly") or {}
        self.times = h.get("time") or []
        self.pos = {t: i for i, t in enumerate(self.times)}
        self.temp = h.get("temperature_2m") or []
        self.wind = h.get("wind_speed_10m") or []
        self.precip = h.get("precipitation") or []

    def at_kickoff(self, kickoff_utc: datetime) -> tuple[float, float, float] | None:
        """(temp_c, wind_kmh, precip_mm-over-3h) at the hour nearest kickoff."""
        if not self.pos or kickoff_utc is None:
            return None
        k = kickoff_utc.astimezone(timezone.utc) + timedelta(minutes=30)
        i = self.pos.get(k.strftime("%Y-%m-%dT%H:00"))
        if i is None or i >= len(self.temp):
            return None
        t, w = self.temp[i], self.wind[i] if i < len(self.wind) else None
        if t is None or w is None:
            return None
        pr = [v for v in self.precip[i:i + PRECIP_WINDOW_H] if v is not None]
        if not pr:
            return None
        return float(t), float(w), float(sum(pr))


def _fetch_archive(lat: float, lon: float, start: date, end: date) -> _HourlyGrid:
    return _HourlyGrid(_get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(),
        "end_date": end.isoformat(), "hourly": HOURLY_VARS, "timezone": "UTC"}))


def _fetch_forecast(lat: float, lon: float, past_days: int = 3,
                    forecast_days: int = 3) -> _HourlyGrid:
    return _HourlyGrid(_get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "hourly": HOURLY_VARS, "timezone": "UTC",
        "past_days": min(max(past_days, 0), 92),
        "forecast_days": min(max(forecast_days, 0), 16)}))


# ------------------------------------------------------------------- games --
def _mlb_games(engine, start: date, end: date) -> list[dict]:
    """One dict per raw.games row: key/game_date/team/season/kickoff (UTC)."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT game_pk, game_date, TRIM(home_team_code) AS team, season,
                   first_pitch_utc AS kickoff
            FROM raw.games WHERE game_date BETWEEN :a AND :b
            ORDER BY game_date, game_pk
        """), {"a": start, "b": end}).mappings().fetchall()
    return [dict(r) for r in rows]


def _nfl_games(engine, start: date, end: date) -> list[dict]:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT g.event_id AS game_pk, g.match_date AS game_date,
                   t.abbrev AS team, g.season, g.start_utc AS kickoff
            FROM nfl.games g JOIN nfl.teams t ON t.team_id = g.home_team_id
            WHERE g.match_date BETWEEN :a AND :b
            ORDER BY g.match_date, g.event_id
        """), {"a": start, "b": end}).mappings().fetchall()
    return [dict(r) for r in rows]


_GAME_SQL = {"mlb": _mlb_games, "nfl": _nfl_games}


def _existing(engine, league: str) -> dict[str, str]:
    """game_key -> source for every stored row of the league."""
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT game_key, source FROM odds.game_weather WHERE league = :lg"),
            {"lg": league}).fetchall()
    return {k: s for k, s in rows}


def _row_for(game: dict, vals: tuple[float, float, float], roof: str, src: str) -> dict:
    return {"key": str(game["game_pk"]), "gd": game["game_date"], "team": game["team"],
            "ko": game["kickoff"], "t": vals[0], "w": vals[1], "p": vals[2],
            "dome": roof == "dome", "src": src}


def _fill_games(engine, league: str, games: list[dict], grid_fetch, src: str) -> tuple[int, int]:
    """Group *games* by stadium, fetch one hourly grid per (stadium, range) via
    grid_fetch(lat, lon, dates), upsert matched rows. Dome games skip the API
    and store neutral constants. Returns (written, missed)."""
    written = missed = 0
    by_park: dict[tuple[float, float], list[tuple[dict, str]]] = {}
    dome_rows: list[dict] = []
    for g in games:
        park = park_for(league, g["team"], g["season"])
        if park is None or g["kickoff"] is None:
            missed += 1
            logger.warning("%s game %s: no stadium/kickoff (team=%s)",
                           league, g["game_pk"], g["team"])
            continue
        lat, lon, roof = park
        if roof == "dome":
            dome_rows.append(_row_for(g, DOME_NEUTRAL, roof, "dome"))
        else:
            by_park.setdefault((lat, lon), []).append((g, roof))
    if dome_rows:
        with engine.begin() as conn:
            _upsert(conn, league, dome_rows)
        written += len(dome_rows)
    for (lat, lon), pack in sorted(by_park.items()):
        dates = [g["game_date"] for g, _ in pack]
        grid = grid_fetch(lat, lon, dates)
        rows = []
        for g, roof in pack:
            vals = grid.at_kickoff(g["kickoff"]) if grid else None
            if vals is None:
                missed += 1
                continue
            rows.append(_row_for(g, vals, roof, src))
        if rows:
            with engine.begin() as conn:
                _upsert(conn, league, rows)
            written += len(rows)
    return written, missed


# ----------------------------------------------------------------- backfill --
def backfill(league: str, config: Config | None = None,
             start: date | None = None, end: date | None = None) -> dict:
    """Historical backfill for one league, per stadium per season (~1 archive
    call per stadium-season). Games younger than the archive lag go through
    the forecast API (past_days). Skips games that already have a 'hist' or
    'dome' row — idempotent, resumable."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_tables(engine)
    today = datetime.now(timezone.utc).date()
    start = start or (date(2023, 4, 1) if league == "mlb" else date(2022, 9, 1))
    end = end or today
    games = _GAME_SQL[league](engine, start, end)
    have = _existing(engine, league)
    todo = [g for g in games if have.get(str(g["game_pk"])) not in ("hist", "dome")]
    # one extra margin day: a game dated D can start after midnight UTC (D+1),
    # so its hours must be safely inside the archive's coverage window
    arch_cut = today - timedelta(days=ARCHIVE_LAG_DAYS)
    old = [g for g in todo if g["game_date"] < arch_cut]
    recent = [g for g in todo if g["game_date"] >= arch_cut]
    written = missed = 0
    # archive part: one call per (stadium, season) date-range (+2 days for
    # next-day-UTC kickoffs and the 3h precip window)
    by_season: dict[tuple, list[dict]] = {}
    for g in old:
        by_season.setdefault((g["team"], g["season"]), []).append(g)
    for (_team, _season), pack in sorted(by_season.items(), key=lambda kv: str(kv[0])):
        w, m = _fill_games(
            engine, league, pack,
            lambda lat, lon, dates: _fetch_archive(
                lat, lon, min(dates), min(max(dates) + timedelta(days=2), arch_cut)),
            "hist")
        written += w
        missed += m
    # recent part (archive lag window incl. today/future): forecast API past_days
    if recent:
        w, m = _fill_games(
            engine, league, recent,
            lambda lat, lon, dates: _fetch_forecast(
                lat, lon, past_days=max((today - min(dates)).days + 1, 1),
                forecast_days=max((max(dates) - today).days + 2, 2)),
            "forecast")
        written += w
        missed += m
    rep = {"league": league, "games": len(games), "already": len(games) - len(todo),
           "written": written, "missed": missed}
    logger.info("weather backfill %s", rep)
    return rep


# -------------------------------------------------------------- daily update --
def daily_update(config: Config | None = None) -> dict:
    """The odds_daily.sh weather step. Per league:
      1. FORECASTS for the pending window [today-2 .. today+2]: every game
         without a row, or whose row is still 'forecast', gets (re)fetched
         from the forecast API (past_days=3) — yesterday's games therefore
         carry measured values by reconcile time, today's carry a fresh
         forecast. 'hist'/'dome' rows are never touched.
      2. HIST TOP-UP: any 'forecast' row older than the archive lag is
         re-fetched from the archive API and flipped to source='hist', so
         training frames converge to measured weather.
    Never raises — a failed stadium just stays on its old row/NaN."""
    cfg = config or load_config()
    engine = create_engine(cfg)
    ensure_tables(engine)
    today = datetime.now(timezone.utc).date()
    rep: dict = {}
    for league in ("mlb", "nfl"):
        try:
            games = _GAME_SQL[league](engine, today - timedelta(days=2),
                                      today + timedelta(days=2))
            have = _existing(engine, league)
            todo = [g for g in games if have.get(str(g["game_pk"])) not in ("hist", "dome")]
            w1, m1 = _fill_games(
                engine, league, todo,
                lambda lat, lon, dates: _fetch_forecast(lat, lon, past_days=3,
                                                        forecast_days=4),
                "forecast")
            # hist top-up: 'forecast' rows old enough for the archive get
            # measured values ('forecast' -> 'hist'); games that somehow have
            # NO row at all in the last 30 days self-heal here too.
            arch_cut = today - timedelta(days=ARCHIVE_LAG_DAYS)
            with engine.begin() as conn:
                stale = conn.execute(text("""
                    SELECT game_key FROM odds.game_weather
                    WHERE league = :lg AND source = 'forecast' AND game_date < :cut
                """), {"lg": league, "cut": arch_cut}).fetchall()
            stale_keys = {k for (k,) in stale}
            w2 = m2 = 0
            allg = _GAME_SQL[league](engine, today - timedelta(days=30),
                                     arch_cut - timedelta(days=1))
            pack = [g for g in allg if str(g["game_pk"]) in stale_keys
                    or str(g["game_pk"]) not in have]
            if pack:
                w2, m2 = _fill_games(
                    engine, league, pack,
                    lambda lat, lon, dates: _fetch_archive(
                        lat, lon, min(dates), min(max(dates) + timedelta(days=2), arch_cut)),
                    "hist")
            rep[league] = {"forecast_written": w1, "forecast_missed": m1,
                           "hist_upgraded": w2, "hist_missed": m2}
        except Exception as e:  # noqa: BLE001 — weather is never fatal
            logger.exception("weather daily_update failed for %s", league)
            rep[league] = {"error": str(e)}
    logger.info("weather daily %s", rep)
    return rep


# ------------------------------------------------------------- live lookup --
_live_cache: dict[tuple[str, str], tuple[float, float, float, float]] = {}
_live_engine = None


def wx_tuple(league: str, row: dict) -> tuple[float, float, float, float]:
    """(wx_temp, wx_wind, wx_precip, wx_dome) from a stored game_weather row
    dict — THE single conversion used by both betmeta paths."""
    def _f(v):
        return float(v) if v is not None else float("nan")
    return (_f(row.get("temp_c")), _f(row.get("wind_kmh")), _f(row.get("precip_mm")),
            roof_score(league, row.get("stadium_team"), bool(row.get("is_dome"))))


def live_wx(league: str, game_key, cfg: Config) -> tuple[float, float, float, float]:
    """Single-row (live) path for the wx covariates: read the game's stored
    weather; when a PENDING/RECENT game has no row yet, fetch a forecast on
    the fly and upsert it (so the daily cron and ad-hoc scoring agree). Any
    failure -> NaNs (never crashes scoring). Cached per (league, game_key)."""
    nan4 = (float("nan"),) * 4
    if game_key is None:
        return nan4
    key = (league, str(game_key))
    if key in _live_cache:
        return _live_cache[key]
    global _live_engine
    try:
        if _live_engine is None:
            _live_engine = create_engine(cfg)
        with _live_engine.begin() as conn:
            row = conn.execute(text("""
                SELECT temp_c, wind_kmh, precip_mm, is_dome, stadium_team
                FROM odds.game_weather WHERE league = :lg AND game_key = :k
            """), {"lg": league, "k": str(game_key)}).mappings().fetchone()
        if row is not None:
            out = wx_tuple(league, dict(row))
            _live_cache[key] = out
            return out
        # no row: fetch a forecast on the fly for pending/recent games only
        today = datetime.now(timezone.utc).date()
        lo, hi = today - timedelta(days=3), today + timedelta(days=7)
        games = [g for g in _GAME_SQL[league](_live_engine, lo, hi)
                 if str(g["game_pk"]) == str(game_key)]
        if not games:
            _live_cache[key] = nan4  # old game with no stored weather: stay NaN
            return nan4
        g = games[0]
        park = park_for(league, g["team"], g["season"])
        if park is None or g["kickoff"] is None:
            _live_cache[key] = nan4
            return nan4
        lat, lon, roof = park
        if roof == "dome":
            row_d = _row_for(g, DOME_NEUTRAL, roof, "dome")
        else:
            vals = _fetch_forecast(lat, lon, past_days=3, forecast_days=7).at_kickoff(g["kickoff"])
            if vals is None:
                _live_cache[key] = nan4
                return nan4
            row_d = _row_for(g, vals, roof, "forecast")
        with _live_engine.begin() as conn:
            _upsert(conn, league, [row_d])
        out = wx_tuple(league, {"temp_c": row_d["t"], "wind_kmh": row_d["w"],
                                "precip_mm": row_d["p"], "is_dome": row_d["dome"],
                                "stadium_team": row_d["team"]})
        _live_cache[key] = out
        return out
    except Exception:  # noqa: BLE001 — weather must never break scoring
        logger.exception("live_wx(%s, %s) failed", league, game_key)
        return nan4


def weather_map(league: str, engine) -> dict[str, tuple[float, float, float, float]]:
    """Bulk path: game_key -> (wx_temp, wx_wind, wx_precip, wx_dome) for every
    stored row of the league. Values via the same wx_tuple() as the live path."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT game_key, temp_c, wind_kmh, precip_mm, is_dome, stadium_team
            FROM odds.game_weather WHERE league = :lg
        """), {"lg": league}).mappings().fetchall()
    return {r["game_key"]: wx_tuple(league, dict(r)) for r in rows}


# ------------------------------------------------------------------- CLI ----
def main() -> None:
    import argparse
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Sandy weather covariates (open-meteo)")
    ap.add_argument("cmd", choices=["backfill", "daily", "ensure"])
    ap.add_argument("--league", choices=["mlb", "nfl"], help="backfill: one league only")
    args = ap.parse_args()
    if args.cmd == "ensure":
        ensure_tables(create_engine(load_config()))
        print("odds.game_weather ready")
    elif args.cmd == "backfill":
        for lg in ([args.league] if args.league else ["mlb", "nfl"]):
            print(json.dumps(backfill(lg), default=str))
    else:
        print(json.dumps(daily_update(), default=str))


if __name__ == "__main__":
    main()
