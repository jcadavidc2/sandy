-- Game-time weather covariates for the MLB / NFL meta-models (open-meteo,
-- keyless). One row per game; game_key = MLB game_pk / NFL event_id (as text).
-- Idempotent: safe to run on every deploy.
--
-- source: 'hist'     = archive-api.open-meteo.com (measured, ~5-day lag)
--         'forecast' = api.open-meteo.com/v1/forecast (pending/recent games;
--                      the daily job replaces these with 'hist' once the
--                      archive covers the date)
--         'dome'     = fixed-roof stadium -> neutral constants (21C, 0, 0);
--                      the is_dome flag itself is the feature.
-- is_dome: TRUE only for FIXED roofs (Tropicana, US Bank, SoFi, ...).
--          Retractable-roof stadiums store real outdoor readings with
--          is_dome=FALSE; the 0.5 "retractable" score lives in code
--          (sandy/weather.py RETRACTABLE) and is derived at feature time.
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.game_weather (
    id           BIGSERIAL PRIMARY KEY,
    league       TEXT        NOT NULL,              -- 'mlb' | 'nfl'
    game_key     TEXT        NOT NULL,              -- mlb game_pk / nfl event_id
    game_date    DATE        NOT NULL,
    stadium_team TEXT        NOT NULL,              -- HOME team code = stadium used
    kickoff_utc  TIMESTAMPTZ,                       -- first pitch / kickoff (UTC)
    temp_c       REAL,                              -- at the hour nearest kickoff
    wind_kmh     REAL,                              -- 10m wind, same hour
    precip_mm    REAL,                              -- sum over kickoff hour + next 2
    is_dome      BOOLEAN     NOT NULL DEFAULT FALSE,
    source       TEXT        NOT NULL CHECK (source IN ('hist', 'forecast', 'dome')),
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league, game_key)
);

CREATE INDEX IF NOT EXISTS idx_game_weather_league_date
    ON odds.game_weather (league, game_date);
