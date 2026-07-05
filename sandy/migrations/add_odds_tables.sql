-- Odds / value layer (TheOddsAPI v4).
-- Idempotent: safe to run on every deploy.
--
-- odds.market_odds: one row per (event, book, market, point, side) snapshot.
--   implied        = 1 / price  (bookmaker probability WITH the vig)
--   implied_novig  = proportional overround removal within the outcome set of
--                    the same (event, book, market, point): implied / sum(implied).
--                    For 3-way soccer h2h the draw participates in the sum even
--                    though only home/away rows are stored.
--   matched/our_*  = linkage to OUR prediction row (league key + our team names
--                    + our match_date) filled by the name-matching step.
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.market_odds (
    id            BIGSERIAL PRIMARY KEY,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    sport_key     TEXT        NOT NULL,               -- TheOddsAPI key (baseball_mlb, ...)
    league        TEXT        NOT NULL,               -- our league key (mlb, nba, soccer_eng, ...)
    event_id      TEXT        NOT NULL,               -- TheOddsAPI event id
    event_home    TEXT        NOT NULL,               -- API team names (verbatim)
    event_away    TEXT        NOT NULL,
    commence_utc  TIMESTAMPTZ NOT NULL,
    book          TEXT        NOT NULL,               -- bookmaker key (pinnacle, bet365, ...)
    -- 'double_chance' is DERIVED per book from the 3-way soccer h2h (see
    -- sandy/odds.py): cuota_1X = 1/(1/cuota_home + 1/cuota_draw)
    market        TEXT        NOT NULL CHECK (market IN ('h2h', 'totals', 'double_chance')),
    point         REAL,                               -- totals line; NULL for h2h/DC
    side          TEXT        NOT NULL
                  CHECK (side IN ('over', 'under', 'home', 'away', 'home_or_draw')),
    price         REAL        NOT NULL,               -- decimal odds
    implied       REAL        NOT NULL,
    implied_novig REAL,
    -- linkage to our prediction rows (filled by the matcher; NULL = unmatched)
    matched       BOOLEAN     NOT NULL DEFAULT FALSE,
    match_date    DATE,                               -- OUR match_date
    our_home      TEXT,                               -- OUR home team name/code
    our_away      TEXT
);

CREATE INDEX IF NOT EXISTS idx_market_odds_league_date
    ON odds.market_odds (league, match_date);
CREATE INDEX IF NOT EXISTS idx_market_odds_sport_fetched
    ON odds.market_odds (sport_key, fetched_at);
CREATE INDEX IF NOT EXISTS idx_market_odds_lookup
    ON odds.market_odds (league, our_home, our_away, market, side);

-- One row per identified VALUE pick (edge >= 3pp at identification time).
-- stake is 1.0 unit flat; reconcile fills result ('win'/'lose') + units
-- (win: cuota-1, lose: -1) from our predictions' outcomes.
CREATE TABLE IF NOT EXISTS odds.value_log (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    date       DATE  NOT NULL,          -- our match_date
    league     TEXT  NOT NULL,
    home       TEXT  NOT NULL,          -- our team names/codes
    away       TEXT  NOT NULL,
    market     TEXT  NOT NULL,          -- our market key (over_7_5, winner, double_chance, ...)
    side       TEXT  NOT NULL CHECK (side IN ('over', 'under', 'home', 'away', 'home_or_draw')),
    line       REAL,                    -- NULL for h2h/winner
    prob       REAL  NOT NULL,          -- OUR side prob (base model, calibrated) — NOT the 🤖 meta
    cuota      REAL  NOT NULL,          -- best decimal price across books at log time
    edge       REAL  NOT NULL,          -- prob - consensus implied_novig
    ev         REAL,                    -- prob*(cuota-1) - (1-prob)
    stake      REAL  NOT NULL DEFAULT 1.0,
    result     TEXT  CHECK (result IN ('win', 'lose')),
    units      REAL,                    -- +(cuota-1)*stake on win, -stake on loss
    settled_at TIMESTAMPTZ
);

-- idempotent daily inserts (line NULL-safe)
CREATE UNIQUE INDEX IF NOT EXISTS uq_value_log_pick
    ON odds.value_log (date, league, home, away, market, side, COALESCE(line, -1.0));

CREATE INDEX IF NOT EXISTS idx_value_log_result ON odds.value_log (result, date);

-- Idempotent widening of the CHECKs for installs created before the derived
-- double_chance market existed (drop auto/named constraint, re-add named).
ALTER TABLE odds.market_odds DROP CONSTRAINT IF EXISTS market_odds_market_check;
ALTER TABLE odds.market_odds DROP CONSTRAINT IF EXISTS ck_market_odds_market;
ALTER TABLE odds.market_odds ADD CONSTRAINT ck_market_odds_market
    CHECK (market IN ('h2h', 'totals', 'double_chance'));
ALTER TABLE odds.market_odds DROP CONSTRAINT IF EXISTS market_odds_side_check;
ALTER TABLE odds.market_odds DROP CONSTRAINT IF EXISTS ck_market_odds_side;
ALTER TABLE odds.market_odds ADD CONSTRAINT ck_market_odds_side
    CHECK (side IN ('over', 'under', 'home', 'away', 'home_or_draw'));
ALTER TABLE odds.value_log DROP CONSTRAINT IF EXISTS value_log_side_check;
ALTER TABLE odds.value_log DROP CONSTRAINT IF EXISTS ck_value_log_side;
ALTER TABLE odds.value_log ADD CONSTRAINT ck_value_log_side
    CHECK (side IN ('over', 'under', 'home', 'away', 'home_or_draw'));
