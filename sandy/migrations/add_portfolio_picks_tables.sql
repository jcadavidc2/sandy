-- 🎯 Portafolio B "Picks del Día" — SECOND paper-money bank (see
-- sandy/portfolio_picks.py). Same shapes as the 🎰 value portfolio's tables
-- (odds.portfolio_log / odds.bankroll) but FULLY SEPARATE: strategy A rows are
-- never touched by strategy B and vice versa — the two bankroll curves are the
-- A/B experiment.
-- Idempotent: safe to run on every deploy / on module startup.
--
-- odds.portfolio_picks_log: one row per TICKET of a day's persisted portfolio.
--   legs         = JSONB array [{date, liga, liga_titulo, partido, home, away,
--                  market, side, line, mercado, pick, cuota, prob, ...}] —
--                  every leg is a DIFFERENT game. Unlike portfolio A, legs are
--                  NOT value_log rows (B bets ✅ accuracy picks even without
--                  positive market edge), so there is no vl_id: settling
--                  re-grades each leg straight from the prediction tables.
--   ticket_cuota = Π leg cuotas; ticket_prob = Π leg probs — leg prob is OUR
--                  RAW calibrated side probability (no market shrink; that is
--                  portfolio B's thesis).
--   status/returned: same grading rule as A (all win → stake·cuota; any lose
--                  → 0; unreconciled beyond the grace window → void → refund).
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.portfolio_picks_log (
    id           BIGSERIAL   PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    date         DATE        NOT NULL,      -- portfolio day (America/Los_Angeles, like odds)
    ticket_id    INTEGER     NOT NULL,      -- 1..n within the day
    legs         JSONB       NOT NULL,
    ticket_cuota REAL        NOT NULL,
    ticket_prob  REAL        NOT NULL,
    stake        REAL        NOT NULL,      -- paper COP-like units, $500 multiples
    status       TEXT        NOT NULL DEFAULT 'open'
                 CHECK (status IN ('open', 'won', 'lost', 'void')),
    returned     REAL,                      -- won: stake*cuota, lost: 0, void: stake
    settled_at   TIMESTAMPTZ,
    UNIQUE (date, ticket_id)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_picks_log_status
    ON odds.portfolio_picks_log (status, date);

-- odds.bankroll_picks: one row per day portfolio B ran — its own compounding
-- paper bankroll (initial $100,000, independent from odds.bankroll).
-- returned/end_bank are filled by the settle pass once every ticket of the
-- day is graded: end_bank = start_bank - staked + returned.
CREATE TABLE IF NOT EXISTS odds.bankroll_picks (
    id         BIGSERIAL   PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    date       DATE        NOT NULL UNIQUE,
    start_bank REAL        NOT NULL,
    staked     REAL        NOT NULL DEFAULT 0,
    returned   REAL,
    end_bank   REAL,
    settled_at TIMESTAMPTZ
);
