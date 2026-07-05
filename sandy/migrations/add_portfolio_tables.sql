-- 🎰 Paper-money daily betting portfolio (BetPlay simulation — the user does
-- NOT bet; see sandy/portfolio.py).
-- Idempotent: safe to run on every deploy / on module startup.
--
-- odds.portfolio_log: one row per TICKET of a day's persisted portfolio.
--   legs         = JSONB array [{vl_id, date, liga, liga_titulo, partido, home,
--                  away, market, side, line, mercado, pick, cuota, prob}, ...]
--                  — every leg is a DIFFERENT game (no intra-game correlation).
--   ticket_cuota = Π leg cuotas; ticket_prob = Π leg probs (independence
--                  assumption across different matches — documented in
--                  sandy/portfolio.py).
--   status/returned: settled from OUR predictions' outcomes via
--                  odds.value_log.result — all legs win → returned =
--                  stake·cuota; any leg loses → 0; a leg that never
--                  reconciles within the grace window (postponed game) VOIDS
--                  the ticket → stake refunded (simplification of the book
--                  rule that re-prices the parlay without the void leg).
CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.portfolio_log (
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

CREATE INDEX IF NOT EXISTS idx_portfolio_log_status
    ON odds.portfolio_log (status, date);

-- odds.bankroll: one row per day the optimizer ran — the compounding paper
-- bankroll. start_bank chains from the previous day's end_bank (initial
-- $100,000). staked = Σ ticket stakes ($0 on no-value days — still logged).
-- returned/end_bank are filled by the settle pass once every ticket of the
-- day is graded: end_bank = start_bank - staked + returned.
CREATE TABLE IF NOT EXISTS odds.bankroll (
    id         BIGSERIAL   PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    date       DATE        NOT NULL UNIQUE,
    start_bank REAL        NOT NULL,
    staked     REAL        NOT NULL DEFAULT 0,
    returned   REAL,
    end_bank   REAL,
    settled_at TIMESTAMPTZ
);
