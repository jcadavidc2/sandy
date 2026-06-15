# Sandy Football — World Cup 2026 Prediction Vertical

A second sport vertical added to Sandy alongside the MLB system, sharing
infrastructure (config, db, logging, Telegram) but **fully separate** — its own
`football` Postgres schema, own models, own daily cron, own log file. Nothing
about the baseball system was changed beyond additive registration hooks.

Built: June 2026.

---

## What it does

- Predicts **World Cup 2026** matches: win/draw/loss, total goals (over/under),
  both-teams-score (BTTS), and the most-likely exact scoreline.
- Updates **daily** (6 AM PST cron): ingest yesterday's results, reconcile,
  refit, recalibrate, predict today's slate, push a **Telegram digest**.
- Is **calibrated**: a 3,000+ match walk-forward backtest measures how much to
  trust each prediction (confidence buckets → real accuracy).
- Has a **dashboard** (Streamlit) for phone/laptop: today's picks, a per-game
  explorer (past + future), a confederation scouting report, calibration, team
  ratings, recent results.

Mode: vibes modeling. Not a wagering tool.

---

## Data — single free source (API-Football / api-sports.io)

- **One source** powers everything: historical training data *and* live WC2026.
- **Free tier, two access modes** (discovered during build):
  - **Season-based** queries (`league=1&season=2026`) are **blocked** beyond
    2024 on the free plan.
  - **Date-based** queries (`/fixtures?date=YYYY-MM-DD`) **work for a ±1 day
    rolling window** (yesterday/today/tomorrow) — exactly what a daily loop
    needs. This is how live WC2026 data flows in.
- **Timezone**: all date queries pass `timezone=America/Los_Angeles` so a PST
  evening kickoff (e.g. 19:00 PST = 02:00 UTC next day) is grouped on the day it
  was watched, not the next UTC date. Configurable via `FOOTBALL_TIMEZONE`.
- **Training set**: 3,956 matches (2019→Jun 2026) across senior men's
  national-team competitions — World Cup, all friendlies, WC qualifiers (every
  confederation), Euro, Copa América, AFCON, Asian Cup, Gold Cup, Nations
  Leagues, etc. The qualifier "season 2024" buckets actually carry matches into
  2025–2026, so recent competitive form is included.
- **Per-match stats** (corners/cards/possession/shots) cost 1 request each
  against the 100/day free cap, so they trickle in over time; the model never
  waits on them. Report columns show "insufficient data" until present.
- **EC2 IPv6 note**: this box has broken IPv6 egress and API-Football is behind
  Cloudflare (IPv6-advertised). The client forces IPv4 resolution per request.

Required env (in `~/.sandy_env`): `APIFOOTBALL_KEY`. Reuses `MLB_DB_*`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

---

## Model — Dixon-Coles

`sandy/football/ratings.py` fits team attack/defense strengths + home advantage
+ the Dixon-Coles low-score correction by weighted maximum likelihood, with
exponential time decay (≈1-year half-life) and per-competition weights
(friendly 1.0 … World Cup 3.0). One global fit (~3.5s) refit nightly.

`sandy/football/predictor.py` turns the model into a full P(home=i, away=j)
scoreline matrix and derives **every market from that one distribution**:
W/D/L, P(over 1.5/2.5/3.5/4.5), BTTS, most-likely score. World Cup matches are
predicted at **neutral venue** (no home advantage).

Sanity: Spain / Argentina / Portugal / Brazil / Germany / England top the
strength table. Mild Asian-confederation inflation (weakly-connected schedules)
is a known artifact — future tempering via a FIFA-rank prior is noted below.

---

## Calibration — walk-forward backtest

`sandy/football/backtest.py` replays history leakage-free: for each match it
fits the model on **only prior data** (refit every 30 days of match calendar),
predicts, then reconciles against the known result. Seeded with 3,055
predictions (2022→2026, incl. WC2022 — "predict the previous World Cup").

`sandy/football/calibrator.py` measures, per market, accuracy + a reliability
table (confidence bucket → actual accuracy) + Brier score, and writes
`football.calibration_snapshots`.

Backtest results: **result 55.9%**, O/U 2.5 54.8%, BTTS 55.2%. Reliability is
monotonic — picks at ~80% confidence were right ~77%, ~40% picks ~43%. WC2022
in isolation was the hardest subset (35.9%, 64 high-variance knockout games),
which is exactly why the calibration/trust layer exists.

---

## Daily loop + cron

`sandy/scripts/football_nightly.sh` (cron: `0 13 * * *` = 6 AM PST), separate
from baseball, logging to `logs/football.log`:

1. `sandy football ingest`    — pull the ±1 day window (PST-aligned)
2. `sandy football reconcile` — fill actuals + was_correct for finished games
3. `sandy football ratings`   — refit Dixon-Coles on all data
4. `sandy football calibrate` — recompute calibration snapshots
5. `sandy football predict --notify` — predict upcoming + send Telegram digest

Predictions made today are reconciled tomorrow (1-day cycle). The cron **must**
run daily — the free date window is only ±1 day, so missed days can't be
backfilled later.

---

## Dashboard

`sandy/dashboard/app.py` (Streamlit, reads the `football` tables):

- **Upcoming picks** — color-coded by confidence, with legend
- **Game explorer** — pick any World Cup match (past WC2022 *or* upcoming
  WC2026): full prediction, and for finished games the actual result +
  per-market ✅/❌ + stats
- **Confederation report** — the original-prompt scouting view: per team over
  last 20 matches — est. W/D/L, GF/GA, O/U 2.5, BTTS, last-5 form, a HOT STAT,
  stat columns (insufficient-data until trickled), grouped by confederation
  (derived from qualifier leagues), with TOP picks
- **Calibration / trust**, **team strength**, **recent results**, **coverage**

Run: `streamlit run sandy/dashboard/app.py` (port 8502).

### Remote access (current: quick Cloudflare tunnel)

`cloudflared tunnel --url http://127.0.0.1:8502 --edge-ip-version 4` gives a
public `*.trycloudflare.com` HTTPS URL. **No inbound ports opened** (tunnel is
outbound-only); TLS at Cloudflare's edge. Free.

Caveats: the URL is **public + unauthenticated** (security by obscurity) and
**ephemeral** (dies on process/EC2 restart; URL changes). For an always-on,
login-gated, stable URL: persistent named tunnel + Cloudflare Access + systemd
(noted as a future step; still free).

---

## Module map (`sandy/football/`)

| File | Role |
|---|---|
| `client.py` | API-Football HTTP client (rate limit, retry, IPv4 force, key auth) |
| `parsers.py` | Pure JSON → typed rows |
| `schemas.py` | Dataclasses (MatchRow, MatchStatRow, FootballPrediction, …) |
| `ingest.py` | Idempotent upserts; season backfill + date-window daily ingest |
| `ratings.py` | Dixon-Coles fit + persistence |
| `predictor.py` | Scoreline matrix → markets; predict + persist |
| `reconciler.py` | Fill actuals + was_correct |
| `backtest.py` | Walk-forward replay → seeds calibration |
| `calibrator.py` | Per-market accuracy + reliability + snapshots |
| `report.py` | Confederation scouting report + TOP picks |
| `notifier.py` | Telegram digest (reuses over_under `send_telegram`) |
| `queries.py` | Shared read queries for digest + dashboard |

CLI: `sandy football {ingest,ratings,reconcile,calibrate,predict,backtest}`.
Schema: `sandy/migrations/add_football_tables.sql` (schema `football`).

---

## Known limitations / future

- Per-match stats (corners/cards/possession) trickle in slowly (100 req/day cap).
- Mild confederation-strength inflation — temper with a FIFA-rank prior.
- Meta-model P(pick correct) not yet built (reliability table covers the gap).
- Free date window is ±1 day — daily cron must not miss days.
- Remote access is an ephemeral public tunnel — upgrade to login-gated systemd
  service for always-on private access.
