#!/usr/bin/env bash
# Odds/value daily pass at 14:15 UTC — right after MLB morning predictions
# (14:00 UTC), so today's MLB games get their odds, edge/EV and value log.
# CREDIT FRUGALITY: sandy.odds only fetches sports with pending predictions
# today and skips any sport already fetched today (the 13:30 metas_extra run
# usually covers the soccer/NBA/NHL slates; this run mostly adds MLB).
set -uo pipefail
cd /home/ec2-user/sandy
# set -a: export EVERYTHING the env file defines. On 2026-07-05 the file was
# edited and ODDS_API_KEY lost its `export` prefix — plain `source` then left
# the key un-exported and the 07-06/07-07 fetches died with "ODDS_API_KEY not
# set". set -a makes the sourcing robust to that class of edit forever.
set -a; source "$HOME/.sandy_env"; set +a

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}

# 🌤️ Weather covariates (open-meteo, keyless — sandy/weather.py): BEFORE the
# value/portfolio steps so today's MLB/NFL candidates score with a stored
# forecast row instead of on-the-fly fetches. Also refreshes yesterday's games
# from the forecast API (measured past_days values, so reconciled training rows
# get actuals) and flips week-old 'forecast' rows to archive 'hist'. NON-FATAL:
# on failure the metas simply score with wx=NaN (trees route to default).
echo "[$(date -Iseconds)] weather daily (forecast hoy + actuals ayer + hist top-up)..."
if ! nice -n 10 .venv/bin/python -m sandy.weather daily; then
    echo "[$(date -Iseconds)] ⚠️ weather daily FAILED (non-fatal — picks salen con clima NaN)"
fi

echo "[$(date -Iseconds)] odds daily (fetch frugal + match + value log + reconcile)..."
# Captura la salida para leer los créditos restantes de The Odds API: el módulo NO falla por
# quota agotada (los 401 se tragan por diseño) — el 25/7 se agotó y nadie se enteró por días.
ODDS_OUT=$(mktemp)
if ! nice -n 10 .venv/bin/python -m sandy.odds daily 2>&1 | tee "$ODDS_OUT"; then
    rm -f "$ODDS_OUT"
    echo "[$(date -Iseconds)] odds daily FAILED"
    tg "⚠️ Capa de cuotas/valor falló hoy — los picks salen sin cuota/edge (nada más se afecta)"
    exit 1
fi
REMAIN=$(grep -o 'remaining=[0-9]*' "$ODDS_OUT" | tail -1 | cut -d= -f2)
rm -f "$ODDS_OUT"
if [ -n "${REMAIN:-}" ]; then
    if [ "$REMAIN" -eq 0 ]; then
        tg "🚨 The Odds API: créditos AGOTADOS (0/500 este mes). Las cuotas/edge NO se actualizan hasta el reinicio mensual del plan (o upgrade). Predicciones y dashboard siguen normales."
    elif [ "$REMAIN" -lt 60 ]; then
        tg "⚠️ The Odds API: quedan solo $REMAIN créditos este mes — se agotarán en ~$((REMAIN / 20)) días al ritmo actual."
    fi
fi
echo "[$(date -Iseconds)] odds daily COMPLETE"

# 🎰 Paper-money portfolio: build + persist TODAY's ticket sheet from the value
# picks logged above (Kelly fraccional Monte Carlo, $500 steps, 30% per-game
# cap — see sandy/portfolio.py). NON-FATAL: a failure only means no paper
# portfolio today; odds/value/picks are unaffected. Idempotent (skips if the
# day is already built). Prints 'portfolio build COMPLETE' into odds.log.
echo "[$(date -Iseconds)] portfolio diario (Kelly fraccional, dinero de papel)..."
.venv/bin/python -m sandy.portfolio settle 2>&1 || true
if ! nice -n 10 .venv/bin/python -m sandy.portfolio build; then
    echo "[$(date -Iseconds)] portfolio build FAILED (non-fatal)"
    tg "⚠️ El portafolio de papel 🎰 falló hoy — cuotas y picks no se afectan"
fi
echo "[$(date -Iseconds)] portfolio daily COMPLETE"

# 🎯 Portafolio B "Picks del Día": SECOND paper bank (own tables, own $100k)
# that bets today's ✅ accuracy picks with our RAW probabilities — even
# without market edge (always-bet floor). The A/B experiment vs the 🎰 value
# portfolio above; see sandy/portfolio_picks.py. NON-FATAL and fully
# separate: a failure never touches odds/value/🎰. Idempotent like A's.
# Prints 'portfolio picks build COMPLETE' into odds.log.
echo "[$(date -Iseconds)] portafolio B picks 🎯 (tesis: apostar nuestras creencias)..."
.venv/bin/python -m sandy.portfolio_picks settle 2>&1 || true
if ! nice -n 10 .venv/bin/python -m sandy.portfolio_picks build; then
    echo "[$(date -Iseconds)] portfolio picks build FAILED (non-fatal)"
    tg "⚠️ El portafolio B de picks 🎯 falló hoy — el 🎰 y los picks no se afectan"
fi
echo "[$(date -Iseconds)] portfolio picks daily COMPLETE"
