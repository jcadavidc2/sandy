#!/usr/bin/env bash
# Odds/value daily pass at 14:15 UTC — right after MLB morning predictions
# (14:00 UTC), so today's MLB games get their odds, edge/EV and value log.
# CREDIT FRUGALITY: sandy.odds only fetches sports with pending predictions
# today and skips any sport already fetched today (the 13:30 metas_extra run
# usually covers the soccer/NBA/NHL slates; this run mostly adds MLB).
set -uo pipefail
cd /home/ec2-user/sandy
source "$HOME/.sandy_env"

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
if ! nice -n 10 .venv/bin/python -m sandy.odds daily; then
    echo "[$(date -Iseconds)] odds daily FAILED"
    tg "⚠️ Capa de cuotas/valor falló hoy — los picks salen sin cuota/edge (nada más se afecta)"
    exit 1
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
