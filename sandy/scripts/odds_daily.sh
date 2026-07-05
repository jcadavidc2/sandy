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

echo "[$(date -Iseconds)] odds daily (fetch frugal + match + value log + reconcile)..."
if ! nice -n 10 .venv/bin/python -m sandy.odds daily; then
    echo "[$(date -Iseconds)] odds daily FAILED"
    tg "⚠️ Capa de cuotas/valor falló hoy — los picks salen sin cuota/edge (nada más se afecta)"
    exit 1
fi
echo "[$(date -Iseconds)] odds daily COMPLETE"
