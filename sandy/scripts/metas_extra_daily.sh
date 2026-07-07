#!/usr/bin/env bash
# Daily (13:30 UTC, after all vertical nightlies): retrain the two metas that
# don't live inside a vertical nightly (worldcup + mlb betmeta), then restart
# the dashboard so the webpage serves TODAY's artifacts (it caches models in
# memory). Keeps the whole system hands-off.
set -euo pipefail
cd /home/ec2-user/sandy
# set -a: robust sourcing — see odds_daily.sh (a 2026-07-05 env edit dropped
# an `export` and silently broke the odds fetch for two days).
set -a; source "$HOME/.sandy_env"; set +a
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}
trap 'tg "❌ Metas extra diarias fallaron (worldcup/mlb o reinicio del dashboard)"' ERR

echo "[$(date -Iseconds)] retraining worldcup + mlb metas..."
.venv/bin/python - <<'PY'
from sandy.betmeta import train_meta
for lg in ("worldcup", "mlb"):
    r = train_meta(lg)
    print(f"{lg}: rows={r['rows']} thr={r['threshold']} auc={r['auc']}")
PY

# Meta² refiner: nightly retrain (10-block OOF, cached per league by data
# signature) + the adaptive-hybrid engine choice per tier (calib matched-volume
# vs meta floors, one-way test clamp). NON-FATAL by design: on any failure the
# ⭐/💎 levels keep serving plain meta floors (nivel_for_pick falls back on its
# own; an artifact >10 days old is ignored), so metas + dashboard refresh still
# run. Takes ~20-45 min on nights when league data changed (sequential, nice'd).
echo "[$(date -Iseconds)] retraining meta² refiner + choosing tier engines..."
if ! nice -n 10 .venv/bin/python - <<'PY'
import json
from sandy.betrefine import train_refiner, choose_engines
rep = train_refiner()
print("refiner:", json.dumps({k: rep[k] for k in ("n_dataset", "calib_auc", "test_auc", "floors")},
                             default=str))
ch = choose_engines()
print("engines:", json.dumps(ch.get("engines"), default=str))
print("hybrid vs meta (test):", json.dumps(ch.get("hybrid_test_report"), default=str))
PY
then
    tg "⚠️ Refinador meta² falló hoy — los niveles ⭐/💎 siguen con umbrales del meta (fallback automático)"
fi

# Odds/value layer (TheOddsAPI): frugal fetch (only sports with pending
# predictions today, once per sport per day), match to our games, value log +
# reconcile. NON-FATAL like the refiner block: picks simply render without
# cuota/edge when this fails. MLB usually isn't covered here (its morning
# predictions land 14:00 UTC) — the 14:15 UTC odds_daily.sh cron picks it up.
echo "[$(date -Iseconds)] odds/value layer (fetch frugal + match + value log)..."
if ! nice -n 10 .venv/bin/python -m sandy.odds daily >> logs/odds.log 2>&1; then
    echo "[$(date -Iseconds)] odds daily FAILED (non-fatal)" | tee -a logs/odds.log
    tg "⚠️ Capa de cuotas/valor falló en el run de las 13:30 — reintenta sola a las 14:15 UTC"
fi

# 🎰 Settle yesterday's paper-money portfolio tickets from the reconciled
# outcomes (the odds step above just filled value_log results) and close the
# bankroll day: end_bank = start − staked + returned. Runs BEFORE the
# dashboard restart so the page serves the updated curve. NON-FATAL: open
# tickets simply wait for the next run. Logs into odds.log ('portfolio settle
# COMPLETE') like the rest of the odds/value layer.
echo "[$(date -Iseconds)] liquidando portafolio de papel 🎰 (tiquetes + bankroll)..."
if ! nice -n 10 .venv/bin/python -m sandy.portfolio settle >> logs/odds.log 2>&1; then
    echo "[$(date -Iseconds)] portfolio settle FAILED (non-fatal)" | tee -a logs/odds.log
    tg "⚠️ La liquidación del portafolio de papel 🎰 falló — se reintenta sola mañana"
fi

# 🎯 Settle portfolio B "Picks del Día" (own tables — legs re-grade straight
# from the prediction tables, not value_log; see sandy/portfolio_picks.py).
# NON-FATAL and independent from the 🎰 settle above. Logs 'portfolio picks
# settle COMPLETE' into odds.log.
echo "[$(date -Iseconds)] liquidando portafolio B de picks 🎯 (tiquetes + bankroll)..."
if ! nice -n 10 .venv/bin/python -m sandy.portfolio_picks settle >> logs/odds.log 2>&1; then
    echo "[$(date -Iseconds)] portfolio picks settle FAILED (non-fatal)" | tee -a logs/odds.log
    tg "⚠️ La liquidación del portafolio B 🎯 falló — se reintenta sola mañana"
fi

echo "[$(date -Iseconds)] restarting dashboard (fresh artifacts for the webpage)..."
PID=$(ss -tlnp 2>/dev/null | grep 8502 | grep -oP 'pid=\K[0-9]+' | head -1 || true)
[ -n "${PID}" ] && kill "$PID" && sleep 3
setsid nohup ./scripts/dashboard.sh >> logs/dashboard.log 2>&1 &
sleep 10
curl -sf -o /dev/null http://localhost:8502/ || { tg "❌ Dashboard no volvió tras el reinicio diario"; exit 1; }
echo "[$(date -Iseconds)] metas extra + dashboard refresh COMPLETE"
