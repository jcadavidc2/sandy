#!/usr/bin/env bash
# ONE-TIME bootstrap for the 6 CUP competitions (2026-07-13): ucl uel lib sud
# lgc ccc. Backfills ~3 seasons from ESPN, refits ratings, walk-forward
# backtests ONLY the cups (domestic leagues untouched), calibrates, trains the
# per-cup metas (with the is_knockout stage covariate), and announces on
# Telegram. Sequential + nice'd end to end — 1.9GB box, one heavy op at a time.
# Safe to re-run (idempotent upserts everywhere).
set -uo pipefail
SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
cd /home/ec2-user/sandy
set -a; source "$HOME/.sandy_env"; set +a
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}

# Resumable: override to skip already-backfilled cups, e.g. CUPS="uel lib" ./cups_bootstrap.sh
CUPS="${CUPS:-ucl uel lib sud lgc ccc}"

for lg in $CUPS; do
    echo "[$(date -Iseconds)] backfill $lg..."
    if ! nice -n 10 $SANDY soccer backfill --league "$lg" --start 2023-07-01 2>&1 | tail -1; then
        tg "❌ Copas bootstrap: backfill de $lg falló — revisa logs/cups_bootstrap.log"
        exit 1
    fi
done

echo "[$(date -Iseconds)] ratings (all leagues, cups included)..."
nice -n 10 $SANDY soccer ratings 2>&1 | tail -10 || { tg "❌ Copas bootstrap: ratings falló"; exit 1; }

for lg in $CUPS; do
    echo "[$(date -Iseconds)] backtest $lg..."
    nice -n 10 $SANDY soccer backtest --league "$lg" 2>&1 | tail -1 || { tg "❌ Copas bootstrap: backtest de $lg falló"; exit 1; }
done

echo "[$(date -Iseconds)] calibrate..."
nice -n 10 $SANDY soccer calibrate 2>&1 | tail -1 || { tg "❌ Copas bootstrap: calibrate falló"; exit 1; }

echo "[$(date -Iseconds)] metas (per-cup, is_knockout covariate)..."
nice -n 10 $SANDY soccer meta 2>&1 | tail -12 || { tg "❌ Copas bootstrap: metas fallaron"; exit 1; }

echo "[$(date -Iseconds)] predict + digest..."
nice -n 10 $SANDY soccer predict --notify 2>&1 | tail -2 || tg "⚠️ Copas bootstrap: predict/digest falló (no fatal)"

tg "✅ COPAS listas en Sandy: ⭐Champions 🇪🇺Europa 🏆Libertadores 🥈Sudamericana 🇺🇸🇲🇽Leagues Cup 🌎Concacaf CC — backfill 3 temporadas + backtest + calibración + metas (con covariable de fase eliminatoria). Corren en el nightly de fútbol de las 7:15 AM Bogotá."
echo "[$(date -Iseconds)] CUPS BOOTSTRAP COMPLETE"
