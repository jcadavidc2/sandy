#!/usr/bin/env bash
# ONE-TIME bootstrap for the NBA vertical (same chain as soccer_bootstrap.sh).
set -euo pipefail
SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
cd /home/ec2-user/sandy
source "$HOME/.sandy_env"
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}
trap 'tg "❌ NBA bootstrap falló en el paso: ${STEP:-?}"' ERR

STEP="backfill"; echo "[$(date -Iseconds)] backfill..."; $SANDY nba backfill --start 2022-10-01 2>&1 | tail -1
STEP="ratings";  echo "[$(date -Iseconds)] ratings...";  $SANDY nba ratings 2>&1 | tail -1
STEP="backtest"; echo "[$(date -Iseconds)] backtest..."; $SANDY nba backtest 2>&1 | tail -1
STEP="calibrate"; echo "[$(date -Iseconds)] calibrate..."; $SANDY nba calibrate 2>&1 | tail -6
STEP="meta";     echo "[$(date -Iseconds)] meta...";     $SANDY nba meta 2>&1 | tail -1

STEP="cron"
if ! crontab -l | grep -q nba_nightly; then
    (crontab -l; echo "# Sandy NBA nightly (5:50 AM PST)";
     echo "50 12 * * * /home/ec2-user/sandy/scripts/nba_nightly.sh >> /home/ec2-user/sandy/logs/nba.log 2>&1") | crontab -
fi

STEP="digest"
$SANDY nba predict --notify 2>&1 | tail -2
tg "✅ Vertical NBA 🏀 LISTO: backfill + backtest + calibración + meta-modelo completos. Digest diario 7:50 AM Bogotá."
echo "[$(date -Iseconds)] NBA BOOTSTRAP COMPLETE"
