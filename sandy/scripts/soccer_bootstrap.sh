#!/usr/bin/env bash
# ONE-TIME bootstrap for the 4-league soccer vertical: backfill each league,
# fit models, walk-forward backtest, calibrate, train metas, install the cron
# line, and announce completion on Telegram. Fully autonomous; safe to re-run.
set -euo pipefail
SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
cd /home/ec2-user/sandy
source "$HOME/.sandy_env"
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

tg() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1" > /dev/null 2>&1 || true
}
trap 'tg "❌ Soccer bootstrap falló en el paso: ${STEP:-?}"' ERR

STEP="backfill"
for lg in col mex esp eng; do
    echo "[$(date -Iseconds)] backfill $lg..."
    $SANDY soccer backfill --league "$lg" --start 2023-07-01 2>&1 | tail -1
done

STEP="ratings";  echo "[$(date -Iseconds)] ratings...";  $SANDY soccer ratings 2>&1 | tail -4
STEP="backtest"; echo "[$(date -Iseconds)] backtest..."; $SANDY soccer backtest 2>&1 | tail -1
STEP="calibrate"; echo "[$(date -Iseconds)] calibrate..."; $SANDY soccer calibrate 2>&1 | tail -1
STEP="meta";     echo "[$(date -Iseconds)] metas...";    $SANDY soccer meta 2>&1 | tail -4

STEP="cron"
if ! crontab -l | grep -q soccer_nightly; then
    (crontab -l; echo "# Sandy Soccer (col/mex/esp/eng) nightly (5:15 AM PST)";
     echo "15 12 * * * /home/ec2-user/sandy/scripts/soccer_nightly.sh >> /home/ec2-user/sandy/logs/soccer.log 2>&1") | crontab -
fi

STEP="digest"
$SANDY soccer predict --notify 2>&1 | tail -2
tg "✅ Vertical de fútbol (🇨🇴🇲🇽🇪🇸🏴) LISTO: backfill + backtest + calibración + meta-modelos completos. Digest diario 7:15 AM Bogotá."
echo "[$(date -Iseconds)] SOCCER BOOTSTRAP COMPLETE"
