#!/usr/bin/env bash
# Daily (13:30 UTC, after all vertical nightlies): retrain the two metas that
# don't live inside a vertical nightly (worldcup + mlb betmeta), then restart
# the dashboard so the webpage serves TODAY's artifacts (it caches models in
# memory). Keeps the whole system hands-off.
set -euo pipefail
cd /home/ec2-user/sandy
source "$HOME/.sandy_env"
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

echo "[$(date -Iseconds)] restarting dashboard (fresh artifacts for the webpage)..."
PID=$(ss -tlnp 2>/dev/null | grep 8502 | grep -oP 'pid=\K[0-9]+' | head -1 || true)
[ -n "${PID}" ] && kill "$PID" && sleep 3
setsid nohup ./scripts/dashboard.sh >> logs/dashboard.log 2>&1 &
sleep 10
curl -sf -o /dev/null http://localhost:8502/ || { tg "❌ Dashboard no volvió tras el reinicio diario"; exit 1; }
echo "[$(date -Iseconds)] metas extra + dashboard refresh COMPLETE"
