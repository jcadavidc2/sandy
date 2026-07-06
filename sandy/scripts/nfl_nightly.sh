#!/usr/bin/env bash
# Sandy NFL Nightly Pipeline — own schema/models/log/cron, mirrors nba_nightly.sh.
# Steps: ingest window → reconcile → refit ratings → calibrate → meta → predict+digest.
# NFL games end ~9 PM PST (MNF); run early morning PST. Suggested: 12:55 UTC (5:55 AM PST).

set -euo pipefail

SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
PROJECT_DIR="/home/ec2-user/sandy"
cd "$PROJECT_DIR"

if [ -f "$HOME/.sandy_env" ]; then
    source "$HOME/.sandy_env"
fi
export MLB_MODEL_DIR="${MLB_MODEL_DIR:-/home/ec2-user/sandy/models}"

send_telegram() {
    local message="$1"
    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
        return 0
    fi
    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${message}" \
        > /dev/null 2>&1 || true
}

fail() {
    send_telegram "❌ NFL pipeline failed at: $1"
    exit 1
}

echo "=========================================="
echo "[$(date -Iseconds)] NFL Nightly Pipeline starting"
echo "=========================================="

echo "[$(date -Iseconds)] Step 1/6: Ingest window..."
$SANDY nfl ingest 2>&1 | tail -1 || fail "ingest"

echo "[$(date -Iseconds)] Step 2/6: Reconcile finished predictions..."
$SANDY nfl reconcile 2>&1 | tail -1 || fail "reconcile"

echo "[$(date -Iseconds)] Step 3/6: Refit points model..."
$SANDY nfl ratings 2>&1 | tail -1 || fail "ratings"

echo "[$(date -Iseconds)] Step 4/6: Recompute calibration..."
$SANDY nfl calibrate 2>&1 | tail -4 || fail "calibrate"

echo "[$(date -Iseconds)] Step 5/6: Retrain meta-model..."
$SANDY nfl meta 2>&1 | tail -1 || fail "meta"

echo "[$(date -Iseconds)] Step 6/6: Predict + send digest..."
$SANDY nfl predict --notify 2>&1 | tail -20 || fail "predict"

echo "=========================================="
echo "[$(date -Iseconds)] NFL Nightly Pipeline COMPLETE"
echo "=========================================="
