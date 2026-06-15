#!/usr/bin/env bash
# Sandy Football (World Cup) Nightly Pipeline.
# SEPARATE from the baseball nightly_pipeline.sh — own schema, own models,
# own log file, own cron line. Does not touch the baseball system.
#
# Steps (each starts only after the previous finishes):
#   1. Ingest the +/-1 day fixture window (yesterday's results, today's slate)
#   2. Reconcile finished predictions vs actuals (fill was_correct)
#   3. Refit the Dixon-Coles ratings on all data
#   4. Recompute calibration snapshots
#   5. Predict upcoming fixtures + send the Telegram digest
#
# Run once daily, early morning PST (after the previous day's WC matches
# finish, before the new day's kick off). Suggested: 13:00 UTC (6 AM PST).

set -euo pipefail

SANDY="/home/ec2-user/sandy/.venv/bin/sandy"
PROJECT_DIR="/home/ec2-user/sandy"
cd "$PROJECT_DIR"

# Load environment (DB creds, APIFOOTBALL_KEY, TELEGRAM_*).
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
    send_telegram "❌ Football pipeline failed at: $1"
    exit 1
}

echo "=========================================="
echo "[$(date -Iseconds)] Football Nightly Pipeline starting"
echo "=========================================="

echo "[$(date -Iseconds)] Step 1/5: Ingest fixture window..."
$SANDY football ingest 2>&1 | tail -1 || fail "ingest"

echo "[$(date -Iseconds)] Step 2/5: Reconcile finished predictions..."
$SANDY football reconcile 2>&1 | tail -1 || fail "reconcile"

echo "[$(date -Iseconds)] Step 3/5: Refit Dixon-Coles ratings..."
$SANDY football ratings 2>&1 | tail -1 || fail "ratings"

echo "[$(date -Iseconds)] Step 4/5: Recompute calibration..."
$SANDY football calibrate 2>&1 | tail -3 || fail "calibrate"

echo "[$(date -Iseconds)] Step 5/5: Predict + send digest..."
$SANDY football predict --notify 2>&1 | tail -20 || fail "predict"

echo "=========================================="
echo "[$(date -Iseconds)] Football Nightly Pipeline COMPLETE"
echo "=========================================="
